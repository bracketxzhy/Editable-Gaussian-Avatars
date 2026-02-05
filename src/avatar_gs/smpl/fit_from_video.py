from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None


# MediaPipe -> SMPL joints (subset for stability)
MP_TO_SMPL = {
    11: 16,  # left shoulder
    12: 17,  # right shoulder
    23: 1,   # left hip
    24: 2,   # right hip
    13: 18,  # left elbow
    14: 19,  # right elbow
    15: 20,  # left wrist
    16: 21,  # right wrist
    25: 4,   # left knee
    26: 5,   # right knee
    27: 7,   # left ankle
    28: 8,   # right ankle
}

# Per-joint reliability weights (tuned for MediaPipe Pose noise)
MP_WEIGHT = {
    11: 1.2, 12: 1.2,
    23: 1.2, 24: 1.2,
    13: 1.0, 14: 1.0,
    15: 0.7, 16: 0.7,
    25: 1.0, 26: 1.0,
    27: 0.8, 28: 0.8,
}


def _load_keypoints_frames(keypoints_json: str | Path) -> tuple[list[str], list[np.ndarray]]:
    with open(keypoints_json, "r", encoding="utf-8") as f:
        kpts_all = json.load(f)
    if len(kpts_all) == 0:
        raise RuntimeError("No keypoints found in keypoints.json")

    frame_names = sorted(kpts_all.keys())
    frames: list[np.ndarray] = []
    for name in frame_names:
        a = np.asarray(kpts_all[name], dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 3:
            raise RuntimeError(f"Bad keypoints format for {name}, expected (33,3), got {a.shape}")
        frames.append(a)
    return frame_names, frames


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k, dtype=int)


def _select_indices(n_total: int, fit_frames: int) -> np.ndarray:
    if fit_frames <= 0:
        return np.arange(n_total, dtype=int)
    return _uniform_indices(n_total, min(int(fit_frames), n_total))


def _prepare_targets(
    frames: list[np.ndarray],
    mp_indices: list[int],
    base_w: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tgts = []
    ws = []
    for fr in frames:
        tgt = fr[mp_indices, :2]  # (M,2)
        conf = fr[mp_indices, 2:3]  # (M,1)
        w = conf * base_w
        w = np.where(w > 1e-3, w, 0.0)
        tgts.append(tgt)
        ws.append(w)
    tgt_t = torch.tensor(np.stack(tgts, axis=0), dtype=torch.float32, device=device)  # (F,M,2)
    w_t = torch.tensor(np.stack(ws, axis=0), dtype=torch.float32, device=device)       # (F,M,1)
    return tgt_t, w_t


def _project_joints(joints: torch.Tensor, K: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # joints: (B,M,3), K: (3,3), scale: (1,)
    X = joints * scale.view(1, 1, 1)
    z = X[..., 2].clamp(min=1e-4)
    x = X[..., 0] / z
    y = X[..., 1] / z
    return torch.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], dim=-1)  # (B,M,2)


def _camera_to_blender(points: np.ndarray) -> np.ndarray:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return np.stack([x, z, -y], axis=-1).astype(np.float32)


def _rotation_between_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src = src / (np.linalg.norm(src) + 1e-8)
    dst = dst / (np.linalg.norm(dst) + 1e-8)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180-degree flip around an arbitrary orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - src * float(np.dot(axis, src))
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
        return R.astype(np.float32)

    K = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + K + K @ K * ((1.0 - c) / (s * s + 1e-8))
    return R.astype(np.float32)


def _compute_upright_rotation_from_joints(joints_cam: np.ndarray | None) -> np.ndarray:
    if joints_cam is None:
        return np.eye(3, dtype=np.float32)
    jb = _camera_to_blender(joints_cam)
    # SMPL joint ids: 1/2 hips, 16/17 shoulders
    hip_center = 0.5 * (jb[1] + jb[2])
    shoulder_center = 0.5 * (jb[16] + jb[17])
    up_vec = shoulder_center - hip_center
    if float(np.linalg.norm(up_vec)) < 1e-6:
        return np.eye(3, dtype=np.float32)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _rotation_between_vectors(up_vec, target)


def _export_obj(path: Path, verts: np.ndarray, faces: np.ndarray | None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if faces is not None:
            for tri in faces:
                i, j, k = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i} {j} {k}\n")


def _build_vertex_neighbors(num_verts: int, faces: np.ndarray | None) -> list[np.ndarray] | None:
    if faces is None or len(faces) == 0:
        return None
    neigh_sets = [set() for _ in range(int(num_verts))]
    for tri in np.asarray(faces, dtype=np.int64):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        neigh_sets[i].update([j, k])
        neigh_sets[j].update([i, k])
        neigh_sets[k].update([i, j])
    return [np.asarray(sorted(s), dtype=np.int64) for s in neigh_sets]


def _laplacian_smooth_with_neighbors(
    verts: np.ndarray,
    neighbors: list[np.ndarray] | None,
    iters: int,
    alpha: float,
) -> np.ndarray:
    if neighbors is None or int(iters) <= 0:
        return verts
    out = verts.astype(np.float32, copy=True)
    a = float(np.clip(alpha, 0.0, 1.0))
    for _ in range(int(iters)):
        prev = out.copy()
        for vi, nb in enumerate(neighbors):
            if nb.size == 0:
                continue
            mean_nb = prev[nb].mean(axis=0)
            out[vi] = (1.0 - a) * prev[vi] + a * mean_nb
    return out


def _transform_export_vertices(
    verts: np.ndarray,
    joints_cam: np.ndarray | None = None,
    enforce_upright: bool = True,
    upright_rot: np.ndarray | None = None,
    center_xy: bool = True,
    xy_center: np.ndarray | None = None,
    put_on_floor: bool = True,
    floor_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float | None, np.ndarray]:
    """
    Convert from camera-like coordinates (x right, y down, z forward) to
    Blender-friendly coordinates (x right, y forward, z up), then:
    - optional XY centering
    - optional grounding so feet stand on the XY plane (z=0)
    """
    out = _camera_to_blender(verts)

    if enforce_upright:
        if upright_rot is None:
            upright_rot = _compute_upright_rotation_from_joints(joints_cam)
        out = (out @ upright_rot.T).astype(np.float32)
    else:
        if upright_rot is None:
            upright_rot = np.eye(3, dtype=np.float32)

    if center_xy:
        if xy_center is None:
            xy_center = 0.5 * (out[:, :2].min(axis=0) + out[:, :2].max(axis=0))
        out[:, :2] -= xy_center.reshape(1, 2)

    if put_on_floor:
        if floor_z is None:
            floor_z = float(out[:, 2].min())
        out[:, 2] -= float(floor_z)

    return out, xy_center, floor_z, upright_rot


def _loss_upright(joints: torch.Tensor) -> torch.Tensor:
    # camera coordinates: +y points down, so body up is ~[0,-1,0]
    hips = 0.5 * (joints[:, 1, :] + joints[:, 2, :])
    shoulders = 0.5 * (joints[:, 16, :] + joints[:, 17, :])
    spine = shoulders - hips
    spine = F.normalize(spine, dim=-1, eps=1e-6)
    target = torch.tensor([0.0, -1.0, 0.0], device=joints.device, dtype=joints.dtype).view(1, 3)
    return ((spine - target) ** 2).mean()


def _loss_knee_priors(
    body_pose: torch.Tensor,
    knee_min_flex: float,
    left_knee_flex_sign: float,
    right_knee_flex_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SMPL body_pose is (B, 23*3), joints indexed 0..22
    pose = body_pose.view(body_pose.shape[0], 23, 3)
    left_knee = pose[:, 4, :]
    right_knee = pose[:, 5, :]

    # Knee should mostly be hinge-like (around one major axis), suppress twist/abduction
    hinge_loss = (left_knee[:, 1:] ** 2).mean() + (right_knee[:, 1:] ** 2).mean()

    # Encourage anatomically plausible bend direction
    dir_l = F.relu(float(knee_min_flex) - float(left_knee_flex_sign) * left_knee[:, 0]).mean()
    dir_r = F.relu(float(knee_min_flex) - float(right_knee_flex_sign) * right_knee[:, 0]).mean()
    direction_loss = dir_l + dir_r
    return hinge_loss, direction_loss


def _require_smplx(smpl_model_path: str | Path) -> Path:
    if smplx is None:
        raise RuntimeError(
            "smplx is not installed. Install it first (pip install smplx), "
            "and provide stage1.smpl_model_path in the config."
        )
    model_path = Path(smpl_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL model path not found: {model_path}")
    return model_path


def fit_smpl_from_keypoints(
    keypoints_json: str | Path,
    out_dir: str | Path,
    K_list: list[float],
    smpl_model_path: str | Path,
    smpl_model_type: str = "smpl",
    smpl_gender: str = "neutral",
    fit_frames: int = 20,
    fit_iters: int = 300,
    fit_lr: float = 5e-2,
    export_smooth_iters: int = 0,
    export_smooth_alpha: float = 0.1,
):
    """
    Fit one static SMPL body from multi-frame 2D keypoints.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _require_smplx(smpl_model_path)

    frame_names_all, frames_all = _load_keypoints_frames(keypoints_json)
    use_idx = _select_indices(len(frames_all), int(fit_frames))
    frame_names = [frame_names_all[i] for i in use_idx]
    frames = [frames_all[i] for i in use_idx]
    use_n = len(frames)

    mp_indices = list(MP_TO_SMPL.keys())
    smpl_indices = list(MP_TO_SMPL.values())
    M = len(mp_indices)
    base_w = np.asarray([MP_WEIGHT.get(mi, 1.0) for mi in mp_indices], dtype=np.float32).reshape(M, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smplx.create(
        model_path=str(model_path),
        model_type=smpl_model_type,
        gender=smpl_gender,
        num_betas=10,
        batch_size=1,
    ).to(device)
    faces = getattr(model, "faces", None)

    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)
    targets_xy, targets_w = _prepare_targets(frames, mp_indices, base_w, device)

    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3

    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, body_pose_dim), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    transl = torch.tensor([[0.0, 0.0, 2.5]], device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)

    optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl, scale], lr=float(fit_lr))
    iters = int(fit_iters)
    if iters <= 0:
        raise ValueError(f"fit_iters must be > 0, got {fit_iters}")

    refine_start = int(0.60 * iters)
    refine_lr = float(fit_lr) * 0.2
    batch_frames = 4 if use_n >= 8 else 2
    rng = np.random.default_rng(42)

    pbar = trange(iters, desc="Stage1 SMPL fit (static)", dynamic_ncols=True)
    for it in pbar:
        if it == refine_start:
            for group in optimizer.param_groups:
                group["lr"] = refine_lr

        optimizer.zero_grad(set_to_none=True)

        idxs = rng.integers(0, use_n, size=batch_frames)
        idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

        out = model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl)
        joints = out.joints[:, smpl_indices_t, :]  # (1,M,3)
        proj = _project_joints(joints, K, scale)   # (1,M,2)
        proj = proj.expand(batch_frames, -1, -1)   # (B,M,2)

        tgt = targets_xy[idx_t]  # (B,M,2)
        w = targets_w[idx_t]     # (B,M,1)
        per = F.smooth_l1_loss(proj, tgt, reduction="none")
        loss_kpt = (per * w).sum() / (w.sum() + 1e-6)

        loss_reg = (
            1e-4 * (global_orient ** 2).sum()
            + 5e-5 * (body_pose ** 2).sum()
            + 5e-4 * (betas ** 2).sum()
            + 1e-3 * ((scale - 1.0) ** 2).sum()
            + 1e-5 * (transl ** 2).sum()
        )
        loss = loss_kpt + loss_reg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scale.clamp_(0.2, 5.0)
            transl[:, 2].clamp_(0.2, 10.0)

        if it % 10 == 0 or it == iters - 1:
            pbar.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                loss=float(loss.detach().cpu()),
                kpt=float(loss_kpt.detach().cpu()),
                reg=float(loss_reg.detach().cpu()),
                scale=float(scale.detach().cpu()),
                z=float(transl[:, 2].detach().cpu().item()),
                frames=use_n,
            )

    with torch.no_grad():
        out = model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl)
        verts = (out.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_full = out.joints.squeeze(0).cpu().numpy()
    verts, _, _, _ = _transform_export_vertices(
        verts,
        joints_cam=joints_full,
        enforce_upright=True,
        center_xy=True,
        put_on_floor=True,
    )
    neighbors = _build_vertex_neighbors(num_verts=verts.shape[0], faces=faces)
    verts = _laplacian_smooth_with_neighbors(
        verts,
        neighbors=neighbors,
        iters=int(export_smooth_iters),
        alpha=float(export_smooth_alpha),
    )

    pose = np.concatenate(
        [global_orient.detach().cpu().numpy().reshape(-1), body_pose.detach().cpu().numpy().reshape(-1)],
        axis=0,
    ).astype(np.float32)
    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    trans_np = transl.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)

    np.savez(out_dir / "smpl_params.npz", pose=pose, betas=betas_np, trans=trans_np, scale=scale_np)
    np.savez(out_dir / "smpl_fit_frames.npz", frame_names=np.asarray(frame_names))
    _export_obj(out_dir / "smpl_mesh.obj", verts, faces)
    return out_dir


def fit_smpl_sequence_from_keypoints(
    keypoints_json: str | Path,
    out_dir: str | Path,
    K_list: list[float],
    smpl_model_path: str | Path,
    smpl_model_type: str = "smpl",
    smpl_gender: str = "neutral",
    fit_frames: int = 120,
    fit_iters: int = 2000,
    fit_lr: float = 3e-2,
    frame_batch: int = 16,
    temporal_pose_weight: float = 5e-4,
    temporal_orient_weight: float = 1e-3,
    temporal_trans_weight: float = 5e-3,
    optimize_betas: bool = False,
    unfreeze_betas_iter: int = -1,
    betas_l2_weight: float = 5e-2,
    betas_abs_max: float = 3.0,
    shape_from_previous_frames: bool = True,
    final_pose_from_last_frame: bool = True,
    last_frame_refine_iters: int = 300,
    upright_prior_weight: float = 1e-2,
    knee_hinge_weight: float = 2e-3,
    knee_direction_weight: float = 5e-3,
    knee_min_flex: float = 0.02,
    left_knee_flex_sign: float = 1.0,
    right_knee_flex_sign: float = 1.0,
    enforce_export_upright: bool = True,
    export_center_xy: bool = True,
    export_on_floor: bool = True,
    export_smooth_iters: int = 0,
    export_smooth_alpha: float = 0.1,
    export_meshes: bool = False,
):
    """
    Fit a moving-person SMPL sequence:
    - shared betas/scale
    - per-frame global_orient/body_pose/trans
    - fixed camera intrinsics
    - by default, body shape is estimated from earlier frames and final pose uses last frame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _require_smplx(smpl_model_path)

    frame_names_all, frames_all = _load_keypoints_frames(keypoints_json)
    use_idx = _select_indices(len(frames_all), int(fit_frames))
    frame_names = [frame_names_all[i] for i in use_idx]
    frames = [frames_all[i] for i in use_idx]
    use_n = len(frames)

    mp_indices = list(MP_TO_SMPL.keys())
    smpl_indices = list(MP_TO_SMPL.values())
    M = len(mp_indices)
    base_w = np.asarray([MP_WEIGHT.get(mi, 1.0) for mi in mp_indices], dtype=np.float32).reshape(M, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = max(1, min(int(frame_batch), use_n))

    model = smplx.create(
        model_path=str(model_path),
        model_type=smpl_model_type,
        gender=smpl_gender,
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    faces = getattr(model, "faces", None)

    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)
    targets_xy, targets_w = _prepare_targets(frames, mp_indices, base_w, device)

    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3

    global_orient = torch.zeros((use_n, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((use_n, body_pose_dim), device=device, requires_grad=True)
    transl = torch.zeros((use_n, 3), device=device, requires_grad=True)
    with torch.no_grad():
        transl[:, 2] = 2.5
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [global_orient, body_pose, transl, scale], "lr": float(fit_lr)},
            {"params": [betas], "lr": float(fit_lr) * 0.2},
        ]
    )
    if not optimize_betas:
        betas.requires_grad_(False)
        optimizer.param_groups[1]["lr"] = 0.0
    iters = int(fit_iters)
    if iters <= 0:
        raise ValueError(f"fit_iters must be > 0, got {fit_iters}")

    refine_start = int(0.65 * iters)
    refine_lr = float(fit_lr) * 0.2
    rng = np.random.default_rng(42)

    pbar = trange(iters, desc="Stage1 SMPL fit (sequence)", dynamic_ncols=True)
    fit_pool_n = use_n - 1 if (shape_from_previous_frames and use_n > 1) else use_n
    for it in pbar:
        if (not optimize_betas) and int(unfreeze_betas_iter) >= 0 and it == int(unfreeze_betas_iter):
            betas.requires_grad_(True)
            optimizer.param_groups[1]["lr"] = float(fit_lr) * 0.2

        if it == refine_start:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * (refine_lr / float(fit_lr))

        optimizer.zero_grad(set_to_none=True)

        if batch_size >= fit_pool_n:
            idxs = np.arange(fit_pool_n, dtype=int)
        else:
            idxs = rng.choice(fit_pool_n, size=batch_size, replace=False)
        idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
        B = idx_t.numel()

        out = model(
            global_orient=global_orient[idx_t],
            body_pose=body_pose[idx_t],
            betas=betas.expand(B, -1),
            transl=transl[idx_t],
        )
        joints = out.joints[:, smpl_indices_t, :]  # (B,M,3)
        proj = _project_joints(joints, K, scale)   # (B,M,2)

        tgt = targets_xy[idx_t]  # (B,M,2)
        w = targets_w[idx_t]     # (B,M,1)
        per = F.smooth_l1_loss(proj, tgt, reduction="none")
        loss_kpt = (per * w).sum() / (w.sum() + 1e-6)

        loss_reg = (
            2e-4 * (global_orient ** 2).mean()
            + 1e-4 * (body_pose ** 2).mean()
            + float(betas_l2_weight) * (betas ** 2).sum()
            + 1e-3 * ((scale - 1.0) ** 2).sum()
            + 1e-5 * (transl[:, :2] ** 2).mean()
            + 1e-4 * ((transl[:, 2] - 2.5) ** 2).mean()
        )
        loss_upright = float(upright_prior_weight) * _loss_upright(out.joints)
        loss_knee_hinge, loss_knee_dir = _loss_knee_priors(
            body_pose=body_pose[idx_t],
            knee_min_flex=float(knee_min_flex),
            left_knee_flex_sign=float(left_knee_flex_sign),
            right_knee_flex_sign=float(right_knee_flex_sign),
        )
        loss_knee = (
            float(knee_hinge_weight) * loss_knee_hinge
            + float(knee_direction_weight) * loss_knee_dir
        )

        loss_temp = torch.zeros((), dtype=torch.float32, device=device)
        if use_n > 1:
            loss_temp = (
                float(temporal_pose_weight) * ((body_pose[1:] - body_pose[:-1]) ** 2).mean()
                + float(temporal_orient_weight) * ((global_orient[1:] - global_orient[:-1]) ** 2).mean()
                + float(temporal_trans_weight) * ((transl[1:] - transl[:-1]) ** 2).mean()
            )

        loss = loss_kpt + loss_reg + loss_temp + loss_upright + loss_knee
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scale.clamp_(0.2, 5.0)
            transl[:, 2].clamp_(0.2, 10.0)
            betas.clamp_(-float(betas_abs_max), float(betas_abs_max))

        if it % 10 == 0 or it == iters - 1:
            pbar.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                loss=float(loss.detach().cpu()),
                kpt=float(loss_kpt.detach().cpu()),
                reg=float(loss_reg.detach().cpu()),
                temp=float(loss_temp.detach().cpu()),
                up=float(loss_upright.detach().cpu()),
                knee=float(loss_knee.detach().cpu()),
                scale=float(scale.detach().cpu()),
                z=float(transl[:, 2].mean().detach().cpu()),
                frames=use_n,
                b=B,
                betan=float(torch.linalg.vector_norm(betas).detach().cpu()),
            )

    if final_pose_from_last_frame and use_n > 0 and int(last_frame_refine_iters) > 0:
        last_idx = use_n - 1
        go_last = global_orient[last_idx:last_idx + 1].detach().clone().requires_grad_(True)
        bp_last = body_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(True)
        tr_last = transl[last_idx:last_idx + 1].detach().clone().requires_grad_(True)
        last_opt = torch.optim.Adam([go_last, bp_last, tr_last], lr=float(fit_lr) * 0.25)
        tgt_last = targets_xy[last_idx:last_idx + 1]
        w_last = targets_w[last_idx:last_idx + 1]
        for _ in range(int(last_frame_refine_iters)):
            last_opt.zero_grad(set_to_none=True)
            out_last = model(
                global_orient=go_last,
                body_pose=bp_last,
                betas=betas.detach(),
                transl=tr_last,
            )
            joints_last = out_last.joints[:, smpl_indices_t, :]
            proj_last = _project_joints(joints_last, K, scale)
            per_last = F.smooth_l1_loss(proj_last, tgt_last, reduction="none")
            loss_last_kpt = (per_last * w_last).sum() / (w_last.sum() + 1e-6)
            loss_last_reg = (
                1e-4 * (go_last ** 2).sum()
                + 5e-5 * (bp_last ** 2).sum()
                + 1e-5 * (tr_last[:, :2] ** 2).sum()
                + 1e-4 * ((tr_last[:, 2] - 2.5) ** 2).sum()
            )
            loss_last = loss_last_kpt + loss_last_reg
            loss_last.backward()
            last_opt.step()
            with torch.no_grad():
                tr_last[:, 2].clamp_(0.2, 10.0)
        with torch.no_grad():
            global_orient[last_idx:last_idx + 1] = go_last
            body_pose[last_idx:last_idx + 1] = bp_last
            transl[last_idx:last_idx + 1] = tr_last

    export_idx = (use_n - 1) if (final_pose_from_last_frame and use_n > 0) else 0
    with torch.no_grad():
        out0 = model(
            global_orient=global_orient[export_idx:export_idx + 1],
            body_pose=body_pose[export_idx:export_idx + 1],
            betas=betas,
            transl=transl[export_idx:export_idx + 1],
        )
        verts0 = (out0.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints0 = out0.joints.squeeze(0).cpu().numpy()
    verts0, export_center_xy_val, export_floor, export_upright_rot = _transform_export_vertices(
        verts0,
        joints_cam=joints0,
        enforce_upright=bool(enforce_export_upright),
        center_xy=bool(export_center_xy),
        put_on_floor=bool(export_on_floor),
    )
    neighbors = _build_vertex_neighbors(num_verts=verts0.shape[0], faces=faces)
    verts0 = _laplacian_smooth_with_neighbors(
        verts0,
        neighbors=neighbors,
        iters=int(export_smooth_iters),
        alpha=float(export_smooth_alpha),
    )

    global_orient_np = global_orient.detach().cpu().numpy().astype(np.float32)
    body_pose_np = body_pose.detach().cpu().numpy().astype(np.float32)
    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    trans_np = transl.detach().cpu().numpy().astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)

    np.savez(
        out_dir / "smpl_params_sequence.npz",
        global_orient=global_orient_np,
        body_pose=body_pose_np,
        betas=betas_np,
        trans=trans_np,
        scale=scale_np,
        frame_names=np.asarray(frame_names),
        frame_indices=use_idx.astype(np.int32),
        export_frame_index=np.asarray([export_idx], dtype=np.int32),
    )

    pose0 = np.concatenate([global_orient_np[export_idx], body_pose_np[export_idx]], axis=0).astype(np.float32)
    np.savez(
        out_dir / "smpl_params.npz",
        pose=pose0,
        betas=betas_np,
        trans=trans_np[export_idx],
        scale=scale_np,
        export_frame_index=np.asarray([export_idx], dtype=np.int32),
    )

    _export_obj(out_dir / "smpl_mesh.obj", verts0, faces)

    if export_meshes:
        mesh_dir = out_dir / "smpl_mesh_sequence"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for fi in range(use_n):
                out_i = model(
                    global_orient=global_orient[fi:fi + 1],
                    body_pose=body_pose[fi:fi + 1],
                    betas=betas,
                    transl=transl[fi:fi + 1],
                )
                verts_i = (out_i.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
                verts_i, _, _, _ = _transform_export_vertices(
                    verts_i,
                    enforce_upright=bool(enforce_export_upright),
                    upright_rot=export_upright_rot,
                    center_xy=bool(export_center_xy),
                    xy_center=export_center_xy_val,
                    put_on_floor=bool(export_on_floor),
                    floor_z=export_floor,
                )
                verts_i = _laplacian_smooth_with_neighbors(
                    verts_i,
                    neighbors=neighbors,
                    iters=int(export_smooth_iters),
                    alpha=float(export_smooth_alpha),
                )
                _export_obj(mesh_dir / f"{fi:06d}.obj", verts_i, faces)

    return out_dir
