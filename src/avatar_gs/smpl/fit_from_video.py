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


def _temporal_window_smooth_loss(x: torch.Tensor, window: int, decay: float) -> torch.Tensor:
    """
    Pairwise temporal smoothness over +/-k neighbors with decaying weights.
    x shape: (T, D...)
    """
    T = int(x.shape[0])
    if T <= 1 or int(window) <= 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    max_k = min(int(window), T - 1)
    d = float(np.clip(decay, 0.0, 1.0))
    total = torch.zeros((), dtype=x.dtype, device=x.device)
    total_w = 0.0
    for k in range(1, max_k + 1):
        wk = d ** (k - 1) if d > 0.0 else 1.0
        total = total + float(wk) * ((x[k:] - x[:-k]) ** 2).mean()
        total_w += float(wk)
    if total_w <= 0.0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return total / float(total_w)


def _temporal_accel_loss(x: torch.Tensor) -> torch.Tensor:
    """Second-order temporal smoothness (discrete acceleration)."""
    T = int(x.shape[0])
    if T <= 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return ((x[2:] - 2.0 * x[1:-1] + x[:-2]) ** 2).mean()


def _export_sequence_motion_files(
    out_dir: Path,
    frame_names: list[str],
    betas_np: np.ndarray,
    global_orient_np: np.ndarray,
    body_pose_np: np.ndarray,
    trans_np: np.ndarray,
    scale_np: np.ndarray,
) -> None:
    """
    Export explicit motion parameters to a single NPZ file:
    - global betas (shape)
    - per-frame theta = [global_orient(3), body_pose(23*3)]
    """
    theta_np = np.concatenate([global_orient_np, body_pose_np], axis=1).astype(np.float32)
    betas_per_frame = np.repeat(betas_np.reshape(1, -1), theta_np.shape[0], axis=0).astype(np.float32)

    np.savez(
        out_dir / "smpl_motion_params.npz",
        frame_names=np.asarray(frame_names),
        betas=betas_np.astype(np.float32),
        betas_per_frame=betas_per_frame,
        theta=theta_np,
        theta_global=global_orient_np.astype(np.float32),
        theta_body=body_pose_np.astype(np.float32),
        trans=trans_np.astype(np.float32),
        scale=scale_np.astype(np.float32),
    )


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
    pelvis = jb[0]
    cands = []
    # SMPL joint ids: 12 neck, 15 head, 16/17 shoulders
    for j in (12, 15):
        v = jb[j] - pelvis
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            cands.append(v / n)
    v_sh = 0.5 * (jb[16] + jb[17]) - pelvis
    n_sh = float(np.linalg.norm(v_sh))
    if n_sh > 1e-6:
        cands.append(v_sh / n_sh)

    if not cands:
        return np.eye(3, dtype=np.float32)
    up_vec = np.sum(np.stack(cands, axis=0), axis=0)
    if float(np.linalg.norm(up_vec)) < 1e-6:
        up_vec = cands[0]
    if up_vec[2] < 0:
        up_vec = -up_vec
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


def _force_world_alignment(
    verts: np.ndarray,
    center_xy: bool,
    center_mode: str,
    put_on_floor: bool,
) -> np.ndarray:
    out = verts.astype(np.float32, copy=True)
    if center_xy:
        mode = str(center_mode).lower()
        if mode == "centroid":
            cxy = out[:, :2].mean(axis=0)
        else:
            cxy = 0.5 * (out[:, :2].min(axis=0) + out[:, :2].max(axis=0))
        out[:, :2] -= cxy.reshape(1, 2)
    if put_on_floor:
        out[:, 2] -= float(out[:, 2].min())
    return out


def _transform_export_vertices(
    verts: np.ndarray,
    joints_cam: np.ndarray | None = None,
    enforce_upright: bool = True,
    upright_rot: np.ndarray | None = None,
    ensure_head_up: bool = False,
    center_xy: bool = True,
    xy_center: np.ndarray | None = None,
    center_xyz: bool = False,
    center_mode: str = "pelvis",
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

    if ensure_head_up and joints_cam is not None:
        jb = _camera_to_blender(joints_cam)
        if enforce_upright:
            jb = (jb @ upright_rot.T).astype(np.float32)
        if jb.shape[0] > 15:
            pelvis = jb[0]
            head = jb[15]
            if float(head[2]) < float(pelvis[2]):
                flip = np.array([[1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0]], dtype=np.float32)
                out = (out @ flip.T).astype(np.float32)
                upright_rot = (flip @ upright_rot).astype(np.float32)

    if center_xy:
        if xy_center is None:
            mode = str(center_mode).lower()
            if mode == "pelvis" and joints_cam is not None:
                pelvis = _camera_to_blender(joints_cam[0:1])[0]
                if enforce_upright:
                    pelvis = (upright_rot @ pelvis.astype(np.float32)).astype(np.float32)
                xy_center = pelvis[:2]
            elif mode == "centroid":
                xy_center = out[:, :2].mean(axis=0)
            else:
                xy_center = 0.5 * (out[:, :2].min(axis=0) + out[:, :2].max(axis=0))
        out[:, :2] -= xy_center.reshape(1, 2)

    if center_xyz:
        mode = str(center_mode).lower()
        if mode == "pelvis" and joints_cam is not None:
            pelvis = _camera_to_blender(joints_cam[0:1])[0]
            if enforce_upright:
                pelvis = (upright_rot @ pelvis.astype(np.float32)).astype(np.float32)
            z_center = float(pelvis[2])
        elif mode == "centroid":
            z_center = float(out[:, 2].mean())
        else:
            z_center = 0.5 * float(out[:, 2].min() + out[:, 2].max())
        out[:, 2] -= z_center

    if put_on_floor:
        if floor_z is None:
            mode = str(center_mode).lower()
            if mode == "pelvis" and joints_cam is not None:
                pelvis = _camera_to_blender(joints_cam[0:1])[0]
                if enforce_upright:
                    pelvis = (upright_rot @ pelvis.astype(np.float32)).astype(np.float32)
                floor_z = float(pelvis[2])
            else:
                floor_z = float(out[:, 2].min())
        out[:, 2] -= float(floor_z)

    return out, xy_center, floor_z, upright_rot


def _pelvis_export_offset(
    joints_cam: np.ndarray,
    enforce_upright: bool,
    upright_rot: np.ndarray,
    center_xy: bool,
    xy_center: np.ndarray | None,
    put_on_floor: bool,
    floor_z: float | None,
) -> np.ndarray:
    pelvis = _camera_to_blender(joints_cam[0:1])[0]
    if enforce_upright:
        pelvis = (upright_rot @ pelvis.astype(np.float32)).astype(np.float32)
    if center_xy and xy_center is not None:
        pelvis[:2] -= xy_center
    if put_on_floor and floor_z is not None:
        pelvis[2] -= float(floor_z)
    return pelvis.astype(np.float32)


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
    knee_hinge_axis: int,
    knee_min_flex: float,
    left_knee_flex_sign: float,
    right_knee_flex_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SMPL body_pose is (B, 23*3), joints indexed 0..22
    pose = body_pose.view(body_pose.shape[0], 23, 3)
    # body_pose indices: 3 left_knee, 4 right_knee
    left_knee = pose[:, 3, :]
    right_knee = pose[:, 4, :]
    axis = int(knee_hinge_axis)
    if axis < 0 or axis > 2:
        axis = 1

    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    # Knee should mostly be hinge-like around one axis, suppress the other two axes.
    hinge_loss = (left_knee[:, other_axes] ** 2).mean() + (right_knee[:, other_axes] ** 2).mean()

    # Encourage anatomically plausible bend direction
    dir_l = F.relu(float(knee_min_flex) - float(left_knee_flex_sign) * left_knee[:, axis]).mean()
    dir_r = F.relu(float(knee_min_flex) - float(right_knee_flex_sign) * right_knee[:, axis]).mean()
    direction_loss = dir_l + dir_r
    return hinge_loss, direction_loss


def _loss_ankle_back_priors(
    body_pose: torch.Tensor,
    ankle_hinge_weight_axes: tuple[float, float] = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = body_pose.view(body_pose.shape[0], 23, 3)
    left_ankle = pose[:, 6, :]
    right_ankle = pose[:, 7, :]
    spine = pose[:, [2, 5, 8], :]  # spine1/spine2/spine3

    # Suppress ankle twist/roll and excessive back bending/twisting
    ankle_twist = (
        float(ankle_hinge_weight_axes[0]) * (left_ankle[:, 1:] ** 2).mean()
        + float(ankle_hinge_weight_axes[1]) * (right_ankle[:, 1:] ** 2).mean()
    )
    back_stability = (spine[:, :, 1:] ** 2).mean() + 0.5 * (spine[:, :, 0] ** 2).mean()
    return ankle_twist, back_stability


def _enforce_knee_hard_constraints(
    body_pose: torch.Tensor,
    knee_hinge_axis: int,
    knee_min_flex: float,
    left_knee_flex_sign: float,
    right_knee_flex_sign: float,
    zero_non_hinge_axes: bool = True,
) -> None:
    # In-place projection to feasible knee subspace.
    pose = body_pose.view(body_pose.shape[0], 23, 3)
    axis = int(knee_hinge_axis)
    if axis < 0 or axis > 2:
        axis = 1
    l0 = pose[:, 3, axis] * float(left_knee_flex_sign)
    r0 = pose[:, 4, axis] * float(right_knee_flex_sign)

    pose[:, 3, axis] = float(left_knee_flex_sign) * torch.clamp(l0, min=float(knee_min_flex))
    pose[:, 4, axis] = float(right_knee_flex_sign) * torch.clamp(r0, min=float(knee_min_flex))

    if zero_non_hinge_axes:
        other_axes = [0, 1, 2]
        other_axes.remove(axis)
        pose[:, 3, other_axes] = 0.0
        pose[:, 4, other_axes] = 0.0


def _enforce_spine_upright_hard(body_pose: torch.Tensor, spine_tilt_max: float = 0.15) -> None:
    pose = body_pose.view(body_pose.shape[0], 23, 3)
    # spine1/spine2/spine3 indices
    for si in (2, 5, 8):
        pose[:, si, 0].clamp_(-float(spine_tilt_max), float(spine_tilt_max))
        pose[:, si, 1:] = 0.0


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
    export_pose_mode: str = "fit",
    export_ensure_head_up: bool = False,
    export_center_xyz: bool = False,
    export_on_floor: bool = True,
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
        pose_mode = str(export_pose_mode or "fit").lower()
        if pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
            export_go = torch.zeros((1, 3), device=device)
            export_bp = torch.zeros((1, body_pose_dim), device=device)
            export_tr = transl.detach()
        else:
            export_go = global_orient
            export_bp = body_pose
            export_tr = transl
        out = model(global_orient=export_go, body_pose=export_bp, betas=betas, transl=export_tr)
        verts = (out.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_full = out.joints.squeeze(0).cpu().numpy()
    verts, _, _, _ = _transform_export_vertices(
        verts,
        joints_cam=joints_full,
        enforce_upright=True,
        ensure_head_up=bool(export_ensure_head_up),
        center_xy=True,
        center_xyz=bool(export_center_xyz),
        put_on_floor=True,
    )
    neighbors = _build_vertex_neighbors(num_verts=verts.shape[0], faces=faces)
    verts = _laplacian_smooth_with_neighbors(
        verts,
        neighbors=neighbors,
        iters=int(export_smooth_iters),
        alpha=float(export_smooth_alpha),
    )
    verts = _force_world_alignment(
        verts=verts,
        center_xy=True,
        center_mode="centroid",
        put_on_floor=bool(export_on_floor),
    )

    if pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
        pose = np.concatenate(
            [export_go.detach().cpu().numpy().reshape(-1), export_bp.detach().cpu().numpy().reshape(-1)],
            axis=0,
        ).astype(np.float32)
        trans_np = export_tr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        go_seq = np.repeat(export_go.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
        bp_seq = np.repeat(export_bp.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
        tr_seq = np.repeat(export_tr.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
    else:
        pose = np.concatenate(
            [global_orient.detach().cpu().numpy().reshape(-1), body_pose.detach().cpu().numpy().reshape(-1)],
            axis=0,
        ).astype(np.float32)
        trans_np = transl.detach().cpu().numpy().reshape(-1).astype(np.float32)
        go_seq = np.repeat(global_orient.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
        bp_seq = np.repeat(body_pose.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
        tr_seq = np.repeat(transl.detach().cpu().numpy().astype(np.float32), use_n, axis=0)
    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)

    np.savez(out_dir / "smpl_params.npz", pose=pose, betas=betas_np, trans=trans_np, scale=scale_np)
    np.savez(out_dir / "smpl_fit_frames.npz", frame_names=np.asarray(frame_names))
    _export_sequence_motion_files(
        out_dir=out_dir,
        frame_names=frame_names,
        betas_np=betas_np,
        global_orient_np=go_seq,
        body_pose_np=bp_seq,
        trans_np=tr_seq,
        scale_np=scale_np,
    )
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
    temporal_window: int = 1,
    temporal_window_decay: float = 0.7,
    temporal_pose_accel_weight: float = 0.0,
    temporal_orient_accel_weight: float = 0.0,
    temporal_trans_accel_weight: float = 0.0,
    optimize_betas: bool = False,
    unfreeze_betas_iter: int = -1,
    betas_l2_weight: float = 5e-2,
    betas_abs_max: float = 3.0,
    shape_from_previous_frames: bool = True,
    final_pose_from_last_frame: bool = True,
    last_frame_refine_iters: int = 300,
    export_pose_mode: str = "last_frame",
    export_ensure_head_up: bool = False,
    upright_prior_weight: float = 1e-2,
    knee_hinge_weight: float = 2e-3,
    knee_direction_weight: float = 5e-3,
    knee_hinge_axis: int = 1,
    knee_min_flex: float = 0.0,
    left_knee_flex_sign: float = -1.0,
    right_knee_flex_sign: float = -1.0,
    auto_knee_sign: bool = False,
    knee_sign_update_every: int = 200,
    enforce_knee_hard: bool = True,
    enforce_right_knee_only: bool = True,
    knee_zero_non_hinge_axes: bool = False,
    ankle_twist_weight: float = 3e-3,
    back_stability_weight: float = 6e-3,
    enforce_strict_upright: bool = False,
    upright_tilt_max: float = 0.0,
    enforce_spine_hard: bool = False,
    spine_tilt_max: float = 0.15,
    enforce_export_upright: bool = True,
    export_center_xy: bool = True,
    export_center_xyz: bool = False,
    export_center_mode: str = "pelvis",
    export_on_floor: bool = True,
    export_smooth_iters: int = 0,
    export_smooth_alpha: float = 0.1,
    full_batch_every: int = 200,
    export_meshes: bool = False,
):
    """
    Fit a moving-person SMPL sequence:
    - shared betas/scale
    - per-frame global_orient/body_pose/trans
    - fixed camera intrinsics
    - temporal continuity over neighboring frames (window + acceleration regularization)
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
    left_sign_cur = float(left_knee_flex_sign)
    right_sign_cur = float(right_knee_flex_sign)
    for it in pbar:
        if (not optimize_betas) and int(unfreeze_betas_iter) >= 0 and it == int(unfreeze_betas_iter):
            betas.requires_grad_(True)
            optimizer.param_groups[1]["lr"] = float(fit_lr) * 0.2

        if it == refine_start:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * (refine_lr / float(fit_lr))

        optimizer.zero_grad(set_to_none=True)

        use_full_batch = int(full_batch_every) > 0 and (it % int(full_batch_every) == 0)
        if use_full_batch or batch_size >= fit_pool_n:
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

        if bool(auto_knee_sign) and int(knee_sign_update_every) > 0 and (it % int(knee_sign_update_every) == 0):
            best_obj = None
            best_signs = (left_sign_cur, right_sign_cur)
            candidates = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
            for ls, rs in candidates:
                _, dir_loss_cand = _loss_knee_priors(
                    body_pose=body_pose[idx_t],
                    knee_hinge_axis=int(knee_hinge_axis),
                    knee_min_flex=float(knee_min_flex),
                    left_knee_flex_sign=float(ls),
                    right_knee_flex_sign=float(rs),
                )
                obj = float(loss_kpt.detach().cpu()) + float(knee_direction_weight) * float(dir_loss_cand.detach().cpu())
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_signs = (float(ls), float(rs))
            left_sign_cur, right_sign_cur = best_signs

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
            knee_hinge_axis=int(knee_hinge_axis),
            knee_min_flex=float(knee_min_flex),
            left_knee_flex_sign=float(left_sign_cur),
            right_knee_flex_sign=float(right_sign_cur),
        )
        loss_knee = (
            float(knee_hinge_weight) * loss_knee_hinge
            + float(knee_direction_weight) * loss_knee_dir
        )
        loss_ankle, loss_back = _loss_ankle_back_priors(body_pose=body_pose[idx_t])
        loss_phys_extra = (
            float(ankle_twist_weight) * loss_ankle
            + float(back_stability_weight) * loss_back
        )

        loss_temp = torch.zeros((), dtype=torch.float32, device=device)
        if use_n > 1:
            loss_temp = (
                float(temporal_pose_weight)
                * _temporal_window_smooth_loss(
                    body_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                )
                + float(temporal_orient_weight)
                * _temporal_window_smooth_loss(
                    global_orient, window=int(temporal_window), decay=float(temporal_window_decay)
                )
                + float(temporal_trans_weight)
                * _temporal_window_smooth_loss(
                    transl, window=int(temporal_window), decay=float(temporal_window_decay)
                )
            )
            loss_temp = (
                loss_temp
                + float(temporal_pose_accel_weight) * _temporal_accel_loss(body_pose)
                + float(temporal_orient_accel_weight) * _temporal_accel_loss(global_orient)
                + float(temporal_trans_accel_weight) * _temporal_accel_loss(transl)
            )

        loss = loss_kpt + loss_reg + loss_temp + loss_upright + loss_knee + loss_phys_extra
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scale.clamp_(0.2, 5.0)
            transl[:, 2].clamp_(0.2, 10.0)
            betas.clamp_(-float(betas_abs_max), float(betas_abs_max))
            if bool(enforce_knee_hard):
                if bool(enforce_right_knee_only):
                    pose_view = body_pose.view(body_pose.shape[0], 23, 3)
                    axis = int(knee_hinge_axis)
                    if axis < 0 or axis > 2:
                        axis = 1
                    r0 = pose_view[:, 4, axis] * float(right_sign_cur)
                    pose_view[:, 4, axis] = float(right_sign_cur) * torch.clamp(r0, min=float(knee_min_flex))
                    if bool(knee_zero_non_hinge_axes):
                        other_axes = [0, 1, 2]
                        other_axes.remove(axis)
                        pose_view[:, 4, other_axes] = 0.0
                else:
                    _enforce_knee_hard_constraints(
                        body_pose=body_pose,
                        knee_hinge_axis=int(knee_hinge_axis),
                        knee_min_flex=float(knee_min_flex),
                        left_knee_flex_sign=float(left_sign_cur),
                        right_knee_flex_sign=float(right_sign_cur),
                        zero_non_hinge_axes=bool(knee_zero_non_hinge_axes),
                    )
            if bool(enforce_strict_upright):
                # Keep only yaw in global orientation to prevent whole-body "lying down".
                global_orient[:, 0].clamp_(-float(upright_tilt_max), float(upright_tilt_max))
                global_orient[:, 1].clamp_(-float(upright_tilt_max), float(upright_tilt_max))
            if bool(enforce_spine_hard):
                _enforce_spine_upright_hard(body_pose=body_pose, spine_tilt_max=float(spine_tilt_max))

        if it % 10 == 0 or it == iters - 1:
            pbar.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                loss=float(loss.detach().cpu()),
                kpt=float(loss_kpt.detach().cpu()),
                reg=float(loss_reg.detach().cpu()),
                temp=float(loss_temp.detach().cpu()),
                up=float(loss_upright.detach().cpu()),
                knee=float(loss_knee.detach().cpu()),
                phys=float(loss_phys_extra.detach().cpu()),
                scale=float(scale.detach().cpu()),
                z=float(transl[:, 2].mean().detach().cpu()),
                frames=use_n,
                b=B,
                betan=float(torch.linalg.vector_norm(betas).detach().cpu()),
                lks=float(left_sign_cur),
                rks=float(right_sign_cur),
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
                if bool(enforce_knee_hard):
                    if bool(enforce_right_knee_only):
                        pose_last = bp_last.view(bp_last.shape[0], 23, 3)
                        axis = int(knee_hinge_axis)
                        if axis < 0 or axis > 2:
                            axis = 1
                        r0_last = pose_last[:, 4, axis] * float(right_sign_cur)
                        pose_last[:, 4, axis] = float(right_sign_cur) * torch.clamp(r0_last, min=float(knee_min_flex))
                        if bool(knee_zero_non_hinge_axes):
                            other_axes = [0, 1, 2]
                            other_axes.remove(axis)
                            pose_last[:, 4, other_axes] = 0.0
                    else:
                        _enforce_knee_hard_constraints(
                            body_pose=bp_last,
                            knee_hinge_axis=int(knee_hinge_axis),
                            knee_min_flex=float(knee_min_flex),
                            left_knee_flex_sign=float(left_sign_cur),
                            right_knee_flex_sign=float(right_sign_cur),
                            zero_non_hinge_axes=bool(knee_zero_non_hinge_axes),
                        )
                if bool(enforce_strict_upright):
                    go_last[:, 0].clamp_(-float(upright_tilt_max), float(upright_tilt_max))
                    go_last[:, 1].clamp_(-float(upright_tilt_max), float(upright_tilt_max))
                if bool(enforce_spine_hard):
                    _enforce_spine_upright_hard(body_pose=bp_last, spine_tilt_max=float(spine_tilt_max))
        with torch.no_grad():
            global_orient[last_idx:last_idx + 1] = go_last
            body_pose[last_idx:last_idx + 1] = bp_last
            transl[last_idx:last_idx + 1] = tr_last

    pose_mode = str(export_pose_mode or "last_frame").lower()
    if pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
        export_idx = 0
        export_go = torch.zeros((1, 3), device=device)
        export_bp = torch.zeros((1, body_pose_dim), device=device)
        export_tr = transl.mean(dim=0, keepdim=True).detach()
    else:
        export_idx = (use_n - 1) if (final_pose_from_last_frame and use_n > 0) else 0
        export_go = global_orient[export_idx:export_idx + 1]
        export_bp = body_pose[export_idx:export_idx + 1]
        export_tr = transl[export_idx:export_idx + 1]
    with torch.no_grad():
        out0 = model(
            global_orient=export_go,
            body_pose=export_bp,
            betas=betas,
            transl=export_tr,
        )
        verts0 = (out0.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints0 = out0.joints.squeeze(0).cpu().numpy()
    verts0, export_center_xy_val, export_floor, export_upright_rot = _transform_export_vertices(
        verts0,
        joints_cam=joints0,
        enforce_upright=bool(enforce_export_upright),
        ensure_head_up=bool(export_ensure_head_up),
        center_xy=bool(export_center_xy),
        center_xyz=bool(export_center_xyz),
        center_mode=str(export_center_mode),
        put_on_floor=bool(export_on_floor),
    )
    neighbors = _build_vertex_neighbors(num_verts=verts0.shape[0], faces=faces)
    verts0 = _laplacian_smooth_with_neighbors(
        verts0,
        neighbors=neighbors,
        iters=int(export_smooth_iters),
        alpha=float(export_smooth_alpha),
    )
    verts0 = _force_world_alignment(
        verts=verts0,
        center_xy=bool(export_center_xy),
        center_mode=str(export_center_mode),
        put_on_floor=bool(export_on_floor),
    )

    global_orient_np = global_orient.detach().cpu().numpy().astype(np.float32)
    body_pose_np = body_pose.detach().cpu().numpy().astype(np.float32)
    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    trans_np = transl.detach().cpu().numpy().astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)
    theta_np = np.concatenate([global_orient_np, body_pose_np], axis=1).astype(np.float32)
    betas_per_frame_np = np.repeat(betas_np.reshape(1, -1), use_n, axis=0).astype(np.float32)

    np.savez(
        out_dir / "smpl_params_sequence.npz",
        global_orient=global_orient_np,
        body_pose=body_pose_np,
        theta=theta_np,
        theta_global=global_orient_np,
        theta_body=body_pose_np,
        betas=betas_np,
        betas_per_frame=betas_per_frame_np,
        trans=trans_np,
        scale=scale_np,
        frame_names=np.asarray(frame_names),
        frame_indices=use_idx.astype(np.int32),
        export_frame_index=np.asarray([export_idx], dtype=np.int32),
        # Export-space metadata for downstream inverse-transform during binding.
        export_upright_rot=export_upright_rot.astype(np.float32),
        export_center_xy=np.asarray([1 if bool(export_center_xy) else 0], dtype=np.int32),
        export_center_xyz=np.asarray([1 if bool(export_center_xyz) else 0], dtype=np.int32),
        export_on_floor=np.asarray([1 if bool(export_on_floor) else 0], dtype=np.int32),
        export_center_xy_value=(
            export_center_xy_val.astype(np.float32)
            if export_center_xy_val is not None
            else np.asarray([0.0, 0.0], dtype=np.float32)
        ),
        export_floor_z=np.asarray(
            [float(export_floor) if export_floor is not None else 0.0],
            dtype=np.float32,
        ),
    )

    if pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
        pose0 = np.concatenate(
            [export_go.detach().cpu().numpy().reshape(-1), export_bp.detach().cpu().numpy().reshape(-1)],
            axis=0,
        ).astype(np.float32)
        trans0 = export_tr.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        pose0 = np.concatenate([global_orient_np[export_idx], body_pose_np[export_idx]], axis=0).astype(np.float32)
        trans0 = trans_np[export_idx]
    np.savez(
        out_dir / "smpl_params.npz",
        pose=pose0,
        betas=betas_np,
        trans=trans0,
        scale=scale_np,
        export_frame_index=np.asarray([export_idx], dtype=np.int32),
    )
    _export_sequence_motion_files(
        out_dir=out_dir,
        frame_names=frame_names,
        betas_np=betas_np,
        global_orient_np=global_orient_np,
        body_pose_np=body_pose_np,
        trans_np=trans_np,
        scale_np=scale_np,
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
                    joints_cam=None,
                    enforce_upright=bool(enforce_export_upright),
                    upright_rot=export_upright_rot,
                    ensure_head_up=bool(export_ensure_head_up),
                    center_xy=bool(export_center_xy),
                    xy_center=export_center_xy_val,
                    center_xyz=bool(export_center_xyz),
                    center_mode=str(export_center_mode),
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
