from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None

from tqdm import trange


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

# Per-joint reliability weights (tuned for MP Pose typical noise)
# shoulders/hips more stable; wrists/ankles more noisy
MP_WEIGHT = {
    11: 1.2, 12: 1.2,
    23: 1.2, 24: 1.2,
    13: 1.0, 14: 1.0,
    15: 0.7, 16: 0.7,
    25: 1.0, 26: 1.0,
    27: 0.8, 28: 0.8,
}


def _load_keypoints_frames(keypoints_json: str | Path) -> list[np.ndarray]:
    with open(keypoints_json, "r", encoding="utf-8") as f:
        kpts_all = json.load(f)
    frame_items = list(kpts_all.values())
    if len(frame_items) == 0:
        raise RuntimeError("No keypoints found in keypoints.json")
    frames = []
    for fr in frame_items:
        a = np.array(fr, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 3:
            raise RuntimeError(f"Bad keypoints format, expected (33,3), got {a.shape}")
        frames.append(a)
    return frames


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k, dtype=int)


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
):
    """
    Fit a single global SMPL (pose/betas/trans/scale) from multi-frame 2D keypoints.

    Key change vs your previous version:
    - DO NOT average keypoints across frames.
      Instead, optimize by sampling frames and minimizing re-projection error across them.
      This avoids "arm disappears" due to temporal averaging / occlusion.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smplx is None:
        raise RuntimeError(
            "smplx is not installed. Install it first (pip install smplx), "
            "and provide stage1.smpl_model_path in the config."
        )

    model_path = Path(smpl_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL model path not found: {model_path}")

    frames_all = _load_keypoints_frames(keypoints_json)
    n_total = len(frames_all)
    use_n = max(1, min(int(fit_frames), n_total))

    # Uniformly sample frames to cover the full clip
    use_idx = _uniform_indices(n_total, use_n)
    frames = [frames_all[i] for i in use_idx]

    mp_indices = list(MP_TO_SMPL.keys())
    smpl_indices = list(MP_TO_SMPL.values())
    M = len(mp_indices)

    # Pre-build per-joint base weights
    base_w = np.array([MP_WEIGHT.get(mi, 1.0) for mi in mp_indices], dtype=np.float32).reshape(M, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smplx.create(
        model_path=str(model_path),
        model_type=smpl_model_type,
        gender=smpl_gender,
        num_betas=10,   # matches your basicmodel 10-PC
        batch_size=1,
    ).to(device)

    # Camera intrinsics
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)

    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3

    # Trainable params
    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, body_pose_dim), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    transl = torch.tensor([[0.0, 0.0, 2.5]], device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)

    optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl, scale], lr=float(fit_lr))

    iters = int(fit_iters)
    if iters <= 0:
        raise ValueError(f"fit_iters must be > 0, got {fit_iters}")

    # Two-stage schedule
    refine_start = int(0.60 * iters)
    refine_lr = float(fit_lr) * 0.2  # 5x smaller later

    # Small batch of frames per iteration (helps stability)
    # Keep it small to avoid slowdowns
    batch_frames = 4 if use_n >= 8 else 2

    rng = np.random.default_rng(42)

    pbar = trange(iters, desc="Stage1 SMPL fit (multi-frame)", dynamic_ncols=True)
    for it in pbar:
        if it == refine_start:
            for g in optimizer.param_groups:
                g["lr"] = refine_lr

        optimizer.zero_grad(set_to_none=True)

        # sample a mini-batch of frames
        idxs = rng.integers(0, use_n, size=batch_frames)
        loss_kpt_total = 0.0

        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        joints = out.joints[:, smpl_indices_t, :]  # (1,M,3)

        # apply scale
        X = joints * scale.view(1, 1, 1)

        z = X[..., 2].clamp(min=1e-4)
        x = X[..., 0] / z
        y = X[..., 1] / z
        proj = torch.stack(
            [K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]],
            dim=-1
        ).squeeze(0)  # (M,2)

        # accumulate over sampled frames
        for fi in idxs:
            fr = frames[int(fi)]  # (33,3)
            # target kpts & conf for selected joints
            tgt = fr[mp_indices, :2]  # (M,2)
            conf = fr[mp_indices, 2:3]  # (M,1)

            # conf-weighted + base reliability weights
            w = torch.tensor(conf * base_w, dtype=torch.float32, device=device)  # (M,1)

            # ignore super-low conf points
            w = torch.where(w > 1e-3, w, torch.zeros_like(w))

            tgt_t = torch.tensor(tgt, dtype=torch.float32, device=device)

            per = F.smooth_l1_loss(proj, tgt_t, reduction="none")  # (M,2)
            loss_kpt_total = loss_kpt_total + (per * w).sum()

        loss_kpt = loss_kpt_total / float(batch_frames)

        # regularization
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

        # safety clamps
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
                b=batch_frames,
            )

    # Export mesh + params
    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        verts = (out.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()

    pose = np.concatenate(
        [
            global_orient.detach().cpu().numpy().reshape(-1),
            body_pose.detach().cpu().numpy().reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    trans_np = transl.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)

    np.savez(
        out_dir / "smpl_params.npz",
        pose=pose,
        betas=betas_np,
        trans=trans_np,
        scale=scale_np,
    )

    with (out_dir / "smpl_mesh.obj").open("w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    return out_dir
