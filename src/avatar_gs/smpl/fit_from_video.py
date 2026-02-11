from __future__ import annotations

import json
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _typ in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("unicode", str),
    ("str", str),
):
    if _name not in np.__dict__:
        setattr(np, _name, _typ)

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None


# MediaPipe pose row layout:
# - 0..32: pose
# - 33..53: left hand (optional)
# - 54..74: right hand (optional)
MP_POSE_KPTS = 33
MP_HAND_KPTS = 21
MP_LEFT_HAND_OFFSET = MP_POSE_KPTS
MP_RIGHT_HAND_OFFSET = MP_POSE_KPTS + MP_HAND_KPTS

# Body anchors shared by SMPL/SMPL-X
MP_TO_BODY = {
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
MP_WEIGHT_BODY = {
    11: 1.25, 12: 1.25,
    23: 1.25, 24: 1.25,
    13: 1.05, 14: 1.05,
    15: 0.80, 16: 0.80,
    25: 1.00, 26: 1.00,
    27: 0.82, 28: 0.82,
}

# Extra face anchors available in SMPL-X joint set.
MP_TO_SMPLX_FACE = {
    0: 55,   # nose
    2: 57,   # left eye
    5: 56,   # right eye
    7: 59,   # left ear
    8: 58,   # right ear
    9: 113,  # left mouth corner
    10: 107, # right mouth corner
}
MP_WEIGHT_FACE = {
    0: 0.45,
    2: 0.32, 5: 0.32,
    7: 0.28, 8: 0.28,
    9: 0.25, 10: 0.25,
}

# Hand anchors (MediaPipe Hands -> SMPL-X joints).
_LH = MP_LEFT_HAND_OFFSET
_RH = MP_RIGHT_HAND_OFFSET
MP_TO_SMPLX_HAND = {
    # Left thumb/index/middle/ring/pinky
    _LH + 1: 37, _LH + 2: 38, _LH + 3: 39, _LH + 4: 66,
    _LH + 5: 25, _LH + 6: 26, _LH + 7: 27, _LH + 8: 67,
    _LH + 9: 28, _LH + 10: 29, _LH + 11: 30, _LH + 12: 68,
    _LH + 13: 34, _LH + 14: 35, _LH + 15: 36, _LH + 16: 69,
    _LH + 17: 31, _LH + 18: 32, _LH + 19: 33, _LH + 20: 70,
    # Right thumb/index/middle/ring/pinky
    _RH + 1: 52, _RH + 2: 53, _RH + 3: 54, _RH + 4: 71,
    _RH + 5: 40, _RH + 6: 41, _RH + 7: 42, _RH + 8: 72,
    _RH + 9: 43, _RH + 10: 44, _RH + 11: 45, _RH + 12: 73,
    _RH + 13: 49, _RH + 14: 50, _RH + 15: 51, _RH + 16: 74,
    _RH + 17: 46, _RH + 18: 47, _RH + 19: 48, _RH + 20: 75,
}
MP_WEIGHT_HAND = {k: 0.18 for k in MP_TO_SMPLX_HAND.keys()}
for _tip in (_LH + 4, _LH + 8, _LH + 12, _LH + 16, _LH + 20, _RH + 4, _RH + 8, _RH + 12, _RH + 16, _RH + 20):
    MP_WEIGHT_HAND[_tip] = 0.22


def _build_mp_smpl_correspondence(smpl_model_type: str) -> tuple[list[int], list[int], np.ndarray]:
    """
    Build MediaPipe->SMPL/SMPL-X correspondence and per-keypoint weights.
    - For SMPL: body + nose(head proxy)
    - For SMPL-X: body + face anchors + optional hand anchors
    """
    model_t = str(smpl_model_type).strip().lower()
    mapping: dict[int, int] = {}
    weights: dict[int, float] = {}

    mapping.update(MP_TO_BODY)
    weights.update(MP_WEIGHT_BODY)

    if model_t == "smplx":
        mapping.update(MP_TO_SMPLX_FACE)
        weights.update(MP_WEIGHT_FACE)
        mapping.update(MP_TO_SMPLX_HAND)
        weights.update(MP_WEIGHT_HAND)
    else:
        # SMPL has no facial landmark joints; keep a weak head anchor.
        mapping[0] = 15
        weights[0] = 0.22

    mp_indices = sorted(mapping.keys())
    smpl_indices = [int(mapping[mi]) for mi in mp_indices]
    base_w = np.asarray([float(weights.get(mi, 1.0)) for mi in mp_indices], dtype=np.float32).reshape(len(mp_indices), 1)
    return mp_indices, smpl_indices, base_w

# Stable limb-chain constraints in image space (MediaPipe ids).
LIMB_VECTOR_MP_EDGES: tuple[tuple[int, int], ...] = (
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (11, 23), (12, 24),  # torso vertical
    (11, 12), (23, 24),  # shoulder/hip width
)

# Pair distances used for weak 3D structure matching against MediaPipe world landmarks.
WORLD_PAIR_MP_EDGES: tuple[tuple[int, int], ...] = (
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (27, 28),
    (11, 23), (12, 24),
    (13, 14), (25, 26),
)

# Stable anchors for coarse global fit (LHM-style stage-1 warmup).
COARSE_FIT_MP_ANCHORS: tuple[int, ...] = (0, 11, 12, 23, 24)
TORSO_MP_IDS: tuple[int, ...] = (0, 11, 12, 13, 14, 23, 24)
HAND_FACE_MP_IDS: tuple[int, ...] = tuple(
    sorted(set(int(k) for k in MP_TO_SMPLX_FACE.keys()) | set(int(k) for k in MP_TO_SMPLX_HAND.keys()))
)


def _load_keypoints_frames(keypoints_json: str | Path) -> tuple[list[str], list[np.ndarray]]:
    with open(keypoints_json, "r", encoding="utf-8") as f:
        kpts_all = json.load(f)
    if len(kpts_all) == 0:
        raise RuntimeError("No keypoints found in keypoints.json")

    frame_names = sorted(kpts_all.keys())
    frames: list[np.ndarray] = []
    for name in frame_names:
        a = np.asarray(kpts_all[name], dtype=np.float32)
        if a.ndim != 2 or a.shape[1] < 3:
            raise RuntimeError(f"Bad keypoints format for {name}, expected (33,>=3), got {a.shape}")
        frames.append(a)
    return frame_names, frames


def _smooth_keypoint_frames(
    frames: list[np.ndarray],
    enable: bool = False,
    method: str = "one_euro",
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.3,
    gaussian_kernel_size: int = 7,
    gaussian_sigma: float = 1.5,
) -> list[np.ndarray]:
    """
    Lightweight temporal smoothing for keypoint coordinates.
    Only x/y are smoothed; confidence and other channels stay unchanged.
    """
    if (not bool(enable)) or len(frames) <= 1:
        return frames
    arrs = [np.asarray(fr, dtype=np.float32) for fr in frames]
    if len(arrs) == 0:
        return frames
    K = int(arrs[0].shape[0])
    D = int(arrs[0].shape[1])
    if K <= 0 or D < 2:
        return frames
    if any((a.shape[0] != K or a.shape[1] != D) for a in arrs):
        return frames

    x = np.stack([a[:, :2] for a in arrs], axis=0).astype(np.float32)  # (T,K,2)
    conf = np.stack([a[:, 2] if a.shape[1] > 2 else np.ones((K,), dtype=np.float32) for a in arrs], axis=0)
    valid = conf > 1e-4
    T, _, _ = x.shape

    m = str(method).strip().lower()
    if m == "gaussian":
        ks = max(3, int(gaussian_kernel_size))
        if ks % 2 == 0:
            ks += 1
        sig = max(float(gaussian_sigma), 1e-3)
        grid = np.arange(ks, dtype=np.float32) - float(ks // 2)
        ker = np.exp(-(grid * grid) / (2.0 * sig * sig))
        ker = ker / max(float(ker.sum()), 1e-8)
        pad = ks // 2
        for j in range(K):
            vj = valid[:, j].astype(np.float32)
            if float(vj.sum()) < 2.0:
                continue
            v0 = np.pad(vj, (pad, pad), mode="edge")
            for d in range(2):
                xd = np.pad(x[:, j, d], (pad, pad), mode="edge")
                num = np.convolve(xd * v0, ker, mode="valid")
                den = np.convolve(v0, ker, mode="valid")
                den = np.clip(den, 1e-6, None)
                x[:, j, d] = num / den
    else:
        # OneEuro-like per-joint filter.
        fps_use = max(float(fps), 1.0)
        min_c = max(float(min_cutoff), 1e-3)
        beta_use = max(float(beta), 0.0)

        def _alpha(cutoff: float) -> float:
            r = 2.0 * np.pi * float(cutoff) / fps_use
            return float(r / (1.0 + r))

        for j in range(K):
            prev = np.zeros((2,), dtype=np.float32)
            dprev = np.zeros((2,), dtype=np.float32)
            has_prev = False
            for t in range(T):
                if not bool(valid[t, j]):
                    continue
                if not has_prev:
                    prev = x[t, j].copy()
                    dprev.fill(0.0)
                    has_prev = True
                    continue
                dx = (x[t, j] - prev) * fps_use
                a_d = _alpha(1.0)
                d_hat = dprev + a_d * (dx - dprev)
                cutoff = min_c + beta_use * float(np.linalg.norm(d_hat))
                a = _alpha(cutoff)
                x_hat = prev + a * (x[t, j] - prev)
                x[t, j] = x_hat
                prev = x_hat
                dprev = d_hat

    out: list[np.ndarray] = []
    for i, fr in enumerate(arrs):
        f = fr.copy()
        f[:, :2] = x[i]
        out.append(f)
    return out


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k, dtype=int)


def _select_indices(n_total: int, fit_frames: int) -> np.ndarray:
    if fit_frames <= 0:
        return np.arange(n_total, dtype=int)
    return _uniform_indices(n_total, min(int(fit_frames), n_total))


def _estimate_translation_ls_single(
    joints_3d: np.ndarray,       # (M,3), canonical/model space
    joints_2d: np.ndarray,       # (M,2), pixel
    joints_w: np.ndarray,        # (M,), confidence weights
    K: np.ndarray,               # (3,3)
    min_weight: float = 1e-4,
    default_tz: float = 2.5,
) -> np.ndarray:
    """
    Weighted least-squares solve for camera translation t=(tx,ty,tz):
      u = fx*(X+tx)/(Z+tz) + cx
      v = fy*(Y+ty)/(Z+tz) + cy
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    w = np.asarray(joints_w, dtype=np.float64).reshape(-1)
    keep = np.isfinite(w) & (w > float(min_weight))
    keep = keep & np.isfinite(joints_3d).all(axis=1) & np.isfinite(joints_2d).all(axis=1)
    if int(keep.sum()) < 4:
        return np.asarray([0.0, 0.0, float(default_tz)], dtype=np.float32)

    S = joints_3d[keep].astype(np.float64)
    J = joints_2d[keep].astype(np.float64)
    W = np.sqrt(np.clip(w[keep], 1e-8, None))

    M = S.shape[0]
    A = np.zeros((2 * M, 3), dtype=np.float64)
    b = np.zeros((2 * M,), dtype=np.float64)
    for i in range(M):
        X, Y, Z = float(S[i, 0]), float(S[i, 1]), float(S[i, 2])
        u, v = float(J[i, 0]), float(J[i, 1])
        du = (u - cx)
        dv = (v - cy)
        wu = float(W[i])
        # fx*tx - du*tz = du*Z - fx*X
        A[2 * i, 0] = wu * fx
        A[2 * i, 1] = 0.0
        A[2 * i, 2] = -wu * du
        b[2 * i] = wu * (du * Z - fx * X)
        # fy*ty - dv*tz = dv*Z - fy*Y
        A[2 * i + 1, 0] = 0.0
        A[2 * i + 1, 1] = wu * fy
        A[2 * i + 1, 2] = -wu * dv
        b[2 * i + 1] = wu * (dv * Z - fy * Y)

    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        tx, ty, tz = float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return np.asarray([0.0, 0.0, float(default_tz)], dtype=np.float32)

    if (not np.isfinite(tz)) or tz < 0.2 or tz > 12.0:
        tz = float(default_tz)
    if not np.isfinite(tx):
        tx = 0.0
    if not np.isfinite(ty):
        ty = 0.0
    return np.asarray([tx, ty, tz], dtype=np.float32)


def _estimate_translation_ls(
    joints_3d_ref: np.ndarray,     # (M,3)
    targets_xy: torch.Tensor,      # (F,M,2)
    targets_w: torch.Tensor,       # (F,M,1)
    K: torch.Tensor,               # (3,3)
    default_tz: float = 2.5,
) -> np.ndarray:
    F_ = int(targets_xy.shape[0])
    K_np = K.detach().cpu().numpy().astype(np.float32)
    t_all = np.zeros((F_, 3), dtype=np.float32)
    for fi in range(F_):
        t_all[fi] = _estimate_translation_ls_single(
            joints_3d=joints_3d_ref,
            joints_2d=targets_xy[fi].detach().cpu().numpy().astype(np.float32),
            joints_w=targets_w[fi, :, 0].detach().cpu().numpy().astype(np.float32),
            K=K_np,
            min_weight=1e-4,
            default_tz=float(default_tz),
        )
    return t_all


def _loss_phase_multipliers(
    progress_01: float,
    schedule: str,
) -> dict[str, float]:
    """
    Balanced sequence schedule:
    - A [0,0.15): torso/camera stabilization
    - B [0.15,0.75): balanced full-body
    - C [0.75,1.0]: limbs/hands refinement with temporal consistency
    """
    sched = str(schedule or "none").strip().lower()
    if sched not in ("balanced_sequence_v1", "balanced"):
        return {
            "kpt_global": 1.0,
            "kpt_torso": 1.0,
            "kpt_hand_face": 1.0,
            "depth": 1.0,
            "world_depth": 1.0,
            "world_pair": 1.0,
            "silhouette": 1.0,
            "silhouette_bbox": 1.0,
            "limb": 1.0,
            "temporal": 1.0,
            "hand_face_temporal": 1.0,
            "hand_face_reg": 1.0,
        }
    p = float(np.clip(progress_01, 0.0, 1.0))
    if p < 0.15:
        return {
            "kpt_global": 1.0,
            "kpt_torso": 1.35,
            "kpt_hand_face": 0.55,
            "depth": 1.15,
            "world_depth": 1.15,
            "world_pair": 1.1,
            "silhouette": 0.8,
            "silhouette_bbox": 0.9,
            "limb": 0.45,
            "temporal": 1.2,
            "hand_face_temporal": 0.8,
            "hand_face_reg": 1.15,
        }
    if p < 0.75:
        return {
            "kpt_global": 1.0,
            "kpt_torso": 1.0,
            "kpt_hand_face": 1.0,
            "depth": 1.0,
            "world_depth": 1.0,
            "world_pair": 1.0,
            "silhouette": 1.0,
            "silhouette_bbox": 1.0,
            "limb": 1.0,
            "temporal": 1.0,
            "hand_face_temporal": 1.0,
            "hand_face_reg": 1.0,
        }
    return {
        "kpt_global": 1.0,
        "kpt_torso": 1.0,
        "kpt_hand_face": 1.35,
        "depth": 0.9,
        "world_depth": 0.9,
        "world_pair": 0.95,
        "silhouette": 1.1,
        "silhouette_bbox": 1.1,
        "limb": 1.35,
        "temporal": 1.08,
        "hand_face_temporal": 1.25,
        "hand_face_reg": 0.9,
    }


def _prepare_targets(
    frames: list[np.ndarray],
    mp_indices: list[int],
    base_w: np.ndarray,
    device: torch.device,
    min_conf: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tgts = []
    ws = []
    M = len(mp_indices)
    for fr in frames:
        tgt = np.zeros((M, 2), dtype=np.float32)
        conf = np.zeros((M, 1), dtype=np.float32)
        if fr.ndim == 2 and fr.shape[1] >= 3:
            n = int(fr.shape[0])
            for i, mi in enumerate(mp_indices):
                if 0 <= int(mi) < n:
                    tgt[i] = fr[int(mi), :2]
                    conf[i, 0] = float(fr[int(mi), 2])
        w = conf * base_w
        if float(min_conf) > 0.0:
            w = np.where(conf >= float(min_conf), w, 0.0)
        w = np.where(w > 1e-3, w, 0.0)
        tgts.append(tgt)
        ws.append(w)
    tgt_t = torch.tensor(np.stack(tgts, axis=0), dtype=torch.float32, device=device)  # (F,M,2)
    w_t = torch.tensor(np.stack(ws, axis=0), dtype=torch.float32, device=device)       # (F,M,1)
    return tgt_t, w_t


def _resolve_mp_subset_indices(
    mp_indices: list[int],
    mp_subset: list[int] | tuple[int, ...],
) -> list[int]:
    id_map = {int(mp): i for i, mp in enumerate(mp_indices)}
    out = []
    for mp in mp_subset:
        if int(mp) in id_map:
            out.append(id_map[int(mp)])
    return out


def _compute_keypoint_frame_scale(
    targets_xy: torch.Tensor,   # (F,M,2)
    targets_w: torch.Tensor,    # (F,M,1)
    min_px: float = 96.0,
) -> torch.Tensor:
    """
    Per-frame person scale from valid keypoint 2D extent (bbox diagonal).

    Output:
    - (F,1,1), clamped by min_px.
    """
    if targets_xy.ndim != 3 or targets_w.ndim != 3:
        raise ValueError("targets_xy/targets_w must be (F,M,*)")
    if targets_xy.shape[:2] != targets_w.shape[:2]:
        raise ValueError("targets_xy/targets_w shape mismatch")

    F_ = int(targets_xy.shape[0])
    out = torch.full((F_,), float(min_px), dtype=targets_xy.dtype, device=targets_xy.device)
    valid = targets_w.squeeze(-1) > 1e-4
    for fi in range(F_):
        m = valid[fi]
        if int(m.sum().item()) < 2:
            continue
        pts = targets_xy[fi][m]
        wh = pts.max(dim=0).values - pts.min(dim=0).values
        diag = torch.linalg.vector_norm(wh, dim=0)
        out[fi] = torch.clamp(diag, min=float(min_px))
    return out.view(F_, 1, 1)


def _gmof_error(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure robust penalty used by SMPLify variants.
    """
    sigma2 = max(float(sigma), 1e-3) ** 2
    x2 = x * x
    return (sigma2 / (sigma2 + x2)) * x2


def _keypoint_reprojection_loss(
    proj_xy: torch.Tensor,           # (B,M,2)
    target_xy: torch.Tensor,         # (B,M,2)
    target_w: torch.Tensor,          # (B,M,1)
    loss_type: str = "smooth_l1",
    gmof_sigma_px: float = 100.0,
    frame_scale: torch.Tensor | None = None,     # (B,1,1)
    conf_thresh: float = 0.0,
    subset_idx: list[int] | None = None,
) -> torch.Tensor:
    """
    Weighted 2D reprojection loss with optional robustifier and scale normalization.
    """
    if proj_xy.ndim != 3 or target_xy.ndim != 3 or target_w.ndim != 3:
        raise ValueError("proj_xy/target_xy/target_w must be (B,M,*)")
    if proj_xy.shape[:2] != target_xy.shape[:2] or proj_xy.shape[:2] != target_w.shape[:2]:
        raise ValueError("proj_xy/target_xy/target_w shape mismatch")
    if frame_scale is not None and (frame_scale.ndim != 3 or frame_scale.shape[-2:] != (1, 1)):
        raise ValueError("frame_scale must be (B,1,1)")

    if subset_idx is not None and len(subset_idx) > 0:
        idx_t = torch.tensor(subset_idx, dtype=torch.long, device=proj_xy.device)
        proj_xy = proj_xy.index_select(1, idx_t)
        target_xy = target_xy.index_select(1, idx_t)
        target_w = target_w.index_select(1, idx_t)

    resid = proj_xy - target_xy
    lt = str(loss_type).lower()
    if lt in ("gmof", "geman_mcclure", "gmc"):
        per = _gmof_error(resid, sigma=float(gmof_sigma_px))
    elif lt in ("l2", "mse"):
        per = resid * resid
    else:
        per = F.smooth_l1_loss(proj_xy, target_xy, reduction="none")

    if frame_scale is not None:
        per = per / frame_scale.clamp(min=1.0)

    w = target_w
    if float(conf_thresh) > 0.0:
        w = torch.where(w >= float(conf_thresh), w, torch.zeros_like(w))

    return (per * w).sum() / (w.sum() + 1e-6)


def _loss_pose_angle_constrain(global_orient: torch.Tensor, body_pose: torch.Tensor) -> torch.Tensor:
    """
    CAR-style soft angle prior to suppress implausible elbow/knee bends.
    """
    if body_pose.ndim != 2 or global_orient.ndim != 2:
        raise ValueError("global_orient/body_pose must be (B,D)")
    if body_pose.shape[0] != global_orient.shape[0]:
        raise ValueError("global_orient/body_pose batch mismatch")

    if int(body_pose.shape[1]) < 69:
        return torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)

    theta = torch.cat([global_orient, body_pose[:, :69]], dim=1)
    if int(theta.shape[1]) <= 59:
        return torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)

    val = (
        torch.exp(theta[:, 55])
        + torch.exp(-theta[:, 58])
        + torch.exp(-theta[:, 12])
        + torch.exp(-theta[:, 15])
        + torch.abs(theta[:, 56])
        + torch.abs(theta[:, 59])
    )
    return torch.mean(val * val)


def _limb_vector_reprojection_loss(
    proj_xy: torch.Tensor,      # (B,M,2)
    target_xy: torch.Tensor,    # (B,M,2)
    target_w: torch.Tensor,     # (B,M,1)
    mp_indices: list[int],
    min_target_len_px: float = 6.0,
    length_ratio_weight: float = 0.2,
) -> torch.Tensor:
    """
    Reprojection loss on limb vectors (direction + relative length).

    This reduces single-image depth ambiguity where point-only reprojection
    can still match 2D while producing implausible 3D articulation.
    """
    if proj_xy.ndim != 3 or target_xy.ndim != 3 or target_w.ndim != 3:
        raise ValueError("proj_xy/target_xy/target_w must be (B,M,*)")
    if proj_xy.shape[:2] != target_xy.shape[:2] or proj_xy.shape[:2] != target_w.shape[:2]:
        raise ValueError("proj_xy/target_xy/target_w shape mismatch")

    id_map = {int(mp): i for i, mp in enumerate(mp_indices)}
    conf = target_w.squeeze(-1).clamp(min=0.0)  # (B,M)
    terms: list[torch.Tensor] = []
    ws: list[torch.Tensor] = []
    for a, b in LIMB_VECTOR_MP_EDGES:
        if a not in id_map or b not in id_map:
            continue
        ia = id_map[a]
        ib = id_map[b]
        pv = proj_xy[:, ib, :] - proj_xy[:, ia, :]
        tv = target_xy[:, ib, :] - target_xy[:, ia, :]
        plen = torch.linalg.vector_norm(pv, dim=-1)
        tlen = torch.linalg.vector_norm(tv, dim=-1)
        pair_w = torch.sqrt(conf[:, ia] * conf[:, ib] + 1e-8)
        valid = (pair_w > 1e-4) & (tlen > float(min_target_len_px))
        if not bool(valid.any()):
            continue

        pdir = pv / (plen.unsqueeze(-1) + 1e-6)
        tdir = tv / (tlen.unsqueeze(-1) + 1e-6)
        loss_dir = F.smooth_l1_loss(pdir, tdir, reduction="none").mean(dim=-1)
        loss_len = F.smooth_l1_loss(
            plen / (tlen + 1e-6),
            torch.ones_like(plen),
            reduction="none",
        )
        loss_edge = loss_dir + float(length_ratio_weight) * loss_len
        terms.append(loss_edge[valid])
        ws.append(pair_w[valid])

    if len(terms) == 0:
        return torch.zeros((), dtype=proj_xy.dtype, device=proj_xy.device)
    t_cat = torch.cat(terms, dim=0)
    w_cat = torch.cat(ws, dim=0)
    return (t_cat * w_cat).sum() / (w_cat.sum() + 1e-6)


def _prepare_depth_targets(
    frames: list[np.ndarray],
    mp_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Build MediaPipe relative-depth targets from landmark z (if available).

    Returns:
    - depth_t: (F,M), zeros when a frame has no depth column
    - valid_t: (F,M) in {0,1}
    - has_depth: whether any frame provides depth
    """
    has_depth = any((fr.ndim == 2 and fr.shape[1] >= 4) for fr in frames)
    F_ = len(frames)
    M = len(mp_indices)
    if not has_depth:
        return (
            torch.zeros((F_, M), dtype=torch.float32, device=device),
            torch.zeros((F_, M), dtype=torch.float32, device=device),
            False,
        )

    zs = []
    vs = []
    for fr in frames:
        z = np.zeros((M,), dtype=np.float32)
        v = np.zeros((M,), dtype=np.float32)
        if fr.ndim == 2 and fr.shape[1] >= 4:
            n = int(fr.shape[0])
            for i, mi in enumerate(mp_indices):
                if 0 <= int(mi) < n:
                    z[i] = float(fr[int(mi), 3])
                    v[i] = 1.0 if float(fr[int(mi), 2]) > 1e-3 else 0.0
        zs.append(z)
        vs.append(v)
    z_t = torch.tensor(np.stack(zs, axis=0), dtype=torch.float32, device=device)  # (F,M)
    v_t = torch.tensor(np.stack(vs, axis=0), dtype=torch.float32, device=device)   # (F,M)
    return z_t, v_t, True


def _prepare_world_targets(
    frames: list[np.ndarray],
    mp_indices: list[int],
    device: torch.device,
    min_visibility: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Build MediaPipe world landmark targets from columns [4:7], if present.

    Returns:
    - world_xyz: (F,M,3)
    - world_valid: (F,M) in {0,1}
    - has_world: whether any frame provides valid world landmarks
    """
    has_world = any((fr.ndim == 2 and fr.shape[1] >= 7) for fr in frames)
    F_ = len(frames)
    M = len(mp_indices)
    if not has_world:
        return (
            torch.zeros((F_, M, 3), dtype=torch.float32, device=device),
            torch.zeros((F_, M), dtype=torch.float32, device=device),
            False,
        )

    worlds = []
    vals = []
    for fr in frames:
        xyz = np.zeros((M, 3), dtype=np.float32)
        v = np.zeros((M,), dtype=np.float32)
        if fr.ndim == 2 and fr.shape[1] >= 7:
            n = int(fr.shape[0])
            for i, mi in enumerate(mp_indices):
                if 0 <= int(mi) < n:
                    xyz_i = np.asarray(fr[int(mi), 4:7], dtype=np.float32)
                    if xyz_i.shape[0] == 3:
                        xyz[i] = xyz_i
                        vis = float(fr[int(mi), 2])
                        finite = float(np.isfinite(xyz_i).all())
                        nonzero = 1.0 if float(np.linalg.norm(xyz_i)) > 1e-8 else 0.0
                        v[i] = (1.0 if vis >= float(min_visibility) else 0.0) * finite * nonzero
        worlds.append(xyz)
        vals.append(v)
    w_t = torch.tensor(np.stack(worlds, axis=0), dtype=torch.float32, device=device)  # (F,M,3)
    v_t = torch.tensor(np.stack(vals, axis=0), dtype=torch.float32, device=device)     # (F,M)
    has_any = bool((v_t > 0.5).any().item())
    return w_t, v_t, has_any


def _monocular_depth_rank_loss(
    smpl_joints_sel: torch.Tensor,
    mp_depth: torch.Tensor,
    mp_weight: torch.Tensor,
    pair_margin: float = 0.0,
    auto_sign: bool = True,
    temp: float = 0.05,
) -> torch.Tensor:
    """
    Monocular depth-order prior from MediaPipe landmark z.

    smpl_joints_sel: (B,M,3) selected SMPL joints in camera coordinates.
    mp_depth:        (B,M) MediaPipe depth (relative).
    mp_weight:       (B,M) confidence weights in [0,+inf).
    """
    if smpl_joints_sel.ndim != 3 or smpl_joints_sel.shape[-1] != 3:
        raise ValueError(f"smpl_joints_sel must be (B,M,3), got {smpl_joints_sel.shape}")
    if mp_depth.ndim != 2 or mp_weight.ndim != 2:
        raise ValueError("mp_depth/mp_weight must be (B,M)")

    B, M, _ = smpl_joints_sel.shape
    if B == 0 or M <= 1:
        return torch.zeros((), dtype=smpl_joints_sel.dtype, device=smpl_joints_sel.device)

    zs = smpl_joints_sel[..., 2]
    zs = zs - zs.mean(dim=1, keepdim=True)
    zm = mp_depth - mp_depth.mean(dim=1, keepdim=True)

    dzs = zs.unsqueeze(2) - zs.unsqueeze(1)  # (B,M,M)
    dzm = zm.unsqueeze(2) - zm.unsqueeze(1)  # (B,M,M)
    ww = mp_weight.unsqueeze(2) * mp_weight.unsqueeze(1)

    valid = ww > 1e-6
    if float(pair_margin) > 0.0:
        valid = valid & (dzm.abs() >= float(pair_margin))
    eye = torch.eye(M, device=smpl_joints_sel.device, dtype=torch.bool).unsqueeze(0)
    valid = valid & (~eye)

    def _loss_with_sign(sign_scale: float) -> torch.Tensor:
        target = torch.sign(sign_scale * dzm)
        keep = valid & (target != 0)
        if not bool(keep.any()):
            return torch.zeros((), dtype=smpl_joints_sel.dtype, device=smpl_joints_sel.device)
        logits = (target * dzs) / max(float(temp), 1e-4)
        rank_loss = F.softplus(-logits)
        w = ww[keep] * dzm.abs()[keep].clamp(min=1e-4)
        return (rank_loss[keep] * w).sum() / (w.sum() + 1e-6)

    lp = _loss_with_sign(1.0)
    if not auto_sign:
        return lp
    ln = _loss_with_sign(-1.0)
    return torch.minimum(lp, ln)


def _normalized_body_coords_from_selected_joints(
    joints: torch.Tensor,          # (B,M,3)
    valid_mask: torch.Tensor,      # (B,M) in {0,1}
    mp_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize 3D joints by hip center and torso scale.

    Returns:
    - normalized joints: (B,M,3)
    - sample_valid: (B,) in {0,1}
    """
    B = int(joints.shape[0])
    out = joints.clone()
    sample_valid = torch.ones((B,), dtype=torch.bool, device=joints.device)
    id_map = {int(mp): i for i, mp in enumerate(mp_indices)}
    req = [11, 12, 23, 24]
    if any(r not in id_map for r in req):
        return out, torch.zeros((B,), dtype=torch.bool, device=joints.device)
    i_ls, i_rs, i_lh, i_rh = id_map[11], id_map[12], id_map[23], id_map[24]

    req_ok = (
        (valid_mask[:, i_ls] > 0.5)
        & (valid_mask[:, i_rs] > 0.5)
        & (valid_mask[:, i_lh] > 0.5)
        & (valid_mask[:, i_rh] > 0.5)
    )
    sample_valid = sample_valid & req_ok

    hip_center = 0.5 * (joints[:, i_lh, :] + joints[:, i_rh, :])  # (B,3)
    out = out - hip_center.unsqueeze(1)
    torso = 0.5 * (
        torch.linalg.vector_norm(joints[:, i_ls, :] - joints[:, i_lh, :], dim=-1)
        + torch.linalg.vector_norm(joints[:, i_rs, :] - joints[:, i_rh, :], dim=-1)
    )
    sample_valid = sample_valid & (torso > 1e-5)
    scale = torso.clamp(min=1e-5).view(B, 1, 1)
    out = out / scale
    return out, sample_valid


def _world_pair_distance_loss(
    smpl_joints_sel: torch.Tensor,  # (B,M,3)
    mp_world_xyz: torch.Tensor,     # (B,M,3)
    mp_world_valid: torch.Tensor,   # (B,M)
    mp_weight: torch.Tensor,        # (B,M)
    mp_indices: list[int],
) -> torch.Tensor:
    """
    Weak 3D structure prior: match normalized pair distances to MediaPipe world landmarks.
    """
    if smpl_joints_sel.ndim != 3 or mp_world_xyz.ndim != 3:
        raise ValueError("smpl_joints_sel/mp_world_xyz must be (B,M,3)")
    if mp_world_valid.ndim != 2 or mp_weight.ndim != 2:
        raise ValueError("mp_world_valid/mp_weight must be (B,M)")
    if smpl_joints_sel.shape != mp_world_xyz.shape:
        raise ValueError("smpl_joints_sel and mp_world_xyz shape mismatch")

    B, M, _ = smpl_joints_sel.shape
    if B == 0 or M <= 1:
        return torch.zeros((), dtype=smpl_joints_sel.dtype, device=smpl_joints_sel.device)

    ones = torch.ones((B, M), dtype=torch.float32, device=smpl_joints_sel.device)
    smpl_n, smpl_ok = _normalized_body_coords_from_selected_joints(
        joints=smpl_joints_sel, valid_mask=ones, mp_indices=mp_indices
    )
    mp_n, mp_ok = _normalized_body_coords_from_selected_joints(
        joints=mp_world_xyz, valid_mask=mp_world_valid, mp_indices=mp_indices
    )
    sample_ok = smpl_ok & mp_ok
    if not bool(sample_ok.any()):
        return torch.zeros((), dtype=smpl_joints_sel.dtype, device=smpl_joints_sel.device)

    id_map = {int(mp): i for i, mp in enumerate(mp_indices)}
    terms = []
    weights = []
    for a, b in WORLD_PAIR_MP_EDGES:
        if a not in id_map or b not in id_map:
            continue
        ia = id_map[a]
        ib = id_map[b]
        d_s = torch.linalg.vector_norm(smpl_n[:, ia, :] - smpl_n[:, ib, :], dim=-1)
        d_m = torch.linalg.vector_norm(mp_n[:, ia, :] - mp_n[:, ib, :], dim=-1)
        pair_valid = (
            sample_ok
            & (mp_world_valid[:, ia] > 0.5)
            & (mp_world_valid[:, ib] > 0.5)
        )
        if not bool(pair_valid.any()):
            continue
        pair_w = torch.sqrt((mp_weight[:, ia] * mp_weight[:, ib]).clamp(min=0.0) + 1e-8)
        loss_edge = F.smooth_l1_loss(d_s, d_m, reduction="none")
        terms.append(loss_edge[pair_valid])
        weights.append(pair_w[pair_valid])

    if len(terms) == 0:
        return torch.zeros((), dtype=smpl_joints_sel.dtype, device=smpl_joints_sel.device)
    t_cat = torch.cat(terms, dim=0)
    w_cat = torch.cat(weights, dim=0)
    return (t_cat * w_cat).sum() / (w_cat.sum() + 1e-6)


def _load_silhouette_distance_maps(
    silhouette_npz: str | Path | None,
    frame_names: list[str],
    device: torch.device,
    dist_clip_px: float = 24.0,
) -> torch.Tensor | None:
    """
    Load person masks and convert them to normalized outside-distance maps.

    Output shape: (F,1,H,W), values in [0,1].
    """
    if silhouette_npz is None:
        return None
    path = Path(silhouette_npz)
    if not path.exists():
        return None
    seq = np.load(path, allow_pickle=True)
    if "frame_names" not in seq.files or "masks" not in seq.files:
        return None
    names = [str(x) for x in seq["frame_names"].tolist()]
    masks = np.asarray(seq["masks"], dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[0] != len(names):
        return None
    name_to_i = {n: i for i, n in enumerate(names)}
    clip_v = max(float(dist_clip_px), 1.0)
    dmaps = []
    for fn in frame_names:
        if fn not in name_to_i:
            # Missing mask: keep zero map (no silhouette supervision for this frame).
            h = int(masks.shape[1])
            w = int(masks.shape[2])
            dmaps.append(np.zeros((h, w), dtype=np.float32))
            continue
        m = (masks[name_to_i[fn]] > 0).astype(np.uint8)
        outside = (1 - m).astype(np.uint8)
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32)
        dist = np.minimum(dist, clip_v) / clip_v
        dmaps.append(dist)
    arr = np.stack(dmaps, axis=0).astype(np.float32)  # (F,H,W)
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)


def _load_silhouette_bboxes(
    silhouette_npz: str | Path | None,
    frame_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[int, int] | None]:
    """
    Load per-frame mask bounding boxes as weak shape constraints.

    Returns:
    - bboxes: (F,4) as [xmin, ymin, xmax, ymax]
    - valid:  (F,) in {0,1}
    - image_hw: (H,W)
    """
    if silhouette_npz is None:
        return None, None, None
    path = Path(silhouette_npz)
    if not path.exists():
        return None, None, None
    seq = np.load(path, allow_pickle=True)
    if "frame_names" not in seq.files or "masks" not in seq.files:
        return None, None, None
    names = [str(x) for x in seq["frame_names"].tolist()]
    masks = np.asarray(seq["masks"], dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[0] != len(names):
        return None, None, None

    h = int(masks.shape[1])
    w = int(masks.shape[2])
    name_to_i = {n: i for i, n in enumerate(names)}
    boxes = []
    vals = []
    for fn in frame_names:
        if fn not in name_to_i:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            vals.append(0.0)
            continue
        m = masks[name_to_i[fn]] > 0
        ys, xs = np.where(m)
        if ys.size == 0 or xs.size == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            vals.append(0.0)
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        vals.append(1.0)
    b_t = torch.tensor(np.asarray(boxes, dtype=np.float32), dtype=torch.float32, device=device)
    v_t = torch.tensor(np.asarray(vals, dtype=np.float32), dtype=torch.float32, device=device)
    return b_t, v_t, (h, w)


def _silhouette_bbox_loss(
    verts_cam: torch.Tensor,    # (B,V,3)
    K: torch.Tensor,            # (3,3)
    bbox_tgt: torch.Tensor,     # (B,4)
    bbox_valid: torch.Tensor,   # (B,)
    image_hw: tuple[int, int] | None,
) -> torch.Tensor:
    if verts_cam.ndim != 3 or verts_cam.shape[-1] != 3:
        raise ValueError(f"verts_cam must be (B,V,3), got {verts_cam.shape}")
    if bbox_tgt.ndim != 2 or bbox_tgt.shape[1] != 4:
        raise ValueError(f"bbox_tgt must be (B,4), got {bbox_tgt.shape}")
    if bbox_valid.ndim != 1 or bbox_valid.shape[0] != verts_cam.shape[0]:
        raise ValueError("bbox_valid must be (B,) and align with verts batch size")

    B = int(verts_cam.shape[0])
    if B == 0:
        return torch.zeros((), dtype=verts_cam.dtype, device=verts_cam.device)

    z = verts_cam[..., 2].clamp(min=1e-5)
    u = K[0, 0] * (verts_cam[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (verts_cam[..., 1] / z) + K[1, 2]
    valid = verts_cam[..., 2] > 1e-5

    if image_hw is not None:
        h, w = int(image_hw[0]), int(image_hw[1])
        valid = (
            valid
            & (u >= -0.5) & (u <= float(w) - 0.5)
            & (v >= -0.5) & (v <= float(h) - 0.5)
        )
        norm = torch.tensor([w, h, w, h], dtype=verts_cam.dtype, device=verts_cam.device).view(1, 4)
    else:
        norm = torch.ones((1, 4), dtype=verts_cam.dtype, device=verts_cam.device)

    has_valid = valid.any(dim=1)
    use = (bbox_valid > 0.5) & has_valid
    if not bool(use.any()):
        return torch.zeros((), dtype=verts_cam.dtype, device=verts_cam.device)

    big = torch.full_like(u, 1e6)
    nbig = torch.full_like(u, -1e6)
    u_min = torch.where(valid, u, big).amin(dim=1)
    u_max = torch.where(valid, u, nbig).amax(dim=1)
    v_min = torch.where(valid, v, big).amin(dim=1)
    v_max = torch.where(valid, v, nbig).amax(dim=1)
    bbox_pred = torch.stack([u_min, v_min, u_max, v_max], dim=1)

    loss_per = F.smooth_l1_loss(
        bbox_pred[use] / norm,
        bbox_tgt[use] / norm,
        reduction="none",
    ).mean(dim=1)
    w_use = bbox_valid[use]
    return (loss_per * w_use).sum() / (w_use.sum() + 1e-6)


def _compute_silhouette_bbox_prior_loss(
    verts_cam: torch.Tensor,          # (B,V,3)
    K: torch.Tensor,                  # (3,3)
    frame_idx: torch.Tensor,          # (B,)
    sil_bbox_all: torch.Tensor | None,        # (F,4)
    sil_bbox_valid_all: torch.Tensor | None,  # (F,)
    sil_bbox_hw: tuple[int, int] | None,
    use_bbox_prior: bool,
) -> torch.Tensor:
    if (
        (not use_bbox_prior)
        or sil_bbox_all is None
        or sil_bbox_valid_all is None
    ):
        return torch.zeros((), dtype=verts_cam.dtype, device=verts_cam.device)
    return _silhouette_bbox_loss(
        verts_cam=verts_cam,
        K=K,
        bbox_tgt=sil_bbox_all[frame_idx],
        bbox_valid=sil_bbox_valid_all[frame_idx],
        image_hw=sil_bbox_hw,
    )


def _compute_side_view_frame_weights(
    targets_xy: torch.Tensor,  # (F,M,2)
    targets_w: torch.Tensor,   # (F,M,1)
    mp_indices: list[int],
    side_view_boost: float = 1.5,
    side_ratio_thr: float = 0.55,
) -> torch.Tensor:
    """
    Estimate per-frame side-view confidence from shoulder/hip width ratios.

    Returns frame weights in [1, 1+side_view_boost].
    """
    F_ = int(targets_xy.shape[0])
    out = torch.ones((F_,), dtype=torch.float32, device=targets_xy.device)
    if side_view_boost <= 0.0:
        return out

    id_map = {int(mp): i for i, mp in enumerate(mp_indices)}
    req = [11, 12, 23, 24]
    if any(r not in id_map for r in req):
        return out
    i_ls, i_rs, i_lh, i_rh = id_map[11], id_map[12], id_map[23], id_map[24]

    xy = targets_xy
    w = targets_w.squeeze(-1)
    conf = torch.stack([w[:, i_ls], w[:, i_rs], w[:, i_lh], w[:, i_rh]], dim=1)  # (F,4)
    valid = (conf > 1e-4).all(dim=1)
    if not bool(valid.any()):
        return out

    ls, rs = xy[:, i_ls], xy[:, i_rs]
    lh, rh = xy[:, i_lh], xy[:, i_rh]
    shoulder_w = torch.linalg.vector_norm(ls - rs, dim=1)
    hip_w = torch.linalg.vector_norm(lh - rh, dim=1)
    torso_h = 0.5 * (
        torch.linalg.vector_norm(ls - lh, dim=1) + torch.linalg.vector_norm(rs - rh, dim=1)
    )
    ratio = 0.5 * (shoulder_w + hip_w) / (torso_h + 1e-6)
    side = ((float(side_ratio_thr) - ratio) / max(float(side_ratio_thr), 1e-6)).clamp(0.0, 1.0)
    boost = 1.0 + float(side_view_boost) * side
    out = torch.where(valid, boost, out)
    return out


def _silhouette_vertex_distance_per_frame(
    verts_cam: torch.Tensor,    # (B,V,3)
    K: torch.Tensor,            # (3,3)
    dist_maps: torch.Tensor,    # (B,1,H,W), normalized [0,1]
    sample_verts: int = 2048,
) -> torch.Tensor:
    """
    Per-frame outside-distance penalty by sampling distance maps at projected vertices.
    """
    if verts_cam.ndim != 3 or verts_cam.shape[-1] != 3:
        raise ValueError(f"verts_cam must be (B,V,3), got {verts_cam.shape}")
    if dist_maps.ndim != 4 or dist_maps.shape[1] != 1:
        raise ValueError(f"dist_maps must be (B,1,H,W), got {dist_maps.shape}")

    B, V, _ = verts_cam.shape
    H = int(dist_maps.shape[2])
    W = int(dist_maps.shape[3])
    if V <= 0:
        return torch.zeros((B,), dtype=torch.float32, device=verts_cam.device)

    if int(sample_verts) > 0 and V > int(sample_verts):
        idx = torch.randperm(V, device=verts_cam.device)[: int(sample_verts)]
        vc = verts_cam.index_select(1, idx)
    else:
        vc = verts_cam

    z = vc[..., 2].clamp(min=1e-5)
    u = K[0, 0] * (vc[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (vc[..., 1] / z) + K[1, 2]

    xn = 2.0 * (u / max(float(W - 1), 1.0)) - 1.0
    yn = 2.0 * (v / max(float(H - 1), 1.0)) - 1.0
    grid = torch.stack([xn, yn], dim=-1).unsqueeze(1)  # (B,1,S,2)
    sampled = F.grid_sample(
        dist_maps,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(1)  # (B,S)
    valid = (vc[..., 2] > 1e-5) & (xn >= -1.0) & (xn <= 1.0) & (yn >= -1.0) & (yn <= 1.0)
    w = valid.float()
    return (sampled * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)


def _compute_depth_prior_loss(
    joints_sel: torch.Tensor,       # (B,M,3)
    frame_idx: torch.Tensor,        # (B,)
    targets_depth: torch.Tensor,    # (F,M)
    depth_weight_map: torch.Tensor, # (F,M)
    use_depth_prior: bool,
    pair_margin: float,
    auto_sign: bool,
) -> torch.Tensor:
    if not use_depth_prior:
        return torch.zeros((), dtype=joints_sel.dtype, device=joints_sel.device)
    return _monocular_depth_rank_loss(
        smpl_joints_sel=joints_sel,
        mp_depth=targets_depth[frame_idx],
        mp_weight=depth_weight_map[frame_idx],
        pair_margin=float(pair_margin),
        auto_sign=bool(auto_sign),
    )


def _compute_world_depth_prior_loss(
    joints_sel: torch.Tensor,           # (B,M,3)
    frame_idx: torch.Tensor,            # (B,)
    targets_world: torch.Tensor,        # (F,M,3)
    world_weight_map: torch.Tensor,     # (F,M)
    use_world_depth_prior: bool,
    pair_margin: float,
    auto_sign: bool,
) -> torch.Tensor:
    if not use_world_depth_prior:
        return torch.zeros((), dtype=joints_sel.dtype, device=joints_sel.device)
    return _monocular_depth_rank_loss(
        smpl_joints_sel=joints_sel,
        mp_depth=targets_world[frame_idx, :, 2],
        mp_weight=world_weight_map[frame_idx],
        pair_margin=float(pair_margin),
        auto_sign=bool(auto_sign),
    )


def _compute_silhouette_prior_loss(
    verts_cam: torch.Tensor,         # (B,V,3)
    K: torch.Tensor,                 # (3,3)
    frame_idx: torch.Tensor,         # (B,)
    sil_dist_maps_all: torch.Tensor | None,  # (F,1,H,W)
    sil_frame_w_all: torch.Tensor | None,    # (F,)
    use_silhouette_prior: bool,
    sample_verts: int,
) -> torch.Tensor:
    if (
        (not use_silhouette_prior)
        or sil_dist_maps_all is None
        or sil_frame_w_all is None
    ):
        return torch.zeros((), dtype=verts_cam.dtype, device=verts_cam.device)
    sil_pf = _silhouette_vertex_distance_per_frame(
        verts_cam=verts_cam,
        K=K,
        dist_maps=sil_dist_maps_all[frame_idx],
        sample_verts=int(sample_verts),
    )
    sil_fw = sil_frame_w_all[frame_idx]
    return (sil_pf * sil_fw).sum() / (sil_fw.sum() + 1e-6)


def _scheduled_weight(base_weight: float, progress_01: float, final_ratio: float) -> float:
    """Linear schedule from base_weight to base_weight * final_ratio."""
    p = float(np.clip(progress_01, 0.0, 1.0))
    r = float(max(final_ratio, 0.0))
    return float(base_weight) * ((1.0 - p) + p * r)


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
    smpl_model_type: str = "smpl",
    smpl_gender: str = "neutral",
    theta_full_np: np.ndarray | None = None,
    left_hand_pose_np: np.ndarray | None = None,
    right_hand_pose_np: np.ndarray | None = None,
    jaw_pose_np: np.ndarray | None = None,
    leye_pose_np: np.ndarray | None = None,
    reye_pose_np: np.ndarray | None = None,
    expression_np: np.ndarray | None = None,
) -> None:
    """
    Export explicit motion parameters to a single NPZ file:
    - global betas (shape)
    - per-frame theta = [global_orient(3), body_pose(23*3)]
    """
    theta_np = np.concatenate([global_orient_np, body_pose_np], axis=1).astype(np.float32)
    betas_per_frame = np.repeat(betas_np.reshape(1, -1), theta_np.shape[0], axis=0).astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "frame_names": np.asarray(frame_names),
        "betas": betas_np.astype(np.float32),
        "betas_per_frame": betas_per_frame,
        "theta": theta_np,
        "theta_global": global_orient_np.astype(np.float32),
        "theta_body": body_pose_np.astype(np.float32),
        "trans": trans_np.astype(np.float32),
        "scale": scale_np.astype(np.float32),
        "model_type": np.asarray([str(smpl_model_type)]),
        "gender": np.asarray([str(smpl_gender)]),
    }
    if theta_full_np is not None:
        payload["theta_full"] = theta_full_np.astype(np.float32)
    if left_hand_pose_np is not None:
        payload["left_hand_pose"] = left_hand_pose_np.astype(np.float32)
    if right_hand_pose_np is not None:
        payload["right_hand_pose"] = right_hand_pose_np.astype(np.float32)
    if jaw_pose_np is not None:
        payload["jaw_pose"] = jaw_pose_np.astype(np.float32)
    if leye_pose_np is not None:
        payload["leye_pose"] = leye_pose_np.astype(np.float32)
    if reye_pose_np is not None:
        payload["reye_pose"] = reye_pose_np.astype(np.float32)
    if expression_np is not None:
        payload["expression"] = expression_np.astype(np.float32)

    np.savez(out_dir / "smpl_motion_params.npz", **payload)


def _export_smpl_joints_sequence(
    out_dir: Path,
    frame_names: list[str],
    joints_cam_np: np.ndarray,
) -> None:
    np.savez(
        out_dir / "smpl_joints_sequence.npz",
        frame_names=np.asarray(frame_names),
        joints_cam=joints_cam_np.astype(np.float32),
    )


def _build_joint_parents_export(
    model,
    joint_count: int,
) -> np.ndarray:
    """
    Build export-friendly parent array with length == joint_count.
    For joints beyond model.parents length, use -1.
    """
    parents = np.full((int(joint_count),), -1, dtype=np.int32)
    p = getattr(model, "parents", None)
    if p is None:
        return parents
    p_np = p.detach().cpu().numpy().astype(np.int32) if torch.is_tensor(p) else np.asarray(p, dtype=np.int32)
    use_n = min(int(joint_count), int(p_np.shape[0]))
    if use_n > 0:
        parents[:use_n] = p_np[:use_n]
    parents[0] = -1
    return parents


def _export_smpl_skeleton_sequence(
    out_dir: Path,
    frame_names: list[str],
    joints_cam_theta_np: np.ndarray,
    joint_parents: np.ndarray,
    model_type: str,
    gender: str,
) -> None:
    np.savez(
        out_dir / "smpl_skeleton_sequence.npz",
        frame_names=np.asarray(frame_names),
        joints_cam_theta=joints_cam_theta_np.astype(np.float32),
        joint_parents=joint_parents.astype(np.int32),
        model_type=np.asarray([str(model_type)]),
        gender=np.asarray([str(gender)]),
        source=np.asarray(["theta"]),
    )


def _export_smpl_vertices_sequence(
    out_dir: Path,
    frame_names: list[str],
    verts_cam_np: np.ndarray,
) -> None:
    np.savez(
        out_dir / "smpl_vertices_sequence.npz",
        frame_names=np.asarray(frame_names),
        verts_cam=verts_cam_np.astype(np.float32),
    )


def _project_joints(joints: torch.Tensor, K: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # joints: (B,M,3), K: (3,3), scale: (1,)
    X = joints * scale.view(1, 1, 1)
    z = X[..., 2].clamp(min=1e-4)
    x = X[..., 0] / z
    y = X[..., 1] / z
    return torch.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], dim=-1)  # (B,M,2)


def _scaled_intrinsics(K_base: torch.Tensor, focal_scale: torch.Tensor) -> torch.Tensor:
    fs = focal_scale.reshape(()).to(dtype=K_base.dtype, device=K_base.device)
    return torch.stack(
        [
            torch.stack([K_base[0, 0] * fs, K_base[0, 1], K_base[0, 2]], dim=0),
            torch.stack([K_base[1, 0], K_base[1, 1] * fs, K_base[1, 2]], dim=0),
            K_base[2],
        ],
        dim=0,
    )


def _resolve_export_frame_index(requested_idx: int, frame_count: int, default_idx: int) -> int:
    if int(frame_count) <= 0:
        return 0
    if int(requested_idx) >= 0:
        return int(np.clip(int(requested_idx), 0, int(frame_count) - 1))
    return int(np.clip(int(default_idx), 0, int(frame_count) - 1))


def _resolve_export_frame_indices(
    requested_indices: list[int] | tuple[int, ...] | None,
    frame_count: int,
    default_idx: int,
) -> list[int]:
    if int(frame_count) <= 0:
        return []
    if requested_indices is None:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw_idx in requested_indices:
        idx = _resolve_export_frame_index(int(raw_idx), frame_count, default_idx)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _cleanup_legacy_stage1_mesh_outputs(out_dir: Path) -> None:
    for name in ("smpl_mesh.obj", "smpl_mesh_selected.obj", "smpl_mesh_last.obj"):
        p = out_dir / name
        if p.exists():
            p.unlink()


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
    head_up_margin: float = 0.06,
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
            margin = float(max(head_up_margin, 1e-3))
            upside = float(head[2]) < float(pelvis[2] - margin)
            shoulder_inv = False
            ankle_inv = False
            if jb.shape[0] > 17:
                shoulder_z = 0.5 * float(jb[16, 2] + jb[17, 2])
                shoulder_inv = shoulder_z < float(pelvis[2] - 0.5 * margin)
            if jb.shape[0] > 8:
                ankle_z = 0.5 * float(jb[7, 2] + jb[8, 2])
                ankle_inv = ankle_z > float(pelvis[2] + 0.5 * margin)
            if upside and (shoulder_inv or ankle_inv):
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
    # Support both SMPL (23 joints) and SMPL-X body pose layouts.
    pose_dim = int(body_pose.shape[1])
    joint_n = pose_dim // 3
    if joint_n <= 4 or pose_dim % 3 != 0:
        zero = torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)
        return zero, zero
    pose = body_pose.view(body_pose.shape[0], joint_n, 3)
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
    pose_dim = int(body_pose.shape[1])
    joint_n = pose_dim // 3
    if joint_n <= 8 or pose_dim % 3 != 0:
        zero = torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)
        return zero, zero
    pose = body_pose.view(body_pose.shape[0], joint_n, 3)
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


def _normalize_smpl_gender(gender: str) -> str:
    g = str(gender).strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female", "woman", "girl"):
        return "female"
    if g in ("n", "neutral", "unknown", "auto"):
        return "neutral"
    return g


def _forward_body_model(
    model,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    left_hand_pose: torch.Tensor | None = None,
    right_hand_pose: torch.Tensor | None = None,
    jaw_pose: torch.Tensor | None = None,
    leye_pose: torch.Tensor | None = None,
    reye_pose: torch.Tensor | None = None,
    expression: torch.Tensor | None = None,
):
    """
    Unified body-model forward.
    For SMPL-X family models, always pass zero hand/face/expression terms so
    batch size is controlled by explicit inputs instead of model default params.
    """
    B = int(global_orient.shape[0])
    if B <= 0:
        raise ValueError("global_orient batch must be > 0")

    if int(betas.shape[0]) == B:
        betas_use = betas
    elif int(betas.shape[0]) == 1:
        betas_use = betas.expand(B, -1)
    else:
        betas_use = betas[:1].expand(B, -1)

    kwargs = dict(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas_use,
        transl=transl,
    )
    dev = global_orient.device
    dt = global_orient.dtype
    if hasattr(model, "left_hand_pose"):
        # Respect the model's configured hand parameterization:
        # PCA hands use low-dim (e.g. 6), full axis-angle hands use 45.
        left_pose_ref = getattr(model, "left_hand_pose")
        right_pose_ref = getattr(model, "right_hand_pose", left_pose_ref)
        left_dim = int(left_pose_ref.shape[-1]) if hasattr(left_pose_ref, "shape") else 0
        right_dim = int(right_pose_ref.shape[-1]) if hasattr(right_pose_ref, "shape") else left_dim
        if left_hand_pose is None:
            left_use = torch.zeros((B, max(0, left_dim)), device=dev, dtype=dt)
        elif int(left_hand_pose.shape[0]) == B:
            left_use = left_hand_pose
        elif int(left_hand_pose.shape[0]) == 1:
            left_use = left_hand_pose.expand(B, -1)
        else:
            left_use = left_hand_pose[:1].expand(B, -1)
        if right_hand_pose is None:
            right_use = torch.zeros((B, max(0, right_dim)), device=dev, dtype=dt)
        elif int(right_hand_pose.shape[0]) == B:
            right_use = right_hand_pose
        elif int(right_hand_pose.shape[0]) == 1:
            right_use = right_hand_pose.expand(B, -1)
        else:
            right_use = right_hand_pose[:1].expand(B, -1)
        kwargs["left_hand_pose"] = left_use
        kwargs["right_hand_pose"] = right_use
    if hasattr(model, "jaw_pose"):
        if jaw_pose is None:
            kwargs["jaw_pose"] = torch.zeros((B, 3), device=dev, dtype=dt)
        elif int(jaw_pose.shape[0]) == B:
            kwargs["jaw_pose"] = jaw_pose
        elif int(jaw_pose.shape[0]) == 1:
            kwargs["jaw_pose"] = jaw_pose.expand(B, -1)
        else:
            kwargs["jaw_pose"] = jaw_pose[:1].expand(B, -1)
    if hasattr(model, "leye_pose"):
        if leye_pose is None:
            kwargs["leye_pose"] = torch.zeros((B, 3), device=dev, dtype=dt)
        elif int(leye_pose.shape[0]) == B:
            kwargs["leye_pose"] = leye_pose
        elif int(leye_pose.shape[0]) == 1:
            kwargs["leye_pose"] = leye_pose.expand(B, -1)
        else:
            kwargs["leye_pose"] = leye_pose[:1].expand(B, -1)
    if hasattr(model, "reye_pose"):
        if reye_pose is None:
            kwargs["reye_pose"] = torch.zeros((B, 3), device=dev, dtype=dt)
        elif int(reye_pose.shape[0]) == B:
            kwargs["reye_pose"] = reye_pose
        elif int(reye_pose.shape[0]) == 1:
            kwargs["reye_pose"] = reye_pose.expand(B, -1)
        else:
            kwargs["reye_pose"] = reye_pose[:1].expand(B, -1)
    if hasattr(model, "expression"):
        expr_ref = getattr(model, "expression")
        expr_dim = int(expr_ref.shape[-1]) if hasattr(expr_ref, "shape") else int(getattr(model, "num_expression_coeffs", 10))
        if expression is None:
            expr_use = torch.zeros((B, max(0, expr_dim)), device=dev, dtype=dt)
        elif int(expression.shape[0]) == B:
            expr_use = expression
        elif int(expression.shape[0]) == 1:
            expr_use = expression.expand(B, -1)
        else:
            expr_use = expression[:1].expand(B, -1)
        kwargs["expression"] = expr_use
    # SMPL-X forward in some versions uses `self.batch_size` internally for
    # landmark barycentric tensors; keep it synchronized with current input B.
    orig_batch_size = getattr(model, "batch_size", None)
    patched_batch_size = False
    if orig_batch_size is not None:
        try:
            if int(orig_batch_size) != B:
                model.batch_size = B
                patched_batch_size = True
        except Exception:
            patched_batch_size = False
    try:
        return model(**kwargs)
    finally:
        if patched_batch_size:
            try:
                model.batch_size = orig_batch_size
            except Exception:
                pass


def _canonicalize_smpl_gender_candidates(candidates: list[str] | tuple[str, ...] | None) -> list[str]:
    default = ["female", "male", "neutral"]
    if candidates is None or len(candidates) == 0:
        return default
    out: list[str] = []
    seen: set[str] = set()
    for g in candidates:
        cg = _normalize_smpl_gender(str(g))
        if cg not in ("female", "male", "neutral"):
            continue
        if cg in seen:
            continue
        seen.add(cg)
        out.append(cg)
    return out if len(out) > 0 else default


def _quick_gender_probe_score(
    model_path: Path,
    smpl_model_type: str,
    smpl_gender: str,
    num_betas: int,
    K_list: list[float],
    targets_xy: torch.Tensor,
    targets_w: torch.Tensor,
    keypoint_loss_type: str,
    keypoint_gmof_sigma_px: float,
    keypoint_conf_thresh: float,
    keypoint_frame_scale_all: torch.Tensor | None,
    fit_iters: int,
    fit_lr: float,
    device: torch.device,
    smpl_indices: list[int],
) -> float:
    use_n = int(targets_xy.shape[0])
    if use_n <= 0:
        return float("inf")

    model_kwargs = dict(
        model_path=str(model_path),
        model_type=str(smpl_model_type),
        gender=_normalize_smpl_gender(str(smpl_gender)),
        num_betas=max(1, int(num_betas)),
        batch_size=use_n,
    )
    if str(smpl_model_type).strip().lower() == "smplx":
        model_kwargs["flat_hand_mean"] = True
    model = smplx.create(**model_kwargs).to(device)
    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3

    global_orient = torch.zeros((use_n, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((use_n, body_pose_dim), device=device, requires_grad=True)
    transl = torch.tensor(
        np.tile(np.asarray([[0.0, 0.0, 2.5]], dtype=np.float32), (use_n, 1)),
        device=device,
        requires_grad=True,
    )
    betas = torch.zeros((1, max(1, int(num_betas))), device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)

    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([global_orient, body_pose, transl, betas, scale], lr=float(fit_lr))

    iters = max(1, int(fit_iters))
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        out = _forward_body_model(
            model=model,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        joints = out.joints[:, smpl_indices_t, :]
        proj = _project_joints(joints, K, scale)
        loss_kpt = _keypoint_reprojection_loss(
            proj_xy=proj,
            target_xy=targets_xy,
            target_w=targets_w,
            loss_type=str(keypoint_loss_type),
            gmof_sigma_px=float(keypoint_gmof_sigma_px),
            frame_scale=keypoint_frame_scale_all,
            conf_thresh=float(keypoint_conf_thresh),
        )
        loss_reg = (
            1e-4 * (global_orient ** 2).sum()
            + 2e-5 * (body_pose ** 2).sum()
            + 5e-4 * (betas ** 2).sum()
            + 5e-3 * ((scale - 1.0) ** 2).sum()
            + 1e-5 * (transl[:, :2] ** 2).sum()
            + 1e-4 * ((transl[:, 2] - 2.5) ** 2).sum()
        )
        loss = loss_kpt + loss_reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scale.clamp_(0.2, 5.0)
            transl[:, 2].clamp_(0.2, 10.0)

    with torch.no_grad():
        out = _forward_body_model(
            model=model,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        joints = out.joints[:, smpl_indices_t, :]
        proj = _project_joints(joints, K, scale)
        score = _keypoint_reprojection_loss(
            proj_xy=proj,
            target_xy=targets_xy,
            target_w=targets_w,
            loss_type=str(keypoint_loss_type),
            gmof_sigma_px=float(keypoint_gmof_sigma_px),
            frame_scale=keypoint_frame_scale_all,
            conf_thresh=float(keypoint_conf_thresh),
        )
    return float(score.detach().cpu())


def select_smpl_gender_from_keypoints(
    keypoints_json: str | Path,
    K_list: list[float],
    smpl_model_path: str | Path,
    smpl_model_type: str = "smplx",
    fit_frames: int = 32,
    num_betas: int = 10,
    keypoint_loss_type: str = "gmof",
    keypoint_gmof_sigma_px: float = 100.0,
    keypoint_conf_thresh: float = 0.02,
    keypoint_bbox_norm: bool = True,
    keypoint_bbox_norm_min_px: float = 96.0,
    probe_iters: int = 80,
    probe_lr: float = 2e-2,
    candidates: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, dict[str, float]]:
    """
    Quick gender probe by fitting each candidate and selecting lowest reprojection loss.
    """
    model_path = _require_smplx(smpl_model_path)
    frame_names_all, frames_all = _load_keypoints_frames(keypoints_json)
    if len(frame_names_all) == 0:
        raise RuntimeError("No keypoints found for gender probe.")
    use_idx = _select_indices(len(frames_all), int(fit_frames))
    frames = [frames_all[i] for i in use_idx]

    mp_indices, smpl_indices, base_w = _build_mp_smpl_correspondence(smpl_model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets_xy, targets_w = _prepare_targets(
        frames,
        mp_indices,
        base_w,
        device,
        min_conf=float(keypoint_conf_thresh),
    )
    keypoint_frame_scale_all = _compute_keypoint_frame_scale(
        targets_xy=targets_xy,
        targets_w=targets_w,
        min_px=float(keypoint_bbox_norm_min_px),
    ) if bool(keypoint_bbox_norm) else None

    gender_scores: dict[str, float] = {}
    candidate_list = _canonicalize_smpl_gender_candidates(candidates)
    for cand in candidate_list:
        try:
            score = _quick_gender_probe_score(
                model_path=model_path,
                smpl_model_type=str(smpl_model_type),
                smpl_gender=str(cand),
                num_betas=max(1, int(num_betas)),
                K_list=K_list,
                targets_xy=targets_xy,
                targets_w=targets_w,
                keypoint_loss_type=str(keypoint_loss_type),
                keypoint_gmof_sigma_px=float(keypoint_gmof_sigma_px),
                keypoint_conf_thresh=float(keypoint_conf_thresh),
                keypoint_frame_scale_all=keypoint_frame_scale_all,
                fit_iters=max(1, int(probe_iters)),
                fit_lr=float(probe_lr),
                device=device,
                smpl_indices=smpl_indices,
            )
        except Exception:
            score = float("inf")
        gender_scores[cand] = float(score)

    valid = {k: v for k, v in gender_scores.items() if np.isfinite(v)}
    if len(valid) == 0:
        return "neutral", gender_scores
    best_gender = min(valid.keys(), key=lambda k: valid[k])
    return best_gender, gender_scores


def fit_smpl_from_keypoints(
    keypoints_json: str | Path,
    out_dir: str | Path,
    K_list: list[float],
    smpl_model_path: str | Path,
    smpl_model_type: str = "smplx",
    smpl_gender: str = "neutral",
    smpl_num_betas: int = 10,
    fit_frames: int = 20,
    fit_iters: int = 300,
    fit_lr: float = 5e-2,
    keypoint_loss_type: str = "gmof",
    keypoint_gmof_sigma_px: float = 100.0,
    keypoint_conf_thresh: float = 0.02,
    keypoint_bbox_norm: bool = True,
    keypoint_bbox_norm_min_px: float = 96.0,
    global_init_iters: int = 60,
    optimize_focal_scale: bool = False,
    focal_scale_l2_weight: float = 1e-3,
    focal_scale_min: float = 0.5,
    focal_scale_max: float = 2.0,
    monocular_depth_weight: float = 0.0,
    monocular_depth_pair_margin: float = 0.01,
    monocular_depth_auto_sign: bool = True,
    world_depth_weight: float = 0.0,
    world_depth_pair_margin: float = 0.01,
    world_depth_auto_sign: bool = True,
    world_min_visibility: float = 0.35,
    world_pair_distance_weight: float = 0.0,
    silhouette_npz: str | Path | None = None,
    silhouette_loss_weight: float = 0.0,
    silhouette_loss_final_ratio: float = 1.0,
    silhouette_dist_clip_px: float = 24.0,
    silhouette_sample_verts: int = 2048,
    silhouette_side_view_boost: float = 1.5,
    silhouette_bbox_weight: float = 0.0,
    limb_vector_weight: float = 0.0,
    limb_vector_len_weight: float = 0.2,
    pose_angle_prior_weight: float = 0.0,
    scale_l2_weight: float = 1e-3,
    init_translation_ls: bool = False,
    loss_schedule: str = "none",
    keypoint_smooth_enable: bool = False,
    keypoint_smooth_method: str = "one_euro",
    keypoint_smooth_fps: float = 30.0,
    keypoint_smooth_min_cutoff: float = 1.0,
    keypoint_smooth_beta: float = 0.3,
    keypoint_smooth_gaussian_kernel_size: int = 7,
    keypoint_smooth_gaussian_sigma: float = 1.5,
    export_pose_mode: str = "fit",
    export_ensure_head_up: bool = False,
    export_head_up_margin: float = 0.06,
    export_center_xyz: bool = False,
    export_on_floor: bool = True,
    export_smooth_iters: int = 0,
    export_smooth_alpha: float = 0.1,
    export_mesh_tpose: bool = False,
    export_frame_index: int = -1,
    enforce_export_upright: bool = False,
    export_center_xy: bool = False,
    export_center_mode: str = "centroid",
    export_meshes: bool = True,
    export_mesh_sequence_indices: list[int] | tuple[int, ...] | None = None,
    export_legacy_mesh_objs: bool = False,
    export_transform_reference: str = "tpose",
):
    """
    Fit one static SMPL body from multi-frame 2D keypoints.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _require_smplx(smpl_model_path)

    frame_names_all, frames_all = _load_keypoints_frames(keypoints_json)
    frames_all = _smooth_keypoint_frames(
        frames=frames_all,
        enable=bool(keypoint_smooth_enable),
        method=str(keypoint_smooth_method),
        fps=float(keypoint_smooth_fps),
        min_cutoff=float(keypoint_smooth_min_cutoff),
        beta=float(keypoint_smooth_beta),
        gaussian_kernel_size=int(keypoint_smooth_gaussian_kernel_size),
        gaussian_sigma=float(keypoint_smooth_gaussian_sigma),
    )
    use_idx = _select_indices(len(frames_all), int(fit_frames))
    frame_names = [frame_names_all[i] for i in use_idx]
    frames = [frames_all[i] for i in use_idx]
    use_n = len(frames)

    mp_indices, smpl_indices, base_w = _build_mp_smpl_correspondence(smpl_model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(
        model_path=str(model_path),
        model_type=smpl_model_type,
        gender=_normalize_smpl_gender(smpl_gender),
        num_betas=max(1, int(smpl_num_betas)),
        batch_size=1,
    )
    if str(smpl_model_type).strip().lower() == "smplx":
        model_kwargs["flat_hand_mean"] = True
    model = smplx.create(**model_kwargs).to(device)
    faces = getattr(model, "faces", None)

    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)
    targets_xy, targets_w = _prepare_targets(
        frames,
        mp_indices,
        base_w,
        device,
        min_conf=float(keypoint_conf_thresh),
    )
    keypoint_frame_scale_all = _compute_keypoint_frame_scale(
        targets_xy=targets_xy,
        targets_w=targets_w,
        min_px=float(keypoint_bbox_norm_min_px),
    ) if bool(keypoint_bbox_norm) else None
    anchor_subset_idx = _resolve_mp_subset_indices(mp_indices, COARSE_FIT_MP_ANCHORS)
    torso_subset_idx = _resolve_mp_subset_indices(mp_indices, TORSO_MP_IDS)
    hand_face_subset_idx = _resolve_mp_subset_indices(mp_indices, HAND_FACE_MP_IDS)
    targets_depth, targets_depth_valid, has_depth = _prepare_depth_targets(frames, mp_indices, device)
    depth_weight_map = targets_w.squeeze(-1) * targets_depth_valid
    use_depth_prior = bool(has_depth) and float(monocular_depth_weight) > 0.0
    targets_world, targets_world_valid, has_world = _prepare_world_targets(
        frames,
        mp_indices,
        device,
        min_visibility=float(world_min_visibility),
    )
    world_weight_map = targets_w.squeeze(-1) * targets_world_valid
    use_world_depth_prior = bool(has_world) and float(world_depth_weight) > 0.0
    use_world_pair_prior = bool(has_world) and float(world_pair_distance_weight) > 0.0
    sil_dist_maps_all = _load_silhouette_distance_maps(
        silhouette_npz=silhouette_npz,
        frame_names=frame_names,
        device=device,
        dist_clip_px=float(silhouette_dist_clip_px),
    )
    use_silhouette_prior = sil_dist_maps_all is not None and float(silhouette_loss_weight) > 0.0
    sil_frame_w_all = _compute_side_view_frame_weights(
        targets_xy=targets_xy,
        targets_w=targets_w,
        mp_indices=mp_indices,
        side_view_boost=float(silhouette_side_view_boost),
    ) if use_silhouette_prior else None
    sil_bbox_all, sil_bbox_valid_all, sil_bbox_hw = _load_silhouette_bboxes(
        silhouette_npz=silhouette_npz,
        frame_names=frame_names,
        device=device,
    )
    use_sil_bbox_prior = (
        sil_bbox_all is not None
        and sil_bbox_valid_all is not None
        and float(silhouette_bbox_weight) > 0.0
    )

    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3

    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, body_pose_dim), device=device, requires_grad=True)
    betas = torch.zeros((1, max(1, int(smpl_num_betas))), device=device, requires_grad=True)
    transl = torch.tensor([[0.0, 0.0, 2.5]], device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)
    focal_scale = torch.tensor(
        [1.0],
        device=device,
        dtype=torch.float32,
        requires_grad=bool(optimize_focal_scale),
    )

    if bool(init_translation_ls):
        with torch.no_grad():
            out_ref = _forward_body_model(
                model=model,
                global_orient=torch.zeros((1, 3), device=device),
                body_pose=torch.zeros((1, body_pose_dim), device=device),
                betas=betas.detach(),
                transl=torch.zeros((1, 3), device=device),
            )
            joints_ref = out_ref.joints[:, smpl_indices_t, :].squeeze(0).cpu().numpy().astype(np.float32)
            t_candidates = _estimate_translation_ls(
                joints_3d_ref=joints_ref,
                targets_xy=targets_xy,
                targets_w=targets_w,
                K=K,
                default_tz=2.5,
            )
            t0 = np.median(t_candidates, axis=0).astype(np.float32)
            transl[:, 0] = float(t0[0])
            transl[:, 1] = float(t0[1])
            transl[:, 2] = float(np.clip(float(t0[2]), 0.2, 10.0))
    batch_frames = 4 if use_n >= 8 else 2
    rng = np.random.default_rng(42)

    warmup_iters = max(int(global_init_iters), 0)
    if warmup_iters > 0 and len(anchor_subset_idx) >= 3:
        warmup_params = [global_orient, transl, scale]
        if bool(optimize_focal_scale):
            warmup_params.append(focal_scale)
        warmup_opt = torch.optim.Adam(warmup_params, lr=float(fit_lr))
        pbar_init = trange(
            warmup_iters,
            desc="Stage1 SMPL warmup (global)",
            dynamic_ncols=True,
        )
        for _ in pbar_init:
            warmup_opt.zero_grad(set_to_none=True)
            idxs = rng.integers(0, use_n, size=batch_frames)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
            out_init = _forward_body_model(
                model=model,
                global_orient=global_orient,
                body_pose=body_pose.detach(),
                betas=betas.detach(),
                transl=transl,
            )
            joints_init = out_init.joints[:, smpl_indices_t, :]
            K_it = _scaled_intrinsics(K, focal_scale)
            proj_init = _project_joints(joints_init, K_it, scale).expand(batch_frames, -1, -1)
            frame_scale_init = (
                keypoint_frame_scale_all[idx_t] if keypoint_frame_scale_all is not None else None
            )
            loss_init_kpt = _keypoint_reprojection_loss(
                proj_xy=proj_init,
                target_xy=targets_xy[idx_t],
                target_w=targets_w[idx_t],
                loss_type=str(keypoint_loss_type),
                gmof_sigma_px=float(keypoint_gmof_sigma_px),
                frame_scale=frame_scale_init,
                conf_thresh=float(keypoint_conf_thresh),
                subset_idx=anchor_subset_idx,
            )
            loss_init_reg = (
                2e-4 * (global_orient ** 2).sum()
                + 1e-5 * (transl[:, :2] ** 2).sum()
                + 1e-4 * ((transl[:, 2] - 2.5) ** 2).sum()
                + 1e-3 * ((scale - 1.0) ** 2).sum()
            )
            if bool(optimize_focal_scale):
                loss_init_reg = loss_init_reg + float(focal_scale_l2_weight) * ((focal_scale - 1.0) ** 2).sum()
            loss_init = loss_init_kpt + loss_init_reg
            loss_init.backward()
            warmup_opt.step()
            with torch.no_grad():
                scale.clamp_(0.2, 5.0)
                transl[:, 2].clamp_(0.2, 10.0)
                if bool(optimize_focal_scale):
                    focal_scale.clamp_(float(focal_scale_min), float(focal_scale_max))
            pbar_init.set_postfix(loss=float(loss_init.detach().cpu()), kpt=float(loss_init_kpt.detach().cpu()))

    train_params = [global_orient, body_pose, betas, transl, scale]
    if bool(optimize_focal_scale):
        train_params.append(focal_scale)
    optimizer = torch.optim.Adam(train_params, lr=float(fit_lr))
    iters = int(fit_iters)
    if iters <= 0:
        raise ValueError(f"fit_iters must be > 0, got {fit_iters}")

    refine_start = int(0.60 * iters)
    refine_lr = float(fit_lr) * 0.2

    pbar = trange(iters, desc="Stage1 SMPL fit (static)", dynamic_ncols=True)
    for it in pbar:
        if it == refine_start:
            for group in optimizer.param_groups:
                group["lr"] = refine_lr

        optimizer.zero_grad(set_to_none=True)

        idxs = rng.integers(0, use_n, size=batch_frames)
        idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

        out = _forward_body_model(
            model=model,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        K_it = _scaled_intrinsics(K, focal_scale)
        joints = out.joints[:, smpl_indices_t, :]  # (1,M,3)
        proj = _project_joints(joints, K_it, scale)   # (1,M,2)
        proj = proj.expand(batch_frames, -1, -1)   # (B,M,2)

        progress = float(it) / float(max(iters - 1, 1))
        phase_w = _loss_phase_multipliers(progress_01=progress, schedule=loss_schedule)

        tgt = targets_xy[idx_t]  # (B,M,2)
        w = targets_w[idx_t]     # (B,M,1)
        w_fit = w * float(phase_w["kpt_global"])
        if len(torso_subset_idx) > 0:
            w_fit[:, torso_subset_idx, :] = w_fit[:, torso_subset_idx, :] * float(phase_w["kpt_torso"])
        if len(hand_face_subset_idx) > 0:
            w_fit[:, hand_face_subset_idx, :] = w_fit[:, hand_face_subset_idx, :] * float(phase_w["kpt_hand_face"])
        frame_scale_t = keypoint_frame_scale_all[idx_t] if keypoint_frame_scale_all is not None else None
        loss_kpt = _keypoint_reprojection_loss(
            proj_xy=proj,
            target_xy=tgt,
            target_w=w_fit,
            loss_type=str(keypoint_loss_type),
            gmof_sigma_px=float(keypoint_gmof_sigma_px),
            frame_scale=frame_scale_t,
            conf_thresh=float(keypoint_conf_thresh),
        )
        loss_depth = _compute_depth_prior_loss(
            joints_sel=joints.expand(batch_frames, -1, -1),
            frame_idx=idx_t,
            targets_depth=targets_depth,
            depth_weight_map=depth_weight_map,
            use_depth_prior=use_depth_prior,
            pair_margin=float(monocular_depth_pair_margin),
            auto_sign=bool(monocular_depth_auto_sign),
        )
        loss_world_depth = _compute_world_depth_prior_loss(
            joints_sel=joints.expand(batch_frames, -1, -1),
            frame_idx=idx_t,
            targets_world=targets_world,
            world_weight_map=world_weight_map,
            use_world_depth_prior=use_world_depth_prior,
            pair_margin=float(world_depth_pair_margin),
            auto_sign=bool(world_depth_auto_sign),
        )
        loss_world_pair = torch.zeros((), dtype=loss_kpt.dtype, device=loss_kpt.device)
        if use_world_pair_prior:
            loss_world_pair = _world_pair_distance_loss(
                smpl_joints_sel=joints.expand(batch_frames, -1, -1),
                mp_world_xyz=targets_world[idx_t],
                mp_world_valid=targets_world_valid[idx_t],
                mp_weight=targets_w[idx_t].squeeze(-1),
                mp_indices=mp_indices,
            )
        loss_sil = _compute_silhouette_prior_loss(
            verts_cam=(out.vertices * scale.view(1, 1, 1)).expand(batch_frames, -1, -1),
            K=K_it,
            frame_idx=idx_t,
            sil_dist_maps_all=sil_dist_maps_all,
            sil_frame_w_all=sil_frame_w_all,
            use_silhouette_prior=use_silhouette_prior,
            sample_verts=int(silhouette_sample_verts),
        )
        loss_sil_bbox = _compute_silhouette_bbox_prior_loss(
            verts_cam=(out.vertices * scale.view(1, 1, 1)).expand(batch_frames, -1, -1),
            K=K_it,
            frame_idx=idx_t,
            sil_bbox_all=sil_bbox_all,
            sil_bbox_valid_all=sil_bbox_valid_all,
            sil_bbox_hw=sil_bbox_hw,
            use_bbox_prior=use_sil_bbox_prior,
        )
        loss_vec = _limb_vector_reprojection_loss(
            proj_xy=proj,
            target_xy=tgt,
            target_w=w_fit,
            mp_indices=mp_indices,
            length_ratio_weight=float(limb_vector_len_weight),
        )
        loss_pose_angle = float(pose_angle_prior_weight) * _loss_pose_angle_constrain(
            global_orient=global_orient,
            body_pose=body_pose,
        )

        loss_reg = (
            1e-4 * (global_orient ** 2).sum()
            + 5e-5 * (body_pose ** 2).sum()
            + 5e-4 * (betas ** 2).sum()
            + float(scale_l2_weight) * ((scale - 1.0) ** 2).sum()
            + 1e-5 * (transl ** 2).sum()
        )
        if bool(optimize_focal_scale):
            loss_reg = loss_reg + float(focal_scale_l2_weight) * ((focal_scale - 1.0) ** 2).sum()
        sil_w_it = _scheduled_weight(
            base_weight=float(silhouette_loss_weight),
            progress_01=progress,
            final_ratio=float(silhouette_loss_final_ratio),
        )
        loss = (
            loss_kpt
            + loss_reg
            + (float(monocular_depth_weight) * float(phase_w["depth"])) * loss_depth
            + (float(world_depth_weight) * float(phase_w["world_depth"])) * loss_world_depth
            + (float(world_pair_distance_weight) * float(phase_w["world_pair"])) * loss_world_pair
            + (float(sil_w_it) * float(phase_w["silhouette"])) * loss_sil
            + (float(silhouette_bbox_weight) * float(phase_w["silhouette_bbox"])) * loss_sil_bbox
            + (float(limb_vector_weight) * float(phase_w["limb"])) * loss_vec
            + loss_pose_angle
        )
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
                dep=float(loss_depth.detach().cpu()),
                wdep=float(loss_world_depth.detach().cpu()),
                wpair=float(loss_world_pair.detach().cpu()),
                sil=float(loss_sil.detach().cpu()),
                silb=float(loss_sil_bbox.detach().cpu()),
                vec=float(loss_vec.detach().cpu()),
                pang=float(loss_pose_angle.detach().cpu()),
                sil_w=float(sil_w_it),
                scale=float(scale.detach().cpu()),
                fsc=float(focal_scale.detach().cpu().item()),
                z=float(transl[:, 2].detach().cpu().item()),
                frames=use_n,
            )

    export_idx = _resolve_export_frame_index(export_frame_index, use_n, max(use_n - 1, 0))
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

        if bool(export_mesh_tpose):
            mesh_go = torch.zeros((1, 3), device=device)
            mesh_bp = torch.zeros((1, body_pose_dim), device=device)
            mesh_tr = transl.detach()
        else:
            mesh_go = export_go
            mesh_bp = export_bp
            mesh_tr = export_tr
        out_mesh = _forward_body_model(
            model=model,
            global_orient=mesh_go,
            body_pose=mesh_bp,
            betas=betas,
            transl=mesh_tr,
        )
        verts_main_raw = (out_mesh.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_main = out_mesh.joints.squeeze(0).cpu().numpy()

        out_last = _forward_body_model(
            model=model,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        verts_last_raw = (out_last.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_last = out_last.joints.squeeze(0).cpu().numpy()

        zero_go = torch.zeros((1, 3), device=device)
        zero_bp = torch.zeros((1, body_pose_dim), device=device)
        out_tpose = _forward_body_model(
            model=model,
            global_orient=zero_go,
            body_pose=zero_bp,
            betas=betas,
            transl=transl.detach(),
        )
        verts_tpose_raw = (out_tpose.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_tpose = out_tpose.joints.squeeze(0).cpu().numpy()

    transform_ref = str(export_transform_reference or "tpose").lower()
    if transform_ref in ("selected", "fit", "pose", "main"):
        ref_verts = verts_main_raw
        ref_joints = joints_main
    elif transform_ref in ("last", "last_frame"):
        ref_verts = verts_last_raw
        ref_joints = joints_last
    else:
        ref_verts = verts_tpose_raw
        ref_joints = joints_tpose

    ref_verts_tx, export_center_xy_val, export_floor, export_upright_rot = _transform_export_vertices(
        ref_verts,
        joints_cam=ref_joints,
        enforce_upright=bool(enforce_export_upright),
        ensure_head_up=bool(export_ensure_head_up),
        head_up_margin=float(export_head_up_margin),
        center_xy=bool(export_center_xy),
        center_xyz=bool(export_center_xyz),
        center_mode=str(export_center_mode),
        put_on_floor=bool(export_on_floor),
    )
    neighbors = _build_vertex_neighbors(num_verts=ref_verts_tx.shape[0], faces=faces)

    def _postprocess_export(verts_raw: np.ndarray, joints_raw: np.ndarray | None) -> np.ndarray:
        verts_tx, _, _, _ = _transform_export_vertices(
            verts_raw,
            joints_cam=joints_raw,
            enforce_upright=bool(enforce_export_upright),
            upright_rot=export_upright_rot,
            ensure_head_up=bool(export_ensure_head_up),
            head_up_margin=float(export_head_up_margin),
            center_xy=bool(export_center_xy),
            xy_center=export_center_xy_val,
            center_xyz=bool(export_center_xyz),
            center_mode=str(export_center_mode),
            put_on_floor=bool(export_on_floor),
            floor_z=export_floor,
        )
        verts_tx = _laplacian_smooth_with_neighbors(
            verts_tx,
            neighbors=neighbors,
            iters=int(export_smooth_iters),
            alpha=float(export_smooth_alpha),
        )
        verts_tx = _force_world_alignment(
            verts=verts_tx,
            center_xy=bool(export_center_xy),
            center_mode=str(export_center_mode),
            put_on_floor=bool(export_on_floor),
        )
        return verts_tx

    verts_main = _postprocess_export(verts_main_raw, joints_main)
    verts_last = _postprocess_export(verts_last_raw, joints_last)
    verts_tpose = _postprocess_export(verts_tpose_raw, joints_tpose)

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
    focal_scale_np = focal_scale.detach().cpu().numpy().reshape(-1).astype(np.float32)
    is_smplx_model = str(smpl_model_type).lower() == "smplx"
    hand_pose_dim = int(getattr(model, "NUM_HAND_JOINTS", 15)) * 3 if is_smplx_model else 0
    expr_dim = int(getattr(model, "num_expression_coeffs", 10)) if is_smplx_model else 0
    left_hand_pose_np = np.zeros((use_n, hand_pose_dim), dtype=np.float32)
    right_hand_pose_np = np.zeros((use_n, hand_pose_dim), dtype=np.float32)
    jaw_pose_np = np.zeros((use_n, 3 if is_smplx_model else 0), dtype=np.float32)
    leye_pose_np = np.zeros((use_n, 3 if is_smplx_model else 0), dtype=np.float32)
    reye_pose_np = np.zeros((use_n, 3 if is_smplx_model else 0), dtype=np.float32)
    expression_np = np.zeros((use_n, expr_dim), dtype=np.float32)
    theta_full_np = np.concatenate(
        [go_seq, bp_seq, left_hand_pose_np, right_hand_pose_np, jaw_pose_np, leye_pose_np, reye_pose_np, expression_np],
        axis=1,
    ).astype(np.float32)
    pose_full = np.concatenate(
        [
            pose,
            np.zeros((hand_pose_dim,), dtype=np.float32),
            np.zeros((hand_pose_dim,), dtype=np.float32),
            np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
            np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
            np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
            np.zeros((expr_dim,), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    np.savez(
        out_dir / "smpl_params.npz",
        pose=pose,
        pose_full=pose_full,
        betas=betas_np,
        trans=trans_np,
        scale=scale_np,
        focal_scale=focal_scale_np,
        model_type=np.asarray([str(smpl_model_type)]),
        gender=np.asarray([_normalize_smpl_gender(smpl_gender)]),
        left_hand_pose=np.zeros((hand_pose_dim,), dtype=np.float32),
        right_hand_pose=np.zeros((hand_pose_dim,), dtype=np.float32),
        jaw_pose=np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
        leye_pose=np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
        reye_pose=np.zeros((3 if is_smplx_model else 0,), dtype=np.float32),
        expression=np.zeros((expr_dim,), dtype=np.float32),
        export_frame_index=np.asarray([export_idx], dtype=np.int32),
    )
    np.savez(out_dir / "smpl_fit_frames.npz", frame_names=np.asarray(frame_names))
    _export_sequence_motion_files(
        out_dir=out_dir,
        frame_names=frame_names,
        betas_np=betas_np,
        global_orient_np=go_seq,
        body_pose_np=bp_seq,
        trans_np=tr_seq,
        scale_np=scale_np,
        smpl_model_type=str(smpl_model_type),
        smpl_gender=_normalize_smpl_gender(smpl_gender),
        theta_full_np=theta_full_np,
        left_hand_pose_np=left_hand_pose_np,
        right_hand_pose_np=right_hand_pose_np,
        jaw_pose_np=jaw_pose_np,
        leye_pose_np=leye_pose_np,
        reye_pose_np=reye_pose_np,
        expression_np=expression_np,
    )
    with torch.no_grad():
        out_j = _forward_body_model(
            model=model,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        joints_one = (out_j.joints.squeeze(0) * scale.view(1, 1)).cpu().numpy().astype(np.float32)
        verts_one = (out_j.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy().astype(np.float32)
    joints_seq = np.repeat(joints_one[None, :, :], use_n, axis=0).astype(np.float32)
    verts_seq = np.repeat(verts_one[None, :, :], use_n, axis=0).astype(np.float32)
    _export_smpl_joints_sequence(out_dir=out_dir, frame_names=frame_names, joints_cam_np=joints_seq)
    joint_parents = _build_joint_parents_export(model=model, joint_count=int(joints_seq.shape[1]))
    _export_smpl_skeleton_sequence(
        out_dir=out_dir,
        frame_names=frame_names,
        joints_cam_theta_np=joints_seq,
        joint_parents=joint_parents,
        model_type=str(smpl_model_type),
        gender=_normalize_smpl_gender(smpl_gender),
    )
    _export_smpl_vertices_sequence(out_dir=out_dir, frame_names=frame_names, verts_cam_np=verts_seq)
    _cleanup_legacy_stage1_mesh_outputs(out_dir)
    if bool(export_legacy_mesh_objs):
        _export_obj(out_dir / "smpl_mesh.obj", verts_main, faces)
        _export_obj(out_dir / "smpl_mesh_last.obj", verts_last, faces)
        _export_obj(out_dir / "smpl_mesh_selected.obj", verts_last, faces)
    _export_obj(out_dir / "smpl_mesh_tpose.obj", verts_tpose, faces)
    mesh_dir = out_dir / "smpl_mesh_sequence"
    if bool(export_meshes):
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for old_obj in mesh_dir.glob("*.obj"):
            old_obj.unlink()
        mesh_export_ids = _resolve_export_frame_indices(
            requested_indices=export_mesh_sequence_indices,
            frame_count=use_n,
            default_idx=max(use_n - 1, 0),
        )
        if len(mesh_export_ids) == 0:
            mesh_export_ids = list(range(use_n))
        for fi in mesh_export_ids:
            _export_obj(mesh_dir / f"{int(fi):06d}.obj", verts_last, faces)
    elif mesh_dir.exists():
        for old_obj in mesh_dir.glob("*.obj"):
            old_obj.unlink()
    return out_dir


def fit_smpl_sequence_from_keypoints(
    keypoints_json: str | Path,
    out_dir: str | Path,
    K_list: list[float],
    smpl_model_path: str | Path,
    smpl_model_type: str = "smplx",
    smpl_gender: str = "neutral",
    smpl_num_betas: int = 10,
    fit_frames: int = 120,
    fit_iters: int = 2000,
    fit_lr: float = 3e-2,
    keypoint_loss_type: str = "gmof",
    keypoint_gmof_sigma_px: float = 100.0,
    keypoint_conf_thresh: float = 0.02,
    keypoint_bbox_norm: bool = True,
    keypoint_bbox_norm_min_px: float = 96.0,
    global_init_iters: int = 120,
    optimize_focal_scale: bool = False,
    focal_scale_l2_weight: float = 1e-3,
    focal_scale_min: float = 0.5,
    focal_scale_max: float = 2.0,
    frame_batch: int = 16,
    temporal_pose_weight: float = 5e-4,
    temporal_orient_weight: float = 1e-3,
    temporal_trans_weight: float = 5e-3,
    temporal_window: int = 1,
    temporal_window_decay: float = 0.7,
    temporal_pose_accel_weight: float = 0.0,
    temporal_orient_accel_weight: float = 0.0,
    temporal_trans_accel_weight: float = 0.0,
    monocular_depth_weight: float = 0.0,
    monocular_depth_pair_margin: float = 0.01,
    monocular_depth_auto_sign: bool = True,
    last_frame_depth_weight: float = -1.0,
    world_depth_weight: float = 0.0,
    world_depth_pair_margin: float = 0.01,
    world_depth_auto_sign: bool = True,
    world_min_visibility: float = 0.35,
    last_frame_world_depth_weight: float = -1.0,
    world_pair_distance_weight: float = 0.0,
    last_frame_world_pair_weight: float = -1.0,
    silhouette_npz: str | Path | None = None,
    silhouette_loss_weight: float = 0.0,
    silhouette_loss_final_ratio: float = 1.0,
    silhouette_dist_clip_px: float = 24.0,
    silhouette_sample_verts: int = 2048,
    silhouette_side_view_boost: float = 1.5,
    silhouette_bbox_weight: float = 0.0,
    last_frame_silhouette_weight: float = -1.0,
    last_frame_silhouette_bbox_weight: float = -1.0,
    limb_vector_weight: float = 0.0,
    limb_vector_len_weight: float = 0.2,
    last_frame_limb_vector_weight: float = -1.0,
    pose_angle_prior_weight: float = 0.0,
    optimize_hand_pose: bool = True,
    optimize_face_pose: bool = True,
    optimize_expression: bool = False,
    hand_pose_reg_weight: float = 2e-4,
    face_pose_reg_weight: float = 2e-4,
    expression_reg_weight: float = 1e-3,
    hand_pose_temporal_weight: float = 5e-4,
    face_pose_temporal_weight: float = 5e-4,
    init_translation_ls: bool = False,
    loss_schedule: str = "none",
    keypoint_smooth_enable: bool = False,
    keypoint_smooth_method: str = "one_euro",
    keypoint_smooth_fps: float = 30.0,
    keypoint_smooth_min_cutoff: float = 1.0,
    keypoint_smooth_beta: float = 0.3,
    keypoint_smooth_gaussian_kernel_size: int = 7,
    keypoint_smooth_gaussian_sigma: float = 1.5,
    optimize_betas: bool = False,
    unfreeze_betas_iter: int = -1,
    betas_l2_weight: float = 5e-2,
    betas_l2_final_ratio: float = 1.0,
    betas_abs_max: float = 3.0,
    scale_l2_weight: float = 1e-3,
    shape_from_previous_frames: bool = True,
    final_pose_from_last_frame: bool = True,
    last_frame_refine_iters: int = 300,
    last_frame_temporal_window: int = 0,
    last_frame_temporal_decay: float = 0.8,
    last_frame_temporal_pose_weight: float = 0.0,
    last_frame_temporal_orient_weight: float = 0.0,
    last_frame_temporal_trans_weight: float = 0.0,
    export_pose_mode: str = "last_frame",
    export_ensure_head_up: bool = False,
    export_head_up_margin: float = 0.06,
    upright_prior_weight: float = 1e-2,
    knee_hinge_weight: float = 2e-3,
    knee_direction_weight: float = 5e-3,
    knee_hinge_axis: int = 1,
    knee_min_flex: float = 0.0,
    left_knee_flex_sign: float = -1.0,
    right_knee_flex_sign: float = -1.0,
    auto_knee_sign: bool = False,
    knee_sign_update_every: int = 200,
    ankle_twist_weight: float = 3e-3,
    back_stability_weight: float = 6e-3,
    enforce_export_upright: bool = True,
    export_center_xy: bool = True,
    export_center_xyz: bool = False,
    export_center_mode: str = "pelvis",
    export_on_floor: bool = True,
    export_smooth_iters: int = 0,
    export_smooth_alpha: float = 0.1,
    export_mesh_tpose: bool = False,
    export_frame_index: int = -1,
    export_mesh_sequence_indices: list[int] | tuple[int, ...] | None = None,
    export_mesh_sequence_last_only: bool = False,
    export_legacy_mesh_objs: bool = False,
    export_transform_reference: str = "tpose",
    last_frame_upper_body_boost: float = 1.0,
    full_batch_every: int = 200,
    export_meshes: bool = True,
):
    """
    Fit a moving-person SMPL sequence:
    - shared betas/scale
    - per-frame global_orient/body_pose/trans
    - base camera intrinsics with optional learnable focal scaling
    - temporal continuity over neighboring frames (window + acceleration regularization)
    - by default, body shape is estimated from earlier frames and final pose uses last frame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _require_smplx(smpl_model_path)

    frame_names_all, frames_all = _load_keypoints_frames(keypoints_json)
    frames_all = _smooth_keypoint_frames(
        frames=frames_all,
        enable=bool(keypoint_smooth_enable),
        method=str(keypoint_smooth_method),
        fps=float(keypoint_smooth_fps),
        min_cutoff=float(keypoint_smooth_min_cutoff),
        beta=float(keypoint_smooth_beta),
        gaussian_kernel_size=int(keypoint_smooth_gaussian_kernel_size),
        gaussian_sigma=float(keypoint_smooth_gaussian_sigma),
    )
    use_idx = _select_indices(len(frames_all), int(fit_frames))
    frame_names = [frame_names_all[i] for i in use_idx]
    frames = [frames_all[i] for i in use_idx]
    use_n = len(frames)

    mp_indices, smpl_indices, base_w = _build_mp_smpl_correspondence(smpl_model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = max(1, min(int(frame_batch), use_n))

    model_kwargs = dict(
        model_path=str(model_path),
        model_type=smpl_model_type,
        gender=_normalize_smpl_gender(smpl_gender),
        num_betas=max(1, int(smpl_num_betas)),
        batch_size=batch_size,
    )
    if str(smpl_model_type).strip().lower() == "smplx":
        # Prefer flat hand mean to reduce unrealistic default finger curl.
        model_kwargs["flat_hand_mean"] = True
    model = smplx.create(**model_kwargs).to(device)
    faces = getattr(model, "faces", None)

    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)
    smpl_indices_t = torch.tensor(smpl_indices, dtype=torch.long, device=device)
    targets_xy, targets_w = _prepare_targets(
        frames,
        mp_indices,
        base_w,
        device,
        min_conf=float(keypoint_conf_thresh),
    )
    keypoint_frame_scale_all = _compute_keypoint_frame_scale(
        targets_xy=targets_xy,
        targets_w=targets_w,
        min_px=float(keypoint_bbox_norm_min_px),
    ) if bool(keypoint_bbox_norm) else None
    anchor_subset_idx = _resolve_mp_subset_indices(mp_indices, COARSE_FIT_MP_ANCHORS)
    torso_subset_idx = _resolve_mp_subset_indices(mp_indices, TORSO_MP_IDS)
    hand_face_subset_idx = _resolve_mp_subset_indices(mp_indices, HAND_FACE_MP_IDS)
    targets_depth, targets_depth_valid, has_depth = _prepare_depth_targets(frames, mp_indices, device)
    depth_weight_map = targets_w.squeeze(-1) * targets_depth_valid
    use_depth_prior = bool(has_depth) and float(monocular_depth_weight) > 0.0
    targets_world, targets_world_valid, has_world = _prepare_world_targets(
        frames,
        mp_indices,
        device,
        min_visibility=float(world_min_visibility),
    )
    world_weight_map = targets_w.squeeze(-1) * targets_world_valid
    use_world_depth_prior = bool(has_world) and (
        float(world_depth_weight) > 0.0 or float(last_frame_world_depth_weight) > 0.0
    )
    use_world_pair_prior = bool(has_world) and (
        float(world_pair_distance_weight) > 0.0 or float(last_frame_world_pair_weight) > 0.0
    )
    sil_dist_maps_all = _load_silhouette_distance_maps(
        silhouette_npz=silhouette_npz,
        frame_names=frame_names,
        device=device,
        dist_clip_px=float(silhouette_dist_clip_px),
    )
    use_silhouette_prior = sil_dist_maps_all is not None and float(silhouette_loss_weight) > 0.0
    sil_frame_w_all = _compute_side_view_frame_weights(
        targets_xy=targets_xy,
        targets_w=targets_w,
        mp_indices=mp_indices,
        side_view_boost=float(silhouette_side_view_boost),
    ) if use_silhouette_prior else None
    sil_bbox_all, sil_bbox_valid_all, sil_bbox_hw = _load_silhouette_bboxes(
        silhouette_npz=silhouette_npz,
        frame_names=frame_names,
        device=device,
    )
    use_sil_bbox_prior = (
        sil_bbox_all is not None
        and sil_bbox_valid_all is not None
        and (
            float(silhouette_bbox_weight) > 0.0
            or float(last_frame_silhouette_bbox_weight) > 0.0
        )
    )
    body_joints = int(getattr(model, "NUM_BODY_JOINTS", 23))
    body_pose_dim = body_joints * 3
    is_smplx_model = str(smpl_model_type).strip().lower() == "smplx"
    hand_pose_dim = int(getattr(model, "left_hand_pose").shape[-1]) if hasattr(model, "left_hand_pose") else 0
    expr_dim = int(getattr(model, "expression").shape[-1]) if hasattr(model, "expression") else 0
    use_hand_opt = bool(is_smplx_model and optimize_hand_pose and hand_pose_dim > 0)
    use_face_pose_opt = bool(is_smplx_model and optimize_face_pose)
    use_expr_opt = bool(is_smplx_model and optimize_expression and expr_dim > 0)

    global_orient = torch.zeros((use_n, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((use_n, body_pose_dim), device=device, requires_grad=True)
    transl = torch.zeros((use_n, 3), device=device, requires_grad=True)
    with torch.no_grad():
        transl[:, 2] = 2.5
    betas = torch.zeros((1, max(1, int(smpl_num_betas))), device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)
    left_hand_pose = torch.zeros((use_n, max(0, hand_pose_dim)), device=device, requires_grad=use_hand_opt)
    right_hand_pose = torch.zeros((use_n, max(0, hand_pose_dim)), device=device, requires_grad=use_hand_opt)
    jaw_pose = torch.zeros((use_n, 3), device=device, requires_grad=use_face_pose_opt)
    leye_pose = torch.zeros((use_n, 3), device=device, requires_grad=use_face_pose_opt)
    reye_pose = torch.zeros((use_n, 3), device=device, requires_grad=use_face_pose_opt)
    expression = torch.zeros((use_n, max(0, expr_dim)), device=device, requires_grad=use_expr_opt)
    focal_scale = torch.tensor(
        [1.0],
        device=device,
        dtype=torch.float32,
        requires_grad=bool(optimize_focal_scale),
    )
    fit_pool_n = use_n - 1 if (shape_from_previous_frames and use_n > 1) else use_n
    rng = np.random.default_rng(42)

    if bool(init_translation_ls) and use_n > 0:
        with torch.no_grad():
            out_ref = _forward_body_model(
                model=model,
                global_orient=torch.zeros((1, 3), device=device),
                body_pose=torch.zeros((1, body_pose_dim), device=device),
                betas=betas.detach(),
                transl=torch.zeros((1, 3), device=device),
                left_hand_pose=(torch.zeros((1, max(0, hand_pose_dim)), device=device) if use_hand_opt else None),
                right_hand_pose=(torch.zeros((1, max(0, hand_pose_dim)), device=device) if use_hand_opt else None),
                jaw_pose=(torch.zeros((1, 3), device=device) if use_face_pose_opt else None),
                leye_pose=(torch.zeros((1, 3), device=device) if use_face_pose_opt else None),
                reye_pose=(torch.zeros((1, 3), device=device) if use_face_pose_opt else None),
                expression=(torch.zeros((1, max(0, expr_dim)), device=device) if use_expr_opt else None),
            )
            joints_ref = out_ref.joints[:, smpl_indices_t, :].squeeze(0).cpu().numpy().astype(np.float32)
            t_init = _estimate_translation_ls(
                joints_3d_ref=joints_ref,
                targets_xy=targets_xy,
                targets_w=targets_w,
                K=K,
                default_tz=2.5,
            )
            transl[:, :] = torch.from_numpy(t_init).to(device=device, dtype=torch.float32)
            transl[:, 2].clamp_(0.2, 10.0)

    warmup_iters = max(int(global_init_iters), 0)
    if warmup_iters > 0 and fit_pool_n > 0 and len(anchor_subset_idx) >= 3:
        warmup_params = [global_orient, transl, scale]
        if bool(optimize_focal_scale):
            warmup_params.append(focal_scale)
        warmup_opt = torch.optim.Adam(warmup_params, lr=float(fit_lr))
        pbar_init = trange(
            warmup_iters,
            desc="Stage1 SMPL warmup (global)",
            dynamic_ncols=True,
        )
        for _ in pbar_init:
            warmup_opt.zero_grad(set_to_none=True)
            if batch_size >= fit_pool_n:
                idxs = np.arange(fit_pool_n, dtype=int)
            else:
                idxs = rng.choice(fit_pool_n, size=batch_size, replace=False)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
            B0 = int(idx_t.numel())
            out_init = _forward_body_model(
                model=model,
                global_orient=global_orient[idx_t],
                body_pose=body_pose[idx_t].detach(),
                betas=betas.detach(),
                transl=transl[idx_t],
                left_hand_pose=(left_hand_pose[idx_t].detach() if use_hand_opt else None),
                right_hand_pose=(right_hand_pose[idx_t].detach() if use_hand_opt else None),
                jaw_pose=(jaw_pose[idx_t].detach() if use_face_pose_opt else None),
                leye_pose=(leye_pose[idx_t].detach() if use_face_pose_opt else None),
                reye_pose=(reye_pose[idx_t].detach() if use_face_pose_opt else None),
                expression=(expression[idx_t].detach() if use_expr_opt else None),
            )
            joints_init = out_init.joints[:, smpl_indices_t, :]
            K_it = _scaled_intrinsics(K, focal_scale)
            proj_init = _project_joints(joints_init, K_it, scale)
            frame_scale_init = (
                keypoint_frame_scale_all[idx_t] if keypoint_frame_scale_all is not None else None
            )
            loss_init_kpt = _keypoint_reprojection_loss(
                proj_xy=proj_init,
                target_xy=targets_xy[idx_t],
                target_w=targets_w[idx_t],
                loss_type=str(keypoint_loss_type),
                gmof_sigma_px=float(keypoint_gmof_sigma_px),
                frame_scale=frame_scale_init,
                conf_thresh=float(keypoint_conf_thresh),
                subset_idx=anchor_subset_idx,
            )
            loss_init_reg = (
                2e-4 * (global_orient[idx_t] ** 2).mean()
                + 1e-5 * (transl[idx_t, :2] ** 2).mean()
                + 1e-4 * ((transl[idx_t, 2] - 2.5) ** 2).mean()
                + 1e-3 * ((scale - 1.0) ** 2).sum()
            )
            if bool(optimize_focal_scale):
                loss_init_reg = loss_init_reg + float(focal_scale_l2_weight) * ((focal_scale - 1.0) ** 2).sum()
            loss_init = loss_init_kpt + loss_init_reg
            loss_init.backward()
            warmup_opt.step()
            with torch.no_grad():
                scale.clamp_(0.2, 5.0)
                transl[:, 2].clamp_(0.2, 10.0)
                if bool(optimize_focal_scale):
                    focal_scale.clamp_(float(focal_scale_min), float(focal_scale_max))
            pbar_init.set_postfix(loss=float(loss_init.detach().cpu()), kpt=float(loss_init_kpt.detach().cpu()))

    main_params = [global_orient, body_pose, transl, scale]
    if bool(optimize_focal_scale):
        main_params.append(focal_scale)
    if use_hand_opt:
        main_params.extend([left_hand_pose, right_hand_pose])
    if use_face_pose_opt:
        main_params.extend([jaw_pose, leye_pose, reye_pose])
    if use_expr_opt:
        main_params.append(expression)

    optimizer = torch.optim.Adam(
        [
            {"params": main_params, "lr": float(fit_lr)},
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

    pbar = trange(iters, desc="Stage1 SMPL fit (sequence)", dynamic_ncols=True)
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
        progress = float(it) / float(max(iters - 1, 1))
        phase_w = _loss_phase_multipliers(progress_01=progress, schedule=loss_schedule)

        out = _forward_body_model(
            model=model,
            global_orient=global_orient[idx_t],
            body_pose=body_pose[idx_t],
            betas=betas,
            transl=transl[idx_t],
            left_hand_pose=(left_hand_pose[idx_t] if use_hand_opt else None),
            right_hand_pose=(right_hand_pose[idx_t] if use_hand_opt else None),
            jaw_pose=(jaw_pose[idx_t] if use_face_pose_opt else None),
            leye_pose=(leye_pose[idx_t] if use_face_pose_opt else None),
            reye_pose=(reye_pose[idx_t] if use_face_pose_opt else None),
            expression=(expression[idx_t] if use_expr_opt else None),
        )
        K_it = _scaled_intrinsics(K, focal_scale)
        joints = out.joints[:, smpl_indices_t, :]  # (B,M,3)
        proj = _project_joints(joints, K_it, scale)   # (B,M,2)

        tgt = targets_xy[idx_t]  # (B,M,2)
        w = targets_w[idx_t]     # (B,M,1)
        w_fit = w * float(phase_w["kpt_global"])
        if len(torso_subset_idx) > 0:
            w_fit[:, torso_subset_idx, :] = w_fit[:, torso_subset_idx, :] * float(phase_w["kpt_torso"])
        if len(hand_face_subset_idx) > 0:
            w_fit[:, hand_face_subset_idx, :] = (
                w_fit[:, hand_face_subset_idx, :] * float(phase_w["kpt_hand_face"])
            )
        frame_scale_t = keypoint_frame_scale_all[idx_t] if keypoint_frame_scale_all is not None else None
        loss_kpt = _keypoint_reprojection_loss(
            proj_xy=proj,
            target_xy=tgt,
            target_w=w_fit,
            loss_type=str(keypoint_loss_type),
            gmof_sigma_px=float(keypoint_gmof_sigma_px),
            frame_scale=frame_scale_t,
            conf_thresh=float(keypoint_conf_thresh),
        )
        loss_depth = _compute_depth_prior_loss(
            joints_sel=joints,
            frame_idx=idx_t,
            targets_depth=targets_depth,
            depth_weight_map=depth_weight_map,
            use_depth_prior=use_depth_prior,
            pair_margin=float(monocular_depth_pair_margin),
            auto_sign=bool(monocular_depth_auto_sign),
        )
        loss_world_depth = _compute_world_depth_prior_loss(
            joints_sel=joints,
            frame_idx=idx_t,
            targets_world=targets_world,
            world_weight_map=world_weight_map,
            use_world_depth_prior=use_world_depth_prior and float(world_depth_weight) > 0.0,
            pair_margin=float(world_depth_pair_margin),
            auto_sign=bool(world_depth_auto_sign),
        )
        loss_world_pair = torch.zeros((), dtype=loss_kpt.dtype, device=loss_kpt.device)
        if use_world_pair_prior and float(world_pair_distance_weight) > 0.0:
            loss_world_pair = _world_pair_distance_loss(
                smpl_joints_sel=joints,
                mp_world_xyz=targets_world[idx_t],
                mp_world_valid=targets_world_valid[idx_t],
                mp_weight=targets_w[idx_t].squeeze(-1),
                mp_indices=mp_indices,
            )
        loss_sil = _compute_silhouette_prior_loss(
            verts_cam=out.vertices * scale.view(1, 1, 1),
            K=K_it,
            frame_idx=idx_t,
            sil_dist_maps_all=sil_dist_maps_all,
            sil_frame_w_all=sil_frame_w_all,
            use_silhouette_prior=use_silhouette_prior,
            sample_verts=int(silhouette_sample_verts),
        )
        loss_sil_bbox = _compute_silhouette_bbox_prior_loss(
            verts_cam=out.vertices * scale.view(1, 1, 1),
            K=K_it,
            frame_idx=idx_t,
            sil_bbox_all=sil_bbox_all,
            sil_bbox_valid_all=sil_bbox_valid_all,
            sil_bbox_hw=sil_bbox_hw,
            use_bbox_prior=use_sil_bbox_prior and float(silhouette_bbox_weight) > 0.0,
        )
        loss_vec = _limb_vector_reprojection_loss(
            proj_xy=proj,
            target_xy=tgt,
            target_w=w_fit,
            mp_indices=mp_indices,
            length_ratio_weight=float(limb_vector_len_weight),
        )
        loss_pose_angle = float(pose_angle_prior_weight) * _loss_pose_angle_constrain(
            global_orient=global_orient[idx_t],
            body_pose=body_pose[idx_t],
        )

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

        beta_l2_w = _scheduled_weight(
            base_weight=float(betas_l2_weight),
            progress_01=progress,
            final_ratio=float(betas_l2_final_ratio),
        )
        loss_reg = (
            2e-4 * (global_orient ** 2).mean()
            + 1e-4 * (body_pose ** 2).mean()
            + float(beta_l2_w) * (betas ** 2).sum()
            + float(scale_l2_weight) * ((scale - 1.0) ** 2).sum()
            + 1e-5 * (transl[:, :2] ** 2).mean()
            + 1e-4 * ((transl[:, 2] - 2.5) ** 2).mean()
        )
        if use_hand_opt:
            loss_reg = loss_reg + (float(hand_pose_reg_weight) * float(phase_w["hand_face_reg"])) * (
                (left_hand_pose[idx_t] ** 2).mean() + (right_hand_pose[idx_t] ** 2).mean()
            )
        if use_face_pose_opt:
            loss_reg = loss_reg + (float(face_pose_reg_weight) * float(phase_w["hand_face_reg"])) * (
                (jaw_pose[idx_t] ** 2).mean()
                + (leye_pose[idx_t] ** 2).mean()
                + (reye_pose[idx_t] ** 2).mean()
            )
        if use_expr_opt:
            loss_reg = loss_reg + (float(expression_reg_weight) * float(phase_w["hand_face_reg"])) * (
                expression[idx_t] ** 2
            ).mean()
        if bool(optimize_focal_scale):
            loss_reg = loss_reg + float(focal_scale_l2_weight) * ((focal_scale - 1.0) ** 2).sum()
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
            temporal_w = float(phase_w["temporal"])
            hand_face_temporal_w = float(phase_w["hand_face_temporal"])
            loss_temp = (
                (float(temporal_pose_weight) * temporal_w)
                * _temporal_window_smooth_loss(
                    body_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                )
                + (float(temporal_orient_weight) * temporal_w)
                * _temporal_window_smooth_loss(
                    global_orient, window=int(temporal_window), decay=float(temporal_window_decay)
                )
                + (float(temporal_trans_weight) * temporal_w)
                * _temporal_window_smooth_loss(
                    transl, window=int(temporal_window), decay=float(temporal_window_decay)
                )
            )
            loss_temp = (
                loss_temp
                + (float(temporal_pose_accel_weight) * temporal_w) * _temporal_accel_loss(body_pose)
                + (float(temporal_orient_accel_weight) * temporal_w) * _temporal_accel_loss(global_orient)
                + (float(temporal_trans_accel_weight) * temporal_w) * _temporal_accel_loss(transl)
            )
            if use_hand_opt and float(hand_pose_temporal_weight) > 0.0:
                loss_temp = loss_temp + (float(hand_pose_temporal_weight) * hand_face_temporal_w) * (
                    _temporal_window_smooth_loss(
                        left_hand_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                    + _temporal_window_smooth_loss(
                        right_hand_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                )
            if use_face_pose_opt and float(face_pose_temporal_weight) > 0.0:
                loss_temp = loss_temp + (float(face_pose_temporal_weight) * hand_face_temporal_w) * (
                    _temporal_window_smooth_loss(
                        jaw_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                    + _temporal_window_smooth_loss(
                        leye_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                    + _temporal_window_smooth_loss(
                        reye_pose, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                )
            if use_expr_opt and float(face_pose_temporal_weight) > 0.0:
                loss_temp = loss_temp + (
                    0.5
                    * float(face_pose_temporal_weight)
                    * hand_face_temporal_w
                    * _temporal_window_smooth_loss(
                        expression, window=int(temporal_window), decay=float(temporal_window_decay)
                    )
                )

        sil_w_it = _scheduled_weight(
            base_weight=float(silhouette_loss_weight),
            progress_01=progress,
            final_ratio=float(silhouette_loss_final_ratio),
        )
        loss = (
            loss_kpt
            + (float(monocular_depth_weight) * float(phase_w["depth"])) * loss_depth
            + (float(world_depth_weight) * float(phase_w["world_depth"])) * loss_world_depth
            + (float(world_pair_distance_weight) * float(phase_w["world_pair"])) * loss_world_pair
            + (float(sil_w_it) * float(phase_w["silhouette"])) * loss_sil
            + (float(silhouette_bbox_weight) * float(phase_w["silhouette_bbox"])) * loss_sil_bbox
            + (float(limb_vector_weight) * float(phase_w["limb"])) * loss_vec
            + loss_pose_angle
            + loss_reg
            + loss_temp
            + loss_upright
            + loss_knee
            + loss_phys_extra
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scale.clamp_(0.2, 5.0)
            transl[:, 2].clamp_(0.2, 10.0)
            if bool(optimize_focal_scale):
                focal_scale.clamp_(float(focal_scale_min), float(focal_scale_max))
            betas.clamp_(-float(betas_abs_max), float(betas_abs_max))

        if it % 10 == 0 or it == iters - 1:
            pbar.set_postfix(
                lr=optimizer.param_groups[0]["lr"],
                loss=float(loss.detach().cpu()),
                kpt=float(loss_kpt.detach().cpu()),
                dep=float(loss_depth.detach().cpu()),
                wdep=float(loss_world_depth.detach().cpu()),
                wpair=float(loss_world_pair.detach().cpu()),
                sil=float(loss_sil.detach().cpu()),
                silb=float(loss_sil_bbox.detach().cpu()),
                vec=float(loss_vec.detach().cpu()),
                pang=float(loss_pose_angle.detach().cpu()),
                sil_w=float(sil_w_it),
                reg=float(loss_reg.detach().cpu()),
                temp=float(loss_temp.detach().cpu()),
                up=float(loss_upright.detach().cpu()),
                knee=float(loss_knee.detach().cpu()),
                phys=float(loss_phys_extra.detach().cpu()),
                scale=float(scale.detach().cpu()),
                fsc=float(focal_scale.detach().cpu().item()),
                z=float(transl[:, 2].mean().detach().cpu()),
                beta_w=float(beta_l2_w),
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
        lh_last = left_hand_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(use_hand_opt)
        rh_last = right_hand_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(use_hand_opt)
        jaw_last = jaw_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(use_face_pose_opt)
        leye_last = leye_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(use_face_pose_opt)
        reye_last = reye_pose[last_idx:last_idx + 1].detach().clone().requires_grad_(use_face_pose_opt)
        expr_last = expression[last_idx:last_idx + 1].detach().clone().requires_grad_(use_expr_opt)
        last_params = [go_last, bp_last, tr_last]
        if use_hand_opt:
            last_params.extend([lh_last, rh_last])
        if use_face_pose_opt:
            last_params.extend([jaw_last, leye_last, reye_last])
        if use_expr_opt:
            last_params.append(expr_last)
        last_opt = torch.optim.Adam(last_params, lr=float(fit_lr) * 0.25)
        tgt_last = targets_xy[last_idx:last_idx + 1]
        w_last = targets_w[last_idx:last_idx + 1]
        w_last_fit = w_last.clone()
        if float(last_frame_upper_body_boost) != 1.0:
            id_map_last = {int(mp): i for i, mp in enumerate(mp_indices)}
            for mp in (11, 12, 13, 14, 15, 16):
                if mp in id_map_last:
                    w_last_fit[:, id_map_last[mp], :] = (
                        w_last_fit[:, id_map_last[mp], :] * float(last_frame_upper_body_boost)
                    )
        last_idx_t = torch.tensor([last_idx], dtype=torch.long, device=device)
        depth_w_last = float(monocular_depth_weight)
        if float(last_frame_depth_weight) >= 0.0:
            depth_w_last = float(last_frame_depth_weight)
        world_depth_w_last = float(world_depth_weight)
        if float(last_frame_world_depth_weight) >= 0.0:
            world_depth_w_last = float(last_frame_world_depth_weight)
        world_pair_w_last = float(world_pair_distance_weight)
        if float(last_frame_world_pair_weight) >= 0.0:
            world_pair_w_last = float(last_frame_world_pair_weight)
        sil_w_last = float(silhouette_loss_weight)
        if float(last_frame_silhouette_weight) >= 0.0:
            sil_w_last = float(last_frame_silhouette_weight)
        sil_bbox_w_last = float(silhouette_bbox_weight)
        if float(last_frame_silhouette_bbox_weight) >= 0.0:
            sil_bbox_w_last = float(last_frame_silhouette_bbox_weight)
        vec_w_last = float(limb_vector_weight)
        if float(last_frame_limb_vector_weight) >= 0.0:
            vec_w_last = float(last_frame_limb_vector_weight)
        ctx_weights = None
        go_ctx = None
        bp_ctx = None
        tr_ctx = None
        if int(last_frame_temporal_window) > 0 and last_idx > 0:
            start = max(0, last_idx - int(last_frame_temporal_window))
            ctx_idx = torch.arange(start, last_idx, dtype=torch.long, device=device)
            if int(ctx_idx.numel()) > 0:
                dist = (last_idx - ctx_idx).to(dtype=torch.float32)
                decay = float(np.clip(last_frame_temporal_decay, 0.0, 1.0))
                if decay <= 0.0:
                    raw_w = torch.ones_like(dist)
                else:
                    raw_w = torch.pow(torch.tensor(decay, dtype=torch.float32, device=device), dist)
                ctx_weights = raw_w / (raw_w.sum() + 1e-6)
                go_ctx = global_orient[ctx_idx].detach()
                bp_ctx = body_pose[ctx_idx].detach()
                tr_ctx = transl[ctx_idx].detach()
        K_last = _scaled_intrinsics(K, focal_scale.detach())
        for _ in range(int(last_frame_refine_iters)):
            last_opt.zero_grad(set_to_none=True)
            out_last = _forward_body_model(
                model=model,
                global_orient=go_last,
                body_pose=bp_last,
                betas=betas.detach(),
                transl=tr_last,
                left_hand_pose=(lh_last if use_hand_opt else None),
                right_hand_pose=(rh_last if use_hand_opt else None),
                jaw_pose=(jaw_last if use_face_pose_opt else None),
                leye_pose=(leye_last if use_face_pose_opt else None),
                reye_pose=(reye_last if use_face_pose_opt else None),
                expression=(expr_last if use_expr_opt else None),
            )
            joints_last = out_last.joints[:, smpl_indices_t, :]
            proj_last = _project_joints(joints_last, K_last, scale)
            frame_scale_last = (
                keypoint_frame_scale_all[last_idx_t] if keypoint_frame_scale_all is not None else None
            )
            loss_last_kpt = _keypoint_reprojection_loss(
                proj_xy=proj_last,
                target_xy=tgt_last,
                target_w=w_last_fit,
                loss_type=str(keypoint_loss_type),
                gmof_sigma_px=float(keypoint_gmof_sigma_px),
                frame_scale=frame_scale_last,
                conf_thresh=float(keypoint_conf_thresh),
            )
            loss_last_depth = _compute_depth_prior_loss(
                joints_sel=joints_last,
                frame_idx=last_idx_t,
                targets_depth=targets_depth,
                depth_weight_map=depth_weight_map,
                use_depth_prior=bool(use_depth_prior and depth_w_last > 0.0),
                pair_margin=float(monocular_depth_pair_margin),
                auto_sign=bool(monocular_depth_auto_sign),
            )
            loss_last_world_depth = _compute_world_depth_prior_loss(
                joints_sel=joints_last,
                frame_idx=last_idx_t,
                targets_world=targets_world,
                world_weight_map=world_weight_map,
                use_world_depth_prior=bool(use_world_depth_prior and world_depth_w_last > 0.0),
                pair_margin=float(world_depth_pair_margin),
                auto_sign=bool(world_depth_auto_sign),
            )
            loss_last_world_pair = torch.zeros((), dtype=loss_last_kpt.dtype, device=loss_last_kpt.device)
            if bool(use_world_pair_prior and world_pair_w_last > 0.0):
                loss_last_world_pair = _world_pair_distance_loss(
                    smpl_joints_sel=joints_last,
                    mp_world_xyz=targets_world[last_idx_t],
                    mp_world_valid=targets_world_valid[last_idx_t],
                    mp_weight=targets_w[last_idx_t].squeeze(-1),
                    mp_indices=mp_indices,
                )
            loss_last_sil = _compute_silhouette_prior_loss(
                verts_cam=out_last.vertices * scale.view(1, 1, 1),
                K=K_last,
                frame_idx=last_idx_t,
                sil_dist_maps_all=sil_dist_maps_all,
                sil_frame_w_all=sil_frame_w_all,
                use_silhouette_prior=bool(use_silhouette_prior and sil_w_last > 0.0),
                sample_verts=int(silhouette_sample_verts),
            )
            loss_last_sil_bbox = _compute_silhouette_bbox_prior_loss(
                verts_cam=out_last.vertices * scale.view(1, 1, 1),
                K=K_last,
                frame_idx=last_idx_t,
                sil_bbox_all=sil_bbox_all,
                sil_bbox_valid_all=sil_bbox_valid_all,
                sil_bbox_hw=sil_bbox_hw,
                use_bbox_prior=bool(use_sil_bbox_prior and sil_bbox_w_last > 0.0),
            )
            loss_last_vec = _limb_vector_reprojection_loss(
                proj_xy=proj_last,
                target_xy=tgt_last,
                target_w=w_last_fit,
                mp_indices=mp_indices,
                length_ratio_weight=float(limb_vector_len_weight),
            )
            loss_last_pose_angle = float(pose_angle_prior_weight) * _loss_pose_angle_constrain(
                global_orient=go_last,
                body_pose=bp_last,
            )
            loss_last_reg = (
                1e-4 * (go_last ** 2).sum()
                + 5e-5 * (bp_last ** 2).sum()
                + 1e-5 * (tr_last[:, :2] ** 2).sum()
                + 1e-4 * ((tr_last[:, 2] - 2.5) ** 2).sum()
            )
            if use_hand_opt:
                loss_last_reg = loss_last_reg + float(hand_pose_reg_weight) * (
                    (lh_last ** 2).mean() + (rh_last ** 2).mean()
                )
            if use_face_pose_opt:
                loss_last_reg = loss_last_reg + float(face_pose_reg_weight) * (
                    (jaw_last ** 2).mean() + (leye_last ** 2).mean() + (reye_last ** 2).mean()
                )
            if use_expr_opt:
                loss_last_reg = loss_last_reg + float(expression_reg_weight) * (expr_last ** 2).mean()
            loss_last_temp = torch.zeros((), dtype=loss_last_reg.dtype, device=loss_last_reg.device)
            if ctx_weights is not None and go_ctx is not None and bp_ctx is not None and tr_ctx is not None:
                cw = ctx_weights.view(-1, 1)
                loss_last_temp = (
                    float(last_frame_temporal_orient_weight)
                    * (cw * ((go_last - go_ctx) ** 2)).sum(dim=0).mean()
                    + float(last_frame_temporal_pose_weight)
                    * (cw * ((bp_last - bp_ctx) ** 2)).sum(dim=0).mean()
                    + float(last_frame_temporal_trans_weight)
                    * (cw * ((tr_last - tr_ctx) ** 2)).sum(dim=0).mean()
                )
            loss_last = (
                loss_last_kpt
                + loss_last_reg
                + depth_w_last * loss_last_depth
                + world_depth_w_last * loss_last_world_depth
                + world_pair_w_last * loss_last_world_pair
                + sil_w_last * loss_last_sil
                + sil_bbox_w_last * loss_last_sil_bbox
                + vec_w_last * loss_last_vec
                + loss_last_pose_angle
                + loss_last_temp
            )
            loss_last.backward()
            last_opt.step()
            with torch.no_grad():
                tr_last[:, 2].clamp_(0.2, 10.0)
        with torch.no_grad():
            global_orient[last_idx:last_idx + 1] = go_last
            body_pose[last_idx:last_idx + 1] = bp_last
            transl[last_idx:last_idx + 1] = tr_last
            if use_hand_opt:
                left_hand_pose[last_idx:last_idx + 1] = lh_last
                right_hand_pose[last_idx:last_idx + 1] = rh_last
            if use_face_pose_opt:
                jaw_pose[last_idx:last_idx + 1] = jaw_last
                leye_pose[last_idx:last_idx + 1] = leye_last
                reye_pose[last_idx:last_idx + 1] = reye_last
            if use_expr_opt:
                expression[last_idx:last_idx + 1] = expr_last

    pose_mode = str(export_pose_mode or "last_frame").lower()
    default_export_idx = (use_n - 1) if (final_pose_from_last_frame and use_n > 0) else 0
    export_idx = _resolve_export_frame_index(export_frame_index, use_n, default_export_idx)
    if pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
        export_go = torch.zeros((1, 3), device=device)
        export_bp = torch.zeros((1, body_pose_dim), device=device)
        export_tr = transl.mean(dim=0, keepdim=True).detach()
    else:
        export_go = global_orient[export_idx:export_idx + 1]
        export_bp = body_pose[export_idx:export_idx + 1]
        export_tr = transl[export_idx:export_idx + 1]

    anchor_idx = _resolve_export_frame_index(export_idx, use_n, 0)
    last_idx_export = _resolve_export_frame_index(-1, use_n, use_n - 1)
    with torch.no_grad():
        out_anchor = _forward_body_model(
            model=model,
            global_orient=global_orient[anchor_idx:anchor_idx + 1],
            body_pose=body_pose[anchor_idx:anchor_idx + 1],
            betas=betas,
            transl=transl[anchor_idx:anchor_idx + 1],
            left_hand_pose=(left_hand_pose[anchor_idx:anchor_idx + 1] if use_hand_opt else None),
            right_hand_pose=(right_hand_pose[anchor_idx:anchor_idx + 1] if use_hand_opt else None),
            jaw_pose=(jaw_pose[anchor_idx:anchor_idx + 1] if use_face_pose_opt else None),
            leye_pose=(leye_pose[anchor_idx:anchor_idx + 1] if use_face_pose_opt else None),
            reye_pose=(reye_pose[anchor_idx:anchor_idx + 1] if use_face_pose_opt else None),
            expression=(expression[anchor_idx:anchor_idx + 1] if use_expr_opt else None),
        )
        verts_selected_raw = (out_anchor.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_selected = out_anchor.joints.squeeze(0).cpu().numpy()

        out_last_mesh = _forward_body_model(
            model=model,
            global_orient=global_orient[last_idx_export:last_idx_export + 1],
            body_pose=body_pose[last_idx_export:last_idx_export + 1],
            betas=betas,
            transl=transl[last_idx_export:last_idx_export + 1],
            left_hand_pose=(left_hand_pose[last_idx_export:last_idx_export + 1] if use_hand_opt else None),
            right_hand_pose=(right_hand_pose[last_idx_export:last_idx_export + 1] if use_hand_opt else None),
            jaw_pose=(jaw_pose[last_idx_export:last_idx_export + 1] if use_face_pose_opt else None),
            leye_pose=(leye_pose[last_idx_export:last_idx_export + 1] if use_face_pose_opt else None),
            reye_pose=(reye_pose[last_idx_export:last_idx_export + 1] if use_face_pose_opt else None),
            expression=(expression[last_idx_export:last_idx_export + 1] if use_expr_opt else None),
        )
        verts_last_raw = (out_last_mesh.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_last = out_last_mesh.joints.squeeze(0).cpu().numpy()

        zero_go = torch.zeros((1, 3), device=device)
        zero_bp = torch.zeros((1, body_pose_dim), device=device)
        out_tpose = _forward_body_model(
            model=model,
            global_orient=zero_go,
            body_pose=zero_bp,
            betas=betas,
            transl=transl.mean(dim=0, keepdim=True).detach(),
            left_hand_pose=(left_hand_pose.mean(dim=0, keepdim=True).detach() if use_hand_opt else None),
            right_hand_pose=(right_hand_pose.mean(dim=0, keepdim=True).detach() if use_hand_opt else None),
            jaw_pose=(jaw_pose.mean(dim=0, keepdim=True).detach() if use_face_pose_opt else None),
            leye_pose=(leye_pose.mean(dim=0, keepdim=True).detach() if use_face_pose_opt else None),
            reye_pose=(reye_pose.mean(dim=0, keepdim=True).detach() if use_face_pose_opt else None),
            expression=(expression.mean(dim=0, keepdim=True).detach() if use_expr_opt else None),
        )
        verts_tpose_raw = (out_tpose.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
        joints_tpose = out_tpose.joints.squeeze(0).cpu().numpy()

    transform_ref = str(export_transform_reference or "tpose").lower()
    if transform_ref in ("selected", "fit", "pose", "main"):
        ref_verts = verts_selected_raw
        ref_joints = joints_selected
    elif transform_ref in ("last", "last_frame"):
        ref_verts = verts_last_raw
        ref_joints = joints_last
    else:
        ref_verts = verts_tpose_raw
        ref_joints = joints_tpose

    ref_verts_tx, export_center_xy_val, export_floor, export_upright_rot = _transform_export_vertices(
        ref_verts,
        joints_cam=ref_joints,
        enforce_upright=bool(enforce_export_upright),
        ensure_head_up=bool(export_ensure_head_up),
        head_up_margin=float(export_head_up_margin),
        center_xy=bool(export_center_xy),
        center_xyz=bool(export_center_xyz),
        center_mode=str(export_center_mode),
        put_on_floor=bool(export_on_floor),
    )
    neighbors = _build_vertex_neighbors(num_verts=ref_verts_tx.shape[0], faces=faces)

    def _postprocess_export(verts_raw: np.ndarray, joints_raw: np.ndarray | None) -> np.ndarray:
        verts_tx, _, _, _ = _transform_export_vertices(
            verts_raw,
            joints_cam=joints_raw,
            enforce_upright=bool(enforce_export_upright),
            upright_rot=export_upright_rot,
            ensure_head_up=bool(export_ensure_head_up),
            head_up_margin=float(export_head_up_margin),
            center_xy=bool(export_center_xy),
            xy_center=export_center_xy_val,
            center_xyz=bool(export_center_xyz),
            center_mode=str(export_center_mode),
            put_on_floor=bool(export_on_floor),
            floor_z=export_floor,
        )
        verts_tx = _laplacian_smooth_with_neighbors(
            verts_tx,
            neighbors=neighbors,
            iters=int(export_smooth_iters),
            alpha=float(export_smooth_alpha),
        )
        verts_tx = _force_world_alignment(
            verts=verts_tx,
            center_xy=bool(export_center_xy),
            center_mode=str(export_center_mode),
            put_on_floor=bool(export_on_floor),
        )
        return verts_tx

    verts_selected = _postprocess_export(verts_selected_raw, joints_selected)
    verts_last = _postprocess_export(verts_last_raw, joints_last)
    verts_tpose = _postprocess_export(verts_tpose_raw, joints_tpose)

    if bool(export_mesh_tpose) or pose_mode in ("tpose", "t-pose", "arms_out", "fixed"):
        verts0 = verts_tpose
    else:
        verts0 = verts_selected

    global_orient_np = global_orient.detach().cpu().numpy().astype(np.float32)
    body_pose_np = body_pose.detach().cpu().numpy().astype(np.float32)
    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)
    trans_np = transl.detach().cpu().numpy().astype(np.float32)
    scale_np = scale.detach().cpu().numpy().reshape(-1).astype(np.float32)
    focal_scale_np = focal_scale.detach().cpu().numpy().reshape(-1).astype(np.float32)
    with torch.no_grad():
        joints_seq = []
        verts_seq = []
        for fi in range(use_n):
            out_i = _forward_body_model(
                model=model,
                global_orient=global_orient[fi:fi + 1],
                body_pose=body_pose[fi:fi + 1],
                betas=betas,
                transl=transl[fi:fi + 1],
                left_hand_pose=(left_hand_pose[fi:fi + 1] if use_hand_opt else None),
                right_hand_pose=(right_hand_pose[fi:fi + 1] if use_hand_opt else None),
                jaw_pose=(jaw_pose[fi:fi + 1] if use_face_pose_opt else None),
                leye_pose=(leye_pose[fi:fi + 1] if use_face_pose_opt else None),
                reye_pose=(reye_pose[fi:fi + 1] if use_face_pose_opt else None),
                expression=(expression[fi:fi + 1] if use_expr_opt else None),
            )
            joints_i = (out_i.joints.squeeze(0) * scale.view(1, 1)).cpu().numpy().astype(np.float32)
            verts_i = (out_i.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy().astype(np.float32)
            joints_seq.append(joints_i)
            verts_seq.append(verts_i)
        joints_seq_np = np.stack(joints_seq, axis=0).astype(np.float32)
        verts_seq_np = np.stack(verts_seq, axis=0).astype(np.float32)

    theta_np = np.concatenate([global_orient_np, body_pose_np], axis=1).astype(np.float32)
    betas_per_frame_np = np.repeat(betas_np.reshape(1, -1), use_n, axis=0).astype(np.float32)
    is_smplx_model = str(smpl_model_type).strip().lower() == "smplx"
    hand_pose_dim = int(left_hand_pose.shape[1]) if is_smplx_model else 0
    expr_dim = int(expression.shape[1]) if is_smplx_model else 0
    left_hand_pose_np = left_hand_pose.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    right_hand_pose_np = right_hand_pose.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    jaw_pose_np = jaw_pose.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    leye_pose_np = leye_pose.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    reye_pose_np = reye_pose.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    expression_np = expression.detach().cpu().numpy().astype(np.float32) if is_smplx_model else np.zeros((use_n, 0), dtype=np.float32)
    theta_full_np = np.concatenate(
        [global_orient_np, body_pose_np, left_hand_pose_np, right_hand_pose_np, jaw_pose_np, leye_pose_np, reye_pose_np, expression_np],
        axis=1,
    ).astype(np.float32)

    requested_mesh_ids = _resolve_export_frame_indices(
        requested_indices=export_mesh_sequence_indices,
        frame_count=use_n,
        default_idx=max(use_n - 1, 0),
    )
    if len(requested_mesh_ids) == 0:
        if bool(export_mesh_sequence_last_only):
            requested_mesh_ids = [max(use_n - 1, 0)]
        else:
            requested_mesh_ids = list(range(use_n))
    mesh_export_ids_np = np.asarray(requested_mesh_ids, dtype=np.int32)

    np.savez(
        out_dir / "smpl_params_sequence.npz",
        global_orient=global_orient_np,
        body_pose=body_pose_np,
        theta=theta_np,
        theta_full=theta_full_np,
        theta_global=global_orient_np,
        theta_body=body_pose_np,
        left_hand_pose=left_hand_pose_np,
        right_hand_pose=right_hand_pose_np,
        jaw_pose=jaw_pose_np,
        leye_pose=leye_pose_np,
        reye_pose=reye_pose_np,
        expression=expression_np,
        betas=betas_np,
        betas_per_frame=betas_per_frame_np,
        trans=trans_np,
        scale=scale_np,
        focal_scale=focal_scale_np,
        frame_names=np.asarray(frame_names),
        frame_indices=use_idx.astype(np.int32),
        mesh_export_indices=mesh_export_ids_np,
        model_type=np.asarray([str(smpl_model_type)]),
        gender=np.asarray([_normalize_smpl_gender(smpl_gender)]),
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
        pose0_lh = left_hand_pose_np.mean(axis=0).astype(np.float32) if hand_pose_dim > 0 else np.zeros((0,), dtype=np.float32)
        pose0_rh = right_hand_pose_np.mean(axis=0).astype(np.float32) if hand_pose_dim > 0 else np.zeros((0,), dtype=np.float32)
        pose0_jaw = jaw_pose_np.mean(axis=0).astype(np.float32) if jaw_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_leye = leye_pose_np.mean(axis=0).astype(np.float32) if leye_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_reye = reye_pose_np.mean(axis=0).astype(np.float32) if reye_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_expr = expression_np.mean(axis=0).astype(np.float32) if expr_dim > 0 else np.zeros((0,), dtype=np.float32)
    else:
        pose0 = np.concatenate([global_orient_np[export_idx], body_pose_np[export_idx]], axis=0).astype(np.float32)
        trans0 = trans_np[export_idx]
        pose0_lh = left_hand_pose_np[export_idx] if hand_pose_dim > 0 else np.zeros((0,), dtype=np.float32)
        pose0_rh = right_hand_pose_np[export_idx] if hand_pose_dim > 0 else np.zeros((0,), dtype=np.float32)
        pose0_jaw = jaw_pose_np[export_idx] if jaw_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_leye = leye_pose_np[export_idx] if leye_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_reye = reye_pose_np[export_idx] if reye_pose_np.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
        pose0_expr = expression_np[export_idx] if expr_dim > 0 else np.zeros((0,), dtype=np.float32)
    pose0_full = np.concatenate(
        [
            pose0,
            pose0_lh,
            pose0_rh,
            pose0_jaw,
            pose0_leye,
            pose0_reye,
            pose0_expr,
        ],
        axis=0,
    ).astype(np.float32)
    np.savez(
        out_dir / "smpl_params.npz",
        pose=pose0,
        pose_full=pose0_full,
        betas=betas_np,
        trans=trans0,
        scale=scale_np,
        focal_scale=focal_scale_np,
        model_type=np.asarray([str(smpl_model_type)]),
        gender=np.asarray([_normalize_smpl_gender(smpl_gender)]),
        left_hand_pose=pose0_lh.astype(np.float32),
        right_hand_pose=pose0_rh.astype(np.float32),
        jaw_pose=pose0_jaw.astype(np.float32),
        leye_pose=pose0_leye.astype(np.float32),
        reye_pose=pose0_reye.astype(np.float32),
        expression=pose0_expr.astype(np.float32),
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
        smpl_model_type=str(smpl_model_type),
        smpl_gender=_normalize_smpl_gender(smpl_gender),
        theta_full_np=theta_full_np,
        left_hand_pose_np=left_hand_pose_np,
        right_hand_pose_np=right_hand_pose_np,
        jaw_pose_np=jaw_pose_np,
        leye_pose_np=leye_pose_np,
        reye_pose_np=reye_pose_np,
        expression_np=expression_np,
    )
    _export_smpl_joints_sequence(out_dir=out_dir, frame_names=frame_names, joints_cam_np=joints_seq_np)
    joint_parents = _build_joint_parents_export(model=model, joint_count=int(joints_seq_np.shape[1]))
    _export_smpl_skeleton_sequence(
        out_dir=out_dir,
        frame_names=frame_names,
        joints_cam_theta_np=joints_seq_np,
        joint_parents=joint_parents,
        model_type=str(smpl_model_type),
        gender=_normalize_smpl_gender(smpl_gender),
    )
    _export_smpl_vertices_sequence(out_dir=out_dir, frame_names=frame_names, verts_cam_np=verts_seq_np)

    _cleanup_legacy_stage1_mesh_outputs(out_dir)
    if bool(export_legacy_mesh_objs):
        _export_obj(out_dir / "smpl_mesh.obj", verts0, faces)
        _export_obj(out_dir / "smpl_mesh_selected.obj", verts_selected, faces)
        _export_obj(out_dir / "smpl_mesh_last.obj", verts_last, faces)
    _export_obj(out_dir / "smpl_mesh_tpose.obj", verts_tpose, faces)

    mesh_dir = out_dir / "smpl_mesh_sequence"
    if export_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for old_obj in mesh_dir.glob("*.obj"):
            old_obj.unlink()
        with torch.no_grad():
            for fi in requested_mesh_ids:
                out_i = _forward_body_model(
                    model=model,
                    global_orient=global_orient[fi:fi + 1],
                    body_pose=body_pose[fi:fi + 1],
                    betas=betas,
                    transl=transl[fi:fi + 1],
                    left_hand_pose=(left_hand_pose[fi:fi + 1] if use_hand_opt else None),
                    right_hand_pose=(right_hand_pose[fi:fi + 1] if use_hand_opt else None),
                    jaw_pose=(jaw_pose[fi:fi + 1] if use_face_pose_opt else None),
                    leye_pose=(leye_pose[fi:fi + 1] if use_face_pose_opt else None),
                    reye_pose=(reye_pose[fi:fi + 1] if use_face_pose_opt else None),
                    expression=(expression[fi:fi + 1] if use_expr_opt else None),
                )
                verts_i = (out_i.vertices.squeeze(0) * scale.view(1, 1)).cpu().numpy()
                verts_i = _postprocess_export(verts_i, None)
                _export_obj(mesh_dir / f"{fi:06d}.obj", verts_i, faces)
    elif mesh_dir.exists():
        for old_obj in mesh_dir.glob("*.obj"):
            old_obj.unlink()

    return out_dir
