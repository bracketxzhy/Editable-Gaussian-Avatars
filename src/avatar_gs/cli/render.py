from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, resolve_path
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices
from avatar_gs.gaussian.binding import (
    BindingFrameData,
    build_binding_frame_data,
    build_vertex_adjacency,
    compute_area_weighted_vertex_normals_np,
    parse_binding_mode_config,
)
from avatar_gs.gaussian.binding_layers import (
    apply_inner_binding_per_frame,
    apply_outer_anchor_per_frame,
)
from avatar_gs.gaussian.rigid_align import apply_rigid_transform_np
from avatar_gs.stage2 import (
    Stage2MeshRuntime,
    project_points_numpy as _project_points_numpy,
    rasterize_mesh_mask as _rasterize_mesh_mask,
)
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.deform.smplx_param_source import load_deform_smplx_vertices_from_npz as _load_deform_smplx_vertices_from_npz


MP_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 31), (31, 29),
    (24, 26), (26, 28), (28, 32), (32, 30),
    (0, 11), (0, 12),
]
MP_BODY_DRAW_IDS = sorted(set([0] + [a for a, _ in MP_POSE_CONNECTIONS] + [b for _, b in MP_POSE_CONNECTIONS]))
SMPL_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 16), (16, 18), (18, 20),
    (12, 17), (17, 19), (19, 21),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
]
SMPLX_OVERLAY_JOINT_COUNT = 49


def _parse_frame_idx_from_name(frame_name: str) -> int | None:
    stem = Path(str(frame_name)).stem.strip().lower()
    if stem.isdigit():
        return int(stem)
    if stem.startswith("frame_") and stem[6:].isdigit():
        return int(stem[6:])
    return None


def _build_pose_drive_indices(
    *,
    frame_names: list[str],
    base_indices: list[int],
    available_ids: list[int],
    mode: str,
    fixed_index: int,
    loop: bool,
) -> list[int]:
    if len(base_indices) != len(frame_names):
        raise ValueError("base_indices and frame_names length mismatch.")
    if len(available_ids) == 0:
        raise RuntimeError("pose-drive source has no available frame indices.")
    avail = sorted(int(x) for x in available_ids)
    avail_set = set(avail)
    fallback = int(avail[-1])
    mode_l = str(mode).strip().lower()
    out: list[int] = []
    for i, fn in enumerate(frame_names):
        idx = None
        if mode_l in ("frame_name", "name"):
            parsed = _parse_frame_idx_from_name(fn)
            if parsed is not None and parsed in avail_set:
                idx = int(parsed)
            elif bool(loop):
                idx = int(avail[i % len(avail)])
        elif mode_l in ("bind", "bind_index"):
            base_idx = int(base_indices[i])
            if base_idx in avail_set:
                idx = base_idx
            elif bool(loop):
                idx = int(avail[i % len(avail)])
        elif mode_l in ("sequential", "seq"):
            idx = int(avail[i % len(avail)]) if bool(loop) else (int(avail[i]) if i < len(avail) else fallback)
        elif mode_l in ("fixed", "single"):
            if int(fixed_index) in avail_set:
                idx = int(fixed_index)
            else:
                idx = fallback
        else:
            raise ValueError(
                "render.pose_drive_index_mode must be one of: frame_name|bind|sequential|fixed. "
                f"Got: {mode}"
            )
        if idx is None:
            idx = fallback
        out.append(int(idx))
    return out


def _load_stage2_smpl_align_subset(
    *,
    cache_path: Path,
    frames: list[str],
    logger,
) -> dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None
    try:
        z = np.load(str(cache_path), allow_pickle=True)
    except Exception as e:
        logger.warning("Failed to read stage2 supervision cache for render-time SMPL alignment (%s): %s", cache_path, e)
        return None
    required = ("frame_names", "smpl_align_scales", "smpl_align_tx_px", "smpl_align_ty_px", "smpl_align_accepted")
    missing = [k for k in required if k not in z.files]
    if missing:
        return None
    cache_frames = [str(x) for x in np.asarray(z["frame_names"]).reshape(-1).tolist()]
    name_to_idx = {n: i for i, n in enumerate(cache_frames)}
    scales = np.ones((len(frames),), dtype=np.float32)
    tx_px = np.zeros((len(frames),), dtype=np.float32)
    ty_px = np.zeros((len(frames),), dtype=np.float32)
    accepted = np.zeros((len(frames),), dtype=np.uint8)
    src_scales = np.asarray(z["smpl_align_scales"], dtype=np.float32).reshape(-1)
    src_tx = np.asarray(z["smpl_align_tx_px"], dtype=np.float32).reshape(-1)
    src_ty = np.asarray(z["smpl_align_ty_px"], dtype=np.float32).reshape(-1)
    src_accepted = np.asarray(z["smpl_align_accepted"], dtype=np.uint8).reshape(-1)
    matched = 0
    for i, fn in enumerate(frames):
        j = name_to_idx.get(str(fn), None)
        if j is None:
            continue
        scales[i] = float(src_scales[j])
        tx_px[i] = float(src_tx[j])
        ty_px[i] = float(src_ty[j])
        accepted[i] = np.uint8(src_accepted[j])
        matched += 1
    if matched <= 0:
        return None
    logger.info(
        "Loaded frame-wise SMPL alignment for render: matched %d/%d frame(s) from %s",
        int(matched),
        int(len(frames)),
        str(cache_path),
    )
    return {
        "frame_names": np.asarray([str(x) for x in frames]),
        "smpl_align_scales": scales,
        "smpl_align_tx_px": tx_px,
        "smpl_align_ty_px": ty_px,
        "smpl_align_accepted": accepted,
    }


def _load_stage1_frame_K_subset(
    *,
    stage1_dir: Path,
    frames: list[str],
    logger,
) -> list[np.ndarray] | None:
    seq_npz_path = stage1_dir / "smpl_params_sequence.npz"
    if not seq_npz_path.exists():
        return None
    try:
        z = np.load(str(seq_npz_path), allow_pickle=True)
    except Exception as e:
        logger.warning("Failed to read stage1 frame intrinsics for render (%s): %s", seq_npz_path, e)
        return None
    required = ("frame_names", "camera_focal_xy", "camera_princpt_xy")
    if any(key not in z.files for key in required):
        return None
    src_frames = [str(x) for x in np.asarray(z["frame_names"]).reshape(-1).tolist()]
    focal_xy = np.asarray(z["camera_focal_xy"], dtype=np.float32).reshape(-1, 2)
    princpt_xy = np.asarray(z["camera_princpt_xy"], dtype=np.float32).reshape(-1, 2)
    n = min(len(src_frames), int(focal_xy.shape[0]), int(princpt_xy.shape[0]))
    name_to_idx = {src_frames[i]: i for i in range(n)}
    frame_Ks: list[np.ndarray] = []
    matched = 0
    for fn in frames:
        j = name_to_idx.get(str(fn), None)
        if j is None:
            return None
        K_np = np.asarray(
            [
                [float(focal_xy[j, 0]), 0.0, float(princpt_xy[j, 0])],
                [0.0, float(focal_xy[j, 1]), float(princpt_xy[j, 1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        frame_Ks.append(K_np)
        matched += 1
    if matched <= 0:
        return None
    logger.info(
        "Loaded frame-wise intrinsics for render: matched %d/%d frame(s) from %s",
        matched,
        len(frames),
        seq_npz_path,
    )
    return frame_Ks


def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read image to (3,H,W) float in [0,1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle -> R."""
    theta = torch.norm(w) + 1e-8
    k = w / theta
    kx, ky, kz = k[0], k[1], k[2]
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    K = torch.stack(
        [
            torch.stack([z, -kz, ky]),
            torch.stack([kz, z, -kx]),
            torch.stack([-ky, kx, z]),
        ],
        dim=0,
    )
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return I + s * K + (1.0 - c) * (K @ K)



def _euler_deg_to_rotmat_torch(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    axis_mode: str = "legacy",
) -> torch.Tensor:
    """
    Convert Euler angles in degrees to a 3x3 rotation matrix.

    axis_mode:
      - legacy: yaw around Y, pitch around X, roll around Z (old behavior)
      - z_up:   yaw around Z, pitch around X, roll around Y
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)

    dev = torch.device(device) if device is not None else None

    # Common X rotation (pitch)
    Rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=dtype,
        device=dev,
    )

    m = str(axis_mode).strip().lower()
    if m == "z_up":
        # yaw around Z, roll around Y
        R_yaw = torch.tensor(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=dev,
        )
        R_roll = torch.tensor(
            [[cz, 0.0, sz], [0.0, 1.0, 0.0], [-sz, 0.0, cz]],
            dtype=dtype,
            device=dev,
        )
        return R_roll @ Rx @ R_yaw

    # legacy: yaw around Y, roll around Z
    R_yaw = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=dtype,
        device=dev,
    )
    R_roll = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=dev,
    )
    return R_roll @ Rx @ R_yaw


def _apply_world_rotation_about_pivot_extrinsics(
    R: torch.Tensor,
    t: torch.Tensor,
    Q: torch.Tensor,
    pivot_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply world-space rotation Q around pivot_world to camera extrinsics (world->cam):
      X_cam = R @ X_world + t
    After rotating world points:
      X_world' = Q @ (X_world - p) + p
    equivalent new extrinsics:
      R' = R @ Q
      t' = t + R @ ((I - Q) @ p)
    """
    Q = Q.to(device=R.device, dtype=R.dtype)
    p = pivot_world.to(device=R.device, dtype=R.dtype).view(3)
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    R_new = R @ Q
    t_new = t + (R @ ((I - Q) @ p.view(3, 1))).view(3)
    return R_new, t_new


def save_rgb(path: str | Path, rgb_chw: torch.Tensor) -> None:
    """Save (3,H,W) RGB float [0,1] as png/jpg, preserving exact H/W."""
    img = (rgb_chw.detach().clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
    bgr = img[..., ::-1]  # BGR
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _clear_image_files(dir_path: Path) -> int:
    removed = 0
    if not dir_path.exists():
        return removed
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for old in dir_path.glob(ext):
            old.unlink()
            removed += 1
    return removed


def _pick_suffix_png(folder: Path, suffix_name: str) -> Path:
    cands = sorted(folder.glob(f"*_{suffix_name}.png"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No rendered file matched '*_{suffix_name}.png' in {folder}")
    return cands[-1]


def _move_or_replace(src: Path, dst: Path) -> None:
    if src == dst:
        return
    try:
        dst.unlink(missing_ok=True)
    except PermissionError:
        pass
    try:
        src.replace(dst)
    except PermissionError:
        import shutil

        shutil.copy2(src, dst)


LAYER_KEYS: tuple[str, str, str] = ("outer", "inner", "combined")


def _resolve_layer_keys(layer_keys_cfg) -> tuple[str, ...]:
    if layer_keys_cfg is None:
        return LAYER_KEYS
    if isinstance(layer_keys_cfg, str):
        raw = [x.strip().lower() for x in layer_keys_cfg.split(",")]
    elif isinstance(layer_keys_cfg, (list, tuple)):
        raw = [str(x).strip().lower() for x in layer_keys_cfg]
    else:
        raise ValueError("render.layer_keys must be null, a comma-separated string, or a list.")
    keys = tuple(x for x in raw if x in LAYER_KEYS)
    if len(keys) <= 0:
        raise ValueError(f"render.layer_keys resolved to empty set. Allowed: {list(LAYER_KEYS)}")
    return keys


def _make_layer_output_dirs(base_dir: Path, *, layer_keys: tuple[str, ...]) -> dict[str, Path]:
    return {
        layer_key: ensure_dir(base_dir / layer_key)
        for layer_key in layer_keys
    }


def _split_gaussians_by_shell_layer(g: Gaussian3D) -> dict[str, Gaussian3D | None]:
    if g.xyz.shape[0] == 0:
        return {"outer": None, "inner": None, "combined": None}
    if g.shell_layer is None:
        # Backward compatibility: no explicit shell labels in old checkpoints.
        return {"outer": None, "inner": g, "combined": g}
    layer = g.shell_layer.view(-1)
    outer_mask = layer > 0
    inner_mask = ~outer_mask
    g_outer = _select_gaussians(g, outer_mask) if bool(outer_mask.any()) else None
    g_inner = _select_gaussians(g, inner_mask) if bool(inner_mask.any()) else None
    return {"outer": g_outer, "inner": g_inner, "combined": g}


def _background_tensor(
    *,
    device: torch.device,
    H: int,
    W: int,
    bg: tuple[float, float, float],
    bg_gt_chw: torch.Tensor | None,
    use_gt_bg: bool,
) -> torch.Tensor:
    if bool(use_gt_bg) and bg_gt_chw is not None:
        return bg_gt_chw
    return torch.tensor(bg, dtype=torch.float32, device=device).view(3, 1, 1).expand(3, H, W)


def _render_layer_preview(
    *,
    g_layer: Gaussian3D | None,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: int,
    W: int,
    bg: tuple[float, float, float],
    bg_gt_chw: torch.Tensor | None,
    bg_gt_hwc: np.ndarray | None,
    use_gt_bg: bool,
) -> torch.Tensor:
    if g_layer is None or g_layer.xyz.shape[0] == 0:
        return _background_tensor(
            device=K.device,
            H=H,
            W=W,
            bg=bg,
            bg_gt_chw=bg_gt_chw,
            use_gt_bg=use_gt_bg,
        )
    return render_gaussians_3d(
        g=g_layer,
        K=K,
        R=R,
        t=t,
        H=H,
        W=W,
        bg=bg_gt_hwc if (bool(use_gt_bg) and bg_gt_hwc is not None) else bg,
    )


def _compute_vertex_normals_np(verts: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    return compute_area_weighted_vertex_normals_np(np.asarray(verts, dtype=np.float32), faces, eps=1e-8)


def _quat_from_normals_torch(normals: torch.Tensor) -> torch.Tensor:
    """
    Build quaternion that rotates local +Z axis to target normal.
    Input: normals (N,3), output: quat (N,4) in (w,x,y,z).
    """
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must be (N,3), got {normals.shape}")
    n = F.normalize(normals, dim=1, eps=1e-8)
    z = torch.zeros_like(n)
    z[:, 2] = 1.0
    dot = (z * n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(z, n, dim=1)
    s = torch.sqrt((1.0 + dot).clamp(min=1e-8) * 2.0)
    q = torch.zeros((n.shape[0], 4), dtype=n.dtype, device=n.device)
    q[:, 0] = 0.5 * s[:, 0]
    q[:, 1:] = cross / s.clamp(min=1e-8)

    # Handle opposite direction (+Z -> -Z).
    opp = (dot[:, 0] < -0.9999)
    if bool(opp.any()):
        q[opp, 0] = 0.0
        q[opp, 1] = 1.0
        q[opp, 2] = 0.0
        q[opp, 3] = 0.0

    q = F.normalize(q, dim=1, eps=1e-8)
    return q


def _render_points_debug(
    g: Gaussian3D,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: int,
    W: int,
    bg: tuple[float, float, float],
    point_radius: int = 2,
    point_alpha: float = 1.0,
    use_gaussian_color: bool = True,
    point_color: tuple[float, float, float] | None = None,
    surface_only: bool = True,
    skeleton_segments: list[tuple[np.ndarray, np.ndarray]] | None = None,
    near_skeleton_px: float = 0.0,
) -> torch.Tensor:
    """Project gaussian centers and draw them as circles for binding debugging."""
    # RGB canvas in [0,1]
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    canvas[..., 0] = float(bg[0])
    canvas[..., 1] = float(bg[1])
    canvas[..., 2] = float(bg[2])

    xyz = g.xyz.detach().cpu().numpy().astype(np.float32)   # (N,3)
    K_np = K.detach().cpu().numpy().astype(np.float32)
    R_np = R.detach().cpu().numpy().astype(np.float32)
    t_np = t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)

    # Keep the same convention as gsplat path: X_cam = R * X + t
    Xc = xyz @ R_np.T + t_np
    z = Xc[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()

    Xc = Xc[valid]
    z = z[valid]

    u = K_np[0, 0] * (Xc[:, 0] / z) + K_np[0, 2]
    v = K_np[1, 1] * (Xc[:, 1] / z) + K_np[1, 2]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()

    u = u[in_img]
    v = v[in_img]
    z = z[in_img]

    if use_gaussian_color:
        cols = g.rgb().detach().cpu().numpy().astype(np.float32)[valid][in_img]  # RGB
    else:
        if point_color is None:
            point_color = (1.0, 0.0, 0.0)
        cols = np.tile(np.asarray(point_color, dtype=np.float32).reshape(1, 3), (u.shape[0], 1))

    # Keep only points near 2D keypoint skeleton if requested.
    if skeleton_segments is not None and len(skeleton_segments) > 0 and float(near_skeleton_px) > 0.0:
        P = np.stack([u, v], axis=1).astype(np.float32)
        min_d2 = np.full((P.shape[0],), np.inf, dtype=np.float32)
        for A, B in skeleton_segments:
            AB = B - A
            denom = float(np.dot(AB, AB) + 1e-8)
            AP = P - A.reshape(1, 2)
            t_line = (AP @ AB.reshape(2, 1)).reshape(-1) / denom
            t_line = np.clip(t_line, 0.0, 1.0)
            Q = A.reshape(1, 2) + t_line.reshape(-1, 1) * AB.reshape(1, 2)
            d2 = np.sum((P - Q) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)
        keep = min_d2 <= (float(near_skeleton_px) ** 2)
        if np.any(keep):
            u = u[keep]
            v = v[keep]
            z = z[keep]
            cols = cols[keep]
        else:
            return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()

    # Keep only nearest point for each pixel to remove back-side clutter.
    if surface_only:
        xi = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        yi = np.clip(np.round(v).astype(np.int32), 0, H - 1)
        key = yi.astype(np.int64) * int(W) + xi.astype(np.int64)
        # sort by depth asc (near -> far), then keep first occurrence per pixel
        order_near = np.argsort(z)
        key_sorted = key[order_near]
        _, first_idx = np.unique(key_sorted, return_index=True)
        keep = order_near[first_idx]
        u = u[keep]
        v = v[keep]
        z = z[keep]
        cols = cols[keep]

    # Draw far-to-near for stable blending
    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    cols = cols[order]

    pr = max(1, int(point_radius))
    a = float(np.clip(point_alpha, 0.0, 1.0))
    for i in range(u.shape[0]):
        x = int(round(float(u[i])))
        y = int(round(float(v[i])))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        # cv2 wants BGR
        bgr = (float(cols[i, 2]), float(cols[i, 1]), float(cols[i, 0]))
        if a >= 0.999:
            cv2.circle(canvas, (x, y), pr, bgr, thickness=-1, lineType=cv2.LINE_AA)
        else:
            x0 = max(0, x - pr)
            x1 = min(W - 1, x + pr)
            y0 = max(0, y - pr)
            y1 = min(H - 1, y + pr)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = canvas[y0 : y1 + 1, x0 : x1 + 1]
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            mask = ((xx - x) ** 2 + (yy - y) ** 2) <= (pr * pr)
            if not np.any(mask):
                continue
            patch[mask] = patch[mask] * (1.0 - a) + np.asarray(bgr, dtype=np.float32) * a

    return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()


def _overlay_keypoints(
    rgb_chw: torch.Tensor,
    keypoints_json: Path,
    frame_name: str,
    conf_min: float = 0.2,
    radius: int = 3,
    draw_skeleton: bool = True,
    body_only: bool = True,
) -> torch.Tensor:
    import json as _json

    if not keypoints_json.exists():
        return rgb_chw

    with keypoints_json.open("r", encoding="utf-8") as f:
        kpts = _json.load(f)
    arr = kpts.get(frame_name, None)
    if arr is None:
        return rgb_chw

    img = (rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    bgr = img[..., ::-1].copy()

    H, W = bgr.shape[:2]
    rr = max(1, int(radius))
    for j, p in enumerate(arr):
        if body_only and j not in MP_BODY_DRAW_IDS:
            continue
        if float(p[2]) < float(conf_min):
            continue
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        cv2.circle(bgr, (x, y), rr, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

    out_rgb = torch.from_numpy(bgr[..., ::-1].copy()).float() / 255.0
    return out_rgb.permute(2, 0, 1).contiguous()


def _load_frame_keypoints(
    keypoints_json: Path,
    frame_name: str,
    conf_min: float,
) -> np.ndarray | None:
    import json as _json

    if not keypoints_json.exists():
        return None
    with keypoints_json.open("r", encoding="utf-8") as f:
        kpts = _json.load(f)
    arr = kpts.get(frame_name, None)
    if arr is None:
        return None
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim != 2 or arr_np.shape[1] < 3:
        return None
    arr_np[arr_np[:, 2] < float(conf_min), :2] = np.nan
    return arr_np


def _build_skeleton_segments_2d(
    keypoints_xyc: np.ndarray | None,
    W: int,
    H: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if keypoints_xyc is None:
        return []
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in MP_POSE_CONNECTIONS:
        if a >= keypoints_xyc.shape[0] or b >= keypoints_xyc.shape[0]:
            continue
        pa = keypoints_xyc[a]
        pb = keypoints_xyc[b]
        if not np.isfinite(pa[0]) or not np.isfinite(pa[1]):
            continue
        if not np.isfinite(pb[0]) or not np.isfinite(pb[1]):
            continue
        xa, ya = float(pa[0]), float(pa[1])
        xb, yb = float(pb[0]), float(pb[1])
        if xa < 0 or xa >= W or ya < 0 or ya >= H:
            continue
        if xb < 0 or xb >= W or yb < 0 or yb >= H:
            continue
        segs.append(
            (
                np.asarray([xa, ya], dtype=np.float32),
                np.asarray([xb, yb], dtype=np.float32),
            )
        )
    return segs


def _build_parent_connections(joint_parents: np.ndarray | None, joint_count: int) -> list[tuple[int, int]]:
    if joint_parents is None:
        return []
    p = np.asarray(joint_parents, dtype=np.int32).reshape(-1)
    n = int(min(int(joint_count), int(p.shape[0])))
    edges: list[tuple[int, int]] = []
    for j in range(n):
        par = int(p[j])
        if par < 0 or par >= n or par == j:
            continue
        edges.append((par, j))
    return edges


def _load_smpl_skeleton_map(
    npz_path: Path,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]], str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"SMPL skeleton sequence not found: {npz_path}")
    seq = np.load(npz_path, allow_pickle=True)
    if "frame_names" not in seq.files:
        raise RuntimeError(f"{npz_path} missing frame_names")

    has_joints = "joints_cam" in seq.files
    has_theta = "joints_cam_theta" in seq.files
    if not has_joints and not has_theta:
        raise RuntimeError(f"{npz_path} missing joints_cam_theta/joints_cam")

    joints_key = "joints_cam" if has_joints else "joints_cam_theta"
    frame_names = [str(x) for x in seq["frame_names"].tolist()]
    joints_cam = np.asarray(seq[joints_key], dtype=np.float32)
    if joints_cam.ndim != 3 or joints_cam.shape[0] != len(frame_names) or joints_cam.shape[2] != 3:
        raise RuntimeError(f"Bad joints shape in {npz_path}: {joints_cam.shape}")

    joint_parents = np.asarray(seq["joint_parents"], dtype=np.int32) if "joint_parents" in seq.files else None
    connections = _build_parent_connections(joint_parents=joint_parents, joint_count=int(joints_cam.shape[1]))
    if len(connections) == 0:
        connections = list(SMPL_SKELETON_CONNECTIONS)
    return {fn: joints_cam[i] for i, fn in enumerate(frame_names)}, connections, joints_key


def _normalize_smplx_body_joints(joints_cam: np.ndarray | None) -> np.ndarray | None:
    if joints_cam is None:
        return None
    arr = np.asarray(joints_cam, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    # Use one fixed SMPL-X visualization subset so fg_only/deform always show same joint count.
    body_joint_count = min(int(arr.shape[0]), int(SMPLX_OVERLAY_JOINT_COUNT))
    if body_joint_count <= 0:
        return None
    return arr[:body_joint_count].astype(np.float32, copy=True)


def _overlay_smpl_skeleton(
    rgb_chw: torch.Tensor,
    joints_cam: np.ndarray | None,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    connections: list[tuple[int, int]] | None = None,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 3,
) -> torch.Tensor:
    joints_use = _normalize_smplx_body_joints(joints_cam)
    if joints_use is None:
        return rgb_chw

    img = (rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    bgr = img[..., ::-1].copy()
    H, W = bgr.shape[:2]

    K_np = K.detach().cpu().numpy().astype(np.float32)
    R_np = R.detach().cpu().numpy().astype(np.float32)
    t_np = t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
    Xc = joints_use.astype(np.float32) @ R_np.T + t_np
    z = Xc[:, 2]
    valid = z > 1e-6
    u = np.zeros((joints_use.shape[0],), dtype=np.float32)
    v = np.zeros((joints_use.shape[0],), dtype=np.float32)
    if np.any(valid):
        Xcv = Xc[valid]
        zv = z[valid]
        u[valid] = K_np[0, 0] * (Xcv[:, 0] / zv) + K_np[0, 2]
        v[valid] = K_np[1, 1] * (Xcv[:, 1] / zv) + K_np[1, 2]

    rr = max(1, int(radius))
    for j in range(len(u)):
        if not valid[j]:
            continue
        x = int(round(float(u[j])))
        y = int(round(float(v[j])))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        cv2.circle(bgr, (x, y), rr, color_bgr, thickness=-1, lineType=cv2.LINE_AA)

    out_rgb = torch.from_numpy(bgr[..., ::-1].copy()).float() / 255.0
    return out_rgb.permute(2, 0, 1).contiguous()

def _fit_subject_camera_numpy(
    verts: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    fill: float = 0.82,
    iters: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if verts is None or verts.size == 0:
        return K, t

    K_new = np.asarray(K, dtype=np.float32).copy()
    t_new = np.asarray(t, dtype=np.float32).reshape(3).copy()
    fill = float(np.clip(fill, 0.2, 0.98))
    iters = max(1, int(iters))

    for _ in range(iters):
        u, v, valid = _project_points_numpy(verts, K_new, R, t_new)
        if int(valid.sum()) < 32:
            break
        uu = u[valid]
        vv = v[valid]
        x0, x1 = float(uu.min()), float(uu.max())
        y0, y1 = float(vv.min()), float(vv.max())
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)

        cx_box = 0.5 * (x0 + x1)
        cy_box = 0.5 * (y0 + y1)
        cx_tgt = 0.5 * (W - 1)
        cy_tgt = 0.5 * (H - 1)

        Xc = verts @ R.T + t_new.reshape(1, 3)
        z = Xc[:, 2]
        z_valid = z[valid]
        if z_valid.size > 0:
            zc = float(np.clip(np.median(z_valid), 0.1, 100.0))
            fx = float(max(1e-6, K_new[0, 0]))
            fy = float(max(1e-6, K_new[1, 1]))
            du = float(cx_tgt - cx_box)
            dv = float(cy_tgt - cy_box)
            t_new[0] += du * zc / fx
            t_new[1] += dv * zc / fy

        u2, v2, valid2 = _project_points_numpy(verts, K_new, R, t_new)
        if int(valid2.sum()) < 32:
            continue
        uu2 = u2[valid2]
        vv2 = v2[valid2]
        bw2 = max(1.0, float(uu2.max() - uu2.min()))
        bh2 = max(1.0, float(vv2.max() - vv2.min()))

        sx = (fill * float(W)) / bw2
        sy = (fill * float(H)) / bh2
        s_fit = min(sx, sy)
        # only zoom out to avoid clipping; do not zoom in automatically
        if s_fit < 1.0:
            K_new[0, 0] *= float(s_fit)
            K_new[1, 1] *= float(s_fit)

    return K_new, t_new


def _load_person_mask_map(npz_path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not npz_path.exists():
        return out
    seq = np.load(str(npz_path), allow_pickle=True)
    if "frame_names" not in seq.files or "masks" not in seq.files:
        return out
    names = [str(x) for x in seq["frame_names"].tolist()]
    masks = np.asarray(seq["masks"], dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[0] != len(names):
        return out
    for i, n in enumerate(names):
        out[n] = ((masks[i] > 0).astype(np.uint8) * 255)
    return out


def _scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale camera intrinsics for image size change (or FOV scaling)."""
    if scale == 1.0:
        return K
    Ks = K.clone()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def _resize_chw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Resize (C,H,W) tensor to (C,H,W) with safe down/up-sampling."""
    if x.shape[1] == H and x.shape[2] == W:
        return x
    mode = "area" if (x.shape[1] > H or x.shape[2] > W) else "bilinear"
    if mode == "area":
        return F.interpolate(x.unsqueeze(0), size=(H, W), mode=mode).squeeze(0)
    return F.interpolate(x.unsqueeze(0), size=(H, W), mode=mode, align_corners=False).squeeze(0)


def _composite_premultiplied(
    fg_rgb: torch.Tensor,
    fg_alpha: torch.Tensor,
    bg_rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Composite premultiplied foreground over background:
      C = C_fg + (1 - A_fg) * C_bg
    """
    return fg_rgb + (1.0 - fg_alpha) * bg_rgb


def _alpha_to_mask(
    alpha: torch.Tensor,
    threshold: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    a = alpha.clamp(0.0, 1.0)
    if float(gamma) != 1.0:
        a = torch.pow(a, float(gamma))
    if float(threshold) > 0.0:
        a = (a > float(threshold)).to(dtype=a.dtype)
    return a


def _select_gaussians(g: Gaussian3D, mask: torch.Tensor) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz[mask],
        log_scales=g.log_scales[mask],
        quat=g.quat[mask],
        opacity_logit=g.opacity_logit[mask],
        rgb_logit=g.rgb_logit[mask],
        bind_idx=g.bind_idx[mask] if g.bind_idx is not None else None,
        bind_offset=g.bind_offset[mask] if g.bind_offset is not None else None,
        bind_face_idx=g.bind_face_idx[mask] if g.bind_face_idx is not None else None,
        bind_bary=g.bind_bary[mask] if g.bind_bary is not None else None,
        bind_hybrid_w=g.bind_hybrid_w[mask] if g.bind_hybrid_w is not None else None,
        bind_face_offset_local=g.bind_face_offset_local[mask] if g.bind_face_offset_local is not None else None,
        shell_layer=g.shell_layer[mask] if g.shell_layer is not None else None,
        bind_mix_weight=g.bind_mix_weight[mask] if g.bind_mix_weight is not None else None,
        prune_state=g.prune_state[mask] if g.prune_state is not None else None,
        inner_bind_mode=g.inner_bind_mode[mask] if g.inner_bind_mode is not None else None,
        quality_fail_counter=g.quality_fail_counter[mask] if g.quality_fail_counter is not None else None,
        color_conf_ema=g.color_conf_ema[mask] if g.color_conf_ema is not None else None,
        geom_conf_ema=g.geom_conf_ema[mask] if g.geom_conf_ema is not None else None,
    )


def _clone_gaussians(g: Gaussian3D) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz.clone(),
        log_scales=g.log_scales.clone(),
        quat=g.quat.clone(),
        opacity_logit=g.opacity_logit.clone(),
        rgb_logit=g.rgb_logit.clone(),
        bind_idx=g.bind_idx.clone() if g.bind_idx is not None else None,
        bind_offset=g.bind_offset.clone() if g.bind_offset is not None else None,
        bind_face_idx=g.bind_face_idx.clone() if g.bind_face_idx is not None else None,
        bind_bary=g.bind_bary.clone() if g.bind_bary is not None else None,
        bind_hybrid_w=g.bind_hybrid_w.clone() if g.bind_hybrid_w is not None else None,
        bind_face_offset_local=g.bind_face_offset_local.clone() if g.bind_face_offset_local is not None else None,
        shell_layer=g.shell_layer.clone() if g.shell_layer is not None else None,
        bind_mix_weight=g.bind_mix_weight.clone() if g.bind_mix_weight is not None else None,
        prune_state=g.prune_state.clone() if g.prune_state is not None else None,
        inner_bind_mode=g.inner_bind_mode.clone() if g.inner_bind_mode is not None else None,
        quality_fail_counter=g.quality_fail_counter.clone() if g.quality_fail_counter is not None else None,
        color_conf_ema=g.color_conf_ema.clone() if g.color_conf_ema is not None else None,
        geom_conf_ema=g.geom_conf_ema.clone() if g.geom_conf_ema is not None else None,
    )


def _apply_vis_override(
    g: Gaussian3D,
    opacity_logit: float,
    scale_mult: float,
    rgb_override: tuple[float, float, float] | None,
) -> Gaussian3D:
    gv = _clone_gaussians(g)
    gv.opacity_logit = torch.full_like(gv.opacity_logit, float(opacity_logit))
    if float(scale_mult) > 0.0 and float(scale_mult) != 1.0:
        gv.log_scales = gv.log_scales + float(math.log(scale_mult))
    gv.log_scales = gv.log_scales.clamp(min=-8.0, max=-2.0)

    if rgb_override is not None:
        c = torch.tensor(
            [rgb_override[0], rgb_override[1], rgb_override[2]],
            dtype=gv.rgb_logit.dtype,
            device=gv.rgb_logit.device,
        ).clamp(1e-3, 1.0 - 1e-3)
        c_logit = torch.log(c / (1.0 - c)).view(1, 3)
        gv.rgb_logit = c_logit.expand_as(gv.rgb_logit).contiguous()
    return gv


def _load_keypoints_boxes(
    keypoints_json: Path,
    frames: list[str],
    H: int,
    W: int,
    conf_min: float,
    pad: float,
) -> list[tuple[float, float, float, float] | None]:
    if not keypoints_json.exists():
        raise FileNotFoundError(f"keypoints_json not found: {keypoints_json}")
    import json as _json

    with keypoints_json.open("r", encoding="utf-8") as f:
        kpts = _json.load(f)
    boxes = []
    for fn in frames:
        arr = kpts.get(fn, None)
        if arr is None:
            boxes.append(None)
            continue
        xs = [p[0] for p in arr if p[2] >= conf_min]
        ys = [p[1] for p in arr if p[2] >= conf_min]
        if not xs or not ys:
            boxes.append(None)
            continue
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = (x1 - x0)
        bh = (y1 - y0)
        x0 = max(0.0, x0 - pad * bw)
        x1 = min(W - 1.0, x1 + pad * bw)
        y0 = max(0.0, y0 - pad * bh)
        y1 = min(H - 1.0, y1 + pad * bh)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _resolve_stage2_ckpt(stage2_dir: Path) -> Path:
    """
    Find a checkpoint file inside stage2_dir.
    Tries common filenames first, then falls back to any .pt found.
    """
    common = [
        stage2_dir / "gaussians.pt",
        stage2_dir / "gaussians3d.pt",
        stage2_dir / "ckpt.pt",
    ]
    for p in common:
        if p.exists():
            return p

    pts = sorted(stage2_dir.glob("*.pt"))
    if pts:
        return pts[-1]

    ckpt_dir = stage2_dir / "checkpoints"
    if ckpt_dir.exists():
        pts2 = sorted(ckpt_dir.glob("*.pt"))
        if pts2:
            return pts2[-1]

    raise FileNotFoundError(f"No checkpoint (*.pt) found under: {stage2_dir}")


def _resolve_pick_index(mode: str, n: int, index: int = -1) -> int:
    if n <= 0:
        return 0
    m = str(mode or "last").strip().lower()
    if m == "first":
        return 0
    if m == "index":
        i = int(index)
        if i < 0:
            i = n + i
        return int(np.clip(i, 0, n - 1))
    return n - 1


def _resolve_deform_mesh_cfg(render_cfg: dict, ckpt: dict, logger) -> str:
    txt = str(render_cfg.get("deform_mesh_obj", "")).strip()
    if txt:
        return txt
    for key in ("bind_deform_obj", "bind_canonical_obj"):
        v = str(ckpt.get(key, "")).strip()
        if v:
            return v
    return ""


def _find_deform_mesh_fallback(stage1_dir: Path) -> Path | None:
    for pat in ("*_deform.obj", "*_canonical.obj"):
        cands = sorted(stage1_dir.glob(pat))
        if cands:
            return cands[-1]
    return None

def main():
    parser = parse_args_base("Render images from Stage2 gaussian checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("render", str(log_dir / "render.log"))

    render_cfg = cfg["render"]
    stage2_dir = project_root / Path(render_cfg.get("stage2_dir", "outputs/stage2"))
    stage2_cfg = cfg.get("stage2", {}) if isinstance(cfg.get("stage2", {}), dict) else {}
    stage1_dir_cfg = render_cfg.get("stage1_dir", stage2_cfg.get("stage1_dir", "outputs/stage1"))
    stage1_dir = project_root / Path(stage1_dir_cfg)
    out_dir = ensure_dir(project_root / Path(render_cfg.get("out_dir", "outputs/renders")))
    save_main_image = bool(render_cfg.get("save_main_image", False))
    save_real_image = bool(render_cfg.get("save_real_image", True))
    save_fg_nobg_skeleton = bool(render_cfg.get("save_fg_nobg_skeleton", True))
    render_deform_image = bool(render_cfg.get("deform_render_image", True))

    render_mode = str(render_cfg.get("mode", "")).strip().lower()
    if render_mode in ("fg", "fg_only"):
        save_main_image = False
        save_real_image = False
        save_fg_nobg_skeleton = True
        render_deform_image = False
    elif render_mode == "deform":
        save_main_image = False
        save_real_image = False
        save_fg_nobg_skeleton = False
        render_deform_image = True

    real_out_subdir = str(render_cfg.get("real_out_subdir", "real")).strip()
    fg_nobg_out_subdir = str(render_cfg.get("fg_nobg_out_subdir", "fg_only")).strip()
    deform_out_subdir = str(render_cfg.get("deform_out_subdir", "deform")).strip()
    layer_split_output = bool(render_cfg.get("layer_split_output", True))
    active_layer_keys = _resolve_layer_keys(render_cfg.get("layer_keys", None)) if layer_split_output else tuple()

    # In single-mode rendering, default to writing directly into out_dir (no nested subfolders).
    if render_mode in ("fg", "fg_only") and fg_nobg_out_subdir == "fg_only":
        fg_nobg_out_subdir = "."

    real_out_dir = ensure_dir(out_dir / Path(real_out_subdir)) if save_real_image else out_dir
    fg_nobg_out_dir = ensure_dir(out_dir / Path(fg_nobg_out_subdir)) if save_fg_nobg_skeleton else out_dir
    real_layer_dirs = _make_layer_output_dirs(real_out_dir, layer_keys=active_layer_keys) if (save_real_image and layer_split_output) else {}
    fg_layer_dirs = _make_layer_output_dirs(fg_nobg_out_dir, layer_keys=active_layer_keys) if (save_fg_nobg_skeleton and layer_split_output) else {}

    deform_out_cfg = render_cfg.get("deform_out_dir", None)
    if render_deform_image:
        if deform_out_cfg is None or not str(deform_out_cfg).strip():
            deform_out_dir = ensure_dir(out_dir / Path(deform_out_subdir))
        else:
            deform_out_dir = Path(resolve_path(str(deform_out_cfg)))
            if not deform_out_dir.is_absolute():
                deform_out_dir = project_root / deform_out_dir
            deform_out_dir = ensure_dir(deform_out_dir)
    else:
        deform_out_dir = out_dir
    deform_layer_dirs = _make_layer_output_dirs(deform_out_dir, layer_keys=active_layer_keys) if (render_deform_image and layer_split_output) else {}
    view_yaw_deg = float(render_cfg.get("view_yaw_deg", 0.0))
    view_pitch_deg = float(render_cfg.get("view_pitch_deg", 0.0))
    view_roll_deg = float(render_cfg.get("view_roll_deg", 0.0))
    view_axis_mode = str(render_cfg.get("view_axis_mode", "legacy")).strip().lower()
    if view_axis_mode not in ("legacy", "z_up"):
        raise ValueError(f"render.view_axis_mode must be legacy|z_up. Got: {view_axis_mode}")
    file_suffix = str(render_cfg.get("file_suffix", "")).strip()
    if file_suffix and (not file_suffix.startswith("_")):
        file_suffix = f"_{file_suffix}"
    six_views_enable = bool(render_cfg.get("six_views_enable", True))
    six_views_internal = bool(render_cfg.get("six_views_internal", False))
    six_views_include_top_bottom = bool(render_cfg.get("six_views_include_top_bottom", True))

    if six_views_enable and (not six_views_internal):
        view_specs: list[tuple[str, float, float]] = [
            ("front", 0.0, 0.0),
            ("back", 180.0, 0.0),
            ("left", 90.0, 0.0),
            ("right", -90.0, 0.0),
        ]
        if six_views_include_top_bottom:
            view_specs.extend([("top", 0.0, 90.0), ("bottom", 0.0, -90.0)])
        suffix_base = file_suffix[1:] if file_suffix.startswith("_") else file_suffix
        logger.info(
            "Six-view mode enabled (dual-layer bound-only): %s",
            [x[0] for x in view_specs],
        )
        for vi, (view_name, yaw_deg, pitch_deg) in enumerate(view_specs):
            emit_real = bool(save_real_image) and (view_name == "front")
            emit_deform = bool(render_deform_image)
            view_suffix = view_name if (not suffix_base) else f"{view_name}_{suffix_base}"
            extra_overrides = [
                "render.six_views_enable=false",
                "render.six_views_internal=true",
                f"render.view_yaw_deg={float(yaw_deg)}",
                f"render.view_pitch_deg={float(pitch_deg)}",
                f"render.view_roll_deg={float(view_roll_deg)}",
                f"render.file_suffix={view_suffix}",
                "render.render_bound_only=true",
                "render.fg_only_render_bound_only=true",
                "render.deform_render_bound_only=true",
                "render.deform_smplx_use_standard_pose=true",
                "render.overlay_smpl_skeleton=false",
                "render.debug_overlay_keypoints=false",
                "render.debug_bind_to_google_skeleton=false",
                f"render.save_real_image={'true' if emit_real else 'false'}",
                f"render.use_gt_background={'true' if emit_real else 'false'}",
            ]
            if emit_real:
                # Real/front should stay in the exact GT camera for best alignment.
                extra_overrides += [
                    "render.center_subject=false",
                    "render.subject_fit_enabled=false",
                    "render.k_scale=1.0",
                ]
            if vi > 0:
                extra_overrides.append("render.clear_out_dir_pngs=false")
            merged_overrides = list(args.override or []) + extra_overrides
            cmd = [sys.executable, "-m", "avatar_gs.cli.render", "--config", str(args.config)]
            if len(merged_overrides) > 0:
                cmd += ["--override", *merged_overrides]
            logger.info("Six-view render [%d/%d]: %s", vi + 1, len(view_specs), view_name)
            subprocess.run(cmd, check=True)
            if emit_deform and ((not layer_split_output) or ("combined" in deform_layer_dirs)):
                deform_combined_dir = deform_layer_dirs.get("combined", deform_out_dir)
                src = _pick_suffix_png(deform_combined_dir, view_suffix)
                dst = deform_combined_dir / f"{view_name}.png"
                _move_or_replace(src, dst)
            if emit_real and ((not layer_split_output) or ("combined" in real_layer_dirs)):
                real_combined_dir = real_layer_dirs.get("combined", real_out_dir)
                src = _pick_suffix_png(real_combined_dir, view_suffix)
                dst = real_combined_dir / "front.png"
                _move_or_replace(src, dst)
        logger.info("Done.")
        return

    if bool(render_cfg.get("clear_out_dir_pngs", False)):
        removed = 0
        removed += _clear_image_files(out_dir)
        removed += _clear_image_files(real_out_dir)
        removed += _clear_image_files(fg_nobg_out_dir)
        removed += _clear_image_files(deform_out_dir)
        for layer_dir in list(real_layer_dirs.values()) + list(fg_layer_dirs.values()) + list(deform_layer_dirs.values()):
            removed += _clear_image_files(layer_dir)
        if removed > 0:
            logger.info(f"Cleared {removed} existing render image(s) in output folders")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoint path: optional explicit, otherwise auto-resolve
    ckpt_path = render_cfg.get("ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).strip():
        ckpt_path = Path(resolve_path(str(ckpt_path)))
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
    else:
        ckpt_path = _resolve_stage2_ckpt(stage2_dir)

    logger.info(f"Stage2 dir: {stage2_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Out dir: {out_dir}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Load gaussians
    #g = Gaussian3D()
    #g.load_state_dict(ckpt["gaussians"])
    #g = g.to(device)

    # Load gaussians (Gaussian3D requires tensors in __init__)
    gsd = ckpt["gaussians"]  # this is g.state_dict() saved in stage2

    required = ["xyz", "log_scales", "quat", "opacity_logit", "rgb_logit"]
    missing = [k for k in required if k not in gsd]
    if missing:
      raise KeyError(f"Checkpoint gaussians missing keys: {missing}. Available keys: {list(gsd.keys())}")

    g = Gaussian3D(
      xyz=gsd["xyz"].to(device),
      log_scales=gsd["log_scales"].to(device),
      quat=gsd["quat"].to(device),
      opacity_logit=gsd["opacity_logit"].to(device),
      rgb_logit=gsd["rgb_logit"].to(device),
      bind_idx=gsd["bind_idx"].to(device) if ("bind_idx" in gsd and gsd["bind_idx"] is not None) else None,
      bind_offset=gsd["bind_offset"].to(device) if ("bind_offset" in gsd and gsd["bind_offset"] is not None) else None,
      bind_face_idx=gsd["bind_face_idx"].to(device) if ("bind_face_idx" in gsd and gsd["bind_face_idx"] is not None) else None,
      bind_bary=gsd["bind_bary"].to(device) if ("bind_bary" in gsd and gsd["bind_bary"] is not None) else None,
      bind_hybrid_w=gsd["bind_hybrid_w"].to(device) if ("bind_hybrid_w" in gsd and gsd["bind_hybrid_w"] is not None) else None,
      bind_face_offset_local=gsd["bind_face_offset_local"].to(device) if ("bind_face_offset_local" in gsd and gsd["bind_face_offset_local"] is not None) else None,
      shell_layer=gsd["shell_layer"].to(device) if ("shell_layer" in gsd and gsd["shell_layer"] is not None) else None,
      bind_mix_weight=gsd["bind_mix_weight"].to(device) if ("bind_mix_weight" in gsd and gsd["bind_mix_weight"] is not None) else None,
      prune_state=gsd["prune_state"].to(device) if ("prune_state" in gsd and gsd["prune_state"] is not None) else None,
      inner_bind_mode=gsd["inner_bind_mode"].to(device) if ("inner_bind_mode" in gsd and gsd["inner_bind_mode"] is not None) else None,
      quality_fail_counter=gsd["quality_fail_counter"].to(device) if ("quality_fail_counter" in gsd and gsd["quality_fail_counter"] is not None) else None,
      color_conf_ema=gsd["color_conf_ema"].to(device) if ("color_conf_ema" in gsd and gsd["color_conf_ema"] is not None) else None,
      geom_conf_ema=gsd["geom_conf_ema"].to(device) if ("geom_conf_ema" in gsd and gsd["geom_conf_ema"] is not None) else None,
    )

    # Intrinsics
    K = torch.as_tensor(ckpt["K"], dtype=torch.float32, device=device).view(3, 3)
    K_base_np = K.detach().cpu().numpy().astype(np.float32)

    # Cameras and frames
    frames = ckpt.get("frames", None)
    if frames is None or len(frames) == 0:
        raise KeyError("Checkpoint missing 'frames'. Cannot render without frame list.")
    frames = [str(x) for x in frames]
    total_frames = len(frames)

    cam_w = ckpt.get("cam_w", None)
    cam_t = ckpt.get("cam_t", None)
    cam_w = torch.as_tensor(cam_w, dtype=torch.float32, device=device) if cam_w is not None else None
    cam_t = torch.as_tensor(cam_t, dtype=torch.float32, device=device) if cam_t is not None else None

    # frames_dir: optional in config, otherwise default to data/frames/input
    frames_dir = render_cfg.get("frames_dir", None)
    if frames_dir is not None and str(frames_dir).strip():
        frames_dir = Path(resolve_path(str(frames_dir)))
        if not frames_dir.is_absolute():
            frames_dir = project_root / frames_dir
    else:
        frames_dir = project_root / Path("data/frames/input")

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    # Frame selection
    frame_select = str(render_cfg.get("frame_select", "first")).lower()
    selected_idx = list(range(total_frames))
    if frame_select == "last":
        selected_idx = [total_frames - 1]
    elif frame_select == "indices":
        indices_cfg = render_cfg.get("frame_indices", [])
        if isinstance(indices_cfg, (int, np.integer, float, np.floating, str)):
            indices_cfg = [indices_cfg]
        if not isinstance(indices_cfg, (list, tuple)) or len(indices_cfg) == 0:
            raise ValueError("render.frame_select=indices but render.frame_indices is empty.")
        selected_idx = []
        for idx in indices_cfg:
            i = int(idx)
            if i < 0:
                i = total_frames + i
            if i < 0 or i >= total_frames:
                raise IndexError(f"render.frame_indices has out-of-range index: {idx}")
            selected_idx.append(i)
    elif frame_select != "first":
        raise ValueError(f"Unsupported render.frame_select: {frame_select}. Use first|last|indices")

    # max_frames still applies after frame_select
    max_frames = int(render_cfg.get("max_frames", -1))
    if max_frames > 0:
        selected_idx = selected_idx[:max_frames]
    if len(selected_idx) == 0:
        raise RuntimeError("No frames selected for rendering.")

    frames = [frames[i] for i in selected_idx]
    if cam_w is not None:
        cam_w = cam_w[selected_idx]
    if cam_t is not None:
        cam_t = cam_t[selected_idx]
    to_render = len(frames)
    frame_K_nps = None
    ckpt_frame_Ks = ckpt.get("frame_Ks", None)
    if ckpt_frame_Ks is not None:
        ckpt_frame_Ks_np = np.asarray(ckpt_frame_Ks, dtype=np.float32).reshape(-1, 3, 3)
        if int(ckpt_frame_Ks_np.shape[0]) == int(total_frames):
            frame_K_nps = [np.asarray(ckpt_frame_Ks_np[i], dtype=np.float32) for i in selected_idx]
            logger.info(
                "Loaded frame-wise intrinsics for render from checkpoint: matched %d/%d frame(s)",
                int(len(frame_K_nps)),
                int(len(frames)),
            )
    if frame_K_nps is None:
        frame_K_nps = _load_stage1_frame_K_subset(
            stage1_dir=stage1_dir,
            frames=[str(x) for x in frames],
            logger=logger,
        )
    frame_K_tensors = None if frame_K_nps is None else [
        torch.from_numpy(np.asarray(K_np_i, dtype=np.float32)).to(device=device, dtype=torch.float32)
        for K_np_i in frame_K_nps
    ]
    prepared_supervision_cache_cfg = render_cfg.get(
        "prepared_supervision_cache_path",
        stage2_cfg.get("prepared_supervision_cache_path", str(stage2_dir / "supervision_cache.npz")),
    )
    prepared_supervision_cache_path = Path(resolve_path(str(prepared_supervision_cache_cfg)))
    if not prepared_supervision_cache_path.is_absolute():
        prepared_supervision_cache_path = project_root / prepared_supervision_cache_path

    # Background (default white)
    bg = render_cfg.get("bg", [1.0, 1.0, 1.0])
    if isinstance(bg, (list, tuple)) and len(bg) == 3:
        bg = (float(bg[0]), float(bg[1]), float(bg[2]))
    else:
        bg = (1.0, 1.0, 1.0)

    logger.info(f"Frames dir: {frames_dir}")
    logger.info(f"Frame select: {frame_select}; selected={to_render}/{total_frames}; bg={bg}")
    if to_render <= 5:
        logger.info(f"Selected frame names: {frames}")
    else:
        logger.info(f"Selected frame range: first={frames[0]}, last={frames[-1]}")
    logger.info(
        f"Render outputs: main={save_main_image}, real={save_real_image}, "
        f"fg_only={save_fg_nobg_skeleton}, deform={render_deform_image}"
    )

    fg_scale = float(render_cfg.get("fg_scale", 1.0))
    bg_scale = float(render_cfg.get("bg_scale", 1.0))
    k_scale = float(render_cfg.get("k_scale", 1.0))
    split_by_bind = bool(render_cfg.get("split_by_bind", False))
    render_bound_only = bool(render_cfg.get("render_bound_only", True))
    fg_only_render_bound_only = bool(render_cfg.get("fg_only_render_bound_only", True))
    deform_render_bound_only = bool(render_cfg.get("deform_render_bound_only", True))
    bound_alpha_threshold = float(render_cfg.get("bound_alpha_threshold", 0.0))
    bound_opacity_min = float(render_cfg.get("bound_opacity_min", 0.0))
    fg_mask_mode = str(render_cfg.get("fg_mask_mode", "none")).lower()
    fg_mask_dilate_px = int(render_cfg.get("fg_mask_dilate_px", 0))
    fg_mask_erode_px = int(render_cfg.get("fg_mask_erode_px", 0))
    visualize_bound_gaussians = bool(render_cfg.get("visualize_bound_gaussians", False))
    vis_opacity_logit = float(render_cfg.get("vis_opacity_logit", 6.0))
    vis_scale_mult = float(render_cfg.get("vis_scale_mult", 4.0))
    debug_point_render = bool(render_cfg.get("debug_point_render", False))
    debug_point_radius = int(render_cfg.get("debug_point_radius", 2))
    debug_point_alpha = float(render_cfg.get("debug_point_alpha", 1.0))
    debug_use_gaussian_color = bool(render_cfg.get("debug_use_gaussian_color", True))
    debug_surface_only = bool(render_cfg.get("debug_surface_only", True))
    debug_bind_to_google_skeleton = bool(render_cfg.get("debug_bind_to_google_skeleton", False))
    debug_skeleton_band_px = float(render_cfg.get("debug_skeleton_band_px", 8.0))
    debug_point_color_raw = render_cfg.get("debug_point_color", None)
    debug_point_color = None
    if isinstance(debug_point_color_raw, (list, tuple)) and len(debug_point_color_raw) == 3:
        debug_point_color = (
            float(debug_point_color_raw[0]),
            float(debug_point_color_raw[1]),
            float(debug_point_color_raw[2]),
        )
    debug_overlay_keypoints = bool(render_cfg.get("debug_overlay_keypoints", False))
    debug_draw_skeleton = bool(render_cfg.get("debug_draw_skeleton", True))
    debug_overlay_body_only = bool(render_cfg.get("debug_overlay_body_only", True))
    overlay_smpl_skeleton = bool(
        render_cfg.get("overlay_smpl_skeleton", render_cfg.get("debug_overlay_smpl_skeleton", False))
    )
    smpl_skeleton_source = str(render_cfg.get("smpl_skeleton_source", "joints")).strip().lower()
    if smpl_skeleton_source == "theta":
        logger.warning(
            "render.smpl_skeleton_source=theta is deprecated for overlay consistency; "
            "forcing joints_cam (SMPL-X unified joints)."
        )
        smpl_skeleton_source = "joints"
    if smpl_skeleton_source != "joints":
        raise ValueError(
            f"render.smpl_skeleton_source must be: joints. Got: {smpl_skeleton_source}"
        )
    smpl_skeleton_npz_cfg = render_cfg.get("smpl_skeleton_npz", None)
    smpl_joints_npz_cfg = render_cfg.get("smpl_joints_npz", render_cfg.get("debug_smpl_joints_npz", None))
    smpl_line_thickness = int(
        render_cfg.get("smpl_skeleton_line_thickness", render_cfg.get("debug_smpl_line_thickness", 2))
    )
    smpl_joint_radius = int(
        render_cfg.get("smpl_skeleton_joint_radius", render_cfg.get("debug_smpl_joint_radius", 3))
    )
    smpl_color_raw = render_cfg.get("smpl_skeleton_color_bgr", render_cfg.get("debug_smpl_color_bgr", [0, 255, 0]))
    if not (isinstance(smpl_color_raw, (list, tuple)) and len(smpl_color_raw) == 3):
        smpl_color_raw = [0, 255, 0]
    smpl_color_bgr = (
        int(smpl_color_raw[0]),
        int(smpl_color_raw[1]),
        int(smpl_color_raw[2]),
    )
    debug_keypoints_json = render_cfg.get("debug_keypoints_json", None)
    debug_kpt_conf_min = float(render_cfg.get("debug_kpt_conf_min", 0.2))
    debug_kpt_radius = int(render_cfg.get("debug_kpt_radius", 3))
    debug_need_kpts = bool(debug_bind_to_google_skeleton or debug_overlay_keypoints)
    center_subject = bool(render_cfg.get("center_subject", True))
    center_subject_use_mesh = bool(render_cfg.get("center_subject_use_mesh", True))
    center_subject_depth_min = float(render_cfg.get("center_subject_depth_min", 0.1))
    center_subject_joint_idx = int(render_cfg.get("center_subject_joint_idx", 0))
    subject_fit_enabled = bool(render_cfg.get("subject_fit_enabled", True))
    subject_fit_fill = float(render_cfg.get("subject_fit_fill", 0.82))
    subject_fit_iters = int(render_cfg.get("subject_fit_iters", 2))
    real_align_to_person_mask = bool(render_cfg.get("real_align_to_person_mask", True))
    real_align_alpha_threshold = float(render_cfg.get("real_align_alpha_threshold", 0.05))
    real_align_scale_min = float(render_cfg.get("real_align_scale_min", 0.4))
    real_align_scale_max = float(render_cfg.get("real_align_scale_max", 2.5))
    person_masks_npz_cfg = render_cfg.get("person_masks_npz", "outputs/stage1/person_masks.npz")
    fg_only_alpha_threshold = float(render_cfg.get("fg_only_alpha_threshold", 0.0))
    fg_only_alpha_gamma = float(render_cfg.get("fg_only_alpha_gamma", 1.0))
    deform_alpha_threshold = float(render_cfg.get("deform_alpha_threshold", fg_only_alpha_threshold))
    deform_alpha_gamma = float(render_cfg.get("deform_alpha_gamma", fg_only_alpha_gamma))
    fg_only_alpha_gain_max = float(render_cfg.get("fg_only_alpha_gain_max", 1.0))
    deform_alpha_gain_max = float(render_cfg.get("deform_alpha_gain_max", 1.0))
    deform_source_raw = str(render_cfg.get("deform_source", "smplx_params")).strip().lower()
    deform_smplx_params_npz_cfg = str(render_cfg.get("deform_smplx_params_npz", "")).strip()
    deform_smplx_frame_index = int(render_cfg.get("deform_smplx_frame_index", -1))
    deform_smplx_frame_name = str(render_cfg.get("deform_smplx_frame_name", "")).strip()
    deform_smplx_model_type = str(render_cfg.get("deform_smplx_model_type", "smplx")).strip().lower()
    deform_smplx_gender = str(render_cfg.get("deform_smplx_gender", "neutral")).strip().lower()
    deform_smplx_model_root = str(render_cfg.get("deform_smplx_model_root", "")).strip()
    deform_smplx_use_standard_pose = bool(render_cfg.get("deform_smplx_use_standard_pose", True))

    deform_source = deform_source_raw
    if deform_source != "smplx_params":
        raise ValueError(
            "render.deform_source only supports smplx_params. "
            "Pass target pose via deform_smplx_params_npz + deform_smplx_frame_index/name. "
            f"Got: {deform_source_raw}"
        )

    vis_rgb_raw = render_cfg.get("vis_rgb", None)
    vis_rgb = None
    if isinstance(vis_rgb_raw, (list, tuple)) and len(vis_rgb_raw) == 3:
        vis_rgb = (
            float(vis_rgb_raw[0]),
            float(vis_rgb_raw[1]),
            float(vis_rgb_raw[2]),
        )
    if fg_scale <= 0.0 or bg_scale <= 0.0:
        raise ValueError(f"fg_scale/bg_scale must be > 0. Got fg_scale={fg_scale}, bg_scale={bg_scale}")
    if bound_alpha_threshold < 0.0 or bound_alpha_threshold >= 1.0:
        raise ValueError(
            f"render.bound_alpha_threshold must be in [0,1). Got {bound_alpha_threshold}"
        )
    if bound_opacity_min < 0.0 or bound_opacity_min >= 1.0:
        raise ValueError(
            f"render.bound_opacity_min must be in [0,1). Got {bound_opacity_min}"
        )
    if fg_only_alpha_threshold < 0.0 or fg_only_alpha_threshold >= 1.0:
        raise ValueError(
            f"render.fg_only_alpha_threshold must be in [0,1). Got {fg_only_alpha_threshold}"
        )
    if deform_alpha_threshold < 0.0 or deform_alpha_threshold >= 1.0:
        raise ValueError(
            f"render.deform_alpha_threshold must be in [0,1). Got {deform_alpha_threshold}"
        )
    if fg_only_alpha_gamma <= 0.0 or deform_alpha_gamma <= 0.0:
        raise ValueError(
            f"render alpha gamma must be > 0. Got fg_only/deform={fg_only_alpha_gamma}/{deform_alpha_gamma}"
        )
    if real_align_alpha_threshold < 0.0 or real_align_alpha_threshold >= 1.0:
        raise ValueError(
            f"render.real_align_alpha_threshold must be in [0,1). Got {real_align_alpha_threshold}"
        )
    if real_align_scale_min <= 0.0 or real_align_scale_max <= 0.0 or real_align_scale_min > real_align_scale_max:
        raise ValueError(
            f"Invalid render.real_align_scale_min/max: {real_align_scale_min}/{real_align_scale_max}"
        )
    if fg_mask_mode not in ("none", "smpl", "mesh"):
        raise ValueError(f"Unsupported fg_mask_mode: {fg_mask_mode}. Use one of: none|smpl|mesh")
    if fg_mask_dilate_px < 0 or fg_mask_erode_px < 0:
        raise ValueError(
            f"render.fg_mask_dilate_px/render.fg_mask_erode_px must be >=0. "
            f"Got {fg_mask_dilate_px}/{fg_mask_erode_px}"
        )
    logger.info(
        f"Bind-only flags: real={render_bound_only}, fg_only={fg_only_render_bound_only}, "
        f"deform={deform_render_bound_only}"
    )
    logger.info(f"Deform source: {deform_source}")
    logger.info(
        "Deform render uses SMPL-X params: npz=%s, name=%s, idx=%d, standard_pose=%s",
        deform_smplx_params_npz_cfg or "outputs/stage1/smpl_params_sequence.npz",
        deform_smplx_frame_name,
        int(deform_smplx_frame_index),
        str(bool(deform_smplx_use_standard_pose)).lower(),
    )
    if fg_scale != 1.0 or bg_scale != 1.0:
        logger.info(f"Render scales: fg_scale={fg_scale}, bg_scale={bg_scale}")
    if k_scale != 1.0:
        logger.info(f"Intrinsic scale (FOV): k_scale={k_scale}")
    if debug_point_render:
        logger.info(
            f"Debug point render enabled: radius={debug_point_radius}, alpha={debug_point_alpha}, "
            f"use_gaussian_color={debug_use_gaussian_color}"
        )
        if debug_bind_to_google_skeleton:
            logger.info(f"Debug skeleton binding enabled: band_px={debug_skeleton_band_px}")

    # Optional keypoint bbox mask for foreground/background split
    mask_mode = str(render_cfg.get("mask_mode", "alpha")).lower()
    kpt_boxes = None
    kp_json = render_cfg.get("keypoints_json", None)
    if kp_json is not None:
        kp_json = Path(resolve_path(str(kp_json)))
        if not kp_json.is_absolute():
            kp_json = project_root / kp_json

    debug_kpt_json_path = None
    if debug_keypoints_json is not None and str(debug_keypoints_json).strip():
        debug_kpt_json_path = Path(resolve_path(str(debug_keypoints_json)))
        if not debug_kpt_json_path.is_absolute():
            debug_kpt_json_path = project_root / debug_kpt_json_path
    if debug_need_kpts and debug_kpt_json_path is None:
        raise ValueError(
            "debug_bind_to_google_skeleton/debug_overlay_keypoints is enabled, "
            "but debug_keypoints_json is not set."
        )
    if debug_need_kpts and debug_kpt_json_path is not None and not debug_kpt_json_path.exists():
        raise FileNotFoundError(f"debug_keypoints_json not found: {debug_kpt_json_path}")
    smpl_joints_map = None
    smpl_skeleton_connections = list(SMPL_SKELETON_CONNECTIONS)
    if overlay_smpl_skeleton:
        if smpl_skeleton_npz_cfg is None or not str(smpl_skeleton_npz_cfg).strip():
            if smpl_joints_npz_cfg is not None and str(smpl_joints_npz_cfg).strip():
                smpl_skeleton_npz_cfg = smpl_joints_npz_cfg
            else:
                smpl_skeleton_npz_cfg = "outputs/stage1/smpl_joints_sequence.npz"
        smpl_skeleton_path = Path(resolve_path(str(smpl_skeleton_npz_cfg)))
        if not smpl_skeleton_path.is_absolute():
            smpl_skeleton_path = project_root / smpl_skeleton_path
        if not smpl_skeleton_path.exists():
            legacy_fallback = project_root / Path("outputs/stage1/smpl_skeleton_sequence.npz")
            if legacy_fallback.exists():
                logger.warning(
                    "Primary SMPL joints npz missing, fallback to legacy skeleton npz: "
                    f"{legacy_fallback}"
                )
                smpl_skeleton_path = legacy_fallback
        smpl_joints_map, smpl_skeleton_connections, loaded_joint_key = _load_smpl_skeleton_map(
            smpl_skeleton_path,
        )
        if loaded_joint_key != "joints_cam":
            logger.warning(
                "Loaded legacy joint key for overlay: %s. "
                "For strict fg_only/deform joint consistency, use npz with joints_cam.",
                loaded_joint_key,
            )
        logger.info(
            f"Loaded SMPL skeleton sequence: {smpl_skeleton_path} "
            f"(source={loaded_joint_key}, edges={len(smpl_skeleton_connections)})"
        )

    # Optional: dynamic binding to SMPL mesh sequence
    bind_to_mesh = bool(render_cfg.get("bind_to_mesh", True))
    strict_binding_topology = bool(render_cfg.get("strict_binding_topology", True))
    pose_drive_enabled = bool(render_cfg.get("pose_drive_enabled", False))
    pose_drive_mesh_dir_cfg = render_cfg.get("pose_drive_mesh_sequence_dir", "")
    pose_drive_vertices_npz_cfg = render_cfg.get("pose_drive_vertices_npz", "")
    pose_drive_index_mode = str(render_cfg.get("pose_drive_index_mode", "frame_name")).strip().lower()
    pose_drive_fixed_index = int(render_cfg.get("pose_drive_fixed_index", -1))
    pose_drive_loop = bool(render_cfg.get("pose_drive_loop", True))
    if pose_drive_enabled:
        logger.info(
            "Pose-drive enabled: source(mesh_dir=%s, verts_npz=%s), mode=%s, fixed=%d, loop=%s",
            str(pose_drive_mesh_dir_cfg),
            str(pose_drive_vertices_npz_cfg),
            pose_drive_index_mode,
            int(pose_drive_fixed_index),
            str(bool(pose_drive_loop)).lower(),
        )

    bind_mesh_to_camera = bool(render_cfg.get("bind_mesh_to_camera", False))
    bind_inverse_export_upright = bool(render_cfg.get("bind_inverse_export_upright", True))
    bind_mesh_obj_space_mode = str(render_cfg.get("bind_mesh_obj_space", "auto")).strip().lower()
    if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        raise ValueError(
            f"render.bind_mesh_obj_space must be auto|camera|blender. Got: {bind_mesh_obj_space_mode}"
        )
    deform_mesh_obj_space_mode = str(render_cfg.get("deform_mesh_obj_space", bind_mesh_obj_space_mode)).strip().lower()
    if deform_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        raise ValueError(
            f"render.deform_mesh_obj_space must be auto|camera|blender. Got: {deform_mesh_obj_space_mode}"
        )
    bind_align_quat_to_mesh_normals = bool(render_cfg.get("bind_align_quat_to_mesh_normals", True))
    bind_quat_normal_blend = float(np.clip(render_cfg.get("bind_quat_normal_blend", 1.0), 0.0, 1.0))
    binding_cfg_dict = render_cfg.get("binding", ckpt.get("binding", None))
    default_bind_mode = "games_face" if (g.bind_face_idx is not None and g.bind_bary is not None) else "legacy"
    binding_cfg = parse_binding_mode_config(binding_cfg_dict, default_mode=default_bind_mode)
    legacy_cam_t_autofix = bool(render_cfg.get("legacy_cam_t_autofix", True))
    bind_mesh_offset_cfg = render_cfg.get("bind_mesh_offset", [0.0, 0.0, 0.0])
    if not (isinstance(bind_mesh_offset_cfg, (list, tuple)) and len(bind_mesh_offset_cfg) == 3):
        raise ValueError("render.bind_mesh_offset must be a 3-value list/tuple, e.g. [0, 0, 0].")
    bind_rigid_align_auto_from_stage2 = bool(render_cfg.get("bind_rigid_align_auto_from_stage2", True))

    render_bind_cfg = {
        "bind_mesh_sequence_dir": render_cfg.get("bind_mesh_sequence_dir", None),
        "bind_sequence_npz": render_cfg.get("bind_sequence_npz", ckpt.get("bind_sequence_npz", None)),
        "bind_vertices_npz": render_cfg.get("bind_vertices_npz", ckpt.get("bind_vertices_npz", None)),
        "bind_deform_obj": render_cfg.get("bind_deform_obj", ckpt.get("bind_deform_obj", None)),
        "bind_mesh_offset": bind_mesh_offset_cfg,
        "bind_mesh_to_camera": bind_mesh_to_camera,
        "bind_inverse_export_upright": bind_inverse_export_upright,
        "bind_mesh_obj_space": bind_mesh_obj_space_mode,
        "bind_rigid_align": render_cfg.get("bind_rigid_align", {"enabled": False}),
    }
    mesh_runtime = Stage2MeshRuntime.from_stage2_config(
        project_root=project_root,
        stage1_dir=stage1_dir,
        cfg=render_bind_cfg,
        frames=frames,
        logger=logger,
        device=device,
        bind_to_mesh=bind_to_mesh,
        auto_from_stage2_report=bind_rigid_align_auto_from_stage2,
        stage2_dir_for_report=stage2_dir,
    )
    prepared_smpl_align = _load_stage2_smpl_align_subset(
        cache_path=prepared_supervision_cache_path,
        frames=[str(x) for x in frames],
        logger=logger,
    )
    if prepared_smpl_align is not None:
        mesh_runtime.set_frame_smpl_align_from_arrays(
            frame_names=[str(x) for x in prepared_smpl_align["frame_names"].tolist()],
            scales=np.asarray(prepared_smpl_align["smpl_align_scales"], dtype=np.float32),
            tx_px=np.asarray(prepared_smpl_align["smpl_align_tx_px"], dtype=np.float32),
            ty_px=np.asarray(prepared_smpl_align["smpl_align_ty_px"], dtype=np.float32),
            accepted=np.asarray(prepared_smpl_align["smpl_align_accepted"], dtype=np.uint8),
        )
    mesh_context = mesh_runtime.context
    bind_seq_npz = mesh_context.bind_seq_npz_path
    bind_vertices_npz = mesh_context.bind_vertices_npz_path
    bind_mesh_indices = [] if mesh_context.bind_mesh_indices is None else [int(x) for x in mesh_context.bind_mesh_indices]
    binding_enabled = bind_to_mesh and mesh_context.bind_mesh_dir is not None and g.bind_idx is not None
    if fg_mask_mode in ("smpl", "mesh") and not binding_enabled:
        raise ValueError("fg_mask_mode=smpl requires bind_mesh_sequence_dir and bound gaussians.")

    bind_mesh_obj_space_mode = str(mesh_context.bind_mesh_obj_space_mode).strip().lower()
    mesh_obj_in_camera_space = bool(mesh_context.mesh_obj_in_camera_space)
    deform_mesh_obj_in_camera_space = bool(deform_mesh_obj_space_mode == "camera")
    bind_mesh_offset_np = np.asarray(mesh_context.bind_mesh_offset_np, dtype=np.float32).reshape(1, 3)
    bind_mesh_offset_t = torch.from_numpy(bind_mesh_offset_np.copy()).to(device=device, dtype=torch.float32)
    bind_rigid_align_apply_cfg = None if mesh_context.bind_rigid_apply_cfg is None else dict(mesh_context.bind_rigid_apply_cfg)
    if bind_rigid_align_apply_cfg is not None and (not binding_enabled):
        logger.warning("bind_rigid_align is enabled but binding is disabled; ignoring rigid align.")
        bind_rigid_align_apply_cfg = None
    if bind_rigid_align_apply_cfg is not None:
        logger.info(
            "Bind rigid align enabled: "
            f"rot=({bind_rigid_align_apply_cfg['rx_deg']:.3f},"
            f"{bind_rigid_align_apply_cfg['ry_deg']:.3f},"
            f"{bind_rigid_align_apply_cfg['rz_deg']:.3f})deg, "
            f"trans=({bind_rigid_align_apply_cfg['tx']:.4f},"
            f"{bind_rigid_align_apply_cfg['ty']:.4f},"
            f"{bind_rigid_align_apply_cfg['tz']:.4f})"
        )

    mesh_normal_cache: dict[int, torch.Tensor] = {}
    bind_idx_t = None
    bind_vertex_offset_t = None
    bind_face_idx_t = None
    bind_bary_t = None
    bind_faces_np = mesh_context.bind_faces_np
    prev_vertex_dirs: torch.Tensor | None = None
    exp_upright_rot = np.asarray(mesh_context.exp_upright_rot, dtype=np.float32)
    exp_center_xy = np.asarray(mesh_context.exp_center_xy, dtype=np.float32).reshape(2,)
    exp_floor_z = float(mesh_context.exp_floor_z)
    exp_has_center_xy = bool(mesh_context.exp_has_center_xy)
    exp_has_floor = bool(mesh_context.exp_has_floor)
    deform_verts_cam_np: np.ndarray | None = None
    deform_default_joints_cam_np: np.ndarray | None = None

    if binding_enabled and bind_faces_np is not None and len(bind_mesh_indices) == 0:
        raise RuntimeError("Binding is enabled but bind mesh indices are empty.")
    if bind_faces_np is None or len(bind_faces_np) == 0:
        if fg_mask_mode in ("smpl", "mesh") and binding_enabled:
            raise RuntimeError("fg_mask_mode=smpl requires mesh faces, but loaded mesh has no faces.")
        if binding_enabled and str(binding_cfg.mode).lower() == "games_face":
            raise RuntimeError("binding.mode=games_face requires mesh faces, but loaded mesh has no faces.")
        bind_faces_np = None

    if bind_seq_npz is not None and bind_seq_npz.exists():
        seq_meta = np.load(bind_seq_npz, allow_pickle=True)
        fit_strategy_hint = None
        if "fit_strategy" in seq_meta.files:
            fs_arr = np.asarray(seq_meta["fit_strategy"]).reshape(-1)
            if fs_arr.size > 0:
                fit_strategy_hint = str(fs_arr[0]).strip().lower()
        if deform_mesh_obj_space_mode == "auto" and fit_strategy_hint in ("stage1_motion_prior", "vibe"):
            deform_mesh_obj_in_camera_space = True
        elif deform_mesh_obj_space_mode == "blender":
            deform_mesh_obj_in_camera_space = False
        deform_verts_key = "verts_canonical_cam" if "verts_canonical_cam" in seq_meta.files else ("verts_tpose_cam" if "verts_tpose_cam" in seq_meta.files else "")
        if deform_verts_key:
            deform_verts_cam_np = np.asarray(seq_meta[deform_verts_key], dtype=np.float32)
            if deform_verts_cam_np.ndim != 2 or deform_verts_cam_np.shape[1] != 3:
                logger.warning(
                    f"Ignore malformed {deform_verts_key} in {bind_seq_npz}: {deform_verts_cam_np.shape}"
                )
                deform_verts_cam_np = None
            else:
                logger.info(
                    f"Loaded camera-space canonical verts from {bind_seq_npz} "
                    f"(source={deform_verts_key}, V={int(deform_verts_cam_np.shape[0])})"
                )
        deform_joints_key = "joints_canonical_cam" if "joints_canonical_cam" in seq_meta.files else ("joints_tpose_cam" if "joints_tpose_cam" in seq_meta.files else "")
        if deform_joints_key:
            deform_default_joints_cam_np = np.asarray(seq_meta[deform_joints_key], dtype=np.float32)
            if deform_default_joints_cam_np.ndim == 3 and deform_default_joints_cam_np.shape[0] == 1:
                deform_default_joints_cam_np = deform_default_joints_cam_np[0]
            if deform_default_joints_cam_np.ndim != 2 or deform_default_joints_cam_np.shape[1] != 3:
                logger.warning(
                    f"Ignore malformed {deform_joints_key} in {bind_seq_npz}: {deform_default_joints_cam_np.shape}"
                )
                deform_default_joints_cam_np = None
            else:
                logger.info(
                    f"Loaded camera-space canonical joints from {bind_seq_npz} "
                    f"(source={deform_joints_key}, J={int(deform_default_joints_cam_np.shape[0])})"
                )

    if binding_enabled:
        bind_idx_t = g.bind_idx.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0
        if g.bind_face_idx is not None:
            bind_face_idx_t = g.bind_face_idx.to(device=device, dtype=torch.long)
        if g.bind_bary is not None:
            bind_bary_t = g.bind_bary.to(device=device, dtype=torch.float32)
        if (
            legacy_cam_t_autofix
            and mesh_context.bind_mesh_to_camera
            and cam_t is not None
            and cam_t.numel() > 0
        ):
            cam_t_np = cam_t.detach().cpu().numpy()
            if (
                np.all(np.abs(cam_t_np[:, 0]) < 1e-6)
                and np.all(np.abs(cam_t_np[:, 1]) < 1e-6)
                and np.all(np.abs(cam_t_np[:, 2] - 2.5) < 1e-3)
            ):
                logger.warning(
                    "Detected legacy cam_t=[0,0,2.5] with bind_mesh_to_camera=true; "
                    "auto-correcting to cam_t=[0,0,0] for alignment."
                )
                cam_t = cam_t.clone()
                cam_t[:, 2] = cam_t[:, 2] - 2.5
        if g.bind_offset is not None:
            bind_vertex_offset_t = g.bind_offset.to(device=device, dtype=torch.float32)

        def _faces_compatible_with_verts(verts_np: np.ndarray, context: str) -> bool:
            if bind_faces_np is None:
                return True
            vmax = int(np.max(bind_faces_np))
            n_v = int(np.asarray(verts_np).shape[0])
            if vmax < n_v:
                return True
            msg = (
                f"Binding topology mismatch at {context}: mesh faces reference vertex {vmax}, "
                f"but current verts only have {n_v} vertices."
            )
            if strict_binding_topology:
                raise RuntimeError(
                    msg + " Regenerate binding assets (faces/verts/deform) from the same mesh topology."
                )
            logger.warning(msg + " Non-strict mode will skip face-dependent operations.")
            return False

        def _load_mesh_verts(frame_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
            K_curr_np = frame_K_nps[int(frame_idx)] if frame_K_nps is not None else K_base_np
            verts_t, mesh_idx_use = mesh_runtime.load_frame_mesh_verts(
                int(frame_idx),
                K_np=K_curr_np,
                bind_cache_size=int(render_cfg.get("bind_cache_size", 8)),
            )
            normals_t = mesh_normal_cache.get(mesh_idx_use, None)
            if normals_t is None and bool(bind_align_quat_to_mesh_normals) and bind_faces_np is not None:
                verts_np = np.asarray(mesh_runtime.frame_mesh_np_cache[int(frame_idx)], dtype=np.float32)
                if _faces_compatible_with_verts(verts_np, f"binding frame {mesh_idx_use}"):
                    normals_np = _compute_vertex_normals_np(verts_np, bind_faces_np)
                    normals_t = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)
                    mesh_normal_cache[mesh_idx_use] = normals_t
            return verts_t, normals_t, mesh_idx_use

        def _load_binding_frame(frame_idx: int) -> BindingFrameData:
            frame_idx = int(frame_idx)
            if frame_idx in mesh_runtime.frame_binding_cache:
                return mesh_runtime.frame_binding_cache[frame_idx]
            if frame_idx not in mesh_runtime.frame_mesh_np_cache:
                K_curr_np = frame_K_nps[int(frame_idx)] if frame_K_nps is not None else K_base_np
                mesh_runtime.load_frame_mesh_verts(
                    frame_idx,
                    K_np=K_curr_np,
                    bind_cache_size=int(render_cfg.get("bind_cache_size", 8)),
                )
            verts_np = np.asarray(mesh_runtime.frame_mesh_np_cache[frame_idx], dtype=np.float32)
            faces_use = bind_faces_np
            if faces_use is not None and (not _faces_compatible_with_verts(verts_np, f"binding frame {frame_idx}")):
                if str(binding_cfg.mode).lower() in ("games_face", "hybrid"):
                    raise RuntimeError(
                        "binding.mode requires mesh faces, but current verts/faces topology is inconsistent."
                    )
                faces_use = None
            adjacency = None
            if faces_use is not None:
                if mesh_runtime.bind_adjacency is None or len(mesh_runtime.bind_adjacency) != int(verts_np.shape[0]):
                    mesh_runtime.bind_adjacency = build_vertex_adjacency(int(verts_np.shape[0]), faces_use)
                adjacency = mesh_runtime.bind_adjacency
            frame = build_binding_frame_data(
                verts_np=verts_np,
                faces_np=faces_use,
                adjacency=adjacency,
                cfg=binding_cfg,
                device=device,
            )
            mesh_runtime.frame_binding_cache[frame_idx] = frame
            return frame

    deform_pick_mode = str(render_cfg.get("deform_frame_source", "last")).strip().lower()
    deform_pick_idx = _resolve_pick_index(
        mode=deform_pick_mode,
        n=to_render,
        index=int(render_cfg.get("deform_frame_index", -1)),
    )
    deform_ctx: dict[str, object] | None = None
    invalid_bind_warned = False
    for i in range(to_render):
        frame_path = frames_dir / frames[i]
        gt = read_image_rgb(frame_path).to(device)  # (3,H,W)
        base_h, base_w = int(gt.shape[1]), int(gt.shape[2])
        use_gt_bg = bool(render_cfg.get("use_gt_background", False))
        bg_gt_chw = gt if use_gt_bg else None
        bg_gt_hwc = gt.permute(1, 2, 0).contiguous() if use_gt_bg else None
        if mask_mode == "kpt" and kpt_boxes is None and kp_json is not None:
            kpt_boxes = _load_keypoints_boxes(
                keypoints_json=kp_json,
                frames=frames,
                H=base_h,
                W=base_w,
                conf_min=float(render_cfg.get("kpt_conf_min", 0.2)),
                pad=float(render_cfg.get("kpt_box_pad", 0.15)),
            )

        # Camera for this frame
        if cam_w is None or cam_t is None:
            R = torch.eye(3, device=device, dtype=torch.float32)
            t = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        else:
            R = axis_angle_to_rotmat(cam_w[i])
            t = cam_t[i]
        K_curr = frame_K_tensors[i] if frame_K_tensors is not None else K

        view_orbit_enabled = (abs(view_yaw_deg) > 1e-8) or (abs(view_pitch_deg) > 1e-8) or (abs(view_roll_deg) > 1e-8)
        R_delta_view = None
        if view_orbit_enabled:
            R_delta_view = _euler_deg_to_rotmat_torch(
                yaw_deg=view_yaw_deg,
                pitch_deg=view_pitch_deg,
                roll_deg=view_roll_deg,
                device=torch.device(device),
                axis_mode=view_axis_mode,
            )

        verts_t = None
        normals_t = None
        bind_mask_curr = bind_mask if binding_enabled else None

        # Update bound xyz for this frame (dynamic binding)
        if binding_enabled:
            mesh_idx = int(bind_mesh_indices[i])
            verts_t, normals_t, mesh_idx_use = _load_mesh_verts(i)
            if bind_mask.any():
                if str(binding_cfg.mode).lower() == "legacy":
                    bind_idx_use = bind_idx_t[bind_mask]
                    valid_local = bind_idx_use < int(verts_t.shape[0])
                    if bool(valid_local.all()):
                        xyz_new = verts_t.index_select(0, bind_idx_use)
                        if bind_vertex_offset_t is not None:
                            xyz_new = xyz_new + bind_vertex_offset_t[bind_mask]
                        g.xyz[bind_mask] = xyz_new.detach()
                        if bool(bind_align_quat_to_mesh_normals) and normals_t is not None and bind_quat_normal_blend > 0.0:
                            q_target = _quat_from_normals_torch(normals_t.index_select(0, bind_idx_use))
                            with torch.no_grad():
                                if bind_quat_normal_blend >= 0.999:
                                    g.quat[bind_mask] = q_target
                                else:
                                    q_old = g.quat[bind_mask]
                                    q_mix = (1.0 - bind_quat_normal_blend) * q_old + bind_quat_normal_blend * q_target
                                    g.quat[bind_mask] = F.normalize(q_mix, dim=1, eps=1e-8)
                    else:
                        # Keep rendering robust when stage2 checkpoint bind topology and
                        # stage1 mesh topology differ (e.g., old checkpoint + new stage1).
                        valid_mask = bind_mask.clone()
                        valid_mask[bind_mask] = valid_local
                        bind_mask_curr = valid_mask
                        if not bool(valid_mask.any()):
                            raise RuntimeError(
                                "All bind indices are out of range for current mesh topology. "
                                "Please retrain stage2 after rerunning stage1."
                            )
                        bad_n = int((~valid_local).sum().item())
                        if strict_binding_topology:
                            raise RuntimeError(
                                "Detected out-of-range bind indices during render "
                                f"({bad_n} invalid). Strict mode is enabled; "
                                "regenerate consistent binding assets and retrain stage2."
                            )
                        if not invalid_bind_warned:
                            logger.warning(
                                "Detected %d out-of-range bind indices for current mesh; "
                                "skipping them for rendering. Retrain stage2 to fully align topology.",
                                bad_n,
                            )
                            invalid_bind_warned = True
                        valid_idx = bind_idx_t[valid_mask]
                        xyz_new = verts_t.index_select(0, valid_idx)
                        if bind_vertex_offset_t is not None:
                            xyz_new = xyz_new + bind_vertex_offset_t[valid_mask]
                        g.xyz[valid_mask] = xyz_new.detach()
                        if bool(bind_align_quat_to_mesh_normals) and normals_t is not None and bind_quat_normal_blend > 0.0:
                            q_target = _quat_from_normals_torch(normals_t.index_select(0, valid_idx))
                            with torch.no_grad():
                                if bind_quat_normal_blend >= 0.999:
                                    g.quat[valid_mask] = q_target
                                else:
                                    q_old = g.quat[valid_mask]
                                    q_mix = (1.0 - bind_quat_normal_blend) * q_old + bind_quat_normal_blend * q_target
                                    g.quat[valid_mask] = F.normalize(q_mix, dim=1, eps=1e-8)
                else:
                    frame_data = _load_binding_frame(i)
                    prev_vertex_dirs, valid_inner = apply_inner_binding_per_frame(
                        g=g,
                        frame=frame_data,
                        bind_mask=bind_mask,
                        bind_vertex_idx=bind_idx_t,
                        bind_vertex_offset=bind_vertex_offset_t,
                        bind_face_idx=bind_face_idx_t,
                        bind_bary=bind_bary_t,
                        cfg=binding_cfg,
                        quat_blend=bind_quat_normal_blend,
                        prev_vertex_dirs=prev_vertex_dirs,
                    )
                    prev_vertex_dirs, valid_outer = apply_outer_anchor_per_frame(
                        g=g,
                        frame=frame_data,
                        bind_mask=bind_mask,
                        bind_vertex_idx=bind_idx_t,
                        bind_vertex_offset=bind_vertex_offset_t,
                        bind_face_idx=bind_face_idx_t,
                        bind_bary=bind_bary_t,
                        cfg=binding_cfg,
                        quat_blend=bind_quat_normal_blend,
                        prev_vertex_dirs=prev_vertex_dirs,
                    )
                    bind_mask_curr = (valid_inner | valid_outer)
                    if bind_mask_curr is not None and not bool(bind_mask_curr.all()) and not invalid_bind_warned:
                        bad_n = int((bind_mask & (~bind_mask_curr)).sum().item())
                        if bad_n > 0:
                            if strict_binding_topology:
                                raise RuntimeError(
                                    "Detected invalid bindings for current topology "
                                    f"({bad_n} invalid). Strict mode is enabled; "
                                    "regenerate consistent binding assets and retrain stage2."
                                )
                            logger.warning(
                                "Detected %d invalid bindings for current mesh topology; "
                                "skipping those gaussians.",
                                bad_n,
                            )
                            invalid_bind_warned = True

        center_world_joint = None
        if center_subject_joint_idx >= 0 and smpl_joints_map is not None:
            joints_center = smpl_joints_map.get(frames[i], None)
            if joints_center is not None:
                joints_center = np.asarray(joints_center, dtype=np.float32).copy()
                if joints_center.ndim == 2 and joints_center.shape[1] == 3 and joints_center.shape[0] > 0:
                    if bind_rigid_align_apply_cfg is not None:
                        joints_center = apply_rigid_transform_np(joints_center, bind_rigid_align_apply_cfg)
                    joints_center = joints_center + bind_mesh_offset_np
                    j_idx = int(np.clip(center_subject_joint_idx, 0, joints_center.shape[0] - 1))
                    center_world_joint = torch.from_numpy(joints_center[j_idx]).to(device=device, dtype=torch.float32)

        # Orbit view should rotate around subject pivot instead of world origin.
        # Otherwise side/back views can move subject behind camera and become blank.
        if view_orbit_enabled and R_delta_view is not None:
            pivot_world = center_world_joint
            if pivot_world is None and center_subject_use_mesh and (verts_t is not None) and (verts_t.numel() > 0):
                pivot_world = verts_t.mean(dim=0)
            elif pivot_world is None and g.xyz is not None and g.xyz.numel() > 0:
                if (g.bind_idx is not None) and bool((g.bind_idx >= 0).any()):
                    pivot_world = g.xyz[g.bind_idx >= 0].mean(dim=0)
                else:
                    pivot_world = g.xyz.mean(dim=0)
            if pivot_world is None:
                pivot_world = torch.zeros((3,), device=device, dtype=torch.float32)
            R, t = _apply_world_rotation_about_pivot_extrinsics(
                R=R,
                t=t,
                Q=R_delta_view,
                pivot_world=pivot_world,
            )

        # Optional: split gaussians by bind_idx
        if split_by_bind and g.bind_idx is not None:
            fg_mask = (g.bind_idx >= 0)
            bg_mask = ~fg_mask
            g_fg = _select_gaussians(g, fg_mask)
            g_bg = _select_gaussians(g, bg_mask) if bg_mask.any() else None
        else:
            g_fg = g
            g_bg = None

        if render_bound_only:
            if g.bind_idx is None:
                raise RuntimeError("render_bound_only=true but bind_idx is missing in checkpoint.")
            fg_mask = (g.bind_idx >= 0)
            if not fg_mask.any():
                raise RuntimeError("render_bound_only=true but no bound gaussians (bind_idx >= 0).")
            g_fg = _select_gaussians(g, fg_mask)
            g_bg = None
            if bound_opacity_min > 0.0:
                opa = torch.sigmoid(g_fg.opacity_logit).squeeze(-1)
                keep = opa >= float(bound_opacity_min)
                if keep.any():
                    g_fg = _select_gaussians(g_fg, keep)
                else:
                    raise RuntimeError(
                        "render_bound_only with bound_opacity_min removed all gaussians. "
                        "Lower render.bound_opacity_min."
                    )

        if visualize_bound_gaussians:
            g_fg = _apply_vis_override(
                g=g_fg,
                opacity_logit=vis_opacity_logit,
                scale_mult=vis_scale_mult,
                rgb_override=vis_rgb,
            )

        # Scale intrinsics for wider/narrower FOV if requested
        K_use = _scale_intrinsics(K_curr, k_scale)


        # Keep subject centered in image to avoid out-of-frame blank renders.
        if center_subject:
            center_world = center_world_joint
            if center_world is None and center_subject_use_mesh and (verts_t is not None) and (verts_t.numel() > 0):
                center_world = verts_t.mean(dim=0)
            elif center_world is None and g.xyz is not None and g.xyz.numel() > 0:
                if (g.bind_idx is not None) and bool((g.bind_idx >= 0).any()):
                    center_world = g.xyz[g.bind_idx >= 0].mean(dim=0)
                else:
                    center_world = g.xyz.mean(dim=0)
            if center_world is not None:
                xc = (R @ center_world.view(3, 1)).view(3) + t
                zc = torch.clamp(xc[2], min=float(center_subject_depth_min))
                fx = K_use[0, 0]
                fy = K_use[1, 1]
                cx = K_use[0, 2]
                cy = K_use[1, 2]
                u_tgt = torch.tensor((base_w - 1) * 0.5, device=device, dtype=torch.float32)
                v_tgt = torch.tensor((base_h - 1) * 0.5, device=device, dtype=torch.float32)
                x_tgt = (u_tgt - cx) * zc / fx
                y_tgt = (v_tgt - cy) * zc / fy
                t = t + torch.stack(
                    [x_tgt - xc[0], y_tgt - xc[1], torch.zeros((), device=device, dtype=torch.float32)],
                    dim=0,
                )
        if center_subject and subject_fit_enabled and (verts_t is not None) and (verts_t.numel() > 0):
            K_np_fit, t_np_fit = _fit_subject_camera_numpy(
                verts=verts_t.detach().cpu().numpy().astype(np.float32),
                K=K_use.detach().cpu().numpy().astype(np.float32),
                R=R.detach().cpu().numpy().astype(np.float32),
                t=t.detach().cpu().numpy().astype(np.float32),
                H=base_h,
                W=base_w,
                fill=float(subject_fit_fill),
                iters=int(subject_fit_iters),
            )
            K_use = torch.from_numpy(K_np_fit).to(device=device, dtype=torch.float32)
            t = torch.from_numpy(t_np_fit).to(device=device, dtype=torch.float32)
        if debug_point_render:
            skeleton_segments = None
            if debug_bind_to_google_skeleton and debug_kpt_json_path is not None:
                frame_kpts = _load_frame_keypoints(
                    keypoints_json=debug_kpt_json_path,
                    frame_name=frames[i],
                    conf_min=debug_kpt_conf_min,
                )
                skeleton_segments = _build_skeleton_segments_2d(
                    keypoints_xyc=frame_kpts,
                    W=base_w,
                    H=base_h,
                )
            pred = _render_points_debug(
                g=g_fg,
                K=K_use,
                R=R,
                t=t,
                H=base_h,
                W=base_w,
                bg=bg,
                point_radius=debug_point_radius,
                point_alpha=debug_point_alpha,
                use_gaussian_color=debug_use_gaussian_color,
                point_color=debug_point_color,
                surface_only=debug_surface_only,
                skeleton_segments=skeleton_segments,
                near_skeleton_px=debug_skeleton_band_px,
            )
            if debug_need_kpts and debug_kpt_json_path is not None:
                pred = _overlay_keypoints(
                    rgb_chw=pred,
                    keypoints_json=debug_kpt_json_path,
                    frame_name=frames[i],
                    conf_min=debug_kpt_conf_min,
                    radius=debug_kpt_radius,
                    draw_skeleton=debug_draw_skeleton,
                    body_only=debug_overlay_body_only,
                )
        elif fg_scale == 1.0 and bg_scale == 1.0:
            if render_bound_only and bound_alpha_threshold > 0.0:
                fg_rgb, fg_alpha = render_gaussians_3d(
                    g=g_fg,
                    K=K_use,
                    R=R,
                    t=t,
                    H=base_h,
                    W=base_w,
                    bg=None,
                    return_alpha=True,
                )
                hard_m = (fg_alpha > float(bound_alpha_threshold)).float()
                if use_gt_bg and bg_gt_chw is not None:
                    bg_fill = bg_gt_chw
                else:
                    bg_fill = torch.tensor(bg, dtype=torch.float32, device=device).view(3, 1, 1).expand_as(fg_rgb)
                pred = fg_rgb * hard_m + bg_fill * (1.0 - hard_m)
            else:
                # IMPORTANT: infer output size from gt (do NOT hardcode H/W)
                pred = render_gaussians_3d(
                    g=g_fg,
                    K=K_use,
                    R=R,
                    t=t,
                    H=None,
                    W=None,
                    bg=bg_gt_hwc if use_gt_bg else bg,
                    ref_image=gt,  # << output size == gt size
                )
        else:
            # Foreground: higher scale, keep alpha for compositing
            fg_h = max(1, int(round(base_h * fg_scale)))
            fg_w = max(1, int(round(base_w * fg_scale)))
            K_fg = _scale_intrinsics(K_use, fg_scale)
            fg_rgb, fg_alpha = render_gaussians_3d(
                g=g_fg,
                K=K_fg,
                R=R,
                t=t,
                H=fg_h,
                W=fg_w,
                bg=None,
                return_alpha=True,
            )
            fg_rgb = _resize_chw(fg_rgb, base_h, base_w)
            fg_alpha = _resize_chw(fg_alpha, base_h, base_w).clamp(0.0, 1.0)

            # Background: either render low-res gaussians, or use GT image
            if use_gt_bg and bg_gt_chw is not None:
                bg_rgb = bg_gt_chw
            else:
                bg_h = max(1, int(round(base_h * bg_scale)))
                bg_w = max(1, int(round(base_w * bg_scale)))
                K_bg = _scale_intrinsics(K_use, bg_scale)
                bg_rgb = render_gaussians_3d(
                    g=g_bg if g_bg is not None else g_fg,
                    K=K_bg,
                    R=R,
                    t=t,
                    H=bg_h,
                    W=bg_w,
                    bg=bg,
                )
                bg_rgb = _resize_chw(bg_rgb, base_h, base_w)

            if render_bound_only and bound_alpha_threshold > 0.0:
                hard_m = (fg_alpha > float(bound_alpha_threshold)).float()
                pred = fg_rgb * hard_m + bg_rgb * (1.0 - hard_m)
            elif mask_mode == "kpt" and kpt_boxes is not None and kpt_boxes[i] is not None:
                x0, y0, x1, y1 = kpt_boxes[i]
                m = torch.zeros((1, base_h, base_w), device=device)
                m[:, int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1.0
                pred = fg_rgb * m + bg_rgb * (1.0 - m)
            else:
                pred = _composite_premultiplied(fg_rgb=fg_rgb, fg_alpha=fg_alpha, bg_rgb=bg_rgb)

        if (
            fg_mask_mode in ("smpl", "mesh")
            and binding_enabled
            and verts_t is not None
            and bind_faces_np is not None
        ):
            mask_np = _rasterize_mesh_mask(
                verts=verts_t.detach().cpu().numpy(),
                faces=bind_faces_np,
                K=K_use.detach().cpu().numpy().astype(np.float32),
                R=R.detach().cpu().numpy().astype(np.float32),
                t=t.detach().cpu().numpy().astype(np.float32),
                H=base_h,
                W=base_w,
                dilate_px=int(fg_mask_dilate_px),
                erode_px=int(fg_mask_erode_px),
            )
            fg_mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            if use_gt_bg and bg_gt_chw is not None:
                bg_fill = bg_gt_chw
            else:
                bg_fill = torch.tensor(bg, dtype=torch.float32, device=device).view(3, 1, 1).expand_as(pred)
            pred = pred * fg_mask + bg_fill * (1.0 - fg_mask)

        frame_out_name = f"{Path(frames[i]).stem}{file_suffix}.png"
        pred_real = pred

        if save_main_image:
            pred_main = pred_real
            if (not debug_point_render) and debug_overlay_keypoints and debug_kpt_json_path is not None:
                pred_main = _overlay_keypoints(
                    rgb_chw=pred_main,
                    keypoints_json=debug_kpt_json_path,
                    frame_name=frames[i],
                    conf_min=debug_kpt_conf_min,
                    radius=debug_kpt_radius,
                    draw_skeleton=debug_draw_skeleton,
                    body_only=debug_overlay_body_only,
                )
            if overlay_smpl_skeleton and smpl_joints_map is not None:
                joints_vis = smpl_joints_map.get(frames[i], None)
                if joints_vis is not None and bind_rigid_align_apply_cfg is not None:
                    joints_vis = apply_rigid_transform_np(
                        np.asarray(joints_vis, dtype=np.float32),
                        bind_rigid_align_apply_cfg,
                    )
                pred_main = _overlay_smpl_skeleton(
                    rgb_chw=pred_main,
                    joints_cam=joints_vis,
                    K=K_use,
                    R=R,
                    t=t,
                    connections=smpl_skeleton_connections,
                    color_bgr=smpl_color_bgr,
                    thickness=smpl_line_thickness,
                    radius=smpl_joint_radius,
                )
            save_rgb(out_dir / frame_out_name, pred_main)

        if save_real_image:
            if layer_split_output:
                real_layers = _split_gaussians_by_shell_layer(g_fg)
                for layer_key in active_layer_keys:
                    if layer_key == "combined":
                        pred_layer = pred_real
                    else:
                        pred_layer = _render_layer_preview(
                            g_layer=real_layers.get(layer_key, None),
                            K=K_use,
                            R=R,
                            t=t,
                            H=base_h,
                            W=base_w,
                            bg=bg,
                            bg_gt_chw=bg_gt_chw,
                            bg_gt_hwc=bg_gt_hwc,
                            use_gt_bg=use_gt_bg,
                        )
                    save_rgb(real_layer_dirs[layer_key] / frame_out_name, pred_layer)
            else:
                save_rgb(real_out_dir / frame_out_name, pred_real)
        if save_fg_nobg_skeleton:
            g_fg_only = g_fg
            if fg_only_render_bound_only:
                if g.bind_idx is None:
                    raise RuntimeError("fg_only_render_bound_only=true but bind_idx is missing in checkpoint.")
                fg_only_mask = bind_mask_curr if bind_mask_curr is not None else (g.bind_idx >= 0)
                if not bool(fg_only_mask.any()):
                    raise RuntimeError(
                        "fg_only_render_bound_only=true but no valid bound gaussians for this frame."
                    )
                g_fg_only = _select_gaussians(g, fg_only_mask)
                if bound_opacity_min > 0.0:
                    opa_fg_only = torch.sigmoid(g_fg_only.opacity_logit).squeeze(-1)
                    keep_fg_only = opa_fg_only >= float(bound_opacity_min)
                    if keep_fg_only.any():
                        g_fg_only = _select_gaussians(g_fg_only, keep_fg_only)
                    else:
                        raise RuntimeError(
                            "fg_only render with bound_opacity_min removed all bound gaussians. "
                            "Lower render.bound_opacity_min."
                        )
            fg_nobg_rgb, fg_nobg_alpha = render_gaussians_3d(
                g=g_fg_only,
                K=K_use,
                R=R,
                t=t,
                H=base_h,
                W=base_w,
                bg=None,
                return_alpha=True,
            )
            fg_mask = _alpha_to_mask(
                fg_nobg_alpha,
                threshold=fg_only_alpha_threshold,
                gamma=fg_only_alpha_gamma,
            )
            if (
                fg_mask_mode in ("smpl", "mesh")
                and binding_enabled
                and verts_t is not None
                and bind_faces_np is not None
            ):
                mask_np = _rasterize_mesh_mask(
                    verts=verts_t.detach().cpu().numpy(),
                    faces=bind_faces_np,
                    K=K_use.detach().cpu().numpy().astype(np.float32),
                    R=R.detach().cpu().numpy().astype(np.float32),
                    t=t.detach().cpu().numpy().astype(np.float32),
                    H=base_h,
                    W=base_w,
                    dilate_px=int(fg_mask_dilate_px),
                    erode_px=int(fg_mask_erode_px),
                )
                fg_mask_mesh = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                fg_mask = fg_mask * fg_mask_mesh
            alpha_src = fg_nobg_alpha.clamp(min=1e-4)
            gain = (fg_mask / alpha_src).clamp(min=0.0, max=float(fg_only_alpha_gain_max))
            fg_rgb_adj = (fg_nobg_rgb * gain).clamp(0.0, 1.0)
            bg_fill_fg = torch.tensor(bg, dtype=torch.float32, device=device).view(3, 1, 1).expand_as(fg_nobg_rgb)
            pred_fg_only = _composite_premultiplied(fg_rgb=fg_rgb_adj, fg_alpha=fg_mask, bg_rgb=bg_fill_fg)
            if overlay_smpl_skeleton and smpl_joints_map is not None:
                joints_vis = smpl_joints_map.get(frames[i], None)
                if joints_vis is not None and bind_rigid_align_apply_cfg is not None:
                    joints_vis = apply_rigid_transform_np(
                        np.asarray(joints_vis, dtype=np.float32),
                        bind_rigid_align_apply_cfg,
                    )
                pred_fg_only = _overlay_smpl_skeleton(
                    rgb_chw=pred_fg_only,
                    joints_cam=joints_vis,
                    K=K_use,
                    R=R,
                    t=t,
                    connections=smpl_skeleton_connections,
                    color_bgr=smpl_color_bgr,
                    thickness=smpl_line_thickness,
                    radius=smpl_joint_radius,
                )
            if layer_split_output:
                fg_layers = _split_gaussians_by_shell_layer(g_fg_only)
                for layer_key in active_layer_keys:
                    if layer_key == "combined":
                        pred_layer = pred_fg_only
                    else:
                        pred_layer = _render_layer_preview(
                            g_layer=fg_layers.get(layer_key, None),
                            K=K_use,
                            R=R,
                            t=t,
                            H=base_h,
                            W=base_w,
                            bg=bg,
                            bg_gt_chw=None,
                            bg_gt_hwc=None,
                            use_gt_bg=False,
                        )
                    save_rgb(fg_layer_dirs[layer_key] / frame_out_name, pred_layer)
            else:
                save_rgb(fg_nobg_out_dir / frame_out_name, pred_fg_only)
        if render_deform_image and i == int(deform_pick_idx):
            deform_ctx = {
                "frame_name": frames[i],
                "H": base_h,
                "W": base_w,
                "K_use": K_use.detach().clone(),
                "R": R.detach().clone(),
                "t": t.detach().clone(),
            }

        if (i + 1) % 50 == 0:
            logger.info(f"Rendered {i+1}/{to_render} ...")

    if render_deform_image and deform_ctx is not None:
        if not binding_enabled:
            logger.warning("Skip deform render: binding is disabled or bind_idx is missing.")
        elif bind_idx_t is None:
            logger.warning("Skip deform render: bind indices unavailable.")
        else:
            verts_deform: torch.Tensor | None = None
            verts_np_bind: np.ndarray | None = None
            deform_joints_cam_np: np.ndarray | None = None
            if deform_source == "smplx_params":
                params_npz_cfg = deform_smplx_params_npz_cfg
                if not params_npz_cfg:
                    if bind_seq_npz is not None:
                        params_npz_cfg = str(bind_seq_npz)
                    else:
                        params_npz_cfg = "outputs/stage1/smpl_params_sequence.npz"
                params_npz_path = Path(resolve_path(str(params_npz_cfg)))
                if not params_npz_path.is_absolute():
                    params_npz_path = project_root / params_npz_path
                req_frame_index = int(deform_smplx_frame_index)
                req_frame_name_cfg = str(deform_smplx_frame_name).strip()
                # If an explicit deform frame index is provided, do not pass render-selected frame_name.
                # This prevents index requests from being implicitly overridden by frame_select=last.
                if req_frame_index >= 0:
                    req_frame_name = req_frame_name_cfg
                else:
                    req_frame_name = req_frame_name_cfg or str(deform_ctx["frame_name"])
                verts_np, deform_joints_cam_np, deform_meta = _load_deform_smplx_vertices_from_npz(
                    params_npz_path=params_npz_path,
                    frame_index=req_frame_index,
                    frame_name=req_frame_name,
                    model_type_default=deform_smplx_model_type,
                    gender_default=deform_smplx_gender,
                    model_root_cfg=deform_smplx_model_root,
                    project_root=project_root,
                    assets_dir=project_root / Path("assets"),
                    ref_vertices_npz_path=bind_vertices_npz,
                    use_standard_pose=deform_smplx_use_standard_pose,
                )
                if bind_rigid_align_apply_cfg is not None:
                    verts_np = apply_rigid_transform_np(np.asarray(verts_np, dtype=np.float32), bind_rigid_align_apply_cfg)
                verts_np_bind = (np.asarray(verts_np, dtype=np.float32) + bind_mesh_offset_np).astype(np.float32)
                if bind_faces_np is not None:
                    _faces_compatible_with_verts(verts_np_bind, "deform(smplx_params)")
                verts_deform = torch.from_numpy(verts_np_bind).to(device=device, dtype=torch.float32)
                logger.info(
                    "Deform render source=%s row=%s model=%s/%s sign=%s",
                    str(deform_meta.get("source", "smplx_rebuild")),
                    str(deform_meta.get("row_idx", "?")),
                    str(deform_meta.get("model_type", "?")),
                    str(deform_meta.get("gender", "?")),
                    str(deform_meta.get("space_match_sign", "[1,1,1]")),
                )

            if verts_deform is None:
                if deform_verts_cam_np is not None:
                    verts_np = np.asarray(deform_verts_cam_np, dtype=np.float32).copy()
                else:
                    deform_mesh_cfg = _resolve_deform_mesh_cfg(render_cfg, ckpt, logger)
                    deform_mesh_obj = Path(resolve_path(str(deform_mesh_cfg)))
                    if not deform_mesh_obj.is_absolute():
                        deform_mesh_obj = project_root / deform_mesh_obj
                    if (not str(deform_mesh_cfg).strip()) or (not deform_mesh_obj.exists()):
                        fallback_mesh = _find_deform_mesh_fallback(project_root / "outputs/stage1")
                        if fallback_mesh is not None:
                            deform_mesh_obj = fallback_mesh
                    if not deform_mesh_obj.exists():
                        raise FileNotFoundError(f"Deform mesh not found: {deform_mesh_obj}")

                    verts_np = load_obj_vertices(deform_mesh_obj)
                    if bind_mesh_to_camera and (not bool(deform_mesh_obj_in_camera_space)):
                        if bind_inverse_export_upright:
                            if exp_has_center_xy:
                                verts_np[:, 0] += float(exp_center_xy[0])
                                verts_np[:, 1] += float(exp_center_xy[1])
                            if exp_has_floor:
                                verts_np[:, 2] += float(exp_floor_z)
                            verts_np = (verts_np @ exp_upright_rot.astype(np.float32))
                        verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
                if bind_rigid_align_apply_cfg is not None:
                    verts_np = apply_rigid_transform_np(np.asarray(verts_np, dtype=np.float32), bind_rigid_align_apply_cfg)
                verts_np_bind = (np.asarray(verts_np, dtype=np.float32) + bind_mesh_offset_np).astype(np.float32)
                if bind_faces_np is not None:
                    _faces_compatible_with_verts(verts_np_bind, "deform")
                verts_deform = torch.from_numpy(verts_np_bind).to(device=device, dtype=torch.float32)

            g_deform = _clone_gaussians(g)
            bind_mask = bind_idx_t >= 0
            if bind_mask.any():
                if str(binding_cfg.mode).lower() == "legacy":
                    bind_idx_use = bind_idx_t[bind_mask]
                    valid_local = bind_idx_use < int(verts_deform.shape[0])
                    if bool(valid_local.all()):
                        xyz_new = verts_deform.index_select(0, bind_idx_use)
                        if bind_vertex_offset_t is not None:
                            xyz_new = xyz_new + bind_vertex_offset_t[bind_mask]
                        g_deform.xyz[bind_mask] = xyz_new
                    else:
                        valid_mask = bind_mask.clone()
                        valid_mask[bind_mask] = valid_local
                        if not bool(valid_mask.any()):
                            raise RuntimeError(
                                "Deform binding failed: all bind indices are out of range for current mesh topology. "
                                "Please rerun stage1 and stage2 to refresh bind topology."
                            )
                        bad_n = int((~valid_local).sum().item())
                        if strict_binding_topology:
                            raise RuntimeError(
                                "Deform binding found out-of-range bind indices "
                                f"({bad_n} invalid). Strict mode is enabled."
                            )
                        logger.warning(
                                "Deform render skipped %d out-of-range bind indices. "
                                "Retrain stage2 for full topology alignment.",
                                bad_n,
                            )
                        valid_idx = bind_idx_t[valid_mask]
                        xyz_new = verts_deform.index_select(0, valid_idx)
                        if bind_vertex_offset_t is not None:
                            xyz_new = xyz_new + bind_vertex_offset_t[valid_mask]
                        g_deform.xyz[valid_mask] = xyz_new
                    if bool(bind_align_quat_to_mesh_normals) and bind_faces_np is not None and bind_quat_normal_blend > 0.0:
                        normals_tp_np = _compute_vertex_normals_np(
                            verts_deform.detach().cpu().numpy(),
                            bind_faces_np,
                        )
                        normals_tp = torch.from_numpy(normals_tp_np).to(device=device, dtype=torch.float32)
                        bind_mask_local = bind_idx_t >= 0
                        if bool(bind_mask_local.any()):
                            bind_idx_use = bind_idx_t[bind_mask_local]
                            valid_local = bind_idx_use < int(normals_tp.shape[0])
                            valid_mask = bind_mask_local.clone()
                            valid_mask[bind_mask_local] = valid_local
                            if bool(valid_mask.any()):
                                q_target = _quat_from_normals_torch(normals_tp.index_select(0, bind_idx_t[valid_mask]))
                                with torch.no_grad():
                                    if bind_quat_normal_blend >= 0.999:
                                        g_deform.quat[valid_mask] = q_target
                                    else:
                                        q_old = g_deform.quat[valid_mask]
                                        q_mix = (1.0 - bind_quat_normal_blend) * q_old + bind_quat_normal_blend * q_target
                                        g_deform.quat[valid_mask] = F.normalize(q_mix, dim=1, eps=1e-8)
                else:
                    deform_adjacency = None
                    if bind_faces_np is not None:
                        if mesh_runtime.bind_adjacency is None or len(mesh_runtime.bind_adjacency) != int(verts_np_bind.shape[0]):
                            mesh_runtime.bind_adjacency = build_vertex_adjacency(int(verts_np_bind.shape[0]), bind_faces_np)
                        deform_adjacency = mesh_runtime.bind_adjacency
                    deform_frame = build_binding_frame_data(
                        verts_np=verts_np_bind,
                        faces_np=bind_faces_np,
                        adjacency=deform_adjacency,
                        cfg=binding_cfg,
                        device=device,
                    )
                    _, valid_inner = apply_inner_binding_per_frame(
                        g=g_deform,
                        frame=deform_frame,
                        bind_mask=bind_mask,
                        bind_vertex_idx=bind_idx_t,
                        bind_vertex_offset=bind_vertex_offset_t,
                        bind_face_idx=bind_face_idx_t,
                        bind_bary=bind_bary_t,
                        cfg=binding_cfg,
                        quat_blend=bind_quat_normal_blend,
                        prev_vertex_dirs=prev_vertex_dirs,
                    )
                    _, valid_outer = apply_outer_anchor_per_frame(
                        g=g_deform,
                        frame=deform_frame,
                        bind_mask=bind_mask,
                        bind_vertex_idx=bind_idx_t,
                        bind_vertex_offset=bind_vertex_offset_t,
                        bind_face_idx=bind_face_idx_t,
                        bind_bary=bind_bary_t,
                        cfg=binding_cfg,
                        quat_blend=bind_quat_normal_blend,
                        prev_vertex_dirs=prev_vertex_dirs,
                    )
                    valid_mask = (valid_inner | valid_outer)
                    if valid_mask is not None and not bool(valid_mask.all()):
                        bad_n = int((bind_mask & (~valid_mask)).sum().item())
                        if bad_n > 0:
                            if strict_binding_topology:
                                raise RuntimeError(
                                    "Deform render found invalid bindings for current topology "
                                    f"({bad_n} invalid). Strict mode is enabled."
                                )
                            logger.warning(
                                "Deform render skipped %d invalid bindings for current topology.",
                                bad_n,
                            )

            g_def_render = g_deform
            if deform_render_bound_only:
                tp_fg_mask = (g_deform.bind_idx >= 0) if g_deform.bind_idx is not None else None
                if tp_fg_mask is None or not tp_fg_mask.any():
                    raise RuntimeError(
                        "Deform render requires bound gaussians when deform_render_bound_only=true."
                    )
                g_def_render = _select_gaussians(g_deform, tp_fg_mask)
                if bound_opacity_min > 0.0:
                    opa = torch.sigmoid(g_def_render.opacity_logit).squeeze(-1)
                    keep = opa >= float(bound_opacity_min)
                    if keep.any():
                        g_def_render = _select_gaussians(g_def_render, keep)
                    else:
                        raise RuntimeError(
                            "Deform render with bound_opacity_min removed all gaussians. "
                            "Lower render.bound_opacity_min."
                        )

            K_def = deform_ctx["K_use"]
            R_def = deform_ctx["R"]
            t_def = deform_ctx["t"]
            H_def = int(deform_ctx["H"])
            W_def = int(deform_ctx["W"])

            if center_subject:
                tp_center_world = None
                joints_center_src = deform_joints_cam_np if deform_joints_cam_np is not None else deform_default_joints_cam_np
                if center_subject_joint_idx >= 0 and joints_center_src is not None:
                    joints_tp_center = np.asarray(joints_center_src, dtype=np.float32).copy()
                    if joints_tp_center.ndim == 2 and joints_tp_center.shape[1] == 3 and joints_tp_center.shape[0] > 0:
                        if bind_rigid_align_apply_cfg is not None:
                            joints_tp_center = apply_rigid_transform_np(joints_tp_center, bind_rigid_align_apply_cfg)
                        joints_tp_center = joints_tp_center + bind_mesh_offset_np
                        j_idx_tp = int(np.clip(center_subject_joint_idx, 0, joints_tp_center.shape[0] - 1))
                        tp_center_world = torch.from_numpy(joints_tp_center[j_idx_tp]).to(device=device, dtype=torch.float32)
                if tp_center_world is None:
                    tp_center_world = verts_deform.mean(dim=0)
                xc_def = (R_def @ tp_center_world.view(3, 1)).view(3) + t_def
                zc_def = torch.clamp(xc_def[2], min=float(center_subject_depth_min))
                fx_def = K_def[0, 0]
                fy_def = K_def[1, 1]
                cx_def = K_def[0, 2]
                cy_def = K_def[1, 2]
                u_tgt_def = torch.tensor((W_def - 1) * 0.5, device=device, dtype=torch.float32)
                v_tgt_def = torch.tensor((H_def - 1) * 0.5, device=device, dtype=torch.float32)
                x_tgt_def = (u_tgt_def - cx_def) * zc_def / fx_def
                y_tgt_def = (v_tgt_def - cy_def) * zc_def / fy_def
                t_def = t_def + torch.stack(
                    [x_tgt_def - xc_def[0], y_tgt_def - xc_def[1], torch.zeros((), device=device, dtype=torch.float32)],
                    dim=0,
                )

            if center_subject and subject_fit_enabled and (verts_deform is not None) and (verts_deform.numel() > 0):
                K_def_np, t_def_np = _fit_subject_camera_numpy(
                    verts=verts_deform.detach().cpu().numpy().astype(np.float32),
                    K=K_def.detach().cpu().numpy().astype(np.float32),
                    R=R_def.detach().cpu().numpy().astype(np.float32),
                    t=t_def.detach().cpu().numpy().astype(np.float32),
                    H=H_def,
                    W=W_def,
                    fill=float(subject_fit_fill),
                    iters=int(subject_fit_iters),
                )
                K_def = torch.from_numpy(K_def_np).to(device=device, dtype=torch.float32)
                t_def = torch.from_numpy(t_def_np).to(device=device, dtype=torch.float32)

            rgb_def, alpha_def = render_gaussians_3d(
                g=g_def_render,
                K=K_def,
                R=R_def,
                t=t_def,
                H=H_def,
                W=W_def,
                bg=None,
                return_alpha=True,
            )
            deform_mask = _alpha_to_mask(
                alpha_def,
                threshold=deform_alpha_threshold,
                gamma=deform_alpha_gamma,
            )
            if fg_mask_mode in ("smpl", "mesh") and bind_faces_np is not None:
                mask_def_np = _rasterize_mesh_mask(
                    verts=verts_deform.detach().cpu().numpy(),
                    faces=bind_faces_np,
                    K=K_def.detach().cpu().numpy().astype(np.float32),
                    R=R_def.detach().cpu().numpy().astype(np.float32),
                    t=t_def.detach().cpu().numpy().astype(np.float32),
                    H=H_def,
                    W=W_def,
                    dilate_px=int(fg_mask_dilate_px),
                    erode_px=int(fg_mask_erode_px),
                )
                deform_mask_mesh = torch.from_numpy(mask_def_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                deform_mask = deform_mask * deform_mask_mesh
            alpha_def_src = alpha_def.clamp(min=1e-4)
            gain_def = (deform_mask / alpha_def_src).clamp(min=0.0, max=float(deform_alpha_gain_max))
            rgb_def_adj = (rgb_def * gain_def).clamp(0.0, 1.0)
            bg_fill_def = torch.tensor(bg, dtype=torch.float32, device=device).view(3, 1, 1).expand_as(rgb_def)
            pred_deform = _composite_premultiplied(fg_rgb=rgb_def_adj, fg_alpha=deform_mask, bg_rgb=bg_fill_def)
            if overlay_smpl_skeleton:
                frame_name = str(deform_ctx["frame_name"])
                joints_raw = deform_joints_cam_np if deform_joints_cam_np is not None else deform_default_joints_cam_np
                joints_deform_use = _normalize_smplx_body_joints(joints_raw)
                if joints_deform_use is None and smpl_joints_map is not None:
                    joints_deform_use = _normalize_smplx_body_joints(smpl_joints_map.get(frame_name, None))
                if joints_deform_use is not None:
                    joints_deform_use = np.asarray(joints_deform_use, dtype=np.float32).copy()
                    if bind_rigid_align_apply_cfg is not None:
                        joints_deform_use = apply_rigid_transform_np(joints_deform_use, bind_rigid_align_apply_cfg)
                    joints_deform_use = joints_deform_use + bind_mesh_offset_np
                pred_deform = _overlay_smpl_skeleton(
                    rgb_chw=pred_deform,
                    joints_cam=joints_deform_use,
                    K=K_def,
                    R=R_def,
                    t=t_def,
                    connections=smpl_skeleton_connections,
                    color_bgr=smpl_color_bgr,
                    thickness=smpl_line_thickness,
                    radius=smpl_joint_radius,
                )
            deform_name = f"{Path(str(deform_ctx['frame_name'])).stem}{file_suffix}.png"
            if layer_split_output:
                deform_layers = _split_gaussians_by_shell_layer(g_def_render)
                for layer_key in active_layer_keys:
                    if layer_key == "combined":
                        pred_layer = pred_deform
                    else:
                        pred_layer = _render_layer_preview(
                            g_layer=deform_layers.get(layer_key, None),
                            K=K_def,
                            R=R_def,
                            t=t_def,
                            H=H_def,
                            W=W_def,
                            bg=bg,
                            bg_gt_chw=None,
                            bg_gt_hwc=None,
                            use_gt_bg=False,
                        )
                    save_rgb(deform_layer_dirs[layer_key] / deform_name, pred_layer)
            else:
                save_rgb(deform_out_dir / deform_name, pred_deform)

    logger.info("Done.")


if __name__ == "__main__":
    main()
