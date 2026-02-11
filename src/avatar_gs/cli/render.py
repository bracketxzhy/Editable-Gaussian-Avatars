from __future__ import annotations

from pathlib import Path
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, resolve_path
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices, load_obj_mesh
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


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


def save_rgb(path: str | Path, rgb_chw: torch.Tensor) -> None:
    """Save (3,H,W) RGB float [0,1] as png/jpg, preserving exact H/W."""
    img = (rgb_chw.detach().clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
    bgr = img[..., ::-1]  # BGR
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


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
    if draw_skeleton:
        for a, b in MP_POSE_CONNECTIONS:
            if a >= len(arr) or b >= len(arr):
                continue
            pa = arr[a]
            pb = arr[b]
            if float(pa[2]) < float(conf_min) or float(pb[2]) < float(conf_min):
                continue
            xa = int(round(float(pa[0])))
            ya = int(round(float(pa[1])))
            xb = int(round(float(pb[0])))
            yb = int(round(float(pb[1])))
            if xa < 0 or xa >= W or ya < 0 or ya >= H:
                continue
            if xb < 0 or xb >= W or yb < 0 or yb >= H:
                continue
            cv2.line(bgr, (xa, ya), (xb, yb), (0, 180, 0), thickness=2, lineType=cv2.LINE_AA)

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
    prefer_theta: bool = True,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]], str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"SMPL skeleton sequence not found: {npz_path}")
    seq = np.load(npz_path, allow_pickle=True)
    if "frame_names" not in seq.files:
        raise RuntimeError(f"{npz_path} missing frame_names")

    has_theta = "joints_cam_theta" in seq.files
    has_joints = "joints_cam" in seq.files
    if not has_theta and not has_joints:
        raise RuntimeError(f"{npz_path} missing joints_cam_theta/joints_cam")

    joints_key = "joints_cam_theta" if (prefer_theta and has_theta) else ("joints_cam" if has_joints else "joints_cam_theta")
    frame_names = [str(x) for x in seq["frame_names"].tolist()]
    joints_cam = np.asarray(seq[joints_key], dtype=np.float32)
    if joints_cam.ndim != 3 or joints_cam.shape[0] != len(frame_names) or joints_cam.shape[2] != 3:
        raise RuntimeError(f"Bad joints shape in {npz_path}: {joints_cam.shape}")

    joint_parents = np.asarray(seq["joint_parents"], dtype=np.int32) if "joint_parents" in seq.files else None
    connections = _build_parent_connections(joint_parents=joint_parents, joint_count=int(joints_cam.shape[1]))
    if len(connections) == 0:
        connections = list(SMPL_SKELETON_CONNECTIONS)
    return {fn: joints_cam[i] for i, fn in enumerate(frame_names)}, connections, joints_key


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
    if joints_cam is None:
        return rgb_chw
    if joints_cam.ndim != 2 or joints_cam.shape[1] != 3:
        return rgb_chw

    img = (rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    bgr = img[..., ::-1].copy()
    H, W = bgr.shape[:2]

    K_np = K.detach().cpu().numpy().astype(np.float32)
    R_np = R.detach().cpu().numpy().astype(np.float32)
    t_np = t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
    Xc = joints_cam.astype(np.float32) @ R_np.T + t_np
    z = Xc[:, 2]
    valid = z > 1e-6
    u = np.zeros((joints_cam.shape[0],), dtype=np.float32)
    v = np.zeros((joints_cam.shape[0],), dtype=np.float32)
    if np.any(valid):
        Xcv = Xc[valid]
        zv = z[valid]
        u[valid] = K_np[0, 0] * (Xcv[:, 0] / zv) + K_np[0, 2]
        v[valid] = K_np[1, 1] * (Xcv[:, 1] / zv) + K_np[1, 2]

    rr = max(1, int(radius))
    th = max(1, int(thickness))
    draw_connections = connections if (connections is not None and len(connections) > 0) else SMPL_SKELETON_CONNECTIONS
    for a, b in draw_connections:
        if a >= len(u) or b >= len(u):
            continue
        if not (valid[a] and valid[b]):
            continue
        xa = int(round(float(u[a])))
        ya = int(round(float(v[a])))
        xb = int(round(float(u[b])))
        yb = int(round(float(v[b])))
        if xa < 0 or xa >= W or ya < 0 or ya >= H:
            continue
        if xb < 0 or xb >= W or yb < 0 or yb >= H:
            continue
        cv2.line(bgr, (xa, ya), (xb, yb), color_bgr, thickness=th, lineType=cv2.LINE_AA)
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


def _project_points_numpy(
    xyz: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc = xyz @ R.T + t.reshape(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = np.zeros((xyz.shape[0],), dtype=np.float32)
    v = np.zeros((xyz.shape[0],), dtype=np.float32)
    if np.any(valid):
        Xcv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (Xcv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (Xcv[:, 1] / zv) + K[1, 2]
    return u, v, valid


def _rasterize_mesh_mask(
    verts: np.ndarray,
    faces: np.ndarray | None,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if faces is None or len(faces) == 0 or verts.shape[0] == 0:
        return mask
    u, v, valid = _project_points_numpy(verts, K, R, t)
    if not np.any(valid):
        return mask
    uv = np.stack([u, v], axis=1).astype(np.float32)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if i0 < 0 or i1 < 0 or i2 < 0:
            continue
        if i0 >= len(valid) or i1 >= len(valid) or i2 >= len(valid):
            continue
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        pts = uv[[i0, i1, i2]]
        if (
            (pts[:, 0] < -2).all()
            or (pts[:, 0] > W + 1).all()
            or (pts[:, 1] < -2).all()
            or (pts[:, 1] > H + 1).all()
        ):
            continue
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(mask, poly, 1)
    if int(dilate_px) > 0:
        k = int(dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if int(erode_px) > 0:
        k = int(erode_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


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


def _select_gaussians(g: Gaussian3D, mask: torch.Tensor) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz[mask],
        log_scales=g.log_scales[mask],
        quat=g.quat[mask],
        opacity_logit=g.opacity_logit[mask],
        rgb_logit=g.rgb_logit[mask],
        bind_idx=g.bind_idx[mask] if g.bind_idx is not None else None,
        bind_offset=g.bind_offset[mask] if g.bind_offset is not None else None,
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
    out_dir = ensure_dir(project_root / Path(render_cfg.get("out_dir", "outputs/renders")))
    if bool(render_cfg.get("clear_out_dir_pngs", False)):
        removed = 0
        for old in out_dir.glob("*.png"):
            old.unlink()
            removed += 1
        for old in out_dir.glob("*.jpg"):
            old.unlink()
            removed += 1
        for old in out_dir.glob("*.jpeg"):
            old.unlink()
            removed += 1
        if removed > 0:
            logger.info(f"Cleared {removed} existing render image(s) in {out_dir}")

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
    )

    # Intrinsics
    K = torch.as_tensor(ckpt["K"], dtype=torch.float32, device=device).view(3, 3)

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

    fg_scale = float(render_cfg.get("fg_scale", 1.0))
    bg_scale = float(render_cfg.get("bg_scale", 1.0))
    k_scale = float(render_cfg.get("k_scale", 1.0))
    split_by_bind = bool(render_cfg.get("split_by_bind", False))
    render_bound_only = bool(render_cfg.get("render_bound_only", False))
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
    smpl_skeleton_source = str(render_cfg.get("smpl_skeleton_source", "theta")).strip().lower()
    if smpl_skeleton_source not in ("theta", "joints"):
        raise ValueError(
            f"render.smpl_skeleton_source must be one of: theta|joints. Got: {smpl_skeleton_source}"
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
    if fg_mask_mode not in ("none", "smpl", "mesh"):
        raise ValueError(f"Unsupported fg_mask_mode: {fg_mask_mode}. Use one of: none|smpl|mesh")
    if fg_mask_dilate_px < 0 or fg_mask_erode_px < 0:
        raise ValueError(
            f"render.fg_mask_dilate_px/render.fg_mask_erode_px must be >=0. "
            f"Got {fg_mask_dilate_px}/{fg_mask_erode_px}"
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
            if smpl_skeleton_source == "theta":
                smpl_skeleton_npz_cfg = "outputs/stage1/smpl_skeleton_sequence.npz"
            elif smpl_joints_npz_cfg is not None and str(smpl_joints_npz_cfg).strip():
                smpl_skeleton_npz_cfg = smpl_joints_npz_cfg
            else:
                smpl_skeleton_npz_cfg = "outputs/stage1/smpl_joints_sequence.npz"
        smpl_skeleton_path = Path(resolve_path(str(smpl_skeleton_npz_cfg)))
        if not smpl_skeleton_path.is_absolute():
            smpl_skeleton_path = project_root / smpl_skeleton_path
        smpl_joints_map, smpl_skeleton_connections, loaded_joint_key = _load_smpl_skeleton_map(
            smpl_skeleton_path,
            prefer_theta=bool(smpl_skeleton_source == "theta"),
        )
        logger.info(
            f"Loaded SMPL skeleton sequence: {smpl_skeleton_path} "
            f"(source={loaded_joint_key}, edges={len(smpl_skeleton_connections)})"
        )

    # Optional: dynamic binding to SMPL mesh sequence
    bind_mesh_dir = render_cfg.get("bind_mesh_sequence_dir", None)
    bind_seq_npz = render_cfg.get("bind_sequence_npz", None)
    bind_vertices_npz = render_cfg.get("bind_vertices_npz", ckpt.get("bind_vertices_npz", None))
    bind_mesh_indices = ckpt.get("bind_mesh_indices", None)
    if bind_mesh_indices is not None:
        bind_mesh_indices = [int(x) for x in bind_mesh_indices]
        if len(bind_mesh_indices) == total_frames:
            bind_mesh_indices = [bind_mesh_indices[i] for i in selected_idx]
        elif len(bind_mesh_indices) != to_render:
            logger.warning(
                "bind_mesh_indices length does not match selected frames; "
                "will remap by frame names from bind_sequence_npz."
            )
            bind_mesh_indices = None
    binding_enabled = bind_mesh_dir is not None and g.bind_idx is not None
    if fg_mask_mode in ("smpl", "mesh") and not binding_enabled:
        raise ValueError("fg_mask_mode=smpl requires bind_mesh_sequence_dir and bound gaussians.")
    bind_mesh_to_camera = bool(render_cfg.get("bind_mesh_to_camera", False))
    bind_inverse_export_upright = bool(render_cfg.get("bind_inverse_export_upright", True))
    legacy_cam_t_autofix = bool(render_cfg.get("legacy_cam_t_autofix", True))
    bind_mesh_offset_cfg = render_cfg.get("bind_mesh_offset", [0.0, 0.0, 0.0])
    if not (isinstance(bind_mesh_offset_cfg, (list, tuple)) and len(bind_mesh_offset_cfg) == 3):
        raise ValueError("render.bind_mesh_offset must be a 3-value list/tuple, e.g. [0, 0, 0].")
    bind_mesh_offset_t = torch.tensor(
        [float(bind_mesh_offset_cfg[0]), float(bind_mesh_offset_cfg[1]), float(bind_mesh_offset_cfg[2])],
        dtype=torch.float32,
        device=device,
    ).view(1, 3)
    if bind_seq_npz is not None:
        bind_seq_npz = Path(resolve_path(str(bind_seq_npz)))
        if not bind_seq_npz.is_absolute():
            bind_seq_npz = project_root / bind_seq_npz
    if bind_vertices_npz is not None:
        bind_vertices_npz = Path(resolve_path(str(bind_vertices_npz)))
        if not bind_vertices_npz.is_absolute():
            bind_vertices_npz = project_root / bind_vertices_npz
    if bind_vertices_npz is None and binding_enabled:
        cand_vseq = project_root / Path("outputs/stage1/smpl_vertices_sequence.npz")
        if cand_vseq.exists():
            bind_vertices_npz = cand_vseq

    mesh_cache: dict[int, torch.Tensor] = {}
    mesh_cache_order: list[int] = []
    bind_idx_t = None
    bind_vertex_offset_t = None
    bind_faces_np = None
    exp_upright_rot = np.eye(3, dtype=np.float32)
    exp_center_xy = np.zeros((2,), dtype=np.float32)
    exp_floor_z = 0.0
    exp_has_center_xy = False
    exp_has_floor = False
    if binding_enabled:
        bind_mesh_dir = Path(resolve_path(str(bind_mesh_dir)))
        if not bind_mesh_dir.is_absolute():
            bind_mesh_dir = project_root / bind_mesh_dir
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(f"bind_mesh_sequence_dir not found: {bind_mesh_dir}")

        if bind_mesh_indices is None:
            # try to map frame names -> mesh indices using smpl_params_sequence.npz
            if bind_seq_npz is None:
                bind_seq_npz = project_root / Path("outputs/stage1/smpl_params_sequence.npz")
            else:
                bind_seq_npz = Path(resolve_path(str(bind_seq_npz)))
                if not bind_seq_npz.is_absolute():
                    bind_seq_npz = project_root / bind_seq_npz
            if bind_seq_npz.exists():
                import numpy as _np
                seq = _np.load(bind_seq_npz, allow_pickle=True)
                seq_names = [str(x) for x in seq["frame_names"].tolist()]
                name_to_idx = {n: i for i, n in enumerate(seq_names)}
                bind_mesh_indices = []
                missing = []
                for fn in frames:
                    if fn in name_to_idx:
                        bind_mesh_indices.append(int(name_to_idx[fn]))
                    else:
                        missing.append(fn)
                if missing and len(seq_names) == len(frames):
                    bind_mesh_indices = list(range(len(frames)))
                elif missing:
                    raise RuntimeError(f"Binding failed: {len(missing)} frames not in {bind_seq_npz}")
            else:
                bind_mesh_indices = list(range(len(frames)))

        verts_seq_np = None
        verts_seq_ids_set: set[int] = set()
        verts_seq_fallback_idx = 0
        if bind_vertices_npz is not None and bind_vertices_npz.exists():
            vseq = np.load(bind_vertices_npz, allow_pickle=True)
            if "verts_cam" in vseq.files:
                verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)
                if verts_seq_np.ndim != 3 or verts_seq_np.shape[-1] != 3:
                    raise RuntimeError(
                        f"Bad verts_cam shape in {bind_vertices_npz}: {verts_seq_np.shape}"
                    )
                verts_seq_ids_set = set(range(int(verts_seq_np.shape[0])))
                verts_seq_fallback_idx = int(max(int(verts_seq_np.shape[0]) - 1, 0))
                logger.info(f"Using dynamic bind vertices from {bind_vertices_npz}")
            else:
                logger.warning(
                    f"{bind_vertices_npz} missing verts_cam; fallback to OBJ sequence binding."
                )

        mesh_ids = []
        for p in bind_mesh_dir.glob("*.obj"):
            stem = p.stem
            if stem.isdigit():
                mesh_ids.append(int(stem))
        if len(mesh_ids) == 0:
            raise FileNotFoundError(f"No numeric OBJ meshes found in {bind_mesh_dir}")
        mesh_ids_set = set(mesh_ids)
        fallback_mesh_idx = int(max(mesh_ids))
        if verts_seq_np is not None:
            missing_seq_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in verts_seq_ids_set]
            if len(missing_seq_ids) > 0:
                logger.warning(
                    "Some bind indices are missing in bind_vertices_npz; "
                    f"fallback to frame {verts_seq_fallback_idx} for {len(missing_seq_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in verts_seq_ids_set else verts_seq_fallback_idx
                    for mi in bind_mesh_indices
                ]
        else:
            missing_mesh_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in mesh_ids_set]
            if len(missing_mesh_ids) > 0:
                logger.warning(
                    "Some bind mesh indices are missing from bind_mesh_sequence_dir; "
                    f"fallback to mesh {fallback_mesh_idx:06d}.obj for {len(missing_mesh_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in mesh_ids_set else fallback_mesh_idx
                    for mi in bind_mesh_indices
                ]

        if fg_mask_mode in ("smpl", "mesh"):
            if len(bind_mesh_indices) == 0:
                raise RuntimeError("fg_mask_mode=smpl requires non-empty bind_mesh_indices.")
            first_mesh_idx = int(bind_mesh_indices[0]) if int(bind_mesh_indices[0]) in mesh_ids_set else fallback_mesh_idx
            first_obj = bind_mesh_dir / f"{first_mesh_idx:06d}.obj"
            if not first_obj.exists():
                raise FileNotFoundError(f"Mesh frame not found for fg mask: {first_obj}")
            _, bind_faces_np = load_obj_mesh(first_obj)
            if bind_faces_np is None or len(bind_faces_np) == 0:
                raise RuntimeError("fg_mask_mode=smpl requires mesh faces, but loaded mesh has no faces.")

        if bind_mesh_to_camera and bind_inverse_export_upright and bind_seq_npz is not None and bind_seq_npz.exists():
            seq_meta = np.load(bind_seq_npz, allow_pickle=True)
            if "export_upright_rot" in seq_meta.files:
                exp_upright_rot = np.asarray(seq_meta["export_upright_rot"], dtype=np.float32).reshape(3, 3)
            if "export_center_xy" in seq_meta.files:
                exp_has_center_xy = bool(int(np.asarray(seq_meta["export_center_xy"]).reshape(-1)[0]))
            if "export_center_xy_value" in seq_meta.files:
                exp_center_xy = np.asarray(seq_meta["export_center_xy_value"], dtype=np.float32).reshape(2,)
            if "export_on_floor" in seq_meta.files:
                exp_has_floor = bool(int(np.asarray(seq_meta["export_on_floor"]).reshape(-1)[0]))
            if "export_floor_z" in seq_meta.files:
                exp_floor_z = float(np.asarray(seq_meta["export_floor_z"], dtype=np.float32).reshape(-1)[0])

        bind_idx_t = g.bind_idx.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0
        if (
            legacy_cam_t_autofix
            and bind_mesh_to_camera
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

        def _load_mesh_verts(mesh_idx: int) -> torch.Tensor:
            if verts_seq_np is not None:
                mesh_idx_use = int(mesh_idx)
                if mesh_idx_use not in verts_seq_ids_set:
                    mesh_idx_use = verts_seq_fallback_idx
                if mesh_idx_use in mesh_cache:
                    return mesh_cache[mesh_idx_use]
                verts_np = verts_seq_np[mesh_idx_use]
                verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
                verts_t = verts_t + bind_mesh_offset_t
                mesh_cache[mesh_idx_use] = verts_t
                mesh_cache_order.append(mesh_idx_use)
                if len(mesh_cache_order) > 8:
                    old = mesh_cache_order.pop(0)
                    mesh_cache.pop(old, None)
                return verts_t

            mesh_idx_use = int(mesh_idx)
            if mesh_idx_use not in mesh_ids_set:
                mesh_idx_use = fallback_mesh_idx
            if mesh_idx_use in mesh_cache:
                return mesh_cache[mesh_idx_use]
            obj_path = bind_mesh_dir / f"{mesh_idx_use:06d}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(f"Mesh frame not found: {obj_path}")
            verts_np = load_obj_vertices(obj_path)
            if bind_mesh_to_camera:
                if bind_inverse_export_upright:
                    if exp_has_center_xy:
                        verts_np[:, 0] += float(exp_center_xy[0])
                        verts_np[:, 1] += float(exp_center_xy[1])
                    if exp_has_floor:
                        verts_np[:, 2] += float(exp_floor_z)
                    # Inverse of stage1 upright rotation: v_blender_raw = v_export @ U
                    verts_np = (verts_np @ exp_upright_rot.astype(np.float32))
                # Inverse of stage1 _camera_to_blender: (x, y, z) = (x_b, -z_b, y_b)
                import numpy as _np
                verts_np = _np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(_np.float32)
            verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
            verts_t = verts_t + bind_mesh_offset_t
            mesh_cache[mesh_idx_use] = verts_t
            mesh_cache_order.append(mesh_idx_use)
            if len(mesh_cache_order) > 8:
                old = mesh_cache_order.pop(0)
                mesh_cache.pop(old, None)
            return verts_t

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
        verts_t = None

        # Update bound xyz for this frame (dynamic binding)
        if binding_enabled:
            mesh_idx = int(bind_mesh_indices[i])
            verts_t = _load_mesh_verts(mesh_idx)
            if bind_mask.any():
                xyz_new = verts_t.index_select(0, bind_idx_t[bind_mask])
                if bind_vertex_offset_t is not None:
                    xyz_new = xyz_new + bind_vertex_offset_t[bind_mask]
                g.xyz[bind_mask] = xyz_new.detach()

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
        K_use = _scale_intrinsics(K, k_scale)

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
                pred = fg_rgb * fg_alpha + bg_rgb * (1.0 - fg_alpha)

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

        # Overlay 2D keypoints/skeleton on top for both debug and normal rendering.
        if (not debug_point_render) and debug_overlay_keypoints and debug_kpt_json_path is not None:
            pred = _overlay_keypoints(
                rgb_chw=pred,
                keypoints_json=debug_kpt_json_path,
                frame_name=frames[i],
                conf_min=debug_kpt_conf_min,
                radius=debug_kpt_radius,
                draw_skeleton=debug_draw_skeleton,
                body_only=debug_overlay_body_only,
            )
        if overlay_smpl_skeleton and smpl_joints_map is not None:
            pred = _overlay_smpl_skeleton(
                rgb_chw=pred,
                joints_cam=smpl_joints_map.get(frames[i], None),
                K=K_use,
                R=R,
                t=t,
                connections=smpl_skeleton_connections,
                color_bgr=smpl_color_bgr,
                thickness=smpl_line_thickness,
                radius=smpl_joint_radius,
            )

        out_path = out_dir / f"{i:06d}.png"
        save_rgb(out_path, pred)

        if (i + 1) % 50 == 0:
            logger.info(f"Rendered {i+1}/{to_render} ...")

    logger.info("Done.")


if __name__ == "__main__":
    main()
