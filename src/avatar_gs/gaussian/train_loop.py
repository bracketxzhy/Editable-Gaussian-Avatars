from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices, load_obj_mesh
from avatar_gs.gaussian.binding import (
    BindingFrameData,
    build_binding_frame_data,
    build_vertex_adjacency,
    export_binding_mode_config,
    parse_binding_mode_config,
    update_gaussians_with_binding,
)
from avatar_gs.gaussian.rigid_align import (
    apply_rigid_transform_np,
    coordinate_descent_rigid_search,
    normalize_rigid_align_config,
)
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.binding.barycentric_binding import (
    apply_barycentric_binding_per_frame,
    init_barycentric_binding,
)
from avatar_gs.losses import (
    canonical_offset_l2,
    compute_robust_photo_loss,
    covariance_penalty,
    layer_alpha_balance_loss,
    opacity_prior_loss,
    outer_color_band_loss,
)
from avatar_gs.training import FrameResidualEMA, frame_weight_from_residual
from avatar_gs.visibility import (
    build_border_attenuation,
    compute_gaussian_visibility,
    sample_border_weight,
)
from avatar_gs.densify import (
    GaussianDensifyStats,
    select_densify_candidates,
    update_densify_stats,
)
from avatar_gs.gaussian.shell_policy import (
    compute_shell_score,
    estimate_outer_budget,
    select_inner_hardprune_candidates,
    select_inner_softfreeze_candidates,
    select_outer_prune_candidates,
    select_outer_spawn_candidates,
)

def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read an image and return float tensor in [0,1], shape (3,H,W)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (3,) axis-angle.
    Fully differentiable. Avoid torch.tensor([...]) with tensor entries (causes warnings).
    """
    theta = torch.norm(w) + 1e-8
    k = w / theta  # (3,)

    kx, ky, kz = k[0], k[1], k[2]
    zeros = torch.zeros((), device=w.device, dtype=w.dtype)

    # Skew-symmetric matrix of k
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky]),
            torch.stack([kz, zeros, -kx]),
            torch.stack([-ky, kx, zeros]),
        ],
        dim=0,
    )

    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    R = I + s * K + (1.0 - c) * (K @ K)
    return R


def _compute_vertex_normals_np(verts: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    if faces is None or len(faces) == 0:
        return np.zeros_like(verts, dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(v, dtype=np.float32)
    tri = v[f]  # (F,3,3)
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    fn = np.cross(e1, e2).astype(np.float32)  # area-weighted face normals
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return (n / norm).astype(np.float32)


def _quat_from_normals_torch(normals: torch.Tensor) -> torch.Tensor:
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
    opp = dot[:, 0] < -0.9999
    if bool(opp.any()):
        q[opp, 0] = 0.0
        q[opp, 1] = 1.0
        q[opp, 2] = 0.0
        q[opp, 3] = 0.0
    return F.normalize(q, dim=1, eps=1e-8)


def _project_points_numpy(
    xyz: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points with camera convention X_cam = R * X + t.
    Returns (u, v, valid_z_mask).
    """
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


def _project_points_torch(
    xyz: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xc = (xyz @ R.transpose(0, 1)) + t.view(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = torch.zeros_like(z)
    v = torch.zeros_like(z)
    if bool(valid.any()):
        xv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (xv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (xv[:, 1] / zv) + K[1, 2]
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
    """
    Rasterize mesh silhouette to a binary mask (H, W) in {0,1}.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if faces is None or len(faces) == 0 or verts.shape[0] == 0:
        return mask

    u, v, valid = _project_points_numpy(verts, K, R, t)
    if not np.any(valid):
        return mask

    uvi = np.stack([u, v], axis=1).astype(np.float32)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if i0 < 0 or i1 < 0 or i2 < 0:
            continue
        if i0 >= len(valid) or i1 >= len(valid) or i2 >= len(valid):
            continue
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        pts = uvi[[i0, i1, i2]]
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


def _erode_binary_mask_torch(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    """
    Erode a binary-like mask tensor (1,H,W) using max-pool based morphology on GPU.
    """
    if int(erode_px) <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    k = int(erode_px) * 2 + 1
    inv = (1.0 - mask).clamp(0.0, 1.0).unsqueeze(0)  # (1,1,H,W)
    dil_inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=int(erode_px))
    out = 1.0 - dil_inv.squeeze(0)
    return out.clamp(0.0, 1.0)


def _dilate_binary_mask_torch(mask: torch.Tensor, dilate_px: int) -> torch.Tensor:
    if int(dilate_px) <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    k = int(dilate_px) * 2 + 1
    x = mask.clamp(0.0, 1.0).unsqueeze(0)
    out = F.max_pool2d(x, kernel_size=k, stride=1, padding=int(dilate_px)).squeeze(0)
    return out.clamp(0.0, 1.0)


def _compute_contour_band_torch(mask: torch.Tensor, width_px: int = 2) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    w = max(int(width_px), 1)
    m = (mask > 0.5).float()
    dil = _dilate_binary_mask_torch(m, w)
    ero = _erode_binary_mask_torch(m, w)
    return (dil - ero).clamp(0.0, 1.0)


def _fscore_binary_torch(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    eval_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    p = (pred_mask > 0.5).float()
    g = (gt_mask > 0.5).float()
    if eval_mask is not None:
        e = (eval_mask > 0.5).float()
        p = p * e
        g = g * e
    tp = (p * g).sum()
    fp = (p * (1.0 - g)).sum()
    fn = ((1.0 - p) * g).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return (2.0 * precision * recall) / (precision + recall + 1e-6)


def train_gaussians_3d_learnable_camera(
    g: Gaussian3D,
    frames_dir: str | Path,
    frames_list: list[str],
    K_list: list[float],
    out_ckpt: str | Path,
    iters: int = 1000,
    lr_gauss: float = 1e-2,
    lr_cam: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cam_warmup_iters: int = 5000,
    l1_weight: float = 0.25,
    use_alpha_mask: bool = True,
    alpha_threshold: float = 0.15,
    camera_mode: str = "learnable",
    cam_t_prior: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_reg_weight: float = 0.0,
    keypoints_json: str | Path | None = None,
    kpt_conf_min: float = 0.2,
    kpt_box_pad: float = 0.15,
    use_kpt_bbox_mask: bool = True,
    face_weight: float = 1.0,
    face_conf_min: float = 0.3,
    face_box_pad: float = 0.2,
    face_use_ears: bool = False,
    frame_weights: list[float] | None = None,
    frame_loss_weights: list[float] | None = None,
    temporal_ref_window: int = 0,
    temporal_ref_decay: float = 0.85,
    temporal_ref_anchor: str = "sample",
    bind_mesh_dir: str | Path | None = None,
    bind_mesh_paths: list[str] | None = None,
    bind_vertex_indices: torch.Tensor | None = None,
    bind_mesh_indices: list[int] | None = None,
    bind_cache_size: int = 8,
    bind_seq_npz: str | Path | None = None,
    bind_vertices_npz: str | Path | None = None,
    bind_deform_obj: str | Path | None = None,
    bind_mesh_to_camera: bool = False,
    bind_inverse_export_upright: bool = True,
    bind_mesh_obj_space: str = "auto",
    bind_align_quat_to_mesh_normals: bool = True,
    bind_quat_normal_blend: float = 1.0,
    bind_surface_aniso_weight: float = 0.0,
    bind_surface_aniso_margin: float = 0.15,
    bind_mesh_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bind_rigid_align: dict | None = None,
    use_smpl_silhouette_mask: bool = False,
    smpl_mask_dilate_px: int = 0,
    smpl_mask_erode_px: int = 0,
    smpl_color_mask_erode_px: int = 0,
    smpl_alpha_leak_weight: float = 0.0,
    smpl_mask_override_alpha: bool = True,
    smpl_mask_precompute: bool = True,
    smpl_mask_precompute_max_frames: int = 64,
    person_masks_npz: str | Path | None = None,
    use_person_mask: bool = False,
    person_mask_dilate_px: int = 0,
    person_mask_erode_px: int = 0,
    person_mask_fuse_mode: str = "intersect",
    separate_fg_bg_training: bool = False,
    fg_region_weight: float = 1.0,
    bg_region_weight: float = 1.0,
    bg_grad_scale: float = 1.0,
    bg_opacity_logit_max: float | None = None,
    bg_log_scale_min: float | None = None,
    bg_log_scale_max: float | None = None,
    fg_opacity_logit_max: float | None = None,
    fg_log_scale_min: float | None = None,
    fg_log_scale_max: float | None = None,
    early_stop_enable: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup_iters: int = 0,
    binding: dict | None = None,
    tau_vis: float = 0.15,
    frame_weight_k: float = 4.0,
    use_robust_loss: bool = True,
    robust_loss_type: str = "charbonnier",
    alpha_prior_weight: float = 0.1,
    cov_min: float = 5e-4,
    cov_max: float = 2.5e-1,
    cov_penalty_weight: float = 0.05,
    use_border_attenuation: bool = True,
    border_attenuation_width: int = 8,
    V_min: int = 8,
    densify_top_percent: float = 10.0,
    phase0_ratio: float = 0.15,
    phase2_ratio: float = 0.15,
    shell_enable: bool = False,
    shell_max_ratio: float = 0.40,
    shell_spawn_top_percent: float = 15.0,
    shell_spawn_every: int = 200,
    shell_prune_every: int = 400,
    shell_offset_max_n: float = 0.04,
    shell_offset_max_t: float = 0.015,
    shell_offset_reg_weight: float = 0.02,
    inner_occlusion_soft_enable: bool = True,
    inner_occlusion_hard_enable: bool = True,
    inner_soft_vis_th: float = 0.05,
    inner_soft_contrib_th: float = 0.01,
    outer_cover_vis_th: float = 0.20,
    outer_cover_contrib_th: float = 0.03,
    inner_soft_patience: int = 300,
    inner_hard_patience: int = 800,
    inner_soft_lr_scale: float = 0.2,
    inner_prune_every: int = 400,
    inner_prune_protect_parts: list[str] | tuple[str, ...] | None = None,
    dual_quality_gate: dict | None = None,
):
    """
    Stage2: optimize 3D Gaussians + (optional) per-frame learnable camera.

    Key fixes vs the original:
    - Use white background compositing to match GT background (reduces colored speckles).
    - Freeze camera for warmup_iters, then enable camera learning (stabilizes convergence).
    - Add L1 term to reduce MSE-induced blur.
    - Random frame sampling (remove periodic bias).
    - Optional alpha-based masking to avoid fitting background.
    - Optional SMPL silhouette masking/leak penalty to keep bound gaussians on the body region.
    - For fixed camera, optional precompute of silhouette masks to reduce per-iteration CPU work.
    """

    frames_dir = Path(frames_dir)
    if len(frames_list) == 0:
        raise ValueError("frames_list is empty.")

    # Intrinsics
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)

    # Load first frame for H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    # Move gaussian to device and make trainable
    g = g.to(device)
    # If binding is enabled, we will override xyz per-frame from mesh; do NOT optimize xyz.
    binding_enabled = bind_mesh_dir is not None and bind_vertex_indices is not None
    if bool(use_smpl_silhouette_mask) and not binding_enabled:
        raise ValueError("use_smpl_silhouette_mask=true requires bind_mesh_dir + bind_vertex_indices.")
    if int(smpl_mask_dilate_px) < 0 or int(smpl_mask_erode_px) < 0 or int(smpl_color_mask_erode_px) < 0:
        raise ValueError(
            "smpl_mask_dilate_px/smpl_mask_erode_px/smpl_color_mask_erode_px must be >= 0. "
            f"Got {smpl_mask_dilate_px}/{smpl_mask_erode_px}/{smpl_color_mask_erode_px}"
        )
    if binding_enabled:
        g.xyz = g.xyz.detach()  # keep as buffer
    else:
        g.xyz = torch.nn.Parameter(g.xyz)
    g.log_scales = torch.nn.Parameter(g.log_scales)
    g.quat = torch.nn.Parameter(g.quat)
    g.opacity_logit = torch.nn.Parameter(g.opacity_logit)
    g.rgb_logit = torch.nn.Parameter(g.rgb_logit)

    # Optional structured fields (kept backward-compatible with old checkpoints)
    n_gauss = int(g.xyz.shape[0])
    if g.bind_face_offset_local is None:
        g.bind_face_offset_local = torch.zeros((n_gauss, 3), dtype=torch.float32, device=device)
    else:
        g.bind_face_offset_local = g.bind_face_offset_local.to(device=device, dtype=torch.float32)
    if g.shell_layer is None:
        shell_base = torch.zeros((n_gauss,), dtype=torch.int64, device=device)
        if g.bind_idx is not None:
            shell_base[g.bind_idx.to(device=device) < 0] = -1
        g.shell_layer = shell_base
    else:
        g.shell_layer = g.shell_layer.to(device=device, dtype=torch.int64)
    if g.bind_mix_weight is None:
        g.bind_mix_weight = torch.full((n_gauss, 1), 0.5, dtype=torch.float32, device=device)
    else:
        g.bind_mix_weight = g.bind_mix_weight.to(device=device, dtype=torch.float32)
    if g.prune_state is None:
        g.prune_state = torch.zeros((n_gauss,), dtype=torch.int64, device=device)
    else:
        g.prune_state = g.prune_state.to(device=device, dtype=torch.int64)
    if g.quality_fail_counter is None:
        g.quality_fail_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    else:
        g.quality_fail_counter = g.quality_fail_counter.to(device=device, dtype=torch.float32)
    if g.color_conf_ema is None:
        g.color_conf_ema = torch.full((n_gauss,), 0.5, dtype=torch.float32, device=device)
    else:
        g.color_conf_ema = g.color_conf_ema.to(device=device, dtype=torch.float32)
    if g.geom_conf_ema is None:
        g.geom_conf_ema = torch.full((n_gauss,), 0.5, dtype=torch.float32, device=device)
    else:
        g.geom_conf_ema = g.geom_conf_ema.to(device=device, dtype=torch.float32)

    # Learnable local face offset for shell/deformation branch.
    g.bind_face_offset_local = torch.nn.Parameter(g.bind_face_offset_local)

    # Camera per-frame (axis-angle + translation)
    nF = len(frames_list)
    ref_window = int(temporal_ref_window)
    if ref_window < 0:
        raise ValueError(f"temporal_ref_window must be >= 0. Got {temporal_ref_window}")
    ref_decay = float(temporal_ref_decay)
    if ref_window > 0 and (ref_decay <= 0.0 or ref_decay > 1.0):
        raise ValueError(
            f"temporal_ref_decay must be in (0,1] when temporal_ref_window>0. Got {temporal_ref_decay}"
        )
    ref_anchor = str(temporal_ref_anchor).lower()
    if ref_anchor not in ("sample", "last"):
        raise ValueError(f"temporal_ref_anchor must be sample|last. Got {temporal_ref_anchor}")
    bind_mesh_obj_space_mode = str(bind_mesh_obj_space).strip().lower()
    if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        raise ValueError(
            f"bind_mesh_obj_space must be auto|camera|blender. Got: {bind_mesh_obj_space}"
        )
    mesh_obj_in_camera_space = bool(bind_mesh_obj_space_mode == "camera")
    bind_quat_normal_blend = float(np.clip(bind_quat_normal_blend, 0.0, 1.0))
    bind_surface_aniso_weight = float(max(bind_surface_aniso_weight, 0.0))
    bind_surface_aniso_margin = float(max(bind_surface_aniso_margin, 0.0))
    default_bind_mode = "games_face" if (g.bind_face_idx is not None and g.bind_bary is not None) else "legacy"
    binding_cfg = parse_binding_mode_config(binding, default_mode=default_bind_mode)
    bind_rigid_align_cfg = normalize_rigid_align_config(bind_rigid_align, default_enabled=False)
    bind_rigid_align_meta: dict | None = None
    bind_rigid_align_enabled = bool(
        bind_rigid_align_cfg is not None and bool(bind_rigid_align_cfg.get("enabled", False))
    )
    if bind_rigid_align_enabled and not binding_enabled:
        print("Warning: bind_rigid_align is enabled but binding is disabled; ignoring rigid align.")
        bind_rigid_align_enabled = False

    cam_t_init = torch.tensor([cam_t_prior], device=device).float().view(1, 3)
    cam_w = torch.zeros((nF, 3), device=device)
    cam_t = cam_t_init.repeat(nF, 1)
    learn_cam = str(camera_mode).lower() != "fixed" and lr_cam > 0
    if learn_cam:
        cam_w = torch.nn.Parameter(cam_w)
        cam_t = torch.nn.Parameter(cam_t)

    # Optimizer (we will dynamically freeze/unfreeze camera by toggling requires_grad)
    gauss_params = [g.log_scales, g.quat, g.opacity_logit, g.rgb_logit, g.bind_face_offset_local]
    if not binding_enabled:
        gauss_params = [g.xyz] + gauss_params
    param_groups = [{"params": gauss_params, "lr": lr_gauss}]
    if learn_cam:
        param_groups.append({"params": [cam_w, cam_t], "lr": lr_cam})
    opt = torch.optim.Adam(param_groups)

    # Preload GT frames (ok for small clips; for long clips stream from disk)
    gts = [read_image_rgb(frames_dir / fn).to(device) for fn in frames_list]

    # Helper to freeze/unfreeze cam params
    def _set_cam_trainable(flag: bool):
        if learn_cam:
            cam_w.requires_grad_(flag)
            cam_t.requires_grad_(flag)

    # Warmup: freeze camera first
    _set_cam_trainable(False)

    # Optional: keypoint-based bbox mask
    kpt_boxes = None
    face_boxes = None
    if keypoints_json is not None:
        kp_path = Path(keypoints_json)
        if kp_path.exists():
            with kp_path.open("r", encoding="utf-8") as f:
                kpts = json.load(f)
            # Build per-frame bbox (x0,y0,x1,y1)
            kpt_boxes = [] if use_kpt_bbox_mask else None
            face_boxes = [] if float(face_weight) > 1.0 else None
            face_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10]  # core face landmarks (exclude ears/hair by default)
            if bool(face_use_ears):
                face_ids = face_ids + [7, 8]
            for fn in frames_list:
                arr = kpts.get(fn, None)
                if arr is None:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    continue
                xs = [p[0] for p in arr if p[2] >= kpt_conf_min]
                ys = [p[1] for p in arr if p[2] >= kpt_conf_min]
                if not xs or not ys:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    continue
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                # pad bbox
                bw = (x1 - x0)
                bh = (y1 - y0)
                x0 = max(0.0, x0 - kpt_box_pad * bw)
                x1 = min(W - 1.0, x1 + kpt_box_pad * bw)
                y0 = max(0.0, y0 - kpt_box_pad * bh)
                y1 = min(H - 1.0, y1 + kpt_box_pad * bh)
                if kpt_boxes is not None:
                    kpt_boxes.append((x0, y0, x1, y1))

                if face_boxes is not None:
                    fx = []
                    fy = []
                    for fi in face_ids:
                        if fi < len(arr) and arr[fi][2] >= face_conf_min:
                            fx.append(arr[fi][0])
                            fy.append(arr[fi][1])
                    if not fx or not fy:
                        face_boxes.append(None)
                    else:
                        fx0, fx1 = min(fx), max(fx)
                        fy0, fy1 = min(fy), max(fy)
                        fbw = (fx1 - fx0)
                        fbh = (fy1 - fy0)
                        fx0 = max(0.0, fx0 - face_box_pad * fbw)
                        fx1 = min(W - 1.0, fx1 + face_box_pad * fbw)
                        fy0 = max(0.0, fy0 - face_box_pad * fbh)
                        fy1 = min(H - 1.0, fy1 + face_box_pad * fbh)
                        face_boxes.append((fx0, fy0, fx1, fy1))

    # Optional: stage1 person masks (works on non-green, cluttered backgrounds too).
    person_masks_frames: list[torch.Tensor | None] | None = None
    person_fuse = str(person_mask_fuse_mode).strip().lower()
    if person_fuse not in ("intersect", "union", "replace"):
        raise ValueError(
            f"person_mask_fuse_mode must be intersect|union|replace. Got: {person_mask_fuse_mode}"
        )
    sep_fg_bg = bool(separate_fg_bg_training)
    fg_w = float(max(fg_region_weight, 0.0))
    bg_w = float(max(bg_region_weight, 0.0))
    if sep_fg_bg and fg_w <= 0.0 and bg_w <= 0.0:
        raise ValueError(
            "separate_fg_bg_training=true requires at least one positive "
            "weight in fg_region_weight/bg_region_weight."
        )
    if bool(use_person_mask):
        if person_masks_npz is None:
            raise ValueError("use_person_mask=true but person_masks_npz is not set.")
        p_path = Path(person_masks_npz)
        if not p_path.exists():
            raise FileNotFoundError(f"person_masks_npz not found: {p_path}")
        pdata = np.load(p_path, allow_pickle=True)
        if "frame_names" not in pdata.files or "masks" not in pdata.files:
            raise RuntimeError(f"{p_path} must contain frame_names and masks.")
        p_names = [str(x) for x in pdata["frame_names"].tolist()]
        p_masks = np.asarray(pdata["masks"])
        if p_masks.ndim != 3:
            raise RuntimeError(f"Bad masks shape in {p_path}: {p_masks.shape}")
        name_to_idx = {n: i for i, n in enumerate(p_names)}
        person_masks_frames = [None] * nF
        pd = max(int(person_mask_dilate_px), 0)
        pe = max(int(person_mask_erode_px), 0)
        k_d = np.ones((2 * pd + 1, 2 * pd + 1), dtype=np.uint8) if pd > 0 else None
        k_e = np.ones((2 * pe + 1, 2 * pe + 1), dtype=np.uint8) if pe > 0 else None
        for i, fn in enumerate(frames_list):
            if fn not in name_to_idx:
                continue
            m = p_masks[int(name_to_idx[fn])].astype(np.float32)
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.uint8)
            if k_d is not None:
                m = cv2.dilate(m, k_d, iterations=1)
            if k_e is not None:
                m = cv2.erode(m, k_e, iterations=1)
            person_masks_frames[i] = torch.from_numpy(m.astype(np.float32)).to(device=device).unsqueeze(0)

    # Optional: small robustness閳ユ敂andomize frame order sampling
    # Optional: frame sampling weights (e.g., emphasize the last frame)
    frame_weights_t = None
    frame_loss_weights_t = None
    if frame_weights is not None:
        if len(frame_weights) != nF:
            raise ValueError("frame_weights length must match number of frames.")
        w = torch.tensor(frame_weights, dtype=torch.float32)
        if (w <= 0).any():
            raise ValueError("frame_weights must be > 0.")
        frame_weights_t = w / w.sum()
    if frame_loss_weights is not None:
        if len(frame_loss_weights) != nF:
            raise ValueError("frame_loss_weights length must match number of frames.")
        frame_loss_weights_t = torch.tensor(frame_loss_weights, dtype=torch.float32)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    # Early-stop state
    es_enable = bool(early_stop_enable) and int(early_stop_patience) > 0
    es_patience = max(int(early_stop_patience), 0)
    es_min_delta = max(float(early_stop_min_delta), 0.0)
    es_warmup = max(int(early_stop_warmup_iters), 0)
    es_best = float("inf")
    es_bad = 0

    # Binding helpers (lazy-load mesh vertices per frame)
    mesh_cache: dict[int, torch.Tensor] = {}
    mesh_normal_cache: dict[int, torch.Tensor] = {}
    mesh_np_cache: dict[int, np.ndarray] = {}
    mesh_binding_cache: dict[int, BindingFrameData] = {}
    mesh_cache_order: list[int] = []
    smpl_mask_cache: dict[int, torch.Tensor] = {}
    bind_idx_t = None
    bind_vertex_offset_t = None
    bind_face_idx_t = None
    bind_bary_t = None
    bind_faces_np = None
    bind_adjacency: list[np.ndarray] | None = None
    prev_vertex_dirs: torch.Tensor | None = None
    K_np = K.detach().cpu().numpy().astype(np.float32)
    if binding_enabled:
        bind_idx_t = bind_vertex_indices.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0
        if g.bind_face_idx is not None:
            bind_face_idx_t = g.bind_face_idx.to(device=device, dtype=torch.long)
        if g.bind_bary is not None:
            bind_bary_t = g.bind_bary.to(device=device, dtype=torch.float32)
        if g.bind_offset is not None:
            bind_vertex_offset_t = g.bind_offset.to(device=device, dtype=torch.float32)
        bind_mesh_offset_t = torch.tensor(bind_mesh_offset, dtype=torch.float32, device=device).view(1, 3)
        bind_mesh_offset_np = bind_mesh_offset_t.detach().cpu().numpy().reshape(1, 3).astype(np.float32)
        bind_rigid_apply_cfg = (
            {
                "enabled": True,
                "rx_deg": float(bind_rigid_align_cfg.get("rx_deg", 0.0)),
                "ry_deg": float(bind_rigid_align_cfg.get("ry_deg", 0.0)),
                "rz_deg": float(bind_rigid_align_cfg.get("rz_deg", 0.0)),
                "tx": float(bind_rigid_align_cfg.get("tx", 0.0)),
                "ty": float(bind_rigid_align_cfg.get("ty", 0.0)),
                "tz": float(bind_rigid_align_cfg.get("tz", 0.0)),
            }
            if bind_rigid_align_enabled and bind_rigid_align_cfg is not None
            else None
        )
        exp_upright_rot = np.eye(3, dtype=np.float32)
        exp_center_xy = np.zeros((2,), dtype=np.float32)
        exp_floor_z = 0.0
        exp_has_center_xy = False
        exp_has_floor = False
        if bind_mesh_indices is None:
            bind_mesh_indices = list(range(len(frames_list)))
        if len(bind_mesh_indices) != len(frames_list):
            raise ValueError("bind_mesh_indices must align with frames_list length.")
        verts_seq_np = None
        verts_seq_ids_set: set[int] = set()
        verts_seq_fallback_idx = 0
        if bind_vertices_npz is not None:
            vseq_path = Path(bind_vertices_npz)
            if vseq_path.exists():
                vseq = np.load(vseq_path, allow_pickle=True)
                if "verts_cam" in vseq.files:
                    verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)
                    if verts_seq_np.ndim != 3 or verts_seq_np.shape[-1] != 3:
                        raise RuntimeError(
                            f"Bad verts_cam shape in {vseq_path}: {verts_seq_np.shape}"
                        )
                    verts_seq_ids_set = set(range(int(verts_seq_np.shape[0])))
                    verts_seq_fallback_idx = int(max(int(verts_seq_np.shape[0]) - 1, 0))
                    print(f"Binding uses dynamic vertices from {vseq_path}")
                else:
                    print(f"Warning: {vseq_path} missing verts_cam; fallback to OBJ sequence binding.")
            else:
                print(f"Warning: bind_vertices_npz not found: {vseq_path}. Fallback to OBJ sequence binding.")
        bind_mesh_dir = Path(bind_mesh_dir)
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(f"bind_mesh_dir not found: {bind_mesh_dir}")
        def _parse_mesh_frame_idx(p: Path) -> int | None:
            stem = p.stem.strip().lower()
            if stem.isdigit():
                return int(stem)
            if stem.startswith("frame_") and stem[6:].isdigit():
                return int(stem[6:])
            return None

        mesh_path_map: dict[int, Path] = {}
        for p in bind_mesh_dir.glob("*.obj"):
            idx = _parse_mesh_frame_idx(p)
            if idx is None:
                continue
            old_p = mesh_path_map.get(idx, None)
            if old_p is None:
                mesh_path_map[idx] = p
            else:
                old_is_canonical = old_p.stem.lower().startswith("frame_")
                new_is_canonical = p.stem.lower().startswith("frame_")
                if new_is_canonical and (not old_is_canonical):
                    mesh_path_map[idx] = p
        mesh_ids = sorted(mesh_path_map.keys())
        if len(mesh_ids) == 0:
            raise FileNotFoundError(f"No sequence OBJ meshes found in bind_mesh_dir: {bind_mesh_dir}")
        mesh_ids_set = set(mesh_ids)
        fallback_mesh_idx = int(mesh_ids[-1])
        if verts_seq_np is not None:
            missing_seq_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in verts_seq_ids_set]
            if len(missing_seq_ids) > 0:
                print(
                    "Warning: some bind indices are missing in verts sequence; "
                    f"fallback to frame {verts_seq_fallback_idx} for {len(missing_seq_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in verts_seq_ids_set else verts_seq_fallback_idx
                    for mi in bind_mesh_indices
                ]
        else:
            missing_mesh_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in mesh_ids_set]
            if len(missing_mesh_ids) > 0:
                print(
                    "Warning: some bind mesh indices are missing; "
                    f"fallback to mesh {fallback_mesh_idx:06d}.obj for {len(missing_mesh_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in mesh_ids_set else fallback_mesh_idx
                    for mi in bind_mesh_indices
                ]
        if (
            bool(use_smpl_silhouette_mask)
            or bool(bind_align_quat_to_mesh_normals)
            or str(binding_cfg.mode).lower() in ("games_face", "hybrid")
        ):
            if len(bind_mesh_indices) == 0:
                raise RuntimeError("use_smpl_silhouette_mask=true but bind_mesh_indices is empty.")
            first_mesh_idx = int(bind_mesh_indices[0]) if int(bind_mesh_indices[0]) in mesh_ids_set else fallback_mesh_idx
            first_obj = mesh_path_map.get(first_mesh_idx, None)
            if first_obj is None:
                raise FileNotFoundError(f"Mesh frame not found for silhouette mask index: {first_mesh_idx}")
            _, bind_faces_np = load_obj_mesh(first_obj)
            if bind_faces_np is None or len(bind_faces_np) == 0:
                if bool(use_smpl_silhouette_mask):
                    raise RuntimeError(
                        "use_smpl_silhouette_mask=true but mesh has no faces; "
                        "cannot build silhouette."
                    )
                if str(binding_cfg.mode).lower() == "games_face":
                    raise RuntimeError(
                        "binding.mode=games_face requires mesh faces, but loaded mesh has no faces."
                    )
                bind_faces_np = None

        if bind_mesh_to_camera and bind_inverse_export_upright and bind_seq_npz is not None:
            seq_path = Path(bind_seq_npz)
            if seq_path.exists():
                seq = np.load(seq_path, allow_pickle=True)
                if bind_mesh_obj_space_mode == "auto" and "fit_strategy" in seq.files:
                    fs_arr = np.asarray(seq["fit_strategy"]).reshape(-1)
                    if fs_arr.size > 0 and str(fs_arr[0]).strip().lower() in ("stage1_motion_prior", "vibe"):
                        mesh_obj_in_camera_space = True
                if "export_upright_rot" in seq.files:
                    exp_upright_rot = np.asarray(seq["export_upright_rot"], dtype=np.float32).reshape(3, 3)
                if "export_center_xy" in seq.files:
                    exp_has_center_xy = bool(int(np.asarray(seq["export_center_xy"]).reshape(-1)[0]))
                if "export_center_xy_value" in seq.files:
                    exp_center_xy = np.asarray(seq["export_center_xy_value"], dtype=np.float32).reshape(2,)
                if "export_on_floor" in seq.files:
                    exp_has_floor = bool(int(np.asarray(seq["export_on_floor"]).reshape(-1)[0]))
                if "export_floor_z" in seq.files:
                    exp_floor_z = float(np.asarray(seq["export_floor_z"], dtype=np.float32).reshape(-1)[0])

        mesh_raw_cache: dict[int, np.ndarray] = {}

        def _resolve_mesh_idx(mesh_idx: int) -> int:
            mesh_idx_use = int(mesh_idx)
            if verts_seq_np is not None:
                if mesh_idx_use not in verts_seq_ids_set:
                    mesh_idx_use = verts_seq_fallback_idx
            else:
                if mesh_idx_use not in mesh_ids_set:
                    mesh_idx_use = fallback_mesh_idx
            return mesh_idx_use

        def _load_mesh_np_raw(mesh_idx: int) -> tuple[np.ndarray, int]:
            mesh_idx_use = _resolve_mesh_idx(mesh_idx)
            if mesh_idx_use in mesh_raw_cache:
                return mesh_raw_cache[mesh_idx_use], mesh_idx_use
            if verts_seq_np is not None:
                verts_np = np.asarray(verts_seq_np[mesh_idx_use], dtype=np.float32)
            else:
                obj_path = mesh_path_map.get(mesh_idx_use, None)
                if obj_path is None:
                    raise FileNotFoundError(f"Mesh frame index not found: {mesh_idx_use}")
                verts_np = load_obj_vertices(obj_path)
                if bind_mesh_to_camera and (not bool(mesh_obj_in_camera_space)):
                    if bind_inverse_export_upright:
                        if exp_has_center_xy:
                            verts_np[:, 0] += float(exp_center_xy[0])
                            verts_np[:, 1] += float(exp_center_xy[1])
                        if exp_has_floor:
                            verts_np[:, 2] += float(exp_floor_z)
                        # Inverse of stage1 upright rotation: v_blender_raw = v_export @ U
                        verts_np = (verts_np @ exp_upright_rot.astype(np.float32))
                    # Inverse of stage1 _camera_to_blender: (x, y, z) = (x_b, -z_b, y_b)
                    verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
            mesh_raw_cache[mesh_idx_use] = np.asarray(verts_np, dtype=np.float32)
            return mesh_raw_cache[mesh_idx_use], mesh_idx_use

        def _apply_mesh_transforms_np(verts_np: np.ndarray) -> np.ndarray:
            v = np.asarray(verts_np, dtype=np.float32)
            if bind_rigid_apply_cfg is not None:
                v = apply_rigid_transform_np(v, bind_rigid_apply_cfg)
            v = v + bind_mesh_offset_np
            return np.asarray(v, dtype=np.float32)

        if (
            bind_rigid_align_enabled
            and bind_rigid_align_cfg is not None
            and bool(bind_rigid_align_cfg.get("auto", False))
            and bind_faces_np is not None
        ):
            pm_cfg = bind_rigid_align_cfg.get("person_masks_npz", None)
            pm_path = Path(pm_cfg) if pm_cfg is not None else (
                Path(person_masks_npz) if person_masks_npz is not None else None
            )
            if pm_path is not None and pm_path.exists():
                pdata = np.load(pm_path, allow_pickle=True)
                if "frame_names" in pdata.files and "masks" in pdata.files:
                    p_names = [str(x) for x in pdata["frame_names"].tolist()]
                    p_masks = np.asarray(pdata["masks"], dtype=np.uint8)
                    if p_masks.ndim == 3 and len(p_names) == int(p_masks.shape[0]):
                        name_to_pm = {n: i for i, n in enumerate(p_names)}
                        valid_frame_ids = [i for i, fn in enumerate(frames_list) if fn in name_to_pm]
                        if len(valid_frame_ids) > 0:
                            sample_n = min(int(bind_rigid_align_cfg.get("sample_frames", 8)), len(valid_frame_ids))
                            pick = np.linspace(0, len(valid_frame_ids) - 1, num=sample_n, dtype=np.float64)
                            pick = np.clip(np.round(pick).astype(np.int64), 0, len(valid_frame_ids) - 1)
                            pick = np.unique(pick)
                            sample_ids = [valid_frame_ids[int(k)] for k in pick.tolist()]

                            ds = float(bind_rigid_align_cfg.get("eval_downsample", 0.5))
                            Hs = max(8, int(round(float(H) * ds)))
                            Ws = max(8, int(round(float(W) * ds)))
                            sx = float(Ws) / float(W)
                            sy = float(Hs) / float(H)
                            K_eval = np.asarray(K_np, dtype=np.float32).copy()
                            K_eval[0, 0] *= sx
                            K_eval[0, 2] *= sx
                            K_eval[1, 1] *= sy
                            K_eval[1, 2] *= sy
                            outside_w = float(bind_rigid_align_cfg.get("outside_weight", 2.0))
                            gt_masks_eval: dict[int, np.ndarray] = {}
                            for fj in sample_ids:
                                pm_i = int(name_to_pm[frames_list[fj]])
                                gt_m = (p_masks[pm_i] > 0).astype(np.uint8)
                                if Hs != H or Ws != W:
                                    gt_m = cv2.resize(gt_m, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
                                gt_masks_eval[fj] = (gt_m > 0)

                            def _eval_score(param_dict: dict) -> float:
                                cfg_eval = {"enabled": True}
                                cfg_eval.update(param_dict)
                                acc = 0.0
                                used = 0
                                for fj in sample_ids:
                                    mesh_i = int(bind_mesh_indices[fj])
                                    verts_raw, _ = _load_mesh_np_raw(mesh_i)
                                    verts_eval = apply_rigid_transform_np(verts_raw, cfg_eval) + bind_mesh_offset_np
                                    R_eval = axis_angle_to_rotmat(cam_w[fj]).detach().cpu().numpy().astype(np.float32)
                                    t_eval = cam_t[fj].detach().cpu().numpy().astype(np.float32)
                                    pred = _rasterize_mesh_mask(
                                        verts=verts_eval,
                                        faces=bind_faces_np,
                                        K=K_eval,
                                        R=R_eval,
                                        t=t_eval,
                                        H=Hs,
                                        W=Ws,
                                        dilate_px=0,
                                        erode_px=0,
                                    ) > 0
                                    gt_m = gt_masks_eval[fj]
                                    inter = float(np.logical_and(pred, gt_m).sum())
                                    union = float(np.logical_or(pred, gt_m).sum())
                                    iou = inter / (union + 1e-6)
                                    pred_area = float(pred.sum())
                                    outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
                                    acc += (iou - outside_w * outside)
                                    used += 1
                                if used <= 0:
                                    return -1e9
                                return float(acc / float(used))

                            def _eval_metrics(param_dict: dict) -> tuple[float, float, float]:
                                cfg_eval = {"enabled": True}
                                cfg_eval.update(param_dict)
                                iou_acc = 0.0
                                outside_acc = 0.0
                                score_acc = 0.0
                                used = 0
                                for fj in sample_ids:
                                    mesh_i = int(bind_mesh_indices[fj])
                                    verts_raw, _ = _load_mesh_np_raw(mesh_i)
                                    verts_eval = apply_rigid_transform_np(verts_raw, cfg_eval) + bind_mesh_offset_np
                                    R_eval = axis_angle_to_rotmat(cam_w[fj]).detach().cpu().numpy().astype(np.float32)
                                    t_eval = cam_t[fj].detach().cpu().numpy().astype(np.float32)
                                    pred = _rasterize_mesh_mask(
                                        verts=verts_eval,
                                        faces=bind_faces_np,
                                        K=K_eval,
                                        R=R_eval,
                                        t=t_eval,
                                        H=Hs,
                                        W=Ws,
                                        dilate_px=0,
                                        erode_px=0,
                                    ) > 0
                                    gt_m = gt_masks_eval[fj]
                                    inter = float(np.logical_and(pred, gt_m).sum())
                                    union = float(np.logical_or(pred, gt_m).sum())
                                    iou = inter / (union + 1e-6)
                                    pred_area = float(pred.sum())
                                    outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
                                    score = iou - outside_w * outside
                                    iou_acc += iou
                                    outside_acc += outside
                                    score_acc += score
                                    used += 1
                                if used <= 0:
                                    return 0.0, 1.0, -1e9
                                n = float(used)
                                return float(iou_acc / n), float(outside_acc / n), float(score_acc / n)

                            init_params = dict(bind_rigid_apply_cfg) if bind_rigid_apply_cfg is not None else {
                                "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0,
                                "tx": 0.0, "ty": 0.0, "tz": 0.0,
                            }
                            init_params.pop("enabled", None)
                            best_params, score_before, score_after = coordinate_descent_rigid_search(
                                score_fn=_eval_score,
                                init_params=init_params,
                                rot_step_deg=float(bind_rigid_align_cfg.get("rot_step_deg", 3.0)),
                                trans_step=tuple(bind_rigid_align_cfg.get("trans_step", (0.04, 0.04, 0.06))),
                                levels=int(bind_rigid_align_cfg.get("levels", 3)),
                                iters_per_level=int(bind_rigid_align_cfg.get("iters_per_level", 2)),
                                max_rot_deg=float(bind_rigid_align_cfg.get("max_rot_deg", 20.0)),
                                max_trans=tuple(bind_rigid_align_cfg.get("max_trans", (0.35, 0.35, 0.45))),
                            )
                            bind_rigid_apply_cfg = {"enabled": True}
                            bind_rigid_apply_cfg.update(best_params)
                            iou_before, outside_before, _ = _eval_metrics(init_params)
                            iou_after, outside_after, _ = _eval_metrics(best_params)
                            bind_rigid_align_meta = {
                                "enabled": True,
                                "auto": True,
                                "params": dict(bind_rigid_apply_cfg),
                                "sample_frames": int(len(sample_ids)),
                                "score_before": float(score_before),
                                "score_after": float(score_after),
                                "iou_before": float(iou_before),
                                "iou_after": float(iou_after),
                                "outside_before": float(outside_before),
                                "outside_after": float(outside_after),
                                "person_masks_npz": str(pm_path),
                            }
                            print(
                                "bind_rigid_align auto: "
                                f"score {score_before:.4f}->{score_after:.4f}, "
                                f"iou {iou_before:.4f}->{iou_after:.4f}, "
                                f"outside {outside_before:.4f}->{outside_after:.4f}, "
                                f"rot=({best_params['rx_deg']:.3f},{best_params['ry_deg']:.3f},{best_params['rz_deg']:.3f})deg, "
                                f"trans=({best_params['tx']:.4f},{best_params['ty']:.4f},{best_params['tz']:.4f})"
                            )
                        else:
                            print("Warning: bind_rigid_align auto skipped (no overlapping frame names in person masks).")
                    else:
                        print(f"Warning: bind_rigid_align auto skipped (bad masks shape in {pm_path}).")
                else:
                    print(f"Warning: bind_rigid_align auto skipped ({pm_path} missing frame_names/masks).")
            else:
                print("Warning: bind_rigid_align auto skipped (person_masks_npz missing).")

        if bind_rigid_align_enabled and bind_rigid_apply_cfg is not None and bind_rigid_align_meta is None:
            bind_rigid_align_meta = {
                "enabled": True,
                "auto": False,
                "params": dict(bind_rigid_apply_cfg),
            }
            print(
                "bind_rigid_align manual: "
                f"rot=({bind_rigid_apply_cfg['rx_deg']:.3f},{bind_rigid_apply_cfg['ry_deg']:.3f},{bind_rigid_apply_cfg['rz_deg']:.3f})deg, "
                f"trans=({bind_rigid_apply_cfg['tx']:.4f},{bind_rigid_apply_cfg['ty']:.4f},{bind_rigid_apply_cfg['tz']:.4f})"
            )

        def _load_mesh_verts(mesh_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
            mesh_idx_use = _resolve_mesh_idx(mesh_idx)
            if mesh_idx_use in mesh_cache:
                return mesh_cache[mesh_idx_use], mesh_normal_cache.get(mesh_idx_use, None), mesh_idx_use
            verts_raw_np, mesh_idx_use = _load_mesh_np_raw(mesh_idx_use)
            verts_np = _apply_mesh_transforms_np(verts_raw_np)
            verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
            mesh_cache[mesh_idx_use] = verts_t
            mesh_np_cache[mesh_idx_use] = np.asarray(verts_np, dtype=np.float32)
            normals_t = None
            if bool(bind_align_quat_to_mesh_normals) and bind_faces_np is not None:
                normals_np = _compute_vertex_normals_np(verts_np, bind_faces_np)
                normals_t = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)
                mesh_normal_cache[mesh_idx_use] = normals_t
            mesh_cache_order.append(mesh_idx_use)
            if len(mesh_cache_order) > int(bind_cache_size):
                old = mesh_cache_order.pop(0)
                mesh_cache.pop(old, None)
                mesh_normal_cache.pop(old, None)
                mesh_np_cache.pop(old, None)
                mesh_binding_cache.pop(old, None)
                mesh_raw_cache.pop(old, None)
            return verts_t, normals_t, mesh_idx_use

        def _load_binding_frame(mesh_idx_use: int) -> BindingFrameData:
            if mesh_idx_use in mesh_binding_cache:
                return mesh_binding_cache[mesh_idx_use]
            if mesh_idx_use not in mesh_np_cache:
                raise RuntimeError(f"Mesh verts cache miss for binding frame: {mesh_idx_use}")
            verts_np = mesh_np_cache[mesh_idx_use]
            nonlocal bind_adjacency
            if bind_faces_np is not None:
                if bind_adjacency is None or len(bind_adjacency) != int(verts_np.shape[0]):
                    bind_adjacency = build_vertex_adjacency(int(verts_np.shape[0]), bind_faces_np)
            frame = build_binding_frame_data(
                verts_np=verts_np,
                faces_np=bind_faces_np,
                adjacency=bind_adjacency,
                cfg=binding_cfg,
                device=device,
            )
            mesh_binding_cache[mesh_idx_use] = frame
            return frame

        if (
            bool(use_smpl_silhouette_mask)
            and bind_faces_np is not None
            and (not learn_cam)
            and bool(smpl_mask_precompute)
            and int(smpl_mask_precompute_max_frames) > 0
            and len(bind_mesh_indices) <= int(smpl_mask_precompute_max_frames)
        ):
            cam_r0 = axis_angle_to_rotmat(cam_w[0]).detach().cpu().numpy().astype(np.float32)
            cam_t0 = cam_t[0].detach().cpu().numpy().astype(np.float32)
            unique_mesh = sorted(set(int(mi) for mi in bind_mesh_indices))
            for mesh_idx_pc in unique_mesh:
                verts_pc, _, _ = _load_mesh_verts(mesh_idx_pc)
                mask_np = _rasterize_mesh_mask(
                    verts=verts_pc.detach().cpu().numpy(),
                    faces=bind_faces_np,
                    K=K_np,
                    R=cam_r0,
                    t=cam_t0,
                    H=H,
                    W=W,
                    dilate_px=int(smpl_mask_dilate_px),
                    erode_px=int(smpl_mask_erode_px),
                )
                smpl_mask_cache[mesh_idx_pc] = (
                    torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                )

    # ---- Adaptive training helpers ----
    phase0_end = max(1, int(round(float(iters) * float(np.clip(phase0_ratio, 0.0, 0.9)))))
    phase2_start = max(phase0_end + 1, int(round(float(iters) * float(1.0 - np.clip(phase2_ratio, 0.0, 0.9)))))
    phase2_lr_scale = 0.35

    binding_raw = binding if isinstance(binding, dict) else {}
    hybrid_dynamic = bool(binding_raw.get("hybrid_dynamic", getattr(binding_cfg, "hybrid_dynamic", False)))
    mix_w_min = float(binding_raw.get("mix_w_min", getattr(binding_cfg, "mix_w_min", 0.05)))
    mix_w_max = float(binding_raw.get("mix_w_max", getattr(binding_cfg, "mix_w_max", 0.95)))
    mix_coef_curv = float(binding_raw.get("mix_coef_curv", 1.0))
    mix_coef_edge = float(binding_raw.get("mix_coef_edge", 0.8))
    mix_coef_residual = float(binding_raw.get("mix_coef_residual", 0.7))
    mix_coef_normal_var = float(binding_raw.get("mix_coef_normal_var", 0.6))
    mix_bias = float(binding_raw.get("mix_bias", 1.1))

    frame_residual_ema: dict[int, FrameResidualEMA] = {i: FrameResidualEMA(alpha=0.9) for i in range(nF)}
    densify_stats = GaussianDensifyStats.create(n_gaussians=n_gauss, device=device)
    border_att_cache: dict[int, np.ndarray] = {}

    inner_soft_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    inner_hard_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)

    dq_raw = dual_quality_gate if isinstance(dual_quality_gate, dict) else {}
    dq_enable = bool(dq_raw.get("enable", False))
    dq_mode = str(dq_raw.get("mode", "hard_and")).strip().lower()
    if dq_mode not in ("hard_and", "hard_or"):
        dq_mode = "hard_and"
    dq_eval_every = max(int(dq_raw.get("eval_every", 100)), 1)
    dq_warmup_iters = max(int(dq_raw.get("warmup_iters", 1200)), 0)

    dq_color = dq_raw.get("color", {}) if isinstance(dq_raw.get("color", {}), dict) else {}
    dq_color_max_error = float(dq_color.get("max_error", 0.045))
    dq_color_unstable_var_max = float(dq_color.get("unstable_var_max", 0.012))
    dq_color_roi = str(dq_color.get("roi", "person_intersect_smpl")).strip().lower()

    dq_geom = dq_raw.get("geometry", {}) if isinstance(dq_raw.get("geometry", {}), dict) else {}
    dq_geom_min_score = float(dq_geom.get("min_score", 0.78))
    dq_geom_residual_top_percent = float(dq_geom.get("residual_top_percent", 20.0))
    dq_geom_edge_top_percent = float(dq_geom.get("edge_top_percent", 30.0))
    dq_geom_roi = str(dq_geom.get("roi", "contour_band")).strip().lower()

    dq_ctrl = dq_raw.get("control", {}) if isinstance(dq_raw.get("control", {}), dict) else {}
    dq_spawn_scale_if_fail = float(dq_ctrl.get("spawn_scale_if_fail", 0.0))
    dq_outer_lr_scale_if_fail = float(dq_ctrl.get("outer_lr_scale_if_fail", 0.5))
    dq_outer_alpha_cap_if_fail = float(dq_ctrl.get("outer_alpha_cap_if_fail", 0.28))
    dq_prune_boost_if_fail = float(dq_ctrl.get("prune_boost_if_fail", 1.25))
    dq_recover_patience = max(int(dq_ctrl.get("recover_patience", 3)), 1)
    dq_outer_color_band_weight = float(dq_ctrl.get("outer_color_band_weight", 0.05))
    dq_layer_alpha_balance_weight = float(dq_ctrl.get("layer_alpha_balance_weight", 0.02))

    dq_fail_active = False
    dq_pass_streak = 0
    dq_last_color_error = 0.0
    dq_last_color_var = 0.0
    dq_last_geom_score = 0.0

    for it in tqdm(range(iters), desc="Stage2 3DGS (learnable camera)"):
        opt.zero_grad(set_to_none=True)

        # Training phase schedule
        if it < phase0_end:
            phase_tag = "phase0"
        elif it >= phase2_start:
            phase_tag = "phase2"
        else:
            phase_tag = "phase1"

        for pg in opt.param_groups:
            base_lr = float(pg.get("initial_lr", pg["lr"]))
            pg.setdefault("initial_lr", base_lr)
            pg["lr"] = base_lr * (phase2_lr_scale if phase_tag == "phase2" else 1.0)

        # Enable camera learning after warmup
        if learn_cam and it == cam_warmup_iters:
            _set_cam_trainable(True)

        # Select anchor frame (current frame to optimize).
        if ref_anchor == "last":
            fi = nF - 1
        elif frame_weights_t is None:
            fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        else:
            fi = int(torch.multinomial(frame_weights_t, 1, replacement=True, generator=rng).item())

        # Build temporal reference set: current frame + previous frames with decay.
        if ref_window > 0:
            start_idx = max(0, fi - ref_window)
            ctx_indices = list(range(start_idx, fi + 1))
            raw_w = np.asarray([ref_decay ** (fi - fj) for fj in ctx_indices], dtype=np.float32)
            raw_w = raw_w / float(raw_w.sum() + 1e-8)
            ctx_weights = raw_w.tolist()
        else:
            ctx_indices = [fi]
            ctx_weights = [1.0]

        loss_value = 0.0
        vis_accum: list[torch.Tensor] = []
        color_eval_vals: list[float] = []
        color_var_vals: list[float] = []
        geom_eval_vals: list[float] = []
        for fj, wj in zip(ctx_indices, ctx_weights):
            gt = gts[fj]  # (3,H,W)

            # Camera pose for this reference frame
            R = axis_angle_to_rotmat(cam_w[fj])
            t = cam_t[fj]

            # Update bound xyz for this reference frame (dynamic binding)
            mesh_idx = None
            verts_t = None
            normals_t = None
            bind_mask_curr = bind_mask if binding_enabled else None
            frame_data = None
            if binding_enabled:
                mesh_idx = int(bind_mesh_indices[fj])
                verts_t, normals_t, mesh_idx_use = _load_mesh_verts(mesh_idx)
                if bind_mask.any():
                    if str(binding_cfg.mode).lower() == "legacy":
                        bind_idx_use = bind_idx_t[bind_mask]
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
                        frame_data = _load_binding_frame(mesh_idx_use)

                        # Dynamic hybrid binding weight per gaussian (position-adaptive).
                        if hybrid_dynamic and str(binding_cfg.mode).lower() == "hybrid":
                            with torch.no_grad():
                                b_ids = torch.nonzero(bind_mask, as_tuple=False).view(-1)
                                if int(b_ids.numel()) > 0:
                                    if bind_face_idx_t is not None and bind_bary_t is not None and frame_data.faces_t is not None:
                                        f_idx_all = bind_face_idx_t.index_select(0, b_ids)
                                        bary_all = bind_bary_t.index_select(0, b_ids)
                                        f_valid = f_idx_all >= 0
                                        curv_sig = torch.zeros((int(b_ids.numel()),), dtype=torch.float32, device=device)
                                        if bool(f_valid.any()):
                                            f_idx = f_idx_all[f_valid].long()
                                            face_vid = frame_data.faces_t.index_select(0, f_idx)
                                            curv_v = frame_data.curv_conf_t.index_select(0, face_vid.reshape(-1)).view(-1, 3)
                                            curv_sig[f_valid] = (curv_v * bary_all[f_valid]).sum(dim=1)
                                    else:
                                        v_idx = bind_idx_t.index_select(0, b_ids).clamp(min=0)
                                        curv_sig = frame_data.curv_conf_t.index_select(0, v_idx)

                                    residual_sig = densify_stats.residual_ema.index_select(0, b_ids)
                                    edge_sig = (curv_sig > float(binding_cfg.curv_conf_tau)).float()
                                    if prev_vertex_dirs is not None and prev_vertex_dirs.shape[0] == frame_data.vertex_n_t.shape[0]:
                                        v_idx = bind_idx_t.index_select(0, b_ids).clamp(min=0)
                                        n_now = frame_data.vertex_n_t.index_select(0, v_idx)
                                        n_prev = prev_vertex_dirs.index_select(0, v_idx)
                                        normal_var_sig = (1.0 - (n_now * n_prev).sum(dim=1).abs()).clamp(0.0, 1.0)
                                    else:
                                        normal_var_sig = torch.zeros_like(curv_sig)

                                    logits = (
                                        mix_coef_curv * curv_sig
                                        + mix_coef_edge * edge_sig
                                        + mix_coef_residual * residual_sig
                                        + mix_coef_normal_var * normal_var_sig
                                        - mix_bias
                                    )
                                    w_dyn = torch.sigmoid(logits).clamp(min=float(mix_w_min), max=float(mix_w_max))
                                    g.bind_mix_weight[b_ids, 0] = w_dyn

                        prev_vertex_dirs, bind_mask_curr = update_gaussians_with_binding(
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

            # Optional silhouette mask from bound SMPL mesh.
            smpl_mask = None
            if binding_enabled and bool(use_smpl_silhouette_mask) and bind_faces_np is not None:
                cache_key = int(mesh_idx) if (mesh_idx is not None and not learn_cam) else -1
                if cache_key >= 0 and cache_key in smpl_mask_cache:
                    smpl_mask = smpl_mask_cache[cache_key]
                else:
                    if verts_t is None:
                        raise RuntimeError("Silhouette mask requires bound verts, but verts_t is None.")
                    mask_np = _rasterize_mesh_mask(
                        verts=verts_t.detach().cpu().numpy(),
                        faces=bind_faces_np,
                        K=K_np,
                        R=R.detach().cpu().numpy().astype(np.float32),
                        t=t.detach().cpu().numpy().astype(np.float32),
                        H=H,
                        W=W,
                        dilate_px=int(smpl_mask_dilate_px),
                        erode_px=int(smpl_mask_erode_px),
                    )
                    smpl_mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                    if cache_key >= 0:
                        smpl_mask_cache[cache_key] = smpl_mask

            # Common mask/weight builders.
            mask = None
            if use_kpt_bbox_mask and kpt_boxes is not None and kpt_boxes[fj] is not None:
                x0, y0, x1, y1 = kpt_boxes[fj]
                m2d = torch.zeros((1, H, W), device=device)
                m2d[:, int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1.0
                mask = m2d

            person_mask_curr = None
            if person_masks_frames is not None:
                p_mask = person_masks_frames[fj]
                person_mask_curr = p_mask
                if p_mask is not None and not sep_fg_bg:
                    if mask is None:
                        mask = p_mask
                    elif person_fuse == "replace":
                        mask = p_mask
                    elif person_fuse == "union":
                        mask = (mask + p_mask).clamp(0.0, 1.0)
                    else:
                        mask = mask * p_mask

            smpl_mask_for_color = None
            if smpl_mask is not None:
                smpl_mask_for_color = smpl_mask
                if int(smpl_color_mask_erode_px) > 0:
                    smpl_mask_for_color = _erode_binary_mask_torch(smpl_mask_for_color, int(smpl_color_mask_erode_px))
                if (not sep_fg_bg) and bool(smpl_mask_override_alpha):
                    mask = smpl_mask_for_color if mask is None else (mask * smpl_mask_for_color)

            weight = None
            if face_boxes is not None and face_boxes[fj] is not None and float(face_weight) > 1.0:
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                w2d = torch.ones((1, H, W), device=device)
                w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_weight)
                weight = w2d

            if sep_fg_bg:
                region_mask = smpl_mask_for_color if smpl_mask_for_color is not None else person_mask_curr
                if region_mask is not None:
                    region_w = (bg_w + (fg_w - bg_w) * region_mask).clamp(min=0.0)
                    weight = region_w if weight is None else (weight * region_w)

            if mask is not None:
                weight = mask if weight is None else (weight * mask)

            gate_color_roi_mask = None
            gate_geom_target = None
            gate_geom_band = None
            if dq_enable:
                if dq_color_roi == "person_intersect_smpl":
                    if person_mask_curr is not None and smpl_mask_for_color is not None:
                        gate_color_roi_mask = (person_mask_curr * smpl_mask_for_color).clamp(0.0, 1.0)
                    elif person_mask_curr is not None:
                        gate_color_roi_mask = person_mask_curr
                    elif smpl_mask_for_color is not None:
                        gate_color_roi_mask = smpl_mask_for_color
                elif dq_color_roi == "person":
                    gate_color_roi_mask = person_mask_curr
                elif dq_color_roi == "smpl":
                    gate_color_roi_mask = smpl_mask_for_color

                if gate_color_roi_mask is None:
                    gate_color_roi_mask = torch.ones((1, H, W), dtype=torch.float32, device=device)

                if dq_geom_roi == "contour_band":
                    gate_geom_target = gate_color_roi_mask
                    gate_geom_band = _compute_contour_band_torch(gate_geom_target, width_px=2)
                else:
                    gate_geom_target = gate_color_roi_mask

            # Pass-1 render: estimate visibility and frame residual.
            pred_rgb_p1, pred_a_p1, meta_p1 = render_gaussians_3d(
                g, K, R, t, H, W, bg=bg_rgb, return_alpha=True, return_meta=True
            )
            diff_p1 = pred_rgb_p1 - gt
            residual_scalar = float(diff_p1.abs().mean().detach().cpu())
            residual_ema_val = frame_residual_ema[fj].update(residual_scalar)
            frame_w_scalar = float(frame_weight_from_residual(residual_ema_val, frame_weight_k))

            visibility = compute_gaussian_visibility(
                xyz=g.xyz.detach(),
                K=K,
                R=R,
                t=t,
                image_h=H,
                image_w=W,
                alpha_map=pred_a_p1.detach(),
                person_mask=person_mask_curr,
                renderer_meta=meta_p1,
                min_alpha=max(float(alpha_threshold), 0.005),
            )

            if bool(use_border_attenuation) and person_mask_curr is not None and int(border_attenuation_width) > 0:
                if fj not in border_att_cache:
                    pm_np = (person_mask_curr[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    border_att_cache[fj] = build_border_attenuation(pm_np, int(border_attenuation_width))
                u_proj, v_proj, valid_proj = _project_points_torch(g.xyz.detach(), K, R, t)
                uv_proj = torch.stack([u_proj, v_proj], dim=1)
                border_w = sample_border_weight(border_att_cache[fj], uv_proj, valid_proj)
                visibility = visibility * border_w

            active_mask = (g.prune_state != 2)
            visibility = visibility * active_mask.float()
            # Keep a tiny visibility floor for currently visible splats during early stabilization,
            # otherwise training can stall when pass-1 alpha is still weak.
            vis_floor = 0.08 if (it < phase0_end) else 0.02
            if vis_floor > 0.0:
                vis_pos = visibility > 0.0
                visibility = torch.where(vis_pos, visibility.clamp(min=vis_floor), visibility)
            visibility = visibility.clamp(0.0, 1.0)
            vis_accum.append(visibility.detach())

            # Densify stats (module hooks for policy).
            update_densify_stats(densify_stats, visibility.detach(), residual_scalar, float(tau_vis))
            _ = select_densify_candidates(
                densify_stats,
                visibility.detach(),
                int(V_min),
                float(densify_top_percent),
            )

            # Pass-2 render with per-gaussian visibility weighting.
            pred_rgb, pred_a = render_gaussians_3d(
                g, K, R, t, H, W, bg=bg_rgb, return_alpha=True, gaussian_weights=visibility.detach()
            )

            diff = pred_rgb - gt
            if bool(use_robust_loss):
                photo_map = compute_robust_photo_loss(
                    diff,
                    loss_type=robust_loss_type,
                    huber_delta=0.02,
                    charbonnier_eps=1e-3,
                )
            else:
                photo_map = diff * diff
            l1_map = diff.abs()

            if weight is not None:
                w_map = weight.expand_as(pred_rgb)
                photo = (photo_map * w_map).sum() / (w_map.sum() + 1e-6)
                l1 = (l1_map * w_map).sum() / (w_map.sum() + 1e-6)
            else:
                photo = photo_map.mean()
                l1 = l1_map.mean()

            outer_band_term = pred_rgb.new_tensor(0.0)
            if dq_enable:
                roi = gate_color_roi_mask
                abs_diff = diff.abs()
                if roi is not None and float(roi.sum().detach().cpu()) > 1.0:
                    roi3 = roi.expand_as(abs_diff)
                    color_err = (abs_diff * roi3).sum() / (roi3.sum() + 1e-6)
                    pix_err = abs_diff.mean(dim=0, keepdim=True)
                    pix_vals = pix_err[roi > 0.5]
                    if int(pix_vals.numel()) > 1:
                        color_var = torch.var(pix_vals, unbiased=False)
                    else:
                        color_var = abs_diff.new_tensor(0.0)
                else:
                    color_err = abs_diff.mean()
                    color_var = abs_diff.new_tensor(0.0)
                color_eval_vals.append(float(color_err.detach().cpu()))
                color_var_vals.append(float(color_var.detach().cpu()))

                if gate_geom_target is not None:
                    pred_bin = (pred_a > max(float(alpha_threshold), 0.01)).float()
                    gt_bin = (gate_geom_target > 0.5).float()
                    geom_score = _fscore_binary_torch(pred_bin, gt_bin, gate_geom_band)
                    geom_eval_vals.append(float(geom_score.detach().cpu()))

                if bool(shell_enable) and gate_geom_band is not None and float(dq_outer_color_band_weight) > 0.0:
                    outer_band_term = outer_color_band_loss(pred_rgb, gt, gate_geom_band)

            frame_loss = photo + float(l1_weight) * l1 + float(dq_outer_color_band_weight) * outer_band_term

            leak_mask = smpl_mask
            if leak_mask is None and person_masks_frames is not None:
                leak_mask = person_masks_frames[fj]
            if leak_mask is not None and float(smpl_alpha_leak_weight) > 0.0:
                outside = (1.0 - leak_mask).clamp(0.0, 1.0)
                leak = (pred_a * outside).sum() / (outside.sum() + 1e-6)
                frame_loss = frame_loss + float(smpl_alpha_leak_weight) * leak

            if frame_loss_weights_t is not None:
                frame_loss = frame_loss * float(frame_loss_weights_t[fj].item())
            frame_loss = float(wj) * float(frame_w_scalar) * frame_loss

            loss_value += float(frame_loss.detach().cpu())
            frame_loss.backward()

            if (
                binding_enabled
                and bind_mask_curr is not None
                and bool(bind_mask_curr.any())
                and bind_surface_aniso_weight > 0.0
            ):
                ls = g.log_scales[bind_mask_curr]
                tang_max = torch.maximum(ls[:, 0], ls[:, 1])
                pen = F.relu(ls[:, 2] - tang_max + float(bind_surface_aniso_margin))
                aniso_loss = float(bind_surface_aniso_weight) * float(wj) * (pen * pen).mean()
                loss_value += float(aniso_loss.detach().cpu())
                aniso_loss.backward()

        vis_mean = torch.stack(vis_accum, dim=0).mean(dim=0) if len(vis_accum) > 0 else torch.ones((n_gauss,), device=device)

        if dq_enable:
            eval_due = ((it + 1) >= int(dq_warmup_iters)) and (((it + 1) % int(dq_eval_every)) == 0)
            if eval_due:
                dq_last_color_error = float(np.mean(color_eval_vals)) if len(color_eval_vals) > 0 else 0.0
                dq_last_color_var = float(np.mean(color_var_vals)) if len(color_var_vals) > 0 else 0.0
                dq_last_geom_score = float(np.mean(geom_eval_vals)) if len(geom_eval_vals) > 0 else 0.0

                color_pass = (dq_last_color_error <= float(dq_color_max_error)) and (
                    dq_last_color_var <= float(dq_color_unstable_var_max)
                )
                geom_pass = dq_last_geom_score >= float(dq_geom_min_score)
                gate_pass = (color_pass and geom_pass) if dq_mode == "hard_and" else (color_pass or geom_pass)

                if gate_pass:
                    dq_pass_streak += 1
                    if dq_pass_streak >= int(dq_recover_patience):
                        dq_fail_active = False
                else:
                    dq_pass_streak = 0
                    dq_fail_active = True

            color_conf_now = max(0.0, min(1.0, 1.0 - (dq_last_color_error / max(float(dq_color_max_error), 1e-6))))
            geom_conf_now = max(0.0, min(1.0, dq_last_geom_score / max(float(dq_geom_min_score), 1e-6)))
            g.color_conf_ema = 0.9 * g.color_conf_ema + 0.1 * float(color_conf_now)
            g.geom_conf_ema = 0.9 * g.geom_conf_ema + 0.1 * float(geom_conf_now)

            if dq_fail_active:
                outer_fail_m = (g.shell_layer == 1) & (g.prune_state != 2)
                g.quality_fail_counter[outer_fail_m] = g.quality_fail_counter[outer_fail_m] + 1.0

        # Shell/occlusion policy updates (state-based, non-destructive).
        if bool(shell_enable):
            inner_mask = (g.shell_layer == 0)
            outer_mask = (g.shell_layer == 1)
            active_mask = (g.prune_state != 2)

            outer_vis_scalar = float(vis_mean[outer_mask].mean().detach().cpu()) if bool(outer_mask.any()) else 0.0
            outer_cov_vec = torch.full_like(vis_mean, outer_vis_scalar)

            inner_soft_cond = select_inner_softfreeze_candidates(
                inner_mask=inner_mask,
                inner_vis=vis_mean,
                inner_contrib=vis_mean,
                outer_cover_vis=outer_cov_vec,
                outer_cover_contrib=outer_cov_vec,
                soft_vis_th=float(inner_soft_vis_th),
                soft_contrib_th=float(inner_soft_contrib_th),
                outer_vis_th=float(outer_cover_vis_th),
                outer_contrib_th=float(outer_cover_contrib_th),
            )
            inner_soft_counter = torch.where(inner_soft_cond, inner_soft_counter + 1.0, torch.zeros_like(inner_soft_counter))
            inner_hard_counter = torch.where(inner_soft_cond, inner_hard_counter + 1.0, torch.zeros_like(inner_hard_counter))

            if bool(inner_occlusion_soft_enable):
                to_soft = inner_mask & (inner_soft_counter >= float(max(int(inner_soft_patience), 1)))
                g.prune_state[to_soft] = 1

            if bool(inner_occlusion_hard_enable) and int(inner_prune_every) > 0 and ((it + 1) % int(inner_prune_every) == 0):
                to_hard = select_inner_hardprune_candidates(
                    inner_mask=inner_mask,
                    soft_counter=inner_soft_counter,
                    hard_counter=inner_hard_counter,
                    soft_patience=int(inner_soft_patience),
                    hard_patience=int(inner_hard_patience),
                )
                g.prune_state[to_hard] = 2

            # Outer score / prune policy.
            edge_metric = (g.bind_mix_weight.view(-1).detach() - 0.5).abs() * 2.0
            shell_score = compute_shell_score(
                residual_ema=densify_stats.residual_ema,
                edge=edge_metric,
                curv_conf=g.bind_mix_weight.view(-1).detach(),
                leak=(1.0 - vis_mean.detach()).clamp(0.0, 1.0),
            )
            if dq_enable:
                res_top = float(np.clip(dq_geom_residual_top_percent, 0.0, 100.0)) / 100.0
                edge_top = float(np.clip(dq_geom_edge_top_percent, 0.0, 100.0)) / 100.0
                if res_top > 0.0 and edge_top > 0.0:
                    res_q = torch.quantile(densify_stats.residual_ema.detach(), max(1e-6, 1.0 - res_top))
                    edge_q = torch.quantile(edge_metric.detach(), max(1e-6, 1.0 - edge_top))
                    geom_focus = (densify_stats.residual_ema.detach() >= res_q) & (edge_metric.detach() >= edge_q)
                    shell_score = shell_score * (1.0 + 0.25 * geom_focus.float())

            prune_low_vis_th = 0.02
            prune_low_score_percent = 10.0
            if dq_enable and dq_fail_active:
                prune_low_vis_th = min(1.0, prune_low_vis_th * float(dq_prune_boost_if_fail))
                prune_low_score_percent = min(100.0, prune_low_score_percent * float(dq_prune_boost_if_fail))

            if int(shell_prune_every) > 0 and ((it + 1) % int(shell_prune_every) == 0):
                to_prune_outer = select_outer_prune_candidates(
                    outer_mask=outer_mask,
                    outer_vis=vis_mean,
                    outer_score=shell_score,
                    low_vis_th=prune_low_vis_th,
                    low_score_percent=prune_low_score_percent,
                )
                if int(to_prune_outer.numel()) > 0:
                    g.prune_state[to_prune_outer] = 2

            if int(shell_spawn_every) > 0 and ((it + 1) % int(shell_spawn_every) == 0):
                n_inner_active = int((inner_mask & active_mask).sum().item())
                n_outer_active = int((outer_mask & active_mask).sum().item())
                _outer_init_budget, outer_budget = estimate_outer_budget(
                    n_inner=max(n_inner_active, 1),
                    init_ratio=0.15,
                    max_ratio=float(shell_max_ratio),
                )
                max_new = max(0, outer_budget - n_outer_active)
                if dq_enable and dq_fail_active:
                    max_new = int(round(float(max_new) * float(max(dq_spawn_scale_if_fail, 0.0))))
                if max_new > 0:
                    spawn_src = select_outer_spawn_candidates(
                        shell_score=shell_score,
                        visibility=vis_mean,
                        vis_count=densify_stats.vis_count,
                        V_min=int(V_min),
                        tau_vis=float(tau_vis),
                        top_percent=float(shell_spawn_top_percent),
                        max_new=max_new,
                        active_mask=active_mask,
                    )
                    # Reuse previously hard-pruned outer slots to keep tensor shapes stable.
                    spawn_slots = torch.nonzero((g.shell_layer == 1) & (g.prune_state == 2), as_tuple=False).view(-1)
                    n_spawn = min(int(spawn_src.numel()), int(spawn_slots.numel()))
                    if n_spawn > 0:
                        src = spawn_src[:n_spawn]
                        dst = spawn_slots[:n_spawn]
                        with torch.no_grad():
                            g.log_scales[dst] = g.log_scales[src]
                            g.quat[dst] = g.quat[src]
                            g.opacity_logit[dst] = g.opacity_logit[src]
                            g.rgb_logit[dst] = g.rgb_logit[src]
                            if g.bind_idx is not None:
                                g.bind_idx[dst] = g.bind_idx[src]
                            if g.bind_offset is not None:
                                g.bind_offset[dst] = g.bind_offset[src]
                            if g.bind_face_idx is not None:
                                g.bind_face_idx[dst] = g.bind_face_idx[src]
                            if g.bind_bary is not None:
                                g.bind_bary[dst] = g.bind_bary[src]
                            g.bind_face_offset_local.data[dst] = g.bind_face_offset_local.data[src]
                            g.bind_face_offset_local.data[dst, 2] = g.bind_face_offset_local.data[dst, 2].clamp(
                                -float(shell_offset_max_n), float(shell_offset_max_n)
                            )
                            g.prune_state[dst] = 0

        # Global regularizers (stronger on low-visibility gaussians).
        low_vis_boost = 1.0 + float((1.0 - vis_mean).mean().detach().cpu())
        phase_alpha_mul = 2.0 if phase_tag == "phase0" else 1.0
        reg_opacity = (
            float(alpha_prior_weight)
            * phase_alpha_mul
            * low_vis_boost
            * opacity_prior_loss(g.opacity(), target=0.5)
        )
        reg_cov = float(cov_penalty_weight) * low_vis_boost * covariance_penalty(g.scales(), float(cov_min), float(cov_max))
        reg_offset = float(shell_offset_reg_weight) * canonical_offset_l2(g.bind_face_offset_local)
        alpha_cap_for_balance = float(dq_outer_alpha_cap_if_fail) if (dq_enable and dq_fail_active) else 0.40
        reg_alpha_balance = float(dq_layer_alpha_balance_weight) * layer_alpha_balance_loss(
            g.opacity(),
            g.shell_layer,
            outer_alpha_cap=alpha_cap_for_balance,
            inner_alpha_floor=0.03,
        )

        reg_total = reg_opacity + reg_cov + reg_offset + reg_alpha_balance
        loss_value += float(reg_total.detach().cpu())
        reg_total.backward()

        if learn_cam and cam_reg_weight > 0:
            # Keep camera near prior (prevents drift)
            reg = (cam_w ** 2).mean() + ((cam_t - cam_t_init) ** 2).mean()
            reg_loss = float(cam_reg_weight) * reg
            loss_value += float(reg_loss.detach().cpu())
            reg_loss.backward()
        # If we want background to be low-precision, down-scale its gradients.
        if binding_enabled and bg_grad_scale < 1.0:
            bg_mask = ~bind_mask
            with torch.no_grad():
                for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit):
                    if p.grad is None:
                        continue
                    p.grad[bg_mask] *= float(bg_grad_scale)

        soft_mask = g.prune_state == 1
        hard_mask = g.prune_state == 2
        if bool(soft_mask.any()) or bool(hard_mask.any()):
            with torch.no_grad():
                for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit, g.bind_face_offset_local):
                    if p.grad is None:
                        continue
                    if bool(soft_mask.any()):
                        p.grad[soft_mask] *= float(inner_soft_lr_scale)
                    if bool(hard_mask.any()):
                        p.grad[hard_mask] = 0.0

        if dq_enable and dq_fail_active and bool(shell_enable):
            outer_active = (g.shell_layer == 1) & (g.prune_state != 2)
            if bool(outer_active.any()):
                with torch.no_grad():
                    for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit, g.bind_face_offset_local):
                        if p.grad is None:
                            continue
                        p.grad[outer_active] *= float(dq_outer_lr_scale_if_fail)

        if phase_tag == "phase0":
            with torch.no_grad():
                if g.log_scales.grad is not None:
                    g.log_scales.grad.zero_()
                if g.bind_face_offset_local.grad is not None:
                    g.bind_face_offset_local.grad.zero_()
        opt.step()

        # Mild regularization: keep parameters in a reasonable range
        with torch.no_grad():
            g.log_scales.clamp_(min=-7.0, max=0.0)      # exp range ~ [0.0009, 1]
            g.opacity_logit.clamp_(min=-8.0, max=8.0)
            if binding_enabled:
                bg_mask = ~bind_mask
                # Foreground constraints (bound gaussians)
                if fg_opacity_logit_max is not None:
                    g.opacity_logit[bind_mask] = g.opacity_logit[bind_mask].clamp(
                        max=float(fg_opacity_logit_max)
                    )
                if fg_log_scale_min is not None or fg_log_scale_max is not None:
                    min_v = -7.0 if fg_log_scale_min is None else float(fg_log_scale_min)
                    max_v = 0.0 if fg_log_scale_max is None else float(fg_log_scale_max)
                    g.log_scales[bind_mask] = g.log_scales[bind_mask].clamp(
                        min=min_v, max=max_v
                    )
                if bg_opacity_logit_max is not None:
                    g.opacity_logit[bg_mask] = g.opacity_logit[bg_mask].clamp(
                        max=float(bg_opacity_logit_max)
                    )
                if bg_log_scale_min is not None or bg_log_scale_max is not None:
                    min_v = -7.0 if bg_log_scale_min is None else float(bg_log_scale_min)
                    max_v = 0.0 if bg_log_scale_max is None else float(bg_log_scale_max)
                    g.log_scales[bg_mask] = g.log_scales[bg_mask].clamp(
                        min=min_v, max=max_v
                    )
            # Optional: keep translations in a sane range to prevent explosion
            if learn_cam:
                cam_t[:, 2].clamp_(min=0.5, max=10.0)
            g.bind_mix_weight.clamp_(min=float(mix_w_min), max=float(mix_w_max))
            g.bind_face_offset_local[:, 0].clamp_(min=-float(shell_offset_max_t), max=float(shell_offset_max_t))
            g.bind_face_offset_local[:, 1].clamp_(min=-float(shell_offset_max_t), max=float(shell_offset_max_t))
            g.bind_face_offset_local[:, 2].clamp_(min=-float(shell_offset_max_n), max=float(shell_offset_max_n))
            if dq_enable and dq_fail_active and bool(shell_enable):
                outer_active = (g.shell_layer == 1) & (g.prune_state != 2)
                if bool(outer_active.any()):
                    cap = float(np.clip(dq_outer_alpha_cap_if_fail, 1e-3, 0.999))
                    cap_logit = float(np.log(cap / (1.0 - cap)))
                    g.opacity_logit[outer_active] = g.opacity_logit[outer_active].clamp(max=cap_logit)
            if bool((g.prune_state == 2).any()):
                g.opacity_logit[g.prune_state == 2] = g.opacity_logit[g.prune_state == 2].clamp(max=-6.0)

        if es_enable:
            if loss_value + es_min_delta < es_best:
                es_best = loss_value
                es_bad = 0
            elif it >= es_warmup:
                es_bad += 1
                if es_bad >= es_patience:
                    print(
                        f"Early stop at iter {it + 1}/{iters}: "
                        f"best_loss={es_best:.6f}, last_loss={loss_value:.6f}, "
                        f"patience={es_patience}, min_delta={es_min_delta}"
                    )
                    break

    # Save checkpoint
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gaussians": g.state_dict(),
            "K": K.detach().cpu(),
            "cam_w": cam_w.detach().cpu() if torch.is_tensor(cam_w) else torch.tensor(cam_w),
            "cam_t": cam_t.detach().cpu() if torch.is_tensor(cam_t) else torch.tensor(cam_t),
            "frames": frames_list,
            "H": H,
            "W": W,
            "bg_rgb": bg_rgb,
            "cam_warmup_iters": cam_warmup_iters,
            "l1_weight": l1_weight,
            "use_alpha_mask": use_alpha_mask,
            "alpha_threshold": alpha_threshold,
            "camera_mode": camera_mode,
            "cam_t_prior": cam_t_prior,
            "cam_reg_weight": cam_reg_weight,
            "keypoints_json": str(keypoints_json) if keypoints_json is not None else None,
            "kpt_conf_min": kpt_conf_min,
            "kpt_box_pad": kpt_box_pad,
            "use_kpt_bbox_mask": use_kpt_bbox_mask,
            "face_weight": face_weight,
            "face_conf_min": face_conf_min,
            "face_box_pad": face_box_pad,
            "face_use_ears": face_use_ears,
            "frame_weights": frame_weights,
            "frame_loss_weights": frame_loss_weights,
            "temporal_ref_window": temporal_ref_window,
            "temporal_ref_decay": temporal_ref_decay,
            "temporal_ref_anchor": temporal_ref_anchor,
            "bind_mesh_dir": str(bind_mesh_dir) if binding_enabled else None,
            "bind_mesh_indices": bind_mesh_indices if binding_enabled else None,
            "bind_sequence_npz": str(bind_seq_npz) if (binding_enabled and bind_seq_npz is not None) else None,
            "bind_vertices_npz": str(bind_vertices_npz) if (binding_enabled and bind_vertices_npz is not None) else None,
            "bind_deform_obj": str(bind_deform_obj) if (binding_enabled and bind_deform_obj is not None) else None,
            "bind_mesh_obj_space": bind_mesh_obj_space,
            "bind_align_quat_to_mesh_normals": bind_align_quat_to_mesh_normals,
            "bind_quat_normal_blend": bind_quat_normal_blend,
            "bind_surface_aniso_weight": bind_surface_aniso_weight,
            "bind_surface_aniso_margin": bind_surface_aniso_margin,
            "bind_rigid_align": bind_rigid_align_meta if binding_enabled else None,
            "binding": export_binding_mode_config(binding_cfg),
            "use_smpl_silhouette_mask": use_smpl_silhouette_mask,
            "smpl_mask_dilate_px": smpl_mask_dilate_px,
            "smpl_mask_erode_px": smpl_mask_erode_px,
            "smpl_color_mask_erode_px": smpl_color_mask_erode_px,
            "smpl_alpha_leak_weight": smpl_alpha_leak_weight,
            "smpl_mask_override_alpha": smpl_mask_override_alpha,
            "smpl_mask_precompute": smpl_mask_precompute,
            "smpl_mask_precompute_max_frames": smpl_mask_precompute_max_frames,
            "person_masks_npz": str(person_masks_npz) if person_masks_npz is not None else None,
            "use_person_mask": use_person_mask,
            "person_mask_dilate_px": person_mask_dilate_px,
            "person_mask_erode_px": person_mask_erode_px,
            "person_mask_fuse_mode": person_mask_fuse_mode,
            "separate_fg_bg_training": separate_fg_bg_training,
            "fg_region_weight": fg_region_weight,
            "bg_region_weight": bg_region_weight,
            "bg_grad_scale": bg_grad_scale,
            "bg_opacity_logit_max": bg_opacity_logit_max,
            "bg_log_scale_min": bg_log_scale_min,
            "bg_log_scale_max": bg_log_scale_max,
            "fg_opacity_logit_max": fg_opacity_logit_max,
            "fg_log_scale_min": fg_log_scale_min,
            "fg_log_scale_max": fg_log_scale_max,
            "early_stop_enable": early_stop_enable,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_warmup_iters": early_stop_warmup_iters,
            "dual_quality_gate": dual_quality_gate,
            "dual_quality_gate_last": {
                "color_error": dq_last_color_error,
                "color_var": dq_last_color_var,
                "geometry_score": dq_last_geom_score,
                "fail_active": dq_fail_active,
            },
        },
        out_ckpt,
    )











