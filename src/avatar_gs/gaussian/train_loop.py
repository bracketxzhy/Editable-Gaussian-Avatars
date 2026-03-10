from __future__ import annotations

from pathlib import Path
import json
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.binding import (
    BindingFrameData,
    export_binding_mode_config,
    parse_binding_mode_config,
)
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.gaussian.binding_layers import (
    apply_inner_binding_per_frame,
    apply_outer_anchor_per_frame,
)
from avatar_gs.losses import (
    canonical_offset_l2,
    compute_robust_photo_loss,
    covariance_penalty,
    layer_alpha_balance_loss,
    opacity_prior_loss,
    outer_color_band_loss,
)
from avatar_gs.training import (
    FrameResidualEMA,
    build_outer_trigger_mask,
    estimate_face_gap_targets,
    frame_weight_from_residual,
    map_trigger_ring_to_faces,
    resolve_inner_supervision_mask,
)
from avatar_gs.gaussian.visibility import (
    build_border_attenuation,
    compute_gaussian_visibility,
    sample_border_weight,
)
from avatar_gs.gaussian.densify import (
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
)
from avatar_gs.stage2 import (
    Stage2FrameAssets,
    Stage2MeshRuntime,
    build_outer_face_records,
    build_outer_spawn_report,
    build_stage2_training_supervision_state,
    compute_contour_band_torch as _compute_contour_band_torch,
    compute_outer_dynamic_caps,
    default_stage2_quality_thresholds,
    estimate_gt_person_contour_mask_np as _estimate_gt_person_contour_mask_np,
    evaluate_stage2_quality,
    fuse_outer_contour_mask_np,
    initialize_stage2_training_binding_state,
    project_points_numpy as _project_points_numpy,
    project_points_torch as _project_points_torch,
    rasterize_mesh_mask as _rasterize_mesh_mask,
    write_stage2_json_report,
    write_stage2_quality_log,
)

class _TrainLoopLogger:
    def info(self, msg: str, *args) -> None:
        print(msg % args if args else msg)

    def warning(self, msg: str, *args) -> None:
        print((msg % args if args else msg))


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


def _barycentric_coords_torch(pts: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    v0 = tri[..., 1, :] - tri[..., 0, :]
    v1 = tri[..., 2, :] - tri[..., 0, :]
    v2 = pts - tri[..., 0, :]
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-8)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


def _compute_outer_airspace_penalty(
    *,
    centers_world: torch.Tensor,
    tangent_radius: torch.Tensor,
    bind_face_idx: torch.Tensor,
    face_centroids: torch.Tensor,
    face_normals: torch.Tensor,
    face_tris: torch.Tensor,
    neighbor_idx: torch.Tensor | None,
    neighbor_w: torch.Tensor | None,
    main_support_scale: float,
) -> torch.Tensor:
    if (
        neighbor_idx is None
        or neighbor_w is None
        or int(neighbor_idx.numel()) <= 0
        or int(centers_world.shape[0]) <= 0
    ):
        return centers_world.new_tensor(0.0)
    valid_face = (bind_face_idx >= 0) & (bind_face_idx < int(face_centroids.shape[0]))
    if not bool(valid_face.any()):
        return centers_world.new_tensor(0.0)
    face_ids = bind_face_idx[valid_face]
    neigh = neighbor_idx.index_select(0, face_ids)
    neigh_weights = neighbor_w.index_select(0, face_ids)
    valid_nb = neigh >= 0
    if not bool(valid_nb.any()):
        return centers_world.new_tensor(0.0)
    safe_neigh = neigh.clamp_min(0)
    tri_nb = face_tris.index_select(0, safe_neigh.reshape(-1)).reshape(*safe_neigh.shape, 3, 3)
    ctr_nb = face_centroids.index_select(0, safe_neigh.reshape(-1)).reshape(*safe_neigh.shape, 3)
    nrm_nb = face_normals.index_select(0, safe_neigh.reshape(-1)).reshape(*safe_neigh.shape, 3)
    ctr = centers_world[valid_face].unsqueeze(1)
    signed_h = ((ctr - ctr_nb) * nrm_nb).sum(dim=-1)
    proj = ctr - signed_h.unsqueeze(-1) * nrm_nb
    bary = _barycentric_coords_torch(proj, tri_nb)
    edge01 = (tri_nb[..., 1, :] - tri_nb[..., 0, :]).norm(dim=-1)
    edge12 = (tri_nb[..., 2, :] - tri_nb[..., 1, :]).norm(dim=-1)
    edge20 = (tri_nb[..., 0, :] - tri_nb[..., 2, :]).norm(dim=-1)
    edge_mean = ((edge01 + edge12 + edge20) / 3.0).clamp_min(1e-6)
    margin = (
        float(max(main_support_scale, 0.0))
        * tangent_radius[valid_face].unsqueeze(1)
        / edge_mean
    ).clamp(min=0.0, max=4.0)
    inside_expanded = bary.min(dim=-1).values >= (-margin)
    invade = F.relu(signed_h) * inside_expanded.float() * neigh_weights * valid_nb.float()
    denom = valid_nb.float().sum().clamp_min(1.0)
    return (invade * invade).sum() / denom


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


def _compute_image_edge_strength_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Estimate image-edge strength map from RGB frame.
    Input: (3,H,W) in [0,1], output: (1,H,W) in [0,1].
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"rgb must be (3,H,W), got {rgb.shape}")
    dev = rgb.device
    dt = rgb.dtype
    gray = (0.299 * rgb[0:1] + 0.587 * rgb[1:2] + 0.114 * rgb[2:3]).clamp(0.0, 1.0)
    gx_k = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dt,
        device=dev,
    ).view(1, 1, 3, 3)
    gy_k = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dt,
        device=dev,
    ).view(1, 1, 3, 3)
    g = gray.unsqueeze(0)
    gx = F.conv2d(g, gx_k, stride=1, padding=1)
    gy = F.conv2d(g, gy_k, stride=1, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0)
    scale = mag.mean() + 2.0 * mag.std(unbiased=False)
    return (mag / scale.clamp(min=1e-6)).clamp(0.0, 1.0)


def _image_laplacian_chw(rgb_chw: torch.Tensor) -> torch.Tensor:
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
        raise ValueError(f"rgb_chw must be (3,H,W), got {rgb_chw.shape}")
    dev = rgb_chw.device
    dt = rgb_chw.dtype
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=dt,
        device=dev,
    ).view(1, 1, 3, 3)
    k = k.repeat(3, 1, 1, 1)
    x = rgb_chw.unsqueeze(0)
    y = F.conv2d(x, k, stride=1, padding=1, groups=3)
    return y.squeeze(0)


def _sample_mask_at_projected_uv(
    mask: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    out = torch.zeros_like(u, dtype=torch.float32)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if int(idx.numel()) <= 0:
        return out
    uu = torch.round(u[idx]).long().clamp(0, int(W) - 1)
    vv = torch.round(v[idx]).long().clamp(0, int(H) - 1)
    out[idx] = mask[0, vv, uu]
    return out.clamp(0.0, 1.0)


def _make_gaussian_with_xyz(g: Gaussian3D, xyz: torch.Tensor) -> Gaussian3D:
    return Gaussian3D(
        xyz=xyz,
        log_scales=g.log_scales,
        quat=g.quat,
        opacity_logit=g.opacity_logit,
        rgb_logit=g.rgb_logit,
        bind_idx=g.bind_idx,
        bind_offset=g.bind_offset,
        bind_face_idx=g.bind_face_idx,
        bind_bary=g.bind_bary,
        bind_hybrid_w=g.bind_hybrid_w,
        bind_face_offset_local=g.bind_face_offset_local,
        shell_layer=g.shell_layer,
        bind_mix_weight=g.bind_mix_weight,
        prune_state=g.prune_state,
        inner_bind_mode=g.inner_bind_mode,
        quality_fail_counter=g.quality_fail_counter,
        color_conf_ema=g.color_conf_ema,
        geom_conf_ema=g.geom_conf_ema,
    )


def _inject_bind_face_offset_straight_through_xyz(
    xyz_base: torch.Tensor,
    g: Gaussian3D,
    frame_data: BindingFrameData | None,
    bind_mask_curr: torch.Tensor | None,
    bind_face_idx_t: torch.Tensor | None,
) -> torch.Tensor:
    """
    Straight-through gradient path for bind_face_offset_local:
    forward value stays unchanged, backward carries dL/d(offset).
    """
    if frame_data is None or bind_mask_curr is None or bind_face_idx_t is None:
        return xyz_base
    if g.bind_face_offset_local is None:
        return xyz_base
    if frame_data.faces_t is None or frame_data.face_t1_t is None or frame_data.face_t2_t is None or frame_data.face_n_t is None:
        return xyz_base
    if not bool(bind_mask_curr.any()):
        return xyz_base

    bound_ids = torch.nonzero(bind_mask_curr, as_tuple=False).view(-1)
    if int(bound_ids.numel()) <= 0:
        return xyz_base

    f_idx = bind_face_idx_t.index_select(0, bound_ids)
    n_faces = int(frame_data.faces_t.shape[0])
    f_valid = (f_idx >= 0) & (f_idx < n_faces)
    if g.shell_layer is not None:
        sl_bound = g.shell_layer.index_select(0, bound_ids).to(dtype=torch.int64, device=xyz_base.device)
        f_valid = f_valid & (sl_bound == 1)
    if not bool(f_valid.any()):
        return xyz_base

    local = torch.nonzero(f_valid, as_tuple=False).view(-1)
    ids = bound_ids.index_select(0, local)
    fs = f_idx.index_select(0, local).long()
    off = g.bind_face_offset_local.index_select(0, ids)
    off_st = off - off.detach()  # zero in forward, non-zero in backward

    t1 = frame_data.face_t1_t.index_select(0, fs)
    t2 = frame_data.face_t2_t.index_select(0, fs)
    nf = frame_data.face_n_t.index_select(0, fs)
    dxyz_st = (
        off_st[:, 0:1] * t1
        + off_st[:, 1:2] * t2
        + off_st[:, 2:3] * nf
    )
    if int(dxyz_st.numel()) <= 0:
        return xyz_base

    xyz_eff = xyz_base.clone()
    xyz_local = xyz_eff.index_select(0, ids)
    xyz_eff = xyz_eff.index_copy(0, ids, xyz_local + dxyz_st)
    return xyz_eff


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
    init_cam_w: torch.Tensor | None = None,
    init_cam_t: torch.Tensor | None = None,
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
    use_kpt_bbox_mask: bool = False,
    face_weight: float = 1.0,
    face_conf_min: float = 0.3,
    face_box_pad: float = 0.2,
    face_use_ears: bool = False,
    hand_weight: float = 1.0,
    hand_conf_min: float = 0.25,
    hand_radius_scale: float = 0.45,
    hand_radius_min_px: int = 10,
    hand_extension_scale: float = 0.35,
    outer_exclude_hands: bool = False,
    outer_hand_expand_px: int = 3,
    frame_weights: list[float] | None = None,
    frame_loss_weights: list[float] | None = None,
    frame_align_scores: list[float] | None = None,
    temporal_ref_window: int = 0,
    temporal_ref_decay: float = 0.85,
    temporal_ref_anchor: str = "sample",
    bind_vertex_indices: torch.Tensor | None = None,
    bind_cache_size: int = 8,
    bind_align_quat_to_mesh_normals: bool = True,
    bind_quat_normal_blend: float = 1.0,
    bind_surface_aniso_weight: float = 0.0,
    bind_surface_aniso_margin: float = 0.15,
    use_smpl_silhouette_mask: bool = False,
    smpl_mask_dilate_px: int = 0,
    smpl_mask_erode_px: int = 0,
    smpl_color_mask_erode_px: int = 0,
    inner_use_strict_smpl_mask: bool = True,
    smpl_alpha_leak_weight: float = 0.0,
    smpl_mask_override_alpha: bool = True,
    smpl_mask_precompute: bool = True,
    smpl_mask_precompute_max_frames: int = 64,
    person_masks_npz: str | Path | None = None,
    prepared_supervision: dict | None = None,
    use_person_mask: bool = False,
    person_mask_dilate_px: int = 0,
    person_mask_erode_px: int = 0,
    person_mask_fuse_mode: str = "intersect",
    person_mask_eval_clean: bool = False,
    person_mask_eval_refine_modnet: bool = False,
    person_mask_contour_width_px: int = 2,
    person_mask_contour_leak_weight: float = 0.0,
    outer_use_gt_contour: bool = True,
    outer_gt_contour_fuse_mode: str = "union",
    outer_gt_contour_modnet_ref_size: int = 512,
    outer_gt_contour_modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
    outer_head_priority_enable: bool = False,
    outer_head_priority_weight: float = 1.5,
    outer_head_priority_face_pad: float = 0.45,
    outer_head_priority_up_scale: float = 2.4,
    outer_head_priority_down_scale: float = 0.35,
    outer_head_priority_person_top_ratio: float = 0.16,
    outer_head_support_th: float = 0.08,
    outer_trigger_smpl_gap_px: int = 3,
    outer_trigger_edge_th: float = 0.06,
    outer_trigger_min_pixels: int = 32,
    outer_trigger_ring_width_px: int = 2,
    outer_trigger_use_edge_gate: bool = True,
    outer_contour_inside_only: bool = False,
    outer_detached_solid_enable: bool = False,
    outer_detached_solid_dilate_px: int = 0,
    outer_detached_solid_close_px: int = 0,
    outer_detached_solid_mix_start: float = 0.0,
    outer_detached_solid_mix_end: float = 0.0,
    outer_head_support_sample_dilate_px: int = 0,
    outer_exclude_face: bool = False,
    outer_face_expand_px: int = 0,
    outer_face_exclude_lower_ratio: float = 1.0,
    outer_silhouette_weight: float = 0.0,
    outer_silhouette_band_px: int = 2,
    outer_silhouette_band_weight: float = 1.0,
    detail_hf_weight: float = 0.0,
    detail_hf_exclude_face: bool = True,
    outer_support_weight: float = 1.2,
    outer_spawn_support_th: float = 0.02,
    outer_inside_support_th: float = 0.05,
    outer_outside_support_th: float = 0.5,
    outer_outside_opacity_cap: float = 0.02,
    outer_outside_offset_decay: float = 0.92,
    outer_outside_opacity_weight: float = 0.10,
    outer_outside_offset_weight: float = 0.02,
    outer_offset_suppress_low_support: bool = False,
    outer_offset_suppress_support_th: float = 0.0,
    outer_offset_suppress_decay: float = 1.0,
    outer_offset_suppress_opacity_cap: float = 1.0,
    outer_spawn_inside_only: bool = True,
    outer_phase_inner_vis_scale: float = 0.0,
    outer_phase_inner_full_mask: bool = False,
    outer_low_support_patience: int = 3,
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
    outer_log_scale_min: float | None = None,
    outer_log_scale_max: float | None = None,
    outer_prune_support_th: float = 0.0,
    outer_prune_dual_enable: bool = False,
    outer_prune_dual_max_vis: float = 0.06,
    outer_prune_dual_min_support: float = 0.001,
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
    inner_opacity_prior_target: float = 0.50,
    outer_opacity_prior_target: float = 0.12,
    cov_min: float = 5e-4,
    cov_max: float = 2.5e-1,
    cov_penalty_weight: float = 0.05,
    use_border_attenuation: bool = True,
    border_attenuation_width: int = 8,
    V_min: int = 8,
    densify_top_percent: float = 10.0,
    phase_ratio_inner_only: float = 0.20,
    phase_ratio_outer_only: float = 0.20,
    phase_ratio_occlusion_simplify: float = 0.30,
    phase_ratio_joint_finetune: float = 0.30,
    shell_enable: bool = False,
    shell_max_ratio: float = 0.40,
    shell_spawn_top_percent: float = 15.0,
    shell_spawn_every: int = 200,
    shell_prune_every: int = 400,
    shell_offset_max_n: float = 0.04,
    shell_offset_max_t: float = 0.015,
    outer_offset_unbounded: bool = True,
    outer_offset_auto_range_enable: bool = True,
    outer_offset_auto_range_min_scale: float = 0.35,
    outer_offset_auto_range_edge_mix: float = 0.75,
    outer_offset_edge_guided_reg: bool = True,
    outer_offset_reg_relax_support: bool = True,
    outer_offset_min_active_weight: float = 0.08,
    outer_offset_min_target_scale: float = 0.20,
    outer_offset_min_support_th: float = 0.02,
    shell_offset_reg_weight: float = 0.02,
    shell_offset_reg_phase0_scale: float = 0.25,
    shell_offset_freeze_phase0: bool = False,
    outer_offset_grad_scale: float = 3.0,
    outer_offset_edge_grad_scale: float = 2.0,
    outer_offset_grad_scale_max: float = 8.0,
    bind_face_offset_straight_through: bool = True,
    inner_bind_mode_default: str = "vertex",
    outer_init_from_inner: bool = False,
    outer_dormant_slot_ratio: float = 0.30,
    outer_trigger_face_activate_enable: bool = True,
    outer_trigger_face_sample_dilate_px: int = 0,
    outer_viewplane_init_enable: bool = True,
    outer_center_init_rule: str = "segment_midpoint",
    outer_center_phase1_normal_tune_enable: bool = True,
    outer_center_phase1_normal_tune_cap: float = 0.04,
    outer_center_freeze_after_phase1: bool = True,
    outer_contact_alpha_edge_th: float = 0.05,
    outer_detached_support_boost: float = 1.8,
    outer_detached_vis_boost: float = 1.5,
    outer_overlap_prune_offset_th: float = 0.004,
    outer_overlap_prune_detached_th: float = 0.02,
    outer_detached_support_th: float = 0.05,
    outer_detached_opacity_floor: float = 0.08,
    outer_detached_log_scale_floor: float = -5.8,
    outer_gap_search_px: int = 24,
    outer_gap_step_px: float = 1.0,
    outer_gap_probe_world: float = 0.01,
    outer_gap_target_support_th: float = 0.05,
    outer_gap_target_weight: float = 0.20,
    outer_gap_scale_weight: float = 0.10,
    outer_gap_tangent_scale_ratio: float = 0.45,
    outer_gap_tangent_scale_weight: float = 0.08,
    outer_gap_hard_target_enable: bool = False,
    outer_gap_hard_target_blend: float = 1.0,
    outer_gap_refine_steps: int = 4,
    outer_gap_fallback_enable: bool = True,
    outer_gap_fallback_support_th: float = 0.03,
    outer_gap_fallback_half_min_n: float = 0.010,
    outer_gap_fallback_half_max_n: float = 0.045,
    outer_geom_fraction: float = 0.65,
    outer_phase_residual_weight: float = 0.15,
    outer_trigger_face_topk: int = 0,
    outer_trigger_face_ema_decay: float = 0.85,
    outer_trigger_face_ema_weight: float = 0.60,
    outer_trigger_face_topk_min_support: float = 0.02,
    outer_airspace_penalty_weight: float = 0.04,
    outer_airspace_main_support_scale: float = 1.5,
    outer_airspace_sharp_allow_dot: float = 0.15,
    outer_airspace_neighbor_rings: int = 1,
    outer_edge_trigger_enable: bool = True,
    outer_edge_trigger_weight: float = 0.35,
    outer_boundary_enforce: str = "hard",
    outer_boundary_tol_px: int = 1,
    outer_boundary_check_every: int = 50,
    outer_boundary_alpha_cap: float = 0.01,
    outer_boundary_scale_decay: float = 0.85,
    outer_boundary_offset_decay: float = 0.90,
    outer_boundary_patience: int = 6,
    outer_boundary_geom_start_frac: float = 0.45,
    outer_offset_cap_geom_start_frac: float = 0.55,
    outer_edge_prune_protect_quantile: float = 0.90,
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
    outer_pruned_opacity_logit_max: float = -12.0,
    outer_quality_log_every: int = 100,
    outer_progress_log_every: int = 20,
    outer_zero_stop_enable: bool = False,
    outer_zero_stop_patience: int = 120,
    outer_zero_stop_warmup_iters: int = 200,
    outer_zero_stop_after_phase_inner: bool = True,
    dual_quality_gate: dict | None = None,
    stage2_frame_assets: Stage2FrameAssets | None = None,
    stage2_mesh_runtime: Stage2MeshRuntime | None = None,
    training_mode: str = "full",
    save_inner_checkpoint_path: str | Path | None = None,
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
    out_ckpt_path = Path(out_ckpt)

    # Intrinsics
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)

    # Load first frame for H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    mesh_runtime = stage2_mesh_runtime
    mesh_context = None if mesh_runtime is None else mesh_runtime.context
    bind_mesh_indices: list[int] = [] if mesh_context is None or mesh_context.bind_mesh_indices is None else [
        int(x) for x in mesh_context.bind_mesh_indices
    ]
    # Move gaussian to device and make trainable
    g = g.to(device)
    # If binding is enabled, we will override xyz per-frame from mesh; do NOT optimize xyz.
    binding_enabled = mesh_runtime is not None and bind_vertex_indices is not None
    if bool(use_smpl_silhouette_mask) and not binding_enabled:
        raise ValueError("use_smpl_silhouette_mask=true requires stage2_mesh_runtime + bind_vertex_indices.")
    if int(smpl_mask_dilate_px) < 0 or int(smpl_mask_erode_px) < 0 or int(smpl_color_mask_erode_px) < 0:
        raise ValueError(
            "smpl_mask_dilate_px/smpl_mask_erode_px/smpl_color_mask_erode_px must be >= 0. "
            f"Got {smpl_mask_dilate_px}/{smpl_mask_erode_px}/{smpl_color_mask_erode_px}"
        )
    strict_inner_mask = bool(inner_use_strict_smpl_mask)
    smpl_mask_dilate_eff = 0 if strict_inner_mask else int(smpl_mask_dilate_px)
    smpl_mask_erode_eff = 0 if strict_inner_mask else int(smpl_mask_erode_px)
    smpl_color_mask_erode_eff = 0 if strict_inner_mask else int(smpl_color_mask_erode_px)
    if int(outer_trigger_smpl_gap_px) < 0:
        raise ValueError(f"outer_trigger_smpl_gap_px must be >= 0. Got: {outer_trigger_smpl_gap_px}")
    if int(outer_trigger_min_pixels) < 0:
        raise ValueError(f"outer_trigger_min_pixels must be >= 0. Got: {outer_trigger_min_pixels}")
    if int(outer_trigger_ring_width_px) <= 0:
        raise ValueError(f"outer_trigger_ring_width_px must be > 0. Got: {outer_trigger_ring_width_px}")
    if float(outer_trigger_edge_th) < 0.0:
        raise ValueError(f"outer_trigger_edge_th must be >= 0. Got: {outer_trigger_edge_th}")
    center_init_rule = str(outer_center_init_rule).strip().lower()
    if center_init_rule not in ("segment_midpoint", "fixed"):
        raise ValueError(
            "outer_center_init_rule must be segment_midpoint|fixed. "
            f"Got: {outer_center_init_rule}"
        )
    inner_opacity_prior_target = float(np.clip(float(inner_opacity_prior_target), 1e-3, 0.999))
    outer_opacity_prior_target = float(np.clip(float(outer_opacity_prior_target), 1e-3, 0.999))
    outer_offset_min_active_weight = float(max(float(outer_offset_min_active_weight), 0.0))
    outer_offset_min_target_scale = float(np.clip(float(outer_offset_min_target_scale), 0.0, 1.5))
    outer_offset_min_support_th = float(np.clip(float(outer_offset_min_support_th), 0.0, 1.0))
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
    default_inner_mode = 1 if str(inner_bind_mode_default).strip().lower() == "face" else 0
    if g.inner_bind_mode is None:
        inner_mode_base = torch.full((n_gauss,), -1, dtype=torch.int64, device=device)
        inner_mode_base[g.shell_layer == 0] = int(default_inner_mode)
        g.inner_bind_mode = inner_mode_base
    else:
        g.inner_bind_mode = g.inner_bind_mode.to(device=device, dtype=torch.int64)
        inner_sel_init = (g.shell_layer == 0) & (g.inner_bind_mode < 0)
        if bool(inner_sel_init.any()):
            g.inner_bind_mode[inner_sel_init] = int(default_inner_mode)
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
    bind_mesh_obj_space_mode = "auto" if mesh_context is None else str(mesh_context.bind_mesh_obj_space_mode).strip().lower()
    if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        raise ValueError(
            f"bind_mesh_obj_space must be auto|camera|blender. Got: {bind_mesh_obj_space_mode}"
        )
    bind_quat_normal_blend = float(np.clip(bind_quat_normal_blend, 0.0, 1.0))
    bind_surface_aniso_weight = float(max(bind_surface_aniso_weight, 0.0))
    bind_surface_aniso_margin = float(max(bind_surface_aniso_margin, 0.0))
    default_bind_mode = "games_face" if (g.bind_face_idx is not None and g.bind_bary is not None) else "legacy"
    binding_cfg = parse_binding_mode_config(binding, default_mode=default_bind_mode)
    bind_rigid_align_cfg = None if mesh_context is None or mesh_context.bind_rigid_align_cfg is None else dict(mesh_context.bind_rigid_align_cfg)
    bind_rigid_align_meta: dict | None = None
    bind_rigid_align_enabled = bool(
        bind_rigid_align_cfg is not None and bool(bind_rigid_align_cfg.get("enabled", False))
    )
    if bind_rigid_align_enabled and not binding_enabled:
        print("Warning: bind_rigid_align is enabled but binding is disabled; ignoring rigid align.")
        bind_rigid_align_enabled = False

    training_mode = str(training_mode).strip().lower()
    if training_mode not in ("full", "inner_only", "outer_only"):
        raise ValueError(f"training_mode must be full|inner_only|outer_only. Got: {training_mode}")
    if training_mode == "outer_only" and (not bool(shell_enable)):
        raise ValueError("training_mode=outer_only requires shell_enable=true.")
    save_inner_checkpoint_path = (
        None if save_inner_checkpoint_path is None else Path(save_inner_checkpoint_path)
    )

    cam_t_init = torch.tensor([cam_t_prior], device=device).float().view(1, 3)
    cam_w = torch.zeros((nF, 3), device=device)
    cam_t = cam_t_init.repeat(nF, 1)
    if init_cam_w is not None:
        init_cam_w_t = torch.as_tensor(init_cam_w, dtype=torch.float32, device=device)
        if init_cam_w_t.shape != cam_w.shape:
            raise ValueError(
                f"init_cam_w shape must be {(nF, 3)}, got {tuple(init_cam_w_t.shape)}"
            )
        cam_w = init_cam_w_t.clone()
    if init_cam_t is not None:
        init_cam_t_t = torch.as_tensor(init_cam_t, dtype=torch.float32, device=device)
        if init_cam_t_t.shape != cam_t.shape:
            raise ValueError(
                f"init_cam_t shape must be {(nF, 3)}, got {tuple(init_cam_t_t.shape)}"
            )
        cam_t = init_cam_t_t.clone()
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

    if stage2_frame_assets is not None:
        gts = [gt.to(device) for gt in stage2_frame_assets.gts]
        if prepared_supervision is None:
            prepared_supervision = stage2_frame_assets.prepared_supervision
        if frame_align_scores is None:
            frame_align_scores = stage2_frame_assets.frame_align_scores
    else:
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
    hand_masks_frames: list[torch.Tensor | None] | None = None
    kpts_map: dict[str, list] | None = None
    if keypoints_json is not None:
        kp_path = Path(keypoints_json)
        if kp_path.exists():
            with kp_path.open("r", encoding="utf-8") as f:
                kpts = json.load(f)
            if isinstance(kpts, dict):
                kpts_map = {str(k): v for k, v in kpts.items()}
            # Build per-frame bbox (x0,y0,x1,y1)
            kpt_boxes = [] if use_kpt_bbox_mask else None
            face_boxes = [] if (float(face_weight) > 1.0 or bool(shell_enable)) else None
            hand_masks_frames = [] if (float(hand_weight) > 1.0 or bool(shell_enable)) else None
            # COCO-style body keypoints: 0..4 are nose/eyes/ears.
            # Do not include shoulders/wrists here, otherwise face ROI becomes upper-body ROI.
            face_ids = [0, 1, 2]  # nose + eyes
            if bool(face_use_ears):
                face_ids = face_ids + [3, 4]
            for fn in frames_list:
                arr = kpts.get(fn, None)
                if arr is None:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    if hand_masks_frames is not None:
                        hand_masks_frames.append(None)
                    continue
                xs = [p[0] for p in arr if p[2] >= kpt_conf_min]
                ys = [p[1] for p in arr if p[2] >= kpt_conf_min]
                if not xs or not ys:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    if hand_masks_frames is not None:
                        hand_masks_frames.append(None)
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
                    if fx and fy:
                        fx0, fx1 = min(fx), max(fx)
                        fy0, fy1 = min(fy), max(fy)
                        fbw = max(fx1 - fx0, 1.0)
                        fbh = max(fy1 - fy0, 1.0)
                        # Face keypoints can be near-collinear; enforce a minimum square-ish support.
                        side = max(fbw, fbh)
                        cx = 0.5 * (fx0 + fx1)
                        cy = 0.5 * (fy0 + fy1)
                        fx0 = cx - 0.5 * side
                        fx1 = cx + 0.5 * side
                        fy0 = cy - 0.55 * side
                        fy1 = cy + 0.45 * side
                        fx0 = max(0.0, fx0 - face_box_pad * side)
                        fx1 = min(W - 1.0, fx1 + face_box_pad * side)
                        fy0 = max(0.0, fy0 - face_box_pad * side)
                        fy1 = min(H - 1.0, fy1 + face_box_pad * side)
                        face_boxes.append((fx0, fy0, fx1, fy1))
                    else:
                        # Fallback: if nose exists, use shoulder span to estimate a compact head ROI.
                        if len(arr) > 0 and arr[0][2] >= face_conf_min:
                            nx = float(arr[0][0])
                            ny = float(arr[0][1])
                            has_ls = len(arr) > 5 and arr[5][2] >= face_conf_min
                            has_rs = len(arr) > 6 and arr[6][2] >= face_conf_min
                            if has_ls and has_rs:
                                shoulder_span = abs(float(arr[5][0]) - float(arr[6][0]))
                                side = max(12.0, 0.45 * shoulder_span)
                            else:
                                side = max(12.0, 0.08 * float(min(H, W)))
                            fx0 = max(0.0, nx - 0.5 * side)
                            fx1 = min(W - 1.0, nx + 0.5 * side)
                            fy0 = max(0.0, ny - 0.7 * side)
                            fy1 = min(H - 1.0, ny + 0.4 * side)
                            face_boxes.append((fx0, fy0, fx1, fy1))
                        else:
                            face_boxes.append(None)

                if hand_masks_frames is not None:
                    hmask = np.zeros((H, W), dtype=np.uint8)
                    # COCO ids: left/right elbow=7/8, wrist=9/10
                    for ei, wi in ((7, 9), (8, 10)):
                        if wi >= len(arr):
                            continue
                        wx, wy, wc = float(arr[wi][0]), float(arr[wi][1]), float(arr[wi][2])
                        if wc < float(hand_conf_min) or (not np.isfinite(wx)) or (not np.isfinite(wy)):
                            continue
                        radius = float(max(int(hand_radius_min_px), 1))
                        hx, hy = wx, wy
                        if ei < len(arr):
                            ex, ey, ec = float(arr[ei][0]), float(arr[ei][1]), float(arr[ei][2])
                            if ec >= float(hand_conf_min) and np.isfinite(ex) and np.isfinite(ey):
                                forearm = float(np.hypot(wx - ex, wy - ey))
                                radius = max(radius, float(hand_radius_scale) * forearm)
                                hx = wx + float(hand_extension_scale) * (wx - ex)
                                hy = wy + float(hand_extension_scale) * (wy - ey)
                        cx = int(np.clip(round(hx), 0, W - 1))
                        cy = int(np.clip(round(hy), 0, H - 1))
                        rr = int(max(1, round(radius)))
                        cv2.circle(hmask, (cx, cy), rr, 1, thickness=-1, lineType=cv2.LINE_AA)
                    if int(hmask.sum()) > 0:
                        hand_masks_frames.append(torch.from_numpy(hmask.astype(np.float32)).to(device=device).unsqueeze(0))
                    else:
                        hand_masks_frames.append(None)

    supervision_state = build_stage2_training_supervision_state(
        frames_dir=frames_dir,
        frames_list=[str(x) for x in frames_list],
        gts=gts,
        device=device,
        image_hw=(H, W),
        prepared_supervision=prepared_supervision,
        use_person_mask=bool(use_person_mask),
        person_masks_npz=person_masks_npz,
        person_mask_dilate_px=int(person_mask_dilate_px),
        person_mask_erode_px=int(person_mask_erode_px),
        person_mask_eval_clean=bool(person_mask_eval_clean),
        person_mask_eval_refine_modnet=bool(person_mask_eval_refine_modnet),
        person_mask_fuse_mode=person_mask_fuse_mode,
        person_mask_contour_width_px=int(person_mask_contour_width_px),
        person_mask_contour_leak_weight=float(person_mask_contour_leak_weight),
        outer_use_gt_contour=bool(outer_use_gt_contour),
        outer_gt_contour_fuse_mode=outer_gt_contour_fuse_mode,
        outer_gt_contour_modnet_ref_size=int(outer_gt_contour_modnet_ref_size),
        outer_gt_contour_modnet_threshold=float(outer_gt_contour_modnet_threshold),
        modnet_ckpt_path=modnet_ckpt_path,
        separate_fg_bg_training=bool(separate_fg_bg_training),
        fg_region_weight=float(fg_region_weight),
        bg_region_weight=float(bg_region_weight),
        face_boxes=face_boxes,
        outer_head_priority_face_pad=float(outer_head_priority_face_pad),
        outer_head_priority_up_scale=float(outer_head_priority_up_scale),
        outer_head_priority_down_scale=float(outer_head_priority_down_scale),
        outer_head_priority_person_top_ratio=float(outer_head_priority_person_top_ratio),
        frame_weights=frame_weights,
        frame_loss_weights=frame_loss_weights,
        frame_align_scores=frame_align_scores,
        out_ckpt_path=out_ckpt_path,
        logger=_TrainLoopLogger(),
    )
    person_masks_frames = supervision_state.person_masks_frames
    prepared_inner_masks_frames = supervision_state.prepared_inner_masks_frames
    prepared_outer_masks_frames = supervision_state.prepared_outer_masks_frames
    prepared_smpl_masks_frames = supervision_state.prepared_smpl_masks_frames
    prepared_gt_train_masks_frames = supervision_state.prepared_gt_train_masks_frames
    prepared_inner_rgb_targets_frames = supervision_state.prepared_inner_rgb_targets_frames
    prepared_outer_rgb_targets_frames = supervision_state.prepared_outer_rgb_targets_frames
    prepared_combined_rgb_targets_frames = supervision_state.prepared_combined_rgb_targets_frames
    outer_gt_masks_frames = supervision_state.outer_gt_masks_frames
    person_contour_band_frames = supervision_state.person_contour_band_frames
    outer_head_eval_masks_frames = supervision_state.outer_head_eval_masks_frames
    frame_weights_t = supervision_state.frame_weights_t
    frame_loss_weights_t = supervision_state.frame_loss_weights_t
    frame_align_scores_t = supervision_state.frame_align_scores_t
    person_fuse = supervision_state.person_fuse
    outer_contour_fuse = supervision_state.outer_contour_fuse

    ignored_outer_bias_flags: list[str] = []
    if bool(outer_head_priority_enable) or float(outer_head_priority_weight) > 0.0:
        ignored_outer_bias_flags.extend(
            [
                "outer_head_priority_enable",
                "outer_head_priority_weight",
                "outer_head_priority_face_pad",
                "outer_head_priority_up_scale",
                "outer_head_priority_down_scale",
                "outer_head_priority_person_top_ratio",
                "outer_head_support_sample_dilate_px",
            ]
        )
    if bool(outer_exclude_face):
        ignored_outer_bias_flags.extend(
            [
                "outer_exclude_face",
                "outer_face_expand_px",
                "outer_face_exclude_lower_ratio",
            ]
        )
    if bool(outer_exclude_hands):
        ignored_outer_bias_flags.extend(
            [
                "outer_exclude_hands",
                "outer_hand_expand_px",
            ]
        )
    if (
        float(outer_trigger_face_ema_decay) > 0.0
        or float(outer_trigger_face_ema_weight) > 0.0
        or float(outer_trigger_face_topk_min_support) > 0.0
        or int(outer_trigger_face_topk) > 0
    ):
        ignored_outer_bias_flags.extend(
            [
                "outer_trigger_face_ema_decay",
                "outer_trigger_face_ema_weight",
                "outer_trigger_face_topk",
                "outer_trigger_face_topk_min_support",
            ]
        )
    if bool(outer_spawn_inside_only) or float(outer_spawn_support_th) > 0.0:
        ignored_outer_bias_flags.extend(
            [
                "outer_spawn_inside_only",
                "outer_spawn_support_th",
            ]
        )
    if len(ignored_outer_bias_flags) > 0:
        print(
            "Warning: outer activation/spawn/prune ignores semantic bias settings: "
            + ", ".join(sorted(set(ignored_outer_bias_flags)))
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    # Early-stop state
    es_enable = bool(early_stop_enable) and int(early_stop_patience) > 0
    es_patience = max(int(early_stop_patience), 0)
    es_min_delta = max(float(early_stop_min_delta), 0.0)
    es_warmup = max(int(early_stop_warmup_iters), 0)
    es_best = float("inf")
    es_bad = 0

    binding_state = initialize_stage2_training_binding_state(
        binding_enabled=bool(binding_enabled),
        mesh_runtime=mesh_runtime,
        bind_vertex_indices=bind_vertex_indices,
        g_bind_face_idx=g.bind_face_idx,
        g_bind_bary=g.bind_bary,
        g_bind_offset=g.bind_offset,
        frames_list=[str(x) for x in frames_list],
        bind_mesh_indices=bind_mesh_indices,
        bind_cache_size=int(bind_cache_size),
        device=device,
        use_smpl_silhouette_mask=bool(use_smpl_silhouette_mask),
        bind_align_quat_to_mesh_normals=bool(bind_align_quat_to_mesh_normals),
        binding_cfg=binding_cfg,
        binding_cfg_dict=binding,
        outer_airspace_neighbor_rings=int(outer_airspace_neighbor_rings),
        outer_airspace_sharp_allow_dot=float(outer_airspace_sharp_allow_dot),
        bind_rigid_align_enabled=bool(bind_rigid_align_enabled),
        bind_rigid_align_cfg=bind_rigid_align_cfg,
        person_masks_npz=person_masks_npz,
        K_np=K_np,
        image_hw=(H, W),
        axis_angle_to_rotmat_fn=axis_angle_to_rotmat,
        cam_w=cam_w,
        cam_t=cam_t,
        logger=_TrainLoopLogger(),
    )
    mesh_normal_cache = binding_state.mesh_normal_cache
    smpl_mask_cache = binding_state.smpl_mask_cache
    bind_idx_t = binding_state.bind_idx_t
    bind_vertex_offset_t = binding_state.bind_vertex_offset_t
    bind_face_idx_t = binding_state.bind_face_idx_t
    bind_bary_t = binding_state.bind_bary_t
    bind_faces_np = binding_state.bind_faces_np
    bind_face_neighbor_idx_t = binding_state.bind_face_neighbor_idx_t
    bind_face_neighbor_w_t = binding_state.bind_face_neighbor_w_t
    outer_face_weight_sum = binding_state.outer_face_weight_sum
    outer_face_gap_half_weighted_sum = binding_state.outer_face_gap_half_weighted_sum
    outer_face_gap_scale_weighted_sum = binding_state.outer_face_gap_scale_weighted_sum
    outer_face_spawned_mask = binding_state.outer_face_spawned_mask
    outer_face_first_hit_frame_idx = binding_state.outer_face_first_hit_frame_idx
    outer_face_contrib_frames = binding_state.outer_face_contrib_frames
    outer_face_spawn_iter = binding_state.outer_face_spawn_iter
    outer_face_spawn_frame_idx = binding_state.outer_face_spawn_frame_idx
    bind_rigid_align_meta = binding_state.bind_rigid_align_meta
    bind_rigid_apply_cfg = binding_state.bind_rigid_apply_cfg
    _load_mesh_np_raw = binding_state.load_mesh_np_raw
    _load_mesh_verts = binding_state.load_mesh_verts
    _load_binding_frame = binding_state.load_binding_frame

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
            dilate_px=int(smpl_mask_dilate_eff),
            erode_px=int(smpl_mask_erode_eff),
        )
        smpl_mask_cache[mesh_idx_pc] = (
            torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        )

    # ---- Adaptive training helpers ----
    phase_ratios_raw = np.asarray(
        [
            max(float(phase_ratio_inner_only), 0.0),
            max(float(phase_ratio_outer_only), 0.0),
            max(float(phase_ratio_occlusion_simplify), 0.0),
            max(float(phase_ratio_joint_finetune), 0.0),
        ],
        dtype=np.float32,
    )
    if float(phase_ratios_raw.sum()) <= 1e-8:
        raise ValueError(
            "phase ratios are all zero. Set phase_ratio_inner_only / "
            "phase_ratio_outer_only / phase_ratio_occlusion_simplify / "
            "phase_ratio_joint_finetune to positive values."
        )
    if training_mode == "inner_only":
        phase_counts = np.asarray([int(iters), 0, 0, 0], dtype=np.int64)
    elif training_mode == "outer_only":
        phase_counts = np.asarray([0, int(iters), 0, 0], dtype=np.int64)
    else:
        phase_ratios = phase_ratios_raw / float(np.clip(phase_ratios_raw.sum(), 1e-8, None))
        phase_counts = np.floor(phase_ratios * float(iters)).astype(np.int64)
        residual_iters = int(iters - int(phase_counts.sum()))
        if residual_iters > 0:
            order = np.argsort(-(phase_ratios * float(iters) - phase_counts.astype(np.float32)))
            for k in order[:residual_iters]:
                phase_counts[int(k)] += 1
    phase_inner_end = int(phase_counts[0])
    phase_outer_end = int(phase_inner_end + phase_counts[1])
    phase_occlusion_end = int(phase_outer_end + phase_counts[2])
    outer_phase_span = max(int(phase_outer_end - phase_inner_end), 1)
    outer_geom_fraction = float(np.clip(float(outer_geom_fraction), 0.0, 1.0))
    outer_geom_end = int(round(float(phase_inner_end) + outer_geom_fraction * float(outer_phase_span)))
    outer_geom_end = min(max(outer_geom_end, phase_inner_end + 1), phase_outer_end)

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
    edge_strength_cache: dict[int, torch.Tensor] = {}
    outer_gt_mask_hint_cache: dict[int, torch.Tensor] = {}
    outer_face_gap_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    inner_soft_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    inner_hard_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_low_support_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_boundary_violation_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_boundary_hard_clamp_count = 0

    outer_spawn_stats_by_frame: dict[str, dict[str, int]] = {
        str(fn): {"total": 0, "trigger": 0, "fallback": 0, "iters_hit": 0}
        for fn in frames_list
    }
    outer_spawn_iter_records: list[dict[str, object]] = []
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    quality_log_path = out_ckpt_path.parent / "outer_quality_metrics.jsonl"
    quality_report_path = out_ckpt_path.parent / "outer_quality_report.json"
    outer_spawn_report_path = out_ckpt_path.with_name("outer_spawn_per_frame.json")
    if quality_log_path.exists():
        quality_log_path.unlink()
    quality_log_every = max(int(outer_quality_log_every), 1)
    progress_log_every = max(int(outer_progress_log_every), 1)
    outer_zero_stop_enable = bool(outer_zero_stop_enable) and bool(shell_enable)
    outer_zero_stop_patience = max(int(outer_zero_stop_patience), 1)
    outer_zero_stop_warmup_iters = max(int(outer_zero_stop_warmup_iters), 0)
    outer_zero_consecutive = 0
    quality_thresholds = default_stage2_quality_thresholds()
    head_detached_coverage_baseline: float | None = None
    last_quality_metrics: dict[str, float] | None = None
    inner_ckpt_saved = False

    def _build_checkpoint_payload(*, quality_pass_now: bool) -> dict[str, object]:
        return {
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
            "frame_align_scores": frame_align_scores,
            "face_weight": face_weight,
            "face_conf_min": face_conf_min,
            "face_box_pad": face_box_pad,
            "face_use_ears": face_use_ears,
            "hand_weight": hand_weight,
            "hand_conf_min": hand_conf_min,
            "hand_radius_scale": hand_radius_scale,
            "hand_radius_min_px": hand_radius_min_px,
            "hand_extension_scale": hand_extension_scale,
            "outer_exclude_hands": outer_exclude_hands,
            "outer_hand_expand_px": outer_hand_expand_px,
            "phase_ratio_inner_only": phase_ratio_inner_only,
            "phase_ratio_outer_only": phase_ratio_outer_only,
            "phase_ratio_occlusion_simplify": phase_ratio_occlusion_simplify,
            "phase_ratio_joint_finetune": phase_ratio_joint_finetune,
            "outer_geom_fraction": outer_geom_fraction,
            "shell_enable": shell_enable,
            "shell_max_ratio": shell_max_ratio,
            "shell_spawn_top_percent": shell_spawn_top_percent,
            "shell_spawn_every": shell_spawn_every,
            "shell_prune_every": shell_prune_every,
            "shell_offset_max_n": shell_offset_max_n,
            "shell_offset_max_t": shell_offset_max_t,
            "inner_bind_mode_default": inner_bind_mode_default,
            "outer_init_from_inner": outer_init_from_inner,
            "outer_dormant_slot_ratio": outer_dormant_slot_ratio,
            "outer_trigger_face_activate_enable": outer_trigger_face_activate_enable,
            "outer_trigger_face_sample_dilate_px": outer_trigger_face_sample_dilate_px,
            "outer_viewplane_init_enable": outer_viewplane_init_enable,
            "outer_center_init_rule": outer_center_init_rule,
            "outer_center_phase1_normal_tune_enable": outer_center_phase1_normal_tune_enable,
            "outer_center_phase1_normal_tune_cap": outer_center_phase1_normal_tune_cap,
            "outer_center_freeze_after_phase1": outer_center_freeze_after_phase1,
            "outer_contact_alpha_edge_th": outer_contact_alpha_edge_th,
            "outer_detached_support_boost": outer_detached_support_boost,
            "outer_detached_vis_boost": outer_detached_vis_boost,
            "outer_overlap_prune_offset_th": outer_overlap_prune_offset_th,
            "outer_overlap_prune_detached_th": outer_overlap_prune_detached_th,
            "outer_detached_support_th": outer_detached_support_th,
            "outer_detached_opacity_floor": outer_detached_opacity_floor,
            "outer_detached_log_scale_floor": outer_detached_log_scale_floor,
            "outer_gap_search_px": outer_gap_search_px,
            "outer_gap_step_px": outer_gap_step_px,
            "outer_gap_probe_world": outer_gap_probe_world,
            "outer_gap_target_support_th": outer_gap_target_support_th,
            "outer_gap_target_weight": outer_gap_target_weight,
            "outer_gap_scale_weight": outer_gap_scale_weight,
            "outer_gap_tangent_scale_ratio": outer_gap_tangent_scale_ratio,
            "outer_gap_tangent_scale_weight": outer_gap_tangent_scale_weight,
            "outer_gap_hard_target_enable": outer_gap_hard_target_enable,
            "outer_gap_hard_target_blend": outer_gap_hard_target_blend,
            "outer_phase_residual_weight": outer_phase_residual_weight,
            "frame_weights": frame_weights,
            "frame_loss_weights": frame_loss_weights,
            "temporal_ref_window": temporal_ref_window,
            "temporal_ref_decay": temporal_ref_decay,
            "temporal_ref_anchor": temporal_ref_anchor,
            "bind_mesh_dir": str(mesh_context.bind_mesh_dir) if (binding_enabled and mesh_context is not None and mesh_context.bind_mesh_dir is not None) else None,
            "bind_mesh_indices": bind_mesh_indices if binding_enabled else None,
            "bind_sequence_npz": str(mesh_context.bind_seq_npz_path) if (binding_enabled and mesh_context is not None and mesh_context.bind_seq_npz_path is not None) else None,
            "bind_vertices_npz": str(mesh_context.bind_vertices_npz_path) if (binding_enabled and mesh_context is not None and mesh_context.bind_vertices_npz_path is not None) else None,
            "bind_deform_obj": str(mesh_context.bind_deform_obj_path) if (binding_enabled and mesh_context is not None and mesh_context.bind_deform_obj_path is not None) else None,
            "bind_mesh_obj_space": bind_mesh_obj_space_mode,
            "bind_align_quat_to_mesh_normals": bind_align_quat_to_mesh_normals,
            "bind_quat_normal_blend": bind_quat_normal_blend,
            "bind_surface_aniso_weight": bind_surface_aniso_weight,
            "bind_surface_aniso_margin": bind_surface_aniso_margin,
            "bind_rigid_align": bind_rigid_align_meta if binding_enabled else None,
            "binding": export_binding_mode_config(binding_cfg),
            "use_smpl_silhouette_mask": use_smpl_silhouette_mask,
            "smpl_mask_dilate_px": smpl_mask_dilate_eff,
            "smpl_mask_erode_px": smpl_mask_erode_eff,
            "smpl_color_mask_erode_px": smpl_color_mask_erode_eff,
            "inner_use_strict_smpl_mask": inner_use_strict_smpl_mask,
            "smpl_alpha_leak_weight": smpl_alpha_leak_weight,
            "smpl_mask_override_alpha": smpl_mask_override_alpha,
            "smpl_mask_precompute": smpl_mask_precompute,
            "smpl_mask_precompute_max_frames": smpl_mask_precompute_max_frames,
            "person_masks_npz": str(person_masks_npz) if person_masks_npz is not None else None,
            "use_person_mask": use_person_mask,
            "person_mask_dilate_px": person_mask_dilate_px,
            "person_mask_erode_px": person_mask_erode_px,
            "person_mask_fuse_mode": person_mask_fuse_mode,
            "person_mask_eval_clean": person_mask_eval_clean,
            "person_mask_eval_refine_modnet": person_mask_eval_refine_modnet,
            "person_mask_contour_width_px": person_mask_contour_width_px,
            "person_mask_contour_leak_weight": person_mask_contour_leak_weight,
            "outer_use_gt_contour": outer_use_gt_contour,
            "outer_gt_contour_fuse_mode": outer_gt_contour_fuse_mode,
            "outer_gt_contour_modnet_ref_size": outer_gt_contour_modnet_ref_size,
            "outer_gt_contour_modnet_threshold": outer_gt_contour_modnet_threshold,
            "modnet_ckpt_path": modnet_ckpt_path,
            "outer_head_priority_enable": outer_head_priority_enable,
            "outer_head_priority_weight": outer_head_priority_weight,
            "outer_head_priority_face_pad": outer_head_priority_face_pad,
            "outer_head_priority_up_scale": outer_head_priority_up_scale,
            "outer_head_priority_down_scale": outer_head_priority_down_scale,
            "outer_head_priority_person_top_ratio": outer_head_priority_person_top_ratio,
            "outer_head_support_th": outer_head_support_th,
            "outer_trigger_smpl_gap_px": outer_trigger_smpl_gap_px,
            "outer_trigger_edge_th": outer_trigger_edge_th,
            "outer_trigger_min_pixels": outer_trigger_min_pixels,
            "outer_trigger_ring_width_px": outer_trigger_ring_width_px,
            "outer_trigger_use_edge_gate": outer_trigger_use_edge_gate,
            "outer_contour_inside_only": outer_contour_inside_only,
            "outer_detached_solid_enable": outer_detached_solid_enable,
            "outer_detached_solid_dilate_px": outer_detached_solid_dilate_px,
            "outer_detached_solid_close_px": outer_detached_solid_close_px,
            "outer_detached_solid_mix_start": outer_detached_solid_mix_start,
            "outer_detached_solid_mix_end": outer_detached_solid_mix_end,
            "outer_head_support_sample_dilate_px": outer_head_support_sample_dilate_px,
            "outer_exclude_face": outer_exclude_face,
            "outer_face_expand_px": outer_face_expand_px,
            "outer_face_exclude_lower_ratio": outer_face_exclude_lower_ratio,
            "outer_silhouette_weight": outer_silhouette_weight,
            "outer_silhouette_band_px": outer_silhouette_band_px,
            "outer_silhouette_band_weight": outer_silhouette_band_weight,
            "detail_hf_weight": detail_hf_weight,
            "detail_hf_exclude_face": detail_hf_exclude_face,
            "outer_inside_support_th": outer_inside_support_th,
            "outer_outside_support_th": outer_outside_support_th,
            "outer_outside_opacity_cap": outer_outside_opacity_cap,
            "outer_outside_offset_decay": outer_outside_offset_decay,
            "outer_outside_opacity_weight": outer_outside_opacity_weight,
            "outer_outside_offset_weight": outer_outside_offset_weight,
            "outer_offset_suppress_low_support": outer_offset_suppress_low_support,
            "outer_offset_suppress_support_th": outer_offset_suppress_support_th,
            "outer_offset_suppress_decay": outer_offset_suppress_decay,
            "outer_offset_suppress_opacity_cap": outer_offset_suppress_opacity_cap,
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
            "outer_log_scale_min": outer_log_scale_min,
            "outer_log_scale_max": outer_log_scale_max,
            "outer_prune_support_th": outer_prune_support_th,
            "outer_prune_dual_enable": outer_prune_dual_enable,
            "outer_prune_dual_max_vis": outer_prune_dual_max_vis,
            "outer_prune_dual_min_support": outer_prune_dual_min_support,
            "early_stop_enable": early_stop_enable,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_warmup_iters": early_stop_warmup_iters,
            "outer_edge_trigger_enable": outer_edge_trigger_enable,
            "outer_edge_trigger_weight": outer_edge_trigger_weight,
            "outer_boundary_enforce": outer_boundary_enforce,
            "outer_boundary_tol_px": outer_boundary_tol_px,
            "outer_boundary_check_every": outer_boundary_check_every,
            "outer_boundary_alpha_cap": outer_boundary_alpha_cap,
            "outer_boundary_scale_decay": outer_boundary_scale_decay,
            "outer_boundary_offset_decay": outer_boundary_offset_decay,
            "outer_boundary_patience": outer_boundary_patience,
            "outer_edge_prune_protect_quantile": outer_edge_prune_protect_quantile,
            "outer_offset_unbounded": outer_offset_unbounded,
            "outer_offset_auto_range_enable": outer_offset_auto_range_enable,
            "outer_offset_auto_range_min_scale": outer_offset_auto_range_min_scale,
            "outer_offset_auto_range_edge_mix": outer_offset_auto_range_edge_mix,
            "outer_offset_edge_guided_reg": outer_offset_edge_guided_reg,
            "outer_offset_reg_relax_support": outer_offset_reg_relax_support,
            "outer_offset_min_active_weight": outer_offset_min_active_weight,
            "outer_offset_min_target_scale": outer_offset_min_target_scale,
            "outer_offset_min_support_th": outer_offset_min_support_th,
            "inner_opacity_prior_target": inner_opacity_prior_target,
            "outer_opacity_prior_target": outer_opacity_prior_target,
            "shell_offset_reg_phase0_scale": shell_offset_reg_phase0_scale,
            "shell_offset_freeze_phase0": shell_offset_freeze_phase0,
            "outer_offset_grad_scale": outer_offset_grad_scale,
            "outer_offset_edge_grad_scale": outer_offset_edge_grad_scale,
            "outer_offset_grad_scale_max": outer_offset_grad_scale_max,
            "outer_support_weight": outer_support_weight,
            "outer_spawn_support_th": outer_spawn_support_th,
            "outer_spawn_inside_only": outer_spawn_inside_only,
            "outer_phase_inner_vis_scale": outer_phase_inner_vis_scale,
            "outer_phase_inner_full_mask": outer_phase_inner_full_mask,
            "outer_low_support_patience": outer_low_support_patience,
            "outer_pruned_opacity_logit_max": outer_pruned_opacity_logit_max,
            "outer_quality_log_every": outer_quality_log_every,
            "outer_progress_log_every": outer_progress_log_every,
            "outer_zero_stop_enable": outer_zero_stop_enable,
            "outer_zero_stop_patience": outer_zero_stop_patience,
            "outer_zero_stop_warmup_iters": outer_zero_stop_warmup_iters,
            "outer_zero_stop_after_phase_inner": outer_zero_stop_after_phase_inner,
            "bind_face_offset_straight_through": bind_face_offset_straight_through,
            "dual_quality_gate": dual_quality_gate,
            "dual_quality_gate_last": {
                "color_error": dq_last_color_error,
                "color_var": dq_last_color_var,
                "geometry_score": dq_last_geom_score,
                "fail_active": dq_fail_active,
            },
            "outer_quality_pass": quality_pass_now,
            "outer_quality_report_path": str(quality_report_path),
            "outer_spawn_report_path": str(outer_spawn_report_path),
            "training_mode": training_mode,
            "save_inner_checkpoint_path": (
                None if save_inner_checkpoint_path is None else str(save_inner_checkpoint_path)
            ),
        }


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

    pbar = tqdm(range(iters), desc="Stage2 3DGS (learnable camera)")
    for it in pbar:
        opt.zero_grad(set_to_none=True)

        # Training phase schedule: 40/30/30 split as
        # [inner-only, outer-only, occlusion-simplify, joint-finetune].
        if it < phase_inner_end:
            phase_tag = "phase_inner"
        elif it < phase_outer_end:
            phase_tag = "phase_outer"
        elif it < phase_occlusion_end:
            phase_tag = "phase_occlusion"
        else:
            phase_tag = "phase_joint"
        is_outer_geom_phase = phase_tag == "phase_outer" and it < outer_geom_end
        is_outer_color_phase = phase_tag == "phase_outer" and (not is_outer_geom_phase)
        outer_geom_phase_progress = 1.0
        if is_outer_geom_phase:
            geom_denom = max(int(outer_geom_end - phase_inner_end), 1)
            outer_geom_phase_progress = float(it - phase_inner_end + 1) / float(geom_denom)
            outer_geom_phase_progress = float(np.clip(outer_geom_phase_progress, 0.0, 1.0))
        if is_outer_geom_phase:
            phase_label = "phase_outer_geom"
        elif is_outer_color_phase:
            phase_label = "phase_outer_color"
        else:
            phase_label = phase_tag

        outer_detached_solid_mix_weight_curr = 0.0
        if bool(outer_detached_solid_enable):
            mix_start = float(np.clip(float(outer_detached_solid_mix_start), 0.0, 1.0))
            mix_end = float(np.clip(float(outer_detached_solid_mix_end), 0.0, 1.0))
            mix_span_start = int(phase_inner_end)
            mix_span_end = int(max(phase_occlusion_end - 1, mix_span_start))
            if mix_span_end <= mix_span_start:
                mix_prog = 1.0
            elif it <= mix_span_start:
                mix_prog = 0.0
            elif it >= mix_span_end:
                mix_prog = 1.0
            else:
                mix_prog = float(it - mix_span_start) / float(max(mix_span_end - mix_span_start, 1))
            outer_detached_solid_mix_weight_curr = (1.0 - mix_prog) * mix_start + mix_prog * mix_end

        for pg in opt.param_groups:
            base_lr = float(pg.get("initial_lr", pg["lr"]))
            pg.setdefault("initial_lr", base_lr)
            if phase_tag == "phase_joint":
                lr_mul = 0.35
            elif phase_tag == "phase_occlusion":
                lr_mul = 0.70
            else:
                lr_mul = 1.0
            pg["lr"] = base_lr * lr_mul

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
        spawn_frame_name = str(frames_list[fi])
        vis_accum: list[torch.Tensor] = []
        edge_accum: list[torch.Tensor] = []
        outer_detached_support_accum: list[torch.Tensor] = []
        outer_gap_half_n_accum: list[torch.Tensor] = []
        outer_gap_scale_accum: list[torch.Tensor] = []
        outer_gap_support_accum: list[torch.Tensor] = []
        outer_inside_support_accum: list[torch.Tensor] = []
        outer_head_support_accum: list[torch.Tensor] = []
        outer_person_support_accum: list[torch.Tensor] = []
        outer_hand_support_accum: list[torch.Tensor] = []
        trigger_frame_data: BindingFrameData | None = None
        color_eval_vals: list[float] = []
        color_var_vals: list[float] = []
        geom_eval_vals: list[float] = []
        for fj, wj in zip(ctx_indices, ctx_weights):
            gt = gts[fj]  # (3,H,W)
            gt_inner_target = (
                prepared_inner_rgb_targets_frames[fj]
                if prepared_inner_rgb_targets_frames is not None
                else gt
            )
            gt_outer_target = (
                prepared_outer_rgb_targets_frames[fj]
                if prepared_outer_rgb_targets_frames is not None
                else gt
            )
            gt_combined_target = (
                prepared_combined_rgb_targets_frames[fj]
                if prepared_combined_rgb_targets_frames is not None
                else gt
            )

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
                        if trigger_frame_data is None or int(fj) == int(fi):
                            trigger_frame_data = frame_data

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

            # Optional silhouette mask from bound SMPL mesh.
            smpl_mask = None
            if prepared_smpl_masks_frames is not None:
                smpl_mask = prepared_smpl_masks_frames[fj]
            elif binding_enabled and bool(use_smpl_silhouette_mask) and bind_faces_np is not None:
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
                        dilate_px=int(smpl_mask_dilate_eff),
                        erode_px=int(smpl_mask_erode_eff),
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
            hand_mask_curr = None
            if hand_masks_frames is not None:
                hand_mask_curr = hand_masks_frames[fj]
            outer_head_eval_mask_curr = None
            if outer_head_eval_masks_frames is not None:
                outer_head_eval_mask_curr = outer_head_eval_masks_frames[fj]

            smpl_mask_for_color = None
            if smpl_mask is not None:
                smpl_mask_for_color = smpl_mask
                if int(smpl_color_mask_erode_eff) > 0:
                    smpl_mask_for_color = _erode_binary_mask_torch(
                        smpl_mask_for_color,
                        int(smpl_color_mask_erode_eff),
                    )
                if (not sep_fg_bg) and bool(smpl_mask_override_alpha):
                    mask = smpl_mask_for_color if mask is None else (mask * smpl_mask_for_color)

            # Layer-aware supervision:
            # - inner gaussians: strict SMPL support (or person fallback if SMPL missing);
            # - outer gaussians: trigger-only support from GT-vs-SMPL residual + edge gate.
            if prepared_inner_masks_frames is not None:
                inner_supervision_mask = prepared_inner_masks_frames[fj]
            else:
                inner_supervision_mask = resolve_inner_supervision_mask(
                    strict_inner_mask=bool(strict_inner_mask),
                    smpl_mask=smpl_mask,
                    smpl_mask_for_color=smpl_mask_for_color,
                    person_mask_curr=person_mask_curr,
                )
            outer_supervision_mask = None
            outer_base_mask = None
            if bool(outer_use_gt_contour):
                if prepared_gt_train_masks_frames is not None and prepared_gt_train_masks_frames[fj] is not None:
                    outer_base_mask = prepared_gt_train_masks_frames[fj]
                elif outer_gt_masks_frames is not None and outer_gt_masks_frames[fj] is not None:
                    outer_base_mask = outer_gt_masks_frames[fj]
                elif fj in outer_gt_mask_hint_cache:
                    outer_base_mask = outer_gt_mask_hint_cache[fj]
                else:
                    smpl_hint_u8 = None
                    if smpl_mask is not None:
                        smpl_hint_u8 = ((smpl_mask[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255)
                    gt_mask_np = _estimate_gt_person_contour_mask_np(
                        gt,
                        hint_mask_u8=smpl_hint_u8,
                        modnet_ref_size=int(max(32, outer_gt_contour_modnet_ref_size)),
                        modnet_threshold=float(np.clip(float(outer_gt_contour_modnet_threshold), 0.0, 1.0)),
                        modnet_ckpt_path=modnet_ckpt_path,
                    )
                    pm_np = None
                    if person_mask_curr is not None:
                        pm_np = (person_mask_curr[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    final_mask_np = fuse_outer_contour_mask_np(
                        gt_mask_np,
                        pm_np,
                        fuse_mode=outer_contour_fuse,
                    )
                    if final_mask_np is not None:
                        outer_base_mask = torch.from_numpy(final_mask_np.astype(np.float32)).to(device=device).unsqueeze(0)
                    if outer_base_mask is not None:
                        outer_gt_mask_hint_cache[fj] = outer_base_mask
            if outer_base_mask is None and outer_gt_masks_frames is not None:
                outer_base_mask = outer_gt_masks_frames[fj]
            if outer_base_mask is None:
                outer_base_mask = person_mask_curr
            edge_map_curr = None
            edge_trigger_map_curr = None
            outer_detached_mask_curr = None
            face_gap_half_curr = None
            face_gap_scale_curr = None
            face_gap_support_curr = None
            smpl_ref = smpl_mask if smpl_mask is not None else smpl_mask_for_color
            need_edge_map = (
                bool(outer_trigger_use_edge_gate)
                or (bool(shell_enable) and bool(outer_edge_trigger_enable))
            )
            if outer_base_mask is not None and need_edge_map:
                if fj not in edge_strength_cache:
                    edge_strength_cache[fj] = _compute_image_edge_strength_torch(gt).detach().to(dtype=torch.float16)
                edge_map_curr = edge_strength_cache[fj].to(dtype=torch.float32)
            if prepared_outer_masks_frames is not None and prepared_outer_masks_frames[fj] is not None:
                outer_supervision_mask = prepared_outer_masks_frames[fj]
            elif outer_base_mask is not None:
                outer_supervision_mask, edge_trigger_map_curr, _trigger_pixels, outer_detached_mask_curr = build_outer_trigger_mask(
                    outer_base_mask=outer_base_mask,
                    smpl_ref=smpl_ref,
                    edge_map_curr=edge_map_curr,
                    outer_trigger_smpl_gap_px=int(outer_trigger_smpl_gap_px),
                    outer_trigger_edge_th=float(outer_trigger_edge_th),
                    outer_trigger_min_pixels=int(outer_trigger_min_pixels),
                    outer_trigger_use_edge_gate=bool(outer_trigger_use_edge_gate),
                    outer_trigger_ring_width_px=int(outer_trigger_ring_width_px),
                    outer_contour_inside_only=bool(outer_contour_inside_only),
                    outer_edge_trigger_enable=bool(outer_edge_trigger_enable),
                    outer_detached_solid_enable=bool(outer_detached_solid_enable),
                    outer_detached_solid_dilate_px=int(max(outer_detached_solid_dilate_px, 0)),
                    outer_detached_solid_close_px=int(max(outer_detached_solid_close_px, 0)),
                    outer_detached_solid_mix_weight=float(outer_detached_solid_mix_weight_curr),
                    dilate_fn=_dilate_binary_mask_torch,
                    erode_fn=_erode_binary_mask_torch,
                    contour_band_fn=_compute_contour_band_torch,
                )

            if (
                bool(shell_enable)
                and outer_base_mask is not None
                and smpl_ref is not None
                and frame_data is not None
                and frame_data.verts_t is not None
                and frame_data.faces_t is not None
                and frame_data.face_n_t is not None
            ):
                use_gap_cache = not bool(learn_cam)
                if (not use_gap_cache) or (fj not in outer_face_gap_cache):
                    gap_half_tmp, gap_scale_tmp, gap_sup_tmp = estimate_face_gap_targets(
                        outer_mask=outer_base_mask,
                        smpl_mask=smpl_ref,
                        verts_t=frame_data.verts_t,
                        faces_t=frame_data.faces_t,
                        face_n_t=frame_data.face_n_t,
                        K=K,
                        R=R,
                        t=t,
                        image_h=H,
                        image_w=W,
                        max_search_px=int(max(outer_gap_search_px, 1)),
                        step_px=float(max(outer_gap_step_px, 0.5)),
                        normal_probe_world=float(max(outer_gap_probe_world, 1e-4)),
                        smpl_rim_px=int(max(outer_trigger_ring_width_px + outer_trigger_smpl_gap_px, 1)),
                        refine_steps=int(max(outer_gap_refine_steps, 0)),
                    )
                    if gap_half_tmp is not None and gap_scale_tmp is not None and gap_sup_tmp is not None:
                        gap_pack = (
                            gap_half_tmp.detach().to(dtype=torch.float16),
                            gap_scale_tmp.detach().to(dtype=torch.float16),
                            gap_sup_tmp.detach().to(dtype=torch.float16),
                        )
                        if use_gap_cache:
                            outer_face_gap_cache[fj] = gap_pack
                        else:
                            face_gap_half_curr = gap_pack[0].to(device=device, dtype=torch.float32)
                            face_gap_scale_curr = gap_pack[1].to(device=device, dtype=torch.float32)
                            face_gap_support_curr = gap_pack[2].to(device=device, dtype=torch.float32)
                if use_gap_cache and fj in outer_face_gap_cache:
                    gap_half_cache, gap_scale_cache, gap_sup_cache = outer_face_gap_cache[fj]
                    face_gap_half_curr = gap_half_cache.to(device=device, dtype=torch.float32)
                    face_gap_scale_curr = gap_scale_cache.to(device=device, dtype=torch.float32)
                    face_gap_support_curr = gap_sup_cache.to(device=device, dtype=torch.float32)

            if outer_supervision_mask is not None and float(outer_supervision_mask.sum().detach().cpu()) < 1.0:
                outer_supervision_mask = None
                edge_trigger_map_curr = None

            if (
                bool(shell_enable)
                and bool(outer_trigger_face_activate_enable)
                and outer_supervision_mask is not None
                and frame_data is not None
                and frame_data.faces_t is not None
            ):
                face_trigger_support, _ = map_trigger_ring_to_faces(
                    trigger_ring_mask=outer_supervision_mask,
                    verts_t=frame_data.verts_t,
                    faces_t=frame_data.faces_t,
                    K=K,
                    R=R,
                    t=t,
                    image_h=H,
                    image_w=W,
                    sample_dilate_px=int(max(outer_trigger_face_sample_dilate_px, 0)),
                )
                if face_trigger_support is not None:
                    if bool(outer_gap_fallback_enable):
                        if face_gap_half_curr is None or face_gap_scale_curr is None or face_gap_support_curr is None:
                            n_faces_curr = int(face_trigger_support.shape[0])
                            face_gap_half_curr = torch.zeros((n_faces_curr,), dtype=torch.float32, device=device)
                            face_gap_scale_curr = torch.zeros((n_faces_curr,), dtype=torch.float32, device=device)
                            face_gap_support_curr = torch.zeros((n_faces_curr,), dtype=torch.float32, device=device)
                        trigger_sup_curr = face_trigger_support.detach().clamp(0.0, 1.0)
                        fallback_mask_curr = (
                            trigger_sup_curr >= float(max(outer_gap_fallback_support_th, 0.0))
                        ) & (face_gap_support_curr <= 0.0)
                        if bool(fallback_mask_curr.any()):
                            half_min_n = float(max(outer_gap_fallback_half_min_n, 1e-4))
                            half_max_n = float(max(outer_gap_fallback_half_max_n, half_min_n))
                            half_max_n = min(half_max_n, float(shell_offset_max_n))
                            half_min_n = min(half_min_n, half_max_n)
                            sup_sel_curr = trigger_sup_curr[fallback_mask_curr]
                            half_target_curr = half_min_n + (half_max_n - half_min_n) * sup_sel_curr
                            face_gap_half_curr[fallback_mask_curr] = torch.maximum(
                                face_gap_half_curr[fallback_mask_curr],
                                half_target_curr,
                            )
                            face_gap_scale_curr[fallback_mask_curr] = torch.maximum(
                                face_gap_scale_curr[fallback_mask_curr],
                                (2.0 * half_target_curr).clamp(max=float(shell_offset_max_n)),
                            )
                            face_gap_support_curr[fallback_mask_curr] = torch.maximum(
                                face_gap_support_curr[fallback_mask_curr],
                                sup_sel_curr,
                            )
                            if not bool(learn_cam):
                                outer_face_gap_cache[fj] = (
                                    face_gap_half_curr.detach().to(dtype=torch.float16),
                                    face_gap_scale_curr.detach().to(dtype=torch.float16),
                                    face_gap_support_curr.detach().to(dtype=torch.float16),
                                )
                    if (
                        outer_face_weight_sum is not None
                        and outer_face_gap_half_weighted_sum is not None
                        and outer_face_gap_scale_weighted_sum is not None
                        and outer_face_first_hit_frame_idx is not None
                        and outer_face_contrib_frames is not None
                        and face_gap_half_curr is not None
                        and face_gap_scale_curr is not None
                    ):
                        align_weight = 1.0
                        if frame_align_scores_t is not None:
                            align_weight = float(frame_align_scores_t[fj].item())
                        if align_weight > 0.0:
                            trigger_hit_mask = (
                                face_trigger_support.detach().view(-1) > 0.0
                            ) & (face_gap_half_curr > 0.0) & (face_gap_scale_curr > 0.0)
                            hit_faces = torch.nonzero(trigger_hit_mask, as_tuple=False).view(-1)
                            if int(hit_faces.numel()) > 0:
                                outer_face_weight_sum.index_add_(
                                    0,
                                    hit_faces,
                                    torch.full(
                                        (int(hit_faces.numel()),),
                                        float(align_weight),
                                        dtype=torch.float32,
                                        device=device,
                                    ),
                                )
                                outer_face_gap_half_weighted_sum.index_add_(
                                    0,
                                    hit_faces,
                                    face_gap_half_curr.index_select(0, hit_faces) * float(align_weight),
                                )
                                outer_face_gap_scale_weighted_sum.index_add_(
                                    0,
                                    hit_faces,
                                    face_gap_scale_curr.index_select(0, hit_faces) * float(align_weight),
                                )
                                for face_idx_i in hit_faces.detach().cpu().tolist():
                                    if outer_face_first_hit_frame_idx[face_idx_i] is None:
                                        outer_face_first_hit_frame_idx[face_idx_i] = int(fj)
                                    outer_face_contrib_frames[face_idx_i].add(int(fj))

            weight = None
            if face_boxes is not None and face_boxes[fj] is not None and float(face_weight) > 1.0:
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                w2d = torch.ones((1, H, W), device=device)
                w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_weight)
                weight = w2d
            if hand_mask_curr is not None and float(hand_weight) > 1.0:
                h_w = torch.ones((1, H, W), device=device)
                h_w = h_w + (float(hand_weight) - 1.0) * hand_mask_curr.clamp(0.0, 1.0)
                weight = h_w if weight is None else (weight * h_w)

            if sep_fg_bg:
                region_mask = inner_supervision_mask
                bg_w_eff = bg_w
                if phase_tag == "phase_inner":
                    # During pure inner training, supervise only inside the SMPL support region.
                    bg_w_eff = 0.0
                if region_mask is None:
                    region_mask = person_mask_curr if person_mask_curr is not None else outer_supervision_mask
                if region_mask is not None:
                    region_w = (bg_w_eff + (fg_w - bg_w_eff) * region_mask).clamp(min=0.0)
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

            xyz_render = g.xyz
            if (
                bool(bind_face_offset_straight_through)
                and binding_enabled
                and frame_data is not None
                and bind_mask_curr is not None
                and bind_face_idx_t is not None
            ):
                xyz_render = _inject_bind_face_offset_straight_through_xyz(
                    xyz_base=g.xyz,
                    g=g,
                    frame_data=frame_data,
                    bind_mask_curr=bind_mask_curr,
                    bind_face_idx_t=bind_face_idx_t,
                )
            g_render = _make_gaussian_with_xyz(g, xyz_render) if (xyz_render is not g.xyz) else g

            # Pass-1 render: estimate visibility and frame residual.
            pred_rgb_p1, pred_a_p1, meta_p1 = render_gaussians_3d(
                g_render, K, R, t, H, W, bg=bg_rgb, return_alpha=True, return_meta=True
            )
            diff_p1 = pred_rgb_p1 - gt
            residual_scalar = float(diff_p1.abs().mean().detach().cpu())
            residual_ema_val = frame_residual_ema[fj].update(residual_scalar)
            frame_w_scalar = float(frame_weight_from_residual(residual_ema_val, frame_weight_k))

            person_mask_for_vis = person_mask_curr
            if bool(shell_enable) and (inner_supervision_mask is not None or outer_supervision_mask is not None):
                person_mask_for_vis = None
            visibility = compute_gaussian_visibility(
                xyz=xyz_render.detach(),
                K=K,
                R=R,
                t=t,
                image_h=H,
                image_w=W,
                alpha_map=pred_a_p1.detach(),
                person_mask=person_mask_for_vis,
                renderer_meta=meta_p1,
                min_alpha=max(float(alpha_threshold), 0.005),
            )

            u_proj = None
            v_proj = None
            valid_proj = None
            if bool(shell_enable) and (inner_supervision_mask is not None or outer_supervision_mask is not None):
                u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K, R, t)
                inside_proj = valid_proj & (u_proj >= 0) & (u_proj <= float(W - 1)) & (v_proj >= 0) & (v_proj <= float(H - 1))
                layer_vis = torch.ones((n_gauss,), dtype=torch.float32, device=device)
                inner_sel = (g.shell_layer == 0)

                if inner_supervision_mask is not None:
                    inner_s = _sample_mask_at_projected_uv(inner_supervision_mask, u_proj, v_proj, inside_proj, H, W)
                    if bool(inner_sel.any()):
                        layer_vis[inner_sel] = layer_vis[inner_sel] * inner_s[inner_sel]

                if outer_supervision_mask is not None:
                    outer_s = _sample_mask_at_projected_uv(outer_supervision_mask, u_proj, v_proj, inside_proj, H, W)
                    if face_gap_half_curr is not None and face_gap_scale_curr is not None and face_gap_support_curr is not None and g.bind_face_idx is not None:
                        face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long)
                        gap_valid = (face_idx_all >= 0) & (face_idx_all < int(face_gap_half_curr.shape[0]))
                        gap_half_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
                        gap_scale_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
                        gap_support_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
                        if bool(gap_valid.any()):
                            face_idx_valid = face_idx_all[gap_valid]
                            gap_half_frame[gap_valid] = face_gap_half_curr.index_select(0, face_idx_valid)
                            gap_scale_frame[gap_valid] = face_gap_scale_curr.index_select(0, face_idx_valid)
                            gap_support_frame[gap_valid] = face_gap_support_curr.index_select(0, face_idx_valid)
                        outer_gap_half_n_accum.append(gap_half_frame.detach())
                        outer_gap_scale_accum.append(gap_scale_frame.detach())
                        outer_gap_support_accum.append(gap_support_frame.detach())
                    if outer_detached_mask_curr is not None:
                        outer_detached_s = _sample_mask_at_projected_uv(
                            outer_detached_mask_curr,
                            u_proj,
                            v_proj,
                            inside_proj,
                            H,
                            W,
                        )
                    else:
                        outer_detached_s = torch.zeros_like(outer_s)
                    # In outer-only phase, suppress inner visibility only on trigger-ring regions
                    # so outer points can absorb contour/color residuals.
                    if phase_tag == "phase_outer" and bool(inner_sel.any()):
                        if bool(outer_phase_inner_full_mask):
                            layer_vis[inner_sel] = 0.0
                        else:
                            inner_scale = float(np.clip(float(outer_phase_inner_vis_scale), 0.0, 1.0))
                            if inner_scale < 1.0:
                                inner_keep = (1.0 - outer_s * (1.0 - inner_scale)).clamp(min=inner_scale, max=1.0)
                                layer_vis[inner_sel] = layer_vis[inner_sel] * inner_keep[inner_sel]
                    outer_inside_s = outer_s
                    if outer_base_mask is not None:
                        outer_inside_s = _sample_mask_at_projected_uv(outer_base_mask, u_proj, v_proj, inside_proj, H, W)
                    outer_inside_support_accum.append(outer_inside_s.detach())
                    outer_detached_support_accum.append(outer_detached_s.detach())
                    person_sample_mask = person_mask_curr if person_mask_curr is not None else outer_base_mask
                    if person_sample_mask is None:
                        person_s = torch.ones_like(outer_s)
                    else:
                        person_s = _sample_mask_at_projected_uv(person_sample_mask, u_proj, v_proj, inside_proj, H, W)
                    outer_person_support_accum.append(person_s.detach())
                    if hand_mask_curr is not None:
                        hand_s_metric = _sample_mask_at_projected_uv(hand_mask_curr, u_proj, v_proj, inside_proj, H, W)
                    else:
                        hand_s_metric = torch.zeros_like(outer_s)
                    outer_hand_support_accum.append(hand_s_metric.detach())
                    if outer_head_eval_mask_curr is not None:
                        head_sample_mask = outer_head_eval_mask_curr
                        if int(outer_head_support_sample_dilate_px) > 0:
                            head_sample_mask = _dilate_binary_mask_torch(
                                head_sample_mask,
                                int(outer_head_support_sample_dilate_px),
                            )
                        head_eval_s = _sample_mask_at_projected_uv(head_sample_mask, u_proj, v_proj, inside_proj, H, W)
                        outer_head_support_accum.append(head_eval_s.detach())
                    outer_sel = (g.shell_layer == 1)
                    if bool(outer_sel.any()):
                        inside_support_th = float(np.clip(float(outer_inside_support_th), 0.0, 1.0))
                        outer_soft = outer_s.clamp(0.0, 1.0)
                        detached_vis_boost = float(max(outer_detached_vis_boost, 0.0))
                        if detached_vis_boost > 0.0:
                            outer_soft = torch.maximum(
                                outer_soft,
                                (outer_detached_s * detached_vis_boost).clamp(0.0, 1.0),
                            )
                        if edge_trigger_map_curr is not None and bool(outer_edge_trigger_enable):
                            outer_edge_s = _sample_mask_at_projected_uv(edge_trigger_map_curr, u_proj, v_proj, inside_proj, H, W)
                        if bool(outer_contour_inside_only):
                            outer_soft = outer_soft * (outer_inside_s >= inside_support_th).float()
                        layer_vis[outer_sel] = layer_vis[outer_sel] * outer_soft[outer_sel]

                if edge_trigger_map_curr is not None and bool(outer_edge_trigger_enable):
                    edge_proj_all = _sample_mask_at_projected_uv(edge_trigger_map_curr, u_proj, v_proj, inside_proj, H, W)
                    edge_accum.append(edge_proj_all.detach())

                boundary_hard_active = str(outer_boundary_enforce).lower() == "hard"
                if boundary_hard_active and is_outer_geom_phase:
                    boundary_hard_active = outer_geom_phase_progress >= float(
                        np.clip(float(outer_boundary_geom_start_frac), 0.0, 1.0)
                    )
                if boundary_hard_active and int(outer_boundary_check_every) > 0 and ((it + 1) % int(outer_boundary_check_every) == 0):
                    if outer_base_mask is not None:
                        outer_boundary_mask = _dilate_binary_mask_torch(outer_base_mask, max(int(outer_boundary_tol_px), 0)).clamp(0.0, 1.0)
                        outside_boundary = (1.0 - _sample_mask_at_projected_uv(outer_boundary_mask, u_proj, v_proj, inside_proj, H, W)).clamp(0.0, 1.0)
                        outer_sel_b = (g.shell_layer == 1)
                        viol = outer_sel_b & (outside_boundary > 0.5)
                        if bool(viol.any()):
                            outer_boundary_hard_clamp_count += int(viol.sum().item())
                            alpha_cap = float(np.clip(float(outer_boundary_alpha_cap), 1e-5, 0.999))
                            cap_logit = math.log(alpha_cap / (1.0 - alpha_cap))
                            with torch.no_grad():
                                g.opacity_logit.data[viol] = torch.minimum(g.opacity_logit.data[viol], torch.full_like(g.opacity_logit.data[viol], cap_logit))
                                if is_outer_geom_phase:
                                    sdec = float(np.clip(float(outer_boundary_scale_decay), 0.0, 1.0))
                                    g.log_scales.data[viol, :2] = g.log_scales.data[viol, :2] + math.log(max(sdec, 1e-6))
                                    odec = float(np.clip(float(outer_boundary_offset_decay), 0.0, 1.0))
                                    g.bind_face_offset_local.data[viol] = g.bind_face_offset_local.data[viol] * odec
                            outer_boundary_violation_counter[viol] += 1.0
                        safe = outer_sel_b & (~viol)
                        if bool(safe.any()):
                            outer_boundary_violation_counter[safe] = 0.0
                        prune_cand = outer_sel_b & (outer_boundary_violation_counter >= float(max(int(outer_boundary_patience), 1))) & (outer_person_support_accum[-1] < 0.1 if len(outer_person_support_accum)>0 else torch.ones_like(outer_sel_b, dtype=torch.float32, device=device)>0)
                        if is_outer_geom_phase and bool(prune_cand.any()):
                            g.prune_state.data[prune_cand] = 2

                visibility = visibility * layer_vis

            if bool(use_border_attenuation) and person_mask_curr is not None and int(border_attenuation_width) > 0:
                if fj not in border_att_cache:
                    pm_np = (person_mask_curr[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    border_att_cache[fj] = build_border_attenuation(pm_np, int(border_attenuation_width))
                if u_proj is None or v_proj is None or valid_proj is None:
                    u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K, R, t)
                uv_proj = torch.stack([u_proj, v_proj], dim=1)
                border_w = sample_border_weight(border_att_cache[fj], uv_proj, valid_proj)
                visibility = visibility * border_w

            active_mask = (g.prune_state != 2)
            visibility = visibility * active_mask.float()
            # Visibility floor is applied to inner points only.
            # This avoids reviving weak outer points outside trigger-support regions.
            inner_vis_floor = 0.08 if (it < phase_inner_end) else 0.02
            if inner_vis_floor > 0.0:
                vis_pos = visibility > 0.0
                if g.shell_layer is not None:
                    inner_pos = vis_pos & (g.shell_layer == 0)
                    visibility = torch.where(inner_pos, visibility.clamp(min=inner_vis_floor), visibility)
                else:
                    visibility = torch.where(vis_pos, visibility.clamp(min=inner_vis_floor), visibility)
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
                g_render, K, R, t, H, W, bg=bg_rgb, return_alpha=True, gaussian_weights=visibility.detach()
            )
            outer_residual_term = pred_rgb.new_tensor(0.0)
            if (
                is_outer_color_phase
                and outer_supervision_mask is not None
                and g.shell_layer is not None
                and float(outer_phase_residual_weight) > 0.0
            ):
                vis_det = visibility.detach()
                inner_only_vis = vis_det.clone()
                inner_only_vis[g.shell_layer == 1] = 0.0
                outer_only_vis = vis_det.clone()
                outer_only_vis[g.shell_layer == 0] = 0.0
                black_bg = gt.new_zeros((3,))
                inner_rgb_black, _ = render_gaussians_3d(
                    g_render,
                    K,
                    R,
                    t,
                    H,
                    W,
                    bg=black_bg,
                    return_alpha=True,
                    gaussian_weights=inner_only_vis,
                )
                outer_rgb_black, _ = render_gaussians_3d(
                    g_render,
                    K,
                    R,
                    t,
                    H,
                    W,
                    bg=black_bg,
                    return_alpha=True,
                    gaussian_weights=outer_only_vis,
                )
                outer_roi = outer_supervision_mask.clamp(0.0, 1.0)
                roi3 = outer_roi.expand_as(outer_rgb_black)
                residual_target = (gt_outer_target * roi3).clamp(0.0, 1.0)
                outer_residual_term = (outer_rgb_black - residual_target).abs()
                outer_residual_term = (outer_residual_term * roi3).sum() / (roi3.sum() + 1e-6)

            if phase_tag == "phase_inner":
                gt_phase_target = gt_inner_target
            elif phase_tag == "phase_outer":
                gt_phase_target = gt_outer_target
            else:
                gt_phase_target = gt_combined_target

            diff = pred_rgb - gt_phase_target
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
                    outer_band_term = outer_color_band_loss(pred_rgb, gt_outer_target, gate_geom_band)

            frame_loss = (
                photo
                + float(l1_weight) * l1
                + float(dq_outer_color_band_weight) * outer_band_term
                + float(outer_phase_residual_weight) * outer_residual_term
            )

            if outer_supervision_mask is not None and float(outer_silhouette_weight) > 0.0:
                sil_gt = outer_supervision_mask.clamp(0.0, 1.0)
                sil_err = (pred_a - sil_gt).abs()
                if float(outer_silhouette_band_weight) > 1.0:
                    sil_band = _compute_contour_band_torch(
                        sil_gt,
                        width_px=max(1, int(outer_silhouette_band_px)),
                    )
                    sil_w = (1.0 + (float(outer_silhouette_band_weight) - 1.0) * sil_band).clamp(1.0, 16.0)
                    sil_loss = (sil_err * sil_w).sum() / (sil_w.sum() + 1e-6)
                else:
                    sil_loss = sil_err.mean()
                frame_loss = frame_loss + float(outer_silhouette_weight) * sil_loss

            if float(detail_hf_weight) > 0.0:
                hf_pred = _image_laplacian_chw(pred_rgb)
                hf_gt = _image_laplacian_chw(gt_phase_target)
                hf_err = (hf_pred - hf_gt).abs()
                if outer_supervision_mask is not None:
                    hf_w = outer_supervision_mask.clamp(0.0, 1.0)
                elif person_mask_curr is not None:
                    hf_w = person_mask_curr.clamp(0.0, 1.0)
                else:
                    hf_w = None
                if hf_w is not None and bool(detail_hf_exclude_face) and face_boxes is not None and face_boxes[fj] is not None:
                    fx0, fy0, fx1, fy1 = face_boxes[fj]
                    x0 = max(0, int(fx0))
                    y0 = max(0, int(fy0))
                    x1 = min(W - 1, int(fx1))
                    y1 = min(H - 1, int(fy1))
                    hf_w = hf_w.clone()
                    hf_w[:, y0:y1 + 1, x0:x1 + 1] = 0.0
                if hf_w is not None:
                    hf_w3 = hf_w.expand_as(hf_err)
                    hf_loss = (hf_err * hf_w3).sum() / (hf_w3.sum() + 1e-6)
                else:
                    hf_loss = hf_err.mean()
                frame_loss = frame_loss + float(detail_hf_weight) * hf_loss

            leak_mask = inner_supervision_mask
            if leak_mask is None:
                leak_mask = person_mask_curr
            if leak_mask is None:
                leak_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
            if leak_mask is None:
                leak_mask = outer_supervision_mask
            if leak_mask is not None and float(smpl_alpha_leak_weight) > 0.0:
                outside = (1.0 - leak_mask).clamp(0.0, 1.0)
                leak = (pred_a * outside).sum() / (outside.sum() + 1e-6)
                frame_loss = frame_loss + float(smpl_alpha_leak_weight) * leak
            if (
                person_contour_band_frames is not None
                and person_mask_curr is not None
                and person_contour_band_frames[fj] is not None
                and float(person_mask_contour_leak_weight) > 0.0
            ):
                contour_band_curr = person_contour_band_frames[fj]
                outside_band = (contour_band_curr * (1.0 - person_mask_curr)).clamp(0.0, 1.0)
                leak_band = (pred_a * outside_band).sum() / (outside_band.sum() + 1e-6)
                frame_loss = frame_loss + float(person_mask_contour_leak_weight) * leak_band

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
        edge_mean = (
            torch.stack(edge_accum, dim=0).mean(dim=0)
            if len(edge_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_detached_support_mean = (
            torch.stack(outer_detached_support_accum, dim=0).mean(dim=0)
            if len(outer_detached_support_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_gap_half_n_mean = (
            torch.stack(outer_gap_half_n_accum, dim=0).mean(dim=0)
            if len(outer_gap_half_n_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_gap_scale_mean = (
            torch.stack(outer_gap_scale_accum, dim=0).mean(dim=0)
            if len(outer_gap_scale_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_gap_support_mean = (
            torch.stack(outer_gap_support_accum, dim=0).mean(dim=0)
            if len(outer_gap_support_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_drive_support_mean = (
            float(max(outer_detached_support_boost, 0.0))
            * outer_detached_support_mean.detach().clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        outer_drive_support_mean = torch.maximum(
            outer_drive_support_mean,
            outer_gap_support_mean.detach().clamp(0.0, 1.0),
        ).clamp(0.0, 1.0)
        outer_head_support_mean = (
            torch.stack(outer_head_support_accum, dim=0).mean(dim=0)
            if len(outer_head_support_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_inside_support_mean = (
            torch.stack(outer_inside_support_accum, dim=0).mean(dim=0)
            if len(outer_inside_support_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_person_support_mean = (
            torch.stack(outer_person_support_accum, dim=0).mean(dim=0)
            if len(outer_person_support_accum) > 0
            else outer_inside_support_mean.detach()
        )
        outer_hand_support_mean = (
            torch.stack(outer_hand_support_accum, dim=0).mean(dim=0)
            if len(outer_hand_support_accum) > 0
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
        outer_face_spawn_weight = None
        outer_face_gap_half_avg = None
        outer_face_gap_scale_avg = None
        if (
            is_outer_geom_phase
            and bool(shell_enable)
            and bool(outer_trigger_face_activate_enable)
            and outer_face_weight_sum is not None
            and outer_face_gap_half_weighted_sum is not None
            and outer_face_gap_scale_weighted_sum is not None
        ):
            outer_face_spawn_weight = outer_face_weight_sum.detach().clamp_min(0.0)
            valid_face_weight = outer_face_spawn_weight > 0.0
            if bool(valid_face_weight.any()):
                denom = outer_face_spawn_weight.clamp_min(1e-8)
                outer_face_gap_half_avg = outer_face_gap_half_weighted_sum.detach() / denom
                outer_face_gap_scale_avg = outer_face_gap_scale_weighted_sum.detach() / denom
        # Keep all policy vectors strictly 1D to avoid accidental broadcasting (NxN).
        vis_mean = vis_mean.view(-1)
        edge_mean = edge_mean.view(-1)
        outer_head_support_mean = outer_head_support_mean.view(-1)
        outer_inside_support_mean = outer_inside_support_mean.view(-1)
        outer_person_support_mean = outer_person_support_mean.view(-1)
        outer_hand_support_mean = outer_hand_support_mean.view(-1)

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
            inside_support_th = float(np.clip(float(outer_inside_support_th), 0.0, 1.0))
            inside_support_mask = outer_inside_support_mean.detach() >= inside_support_th
            outside_support_mask = ~inside_support_mask
            low_support_th = max(float(outer_prune_support_th), inside_support_th)
            low_support_cond = outer_mask & active_mask & (outer_drive_support_mean.detach() < float(low_support_th))
            outer_low_support_counter = torch.where(
                low_support_cond,
                outer_low_support_counter + 1.0,
                torch.zeros_like(outer_low_support_counter),
            )

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

            if bool(inner_occlusion_soft_enable) and phase_tag == "phase_occlusion":
                to_soft = inner_mask & (inner_soft_counter >= float(max(int(inner_soft_patience), 1)))
                g.prune_state[to_soft] = 1

            if (
                bool(inner_occlusion_hard_enable)
                and phase_tag == "phase_occlusion"
                and int(inner_prune_every) > 0
                and ((it + 1) % int(inner_prune_every) == 0)
            ):
                to_hard = select_inner_hardprune_candidates(
                    inner_mask=inner_mask,
                    soft_counter=inner_soft_counter,
                    hard_counter=inner_hard_counter,
                    soft_patience=int(inner_soft_patience),
                    hard_patience=int(inner_hard_patience),
                )
                g.prune_state[to_hard] = 2

            # Outer score / prune policy.
            edge_metric_base = (g.bind_mix_weight.view(-1).detach() - 0.5).abs() * 2.0
            edge_metric = edge_metric_base
            if bool(outer_edge_trigger_enable):
                ew = float(np.clip(float(outer_edge_trigger_weight), 0.0, 1.0))
                edge_metric = ((1.0 - ew) * edge_metric_base + ew * edge_mean.detach()).clamp(0.0, 1.0)
            shell_score = compute_shell_score(
                residual_ema=densify_stats.residual_ema,
                edge=edge_metric,
                curv_conf=g.bind_mix_weight.view(-1).detach(),
                leak=(1.0 - vis_mean.detach()).clamp(0.0, 1.0),
            )
            if float(outer_support_weight) > 0.0:
                shell_score = shell_score + float(outer_support_weight) * outer_drive_support_mean.detach().clamp(0.0, 1.0)
            if float(outer_detached_support_boost) > 0.0:
                shell_score = shell_score + 0.5 * float(outer_detached_support_boost) * outer_detached_support_mean.detach().clamp(0.0, 1.0)
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

            if (
                is_outer_geom_phase
                and int(shell_prune_every) > 0
                and ((it + 1) % int(shell_prune_every) == 0)
            ):
                to_prune_outer = select_outer_prune_candidates(
                    outer_mask=outer_mask,
                    outer_vis=vis_mean,
                    outer_score=shell_score,
                    low_vis_th=prune_low_vis_th,
                    low_score_percent=prune_low_score_percent,
                )
                if float(outer_prune_support_th) > 0.0 and (it + 1) > int(phase_inner_end):
                    sup_th = float(max(float(outer_prune_support_th), float(outer_prune_dual_min_support)))
                    low_sup = outer_mask & active_mask & (outer_drive_support_mean.detach() < sup_th)
                    if bool(outer_prune_dual_enable):
                        low_sup = low_sup & (vis_mean.detach() <= float(max(outer_prune_dual_max_vis, 0.0)))
                    low_sup_idx = torch.nonzero(low_sup, as_tuple=False).view(-1)
                    if int(low_sup_idx.numel()) > 0:
                        if int(to_prune_outer.numel()) > 0:
                            to_prune_outer = torch.unique(torch.cat([to_prune_outer, low_sup_idx], dim=0))
                        else:
                            to_prune_outer = low_sup_idx
                if int(outer_low_support_patience) > 0:
                    force_low = outer_mask & active_mask & (
                        outer_low_support_counter >= float(max(int(outer_low_support_patience), 1))
                    )
                    if bool(outer_prune_dual_enable):
                        sup_th_force = float(max(float(outer_prune_support_th), float(outer_prune_dual_min_support), 1e-6))
                        force_low = (
                            force_low
                            & (outer_drive_support_mean.detach() < sup_th_force)
                            & (vis_mean.detach() <= float(max(outer_prune_dual_max_vis, 0.0)))
                        )
                    force_low_idx = torch.nonzero(force_low, as_tuple=False).view(-1)
                    if int(force_low_idx.numel()) > 0:
                        if int(to_prune_outer.numel()) > 0:
                            to_prune_outer = torch.unique(torch.cat([to_prune_outer, force_low_idx], dim=0))
                        else:
                            to_prune_outer = force_low_idx
                overlap_offset_th = float(max(float(outer_overlap_prune_offset_th), 0.0))
                overlap_detached_th = float(np.clip(float(outer_overlap_prune_detached_th), 0.0, 1.0))
                if overlap_offset_th > 0.0:
                    overlap_mask = (
                        outer_mask
                        & active_mask
                        & (g.bind_face_offset_local[:, 2].abs() <= overlap_offset_th)
                        & (outer_detached_support_mean.detach() < overlap_detached_th)
                    )
                    overlap_idx = torch.nonzero(overlap_mask, as_tuple=False).view(-1)
                    if int(overlap_idx.numel()) > 0:
                        if int(to_prune_outer.numel()) > 0:
                            to_prune_outer = torch.unique(torch.cat([to_prune_outer, overlap_idx], dim=0))
                        else:
                            to_prune_outer = overlap_idx
                if bool(outer_edge_trigger_enable) and int(to_prune_outer.numel()) > 0:
                    q = float(np.clip(float(outer_edge_prune_protect_quantile), 0.0, 1.0))
                    if q > 0.0:
                        edge_q = torch.quantile(edge_mean.detach(), q)
                        protect_m = (edge_mean.detach() >= edge_q) & inside_support_mask
                        keep_m = ~protect_m[to_prune_outer]
                        to_prune_outer = to_prune_outer[keep_m]
                if int(to_prune_outer.numel()) > 0:
                    g.prune_state[to_prune_outer] = 2

            if is_outer_geom_phase:
                n_inner_active = int((inner_mask & active_mask).sum().item())
                n_outer_active = int((outer_mask & active_mask).sum().item())
                if float(shell_max_ratio) <= 0.0:
                    # Disable ratio-based outer cap: spawn is only limited by available outer slots.
                    max_new = int((g.shell_layer == 1).sum().item())
                else:
                    _outer_init_budget, outer_budget = estimate_outer_budget(
                        n_inner=max(n_inner_active, 1),
                        init_ratio=0.15,
                        max_ratio=float(shell_max_ratio),
                    )
                    max_new = max(0, outer_budget - n_outer_active)
                if dq_enable and dq_fail_active:
                    max_new = int(round(float(max_new) * float(max(dq_spawn_scale_if_fail, 0.0))))
                if max_new > 0:
                    # Reuse previously hard-pruned outer slots to keep tensor shapes stable.
                    spawn_slots = torch.nonzero((g.shell_layer == 1) & (g.prune_state == 2), as_tuple=False).view(-1)
                    occupied_outer_faces = None
                    if g.bind_face_idx is not None:
                        occupied_outer_faces = g.bind_face_idx[
                            (g.shell_layer == 1) & (g.prune_state != 2)
                        ].to(device=device, dtype=torch.long)
                        occupied_outer_faces = occupied_outer_faces[occupied_outer_faces >= 0]
                        if int(occupied_outer_faces.numel()) > 0:
                            occupied_outer_faces = torch.unique(occupied_outer_faces)
                        else:
                            occupied_outer_faces = None
                    if (
                        bool(outer_trigger_face_activate_enable)
                        and outer_face_spawn_weight is not None
                        and outer_face_gap_half_avg is not None
                        and outer_face_gap_scale_avg is not None
                        and outer_face_spawned_mask is not None
                        and outer_face_spawn_iter is not None
                        and outer_face_spawn_frame_idx is not None
                        and trigger_frame_data is not None
                        and trigger_frame_data.verts_t is not None
                        and trigger_frame_data.faces_t is not None
                        and trigger_frame_data.face_n_t is not None
                        and g.bind_face_idx is not None
                        and g.bind_bary is not None
                        and int(spawn_slots.numel()) > 0
                    ):
                        face_score = outer_face_spawn_weight.view(-1)
                        face_candidate_mask = (
                            (face_score > 0.0)
                            & (outer_face_gap_half_avg.view(-1) > 0.0)
                            & (outer_face_gap_scale_avg.view(-1) > 0.0)
                            & (~outer_face_spawned_mask.view(-1))
                        )
                        face_idx = torch.nonzero(face_candidate_mask, as_tuple=False).view(-1)
                        if occupied_outer_faces is not None and int(face_idx.numel()) > 0:
                            face_idx = face_idx[~torch.isin(face_idx, occupied_outer_faces)]
                        if int(face_idx.numel()) > 0:
                            max_face_spawn = min(int(max_new), int(spawn_slots.numel()))
                            if int(face_idx.numel()) > int(max_face_spawn):
                                top = torch.topk(face_score.index_select(0, face_idx), k=int(max_face_spawn), largest=True).indices
                                face_idx = face_idx.index_select(0, top)
                            n_spawn = min(int(face_idx.numel()), int(spawn_slots.numel()))
                            if n_spawn > 0:
                                dst = spawn_slots[:n_spawn]
                                fs = face_idx[:n_spawn]
                                with torch.no_grad():
                                    ftri = trigger_frame_data.faces_t.index_select(0, fs)
                                    tri = trigger_frame_data.verts_t.index_select(0, ftri.reshape(-1)).reshape(-1, 3, 3)
                                    cent = tri.mean(dim=1)
                                    nf = trigger_frame_data.face_n_t.index_select(0, fs)
                                    gap_half_face = outer_face_gap_half_avg.index_select(0, fs).clamp(min=0.0)
                                    gap_scale_face = outer_face_gap_scale_avg.index_select(0, fs).clamp(min=0.0)
                                    if bool(outer_viewplane_init_enable):
                                        n_init = gap_half_face.clamp(max=float(shell_offset_max_n))
                                    elif center_init_rule == "segment_midpoint":
                                        n_init = torch.full((n_spawn,), 0.5 * float(shell_offset_max_n), dtype=torch.float32, device=device)
                                    else:
                                        n_init = torch.full((n_spawn,), 0.25 * float(shell_offset_max_n), dtype=torch.float32, device=device)
                                    if not bool(outer_viewplane_init_enable):
                                        n_init = n_init.clamp(max=float(max(outer_center_phase1_normal_tune_cap, 0.0)))
                                    xyz_init = cent + n_init.unsqueeze(1) * nf
                                    anchor = ftri[:, 0]
                                    g.xyz[dst] = xyz_init
                                    g.bind_face_idx[dst] = fs
                                    g.bind_bary[dst, :] = (1.0 / 3.0)
                                    if g.bind_idx is not None:
                                        g.bind_idx[dst] = anchor
                                    if g.bind_offset is not None:
                                        av = trigger_frame_data.verts_t.index_select(0, anchor)
                                        g.bind_offset[dst] = cent - av
                                    g.bind_face_offset_local.data[dst, :] = 0.0
                                    g.bind_face_offset_local.data[dst, 2] = n_init
                                    g.opacity_logit[dst] = -5.0
                                    scale_n = gap_scale_face.clamp(min=1e-4, max=float(shell_offset_max_n))
                                    g.log_scales.data[dst, 2] = torch.log(scale_n)
                                    scale_xy = (
                                        float(max(outer_gap_tangent_scale_ratio, 0.0)) * gap_scale_face
                                    ).clamp(min=1e-4, max=float(shell_offset_max_n))
                                    g.log_scales.data[dst, 0] = torch.maximum(
                                        g.log_scales.data[dst, 0],
                                        torch.log(scale_xy),
                                    )
                                    g.log_scales.data[dst, 1] = torch.maximum(
                                        g.log_scales.data[dst, 1],
                                        torch.log(scale_xy),
                                    )
                                    if g.inner_bind_mode is not None:
                                        g.inner_bind_mode[dst] = -1
                                    g.prune_state[dst] = 0
                                    outer_face_spawned_mask[fs] = True
                                    for face_idx_i in fs.detach().cpu().tolist():
                                        outer_face_spawn_iter[face_idx_i] = int(it + 1)
                                        outer_face_spawn_frame_idx[face_idx_i] = int(fi)
                                outer_spawn_stats_by_frame.setdefault(
                                    spawn_frame_name,
                                    {"total": 0, "trigger": 0, "fallback": 0, "iters_hit": 0},
                                )
                                outer_spawn_stats_by_frame[spawn_frame_name]["total"] += int(n_spawn)
                                outer_spawn_stats_by_frame[spawn_frame_name]["trigger"] += int(n_spawn)
                                outer_spawn_stats_by_frame[spawn_frame_name]["iters_hit"] += 1
                                outer_spawn_iter_records.append({
                                    "iter": int(it + 1),
                                    "frame_idx": int(fi),
                                    "frame_name": spawn_frame_name,
                                    "mode": "trigger",
                                    "count": int(n_spawn),
                                    "face_indices": [int(x) for x in fs.detach().cpu().tolist()],
                                })

        # Global regularizers (stronger on low-visibility gaussians).
        low_vis_boost = 1.0 + float((1.0 - vis_mean).mean().detach().cpu())
        phase_alpha_mul = 2.0 if phase_tag in ("phase_inner", "phase_outer") else 1.0
        opacity_all = g.opacity()
        active_for_prior = (g.prune_state != 2)
        if g.shell_layer is not None:
            inner_prior_mask = active_for_prior & (g.shell_layer == 0)
            outer_prior_mask = active_for_prior & (g.shell_layer == 1)
            reg_opacity_inner = (
                opacity_prior_loss(opacity_all[inner_prior_mask], target=float(inner_opacity_prior_target))
                if bool(inner_prior_mask.any())
                else opacity_all.new_tensor(0.0)
            )
            reg_opacity_outer = (
                opacity_prior_loss(opacity_all[outer_prior_mask], target=float(outer_opacity_prior_target))
                if bool(outer_prior_mask.any())
                else opacity_all.new_tensor(0.0)
            )
            reg_opacity_term = reg_opacity_inner + reg_opacity_outer
        else:
            reg_opacity_term = (
                opacity_prior_loss(opacity_all[active_for_prior], target=0.5)
                if bool(active_for_prior.any())
                else opacity_all.new_tensor(0.0)
            )
        reg_opacity = float(alpha_prior_weight) * phase_alpha_mul * low_vis_boost * reg_opacity_term
        reg_cov = float(cov_penalty_weight) * low_vis_boost * covariance_penalty(g.scales(), float(cov_min), float(cov_max))
        offset_reg_w_eff = float(shell_offset_reg_weight)
        if phase_tag in ("phase_inner", "phase_outer"):
            offset_reg_w_eff *= float(max(shell_offset_reg_phase0_scale, 0.0))
        if g.bind_face_offset_local is not None and g.shell_layer is not None and bool((g.shell_layer == 1).any()):
            outer_sel = (g.shell_layer == 1)
            outer_offsets = g.bind_face_offset_local[outer_sel]
            if bool(outer_offset_edge_guided_reg):
                edge_outer = edge_mean.detach()[outer_sel].clamp(0.0, 1.0)
                non_edge_w = (1.0 - edge_outer).clamp(0.0, 1.0)
                if bool(outer_offset_reg_relax_support):
                    sup_outer = outer_drive_support_mean.detach()[outer_sel].clamp(0.0, 1.0)
                    non_edge_w = non_edge_w * (1.0 - sup_outer)
                reg_offset_term = ((outer_offsets * outer_offsets).sum(dim=1) * non_edge_w).sum() / (non_edge_w.sum() + 1e-6)
                reg_offset = float(offset_reg_w_eff) * reg_offset_term
            else:
                reg_offset = float(offset_reg_w_eff) * canonical_offset_l2(outer_offsets)
            if float(outer_offset_min_active_weight) > 0.0:
                active_outer_sel = outer_sel & active_for_prior
                if bool(active_outer_sel.any()):
                    sup_act = outer_drive_support_mean.detach()[active_outer_sel].clamp(0.0, 1.0)
                    if float(outer_offset_min_support_th) > 0.0:
                        sup_act = sup_act * (sup_act >= float(outer_offset_min_support_th)).float()
                    target_n = float(shell_offset_max_n) * float(outer_offset_min_target_scale) * sup_act
                    offset_n_abs = g.bind_face_offset_local[active_outer_sel, 2].abs()
                    reg_offset_min = float(outer_offset_min_active_weight) * F.relu(target_n - offset_n_abs).pow(2).mean()
                else:
                    reg_offset_min = opacity_all.new_tensor(0.0)
            else:
                reg_offset_min = opacity_all.new_tensor(0.0)
        else:
            reg_offset = float(offset_reg_w_eff) * canonical_offset_l2(g.bind_face_offset_local)
            reg_offset_min = opacity_all.new_tensor(0.0)
        alpha_cap_for_balance = float(dq_outer_alpha_cap_if_fail) if (dq_enable and dq_fail_active) else 0.40
        reg_alpha_balance = float(dq_layer_alpha_balance_weight) * layer_alpha_balance_loss(
            g.opacity(),
            g.shell_layer,
            outer_alpha_cap=alpha_cap_for_balance,
            inner_alpha_floor=0.03,
        )
        outside_support_th = float(np.clip(float(outer_outside_support_th), 0.0, 1.0))
        outer_outside_mask = (
            (g.shell_layer == 1)
            & active_for_prior
            & (outer_inside_support_mean.detach() < outside_support_th)
        )
        cap_a = float(np.clip(float(outer_outside_opacity_cap), 1e-4, 0.999))
        if bool(outer_outside_mask.any()):
            outer_out_opacity = opacity_all[outer_outside_mask]
            reg_outer_outside_opacity = float(max(outer_outside_opacity_weight, 0.0)) * (
                F.relu(outer_out_opacity - cap_a).pow(2).mean()
            )
            outer_out_offset = g.bind_face_offset_local[outer_outside_mask]
            reg_outer_outside_offset = float(max(outer_outside_offset_weight, 0.0)) * (
                (outer_out_offset * outer_out_offset).sum(dim=1).mean()
            )
        else:
            reg_outer_outside_opacity = opacity_all.new_tensor(0.0)
            reg_outer_outside_offset = opacity_all.new_tensor(0.0)
        if (
            bool(shell_enable)
            and g.shell_layer is not None
            and float(outer_contact_alpha_edge_th) > 0.0
            and bool((g.shell_layer == 1).any())
            and is_outer_geom_phase
        ):
            contact_sel = (g.shell_layer == 1) & active_for_prior
            if bool(contact_sel.any()):
                support_inside = outer_inside_support_mean.detach()[contact_sel].clamp(0.0, 1.0)
                support_detached = outer_detached_support_mean.detach()[contact_sel].clamp(0.0, 1.0)
                target_scale = (0.10 * support_inside + 0.85 * support_detached).clamp(0.0, 1.0)
                target_n = float(shell_offset_max_n) * target_scale
                offset_n = g.bind_face_offset_local[contact_sel, 2].abs()
                reg_outer_contact = float(outer_contact_alpha_edge_th) * (offset_n - target_n).pow(2).mean()
            else:
                reg_outer_contact = opacity_all.new_tensor(0.0)
        else:
            reg_outer_contact = opacity_all.new_tensor(0.0)

        if (
            bool(shell_enable)
            and g.shell_layer is not None
            and bool((g.shell_layer == 1).any())
            and is_outer_geom_phase
            and (not bool(outer_gap_hard_target_enable))
        ):
            gap_sel = (
                (g.shell_layer == 1)
                & active_for_prior
                & (outer_gap_support_mean.detach() >= float(max(outer_gap_target_support_th, 0.0)))
            )
            if bool(gap_sel.any()):
                target_half = outer_gap_half_n_mean.detach()[gap_sel].clamp(min=0.0, max=float(shell_offset_max_n))
                offset_n = g.bind_face_offset_local[gap_sel, 2].abs()
                reg_outer_gap = float(max(outer_gap_target_weight, 0.0)) * (offset_n - target_half).pow(2).mean()
                target_scale_n = outer_gap_scale_mean.detach()[gap_sel].clamp(min=1e-4, max=float(shell_offset_max_n))
                scale_n = g.scales()[gap_sel, 2].clamp_min(1e-6)
                reg_outer_gap_scale = float(max(outer_gap_scale_weight, 0.0)) * (scale_n - target_scale_n).pow(2).mean()
                target_scale_xy = (
                    float(max(outer_gap_tangent_scale_ratio, 0.0))
                    * outer_gap_scale_mean.detach()[gap_sel]
                ).clamp(min=1e-4, max=float(shell_offset_max_n))
                scale_xy = g.scales()[gap_sel, :2].mean(dim=1).clamp_min(1e-6)
                reg_outer_gap_tangent = float(max(outer_gap_tangent_scale_weight, 0.0)) * (
                    scale_xy - target_scale_xy
                ).pow(2).mean()
            else:
                reg_outer_gap = opacity_all.new_tensor(0.0)
                reg_outer_gap_scale = opacity_all.new_tensor(0.0)
                reg_outer_gap_tangent = opacity_all.new_tensor(0.0)
        else:
            reg_outer_gap = opacity_all.new_tensor(0.0)
            reg_outer_gap_scale = opacity_all.new_tensor(0.0)
            reg_outer_gap_tangent = opacity_all.new_tensor(0.0)
        if (
            float(outer_airspace_penalty_weight) > 0.0
            and bool(shell_enable)
            and g.shell_layer is not None
            and binding_enabled
            and frame_data is not None
            and bind_face_neighbor_idx_t is not None
            and bind_face_neighbor_w_t is not None
            and g.bind_face_idx is not None
        ):
            air_sel = (g.shell_layer == 1) & active_for_prior & (g.bind_face_idx >= 0)
            if bool(air_sel.any()):
                outer_idx = torch.nonzero(air_sel, as_tuple=False).view(-1)
                tri_frame = frame_data.verts_t.index_select(0, frame_data.faces_t.reshape(-1)).reshape(-1, 3, 3)
                face_centroids_frame = tri_frame.mean(dim=1)
                face_normals_frame = frame_data.face_n_t
                xyz_airspace = g.xyz
                if (
                    bool(bind_face_offset_straight_through)
                    and bind_mask_curr is not None
                    and bind_face_idx_t is not None
                ):
                    # Rebuild a fresh straight-through center tensor for the airspace regularizer.
                    # Reusing xyz_render would backprop through a graph already consumed by frame_loss.
                    xyz_airspace = _inject_bind_face_offset_straight_through_xyz(
                        xyz_base=g.xyz,
                        g=g,
                        frame_data=frame_data,
                        bind_mask_curr=bind_mask_curr,
                        bind_face_idx_t=bind_face_idx_t,
                    )
                reg_outer_airspace = float(outer_airspace_penalty_weight) * _compute_outer_airspace_penalty(
                    centers_world=xyz_airspace.index_select(0, outer_idx),
                    tangent_radius=g.scales().index_select(0, outer_idx)[:, :2].mean(dim=1),
                    bind_face_idx=g.bind_face_idx.index_select(0, outer_idx).to(device=device, dtype=torch.long),
                    face_centroids=face_centroids_frame,
                    face_normals=face_normals_frame,
                    face_tris=tri_frame,
                    neighbor_idx=bind_face_neighbor_idx_t,
                    neighbor_w=bind_face_neighbor_w_t,
                    main_support_scale=float(outer_airspace_main_support_scale),
                )
            else:
                reg_outer_airspace = opacity_all.new_tensor(0.0)
        else:
            reg_outer_airspace = opacity_all.new_tensor(0.0)

        reg_total = (
            reg_opacity
            + reg_cov
            + reg_offset
            + reg_offset_min
            + reg_alpha_balance
            + reg_outer_outside_opacity
            + reg_outer_outside_offset
            + reg_outer_contact
            + reg_outer_gap
            + reg_outer_gap_scale
            + reg_outer_gap_tangent
            + reg_outer_airspace
        )
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

        freeze_outer_center_now = False
        outer_center_ref: torch.Tensor | None = None
        if bool(shell_enable):
            inner_layer = (g.shell_layer == 0)
            outer_layer = (g.shell_layer == 1)
            if phase_tag == "phase_inner":
                with torch.no_grad():
                    for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit):
                        if p.grad is not None:
                            p.grad[outer_layer] = 0.0
            elif phase_tag == "phase_outer":
                with torch.no_grad():
                    for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit):
                        if p.grad is not None:
                            p.grad[inner_layer] = 0.0
                    if is_outer_geom_phase:
                        for p in (g.opacity_logit, g.rgb_logit):
                            if p.grad is not None:
                                p.grad[outer_layer] = 0.0
                    elif is_outer_color_phase:
                        for p in (g.log_scales, g.quat):
                            if p.grad is not None:
                                p.grad[outer_layer] = 0.0
            else:
                freeze_outer_center_now = bool(outer_center_freeze_after_phase1)

        # Offset center policy:
        # - inner center is always frozen;
        # - outer center can tune only in phase_outer (P1b).
        if bool(shell_enable) and g.bind_face_offset_local.grad is not None:
            outer_active = (g.shell_layer == 1) & (g.prune_state != 2)
            inner_or_bg = ~outer_active
            with torch.no_grad():
                g.bind_face_offset_local.grad[inner_or_bg] = 0.0
            allow_outer_center_tune = (
                is_outer_geom_phase
                and bool(outer_center_phase1_normal_tune_enable)
                and (not bool(freeze_outer_center_now))
            )
            if bool(outer_active.any()):
                if not allow_outer_center_tune:
                    with torch.no_grad():
                        g.bind_face_offset_local.grad[outer_active] = 0.0
                else:
                    base_scale = float(max(outer_offset_grad_scale, 0.0))
                    edge_scale = float(max(outer_offset_edge_grad_scale, 0.0))
                    max_scale = float(max(outer_offset_grad_scale_max, 1.0))
                    if base_scale <= 0.0:
                        with torch.no_grad():
                            g.bind_face_offset_local.grad[outer_active] = 0.0
                    else:
                        grad_mul = torch.ones((n_gauss,), dtype=torch.float32, device=device)
                        grad_mul[outer_active] = base_scale
                        if edge_scale > 0.0:
                            edge_w = edge_mean.detach().clamp(0.0, 1.0)
                            grad_mul[outer_active] = grad_mul[outer_active] * (1.0 + edge_scale * edge_w[outer_active])
                        grad_mul = grad_mul.clamp(min=0.0, max=max_scale)
                        with torch.no_grad():
                            g.bind_face_offset_local.grad[outer_active] *= grad_mul[outer_active].unsqueeze(1)
                            if float(outer_center_phase1_normal_tune_cap) > 0.0:
                                g.bind_face_offset_local.grad[outer_active, 0:2] = 0.0

        if bool(freeze_outer_center_now):
            with torch.no_grad():
                outer_center_ref = g.bind_face_offset_local.detach().clone()
        opt.step()
        if (
            bool(shell_enable)
            and g.shell_layer is not None
            and bool(outer_gap_hard_target_enable)
            and is_outer_geom_phase
        ):
            gap_sel_hard = (
                (g.shell_layer == 1)
                & (g.prune_state != 2)
                & (outer_gap_support_mean.detach() >= float(max(outer_gap_target_support_th, 0.0)))
            )
            if bool(gap_sel_hard.any()):
                hard_blend = float(np.clip(float(outer_gap_hard_target_blend), 0.0, 1.0))
                with torch.no_grad():
                    target_half_h = outer_gap_half_n_mean.detach()[gap_sel_hard].clamp(
                        min=0.0, max=float(shell_offset_max_n)
                    )
                    target_scale_n_h = outer_gap_scale_mean.detach()[gap_sel_hard].clamp(
                        min=1e-4, max=float(shell_offset_max_n)
                    )
                    target_scale_xy_h = (
                        float(max(outer_gap_tangent_scale_ratio, 0.0))
                        * outer_gap_scale_mean.detach()[gap_sel_hard]
                    ).clamp(min=1e-4, max=float(shell_offset_max_n))
                    if hard_blend >= 1.0 - 1e-6:
                        g.bind_face_offset_local[gap_sel_hard, 2] = target_half_h
                        g.log_scales[gap_sel_hard, 2] = torch.log(target_scale_n_h)
                        g.log_scales[gap_sel_hard, 0] = torch.log(target_scale_xy_h)
                        g.log_scales[gap_sel_hard, 1] = torch.log(target_scale_xy_h)
                    elif hard_blend > 0.0:
                        target_log_n_h = torch.log(target_scale_n_h)
                        target_log_xy_h = torch.log(target_scale_xy_h)
                        g.bind_face_offset_local[gap_sel_hard, 2] = (
                            (1.0 - hard_blend) * g.bind_face_offset_local[gap_sel_hard, 2]
                            + hard_blend * target_half_h
                        )
                        g.log_scales[gap_sel_hard, 2] = (
                            (1.0 - hard_blend) * g.log_scales[gap_sel_hard, 2]
                            + hard_blend * target_log_n_h
                        )
                        g.log_scales[gap_sel_hard, 0] = (
                            (1.0 - hard_blend) * g.log_scales[gap_sel_hard, 0]
                            + hard_blend * target_log_xy_h
                        )
                        g.log_scales[gap_sel_hard, 1] = (
                            (1.0 - hard_blend) * g.log_scales[gap_sel_hard, 1]
                            + hard_blend * target_log_xy_h
                        )
        if outer_center_ref is not None:
            with torch.no_grad():
                outer_layer = (g.shell_layer == 1)
                g.bind_face_offset_local[outer_layer] = outer_center_ref[outer_layer]

        # Mild regularization: keep parameters in a reasonable range
        with torch.no_grad():
            global_log_scale_min = -7.0
            if fg_log_scale_min is not None:
                global_log_scale_min = min(global_log_scale_min, float(fg_log_scale_min))
            if bg_log_scale_min is not None:
                global_log_scale_min = min(global_log_scale_min, float(bg_log_scale_min))
            g.log_scales.clamp_(min=float(global_log_scale_min), max=0.0)      # exp range ~ [exp(min), 1]
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
                if (
                    g.shell_layer is not None
                    and (outer_log_scale_min is not None or outer_log_scale_max is not None)
                ):
                    outer_b = bind_mask & (g.shell_layer == 1)
                    if bool(outer_b.any()):
                        min_v = -9.0 if outer_log_scale_min is None else float(outer_log_scale_min)
                        max_v = 0.0 if outer_log_scale_max is None else float(outer_log_scale_max)
                        g.log_scales[outer_b] = g.log_scales[outer_b].clamp(min=min_v, max=max_v)
                        detached_floor_th = float(np.clip(float(outer_detached_support_th), 0.0, 1.0))
                        detached_floor_mask = (
                            outer_b
                            & (g.prune_state != 2)
                            & (
                                (outer_detached_support_mean.detach() >= detached_floor_th)
                                | (outer_gap_support_mean.detach() >= detached_floor_th)
                            )
                        )
                        if bool(detached_floor_mask.any()):
                            floor_log_scale = float(outer_detached_log_scale_floor)
                            gap_xy_floor = (
                                float(max(outer_gap_tangent_scale_ratio, 0.0))
                                * outer_gap_scale_mean.detach()[detached_floor_mask]
                            ).clamp(min=float(np.exp(floor_log_scale)), max=float(shell_offset_max_n))
                            gap_xy_floor_log = torch.log(gap_xy_floor.clamp_min(1e-6)).unsqueeze(1).expand(-1, 2)
                            g.log_scales[detached_floor_mask, :2] = torch.maximum(
                                g.log_scales[detached_floor_mask, :2],
                                torch.maximum(
                                    gap_xy_floor_log,
                                    torch.full_like(g.log_scales[detached_floor_mask, :2], floor_log_scale),
                                ),
                            )
                            floor_opacity = float(np.clip(float(outer_detached_opacity_floor), 1e-4, 0.999))
                            floor_logit = float(np.log(floor_opacity / (1.0 - floor_opacity)))
                            g.opacity_logit[detached_floor_mask] = torch.maximum(
                                g.opacity_logit[detached_floor_mask],
                                torch.full_like(g.opacity_logit[detached_floor_mask], floor_logit),
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
            if g.shell_layer is not None:
                inner_m = (g.shell_layer == 0)
                if bool(inner_m.any()):
                    g.bind_face_offset_local[inner_m] = 0.0
                offset_cap_active = not bool(outer_offset_unbounded)
                if offset_cap_active and is_outer_geom_phase:
                    offset_cap_active = outer_geom_phase_progress >= float(
                        np.clip(float(outer_offset_cap_geom_start_frac), 0.0, 1.0)
                    )
                if offset_cap_active:
                    outer_m = (g.shell_layer == 1)
                    if bool(outer_m.any()):
                        outer_idx = torch.nonzero(outer_m, as_tuple=False).view(-1)
                        outer_edge = edge_mean.detach().index_select(0, outer_idx)
                        outer_sup = outer_drive_support_mean.detach().index_select(0, outer_idx)
                        t_cap, n_cap = compute_outer_dynamic_caps(
                            edge_vals=outer_edge,
                            support_vals=outer_sup,
                            outer_offset_auto_range_enable=bool(outer_offset_auto_range_enable),
                            outer_offset_auto_range_min_scale=float(outer_offset_auto_range_min_scale),
                            outer_offset_auto_range_edge_mix=float(outer_offset_auto_range_edge_mix),
                            shell_offset_max_t=float(shell_offset_max_t),
                            shell_offset_max_n=float(shell_offset_max_n),
                        )
                        offs = g.bind_face_offset_local.index_select(0, outer_idx)
                        offs[:, 0] = torch.maximum(torch.minimum(offs[:, 0], t_cap), -t_cap)
                        offs[:, 1] = torch.maximum(torch.minimum(offs[:, 1], t_cap), -t_cap)
                        offs[:, 2] = torch.maximum(torch.minimum(offs[:, 2], n_cap), -n_cap)
                        if is_outer_geom_phase and float(outer_center_phase1_normal_tune_cap) > 0.0:
                            n_cap_fixed = torch.full_like(offs[:, 2], float(max(outer_center_phase1_normal_tune_cap, 0.0)))
                            gap_cap_mask = outer_gap_support_mean.detach().index_select(0, outer_idx) >= float(max(outer_gap_target_support_th, 0.0))
                            if bool(gap_cap_mask.any()):
                                n_cap_fixed[gap_cap_mask] = float(shell_offset_max_n)
                            offs[:, 2] = torch.maximum(torch.minimum(offs[:, 2], n_cap_fixed), -n_cap_fixed)
                        g.bind_face_offset_local[outer_idx] = offs
            elif (not bool(outer_offset_unbounded)) and (not is_outer_geom_phase or outer_geom_phase_progress >= float(np.clip(float(outer_offset_cap_geom_start_frac), 0.0, 1.0))):
                g.bind_face_offset_local[:, 0].clamp_(min=-float(shell_offset_max_t), max=float(shell_offset_max_t))
                g.bind_face_offset_local[:, 1].clamp_(min=-float(shell_offset_max_t), max=float(shell_offset_max_t))
                g.bind_face_offset_local[:, 2].clamp_(min=-float(shell_offset_max_n), max=float(shell_offset_max_n))
            if dq_enable and dq_fail_active and bool(shell_enable):
                outer_active = (g.shell_layer == 1) & (g.prune_state != 2)
                if bool(outer_active.any()):
                    cap = float(np.clip(dq_outer_alpha_cap_if_fail, 1e-3, 0.999))
                    cap_logit = float(np.log(cap / (1.0 - cap)))
                    g.opacity_logit[outer_active] = g.opacity_logit[outer_active].clamp(max=cap_logit)
            if bool(shell_enable):
                outside_support_th = float(np.clip(float(outer_outside_support_th), 0.0, 1.0))
                outer_outside_active = (
                    (g.shell_layer == 1)
                    & (g.prune_state != 2)
                    & (outer_inside_support_mean.detach() < outside_support_th)
                )
                if bool(outer_outside_active.any()):
                    cap = float(np.clip(float(outer_outside_opacity_cap), 1e-4, 0.999))
                    cap_logit = float(np.log(cap / (1.0 - cap)))
                    g.opacity_logit[outer_outside_active] = g.opacity_logit[outer_outside_active].clamp(max=cap_logit)
                    if is_outer_geom_phase:
                        decay = float(np.clip(float(outer_outside_offset_decay), 0.0, 1.0))
                        g.bind_face_offset_local[outer_outside_active] = g.bind_face_offset_local[outer_outside_active] * decay
                if bool(outer_offset_suppress_low_support):
                    sup_th = float(np.clip(float(outer_offset_suppress_support_th), 0.0, 1.0))
                    low_sup_mask = (
                        (g.shell_layer == 1)
                        & (g.prune_state != 2)
                        & (outer_drive_support_mean.detach() < sup_th)
                    )
                    if is_outer_geom_phase and bool(low_sup_mask.any()):
                        decay = float(np.clip(float(outer_offset_suppress_decay), 0.0, 1.0))
                        cap = float(np.clip(float(outer_offset_suppress_opacity_cap), 1e-4, 0.999))
                        cap_logit = float(np.log(cap / (1.0 - cap)))
                        g.bind_face_offset_local[low_sup_mask] = g.bind_face_offset_local[low_sup_mask] * decay
                        g.opacity_logit[low_sup_mask] = g.opacity_logit[low_sup_mask].clamp(max=cap_logit)
            if bool((g.prune_state == 2).any()):
                g.opacity_logit[g.prune_state == 2] = g.opacity_logit[g.prune_state == 2].clamp(
                    max=float(outer_pruned_opacity_logit_max)
                )

        inner_active_iter = -1
        outer_active_iter = -1
        outer_pruned_iter = -1
        if bool(shell_enable):
            with torch.no_grad():
                outer_active_q = (g.shell_layer == 1) & (g.prune_state != 2)
                outer_total = int(outer_active_q.sum().item())
                hard_pruned_outer = int(((g.shell_layer == 1) & (g.prune_state == 2)).sum().item())
                inner_total = int(((g.shell_layer == 0) & (g.prune_state != 2)).sum().item())
                inner_active_iter = int(inner_total)
                outer_active_iter = int(outer_total)
                outer_pruned_iter = int(hard_pruned_outer)
                inside_support_th = float(np.clip(float(outer_inside_support_th), 0.0, 1.0))
                outside_support_th = float(np.clip(float(outer_outside_support_th), 0.0, 1.0))
                outside_contour_q = outer_active_q & (outer_inside_support_mean.detach() < outside_support_th)
                outside_person_q = outer_active_q & (outer_person_support_mean.detach() < outside_support_th)
                # Keep opacity as (N,) to avoid accidental NxN broadcasting in boolean ops.
                opacity_now = g.opacity().detach().view(-1)
                hand_leak_q = outer_active_q & (outer_hand_support_mean.detach() > 0.5)
                arm_nontrigger_q = hand_leak_q & (outer_inside_support_mean.detach() < inside_support_th)
                head_ref_q = outer_head_support_mean.detach() >= float(max(outer_head_support_th, inside_support_th))
                head_cov_num = float((outer_active_q & head_ref_q).sum().item())
                head_cov_den = float(max(int(head_ref_q.sum().item()), 1))
                head_detached_coverage = head_cov_num / head_cov_den
                if head_detached_coverage_baseline is None:
                    head_detached_coverage_baseline = float(head_detached_coverage)

                if outer_total > 0:
                    outside_contour_ratio = float(outside_contour_q.sum().item()) / float(outer_total)
                    outside_person_ratio = float(outside_person_q.sum().item()) / float(outer_total)
                    outside_alpha_count = int((opacity_now[outside_contour_q] > 0.05).sum().item())
                    outside_alpha_ratio = float(outside_alpha_count) / float(outer_total)
                    hand_leak_ratio = float(hand_leak_q.sum().item()) / float(outer_total)
                    arm_nontrigger_ratio = float(arm_nontrigger_q.sum().item()) / float(outer_total)
                else:
                    outside_contour_ratio = 0.0
                    outside_person_ratio = 0.0
                    outside_alpha_ratio = 0.0
                    hand_leak_ratio = 0.0
                    arm_nontrigger_ratio = 0.0

                nontrigger_person_q = outer_active_q & (outer_inside_support_mean.detach() < inside_support_th) & (outer_person_support_mean.detach() > 0.5)
                outside_anyframe_q = outer_active_q & (outer_person_support_mean.detach() < outside_support_th)
                last_quality_metrics = {
                    "iter": int(it + 1),
                    "active_outer_count": int(outer_total),
                    "hard_pruned_outer_count": int(hard_pruned_outer),
                    "outer_active_outside_contour_ratio": float(outside_contour_ratio),
                    "outer_active_outside_person_ratio": float(outside_person_ratio),
                    "outer_active_outside_alpha_gt_0.05_ratio": float(outside_alpha_ratio),
                    "hand_leakage_ratio": float(hand_leak_ratio),
                    "head_detached_coverage": float(head_detached_coverage),
                    "outer_active_on_nontrigger_person_ratio": float(nontrigger_person_q.sum().item()) / float(max(outer_total, 1)),
                    "outer_projection_outside_anyframe_ratio": float(outside_anyframe_q.sum().item()) / float(max(outer_total, 1)),
                    "outer_arm_nontrigger_activation_ratio": float(arm_nontrigger_ratio),
                    "outer_head_ring_coverage": float(head_detached_coverage),
                    "outer_boundary_hard_clamp_count": int(outer_boundary_hard_clamp_count),
                }
                if ((it + 1) % quality_log_every) == 0 or (it == 0) or ((it + 1) == int(iters)):
                    write_stage2_quality_log(quality_log_path, last_quality_metrics)
                if ((it + 1) % progress_log_every) == 0 or (it == 0):
                    pbar.set_postfix(
                        phase=phase_label,
                        inner=inner_active_iter,
                        outer=outer_active_iter,
                        outer_pruned=outer_pruned_iter,
                    )
                    print(
                        f"[Iter {it + 1}/{iters}] phase={phase_label} "
                        f"inner_active={inner_active_iter} outer_active={outer_active_iter} "
                        f"outer_pruned={outer_pruned_iter}"
                    )
                if outer_zero_stop_enable:
                    phase_ok = (not bool(outer_zero_stop_after_phase_inner)) or (phase_tag != "phase_inner")
                    if phase_ok and (it + 1) >= int(outer_zero_stop_warmup_iters):
                        if outer_active_iter <= 0:
                            outer_zero_consecutive += 1
                        else:
                            outer_zero_consecutive = 0
                        if outer_zero_consecutive >= int(outer_zero_stop_patience):
                            print(
                                f"Early stop at iter {it + 1}/{iters}: "
                                f"outer_active_count stayed 0 for {outer_zero_consecutive} consecutive iterations."
                            )
                            break
                    else:
                        outer_zero_consecutive = 0

        if (
            (not inner_ckpt_saved)
            and save_inner_checkpoint_path is not None
            and phase_inner_end > 0
            and (it + 1) >= int(phase_inner_end)
            and training_mode in ("full", "inner_only")
        ):
            save_inner_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _build_checkpoint_payload(quality_pass_now=False),
                save_inner_checkpoint_path,
            )
            inner_ckpt_saved = True
            print(f"Saved inner checkpoint: {save_inner_checkpoint_path}")

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

    quality_eval = evaluate_stage2_quality(
        shell_enable=bool(shell_enable),
        quality_thresholds=quality_thresholds,
        head_detached_coverage_baseline=head_detached_coverage_baseline,
        last_quality_metrics=last_quality_metrics,
    )
    quality_pass = bool(quality_eval.quality_pass)
    write_stage2_json_report(quality_report_path, quality_eval.report)

    if (
        (not inner_ckpt_saved)
        and save_inner_checkpoint_path is not None
        and training_mode in ("full", "inner_only")
    ):
        save_inner_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_checkpoint_payload(quality_pass_now=bool(quality_pass)),
            save_inner_checkpoint_path,
        )
        inner_ckpt_saved = True
        print(f"Saved inner checkpoint: {save_inner_checkpoint_path}")

    outer_face_records = build_outer_face_records(
        outer_face_weight_sum=outer_face_weight_sum,
        outer_face_gap_half_weighted_sum=outer_face_gap_half_weighted_sum,
        outer_face_gap_scale_weighted_sum=outer_face_gap_scale_weighted_sum,
        outer_face_first_hit_frame_idx=outer_face_first_hit_frame_idx,
        outer_face_contrib_frames=outer_face_contrib_frames,
        outer_face_spawned_mask=outer_face_spawned_mask,
        outer_face_spawn_iter=outer_face_spawn_iter,
        outer_face_spawn_frame_idx=outer_face_spawn_frame_idx,
        frames_list=[str(fn) for fn in frames_list],
    )
    outer_spawn_report = build_outer_spawn_report(
        frames_list=[str(fn) for fn in frames_list],
        outer_spawn_stats_by_frame=outer_spawn_stats_by_frame,
        outer_spawn_iter_records=outer_spawn_iter_records,
        outer_face_records=outer_face_records,
    )
    write_stage2_json_report(outer_spawn_report_path, outer_spawn_report)

    # Save checkpoint
    out_ckpt = out_ckpt_path
    torch.save(_build_checkpoint_payload(quality_pass_now=bool(quality_pass)), out_ckpt)











