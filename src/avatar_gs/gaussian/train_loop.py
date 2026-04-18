from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avatar_gs.gaussian.structured_gaussian import Gaussian3D, sanitize_inner_surface_binding_
from avatar_gs.gaussian.effective_face import compute_effective_face_indices_torch
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
from avatar_gs.gaussian.binding_layers.binding_update import quat_from_rotmat_torch
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
from avatar_gs.utils.progress import tqdm
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
from avatar_gs.segmentation.smpl_face_semantics import (
    rasterize_visible_face_index_map as _rasterize_visible_face_index_map,
)
from avatar_gs.stage2 import (
    Stage2FaceSemantics,
    Stage2FrameAssets,
    Stage2MeshRuntime,
    Stage2TrainLoopConfig,
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
    load_stage2_face_semantics,
    prepare_stage2_report_paths,
    project_points_numpy as _project_points_numpy,
    project_points_torch as _project_points_torch,
    rasterize_mesh_mask as _rasterize_mesh_mask,
    write_stage2_json_report,
    write_stage2_quality_log,
)

def _log_info(logger, msg: str, *args) -> None:
    if logger is not None:
        logger.info(msg, *args)
        return
    print(msg % args if args else msg)


def _log_warning(logger, msg: str, *args) -> None:
    if logger is not None:
        logger.warning(msg, *args)
        return
    print(msg % args if args else msg)


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


def _compute_inner_local_color_prior(
    *,
    rgb: torch.Tensor,
    shell_layer: torch.Tensor | None,
    prune_state: torch.Tensor | None,
    effective_face_idx: torch.Tensor | None,
    color_conf: torch.Tensor,
    vis_mean: torch.Tensor,
    support_mean: torch.Tensor,
    opacity: torch.Tensor,
    face_raw_semantic: torch.Tensor,
    face_semantic_conf: torch.Tensor,
    neighbor_idx: torch.Tensor | None,
    neighbor_w: torch.Tensor | None,
    target_conf_max: float,
    target_vis_max: float,
    target_support_max: float,
    donor_conf_min: float,
    donor_opacity_min: float,
    donor_vis_min: float,
    semantic_conf_min: float,
    face_reserve_min_active_per_face: float,
    min_donor_weight: float,
    ignore_raw_semantic_ids: torch.Tensor | None = None,
    rgb_logit_param: torch.Tensor | None = None,
    repair_blend: float = 0.0,
    repair_min_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = rgb.new_tensor(0.0)
    stats: dict[str, float] = {
        "targets": 0.0,
        "assisted": 0.0,
        "donors": 0.0,
        "repaired": 0.0,
        "mean_target_conf": 0.0,
    }
    if (
        shell_layer is None
        or prune_state is None
        or effective_face_idx is None
        or int(face_raw_semantic.numel()) <= 0
    ):
        return zero, stats

    n_faces = int(face_raw_semantic.shape[0])
    face_idx = effective_face_idx.to(device=rgb.device, dtype=torch.long).view(-1)
    face_valid = (face_idx >= 0) & (face_idx < n_faces)
    if not bool(face_valid.any()):
        return zero, stats

    face_raw = torch.full((int(face_idx.shape[0]),), -1, dtype=torch.long, device=rgb.device)
    face_conf = torch.zeros((int(face_idx.shape[0]),), dtype=torch.float32, device=rgb.device)
    face_ids_valid = face_idx[face_valid]
    face_raw[face_valid] = face_raw_semantic.index_select(0, face_ids_valid)
    face_conf[face_valid] = face_semantic_conf.index_select(0, face_ids_valid)
    ignore_semantic = torch.zeros_like(face_raw, dtype=torch.bool)
    if ignore_raw_semantic_ids is not None and int(ignore_raw_semantic_ids.numel()) > 0:
        ignore_ids = ignore_raw_semantic_ids.to(device=rgb.device, dtype=torch.long).view(-1)
        ignore_semantic = torch.isin(face_raw, ignore_ids)

    semantic_valid = face_valid & (face_raw >= 0) & (face_conf >= float(semantic_conf_min))
    active_inner = (shell_layer == 0) & (prune_state != 2) & semantic_valid
    if not bool(active_inner.any()):
        return zero, stats

    color_conf_det = color_conf.detach().clamp(0.0, 1.0).view(-1)
    vis_det = vis_mean.detach().clamp(0.0, 1.0).view(-1)
    support_det = support_mean.detach().clamp(0.0, 1.0).view(-1)
    opacity_det = opacity.detach().clamp(0.0, 1.0).view(-1)

    donor_mask = (
        active_inner
        & (~ignore_semantic)
        & (color_conf_det >= float(donor_conf_min))
        & (vis_det >= float(donor_vis_min))
        & (opacity_det >= float(donor_opacity_min))
    )
    weak_conf = color_conf_det < float(target_conf_max)
    weak_vis = vis_det < float(target_vis_max)
    weak_support = support_det < float(target_support_max)
    target_mask = active_inner & (~ignore_semantic) & (weak_conf | weak_vis | weak_support)
    stats["targets"] = float(target_mask.sum().item())
    stats["donors"] = float(donor_mask.sum().item())
    if not bool(donor_mask.any()) or not bool(target_mask.any()):
        return zero, stats

    donor_face_ids = face_idx[donor_mask]
    donor_w = (
        color_conf_det[donor_mask]
        * vis_det[donor_mask]
        * opacity_det[donor_mask]
    ).clamp_min(1e-4)
    donor_rgb = rgb.detach()[donor_mask]
    face_rgb_sum = torch.zeros((n_faces, 3), dtype=rgb.dtype, device=rgb.device)
    face_w_sum = torch.zeros((n_faces,), dtype=torch.float32, device=rgb.device)
    face_rgb_sum.index_add_(0, donor_face_ids, donor_rgb * donor_w.unsqueeze(1))
    face_w_sum.index_add_(0, donor_face_ids, donor_w)

    face_total_rgb = face_rgb_sum.clone()
    face_total_w = face_w_sum.clone()
    if (
        neighbor_idx is not None
        and neighbor_w is not None
        and int(neighbor_idx.numel()) > 0
        and int(neighbor_idx.shape[0]) == n_faces
    ):
        safe_neigh = neighbor_idx.clamp_min(0)
        neigh_face_w = face_w_sum.index_select(0, safe_neigh.reshape(-1)).reshape_as(neighbor_idx)
        neigh_face_rgb = face_rgb_sum.index_select(0, safe_neigh.reshape(-1)).reshape(n_faces, -1, 3)
        neigh_sem = face_raw_semantic.index_select(0, safe_neigh.reshape(-1)).reshape_as(neighbor_idx)
        neigh_conf = face_semantic_conf.index_select(0, safe_neigh.reshape(-1)).reshape_as(neighbor_idx)
        center_sem = face_raw_semantic.view(-1, 1)
        valid_neigh = (
            (neighbor_idx >= 0)
            & (neighbor_w > 0.0)
            & (center_sem >= 0)
            & (neigh_sem == center_sem)
            & (neigh_conf >= float(semantic_conf_min))
        )
        if bool(valid_neigh.any()):
            neigh_weight = neighbor_w * valid_neigh.float() * neigh_face_w
            face_total_w = face_total_w + neigh_weight.sum(dim=1)
            face_total_rgb = face_total_rgb + (neigh_face_rgb * neigh_weight.unsqueeze(-1)).sum(dim=1)

    target_idx = torch.nonzero(target_mask, as_tuple=False).view(-1)
    target_face_ids = face_idx.index_select(0, target_idx)
    target_face_total_w = face_total_w.index_select(0, target_face_ids)
    assisted_mask = target_face_total_w >= float(max(min_donor_weight, 1e-6))
    stats["assisted"] = float(assisted_mask.sum().item())
    if not bool(assisted_mask.any()):
        return zero, stats

    target_idx = target_idx[assisted_mask]
    target_face_ids = target_face_ids[assisted_mask]
    target_face_total_w = target_face_total_w[assisted_mask].clamp_min(1e-6)
    prior_rgb = face_total_rgb.index_select(0, target_face_ids) / target_face_total_w.unsqueeze(1)
    target_conf = color_conf_det.index_select(0, target_idx)
    target_vis = vis_det.index_select(0, target_idx)
    target_support = support_det.index_select(0, target_idx)
    conf_need = ((float(target_conf_max) - target_conf) / max(float(target_conf_max), 1e-6)).clamp(0.0, 1.0)
    vis_need = ((float(target_vis_max) - target_vis) / max(float(target_vis_max), 1e-6)).clamp(0.0, 1.0)
    support_need = ((float(target_support_max) - target_support) / max(float(target_support_max), 1e-6)).clamp(0.0, 1.0)
    target_need = torch.maximum(conf_need, torch.maximum(vis_need, support_need))
    target_weight = target_need * face_conf.index_select(0, target_idx).clamp(0.0, 1.0)
    if not bool((target_weight > 0.0).any()):
        return zero, stats

    stats["mean_target_conf"] = float(target_conf.mean().item())
    color_err = (rgb.index_select(0, target_idx) - prior_rgb).abs().mean(dim=1)
    if rgb_logit_param is not None and float(repair_blend) > 0.0:
        repair_mask = target_weight >= float(max(repair_min_weight, 0.0))
        if bool(repair_mask.any()):
            repair_idx = target_idx[repair_mask]
            repair_prior = prior_rgb[repair_mask]
            repair_rgb = rgb.index_select(0, repair_idx).detach()
            repair_alpha = (
                float(repair_blend) * target_weight[repair_mask].detach().clamp(0.0, 1.0)
            ).clamp(0.0, 0.95).unsqueeze(1)
            repair_rgb = ((1.0 - repair_alpha) * repair_rgb + repair_alpha * repair_prior.detach()).clamp(1e-4, 1.0 - 1e-4)
            with torch.no_grad():
                rgb_logit_param[repair_idx] = torch.log(repair_rgb / (1.0 - repair_rgb))
            stats["repaired"] = float(repair_mask.sum().item())
    loss = (color_err * target_weight).sum() / (target_weight.sum() + 1e-6)
    return loss, stats


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


def _sharpen_rgb_chw(rgb_chw: torch.Tensor, amount: float, *, kernel_size: int = 5) -> torch.Tensor:
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3 or float(amount) <= 0.0:
        return rgb_chw
    k = int(max(int(kernel_size), 3))
    if (k % 2) == 0:
        k += 1
    pad = k // 2
    blur = F.avg_pool2d(rgb_chw.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)
    return (rgb_chw + float(amount) * (rgb_chw - blur)).clamp(0.0, 1.0)


def _masked_multiscale_l1_chw(
    pred_rgb_chw: torch.Tensor,
    gt_rgb_chw: torch.Tensor,
    mask_chw: torch.Tensor | None,
    *,
    levels: int = 3,
) -> torch.Tensor:
    if pred_rgb_chw.ndim != 3 or pred_rgb_chw.shape[0] != 3:
        raise ValueError(f"pred_rgb_chw must be (3,H,W), got {pred_rgb_chw.shape}")
    if gt_rgb_chw.shape != pred_rgb_chw.shape:
        raise ValueError(f"gt_rgb_chw shape mismatch: {gt_rgb_chw.shape} vs {pred_rgb_chw.shape}")
    levels = max(int(levels), 1)
    pred = pred_rgb_chw.unsqueeze(0)
    gt = gt_rgb_chw.unsqueeze(0)
    mask = None
    if mask_chw is not None:
        if mask_chw.ndim != 3 or mask_chw.shape[0] != 1:
            raise ValueError(f"mask_chw must be (1,H,W), got {mask_chw.shape}")
        mask = mask_chw.unsqueeze(0).clamp(0.0, 1.0)
    total = pred_rgb_chw.new_tensor(0.0)
    weight_sum = 0.0
    scale_w = 1.0
    for level in range(levels):
        err = (pred - gt).abs()
        if mask is not None:
            mask3 = mask.expand(-1, err.shape[1], -1, -1)
            total = total + scale_w * (err * mask3).sum() / (mask3.sum() + 1e-6)
        else:
            total = total + scale_w * err.mean()
        weight_sum += scale_w
        if level + 1 >= levels or min(int(pred.shape[-2]), int(pred.shape[-1])) < 4:
            break
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        gt = F.avg_pool2d(gt, kernel_size=2, stride=2)
        if mask is not None:
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2).clamp(0.0, 1.0)
        scale_w *= 0.5
    return total / max(weight_sum, 1e-6)


def _crop_intrinsics_to_box(
    K: torch.Tensor,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    crop_w = max(int(x1) - int(x0) + 1, 1)
    crop_h = max(int(y1) - int(y0) + 1, 1)
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)
    Kc = K.clone()
    Kc[0, 0] = Kc[0, 0] * sx
    Kc[1, 1] = Kc[1, 1] * sy
    Kc[0, 2] = (Kc[0, 2] - float(x0)) * sx
    Kc[1, 2] = (Kc[1, 2] - float(y0)) * sy
    return Kc


def _render_box_resized(
    *,
    g: Gaussian3D,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    out_h: int,
    out_w: int,
    bg_rgb: tuple[float, float, float] | torch.Tensor,
    gaussian_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    K_crop = _crop_intrinsics_to_box(K=K, x0=x0, y0=y0, x1=x1, y1=y1, out_h=out_h, out_w=out_w)
    rgb_crop, alpha_crop = render_gaussians_3d(
        g,
        K_crop,
        R,
        t,
        int(out_h),
        int(out_w),
        bg=bg_rgb,
        return_alpha=True,
        gaussian_weights=gaussian_weights,
    )
    return rgb_crop, alpha_crop


def _build_face_feature_mask_from_points(
    *,
    patch_h: int,
    patch_w: int,
    box_xyxy: tuple[int, int, int, int],
    feature_points: dict[str, tuple[float, float]] | None,
    sigma_frac: float,
    device: torch.device,
) -> torch.Tensor | None:
    if feature_points is None or len(feature_points) == 0:
        return None
    x0, y0, x1, y1 = box_xyxy
    box_w = max(float(int(x1) - int(x0) + 1), 1.0)
    box_h = max(float(int(y1) - int(y0) + 1), 1.0)
    sigma_base = max(float(min(patch_h, patch_w)) * float(max(sigma_frac, 1e-3)), 2.5)
    yy = torch.arange(int(patch_h), device=device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(int(patch_w), device=device, dtype=torch.float32).view(1, -1)
    mask = torch.zeros((1, int(patch_h), int(patch_w)), dtype=torch.float32, device=device)

    def _accumulate_point(px_src: float, py_src: float, sigma_mul: float, amp: float) -> None:
        px = (float(px_src) - float(x0)) / box_w * float(max(int(patch_w) - 1, 1))
        py = (float(py_src) - float(y0)) / box_h * float(max(int(patch_h) - 1, 1))
        sigma = max(float(sigma_base) * float(sigma_mul), 1.5)
        gauss = torch.exp(-0.5 * (((xx - px) / sigma).pow(2) + ((yy - py) / sigma).pow(2)))
        mask[0].copy_(torch.maximum(mask[0], gauss * float(amp)))

    for name, (px, py) in feature_points.items():
        sigma_mul = 1.0
        amp = 1.0
        if 'eye' in name:
            sigma_mul = 0.85
            amp = 1.20
        elif 'nose' in name:
            sigma_mul = 0.75
            amp = 1.35
        elif 'mouth' in name:
            sigma_mul = 0.95
            amp = 1.15
        _accumulate_point(float(px), float(py), sigma_mul, amp)
    return mask.clamp(0.0, 1.0)


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
    uu = torch.round(u[valid]).long().clamp(0, int(W) - 1)
    vv = torch.round(v[valid]).long().clamp(0, int(H) - 1)
    out[valid] = mask[0, vv, uu]
    return out.clamp(0.0, 1.0)


def _sample_rgb_at_projected_uv(
    image: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be (3,H,W), got {image.shape}")
    out = torch.zeros((int(u.shape[0]), 3), dtype=image.dtype, device=image.device)
    if bool(valid.any()):
        uu = torch.round(u[valid]).long().clamp(0, int(W) - 1)
        vv = torch.round(v[valid]).long().clamp(0, int(H) - 1)
        out[valid] = image[:, vv, uu].permute(1, 0)
    return out


def _compute_face_feature_weight_from_projected_uv(
    u: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    *,
    box_xyxy: tuple[int, int, int, int],
    feature_points: dict[str, tuple[float, float]] | None,
    sigma_frac: float,
) -> torch.Tensor:
    out = torch.zeros_like(u, dtype=torch.float32)
    if (not feature_points) or (not bool(valid.any())):
        return out
    x0, y0, x1, y1 = box_xyxy
    side = max(float(max(int(x1) - int(x0) + 1, int(y1) - int(y0) + 1)), 1.0)
    sigma_base = max(side * float(max(sigma_frac, 1e-3)), 1.5)
    uv_u = u[valid]
    uv_v = v[valid]
    w_acc = torch.zeros_like(uv_u, dtype=torch.float32)
    for name, (px, py) in feature_points.items():
        sigma_mul = 1.0
        amp = 1.0
        if 'eye' in name:
            sigma_mul = 0.80
            amp = 1.25
        elif 'nose' in name:
            sigma_mul = 0.72
            amp = 1.40
        elif 'mouth' in name:
            sigma_mul = 0.92
            amp = 1.20
        sigma = max(float(sigma_base) * float(sigma_mul), 1.25)
        dist2 = (uv_u - float(px)).pow(2) + (uv_v - float(py)).pow(2)
        w = torch.exp(-0.5 * dist2 / float(sigma * sigma)) * float(amp)
        w_acc = torch.maximum(w_acc, w)
    out[valid] = w_acc.clamp(0.0, 1.0)
    return out


def _apply_face_texture_bake(
    *,
    g: Gaussian3D,
    xyz_render: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    gt_rgb: torch.Tensor,
    visibility: torch.Tensor,
    face_box_xyxy: tuple[int, int, int, int],
    feature_points: dict[str, tuple[float, float]] | None,
    face_gaussian_mask: torch.Tensor,
    view_score: float,
    base_blend: float,
    max_blend: float,
    visibility_min: float,
    feature_boost: float,
    sharpen_amount: float,
    feature_sigma_frac: float,
) -> int:
    if not bool(face_gaussian_mask.any()):
        return 0
    H = int(gt_rgb.shape[-2])
    W = int(gt_rgb.shape[-1])
    u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K, R, t)
    x0, y0, x1, y1 = face_box_xyxy
    face_proj = (
        valid_proj
        & (u_proj >= float(x0))
        & (u_proj <= float(x1))
        & (v_proj >= float(y0))
        & (v_proj <= float(y1))
        & (visibility >= float(visibility_min))
        & face_gaussian_mask
    )
    if not bool(face_proj.any()):
        return 0
    sampled_rgb = _sample_rgb_at_projected_uv(gt_rgb, u_proj, v_proj, face_proj, H, W).clamp(0.0, 1.0)
    feature_weight = _compute_face_feature_weight_from_projected_uv(
        u_proj,
        v_proj,
        face_proj,
        box_xyxy=face_box_xyxy,
        feature_points=feature_points,
        sigma_frac=float(feature_sigma_frac),
    )
    target_rgb = sampled_rgb
    if float(sharpen_amount) > 0.0:
        blur = F.avg_pool2d(gt_rgb.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        sharp_rgb = (gt_rgb + float(sharpen_amount) * (gt_rgb - blur)).clamp(0.0, 1.0)
        sampled_sharp = _sample_rgb_at_projected_uv(sharp_rgb, u_proj, v_proj, face_proj, H, W).clamp(0.0, 1.0)
        feat_mix = feature_weight[face_proj].unsqueeze(1).clamp(0.0, 1.0)
        target_rgb = sampled_rgb * (1.0 - feat_mix) + sampled_sharp * feat_mix
    curr_rgb = torch.sigmoid(g.rgb_logit.data[face_proj]).clamp(1e-4, 1.0 - 1e-4)
    vis_proj = visibility[face_proj].detach().clamp(0.0, 1.0)
    vis_gate = 0.35 + 0.65 * torch.sqrt(vis_proj.clamp(min=float(visibility_min), max=1.0))
    blend = float(base_blend) * float(max(view_score, 0.0)) * vis_gate
    feat_proj = feature_weight[face_proj].clamp(0.0, 1.0)
    if float(feature_boost) > 0.0:
        # Stop spending most updates on cheek/forehead skin; push color budget into eyes/nose/mouth.
        blend = blend * (0.12 + 0.88 * feat_proj) * (1.0 + float(feature_boost) * feat_proj)
    else:
        blend = blend * (0.35 + 0.65 * feat_proj)
    blend = blend.clamp(0.0, float(max_blend))

    face_ids_all = None
    if g.effective_face_idx is not None and int(g.effective_face_idx.numel()) == int(face_gaussian_mask.numel()):
        face_ids_all = g.effective_face_idx.to(device=g.rgb_logit.device, dtype=torch.long).view(-1)
    if face_ids_all is not None:
        proj_face_ids = face_ids_all[face_proj]
        valid_face_ids = proj_face_ids >= 0
        if bool(valid_face_ids.any()):
            tgt_valid = target_rgb[valid_face_ids]
            blend_valid = blend[valid_face_ids]
            curr_valid = curr_rgb[valid_face_ids]
            face_ids_valid = proj_face_ids[valid_face_ids]
            n_faces = int(face_ids_valid.max().item()) + 1
            face_rgb_sum = torch.zeros((n_faces, 3), dtype=tgt_valid.dtype, device=tgt_valid.device)
            face_w_sum = torch.zeros((n_faces,), dtype=blend_valid.dtype, device=blend_valid.device)
            face_cnt = torch.zeros((n_faces,), dtype=blend_valid.dtype, device=blend_valid.device)
            w = blend_valid.clamp_min(1e-4)
            face_rgb_sum.index_add_(0, face_ids_valid, tgt_valid * w.unsqueeze(1))
            face_w_sum.index_add_(0, face_ids_valid, w)
            face_cnt.index_add_(0, face_ids_valid, torch.ones_like(w))
            face_target = face_rgb_sum.index_select(0, face_ids_valid) / face_w_sum.index_select(0, face_ids_valid).unsqueeze(1).clamp_min(1e-6)
            face_blend = (face_w_sum.index_select(0, face_ids_valid) / face_cnt.index_select(0, face_ids_valid).clamp_min(1.0)).clamp(0.0, float(max_blend))
            new_valid = curr_valid * (1.0 - face_blend.unsqueeze(1)) + face_target * face_blend.unsqueeze(1)
            new_valid = new_valid.clamp(1e-4, 1.0 - 1e-4)
            curr_rgb = curr_rgb.clone()
            curr_rgb[valid_face_ids] = new_valid
    else:
        new_rgb = curr_rgb * (1.0 - blend.unsqueeze(1)) + target_rgb * blend.unsqueeze(1)
        curr_rgb = new_rgb.clamp(1e-4, 1.0 - 1e-4)

    g.rgb_logit.data[face_proj] = torch.log(curr_rgb / (1.0 - curr_rgb))
    return int(face_proj.sum().item())



def _compute_face_surface_color_term(
    *,
    g: Gaussian3D,
    xyz_render: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    gt_rgb: torch.Tensor,
    visibility: torch.Tensor,
    face_box_xyxy: tuple[int, int, int, int],
    feature_points: dict[str, tuple[float, float]] | None,
    face_gaussian_mask: torch.Tensor,
    visibility_min: float,
    feature_sigma_frac: float,
    sharpen_amount: float,
) -> torch.Tensor:
    zero = g.rgb_logit.sum() * 0.0
    if (not bool(face_gaussian_mask.any())) or g.effective_face_idx is None:
        return zero
    H = int(gt_rgb.shape[-2])
    W = int(gt_rgb.shape[-1])
    u_proj, v_proj, valid_proj = _project_points_torch(xyz_render, K, R, t)
    x0, y0, x1, y1 = face_box_xyxy
    face_proj = (
        valid_proj
        & (u_proj >= float(x0))
        & (u_proj <= float(x1))
        & (v_proj >= float(y0))
        & (v_proj <= float(y1))
        & (visibility >= float(visibility_min))
        & face_gaussian_mask
    )
    if not bool(face_proj.any()):
        return zero

    sampled_rgb_all = _sample_rgb_at_projected_uv(gt_rgb, u_proj, v_proj, face_proj, H, W).clamp(0.0, 1.0)
    sampled_rgb = sampled_rgb_all[face_proj]
    feature_weight_all = _compute_face_feature_weight_from_projected_uv(
        u_proj,
        v_proj,
        face_proj,
        box_xyxy=face_box_xyxy,
        feature_points=feature_points,
        sigma_frac=float(feature_sigma_frac),
    )
    feature_weight = feature_weight_all[face_proj]
    target_rgb = sampled_rgb
    if float(sharpen_amount) > 0.0:
        blur = F.avg_pool2d(gt_rgb.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        sharp_rgb = (gt_rgb + float(sharpen_amount) * (gt_rgb - blur)).clamp(0.0, 1.0)
        sampled_sharp_all = _sample_rgb_at_projected_uv(sharp_rgb, u_proj, v_proj, face_proj, H, W).clamp(0.0, 1.0)
        sampled_sharp = sampled_sharp_all[face_proj]
        feat_mix = feature_weight.unsqueeze(1).clamp(0.0, 1.0)
        target_rgb = sampled_rgb * (1.0 - feat_mix) + sampled_sharp * feat_mix

    current_valid = torch.sigmoid(g.rgb_logit[face_proj]).clamp(1e-4, 1.0 - 1e-4)
    vis_valid = visibility[face_proj].detach().clamp(0.0, 1.0)
    feat_valid = feature_weight.clamp(0.0, 1.0)
    sample_weight = (0.10 + 0.90 * feat_valid) * (0.35 + 0.65 * torch.sqrt(vis_valid.clamp(min=float(visibility_min), max=1.0)))
    sample_weight = sample_weight.clamp_min(1e-4)

    err = (current_valid - target_rgb).abs().mean(dim=1)
    return (err * sample_weight).sum() / (sample_weight.sum() + 1e-6)



def _canonical_face_anchor_triplet(
    feature_points: dict[str, tuple[float, float]] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not feature_points:
        return None
    required = ("left_eye", "right_eye", "mouth_center")
    if any(name not in feature_points for name in required):
        return None
    return torch.tensor(
        [
            [float(feature_points["left_eye"][0]), float(feature_points["left_eye"][1])],
            [float(feature_points["right_eye"][0]), float(feature_points["right_eye"][1])],
            [float(feature_points["mouth_center"][0]), float(feature_points["mouth_center"][1])],
        ],
        device=device,
        dtype=dtype,
    )


def _canonical_face_dst_triplet(*, out_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    s = float(max(int(out_size) - 1, 1))
    return torch.tensor(
        [
            [0.36 * s, 0.38 * s],
            [0.64 * s, 0.38 * s],
            [0.50 * s, 0.72 * s],
        ],
        device=device,
        dtype=dtype,
    )


def _warp_face_patch_to_canonical(
    image_chw: torch.Tensor,
    *,
    box_xyxy: tuple[int, int, int, int],
    feature_points: dict[str, tuple[float, float]] | None,
    out_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if image_chw.ndim != 3 or int(image_chw.shape[0]) != 3:
        return None, None
    patch_h = int(image_chw.shape[-2])
    patch_w = int(image_chw.shape[-1])
    x0, y0, x1, y1 = box_xyxy
    src_pts_img = _canonical_face_anchor_triplet(feature_points, device=image_chw.device, dtype=image_chw.dtype)
    if src_pts_img is None:
        return None, None
    box_w = max(float(int(x1) - int(x0) + 1), 1.0)
    box_h = max(float(int(y1) - int(y0) + 1), 1.0)
    src_pts = src_pts_img.clone()
    src_pts[:, 0] = (src_pts[:, 0] - float(x0)) / box_w * float(max(patch_w - 1, 1))
    src_pts[:, 1] = (src_pts[:, 1] - float(y0)) / box_h * float(max(patch_h - 1, 1))
    dst_pts = _canonical_face_dst_triplet(out_size=int(out_size), device=image_chw.device, dtype=image_chw.dtype)
    X = torch.cat([src_pts, torch.ones((3, 1), device=image_chw.device, dtype=image_chw.dtype)], dim=1)
    try:
        A_t = torch.linalg.lstsq(X, dst_pts).solution
    except RuntimeError:
        return None, None
    A = A_t.transpose(0, 1)
    A_aug = torch.eye(3, device=image_chw.device, dtype=image_chw.dtype)
    A_aug[:2, :] = A
    try:
        A_inv = torch.linalg.inv(A_aug)
    except RuntimeError:
        return None, None
    in_wf = float(max(patch_w - 1, 1))
    in_hf = float(max(patch_h - 1, 1))
    out_wf = float(max(int(out_size) - 1, 1))
    out_hf = float(max(int(out_size) - 1, 1))
    T_in = torch.tensor([[in_wf / 2.0, 0.0, in_wf / 2.0], [0.0, in_hf / 2.0, in_hf / 2.0], [0.0, 0.0, 1.0]], device=image_chw.device, dtype=image_chw.dtype)
    T_out = torch.tensor([[out_wf / 2.0, 0.0, out_wf / 2.0], [0.0, out_hf / 2.0, out_hf / 2.0], [0.0, 0.0, 1.0]], device=image_chw.device, dtype=image_chw.dtype)
    T_in_inv = torch.linalg.inv(T_in)
    theta = (T_in_inv @ A_inv @ T_out)[:2, :]
    grid = F.affine_grid(theta.unsqueeze(0), size=(1, 3, int(out_size), int(out_size)), align_corners=True)
    warped = F.grid_sample(image_chw.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
    ones = torch.ones((1, 1, patch_h, patch_w), device=image_chw.device, dtype=image_chw.dtype)
    valid = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).clamp(0.0, 1.0)
    return warped, valid


def _build_canonical_face_template(
    *,
    gts: list[torch.Tensor],
    face_boxes: list[tuple[float, float, float, float] | None] | None,
    face_feature_points_frames: list[dict[str, tuple[float, float]] | None] | None,
    teacher_ids: np.ndarray | None,
    teacher_weights: np.ndarray | None,
    out_size: int,
    device: torch.device,
    teacher_weight_power: float = 1.0,
    max_teachers: int = 0,
    sharpen_amount: float = 0.0,
    feature_sigma_frac: float = 0.045,
    symmetry_blend: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    if face_boxes is None or face_feature_points_frames is None or teacher_ids is None or int(teacher_ids.size) <= 0:
        return None, None, None, 0
    ordered_ids = [int(x) for x in teacher_ids.tolist() if 0 <= int(x) < len(gts)]
    if teacher_weights is not None and len(ordered_ids) > 1:
        ordered_ids.sort(key=lambda ii: float(teacher_weights[ii]), reverse=True)
    if int(max_teachers) > 0 and len(ordered_ids) > int(max_teachers):
        ordered_ids = ordered_ids[: int(max_teachers)]
    rgb_acc = None
    w_acc = None
    used = 0
    weight_power = float(max(float(teacher_weight_power), 0.0))
    for ii in ordered_ids:
        box = face_boxes[ii]
        feat = face_feature_points_frames[ii]
        if box is None or feat is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        crop = gts[ii][:, y0:y1 + 1, x0:x1 + 1]
        warped, valid = _warp_face_patch_to_canonical(crop, box_xyxy=(x0, y0, x1, y1), feature_points=feat, out_size=int(out_size))
        if warped is None or valid is None:
            continue
        base_w = 1.0 if teacher_weights is None else float(max(float(teacher_weights[ii]), 1e-6))
        w = base_w if weight_power == 1.0 else float(base_w ** weight_power)
        if rgb_acc is None:
            rgb_acc = warped * valid * float(w)
            w_acc = valid * float(w)
        else:
            rgb_acc = rgb_acc + warped * valid * float(w)
            w_acc = w_acc + valid * float(w)
        used += 1
    if rgb_acc is None or w_acc is None or used <= 0:
        return None, None, None, 0
    template = (rgb_acc / w_acc.clamp_min(1e-6)).clamp(0.0, 1.0)
    template_mask = (w_acc > 0.05).to(template.dtype)
    canonical_span = float(max(int(out_size) - 1, 1))
    canonical_points = {
        'left_eye': (0.36 * canonical_span, 0.38 * canonical_span),
        'right_eye': (0.64 * canonical_span, 0.38 * canonical_span),
        'nose': (0.50 * canonical_span, 0.55 * canonical_span),
        'mouth_left': (0.42 * canonical_span, 0.72 * canonical_span),
        'mouth_center': (0.50 * canonical_span, 0.72 * canonical_span),
        'mouth_right': (0.58 * canonical_span, 0.72 * canonical_span),
    }
    feature_mask = _build_face_feature_mask_from_points(
        patch_h=int(out_size),
        patch_w=int(out_size),
        box_xyxy=(0, 0, int(out_size) - 1, int(out_size) - 1),
        feature_points=canonical_points,
        sigma_frac=float(np.clip(float(feature_sigma_frac), 1e-3, 0.20)),
        device=device,
    )
    if feature_mask is not None:
        feature_mask = (feature_mask * template_mask).clamp(0.0, 1.0)
    if float(sharpen_amount) > 0.0:
        sharpen_mask = feature_mask if feature_mask is not None else template_mask
        sharpen_rgb = _sharpen_rgb_chw(template, float(sharpen_amount), kernel_size=5)
        sharpen_mask3 = sharpen_mask.expand_as(template)
        template = (template * (1.0 - sharpen_mask3) + sharpen_rgb * sharpen_mask3).clamp(0.0, 1.0)
    if float(symmetry_blend) > 0.0:
        sym_blend = float(np.clip(float(symmetry_blend), 0.0, 1.0))
        if sym_blend > 0.0:
            mirror_template = torch.flip(template, dims=[2])
            template = ((1.0 - sym_blend) * template + sym_blend * 0.5 * (template + mirror_template)).clamp(0.0, 1.0)
            if feature_mask is not None:
                mirror_feature = torch.flip(feature_mask, dims=[2])
                feature_mask = ((1.0 - sym_blend) * feature_mask + sym_blend * 0.5 * (feature_mask + mirror_feature)).clamp(0.0, 1.0)
    return template, template_mask, feature_mask, int(used)

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
        effective_face_idx=g.effective_face_idx,
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


def _aggregate_face_scalar_mean(
    face_ids: torch.Tensor,
    values: torch.Tensor,
    n_faces: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_sum = torch.zeros((int(n_faces),), dtype=values.dtype, device=values.device)
    out_cnt = torch.zeros((int(n_faces),), dtype=values.dtype, device=values.device)
    ones = torch.ones_like(values, dtype=values.dtype)
    out_sum.index_add_(0, face_ids, values)
    out_cnt.index_add_(0, face_ids, ones)
    out_mean = out_sum / out_cnt.clamp_min(1.0)
    return out_mean, out_cnt


def _select_inner_coverage_face_candidates(
    *,
    g: Gaussian3D,
    effective_face_idx: torch.Tensor | None,
    frame_data: BindingFrameData | None,
    vis_mean: torch.Tensor,
    inner_color_support_mean: torch.Tensor,
    residual_ema: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty_idx = torch.zeros((0,), dtype=torch.long, device=vis_mean.device)
    empty_score = torch.zeros((0,), dtype=torch.float32, device=vis_mean.device)
    if (
        effective_face_idx is None
        or frame_data is None
        or frame_data.face_s1_t is None
        or frame_data.face_s2_t is None
        or g.shell_layer is None
        or g.prune_state is None
        or int(topk) <= 0
    ):
        return empty_idx, empty_score

    n_faces = int(frame_data.face_s1_t.shape[0])
    eff_face = effective_face_idx.to(device=vis_mean.device, dtype=torch.long).view(-1)
    active_inner = (
        (g.shell_layer == 0)
        & (g.prune_state != 2)
        & (eff_face >= 0)
        & (eff_face < n_faces)
    )
    if not bool(active_inner.any()):
        return empty_idx, empty_score

    face_ids = eff_face[active_inner]
    scales_inner = g.scales()[active_inner]
    tangential = torch.sort(scales_inner[:, :2], dim=1, descending=True).values
    opacity_inner = g.opacity().view(-1)[active_inner].detach().clamp(0.0, 1.0)
    vis_inner = vis_mean[active_inner].detach().clamp(0.0, 1.0)
    support_inner = inner_color_support_mean[active_inner].detach().clamp(0.0, 1.0)
    residual_inner = residual_ema[active_inner].detach().clamp(min=0.0)

    face_s1 = frame_data.face_s1_t.index_select(0, face_ids).clamp(min=1e-6)
    face_s2 = frame_data.face_s2_t.index_select(0, face_ids).clamp(min=1e-6)
    face_area = (face_s1 * face_s2).clamp(min=1e-6)
    footprint = (tangential[:, 0] * tangential[:, 1]).clamp(min=1e-8)
    coverage_contrib = (
        (footprint / face_area)
        * (0.25 + 0.75 * vis_inner)
        * (0.25 + 0.75 * opacity_inner)
    ).clamp(min=0.0, max=4.0)

    face_coverage = torch.zeros((n_faces,), dtype=torch.float32, device=vis_mean.device)
    face_coverage.index_add_(0, face_ids, coverage_contrib.to(dtype=torch.float32))
    face_support, face_count = _aggregate_face_scalar_mean(
        face_ids,
        support_inner.to(dtype=torch.float32),
        n_faces,
    )
    face_residual, _ = _aggregate_face_scalar_mean(
        face_ids,
        residual_inner.to(dtype=torch.float32),
        n_faces,
    )

    low_coverage = F.relu(1.0 - face_coverage)
    face_score = low_coverage * (0.20 + 0.80 * face_support) * (0.15 + face_residual)
    eligible = (
        (face_count > 0.0)
        & (low_coverage > 0.02)
        & (face_score > 0.0)
    )
    face_idx = torch.nonzero(eligible, as_tuple=False).view(-1)
    if int(face_idx.numel()) <= 0:
        return empty_idx, empty_score
    face_score_sel = face_score.index_select(0, face_idx)
    if int(face_idx.numel()) > int(topk):
        top = torch.topk(face_score_sel, k=int(topk), largest=True).indices
        face_idx = face_idx.index_select(0, top)
        face_score_sel = face_score_sel.index_select(0, top)
    return face_idx, face_score_sel


def _select_face_reserve_candidates(
    *,
    g: Gaussian3D,
    effective_face_idx: torch.Tensor | None,
    frame_data: BindingFrameData | None,
    face_semantic_raw_t: torch.Tensor | None,
    face_semantic_conf_t: torch.Tensor | None,
    target_raw_ids: torch.Tensor | None,
    topk: int,
    semantic_conf_min: float,
    face_reserve_min_active_per_face: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = g.xyz.device
    empty_idx = torch.zeros((0,), dtype=torch.long, device=device)
    empty_score = torch.zeros((0,), dtype=torch.float32, device=device)
    if (
        effective_face_idx is None
        or frame_data is None
        or frame_data.face_s1_t is None
        or frame_data.face_s2_t is None
        or face_semantic_raw_t is None
        or target_raw_ids is None
        or g.shell_layer is None
        or g.prune_state is None
        or int(topk) <= 0
    ):
        return empty_idx, empty_score

    n_faces = int(frame_data.face_s1_t.shape[0])
    if int(n_faces) <= 0:
        return empty_idx, empty_score
    raw_sem = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
    if int(raw_sem.shape[0]) < n_faces:
        return empty_idx, empty_score
    raw_conf = (
        torch.ones((n_faces,), dtype=torch.float32, device=device)
        if face_semantic_conf_t is None
        else face_semantic_conf_t.to(device=device, dtype=torch.float32).view(-1)[:n_faces]
    )
    target_ids = target_raw_ids.to(device=device, dtype=torch.long).view(-1)
    semantic_face = torch.isin(raw_sem[:n_faces], target_ids) & (raw_conf >= float(semantic_conf_min))
    if not bool(semantic_face.any()):
        return empty_idx, empty_score

    face_ids_all = effective_face_idx.to(device=device, dtype=torch.long).view(-1)
    face_valid_all = (face_ids_all >= 0) & (face_ids_all < n_faces)
    active_inner = (g.shell_layer == 0) & (g.prune_state != 2) & face_valid_all
    face_active_counts = torch.zeros((n_faces,), dtype=torch.float32, device=device)
    if bool(active_inner.any()):
        active_face_ids = face_ids_all[active_inner]
        face_active_counts.index_add_(
            0,
            active_face_ids,
            torch.ones((int(active_face_ids.numel()),), dtype=torch.float32, device=device),
        )

    face_area = (frame_data.face_s1_t * frame_data.face_s2_t).to(device=device, dtype=torch.float32).clamp(min=1e-6)
    area_ref = torch.median(face_area[semantic_face]) if bool(semantic_face.any()) else torch.median(face_area)
    detail_boost = (area_ref / face_area).sqrt().clamp(min=0.7, max=2.5)
    reserve_gap = (float(face_reserve_min_active_per_face) - face_active_counts).clamp(min=0.0)
    face_score = semantic_face.to(dtype=torch.float32) * raw_conf * detail_boost * (1.0 + reserve_gap)
    candidate_mask = semantic_face & (reserve_gap > 0.0)
    face_idx = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    if int(face_idx.numel()) <= 0:
        return empty_idx, empty_score
    face_score_sel = face_score.index_select(0, face_idx)
    if int(face_idx.numel()) > int(topk):
        top = torch.topk(face_score_sel, k=int(topk), largest=True).indices
        face_idx = face_idx.index_select(0, top)
        face_score_sel = face_score_sel.index_select(0, top)
    return face_idx, face_score_sel


def _select_face_repurpose_slots(
    *,
    g: Gaussian3D,
    effective_face_idx: torch.Tensor | None,
    face_semantic_raw_t: torch.Tensor | None,
    target_raw_ids: torch.Tensor | None,
    topk: int,
    nonface_conf_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = g.xyz.device
    empty_idx = torch.zeros((0,), dtype=torch.long, device=device)
    empty_score = torch.zeros((0,), dtype=torch.float32, device=device)
    if (
        effective_face_idx is None
        or face_semantic_raw_t is None
        or target_raw_ids is None
        or g.shell_layer is None
        or g.prune_state is None
        or g.color_conf_ema is None
        or g.geom_conf_ema is None
        or int(topk) <= 0
    ):
        return empty_idx, empty_score

    face_ids_all = effective_face_idx.to(device=device, dtype=torch.long).view(-1)
    raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
    valid = (face_ids_all >= 0) & (face_ids_all < int(raw_sem_all.shape[0]))
    if not bool(valid.any()):
        return empty_idx, empty_score
    raw_local = torch.full_like(face_ids_all, -1)
    raw_local[valid] = raw_sem_all.index_select(0, face_ids_all[valid])
    active_inner = (g.shell_layer == 0) & (g.prune_state != 2)
    nonface_mask = active_inner & valid & (~torch.isin(raw_local, target_raw_ids.to(device=device, dtype=torch.long)))
    if not bool(nonface_mask.any()):
        return empty_idx, empty_score

    slot_idx = torch.nonzero(nonface_mask, as_tuple=False).view(-1)
    color_conf = g.color_conf_ema.index_select(0, slot_idx).to(dtype=torch.float32).clamp(0.0, 1.0)
    geom_conf = g.geom_conf_ema.index_select(0, slot_idx).to(dtype=torch.float32).clamp(0.0, 1.0)
    scale_mean = g.scales().detach().index_select(0, slot_idx).mean(dim=1).clamp(min=1e-5)
    conf_score = (0.65 * color_conf + 0.35 * geom_conf).clamp(0.0, 1.0)
    eligible = conf_score <= float(nonface_conf_max)
    if not bool(eligible.any()):
        return empty_idx, empty_score
    slot_idx = slot_idx[eligible]
    conf_score = conf_score[eligible]
    scale_bonus = torch.sqrt(scale_mean[eligible] / scale_mean[eligible].median().clamp(min=1e-5)).clamp(min=0.6, max=2.5)
    repurpose_score = conf_score / scale_bonus
    if int(slot_idx.numel()) > int(topk):
        take = torch.topk(-repurpose_score, k=int(topk), largest=True).indices
        slot_idx = slot_idx.index_select(0, take)
        repurpose_score = repurpose_score.index_select(0, take)
    return slot_idx, repurpose_score


def _spawn_inner_face_gaussians_from_slots(
    *,
    g: Gaussian3D,
    frame_data: BindingFrameData,
    effective_face_idx: torch.Tensor | None,
    face_ids: torch.Tensor,
    spawn_slots: torch.Tensor,
    inner_local_scale_cap_ratio: float,
    inner_local_scale_cap_normal_ratio: float,
    face_semantic_raw_t: torch.Tensor | None,
    spawn_scale_mult: float,
    spawn_opacity_logit_init: float,
    spawn_opacity_logit_cap: float,
) -> int:
    if (
        int(face_ids.numel()) <= 0
        or int(spawn_slots.numel()) <= 0
        or frame_data.faces_t is None
        or frame_data.face_t1_t is None
        or frame_data.face_t2_t is None
        or frame_data.face_n_t is None
        or frame_data.face_s1_t is None
        or frame_data.face_s2_t is None
        or g.bind_face_idx is None
        or g.bind_bary is None
        or g.bind_face_offset_local is None
        or g.bind_idx is None
        or g.bind_offset is None
        or g.shell_layer is None
        or g.prune_state is None
        or g.inner_bind_mode is None
        or g.color_conf_ema is None
        or g.geom_conf_ema is None
        or g.quality_fail_counter is None
    ):
        return 0

    n_spawn = min(int(face_ids.numel()), int(spawn_slots.numel()))
    if n_spawn <= 0:
        return 0
    dst = spawn_slots[:n_spawn]
    fs = face_ids[:n_spawn].to(device=g.xyz.device, dtype=torch.long)

    active_inner = (g.shell_layer == 0) & (g.prune_state != 2)
    eff_face_all = (
        effective_face_idx.to(device=g.xyz.device, dtype=torch.long).view(-1)
        if effective_face_idx is not None
        else torch.full((int(g.xyz.shape[0]),), -1, dtype=torch.long, device=g.xyz.device)
    )
    rgb_all = g.rgb().detach()
    scale_all = g.scales().detach()
    opacity_logit_all = g.opacity_logit.detach().view(-1, 1)
    color_conf_all = g.color_conf_ema.detach().view(-1)
    geom_conf_all = g.geom_conf_ema.detach().view(-1)
    active_inner_ids = torch.nonzero(active_inner, as_tuple=False).view(-1)
    active_eff_faces = eff_face_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None
    active_rgb = rgb_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None
    active_scale = scale_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None
    active_opacity_logit = opacity_logit_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None
    active_color_conf = color_conf_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None
    active_geom_conf = geom_conf_all.index_select(0, active_inner_ids) if int(active_inner_ids.numel()) > 0 else None

    tri_idx = frame_data.faces_t.index_select(0, fs)
    tri = frame_data.verts_t.index_select(0, tri_idx.reshape(-1)).reshape(-1, 3, 3)
    cent = tri.mean(dim=1)
    t1 = frame_data.face_t1_t.index_select(0, fs)
    t2 = frame_data.face_t2_t.index_select(0, fs)
    nf = frame_data.face_n_t.index_select(0, fs)
    s1 = (
        float(max(inner_local_scale_cap_ratio, 0.0))
        * frame_data.face_s1_t.index_select(0, fs)
    ).clamp(min=1e-4)
    s2 = (
        float(max(inner_local_scale_cap_ratio, 0.0))
        * frame_data.face_s2_t.index_select(0, fs)
    ).clamp(min=1e-4)
    s_n = (
        float(max(inner_local_scale_cap_normal_ratio, 0.0))
        * torch.minimum(s1, s2)
    ).clamp(min=1e-4)

    rgb_init = torch.full((n_spawn, 3), 0.5, dtype=torch.float32, device=g.xyz.device)
    scale_init = float(max(spawn_scale_mult, 1e-3)) * torch.stack([s1, s2, s_n], dim=1)
    opacity_init = torch.full((n_spawn, 1), float(spawn_opacity_logit_init), dtype=torch.float32, device=g.xyz.device)
    color_conf_init = torch.full((n_spawn,), 0.15, dtype=torch.float32, device=g.xyz.device)
    geom_conf_init = torch.full((n_spawn,), 0.5, dtype=torch.float32, device=g.xyz.device)

    for j in range(n_spawn):
        fj = fs[j]
        face_same = None
        if active_eff_faces is not None:
            same_face_local = active_eff_faces == fj
            if bool(same_face_local.any()):
                face_same = torch.nonzero(same_face_local, as_tuple=False).view(-1)
        src_local = face_same
        if src_local is None and face_semantic_raw_t is not None and active_eff_faces is not None:
            sem_id = int(face_semantic_raw_t[fj].item())
            if sem_id >= 0:
                active_sem = torch.full_like(active_eff_faces, -1)
                valid_sem = (active_eff_faces >= 0) & (active_eff_faces < int(face_semantic_raw_t.shape[0]))
                if bool(valid_sem.any()):
                    active_sem[valid_sem] = face_semantic_raw_t.index_select(0, active_eff_faces[valid_sem])
                same_sem_local = active_sem == sem_id
                if bool(same_sem_local.any()):
                    src_local = torch.nonzero(same_sem_local, as_tuple=False).view(-1)
        if src_local is not None and int(src_local.numel()) > 0:
            rgb_init[j] = active_rgb.index_select(0, src_local).mean(dim=0)
            scale_init[j] = torch.minimum(
                scale_init[j],
                active_scale.index_select(0, src_local).mean(dim=0),
            ).clamp(min=1e-4)
            opacity_init[j] = torch.minimum(
                active_opacity_logit.index_select(0, src_local).mean(dim=0),
                torch.full((1,), float(spawn_opacity_logit_cap), dtype=torch.float32, device=g.xyz.device),
            )
            color_conf_init[j] = active_color_conf.index_select(0, src_local).mean()
            geom_conf_init[j] = active_geom_conf.index_select(0, src_local).mean()

    R_f = torch.stack([t1, t2, nf], dim=-1)
    quat_init = quat_from_rotmat_torch(R_f)
    anchor = tri_idx[:, 0]
    rgb_init = rgb_init.clamp(1e-4, 1.0 - 1e-4)

    with torch.no_grad():
        g.xyz[dst] = cent
        g.quat[dst] = quat_init
        g.log_scales[dst] = torch.log(scale_init.clamp_min(1e-6))
        g.opacity_logit[dst] = opacity_init
        g.rgb_logit[dst] = torch.log(rgb_init / (1.0 - rgb_init))
        g.bind_idx[dst] = anchor
        g.bind_offset[dst] = 0.0
        g.bind_face_idx[dst] = fs
        if g.effective_face_idx is not None:
            g.effective_face_idx[dst] = fs
        g.bind_bary[dst, :] = (1.0 / 3.0)
        g.bind_face_offset_local[dst] = 0.0
        g.shell_layer[dst] = 0
        g.inner_bind_mode[dst] = 1
        g.bind_mix_weight[dst] = 0.5
        g.prune_state[dst] = 0
        g.quality_fail_counter[dst] = 0.0
        g.color_conf_ema[dst] = color_conf_init
        g.geom_conf_ema[dst] = geom_conf_init
    return n_spawn


def _clamp_inner_log_scales_with_local_face_caps(
    *,
    g: Gaussian3D,
    bind_mask: torch.Tensor,
    frame_data: BindingFrameData | None,
    fg_log_scale_min: float | None,
    fg_log_scale_max: float | None,
    local_ratio: float,
    local_normal_ratio: float,
) -> torch.Tensor:
    """
    Clamp inner bound gaussians using per-face local mesh scales instead of one
    global foreground max. This keeps face/hands/feet small while allowing
    torso-sized faces to grow enough to form a smooth shell.
    Returns the subset mask that was clamped locally so caller can skip a
    second global max clamp on the same gaussians.
    """
    device = g.log_scales.device
    local_applied = torch.zeros_like(bind_mask, dtype=torch.bool, device=device)
    if (
        frame_data is None
        or frame_data.face_s1_t is None
        or frame_data.face_s2_t is None
        or g.shell_layer is None
        or g.effective_face_idx is None
    ):
        return local_applied

    n_faces = int(frame_data.face_s1_t.shape[0])
    inner_face_mask = (
        bind_mask
        & (g.shell_layer == 0)
        & (g.effective_face_idx >= 0)
        & (g.effective_face_idx < n_faces)
    )
    if not bool(inner_face_mask.any()):
        return local_applied

    face_ids = g.effective_face_idx[inner_face_mask].to(device=device, dtype=torch.long)
    face_s1 = frame_data.face_s1_t.index_select(0, face_ids).clamp(min=1e-6)
    face_s2 = frame_data.face_s2_t.index_select(0, face_ids).clamp(min=1e-6)
    tang_cap = (
        float(max(local_ratio, 0.0))
        * torch.maximum(face_s1, face_s2)
    ).clamp(min=1e-6)
    normal_cap = (
        float(max(local_normal_ratio, 0.0))
        * torch.minimum(face_s1, face_s2)
    ).clamp(min=1e-6)

    min_v = -7.0 if fg_log_scale_min is None else float(fg_log_scale_min)
    tang_log_max = torch.log(tang_cap).clamp(min=min_v, max=0.0)
    normal_log_max = torch.log(normal_cap).clamp(min=min_v, max=0.0)
    # Do not reintroduce a global tangential ceiling here.
    # Small SMPL faces must be allowed to stay smaller than the old global
    # fg_log_scale_max, otherwise face/hands/feet are all lifted to the same
    # constant disk size and the whole shell collapses into a dot matrix.
    normal_log_max = torch.minimum(normal_log_max, tang_log_max)

    with torch.no_grad():
        g.log_scales[inner_face_mask, 0] = g.log_scales[inner_face_mask, 0].clamp(min=min_v)
        g.log_scales[inner_face_mask, 1] = g.log_scales[inner_face_mask, 1].clamp(min=min_v)
        g.log_scales[inner_face_mask, 2] = g.log_scales[inner_face_mask, 2].clamp(min=min_v)
        g.log_scales[inner_face_mask, 0] = torch.minimum(g.log_scales[inner_face_mask, 0], tang_log_max)
        g.log_scales[inner_face_mask, 1] = torch.minimum(g.log_scales[inner_face_mask, 1], tang_log_max)
        g.log_scales[inner_face_mask, 2] = torch.minimum(g.log_scales[inner_face_mask, 2], normal_log_max)
    local_applied[inner_face_mask] = True
    return local_applied


def _clamp_face_semantic_inner_log_scales(
    *,
    g: Gaussian3D,
    bind_mask: torch.Tensor,
    frame_data: BindingFrameData | None,
    face_ids_focus: torch.Tensor | None,
    face_semantic_raw_t: torch.Tensor | None,
    fg_log_scale_min: float | None,
    ratio: float,
    normal_ratio: float,
) -> torch.Tensor:
    device = g.log_scales.device
    applied = torch.zeros_like(bind_mask, dtype=torch.bool, device=device)
    if (
        frame_data is None
        or frame_data.face_s1_t is None
        or frame_data.face_s2_t is None
        or g.shell_layer is None
        or g.effective_face_idx is None
        or face_ids_focus is None
        or face_semantic_raw_t is None
    ):
        return applied
    n_faces = int(frame_data.face_s1_t.shape[0])
    if int(n_faces) <= 0:
        return applied
    eff = g.effective_face_idx.to(device=device, dtype=torch.long)
    valid = (eff >= 0) & (eff < n_faces)
    raw = torch.full_like(eff, -1)
    raw[valid] = face_semantic_raw_t.index_select(0, eff[valid])
    mask = bind_mask & (g.shell_layer == 0) & valid & torch.isin(raw, face_ids_focus.to(device=device, dtype=torch.long))
    if not bool(mask.any()):
        return applied
    face_idx = eff[mask]
    face_s1 = frame_data.face_s1_t.index_select(0, face_idx).clamp(min=1e-6)
    face_s2 = frame_data.face_s2_t.index_select(0, face_idx).clamp(min=1e-6)
    tang_cap = (float(max(ratio, 0.0)) * torch.maximum(face_s1, face_s2)).clamp(min=1e-6)
    normal_cap = (float(max(normal_ratio, 0.0)) * torch.minimum(face_s1, face_s2)).clamp(min=1e-6)
    min_v = -7.0 if fg_log_scale_min is None else float(fg_log_scale_min)
    tang_log_max = torch.log(tang_cap).clamp(min=min_v, max=0.0)
    normal_log_max = torch.log(normal_cap).clamp(min=min_v, max=0.0)
    normal_log_max = torch.minimum(normal_log_max, tang_log_max)
    with torch.no_grad():
        g.log_scales[mask, 0] = torch.minimum(g.log_scales[mask, 0].clamp(min=min_v), tang_log_max)
        g.log_scales[mask, 1] = torch.minimum(g.log_scales[mask, 1].clamp(min=min_v), tang_log_max)
        g.log_scales[mask, 2] = torch.minimum(g.log_scales[mask, 2].clamp(min=min_v), normal_log_max)
    applied[mask] = True
    return applied


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


def _set_camera_trainable(cam_w: torch.Tensor, cam_t: torch.Tensor, *, learn_cam: bool, flag: bool) -> None:
    if learn_cam:
        cam_w.requires_grad_(flag)
        cam_t.requires_grad_(flag)


def _checkpoint_cpu_tensor(value: torch.Tensor | object) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.tensor(value)


def _build_stage2_checkpoint_payload(
    *,
    g: Gaussian3D,
    K: torch.Tensor,
    frame_Ks: list[torch.Tensor] | None,
    cam_w: torch.Tensor | object,
    cam_t: torch.Tensor | object,
    checkpoint_metadata: dict[str, object],
    quality_pass_now: bool,
) -> dict[str, object]:
    payload = {
        "gaussians": g.state_dict(),
        "K": K.detach().cpu(),
        "cam_w": _checkpoint_cpu_tensor(cam_w),
        "cam_t": _checkpoint_cpu_tensor(cam_t),
        "frame_Ks": None if frame_Ks is None else torch.stack([row.detach().cpu() for row in frame_Ks], dim=0),
    }
    payload.update(checkpoint_metadata)
    payload["outer_quality_pass"] = quality_pass_now
    return payload


def _save_stage2_checkpoint(
    *,
    path: str | Path,
    payload: dict[str, object],
    logger,
    label: str,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    _log_info(logger, "Saved %s checkpoint: %s", label, ckpt_path)


def _build_dual_quality_gate_last(
    *,
    color_error: float,
    color_var: float,
    geometry_score: float,
    fail_active: bool,
) -> dict[str, object]:
    return {
        "color_error": color_error,
        "color_var": color_var,
        "geometry_score": geometry_score,
        "fail_active": fail_active,
    }


def train_gaussians_3d_learnable_camera(
    g: Gaussian3D,
    train_cfg: Stage2TrainLoopConfig,
):
    return _train_gaussians_3d_learnable_camera_impl(g=g, **train_cfg.to_train_kwargs())


def _train_gaussians_3d_learnable_camera_impl(
    g: Gaussian3D,
    frames_dir: str | Path,
    frames_list: list[str],
    K_list: list[float],
    out_ckpt: str | Path,
    frame_Ks: list[list[float]] | None = None,
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
    face_align_quality_enable: bool = False,
    face_align_weight_low: float = 0.25,
    face_align_weight_high: float = 3.0,
    face_align_score_min: float = 0.86,
    face_align_score_max: float = 0.92,
    face_align_score_power: float = 2.0,
    face_frontal_quality_enable: bool = False,
    face_frontal_weight_low: float = 0.85,
    face_frontal_weight_high: float = 1.5,
    face_frontal_power: float = 1.5,
    inner_legacy_e526_compat_enable: bool = False,
    face_hard_frame_select_enable: bool = False,
    face_hard_frame_keep_topk: int = 0,
    face_hard_align_min: float = 0.90,
    face_hard_frontal_min: float = 0.75,
    face_hard_reject_weight: float = 0.0,
    face_hard_exclusive_enable: bool = False,
    face_hard_sample_prob: float = 0.0,
    face_reference_frame_mode: str = "last",
    face_reference_frame_name: str = "",
    face_reference_frame_index: int = -1,
    face_reference_weight: float = 1.0,
    face_local_loss_weight: float = 0.0,
    face_local_box_shrink: float = 0.0,
    face_local_view_min: float = 0.0,
    face_local_core_enable: bool = False,
    face_patch_loss_weight: float = 0.0,
    face_patch_hf_weight: float = 0.0,
    face_patch_edge_weight: float = 0.0,
    face_patch_size: int = 160,
    face_patch_view_min: float = 0.0,
    face_patch_use_raw_gt: bool = False,
    face_feature_use_raw_gt: bool = False,
    face_feature_loss_weight: float = 0.0,
    face_feature_hf_weight: float = 0.0,
    face_feature_edge_weight: float = 0.0,
    face_feature_sigma_frac: float = 0.055,
    face_feature_view_min: float = 0.0,
    face_feature_sharpen_amount: float = 0.0,
    face_reserve_enable: bool = False,
    face_reserve_target_count: int = 0,
    face_reserve_bootstrap_every: int = 60,
    face_reserve_bootstrap_stop_iter: int = 1200,
    face_reserve_spawn_topk: int = 512,
    face_reserve_semantic_conf_min: float = 0.55,
    face_reserve_min_active_per_face: float = 12.0,
    face_reserve_scale_mult: float = 0.55,
    face_reserve_opacity_logit_init: float = -2.4,
    face_reserve_opacity_logit_cap: float = -1.9,
    face_repurpose_enable: bool = False,
    face_repurpose_target_count: int = 0,
    face_repurpose_every: int = 80,
    face_repurpose_stop_iter: int = 1200,
    face_repurpose_topk: int = 768,
    face_repurpose_nonface_conf_max: float = 0.45,
    face_refine_tail_ratio: float = 0.0,
    face_refine_lr_mul: float = 0.65,
    face_refine_sample_exclusive: bool = False,
    face_refine_face_weight_mul: float = 1.0,
    face_refine_local_weight_mul: float = 1.0,
    face_refine_patch_weight_mul: float = 1.0,
    face_refine_patch_hf_mul: float = 1.0,
    face_refine_patch_edge_mul: float = 1.0,
    face_refine_patch_size: int = 0,
    face_refine_feature_weight_mul: float = 1.0,
    face_refine_feature_hf_mul: float = 1.0,
    face_refine_feature_edge_mul: float = 1.0,
    face_refine_feature_sigma_mul: float = 1.0,
    face_refine_global_loss_scale: float = 1.0,
    face_refine_body_preserve_enable: bool = False,
    face_refine_body_preserve_weight: float = 0.0,
    face_refine_body_preserve_exclude_face: bool = True,
    face_refine_nonselected_face_exclude: bool = False,
    face_teacher_warmup_iters: int = 0,
    face_lock_enable: bool = False,
    face_lock_start_iter: int = 0,
    face_refine_nonface_grad_scale: float = 1.0,
    face_refine_face_grad_boost: float = 1.0,
    face_supervision_selected_only: bool = False,
    face_supervision_exclude_nonselected: bool = False,
    face_detail_selected_only: bool = False,
    face_nonselected_local_scale: float = 1.0,
    face_selected_frame_names: list[str] | None = None,
    face_priority_frame_names: list[str] | None = None,
    face_priority_core_frame_names: list[str] | None = None,
    face_priority_refine_exclusive: bool = False,
    face_priority_refine_sample_prob: float = 0.0,
    face_priority_frame_loss_mul: float = 1.0,
    face_priority_core_refine_sample_prob: float = 0.0,
    face_priority_core_frame_loss_mul: float = 1.0,
    face_priority_core_face_weight_scale: float = 1.0,
    face_priority_core_detail_target_scale: float = 1.0,
    face_priority_core_canonical_mul: float = 1.0,
    face_priority_core_exclude_photo: bool = False,
    face_teacher_ctx_enable: bool = False,
    face_teacher_ctx_weight: float = 1.0,
    face_teacher_ctx_count: int = 1,
    face_teacher_ctx_nonselected_only: bool = True,
    face_teacher_patch_enable: bool = False,
    face_teacher_patch_weight: float = 1.0,
    face_teacher_patch_count: int = 1,
    face_teacher_patch_nonselected_only: bool = True,
    face_teacher_patch_start_iter: int = 0,
    face_teacher_fixed_bank_enable: bool = False,
    face_teacher_fixed_bank_size: int = 4,
    face_texture_bake_enable: bool = False,
    face_texture_bake_every: int = 1,
    face_texture_bake_start_iter: int = 0,
    face_texture_bake_use_raw_gt: bool = False,
    face_texture_bake_selected_only: bool = False,
    face_texture_bake_view_min: float = 0.75,
    face_texture_bake_visibility_min: float = 0.08,
    face_texture_bake_blend: float = 0.12,
    face_texture_bake_max_blend: float = 0.40,
    face_texture_bake_feature_boost: float = 1.5,
    face_texture_bake_sharpen_amount: float = 0.35,
    face_texture_bake_feature_sigma_frac: float = 0.050,
    canonical_face_prior_enable: bool = False,
    canonical_face_prior_weight: float = 0.0,
    canonical_face_prior_hf_weight: float = 0.0,
    canonical_face_prior_edge_weight: float = 0.0,
    canonical_face_prior_size: int = 256,
    canonical_face_prior_view_min: float = 0.0,
    canonical_face_prior_selected_only: bool = False,
    canonical_face_prior_refine_only: bool = False,
    canonical_face_prior_teacher_weight_power: float = 1.0,
    canonical_face_prior_max_teachers: int = 0,
    canonical_face_prior_template_sharpen_amount: float = 0.0,
    canonical_face_prior_feature_sigma_frac: float = 0.045,
    canonical_face_prior_symmetry_weight: float = 0.0,
    canonical_face_prior_symmetry_blend: float = 0.0,
    face_semantic_scale_clamp_enable: bool = False,
    face_semantic_scale_clamp_ratio: float = 0.45,
    face_semantic_scale_clamp_normal_ratio: float = 0.14,
    face_semantic_scale_clamp_refine_only: bool = True,
    hand_weight: float = 1.0,
    hand_local_loss_weight: float = 0.0,
    hand_local_radius_scale: float = 0.18,
    hand_local_radius_min_px: int = 8,
    hand_local_radius_max_px: int = 48,
    hand_local_extension_scale: float = 0.12,
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
    inner_local_scale_cap_weight: float = 0.0,
    inner_local_scale_cap_ratio: float = 1.2,
    inner_local_scale_cap_normal_ratio: float = 0.7,
    inner_local_scale_clamp_enable: bool = False,
    inner_local_scale_clamp_ratio: float = 1.2,
    inner_local_scale_clamp_normal_ratio: float = 0.7,
    smpl_face_semantics_npz: str | Path | None = None,
    outer_block_exposed_faces: bool = True,
    outer_block_raw_parts: list[str] | tuple[str, ...] | str | None = None,
    outer_block_raw_conf_min: float = 0.0,
    outer_semantic_cap_enable: bool = False,
    outer_semantic_cap_parts: list[str] | tuple[str, ...] | str | None = None,
    outer_semantic_cap_conf_min: float = 0.0,
    outer_semantic_cap_opacity: float = 1.0,
    outer_semantic_cap_scale_xy_max: float | None = None,
    inner_local_color_prior_enable: bool = False,
    inner_local_color_prior_weight: float = 0.03,
    inner_local_color_prior_start_iter: int = 200,
    inner_local_color_prior_target_conf_max: float = 0.30,
    inner_local_color_prior_target_vis_max: float = 0.08,
    inner_local_color_prior_target_support_max: float = 0.15,
    inner_local_color_prior_donor_conf_min: float = 0.65,
    inner_local_color_prior_donor_opacity_min: float = 0.08,
    inner_local_color_prior_donor_vis_min: float = 0.05,
    inner_local_color_prior_semantic_conf_min: float = 0.55,
    inner_local_color_prior_min_donor_weight: float = 0.20,
    inner_local_color_exposed_only: bool = False,
    inner_local_color_repair_enable: bool = False,
    inner_local_color_repair_blend: float = 0.20,
    inner_local_color_repair_min_weight: float = 0.35,
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
    detail_ms_weight: float = 0.0,
    detail_ms_levels: int = 3,
    detail_ms_exclude_face: bool = False,
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
    inner_phase_use_raw_gt_target: bool = False,
    inner_phase_keep_bg_weight: bool = False,
    face_only_raw_gt_target_enable: bool = False,
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
    outer_align_prune_protect_enable: bool = True,
    outer_align_prune_protect_min_iou: float = 0.998,
    outer_align_prune_protect_min_vis: float = 0.006,
    outer_align_prune_protect_min_support: float = 0.006,
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
    inner_coverage_densify_enable: bool = True,
    inner_coverage_eval_every: int = 200,
    inner_coverage_spawn_topk: int = 256,
    inner_coverage_stop_ratio: float = 0.82,
    inner_coverage_spawn_scale_mult: float = 0.82,
    inner_coverage_spawn_opacity_logit_init: float = -2.6,
    inner_coverage_spawn_opacity_logit_cap: float = -2.0,
    inner_count_max: int = 260000,
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
    outer_spawn_min_contrib_frames: int = 3,
    outer_spawn_min_frame_ratio: float = 0.03,
    outer_spawn_min_weight_sum: float = 2.0,
    outer_spawn_score_use_persistence: bool = True,
    outer_prune_persist_protect_enable: bool = True,
    outer_prune_persist_min_contrib_frames: int = 4,
    outer_prune_persist_min_frame_ratio: float = 0.05,
    outer_prune_persist_min_weight_sum: float = 2.8,
    outer_color_refine_rgb_only: bool = False,
    outer_color_train_parts: list[str] | tuple[str, ...] | str | None = None,
    outer_color_train_conf_min: float = 0.0,
    outer_color_train_opacity_parts: list[str] | tuple[str, ...] | str | None = None,
    outer_color_train_opacity_conf_min: float = 0.0,
    outer_color_masked_enable: bool = False,
    outer_color_masked_weight: float = 0.0,
    outer_color_masked_parts: list[str] | tuple[str, ...] | str | None = None,
    outer_color_masked_conf_min: float = 0.0,
    outer_color_masked_outer_alpha_min: float = 0.01,
    outer_color_masked_replace_global: bool = False,
    outer_color_masked_use_outer_alpha: bool = True,
    outer_color_masked_exclude_hands: bool = True,
    outer_color_masked_exclude_face_box: bool = True,
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
    inner_sparse_cleanup_enable: bool = True,
    inner_sparse_cleanup_vis_th: float = 0.02,
    inner_sparse_cleanup_opacity_th: float = 0.08,
    inner_sparse_cleanup_warmup_iters: int = 800,
    inner_sparse_cleanup_patience: int = 2,
    inner_sparse_cleanup_every: int = 200,
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
    logger=None,
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
    K_np = K.detach().cpu().numpy().astype(np.float32)
    frame_K_tensors: list[torch.Tensor] | None = None
    frame_K_nps: list[np.ndarray] | None = None
    if frame_Ks is not None:
        if len(frame_Ks) != len(frames_list):
            raise ValueError("frame_Ks length must match frames_list length.")
        frame_K_tensors = []
        frame_K_nps = []
        for row in frame_Ks:
            K_row = torch.tensor(row, dtype=torch.float32, device=device).view(3, 3)
            frame_K_tensors.append(K_row)
            frame_K_nps.append(K_row.detach().cpu().numpy().astype(np.float32))

    # Load first frame for H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    train_logger = logger if logger is not None else logging.getLogger(__name__)
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
    inner_surface_stats = sanitize_inner_surface_binding_(g)
    if inner_surface_stats["inner_bind_offset_zeroed"] > 0 or inner_surface_stats["inner_face_offset_zeroed"] > 0:
        train_logger.info(
            "Inner surface binding cleanup: zeroed inner bind_offset=%d, bind_face_offset_local=%d",
            int(inner_surface_stats["inner_bind_offset_zeroed"]),
            int(inner_surface_stats["inner_face_offset_zeroed"]),
        )
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
    if g.effective_face_idx is not None:
        g.effective_face_idx = g.effective_face_idx.to(device=device, dtype=torch.int64)

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
    ref_weight_table: list[list[float]] | None = None
    if ref_window > 0:
        ref_weight_table = []
        for ctx_len in range(1, ref_window + 2):
            raw_w = np.asarray([ref_decay ** step for step in range(ctx_len - 1, -1, -1)], dtype=np.float32)
            raw_w = raw_w / float(raw_w.sum() + 1e-8)
            ref_weight_table.append(raw_w.tolist())
    bind_mesh_obj_space_mode = "auto" if mesh_context is None else str(mesh_context.bind_mesh_obj_space_mode).strip().lower()
    if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        raise ValueError(
            f"bind_mesh_obj_space must be auto|camera|blender. Got: {bind_mesh_obj_space_mode}"
        )
    bind_quat_normal_blend = float(np.clip(bind_quat_normal_blend, 0.0, 1.0))
    bind_surface_aniso_weight = float(max(bind_surface_aniso_weight, 0.0))
    bind_surface_aniso_margin = float(max(bind_surface_aniso_margin, 0.0))
    inner_local_scale_cap_weight = float(max(inner_local_scale_cap_weight, 0.0))
    inner_local_scale_cap_ratio = float(max(inner_local_scale_cap_ratio, 0.0))
    inner_local_scale_cap_normal_ratio = float(max(inner_local_scale_cap_normal_ratio, 0.0))
    inner_local_scale_clamp_enable = bool(inner_local_scale_clamp_enable)
    inner_local_scale_clamp_ratio = float(max(inner_local_scale_clamp_ratio, 0.0))
    inner_local_scale_clamp_normal_ratio = float(max(inner_local_scale_clamp_normal_ratio, 0.0))
    inner_local_color_prior_enable = bool(inner_local_color_prior_enable)
    detail_ms_weight = float(max(detail_ms_weight, 0.0))
    detail_ms_levels = int(max(int(detail_ms_levels), 1))
    detail_ms_exclude_face = bool(detail_ms_exclude_face)
    inner_local_color_prior_weight = float(max(inner_local_color_prior_weight, 0.0))
    inner_local_color_prior_start_iter = max(int(inner_local_color_prior_start_iter), 0)
    inner_local_color_prior_target_conf_max = float(np.clip(float(inner_local_color_prior_target_conf_max), 0.0, 1.0))
    inner_local_color_prior_target_vis_max = float(np.clip(float(inner_local_color_prior_target_vis_max), 0.0, 1.0))
    inner_local_color_prior_target_support_max = float(np.clip(float(inner_local_color_prior_target_support_max), 0.0, 1.0))
    inner_local_color_prior_donor_conf_min = float(np.clip(float(inner_local_color_prior_donor_conf_min), 0.0, 1.0))
    inner_local_color_prior_donor_opacity_min = float(np.clip(float(inner_local_color_prior_donor_opacity_min), 0.0, 1.0))
    inner_local_color_prior_donor_vis_min = float(np.clip(float(inner_local_color_prior_donor_vis_min), 0.0, 1.0))
    inner_local_color_prior_semantic_conf_min = float(np.clip(float(inner_local_color_prior_semantic_conf_min), 0.0, 1.0))
    inner_local_color_prior_min_donor_weight = float(max(inner_local_color_prior_min_donor_weight, 0.0))
    inner_local_color_exposed_only = bool(inner_local_color_exposed_only)
    inner_local_color_repair_enable = bool(inner_local_color_repair_enable)
    inner_local_color_repair_blend = float(np.clip(float(inner_local_color_repair_blend), 0.0, 1.0))
    inner_local_color_repair_min_weight = float(np.clip(float(inner_local_color_repair_min_weight), 0.0, 1.0))
    inner_coverage_densify_enable = bool(inner_coverage_densify_enable)
    inner_coverage_eval_every = max(int(inner_coverage_eval_every), 1)
    inner_coverage_spawn_topk = max(int(inner_coverage_spawn_topk), 0)
    inner_coverage_stop_ratio = float(np.clip(float(inner_coverage_stop_ratio), 0.0, 1.0))
    inner_coverage_spawn_scale_mult = float(max(inner_coverage_spawn_scale_mult, 1e-3))
    inner_count_max = max(int(inner_count_max), 0)
    smpl_face_semantics_path = None if smpl_face_semantics_npz is None else Path(str(smpl_face_semantics_npz))
    if (
        inner_local_color_prior_enable
        and inner_local_color_prior_target_conf_max > inner_local_color_prior_donor_conf_min
    ):
        _log_warning(
            train_logger,
            "inner_local_color_prior_target_conf_max=%.3f exceeds donor_conf_min=%.3f; clamping target threshold.",
            inner_local_color_prior_target_conf_max,
            inner_local_color_prior_donor_conf_min,
        )
        inner_local_color_prior_target_conf_max = float(inner_local_color_prior_donor_conf_min)
    inner_sparse_cleanup_enable = bool(inner_sparse_cleanup_enable)
    inner_sparse_cleanup_vis_th = float(np.clip(float(inner_sparse_cleanup_vis_th), 0.0, 1.0))
    inner_sparse_cleanup_opacity_th = float(np.clip(float(inner_sparse_cleanup_opacity_th), 0.0, 1.0))
    inner_sparse_cleanup_warmup_iters = max(int(inner_sparse_cleanup_warmup_iters), 0)
    inner_sparse_cleanup_patience = max(int(inner_sparse_cleanup_patience), 1)
    inner_sparse_cleanup_every = max(int(inner_sparse_cleanup_every), 1)
    default_bind_mode = "games_face" if (g.bind_face_idx is not None and g.bind_bary is not None) else "legacy"
    binding_cfg = parse_binding_mode_config(binding, default_mode=default_bind_mode)
    bind_rigid_align_cfg = None if mesh_context is None or mesh_context.bind_rigid_align_cfg is None else dict(mesh_context.bind_rigid_align_cfg)
    bind_rigid_align_meta: dict | None = None
    bind_rigid_align_enabled = bool(
        bind_rigid_align_cfg is not None and bool(bind_rigid_align_cfg.get("enabled", False))
    )
    if bind_rigid_align_enabled and not binding_enabled:
        _log_warning(train_logger, "bind_rigid_align is enabled but binding is disabled; ignoring rigid align.")
        bind_rigid_align_enabled = False

    training_mode = str(training_mode).strip().lower()
    if training_mode not in ("full", "inner_only", "outer_only"):
        raise ValueError(f"training_mode must be full|inner_only|outer_only. Got: {training_mode}")
    if training_mode == "outer_only" and (not bool(shell_enable)):
        raise ValueError("training_mode=outer_only requires shell_enable=true.")
    sep_fg_bg = bool(separate_fg_bg_training)
    inner_phase_use_raw_gt_target = bool(inner_phase_use_raw_gt_target)
    inner_phase_keep_bg_weight = bool(inner_phase_keep_bg_weight)
    face_only_raw_gt_target_enable = bool(face_only_raw_gt_target_enable)
    inner_legacy_e526_compat_enable = False
    if training_mode == "inner_only":
        allow_face_only_raw_gt = bool(face_only_raw_gt_target_enable) and bool(face_supervision_selected_only) and bool(face_detail_selected_only)
        if inner_phase_use_raw_gt_target and not allow_face_only_raw_gt:
            _log_warning(
                train_logger,
                "training_mode=inner_only overrides inner_phase_use_raw_gt_target=false to avoid learning raw GT instead of prepared inner targets.",
            )
        if inner_phase_keep_bg_weight:
            _log_warning(
                train_logger,
                "training_mode=inner_only overrides inner_phase_keep_bg_weight=false to avoid background spill during inner-only validation.",
            )
        if not allow_face_only_raw_gt:
            inner_phase_use_raw_gt_target = False
        inner_phase_keep_bg_weight = False
    outer_align_prune_protect_enable = bool(outer_align_prune_protect_enable)
    outer_align_prune_protect_min_iou = float(np.clip(float(outer_align_prune_protect_min_iou), 0.0, 1.0))
    outer_align_prune_protect_min_vis = float(max(float(outer_align_prune_protect_min_vis), 0.0))
    outer_align_prune_protect_min_support = float(max(float(outer_align_prune_protect_min_support), 0.0))
    fg_w = float(fg_region_weight)
    bg_w = float(bg_region_weight)
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

    # Warmup: freeze camera first
    _set_camera_trainable(cam_w, cam_t, learn_cam=learn_cam, flag=False)

    # Optional: keypoint-based bbox mask
    kpt_boxes = None
    face_boxes = None
    face_view_scores: list[float] | None = None
    face_feature_points_frames: list[dict[str, tuple[float, float]] | None] | None = None
    hand_masks_frames: list[torch.Tensor | None] | None = None
    hand_local_masks_frames: list[torch.Tensor | None] | None = None
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
            need_face_boxes = (
                float(face_weight) > 1.0
                or bool(shell_enable)
                or bool(face_align_quality_enable)
                or bool(face_frontal_quality_enable)
                or float(face_local_loss_weight) > 0.0
                or float(face_patch_loss_weight) > 0.0
                or float(face_patch_hf_weight) > 0.0
                or float(face_patch_edge_weight) > 0.0
                or float(face_feature_loss_weight) > 0.0
                or float(face_feature_hf_weight) > 0.0
                or float(face_feature_edge_weight) > 0.0
            )
            face_boxes = [] if need_face_boxes else None
            face_view_scores = [] if face_boxes is not None else None
            face_feature_points_frames = [] if (face_boxes is not None and (
                float(face_feature_loss_weight) > 0.0
                or float(face_feature_hf_weight) > 0.0
                or float(face_feature_edge_weight) > 0.0
            )) else None
            hand_masks_frames = [] if (float(hand_weight) > 1.0 or bool(shell_enable)) else None
            hand_local_masks_frames = [] if float(hand_local_loss_weight) > 0.0 else None
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
                    if face_view_scores is not None:
                        face_view_scores.append(0.0)
                    if face_feature_points_frames is not None:
                        face_feature_points_frames.append(None)
                    if hand_masks_frames is not None:
                        hand_masks_frames.append(None)
                    if hand_local_masks_frames is not None:
                        hand_local_masks_frames.append(None)
                    continue
                xs = [p[0] for p in arr if p[2] >= kpt_conf_min]
                ys = [p[1] for p in arr if p[2] >= kpt_conf_min]
                if not xs or not ys:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    if face_view_scores is not None:
                        face_view_scores.append(0.0)
                    if hand_masks_frames is not None:
                        hand_masks_frames.append(None)
                    if hand_local_masks_frames is not None:
                        hand_local_masks_frames.append(None)
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

                face_view_score = 0.0
                face_feature_points: dict[str, tuple[float, float]] | None = None
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
                        nose_ok = len(arr) > 0 and float(arr[0][2]) >= float(face_conf_min)
                        le_ok = len(arr) > 1 and float(arr[1][2]) >= float(face_conf_min)
                        re_ok = len(arr) > 2 and float(arr[2][2]) >= float(face_conf_min)
                        leare_ok = len(arr) > 3 and float(arr[3][2]) >= float(face_conf_min)
                        reare_ok = len(arr) > 4 and float(arr[4][2]) >= float(face_conf_min)
                        both_eyes = le_ok and re_ok
                        eye_pair_score = 1.0 if both_eyes else (0.35 if (le_ok or re_ok) else 0.0)
                        ear_pair_score = 1.0 if (leare_ok and reare_ok) else (0.55 if (leare_ok or reare_ok) else 0.15)
                        sym_score = 0.0
                        eye_span_score = 0.0
                        if both_eyes:
                            lx = float(arr[1][0])
                            rx = float(arr[2][0])
                            eye_span = abs(rx - lx)
                            eye_span_score = float(np.clip((eye_span / max(float(side), 1e-6)) / 0.32, 0.0, 1.0))
                            if nose_ok:
                                nx = float(arr[0][0])
                                dl = abs(nx - lx)
                                dr = abs(rx - nx)
                                sym_score = float(1.0 - np.clip(abs(dl - dr) / max(dl + dr, 1e-6), 0.0, 1.0))
                        face_view_score = float(
                            np.clip(
                                0.45 * eye_pair_score
                                + 0.25 * sym_score
                                + 0.20 * eye_span_score
                                + 0.10 * ear_pair_score,
                                0.0,
                                1.0,
                            )
                        )
                        face_feature_points = {}
                        if nose_ok:
                            face_feature_points["nose"] = (float(arr[0][0]), float(arr[0][1]))
                        if le_ok:
                            face_feature_points["left_eye"] = (float(arr[1][0]), float(arr[1][1]))
                        if re_ok:
                            face_feature_points["right_eye"] = (float(arr[2][0]), float(arr[2][1]))
                        mouth_cx = None
                        mouth_cy = None
                        mouth_half = max(0.10 * float(side), 2.0)
                        if both_eyes:
                            mouth_cx = 0.5 * (float(arr[1][0]) + float(arr[2][0]))
                            mouth_half = max(0.22 * abs(float(arr[2][0]) - float(arr[1][0])), mouth_half)
                        if nose_ok:
                            mouth_cx = float(arr[0][0]) if mouth_cx is None else float(mouth_cx)
                            mouth_cy = float(arr[0][1]) + 0.24 * float(side)
                        elif mouth_cx is not None:
                            mouth_cy = float(cy) + 0.18 * float(side)
                        if mouth_cx is not None and mouth_cy is not None:
                            mouth_cy = float(np.clip(mouth_cy, float(fy0), float(fy1)))
                            mouth_left = float(np.clip(mouth_cx - mouth_half, float(fx0), float(fx1)))
                            mouth_right = float(np.clip(mouth_cx + mouth_half, float(fx0), float(fx1)))
                            mouth_cx = float(np.clip(mouth_cx, float(fx0), float(fx1)))
                            face_feature_points["mouth_center"] = (mouth_cx, mouth_cy)
                            face_feature_points["mouth_left"] = (mouth_left, mouth_cy)
                            face_feature_points["mouth_right"] = (mouth_right, mouth_cy)
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
                            face_view_score = 0.25
                            if len(arr) > 0 and arr[0][2] >= face_conf_min:
                                mouth_y = float(np.clip(ny + 0.18 * float(side), float(fy0), float(fy1)))
                                face_feature_points = {
                                    "nose": (float(nx), float(ny)),
                                    "mouth_center": (float(nx), float(mouth_y)),
                                }
                        else:
                            face_boxes.append(None)
                            face_view_score = 0.0
                    if face_view_scores is not None:
                        face_view_scores.append(float(face_view_score))
                    if face_feature_points_frames is not None:
                        face_feature_points_frames.append(face_feature_points)

                if hand_masks_frames is not None or hand_local_masks_frames is not None:
                    hmask = np.zeros((H, W), dtype=np.uint8) if hand_masks_frames is not None else None
                    hmask_local = np.zeros((H, W), dtype=np.uint8) if hand_local_masks_frames is not None else None
                    # COCO ids: left/right elbow=7/8, wrist=9/10
                    for ei, wi in ((7, 9), (8, 10)):
                        if wi >= len(arr):
                            continue
                        wx, wy, wc = float(arr[wi][0]), float(arr[wi][1]), float(arr[wi][2])
                        if wc < float(hand_conf_min) or (not np.isfinite(wx)) or (not np.isfinite(wy)):
                            continue
                        radius = float(max(int(hand_radius_min_px), 1))
                        local_radius = float(max(int(hand_local_radius_min_px), 1))
                        hx, hy = wx, wy
                        hx_local, hy_local = wx, wy
                        if ei < len(arr):
                            ex, ey, ec = float(arr[ei][0]), float(arr[ei][1]), float(arr[ei][2])
                            if ec >= float(hand_conf_min) and np.isfinite(ex) and np.isfinite(ey):
                                forearm = float(np.hypot(wx - ex, wy - ey))
                                radius = max(radius, float(hand_radius_scale) * forearm)
                                hx = wx + float(hand_extension_scale) * (wx - ex)
                                hy = wy + float(hand_extension_scale) * (wy - ey)
                                local_radius = max(local_radius, float(hand_local_radius_scale) * forearm)
                                hx_local = wx + float(hand_local_extension_scale) * (wx - ex)
                                hy_local = wy + float(hand_local_extension_scale) * (wy - ey)
                        if hmask is not None:
                            cx = int(np.clip(round(hx), 0, W - 1))
                            cy = int(np.clip(round(hy), 0, H - 1))
                            rr = int(max(1, round(radius)))
                            cv2.circle(hmask, (cx, cy), rr, 1, thickness=-1, lineType=cv2.LINE_AA)
                        if hmask_local is not None:
                            cx_local = int(np.clip(round(hx_local), 0, W - 1))
                            cy_local = int(np.clip(round(hy_local), 0, H - 1))
                            rr_local = int(max(1, round(min(float(hand_local_radius_max_px), local_radius))))
                            cv2.circle(hmask_local, (cx_local, cy_local), rr_local, 1, thickness=-1, lineType=cv2.LINE_AA)
                    if hand_masks_frames is not None:
                        if int(hmask.sum()) > 0:
                            hand_masks_frames.append(torch.from_numpy(hmask.astype(np.float32)).to(device=device).unsqueeze(0))
                        else:
                            hand_masks_frames.append(None)
                    if hand_local_masks_frames is not None:
                        if int(hmask_local.sum()) > 0:
                            hand_local_masks_frames.append(torch.from_numpy(hmask_local.astype(np.float32)).to(device=device).unsqueeze(0))
                        else:
                            hand_local_masks_frames.append(None)

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
        logger=train_logger,
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
    face_align_quality_enable = bool(face_align_quality_enable) and (face_boxes is not None)
    face_align_weight_low = float(max(face_align_weight_low, 0.0))
    face_align_weight_high = float(max(face_align_weight_high, face_align_weight_low))
    face_align_score_power = float(max(face_align_score_power, 1e-6))
    if float(face_align_score_max) < float(face_align_score_min):
        face_align_score_max = float(face_align_score_min)
    face_frontal_quality_enable = bool(face_frontal_quality_enable)
    face_frontal_weight_low = float(max(face_frontal_weight_low, 0.0))
    face_frontal_weight_high = float(max(face_frontal_weight_high, face_frontal_weight_low))
    face_frontal_power = float(max(face_frontal_power, 1e-6))
    face_hard_frame_select_enable = bool(face_hard_frame_select_enable) and (face_boxes is not None)
    face_hard_frame_keep_topk = int(max(face_hard_frame_keep_topk, 0))
    face_hard_align_min = float(np.clip(float(face_hard_align_min), 0.0, 1.0))
    face_hard_frontal_min = float(np.clip(float(face_hard_frontal_min), 0.0, 1.0))
    face_hard_reject_weight = float(np.clip(float(face_hard_reject_weight), 0.0, 1.0))
    face_hard_exclusive_enable = bool(face_hard_exclusive_enable) and bool(face_hard_frame_select_enable)
    face_hard_sample_prob = float(np.clip(float(face_hard_sample_prob), 0.0, 1.0))
    face_reference_weight = float(max(face_reference_weight, 1.0))
    face_local_loss_weight = float(max(face_local_loss_weight, 0.0))
    face_local_box_shrink = float(np.clip(float(face_local_box_shrink), 0.0, 0.45))
    face_local_view_min = float(np.clip(float(face_local_view_min), 0.0, 1.0))
    face_local_core_enable = bool(face_local_core_enable)
    face_patch_loss_weight = float(max(face_patch_loss_weight, 0.0))
    face_patch_hf_weight = float(max(face_patch_hf_weight, 0.0))
    face_patch_edge_weight = float(max(face_patch_edge_weight, 0.0))
    face_patch_size = int(max(int(face_patch_size), 32))
    face_patch_view_min = float(np.clip(float(face_patch_view_min), 0.0, 1.0))
    face_patch_use_raw_gt = bool(face_patch_use_raw_gt)
    face_feature_use_raw_gt = bool(face_feature_use_raw_gt)
    face_feature_loss_weight = float(max(face_feature_loss_weight, 0.0))
    face_feature_hf_weight = float(max(face_feature_hf_weight, 0.0))
    face_feature_edge_weight = float(max(face_feature_edge_weight, 0.0))
    face_feature_sigma_frac = float(np.clip(float(face_feature_sigma_frac), 1e-3, 0.35))
    face_feature_view_min = float(np.clip(float(face_feature_view_min), 0.0, 1.0))
    face_feature_sharpen_amount = float(np.clip(float(face_feature_sharpen_amount), 0.0, 2.0))
    face_reserve_enable = bool(face_reserve_enable)
    face_reserve_target_count = int(max(int(face_reserve_target_count), 0))
    face_reserve_bootstrap_every = int(max(int(face_reserve_bootstrap_every), 1))
    face_reserve_bootstrap_stop_iter = int(max(int(face_reserve_bootstrap_stop_iter), 0))
    face_reserve_spawn_topk = int(max(int(face_reserve_spawn_topk), 0))
    face_reserve_semantic_conf_min = float(np.clip(float(face_reserve_semantic_conf_min), 0.0, 1.0))
    face_reserve_min_active_per_face = float(max(float(face_reserve_min_active_per_face), 1.0))
    face_reserve_scale_mult = float(max(float(face_reserve_scale_mult), 1e-3))
    face_reserve_opacity_logit_init = float(face_reserve_opacity_logit_init)
    face_reserve_opacity_logit_cap = float(min(face_reserve_opacity_logit_cap, face_reserve_opacity_logit_init + 1.5))
    face_repurpose_enable = bool(face_repurpose_enable)
    face_repurpose_target_count = int(max(int(face_repurpose_target_count), 0))
    face_repurpose_every = int(max(int(face_repurpose_every), 1))
    face_repurpose_stop_iter = int(max(int(face_repurpose_stop_iter), 0))
    face_repurpose_topk = int(max(int(face_repurpose_topk), 0))
    face_repurpose_nonface_conf_max = float(np.clip(float(face_repurpose_nonface_conf_max), 0.0, 1.0))
    face_refine_tail_ratio = float(np.clip(float(face_refine_tail_ratio), 0.0, 0.95))
    face_refine_lr_mul = float(np.clip(float(face_refine_lr_mul), 0.05, 2.0))
    face_refine_sample_exclusive = bool(face_refine_sample_exclusive)
    face_refine_face_weight_mul = float(max(float(face_refine_face_weight_mul), 1.0))
    face_refine_local_weight_mul = float(max(float(face_refine_local_weight_mul), 1.0))
    face_refine_patch_weight_mul = float(max(float(face_refine_patch_weight_mul), 1.0))
    face_refine_patch_hf_mul = float(max(float(face_refine_patch_hf_mul), 1.0))
    face_refine_patch_edge_mul = float(max(float(face_refine_patch_edge_mul), 1.0))
    face_refine_patch_size = int(max(int(face_refine_patch_size), 0))
    face_refine_feature_weight_mul = float(max(float(face_refine_feature_weight_mul), 1.0))
    face_refine_feature_hf_mul = float(max(float(face_refine_feature_hf_mul), 1.0))
    face_refine_feature_edge_mul = float(max(float(face_refine_feature_edge_mul), 1.0))
    face_refine_feature_sigma_mul = float(max(float(face_refine_feature_sigma_mul), 1.0))
    face_refine_global_loss_scale = float(np.clip(float(face_refine_global_loss_scale), 0.0, 1.0))
    face_refine_body_preserve_enable = bool(face_refine_body_preserve_enable)
    face_refine_body_preserve_weight = float(max(float(face_refine_body_preserve_weight), 0.0))
    face_refine_body_preserve_exclude_face = bool(face_refine_body_preserve_exclude_face)
    face_refine_nonselected_face_exclude = bool(face_refine_nonselected_face_exclude)
    face_teacher_warmup_iters = int(max(int(face_teacher_warmup_iters), 0))
    face_lock_enable = bool(face_lock_enable)
    face_lock_start_iter = int(max(int(face_lock_start_iter), 0))
    face_refine_nonface_grad_scale = float(np.clip(float(face_refine_nonface_grad_scale), 0.0, 1.0))
    face_refine_face_grad_boost = float(max(float(face_refine_face_grad_boost), 1.0))
    if face_priority_frame_names is None:
        face_priority_frame_names = []
    elif isinstance(face_priority_frame_names, str):
        face_priority_frame_names = [str(face_priority_frame_names)]
    else:
        face_priority_frame_names = [str(x) for x in face_priority_frame_names if str(x).strip()]
    if face_priority_core_frame_names is None:
        face_priority_core_frame_names = []
    elif isinstance(face_priority_core_frame_names, str):
        face_priority_core_frame_names = [str(face_priority_core_frame_names)]
    else:
        face_priority_core_frame_names = [str(x) for x in face_priority_core_frame_names if str(x).strip()]
    face_priority_refine_exclusive = bool(face_priority_refine_exclusive)
    face_priority_refine_sample_prob = float(np.clip(float(face_priority_refine_sample_prob), 0.0, 1.0))
    face_priority_frame_loss_mul = float(max(float(face_priority_frame_loss_mul), 1.0))
    face_priority_core_refine_sample_prob = float(np.clip(float(face_priority_core_refine_sample_prob), 0.0, 1.0))
    face_priority_core_frame_loss_mul = float(max(float(face_priority_core_frame_loss_mul), 1.0))
    face_priority_core_face_weight_scale = float(np.clip(float(face_priority_core_face_weight_scale), 0.0, 1.0))
    face_priority_core_detail_target_scale = float(np.clip(float(face_priority_core_detail_target_scale), 0.0, 1.0))
    face_priority_core_canonical_mul = float(max(float(face_priority_core_canonical_mul), 1.0))
    face_priority_core_exclude_photo = bool(face_priority_core_exclude_photo)
    face_supervision_selected_only = bool(face_supervision_selected_only)
    face_supervision_exclude_nonselected = bool(face_supervision_exclude_nonselected)
    face_detail_selected_only = bool(face_detail_selected_only)
    face_nonselected_local_scale = float(np.clip(float(face_nonselected_local_scale), 0.0, 1.0))
    if face_selected_frame_names is None:
        face_selected_frame_names = []
    elif isinstance(face_selected_frame_names, str):
        face_selected_frame_names = [str(face_selected_frame_names)]
    else:
        face_selected_frame_names = [str(x) for x in face_selected_frame_names if str(x).strip()]
    face_teacher_ctx_enable = bool(face_teacher_ctx_enable)
    face_teacher_ctx_weight = float(max(float(face_teacher_ctx_weight), 0.0))
    face_teacher_ctx_count = int(max(int(face_teacher_ctx_count), 1))
    face_teacher_ctx_nonselected_only = bool(face_teacher_ctx_nonselected_only)
    face_teacher_patch_enable = bool(face_teacher_patch_enable)
    face_teacher_patch_weight = float(max(float(face_teacher_patch_weight), 0.0))
    face_teacher_patch_count = int(max(int(face_teacher_patch_count), 1))
    face_teacher_patch_nonselected_only = bool(face_teacher_patch_nonselected_only)
    face_teacher_patch_start_iter = int(max(int(face_teacher_patch_start_iter), 0))
    face_teacher_fixed_bank_enable = bool(face_teacher_fixed_bank_enable)
    face_teacher_fixed_bank_size = int(max(int(face_teacher_fixed_bank_size), 1))
    face_texture_bake_enable = bool(face_texture_bake_enable)
    face_texture_bake_every = int(max(int(face_texture_bake_every), 1))
    face_texture_bake_start_iter = int(max(int(face_texture_bake_start_iter), 0))
    face_texture_bake_use_raw_gt = bool(face_texture_bake_use_raw_gt)
    face_texture_bake_selected_only = bool(face_texture_bake_selected_only)
    face_texture_bake_view_min = float(np.clip(float(face_texture_bake_view_min), 0.0, 1.0))
    face_texture_bake_visibility_min = float(np.clip(float(face_texture_bake_visibility_min), 0.0, 1.0))
    face_texture_bake_blend = float(np.clip(float(face_texture_bake_blend), 0.0, 1.0))
    face_texture_bake_max_blend = float(np.clip(float(face_texture_bake_max_blend), 0.0, 1.0))
    face_texture_bake_feature_boost = float(max(float(face_texture_bake_feature_boost), 0.0))
    face_texture_bake_sharpen_amount = float(np.clip(float(face_texture_bake_sharpen_amount), 0.0, 2.0))
    face_texture_bake_feature_sigma_frac = float(np.clip(float(face_texture_bake_feature_sigma_frac), 1e-3, 0.35))
    canonical_face_prior_enable = bool(canonical_face_prior_enable)
    canonical_face_prior_weight = float(max(float(canonical_face_prior_weight), 0.0))
    canonical_face_prior_hf_weight = float(max(float(canonical_face_prior_hf_weight), 0.0))
    canonical_face_prior_edge_weight = float(max(float(canonical_face_prior_edge_weight), 0.0))
    canonical_face_prior_size = int(max(int(canonical_face_prior_size), 64))
    canonical_face_prior_view_min = float(np.clip(float(canonical_face_prior_view_min), 0.0, 1.0))
    canonical_face_prior_selected_only = bool(canonical_face_prior_selected_only)
    canonical_face_prior_refine_only = bool(canonical_face_prior_refine_only)
    canonical_face_prior_teacher_weight_power = float(max(float(canonical_face_prior_teacher_weight_power), 0.0))
    canonical_face_prior_max_teachers = int(max(int(canonical_face_prior_max_teachers), 0))
    canonical_face_prior_template_sharpen_amount = float(max(float(canonical_face_prior_template_sharpen_amount), 0.0))
    canonical_face_prior_feature_sigma_frac = float(np.clip(float(canonical_face_prior_feature_sigma_frac), 1e-3, 0.20))
    canonical_face_prior_symmetry_weight = float(max(float(canonical_face_prior_symmetry_weight), 0.0))
    canonical_face_prior_symmetry_blend = float(np.clip(float(canonical_face_prior_symmetry_blend), 0.0, 1.0))
    face_semantic_scale_clamp_enable = bool(face_semantic_scale_clamp_enable)
    face_semantic_scale_clamp_ratio = float(max(float(face_semantic_scale_clamp_ratio), 0.0))
    face_semantic_scale_clamp_normal_ratio = float(max(float(face_semantic_scale_clamp_normal_ratio), 0.0))
    face_semantic_scale_clamp_refine_only = bool(face_semantic_scale_clamp_refine_only)
    if outer_block_raw_parts is None:
        outer_block_raw_part_names: list[str] = []
    elif isinstance(outer_block_raw_parts, str):
        outer_block_raw_part_names = [
            p.strip() for p in outer_block_raw_parts.replace(",", " " ).split() if p.strip()
        ]
    else:
        outer_block_raw_part_names = [str(p).strip() for p in outer_block_raw_parts if str(p).strip()]
    outer_block_raw_part_names = sorted(set(outer_block_raw_part_names))
    outer_block_raw_conf_min = float(np.clip(float(outer_block_raw_conf_min), 0.0, 1.0))
    outer_semantic_cap_enable = bool(outer_semantic_cap_enable) and bool(shell_enable)
    if outer_semantic_cap_parts is None:
        outer_semantic_cap_part_names: list[str] = []
    elif isinstance(outer_semantic_cap_parts, str):
        outer_semantic_cap_part_names = [
            p.strip() for p in outer_semantic_cap_parts.replace(",", " ").split() if p.strip()
        ]
    else:
        outer_semantic_cap_part_names = [str(p).strip() for p in outer_semantic_cap_parts if str(p).strip()]
    outer_semantic_cap_part_names = sorted(set(outer_semantic_cap_part_names))
    if outer_semantic_cap_enable and len(outer_semantic_cap_part_names) == 0:
        _log_warning(train_logger, "outer_semantic_cap_enable=true but outer_semantic_cap_parts is empty; disabling cap.")
        outer_semantic_cap_enable = False
    outer_semantic_cap_conf_min = float(np.clip(float(outer_semantic_cap_conf_min), 0.0, 1.0))
    outer_semantic_cap_opacity = float(np.clip(float(outer_semantic_cap_opacity), 1e-4, 1.0))
    outer_color_refine_rgb_only = bool(outer_color_refine_rgb_only)
    if outer_color_train_parts is None:
        outer_color_train_part_names: list[str] = []
    elif isinstance(outer_color_train_parts, str):
        outer_color_train_part_names = [
            p.strip() for p in outer_color_train_parts.replace(',', ' ').split() if p.strip()
        ]
    else:
        outer_color_train_part_names = [str(p).strip() for p in outer_color_train_parts if str(p).strip()]
    outer_color_train_part_names = sorted(set(outer_color_train_part_names))
    outer_color_train_conf_min = float(np.clip(float(outer_color_train_conf_min), 0.0, 1.0))
    outer_color_train_parts_enable = bool(shell_enable) and len(outer_color_train_part_names) > 0
    if outer_color_train_opacity_parts is None:
        outer_color_train_opacity_part_names: list[str] = []
    elif isinstance(outer_color_train_opacity_parts, str):
        outer_color_train_opacity_part_names = [
            p.strip() for p in outer_color_train_opacity_parts.replace(',', ' ').split() if p.strip()
        ]
    else:
        outer_color_train_opacity_part_names = [str(p).strip() for p in outer_color_train_opacity_parts if str(p).strip()]
    outer_color_train_opacity_part_names = sorted(set(outer_color_train_opacity_part_names))
    outer_color_train_opacity_conf_min = float(np.clip(float(outer_color_train_opacity_conf_min), 0.0, 1.0))
    outer_color_train_opacity_parts_enable = bool(shell_enable) and len(outer_color_train_opacity_part_names) > 0
    outer_color_masked_enable = bool(outer_color_masked_enable) and bool(shell_enable)
    outer_color_masked_weight = float(max(float(outer_color_masked_weight), 0.0))
    if outer_color_masked_parts is None:
        outer_color_masked_part_names: list[str] = []
    elif isinstance(outer_color_masked_parts, str):
        outer_color_masked_part_names = [
            p.strip() for p in outer_color_masked_parts.replace(",", " " ).split() if p.strip()
        ]
    else:
        outer_color_masked_part_names = [str(p).strip() for p in outer_color_masked_parts if str(p).strip()]
    outer_color_masked_part_names = sorted(set(outer_color_masked_part_names))
    if outer_color_masked_enable and len(outer_color_masked_part_names) == 0:
        _log_warning(train_logger, "outer_color_masked_enable=true but outer_color_masked_parts is empty; disabling masked color loss.")
        outer_color_masked_enable = False
    outer_color_masked_conf_min = float(np.clip(float(outer_color_masked_conf_min), 0.0, 1.0))
    outer_color_masked_outer_alpha_min = float(np.clip(float(outer_color_masked_outer_alpha_min), 0.0, 1.0))
    outer_color_masked_replace_global = bool(outer_color_masked_replace_global)
    outer_color_masked_use_outer_alpha = bool(outer_color_masked_use_outer_alpha)
    outer_color_masked_exclude_hands = bool(outer_color_masked_exclude_hands)
    outer_color_masked_exclude_face_box = bool(outer_color_masked_exclude_face_box)
    if outer_color_masked_enable and outer_color_masked_weight <= 0.0:
        _log_warning(train_logger, "outer_color_masked_enable=true but outer_color_masked_weight<=0; disabling masked color loss.")
        outer_color_masked_enable = False
    if outer_semantic_cap_scale_xy_max is not None:
        outer_semantic_cap_scale_xy_max = float(outer_semantic_cap_scale_xy_max)
        if float(outer_semantic_cap_scale_xy_max) <= 0.0:
            outer_semantic_cap_scale_xy_max = None
    hand_local_loss_weight = float(max(hand_local_loss_weight, 0.0))
    hand_local_radius_scale = float(max(hand_local_radius_scale, 0.0))
    hand_local_radius_min_px = int(max(int(hand_local_radius_min_px), 1))
    hand_local_radius_max_px = int(max(int(hand_local_radius_max_px), hand_local_radius_min_px))
    hand_local_extension_scale = float(np.clip(float(hand_local_extension_scale), 0.0, 0.35))
    face_reference_frame_mode = str(face_reference_frame_mode).strip().lower()
    if face_reference_frame_mode not in ("last", "first", "index", "name", "none"):
        face_reference_frame_mode = "last"
    face_selected_mask_np = None
    face_selected_weight_np = None
    face_selected_ids_np = None
    face_priority_ids_np = None
    face_priority_core_ids_np = None
    face_teacher_bank_ids_np = None
    canonical_face_prior_template = None
    canonical_face_prior_mask = None
    canonical_face_prior_feature_mask = None
    if bool(face_hard_frame_select_enable):
        valid_face_np = np.asarray(
            [(face_boxes[i] is not None) for i in range(len(frames_list))],
            dtype=bool,
        )
        align_np = np.zeros((len(frames_list),), dtype=np.float32)
        if frame_align_scores_t is not None:
            align_np = frame_align_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
        frontal_np = np.zeros((len(frames_list),), dtype=np.float32)
        if face_view_scores is not None:
            frontal_np = np.asarray(face_view_scores, dtype=np.float32)
        candidate_np = (
            valid_face_np
            & (align_np >= float(face_hard_align_min))
            & (frontal_np >= float(face_hard_frontal_min))
        )
        score_np = (align_np * frontal_np).astype(np.float32, copy=False)
        selected_np = np.zeros((len(frames_list),), dtype=bool)
        keep_topk = int(face_hard_frame_keep_topk) if int(face_hard_frame_keep_topk) > 0 else 1
        selected_ranked: list[int] = []
        cand_ids = np.nonzero(candidate_np)[0].astype(np.int64)
        if int(cand_ids.size) > 0:
            order = cand_ids[np.argsort(-score_np[cand_ids], kind="stable")]
            selected_ranked = [int(x) for x in order[: min(int(keep_topk), int(order.size))].tolist()]
            if selected_ranked:
                selected_np[np.asarray(selected_ranked, dtype=np.int64)] = True
        else:
            valid_ids = np.nonzero(valid_face_np)[0].astype(np.int64)
            if int(valid_ids.size) > 0:
                best_idx = int(valid_ids[np.argmax(score_np[valid_ids])])
                selected_ranked = [best_idx]
                selected_np[best_idx] = True

        ref_idx = None
        if face_reference_frame_mode == "last":
            ref_idx = len(frames_list) - 1
        elif face_reference_frame_mode == "first":
            ref_idx = 0
        elif face_reference_frame_mode == "index":
            if 0 <= int(face_reference_frame_index) < len(frames_list):
                ref_idx = int(face_reference_frame_index)
        elif face_reference_frame_mode == "name":
            try:
                ref_idx = [str(x) for x in frames_list].index(str(face_reference_frame_name))
            except ValueError:
                ref_idx = None
        if ref_idx is not None and 0 <= int(ref_idx) < len(frames_list) and bool(valid_face_np[int(ref_idx)]):
            ref_idx_i = int(ref_idx)
            if not bool(selected_np[ref_idx_i]):
                selected_ids_now = list(selected_ranked)
                if int(len(selected_ids_now)) >= int(keep_topk) and int(len(selected_ids_now)) > 0:
                    worst_idx = min(selected_ids_now, key=lambda ii: float(score_np[int(ii)]))
                    selected_np[int(worst_idx)] = False
                    selected_ranked = [int(ii) for ii in selected_ranked if int(ii) != int(worst_idx)]
                selected_np[ref_idx_i] = True
                selected_ranked = [ref_idx_i] + [int(ii) for ii in selected_ranked if int(ii) != int(ref_idx_i)]
            else:
                selected_ranked = [ref_idx_i] + [int(ii) for ii in selected_ranked if int(ii) != int(ref_idx_i)]

        if face_selected_frame_names:
            name_to_idx_manual = {str(x): i for i, x in enumerate(frames_list)}
            manual_ranked = [int(name_to_idx_manual[n]) for n in face_selected_frame_names if n in name_to_idx_manual and bool(valid_face_np[int(name_to_idx_manual[n])])]
            if manual_ranked:
                selected_np = np.zeros((len(frames_list),), dtype=bool)
                selected_np[np.asarray(manual_ranked, dtype=np.int64)] = True
                selected_ranked = manual_ranked
                ref_idx = manual_ranked[0]
        weight_np = np.ones((len(frames_list),), dtype=np.float32)
        if ref_idx is not None and 0 <= int(ref_idx) < len(frames_list) and bool(selected_np[int(ref_idx)]):
            weight_np[int(ref_idx)] = float(face_reference_weight)
        face_selected_mask_np = selected_np
        face_selected_weight_np = weight_np
        face_selected_ids_np = np.asarray(selected_ranked, dtype=np.int64)
        if bool(face_teacher_fixed_bank_enable) and int(face_selected_ids_np.size) > 0:
            face_teacher_bank_ids_np = face_selected_ids_np[: min(int(face_teacher_fixed_bank_size), int(face_selected_ids_np.size))].astype(np.int64)
        selected_ids = face_selected_ids_np
        selected_desc = [
            f"{frames_list[int(i)]}(align={align_np[int(i)]:.3f}, frontal={frontal_np[int(i)]:.3f}, w={weight_np[int(i)]:.2f})"
            for i in selected_ids.tolist()
        ]
        _log_info(
            train_logger,
            "Face hard-frame selection: kept %d/%d frame(s); align>=%.3f frontal>=%.3f topk=%d ref_mode=%s frames=%s",
            int(selected_np.sum()),
            int(len(frames_list)),
            float(face_hard_align_min),
            float(face_hard_frontal_min),
            int(keep_topk),
            str(face_reference_frame_mode),
            selected_desc,
        )
        face_priority_core_ids_np = None
        if face_priority_frame_names or face_priority_core_frame_names:
            name_to_idx = {str(x): i for i, x in enumerate(frames_list)}
            if face_priority_frame_names:
                pri_ids = [int(name_to_idx[n]) for n in face_priority_frame_names if n in name_to_idx]
                if pri_ids:
                    face_priority_ids_np = np.asarray(pri_ids, dtype=np.int64)
                    _log_info(
                        train_logger,
                        "Face priority frames: count=%d frames=%s refine_exclusive=%s sample_prob=%.2f loss_mul=%.2f",
                        int(face_priority_ids_np.size),
                        [str(frames_list[i]) for i in face_priority_ids_np.tolist()],
                        str(bool(face_priority_refine_exclusive)),
                        float(face_priority_refine_sample_prob),
                        float(face_priority_frame_loss_mul),
                    )
            if face_priority_core_frame_names:
                core_ids = [int(name_to_idx[n]) for n in face_priority_core_frame_names if n in name_to_idx]
                if core_ids:
                    face_priority_core_ids_np = np.asarray(core_ids, dtype=np.int64)
                    _log_info(
                        train_logger,
                        "Face priority core frames: count=%d frames=%s sample_prob=%.2f loss_mul=%.2f",
                        int(face_priority_core_ids_np.size),
                        [str(frames_list[i]) for i in face_priority_core_ids_np.tolist()],
                        float(face_priority_core_refine_sample_prob),
                        float(face_priority_core_frame_loss_mul),
                    )
        if bool(canonical_face_prior_enable):
            teacher_source_np = face_teacher_bank_ids_np if (face_teacher_bank_ids_np is not None and int(face_teacher_bank_ids_np.size) > 0) else face_selected_ids_np
            canonical_face_prior_template, canonical_face_prior_mask, canonical_face_prior_feature_mask, canonical_used = _build_canonical_face_template(
                gts=gts,
                face_boxes=face_boxes,
                face_feature_points_frames=face_feature_points_frames,
                teacher_ids=teacher_source_np,
                teacher_weights=face_selected_weight_np,
                out_size=int(canonical_face_prior_size),
                device=device,
                teacher_weight_power=float(canonical_face_prior_teacher_weight_power),
                max_teachers=int(canonical_face_prior_max_teachers),
                sharpen_amount=float(canonical_face_prior_template_sharpen_amount),
                feature_sigma_frac=float(canonical_face_prior_feature_sigma_frac),
                symmetry_blend=float(canonical_face_prior_symmetry_blend),
            )
            _log_info(
                train_logger,
                "Canonical face prior: enabled=%s teacher_frames=%d size=%d built=%s max_teachers=%d teacher_power=%.2f sharpen=%.2f sigma=%.3f refine_only=%s",
                str(bool(canonical_face_prior_enable)),
                0 if teacher_source_np is None else int(teacher_source_np.size),
                int(canonical_face_prior_size),
                int(canonical_used),
                int(canonical_face_prior_max_teachers),
                float(canonical_face_prior_teacher_weight_power),
                float(canonical_face_prior_template_sharpen_amount),
                float(canonical_face_prior_feature_sigma_frac),
                str(bool(canonical_face_prior_refine_only)),
            )
    person_fuse = supervision_state.person_fuse
    outer_contour_fuse = supervision_state.outer_contour_fuse
    if mesh_runtime is not None and prepared_supervision is not None:
        smpl_align_scales = prepared_supervision.get("smpl_align_scales", None)
        smpl_align_tx_px = prepared_supervision.get("smpl_align_tx_px", None)
        smpl_align_ty_px = prepared_supervision.get("smpl_align_ty_px", None)
        smpl_align_accepted = prepared_supervision.get("smpl_align_accepted", None)
        if (
            smpl_align_scales is not None
            and smpl_align_tx_px is not None
            and smpl_align_ty_px is not None
            and smpl_align_accepted is not None
        ):
            mesh_runtime.set_frame_smpl_align_from_arrays(
                frame_names=[str(x) for x in prepared_supervision["frame_names"]],
                scales=np.asarray(smpl_align_scales, dtype=np.float32),
                tx_px=np.asarray(smpl_align_tx_px, dtype=np.float32),
                ty_px=np.asarray(smpl_align_ty_px, dtype=np.float32),
                accepted=np.asarray(smpl_align_accepted, dtype=np.uint8),
            )
            _log_info(
                train_logger,
                "Applied prepared frame-wise SMPL similarity alignment to binding mesh runtime.",
            )

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
        _log_warning(
            train_logger,
            "outer activation/spawn/prune ignores semantic bias settings: %s",
            ", ".join(sorted(set(ignored_outer_bias_flags))),
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
        frame_K_nps=frame_K_nps,
        image_hw=(H, W),
        axis_angle_to_rotmat_fn=axis_angle_to_rotmat,
        cam_w=cam_w,
        cam_t=cam_t,
        logger=train_logger,
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
    outer_face_align_peak = binding_state.outer_face_align_peak
    outer_face_gap_half_weighted_sum = binding_state.outer_face_gap_half_weighted_sum
    outer_face_gap_scale_weighted_sum = binding_state.outer_face_gap_scale_weighted_sum
    outer_face_spawned_mask = binding_state.outer_face_spawned_mask
    outer_face_first_hit_frame_idx = binding_state.outer_face_first_hit_frame_idx
    outer_face_contrib_frames = binding_state.outer_face_contrib_frames
    outer_face_spawn_iter = binding_state.outer_face_spawn_iter
    outer_face_spawn_frame_idx = binding_state.outer_face_spawn_frame_idx
    bind_rigid_align_meta = binding_state.bind_rigid_align_meta
    face_semantics_asset: Stage2FaceSemantics | None = None
    face_semantic_raw_t: torch.Tensor | None = None
    face_semantic_conf_t: torch.Tensor | None = None
    face_semantic_focus_ids_t: torch.Tensor | None = None
    face_semantic_color_prior_ignore_ids_t: torch.Tensor | None = None
    outer_blocked_face_mask_t: torch.Tensor | None = None
    outer_semantic_cap_raw_ids_t: torch.Tensor | None = None
    outer_color_train_raw_ids_t: torch.Tensor | None = None
    outer_color_train_opacity_raw_ids_t: torch.Tensor | None = None
    outer_color_masked_raw_ids_t: torch.Tensor | None = None
    need_face_semantics = bool(inner_local_color_prior_enable) or bool(face_reserve_enable) or (
        bool(shell_enable) and bool(outer_block_exposed_faces)
        or bool(outer_semantic_cap_enable)
        or bool(outer_color_train_parts_enable)
        or bool(outer_color_train_opacity_parts_enable)
        or bool(outer_color_masked_enable)
    )
    if need_face_semantics:
        if not bool(binding_enabled) or g.bind_face_idx is None:
            _log_warning(
                train_logger,
                "Semantic face policies require mesh binding with valid bind_face_idx; disabling them for this run.",
            )
            inner_local_color_prior_enable = False
            outer_block_exposed_faces = False
            outer_semantic_cap_enable = False
            outer_color_train_parts_enable = False
            outer_color_train_opacity_parts_enable = False
            outer_color_masked_enable = False
            face_reserve_enable = False
        else:
            expected_face_count = None
            if bind_faces_np is not None:
                expected_face_count = int(bind_faces_np.shape[0])
            elif bind_face_neighbor_idx_t is not None:
                expected_face_count = int(bind_face_neighbor_idx_t.shape[0])
            if smpl_face_semantics_path is None:
                _log_warning(
                    train_logger,
                    "Semantic face policies requested but smpl_face_semantics_npz is unset; disabling them for this run.",
                )
                inner_local_color_prior_enable = False
                outer_block_exposed_faces = False
                outer_semantic_cap_enable = False
                outer_color_train_parts_enable = False
                outer_color_train_opacity_parts_enable = False
                outer_color_masked_enable = False
                face_reserve_enable = False
            else:
                try:
                    face_semantics_asset = load_stage2_face_semantics(
                        smpl_face_semantics_path,
                        expected_face_count=expected_face_count,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    _log_warning(
                        train_logger,
                        "SMPL face semantics unavailable (%s); disabling semantic face policies for this run.",
                        str(exc),
                    )
                    inner_local_color_prior_enable = False
                    outer_block_exposed_faces = False
                    outer_semantic_cap_enable = False
                    outer_color_train_parts_enable = False
                    outer_color_train_opacity_parts_enable = False
                    outer_color_masked_enable = False
                    face_reserve_enable = False
                    face_semantics_asset = None
                if face_semantics_asset is not None:
                    face_semantic_raw_t = torch.as_tensor(
                        face_semantics_asset.raw_label_ids,
                        device=device,
                        dtype=torch.long,
                    )
                    face_semantic_conf_t = torch.as_tensor(
                        face_semantics_asset.raw_label_confidence,
                        device=device,
                        dtype=torch.float32,
                    )
                    ignore_raw_ids: list[int] = []
                    if bool(inner_local_color_exposed_only) and face_semantics_asset.exclude_outer_mask is not None:
                        raw_ids_np = np.asarray(face_semantics_asset.raw_label_ids, dtype=np.int64).reshape(-1)
                        exposed_face_np = np.asarray(face_semantics_asset.exclude_outer_mask, dtype=np.uint8).reshape(-1) > 0
                        ignore_mask_np = (raw_ids_np >= 0) & (~exposed_face_np)
                        if bool(ignore_mask_np.any()):
                            ignore_raw_ids.extend(np.unique(raw_ids_np[ignore_mask_np]).astype(np.int64).tolist())
                    else:
                        for _name in ("face", "hair"):
                            try:
                                ignore_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                            except ValueError:
                                pass
                    if ignore_raw_ids:
                        face_semantic_color_prior_ignore_ids_t = torch.as_tensor(
                            np.asarray(sorted(set(ignore_raw_ids)), dtype=np.int64),
                            device=device,
                            dtype=torch.long,
                        )
                    focus_raw_ids: list[int] = []
                    for _name in ("face",):
                        try:
                            focus_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                        except ValueError:
                            pass
                    if focus_raw_ids:
                        face_semantic_focus_ids_t = torch.as_tensor(
                            np.asarray(sorted(set(focus_raw_ids)), dtype=np.int64),
                            device=device,
                            dtype=torch.long,
                        )
                    elif bool(face_reserve_enable):
                        _log_warning(
                            train_logger,
                            "face_reserve_enable=true but SMPL face semantics asset has no raw label named 'face'; disabling face reserve.",
                        )
                        face_reserve_enable = False
                    if bool(outer_block_exposed_faces):
                        blocked_np = face_semantics_asset.exclude_outer_mask
                        if blocked_np is None:
                            blocked_np = (np.asarray(face_semantics_asset.group_ids, dtype=np.int16).reshape(-1) == 1).astype(
                                np.uint8
                            )
                        blocked_mask_np = np.asarray(blocked_np, dtype=np.uint8).reshape(-1) > 0
                        if len(outer_block_raw_part_names) > 0:
                            raw_block_ids: list[int] = []
                            for _name in outer_block_raw_part_names:
                                try:
                                    raw_block_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                                except ValueError:
                                    pass
                            if raw_block_ids:
                                raw_ids_np = np.asarray(face_semantics_asset.raw_label_ids, dtype=np.int64).reshape(-1)
                                raw_conf_np = np.asarray(face_semantics_asset.raw_label_confidence, dtype=np.float32).reshape(-1)
                                raw_block_np = np.isin(raw_ids_np, np.asarray(sorted(set(raw_block_ids)), dtype=np.int64))
                                if float(outer_block_raw_conf_min) > 0.0:
                                    raw_block_np = raw_block_np & (raw_conf_np >= float(outer_block_raw_conf_min))
                                blocked_mask_np = blocked_mask_np | raw_block_np
                                _log_info(
                                    train_logger,
                                    "Enabled outer raw-part block: parts=%s conf_min=%.3f blocked_faces=%d",
                                    outer_block_raw_part_names,
                                    float(outer_block_raw_conf_min),
                                    int(raw_block_np.sum()),
                                )
                            else:
                                _log_warning(train_logger, "outer_block_raw_parts not found in SMPL semantics: %s", outer_block_raw_part_names)
                        outer_blocked_face_mask_t = torch.as_tensor(
                            blocked_mask_np,
                            device=device,
                            dtype=torch.bool,
                        )
                    if bool(outer_semantic_cap_enable):
                        cap_raw_ids: list[int] = []
                        for _name in outer_semantic_cap_part_names:
                            try:
                                cap_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                            except ValueError:
                                pass
                        if cap_raw_ids:
                            outer_semantic_cap_raw_ids_t = torch.as_tensor(
                                np.asarray(sorted(set(cap_raw_ids)), dtype=np.int64),
                                device=device,
                                dtype=torch.long,
                            )
                            _log_info(
                                train_logger,
                                "Enabled outer semantic cap: parts=%s opacity<=%.4f scale_xy_max=%s conf_min=%.3f",
                                outer_semantic_cap_part_names,
                                float(outer_semantic_cap_opacity),
                                str(outer_semantic_cap_scale_xy_max),
                                float(outer_semantic_cap_conf_min),
                            )
                        else:
                            _log_warning(train_logger, "outer semantic cap parts not found in SMPL semantics: %s; disabling cap.", outer_semantic_cap_part_names)
                            outer_semantic_cap_enable = False
                    if bool(outer_color_train_parts_enable):
                        color_train_raw_ids: list[int] = []
                        for _name in outer_color_train_part_names:
                            try:
                                color_train_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                            except ValueError:
                                pass
                        if color_train_raw_ids:
                            outer_color_train_raw_ids_t = torch.as_tensor(
                                np.asarray(sorted(set(color_train_raw_ids)), dtype=np.int64),
                                device=device,
                                dtype=torch.long,
                            )
                            _log_info(
                                train_logger,
                                'Enabled outer RGB train-part gate: parts=%s conf_min=%.3f',
                                outer_color_train_part_names,
                                float(outer_color_train_conf_min),
                            )
                        else:
                            _log_warning(train_logger, 'outer_color_train_parts not found in SMPL semantics: %s; disabling RGB train-part gate.', outer_color_train_part_names)
                            outer_color_train_parts_enable = False
                    if bool(outer_color_train_opacity_parts_enable):
                        opacity_train_raw_ids: list[int] = []
                        for _name in outer_color_train_opacity_part_names:
                            try:
                                opacity_train_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                            except ValueError:
                                pass
                        if opacity_train_raw_ids:
                            outer_color_train_opacity_raw_ids_t = torch.as_tensor(
                                np.asarray(sorted(set(opacity_train_raw_ids)), dtype=np.int64),
                                device=device,
                                dtype=torch.long,
                            )
                            _log_info(
                                train_logger,
                                'Enabled outer opacity train-part gate: parts=%s conf_min=%.3f',
                                outer_color_train_opacity_part_names,
                                float(outer_color_train_opacity_conf_min),
                            )
                        else:
                            _log_warning(train_logger, 'outer_color_train_opacity_parts not found in SMPL semantics: %s; disabling opacity train-part gate.', outer_color_train_opacity_part_names)
                            outer_color_train_opacity_parts_enable = False
                    if bool(outer_color_masked_enable):
                        masked_raw_ids: list[int] = []
                        for _name in outer_color_masked_part_names:
                            try:
                                masked_raw_ids.append(int(face_semantics_asset.raw_label_names.index(_name)))
                            except ValueError:
                                pass
                        if masked_raw_ids:
                            outer_color_masked_raw_ids_t = torch.as_tensor(
                                np.asarray(sorted(set(masked_raw_ids)), dtype=np.int64),
                                device=device,
                                dtype=torch.long,
                            )
                            _log_info(
                                train_logger,
                                "Enabled outer masked combined color loss: parts=%s weight=%.4f conf_min=%.3f alpha_min=%.4f replace_global=%s use_outer_alpha=%s",
                                outer_color_masked_part_names,
                                float(outer_color_masked_weight),
                                float(outer_color_masked_conf_min),
                                float(outer_color_masked_outer_alpha_min),
                                str(bool(outer_color_masked_replace_global)),
                                str(bool(outer_color_masked_use_outer_alpha)),
                            )
                        else:
                            _log_warning(train_logger, "outer masked color parts not found in SMPL semantics: %s; disabling masked color loss.", outer_color_masked_part_names)
                            outer_color_masked_enable = False
                    _log_info(
                        train_logger,
                        "Loaded SMPL face semantics: faces=%d asset=%s",
                        int(face_semantics_asset.face_count),
                        str(face_semantics_asset.npz_path),
                    )
                    if bool(inner_local_color_prior_enable):
                        _log_info(
                            train_logger,
                            "Enabled inner local color prior.",
                        )
                    if bool(outer_block_exposed_faces) and outer_blocked_face_mask_t is not None:
                        _log_info(
                            train_logger,
                            "Enabled outer exposed-face block: blocked_faces=%d",
                            int(outer_blocked_face_mask_t.sum().item()),
                        )
    bind_rigid_apply_cfg = binding_state.bind_rigid_apply_cfg
    _load_mesh_np_raw = binding_state.load_mesh_np_raw
    _load_mesh_verts = binding_state.load_mesh_verts
    _load_binding_frame = binding_state.load_binding_frame
    bind_mask = (
        (bind_idx_t >= 0)
        if bind_idx_t is not None
        else torch.zeros((int(g.xyz.shape[0]),), dtype=torch.bool, device=device)
    )
    if g.effective_face_idx is None:
        g.effective_face_idx = torch.full((n_gauss,), -1, dtype=torch.int64, device=device)
    need_rebuild_effective_face = False
    if int(g.effective_face_idx.shape[0]) != n_gauss:
        g.effective_face_idx = torch.full((n_gauss,), -1, dtype=torch.int64, device=device)
        need_rebuild_effective_face = True
    if (
        bind_faces_np is not None
        and int(len(bind_faces_np)) > 0
        and bind_idx_t is not None
        and g.shell_layer is not None
    ):
        inner_missing_effective = (
            (g.shell_layer == 0)
            & bind_mask
            & (
                (g.effective_face_idx < 0)
                | (g.effective_face_idx >= int(len(bind_faces_np)))
            )
        )
        need_rebuild_effective_face = need_rebuild_effective_face or bool(inner_missing_effective.any())
        if need_rebuild_effective_face:
            verts_ref_np, _ = _load_mesh_np_raw(int(bind_mesh_indices[0]))
            g.effective_face_idx = compute_effective_face_indices_torch(
                bind_idx_t=g.bind_idx,
                bind_face_idx_t=g.bind_face_idx,
                shell_layer_t=g.shell_layer,
                verts_np=verts_ref_np,
                faces_np=bind_faces_np,
                device=device,
            )
            inner_effective_ok = int(
                (
                    (g.shell_layer == 0)
                    & bind_mask
                    & (g.effective_face_idx >= 0)
                    & (g.effective_face_idx < int(len(bind_faces_np)))
                ).sum().item()
            )
            inner_total_bound = int(((g.shell_layer == 0) & bind_mask).sum().item())
            _log_info(
                train_logger,
                "Effective inner face refs: valid=%d/%d",
                inner_effective_ok,
                inner_total_bound,
            )

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
        first_frame_by_mesh_idx: dict[int, int] = {}
        for fj_pc, mesh_idx_pc in enumerate(bind_mesh_indices):
            mesh_idx_key = int(mesh_idx_pc)
            if mesh_idx_key not in first_frame_by_mesh_idx:
                first_frame_by_mesh_idx[mesh_idx_key] = int(fj_pc)
        for mesh_idx_pc, fj_pc in sorted(first_frame_by_mesh_idx.items()):
            verts_pc, _, _ = _load_mesh_verts(fj_pc)
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
    if outer_geom_fraction <= 0.0:
        outer_geom_end = int(phase_inner_end)
    else:
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

    frame_residual_ema_alpha = 0.9
    frame_residual_ema = torch.zeros((nF,), dtype=torch.float32, device=device)
    frame_residual_ema_initialized = torch.zeros((nF,), dtype=torch.bool, device=device)
    densify_stats = GaussianDensifyStats.create(n_gaussians=n_gauss, device=device)
    border_att_cache: dict[int, torch.Tensor] = {}
    edge_strength_cache: dict[int, torch.Tensor] = {}
    outer_gt_mask_hint_cache: dict[int, torch.Tensor] = {}
    outer_face_gap_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    outer_color_semantic_mask_cache: dict[int, torch.Tensor] = {}
    prev_vertex_dirs: torch.Tensor | None = None

    inner_soft_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    inner_hard_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    inner_sparse_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_low_support_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_boundary_violation_counter = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
    outer_boundary_hard_clamp_count = 0
    inner_sparse_pruned_total = 0

    outer_spawn_stats_by_frame: dict[str, dict[str, int]] = {
        str(fn): {"total": 0, "trigger": 0, "fallback": 0, "iters_hit": 0}
        for fn in frames_list
    }
    outer_spawn_iter_records: list[dict[str, object]] = []
    report_paths = prepare_stage2_report_paths(out_ckpt_path=out_ckpt_path, reset_quality_log=True)
    quality_log_path = report_paths.quality_log_path
    quality_report_path = report_paths.quality_report_path
    outer_spawn_report_path = report_paths.outer_spawn_report_path
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
    inner_color_prior_last_loss = 0.0
    inner_color_prior_last_targets = 0
    inner_color_prior_last_assisted = 0
    inner_color_prior_last_donors = 0
    inner_coverage_last_spawned = 0

    checkpoint_metadata_base = {
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
        "face_local_loss_weight": face_local_loss_weight,
        "face_local_core_enable": face_local_core_enable,
        "face_local_box_shrink": face_local_box_shrink,
        "face_hard_sample_prob": face_hard_sample_prob,
        "face_patch_loss_weight": face_patch_loss_weight,
        "face_patch_hf_weight": face_patch_hf_weight,
        "face_patch_edge_weight": face_patch_edge_weight,
        "face_patch_size": face_patch_size,
        "face_patch_view_min": face_patch_view_min,
        "face_feature_loss_weight": face_feature_loss_weight,
        "face_feature_hf_weight": face_feature_hf_weight,
        "face_feature_edge_weight": face_feature_edge_weight,
        "face_feature_sigma_frac": face_feature_sigma_frac,
        "face_feature_view_min": face_feature_view_min,
        "face_feature_sharpen_amount": face_feature_sharpen_amount,
        "face_reserve_enable": face_reserve_enable,
        "face_reserve_target_count": face_reserve_target_count,
        "face_reserve_bootstrap_every": face_reserve_bootstrap_every,
        "face_reserve_bootstrap_stop_iter": face_reserve_bootstrap_stop_iter,
        "face_reserve_spawn_topk": face_reserve_spawn_topk,
        "face_reserve_semantic_conf_min": face_reserve_semantic_conf_min,
        "face_reserve_min_active_per_face": face_reserve_min_active_per_face,
        "face_reserve_scale_mult": face_reserve_scale_mult,
        "face_reserve_opacity_logit_init": face_reserve_opacity_logit_init,
        "face_reserve_opacity_logit_cap": face_reserve_opacity_logit_cap,
        "face_repurpose_enable": face_repurpose_enable,
        "face_repurpose_target_count": face_repurpose_target_count,
        "face_repurpose_every": face_repurpose_every,
        "face_repurpose_stop_iter": face_repurpose_stop_iter,
        "face_repurpose_topk": face_repurpose_topk,
        "face_repurpose_nonface_conf_max": face_repurpose_nonface_conf_max,
        "face_refine_tail_ratio": face_refine_tail_ratio,
        "face_refine_lr_mul": face_refine_lr_mul,
        "face_refine_sample_exclusive": face_refine_sample_exclusive,
        "face_refine_face_weight_mul": face_refine_face_weight_mul,
        "face_refine_local_weight_mul": face_refine_local_weight_mul,
        "face_refine_patch_weight_mul": face_refine_patch_weight_mul,
        "face_refine_patch_hf_mul": face_refine_patch_hf_mul,
        "face_refine_patch_edge_mul": face_refine_patch_edge_mul,
        "face_refine_patch_size": face_refine_patch_size,
        "face_refine_feature_weight_mul": face_refine_feature_weight_mul,
        "face_refine_feature_hf_mul": face_refine_feature_hf_mul,
        "face_refine_feature_edge_mul": face_refine_feature_edge_mul,
        "face_refine_feature_sigma_mul": face_refine_feature_sigma_mul,
        "face_refine_global_loss_scale": face_refine_global_loss_scale,
        "face_refine_body_preserve_enable": face_refine_body_preserve_enable,
        "face_refine_body_preserve_weight": face_refine_body_preserve_weight,
        "face_refine_body_preserve_exclude_face": face_refine_body_preserve_exclude_face,
        "face_refine_nonselected_face_exclude": face_refine_nonselected_face_exclude,
        "face_teacher_warmup_iters": face_teacher_warmup_iters,
        "face_lock_enable": face_lock_enable,
        "face_lock_start_iter": face_lock_start_iter,
        "face_refine_nonface_grad_scale": face_refine_nonface_grad_scale,
        "face_refine_face_grad_boost": face_refine_face_grad_boost,
        "face_priority_frame_names": list(face_priority_frame_names),
        "face_priority_refine_exclusive": face_priority_refine_exclusive,
        "canonical_face_prior_enable": canonical_face_prior_enable,
        "canonical_face_prior_weight": canonical_face_prior_weight,
        "canonical_face_prior_hf_weight": canonical_face_prior_hf_weight,
        "canonical_face_prior_edge_weight": canonical_face_prior_edge_weight,
        "canonical_face_prior_size": canonical_face_prior_size,
        "canonical_face_prior_view_min": canonical_face_prior_view_min,
        "canonical_face_prior_selected_only": canonical_face_prior_selected_only,
        "canonical_face_prior_refine_only": canonical_face_prior_refine_only,
        "canonical_face_prior_teacher_weight_power": canonical_face_prior_teacher_weight_power,
        "canonical_face_prior_max_teachers": canonical_face_prior_max_teachers,
        "canonical_face_prior_template_sharpen_amount": canonical_face_prior_template_sharpen_amount,
        "canonical_face_prior_feature_sigma_frac": canonical_face_prior_feature_sigma_frac,
        "face_semantic_scale_clamp_enable": face_semantic_scale_clamp_enable,
        "face_semantic_scale_clamp_ratio": face_semantic_scale_clamp_ratio,
        "face_semantic_scale_clamp_normal_ratio": face_semantic_scale_clamp_normal_ratio,
        "face_semantic_scale_clamp_refine_only": face_semantic_scale_clamp_refine_only,
        "hand_weight": hand_weight,
        "hand_local_loss_weight": hand_local_loss_weight,
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
        "inner_local_scale_cap_weight": inner_local_scale_cap_weight,
        "inner_local_scale_cap_ratio": inner_local_scale_cap_ratio,
        "inner_local_scale_cap_normal_ratio": inner_local_scale_cap_normal_ratio,
        "outer_block_raw_parts": outer_block_raw_part_names,
        "outer_block_raw_conf_min": outer_block_raw_conf_min,
        "outer_semantic_cap_enable": outer_semantic_cap_enable,
        "outer_semantic_cap_parts": outer_semantic_cap_part_names,
        "outer_semantic_cap_conf_min": outer_semantic_cap_conf_min,
        "outer_semantic_cap_opacity": outer_semantic_cap_opacity,
        "outer_semantic_cap_scale_xy_max": outer_semantic_cap_scale_xy_max,
        "smpl_face_semantics_npz": None if smpl_face_semantics_path is None else str(smpl_face_semantics_path),
        "inner_local_color_prior_enable": inner_local_color_prior_enable,
        "inner_local_color_prior_weight": inner_local_color_prior_weight,
        "inner_local_color_prior_start_iter": inner_local_color_prior_start_iter,
        "inner_local_color_prior_target_conf_max": inner_local_color_prior_target_conf_max,
        "inner_local_color_prior_donor_conf_min": inner_local_color_prior_donor_conf_min,
        "inner_local_color_prior_donor_opacity_min": inner_local_color_prior_donor_opacity_min,
        "inner_local_color_prior_donor_vis_min": inner_local_color_prior_donor_vis_min,
        "inner_local_color_prior_semantic_conf_min": inner_local_color_prior_semantic_conf_min,
        "inner_local_color_prior_min_donor_weight": inner_local_color_prior_min_donor_weight,
        "inner_local_color_exposed_only": inner_local_color_exposed_only,
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
        "detail_ms_weight": detail_ms_weight,
        "detail_ms_levels": detail_ms_levels,
        "detail_ms_exclude_face": detail_ms_exclude_face,
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
        "inner_phase_use_raw_gt_target": inner_phase_use_raw_gt_target,
        "inner_phase_keep_bg_weight": inner_phase_keep_bg_weight,
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
        "outer_align_prune_protect_enable": outer_align_prune_protect_enable,
        "outer_align_prune_protect_min_iou": outer_align_prune_protect_min_iou,
        "outer_align_prune_protect_min_vis": outer_align_prune_protect_min_vis,
        "outer_align_prune_protect_min_support": outer_align_prune_protect_min_support,
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
        "outer_color_refine_rgb_only": outer_color_refine_rgb_only,
        "outer_color_train_parts": outer_color_train_part_names,
        "outer_color_train_conf_min": outer_color_train_conf_min,
        "outer_color_train_opacity_parts": outer_color_train_opacity_part_names,
        "outer_color_train_opacity_conf_min": outer_color_train_opacity_conf_min,
        "outer_color_masked_enable": outer_color_masked_enable,
        "outer_color_masked_weight": outer_color_masked_weight,
        "outer_color_masked_parts": outer_color_masked_part_names,
        "outer_color_masked_conf_min": outer_color_masked_conf_min,
        "outer_color_masked_outer_alpha_min": outer_color_masked_outer_alpha_min,
        "outer_color_masked_replace_global": outer_color_masked_replace_global,
        "outer_color_masked_use_outer_alpha": outer_color_masked_use_outer_alpha,
        "outer_color_masked_exclude_hands": outer_color_masked_exclude_hands,
        "outer_color_masked_exclude_face_box": outer_color_masked_exclude_face_box,
        "outer_quality_log_every": outer_quality_log_every,
        "outer_progress_log_every": outer_progress_log_every,
        "outer_zero_stop_enable": outer_zero_stop_enable,
        "outer_zero_stop_patience": outer_zero_stop_patience,
        "outer_zero_stop_warmup_iters": outer_zero_stop_warmup_iters,
        "outer_zero_stop_after_phase_inner": outer_zero_stop_after_phase_inner,
        "bind_face_offset_straight_through": bind_face_offset_straight_through,
        "dual_quality_gate": dual_quality_gate,
        "outer_quality_report_path": str(quality_report_path),
        "outer_spawn_report_path": str(outer_spawn_report_path),
        "training_mode": training_mode,
        "save_inner_checkpoint_path": None if save_inner_checkpoint_path is None else str(save_inner_checkpoint_path),
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
    face_refine_tail_iters = int(round(float(iters) * float(face_refine_tail_ratio)))
    face_refine_tail_iters = max(0, min(int(iters), face_refine_tail_iters))
    face_refine_start_iter = max(int(iters) - face_refine_tail_iters, 0)

    pbar = tqdm(range(iters), desc="Stage2 3DGS (learnable camera)")
    for it in pbar:
        opt.zero_grad(set_to_none=True)
        face_refine_phase = bool(face_refine_tail_iters > 0 and it >= face_refine_start_iter)
        face_lock_active = bool(face_lock_enable) and int(face_lock_start_iter) > 0 and it >= int(face_lock_start_iter)

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
            if face_refine_phase:
                lr_mul = lr_mul * float(face_refine_lr_mul)
            pg["lr"] = base_lr * lr_mul

        # Enable camera learning after warmup
        if learn_cam and it == cam_warmup_iters:
            _set_camera_trainable(cam_w, cam_t, learn_cam=learn_cam, flag=True)

        # Select anchor frame (current frame to optimize).
        if bool(face_lock_active):
            fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        elif (
            face_refine_phase
            and int(face_teacher_warmup_iters) > 0
            and it < int(face_teacher_warmup_iters)
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
        ):
            sample_j = int(torch.randint(low=0, high=int(face_selected_ids_np.size), size=(1,), generator=rng).item())
            fi = int(face_selected_ids_np[sample_j])
        elif (
            face_refine_phase
            and face_priority_core_ids_np is not None
            and int(face_priority_core_ids_np.size) > 0
            and float(face_priority_core_refine_sample_prob) > 0.0
            and float(torch.rand(size=(1,), generator=rng).item()) < float(face_priority_core_refine_sample_prob)
        ):
            sample_j = int(torch.randint(low=0, high=int(face_priority_core_ids_np.size), size=(1,), generator=rng).item())
            fi = int(face_priority_core_ids_np[sample_j])
        elif (
            face_refine_phase
            and face_priority_ids_np is not None
            and int(face_priority_ids_np.size) > 0
            and (
                bool(face_priority_refine_exclusive)
                or (
                    float(face_priority_refine_sample_prob) > 0.0
                    and float(torch.rand(size=(1,), generator=rng).item()) < float(face_priority_refine_sample_prob)
                )
            )
        ):
            sample_j = int(torch.randint(low=0, high=int(face_priority_ids_np.size), size=(1,), generator=rng).item())
            fi = int(face_priority_ids_np[sample_j])
        elif (
            face_refine_phase
            and bool(face_refine_sample_exclusive)
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
        ):
            sample_j = int(torch.randint(low=0, high=int(face_selected_ids_np.size), size=(1,), generator=rng).item())
            fi = int(face_selected_ids_np[sample_j])
        elif ref_anchor == "last":
            fi = nF - 1
        elif (
            face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
            and float(face_hard_sample_prob) > 0.0
            and float(torch.rand(size=(1,), generator=rng).item()) < float(face_hard_sample_prob)
        ):
            sample_j = int(torch.randint(low=0, high=int(face_selected_ids_np.size), size=(1,), generator=rng).item())
            fi = int(face_selected_ids_np[sample_j])
        elif frame_weights_t is None:
            fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        else:
            fi = int(torch.multinomial(frame_weights_t, 1, replacement=True, generator=rng).item())

        # Build temporal reference set: current frame + previous frames with decay.
        if ref_window > 0:
            start_idx = max(0, fi - ref_window)
            ctx_indices = list(range(start_idx, fi + 1))
            ctx_weights = ref_weight_table[len(ctx_indices) - 1] if ref_weight_table is not None else [1.0]
        else:
            ctx_indices = [fi]
            ctx_weights = [1.0]
        ctx_face_only_flags = [False for _ in ctx_indices]
        strict_twoframe_face_fit_mode = (
            int(nF) <= 2
            and bool(face_supervision_selected_only)
            and bool(face_detail_selected_only)
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
        )
        if strict_twoframe_face_fit_mode:
            ctx_face_only_flags = [True for _ in ctx_indices]

        if (
            bool(face_teacher_ctx_enable)
            and bool(face_refine_phase)
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
            and float(face_teacher_ctx_weight) > 0.0
            and (not bool(face_lock_active))
        ):
            use_teacher_ctx = True
            if bool(face_teacher_ctx_nonselected_only) and face_selected_mask_np is not None and bool(face_selected_mask_np[fi]):
                use_teacher_ctx = False
            if use_teacher_ctx:
                teacher_source_np = face_teacher_bank_ids_np if (face_teacher_bank_ids_np is not None and int(face_teacher_bank_ids_np.size) > 0) else face_selected_ids_np
                teacher_pool = [int(x) for x in teacher_source_np.tolist() if int(x) != int(fi)]
                if not teacher_pool:
                    teacher_pool = [int(x) for x in teacher_source_np.tolist()]
                if teacher_pool:
                    max_take = min(int(face_teacher_ctx_count), len(teacher_pool))
                    picked = teacher_pool[:max_take]
                    for teacher_idx in picked:
                        if teacher_idx in ctx_indices:
                            pos = ctx_indices.index(teacher_idx)
                            ctx_weights[pos] = float(ctx_weights[pos]) + float(face_teacher_ctx_weight)
                        else:
                            ctx_indices.append(teacher_idx)
                            ctx_weights.append(float(face_teacher_ctx_weight))
                            ctx_face_only_flags.append(
                                bool(
                                    int(face_teacher_warmup_iters) > 0
                                    and it < int(face_teacher_warmup_iters)
                                )
                            )

        if (
            bool(face_refine_phase)
            and int(face_teacher_warmup_iters) > 0
            and it < int(face_teacher_warmup_iters)
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
        ):
            warmup_teacher_ids = {int(x) for x in face_selected_ids_np.tolist()}
            for _pos, _fj in enumerate(ctx_indices):
                if int(_fj) in warmup_teacher_ids:
                    ctx_face_only_flags[_pos] = True

        if (
            bool(face_teacher_patch_enable)
            and (int(it + 1) >= int(face_teacher_patch_start_iter))
            and face_selected_ids_np is not None
            and int(face_selected_ids_np.size) > 0
            and float(face_teacher_patch_weight) > 0.0
            and (not bool(face_lock_active))
        ):
            use_teacher_patch = True
            if bool(face_teacher_patch_nonselected_only) and face_selected_mask_np is not None and bool(face_selected_mask_np[fi]):
                use_teacher_patch = False
            if use_teacher_patch:
                teacher_source_np = face_teacher_bank_ids_np if (face_teacher_bank_ids_np is not None and int(face_teacher_bank_ids_np.size) > 0) else face_selected_ids_np
                teacher_pool = [int(x) for x in teacher_source_np.tolist() if int(x) != int(fi)]
                if not teacher_pool:
                    teacher_pool = [int(x) for x in teacher_source_np.tolist()]
                if teacher_pool:
                    max_take = min(int(face_teacher_patch_count), len(teacher_pool))
                    picked = teacher_pool[:max_take]
                    for teacher_idx in picked:
                        if teacher_idx in ctx_indices:
                            pos = ctx_indices.index(teacher_idx)
                            ctx_weights[pos] = float(ctx_weights[pos]) + float(face_teacher_patch_weight)
                            ctx_face_only_flags[pos] = True
                        else:
                            ctx_indices.append(teacher_idx)
                            ctx_weights.append(float(face_teacher_patch_weight))
                            ctx_face_only_flags.append(True)

        loss_value_t: torch.Tensor | None = None
        spawn_frame_name = str(frames_list[fi])
        vis_accum: list[torch.Tensor] = []
        inner_color_support_accum: list[torch.Tensor] = []
        edge_accum: list[torch.Tensor] = []
        outer_detached_support_accum: list[torch.Tensor] = []
        outer_gap_half_n_accum: list[torch.Tensor] = []
        best_face_bake_state: dict[str, object] | None = None
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
        for fj, wj, ctx_face_only in zip(ctx_indices, ctx_weights, ctx_face_only_flags):
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
            K_curr = frame_K_tensors[fj] if frame_K_tensors is not None else K

            # Update bound xyz for this reference frame (dynamic binding)
            mesh_idx = None
            verts_t = None
            normals_t = None
            bind_mask_curr = bind_mask if binding_enabled else None
            frame_data = None
            if binding_enabled:
                mesh_idx = int(bind_mesh_indices[fj])
                verts_t, normals_t, mesh_idx_use = _load_mesh_verts(int(fj))
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
                        frame_data = _load_binding_frame(int(fj))
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
            hand_local_mask_curr = None
            if hand_local_masks_frames is not None:
                hand_local_mask_curr = hand_local_masks_frames[fj]
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
                        K=K_curr,
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
                        K=K_curr,
                    R=R,
                    t=t,
                    image_h=H,
                    image_w=W,
                    sample_dilate_px=int(max(outer_trigger_face_sample_dilate_px, 0)),
                )
                if face_trigger_support is not None:
                    if (
                        bool(outer_block_exposed_faces)
                        and outer_blocked_face_mask_t is not None
                        and int(outer_blocked_face_mask_t.shape[0]) == int(face_trigger_support.shape[0])
                    ):
                        face_trigger_support = face_trigger_support.clone()
                        face_trigger_support[outer_blocked_face_mask_t] = 0.0
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
                                if outer_face_align_peak is not None:
                                    peak_now = outer_face_align_peak.index_select(0, hit_faces)
                                    peak_new = torch.maximum(
                                        peak_now,
                                        torch.full_like(peak_now, float(align_weight)),
                                    )
                                    outer_face_align_peak.index_copy_(0, hit_faces, peak_new)
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
            face_exclude_mask = None
            face_weight_eff = float(face_weight)
            face_local_loss_weight_eff = float(face_local_loss_weight)
            face_patch_loss_weight_eff = float(face_patch_loss_weight)
            face_patch_hf_weight_eff = float(face_patch_hf_weight)
            face_patch_edge_weight_eff = float(face_patch_edge_weight)
            face_patch_size_eff = int(face_patch_size)
            face_feature_loss_weight_eff = float(face_feature_loss_weight)
            face_feature_hf_weight_eff = float(face_feature_hf_weight)
            face_feature_edge_weight_eff = float(face_feature_edge_weight)
            face_feature_sigma_frac_eff = float(face_feature_sigma_frac)
            face_feature_sharpen_amount_eff = float(face_feature_sharpen_amount)
            canonical_face_prior_weight_eff = float(canonical_face_prior_weight)
            canonical_face_prior_hf_weight_eff = float(canonical_face_prior_hf_weight)
            canonical_face_prior_edge_weight_eff = float(canonical_face_prior_edge_weight)
            face_selected_active = True
            if bool(face_hard_frame_select_enable) and face_selected_mask_np is not None:
                face_selected_active = bool(face_selected_mask_np[fj])
            if face_refine_phase:
                face_weight_eff = face_weight_eff * float(face_refine_face_weight_mul)
                face_local_loss_weight_eff = face_local_loss_weight_eff * float(face_refine_local_weight_mul)
                face_patch_loss_weight_eff = face_patch_loss_weight_eff * float(face_refine_patch_weight_mul)
                face_patch_hf_weight_eff = face_patch_hf_weight_eff * float(face_refine_patch_hf_mul)
                face_patch_edge_weight_eff = face_patch_edge_weight_eff * float(face_refine_patch_edge_mul)
                face_feature_loss_weight_eff = face_feature_loss_weight_eff * float(face_refine_feature_weight_mul)
                face_feature_hf_weight_eff = face_feature_hf_weight_eff * float(face_refine_feature_hf_mul)
                face_feature_edge_weight_eff = face_feature_edge_weight_eff * float(face_refine_feature_edge_mul)
                face_feature_sigma_frac_eff = face_feature_sigma_frac_eff * float(face_refine_feature_sigma_mul)
                if int(face_refine_patch_size) > 0:
                    face_patch_size_eff = max(int(face_patch_size_eff), int(face_refine_patch_size))
            face_priority_core_active = False
            if face_priority_core_ids_np is not None and int(face_priority_core_ids_np.size) > 0:
                face_priority_core_active = bool(int(fj) in [int(x) for x in face_priority_core_ids_np.tolist()])
            face_priority_active = False
            if face_priority_ids_np is not None and int(face_priority_ids_np.size) > 0:
                face_priority_active = bool(int(fj) in [int(x) for x in face_priority_ids_np.tolist()])
            if face_priority_core_active and float(face_priority_core_frame_loss_mul) > 1.0:
                face_weight_eff = face_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_face_weight_scale)
                face_local_loss_weight_eff = face_local_loss_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_patch_loss_weight_eff = face_patch_loss_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_patch_hf_weight_eff = face_patch_hf_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_patch_edge_weight_eff = face_patch_edge_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_feature_loss_weight_eff = face_feature_loss_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_feature_hf_weight_eff = face_feature_hf_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                face_feature_edge_weight_eff = face_feature_edge_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_detail_target_scale)
                canonical_face_prior_weight_eff = canonical_face_prior_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_canonical_mul)
                canonical_face_prior_hf_weight_eff = canonical_face_prior_hf_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_canonical_mul)
                canonical_face_prior_edge_weight_eff = canonical_face_prior_edge_weight_eff * float(face_priority_core_frame_loss_mul) * float(face_priority_core_canonical_mul)
            elif face_priority_active and float(face_priority_frame_loss_mul) > 1.0:
                face_weight_eff = face_weight_eff * float(face_priority_frame_loss_mul)
                face_local_loss_weight_eff = face_local_loss_weight_eff * float(face_priority_frame_loss_mul)
                face_patch_loss_weight_eff = face_patch_loss_weight_eff * float(face_priority_frame_loss_mul)
                face_patch_hf_weight_eff = face_patch_hf_weight_eff * float(face_priority_frame_loss_mul)
                face_patch_edge_weight_eff = face_patch_edge_weight_eff * float(face_priority_frame_loss_mul)
                face_feature_loss_weight_eff = face_feature_loss_weight_eff * float(face_priority_frame_loss_mul)
                face_feature_hf_weight_eff = face_feature_hf_weight_eff * float(face_priority_frame_loss_mul)
                face_feature_edge_weight_eff = face_feature_edge_weight_eff * float(face_priority_frame_loss_mul)
                canonical_face_prior_weight_eff = canonical_face_prior_weight_eff * float(face_priority_frame_loss_mul)
                canonical_face_prior_hf_weight_eff = canonical_face_prior_hf_weight_eff * float(face_priority_frame_loss_mul)
                canonical_face_prior_edge_weight_eff = canonical_face_prior_edge_weight_eff * float(face_priority_frame_loss_mul)
            if bool(face_supervision_selected_only) and bool(face_hard_frame_select_enable) and face_selected_mask_np is not None and (not bool(face_selected_active)):
                face_weight_eff = 1.0
                face_local_loss_weight_eff = 0.0
                face_patch_loss_weight_eff = 0.0
                face_patch_hf_weight_eff = 0.0
                face_patch_edge_weight_eff = 0.0
                face_feature_loss_weight_eff = 0.0
                face_feature_hf_weight_eff = 0.0
                face_feature_edge_weight_eff = 0.0
            elif bool(face_detail_selected_only) and bool(face_hard_frame_select_enable) and face_selected_mask_np is not None and (not bool(face_selected_active)):
                face_local_loss_weight_eff = face_local_loss_weight_eff * float(face_nonselected_local_scale)
                face_patch_loss_weight_eff = 0.0
                face_patch_hf_weight_eff = 0.0
                face_patch_edge_weight_eff = 0.0
                face_feature_loss_weight_eff = 0.0
                face_feature_hf_weight_eff = 0.0
                face_feature_edge_weight_eff = 0.0
            if bool(face_lock_active):
                face_weight_eff = 0.0
                face_local_loss_weight_eff = 0.0
                face_patch_loss_weight_eff = 0.0
                face_patch_hf_weight_eff = 0.0
                face_patch_edge_weight_eff = 0.0
                face_feature_loss_weight_eff = 0.0
                face_feature_hf_weight_eff = 0.0
                face_feature_edge_weight_eff = 0.0
                canonical_face_prior_weight_eff = 0.0
                canonical_face_prior_hf_weight_eff = 0.0
                canonical_face_prior_edge_weight_eff = 0.0
            if (
                face_refine_phase
                and bool(face_refine_nonselected_face_exclude)
                and bool(face_hard_frame_select_enable)
                and face_selected_mask_np is not None
                and (not bool(face_selected_active))
                and face_boxes is not None
                and face_boxes[fj] is not None
            ):
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                face_exclude_mask = torch.zeros((1, H, W), device=device)
                face_exclude_mask[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = 1.0
                face_local_loss_weight_eff = 0.0
                face_patch_loss_weight_eff = 0.0
                face_patch_hf_weight_eff = 0.0
                face_patch_edge_weight_eff = 0.0
                face_feature_loss_weight_eff = 0.0
                face_feature_hf_weight_eff = 0.0
                face_feature_edge_weight_eff = 0.0
                canonical_face_prior_weight_eff = 0.0
                canonical_face_prior_hf_weight_eff = 0.0
                canonical_face_prior_edge_weight_eff = 0.0
            if (not bool(face_lock_active)) and face_boxes is not None and face_boxes[fj] is not None:
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                if face_priority_core_active and bool(face_priority_core_exclude_photo):
                    face_exclude_mask = torch.zeros((1, H, W), device=device)
                    face_exclude_mask[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = 1.0
                w2d = torch.ones((1, H, W), device=device)
                if bool(face_hard_frame_select_enable) and face_selected_mask_np is not None:
                    if bool(face_selected_mask_np[fj]):
                        face_roi_weight = max(float(face_weight_eff), 1.0)
                        if bool(face_align_quality_enable) and frame_align_scores_t is not None:
                            align_score_face = float(frame_align_scores_t[fj].item())
                            denom = max(float(face_align_score_max) - float(face_align_score_min), 1e-6)
                            align_norm = float(np.clip((align_score_face - float(face_align_score_min)) / denom, 0.0, 1.0))
                            align_norm = float(np.power(align_norm, float(face_align_score_power)))
                            face_roi_weight = face_roi_weight * (
                                float(face_align_weight_low)
                                + (float(face_align_weight_high) - float(face_align_weight_low)) * align_norm
                            )
                        if bool(face_frontal_quality_enable) and face_view_scores is not None:
                            face_view_norm = float(np.clip(float(face_view_scores[fj]), 0.0, 1.0))
                            face_view_norm = float(np.power(face_view_norm, float(face_frontal_power)))
                            face_roi_weight = face_roi_weight * (
                                float(face_frontal_weight_low)
                                + (float(face_frontal_weight_high) - float(face_frontal_weight_low)) * face_view_norm
                            )
                        if face_selected_weight_np is not None:
                            face_roi_weight = face_roi_weight * float(face_selected_weight_np[fj])
                        face_roi_weight = float(np.clip(face_roi_weight, 0.1, 8.0))
                        if abs(float(face_roi_weight) - 1.0) > 1e-6:
                            w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_roi_weight)
                    else:
                        w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_hard_reject_weight)
                        if bool(face_hard_exclusive_enable) or (bool(face_supervision_selected_only) and bool(face_supervision_exclude_nonselected)):
                            face_exclude_mask = torch.zeros((1, H, W), device=device)
                            face_exclude_mask[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = 1.0
                    weight = w2d if weight is None else (weight * w2d)
                else:
                    face_roi_weight = max(float(face_weight_eff), 1.0)
                    if bool(face_align_quality_enable) and frame_align_scores_t is not None:
                        align_score_face = float(frame_align_scores_t[fj].item())
                        denom = max(float(face_align_score_max) - float(face_align_score_min), 1e-6)
                        align_norm = float(np.clip((align_score_face - float(face_align_score_min)) / denom, 0.0, 1.0))
                        align_norm = float(np.power(align_norm, float(face_align_score_power)))
                        face_roi_weight = face_roi_weight * (
                            float(face_align_weight_low)
                            + (float(face_align_weight_high) - float(face_align_weight_low)) * align_norm
                        )
                    if bool(face_frontal_quality_enable) and face_view_scores is not None:
                        face_view_norm = float(np.clip(float(face_view_scores[fj]), 0.0, 1.0))
                        face_view_norm = float(np.power(face_view_norm, float(face_frontal_power)))
                        face_roi_weight = face_roi_weight * (
                            float(face_frontal_weight_low)
                            + (float(face_frontal_weight_high) - float(face_frontal_weight_low)) * face_view_norm
                        )
                    face_roi_weight = float(np.clip(face_roi_weight, 0.1, 4.0))
                    if abs(float(face_roi_weight) - 1.0) > 1e-6:
                        w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_roi_weight)
                        weight = w2d if weight is None else (weight * w2d)
            if hand_mask_curr is not None and float(hand_weight) > 1.0:
                h_w = torch.ones((1, H, W), device=device)
                h_w = h_w + (float(hand_weight) - 1.0) * hand_mask_curr.clamp(0.0, 1.0)
                weight = h_w if weight is None else (weight * h_w)

            if sep_fg_bg:
                region_mask = inner_supervision_mask
                bg_w_eff = bg_w
                if phase_tag == "phase_inner" and (not bool(inner_phase_keep_bg_weight)):
                    # Aggressive mode: ignore background during inner-only phase.
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
                g_render, K_curr, R, t, H, W, bg=bg_rgb, return_alpha=True, return_meta=True
            )
            diff_p1 = pred_rgb_p1 - gt
            residual_scalar_t = diff_p1.abs().mean().detach()
            residual_prev_t = frame_residual_ema[fj]
            residual_blend_t = (float(frame_residual_ema_alpha) * residual_prev_t) + ((1.0 - float(frame_residual_ema_alpha)) * residual_scalar_t)
            residual_ema_t = torch.where(frame_residual_ema_initialized[fj], residual_blend_t, residual_scalar_t)
            frame_residual_ema[fj] = residual_ema_t
            frame_residual_ema_initialized[fj] = True
            frame_w_scalar_t = torch.exp((-float(frame_weight_k)) * residual_ema_t)

            inner_only_phase = (training_mode == "inner_only") and (phase_tag == "phase_inner")
            shell_visibility_active = bool(shell_enable) and (inner_supervision_mask is not None or outer_supervision_mask is not None)
            outer_shell_runtime_active = bool(outer_supervision_mask is not None)

            person_mask_for_vis = person_mask_curr
            if shell_visibility_active:
                person_mask_for_vis = None
            visibility = compute_gaussian_visibility(
                xyz=xyz_render.detach(),
                K=K_curr,
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
            if shell_visibility_active:
                u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K_curr, R, t)
                inside_proj = valid_proj & (u_proj >= 0) & (u_proj <= float(W - 1)) & (v_proj >= 0) & (v_proj <= float(H - 1))
                layer_vis = torch.ones((n_gauss,), dtype=torch.float32, device=device)
                inner_sel = (g.shell_layer == 0)

                if inner_supervision_mask is not None:
                    inner_s = _sample_mask_at_projected_uv(inner_supervision_mask, u_proj, v_proj, inside_proj, H, W)
                    layer_vis[inner_sel] = layer_vis[inner_sel] * inner_s[inner_sel]

                if outer_shell_runtime_active:
                    outer_s = _sample_mask_at_projected_uv(outer_supervision_mask, u_proj, v_proj, inside_proj, H, W)
                    if face_gap_half_curr is not None and face_gap_scale_curr is not None and face_gap_support_curr is not None and g.bind_face_idx is not None:
                        face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long)
                        gap_valid = (face_idx_all >= 0) & (face_idx_all < int(face_gap_half_curr.shape[0]))
                        gap_half_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
                        gap_scale_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
                        gap_support_frame = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
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
                    if phase_tag == "phase_outer":
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
                    inside_support_th = float(np.clip(float(outer_inside_support_th), 0.0, 1.0))
                    outer_soft = outer_s.clamp(0.0, 1.0)
                    detached_vis_boost = float(max(outer_detached_vis_boost, 0.0))
                    if detached_vis_boost > 0.0:
                        outer_soft = torch.maximum(
                            outer_soft,
                            (outer_detached_s * detached_vis_boost).clamp(0.0, 1.0),
                        )
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
                            outer_boundary_violation_counter[safe] = 0.0
                            prune_cand = outer_sel_b & (outer_boundary_violation_counter >= float(max(int(outer_boundary_patience), 1))) & (outer_person_support_accum[-1] < 0.1 if len(outer_person_support_accum)>0 else torch.ones_like(outer_sel_b, dtype=torch.float32, device=device)>0)
                            if is_outer_geom_phase and bool(prune_cand.any()):
                                g.prune_state.data[prune_cand] = 2

                visibility = visibility * layer_vis

            if bool(use_border_attenuation) and person_mask_curr is not None and int(border_attenuation_width) > 0:
                if fj not in border_att_cache:
                    pm_np = (person_mask_curr[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    border_att_cache[fj] = torch.from_numpy(build_border_attenuation(pm_np, int(border_attenuation_width))).to(device=device, dtype=torch.float32)
                if u_proj is None or v_proj is None or valid_proj is None:
                    u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K_curr, R, t)
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
            if bool(inner_local_color_prior_enable):
                color_support_mask = inner_supervision_mask
                if color_support_mask is None:
                    color_support_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
                if color_support_mask is not None:
                    if u_proj is None or v_proj is None or valid_proj is None:
                        u_proj, v_proj, valid_proj = _project_points_torch(xyz_render.detach(), K_curr, R, t)
                    inside_color_proj = (
                        valid_proj
                        & (u_proj >= 0)
                        & (u_proj <= float(W - 1))
                        & (v_proj >= 0)
                        & (v_proj <= float(H - 1))
                    )
                    color_support = _sample_mask_at_projected_uv(
                        color_support_mask,
                        u_proj,
                        v_proj,
                        inside_color_proj,
                        H,
                        W,
                    )
                    inner_color_support_accum.append((color_support * visibility.detach()).clamp(0.0, 1.0))
            vis_accum.append(visibility.detach())

            # Densify stats (module hooks for policy).
            update_densify_stats(densify_stats, visibility.detach(), residual_scalar_t, float(tau_vis))

            # In inner-only runs, the visibility estimation pass is sufficient.
            # Re-running gsplat with per-gaussian weights roughly doubles the heaviest
            # kernel in the loop and has proven brittle on large shared-beta runs.
            use_pass1_render_directly = (training_mode == "inner_only") and (phase_tag == "phase_inner")
            if use_pass1_render_directly:
                pred_rgb, pred_a = pred_rgb_p1, pred_a_p1
            else:
                # Pass-2 render with per-gaussian visibility weighting.
                pred_rgb, pred_a = render_gaussians_3d(
                    g_render, K_curr, R, t, H, W, bg=bg_rgb, return_alpha=True, gaussian_weights=visibility.detach()
                )
            outer_residual_term = pred_rgb.new_tensor(0.0)
            outer_rgb_black: torch.Tensor | None = None
            outer_alpha_black: torch.Tensor | None = None
            need_outer_color_render = (
                is_outer_color_phase
                and g.shell_layer is not None
                and (
                    (outer_supervision_mask is not None and float(outer_phase_residual_weight) > 0.0)
                    or (bool(outer_color_masked_enable) and float(outer_color_masked_weight) > 0.0)
                )
            )
            if need_outer_color_render:
                vis_det = visibility.detach()
                outer_only_vis = vis_det.clone()
                outer_only_vis[g.shell_layer == 0] = 0.0
                black_bg = gt.new_zeros((3,))
                outer_rgb_black, outer_alpha_black = render_gaussians_3d(
                    g_render,
                    K_curr,
                    R,
                    t,
                    H,
                    W,
                    bg=black_bg,
                    return_alpha=True,
                    gaussian_weights=outer_only_vis,
                )
            if (
                is_outer_color_phase
                and outer_supervision_mask is not None
                and outer_rgb_black is not None
                and float(outer_phase_residual_weight) > 0.0
            ):
                outer_roi = outer_supervision_mask.clamp(0.0, 1.0)
                roi3 = outer_roi.expand_as(outer_rgb_black)
                residual_target = (gt_outer_target * roi3).clamp(0.0, 1.0)
                outer_residual_term = (outer_rgb_black - residual_target).abs()
                outer_residual_term = (outer_residual_term * roi3).sum() / (roi3.sum() + 1e-6)

            if phase_tag == "phase_inner":
                gt_phase_target = gt if bool(inner_phase_use_raw_gt_target) else gt_inner_target
            elif phase_tag == "phase_outer":
                gt_phase_target = gt_outer_target
            else:
                gt_phase_target = gt_combined_target

            if face_exclude_mask is not None:
                face_exclude_mask3 = face_exclude_mask.expand_as(pred_rgb)
                gt_phase_target = torch.where(face_exclude_mask3 > 0.5, pred_rgb.detach(), gt_phase_target)

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

            outer_color_masked_term = pred_rgb.new_tensor(0.0)
            outer_color_masked_valid = False
            if (
                is_outer_color_phase
                and bool(outer_color_masked_enable)
                and float(outer_color_masked_weight) > 0.0
                and outer_color_masked_raw_ids_t is not None
                and face_semantic_raw_t is not None
                and frame_data is not None
                and frame_data.verts_t is not None
                and frame_data.faces_t is not None
            ):
                sem_mask = None
                if int(fj) in outer_color_semantic_mask_cache:
                    sem_mask = outer_color_semantic_mask_cache[int(fj)].to(device=device, dtype=torch.float32)
                else:
                    verts_cam_np = (
                        frame_data.verts_t.detach().cpu().numpy().astype(np.float32)
                        @ R.detach().cpu().numpy().astype(np.float32).T
                    ) + t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
                    face_index_map, _ = _rasterize_visible_face_index_map(
                        verts_cam=verts_cam_np,
                        faces=frame_data.faces_t.detach().cpu().numpy().astype(np.int32),
                        K=K_curr.detach().cpu().numpy().astype(np.float32),
                        image_hw=(H, W),
                    )
                    raw_ids_np = face_semantic_raw_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
                    conf_np = (
                        face_semantic_conf_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
                        if face_semantic_conf_t is not None
                        else np.ones_like(raw_ids_np, dtype=np.float32)
                    )
                    allowed_ids_np = outer_color_masked_raw_ids_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
                    valid_face_np = (face_index_map >= 0) & (face_index_map < int(raw_ids_np.shape[0]))
                    allowed_np = np.zeros((H, W), dtype=np.float32)
                    if bool(valid_face_np.any()) and int(allowed_ids_np.size) > 0:
                        face_ids_px = face_index_map[valid_face_np]
                        raw_px = raw_ids_np[face_ids_px]
                        conf_px = conf_np[face_ids_px]
                        keep_px = np.isin(raw_px, allowed_ids_np) & (conf_px >= float(outer_color_masked_conf_min))
                        allowed_np[valid_face_np] = keep_px.astype(np.float32)
                    sem_mask = torch.from_numpy(allowed_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                    outer_color_semantic_mask_cache[int(fj)] = sem_mask.detach().to(dtype=torch.float16)
                color_mask = sem_mask.clamp(0.0, 1.0)
                if outer_supervision_mask is not None:
                    color_mask = (color_mask * outer_supervision_mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)
                elif person_mask_curr is not None:
                    color_mask = (color_mask * person_mask_curr.clamp(0.0, 1.0)).clamp(0.0, 1.0)
                if bool(outer_color_masked_use_outer_alpha) and outer_alpha_black is not None:
                    alpha_gate = (outer_alpha_black.detach() / max(float(outer_color_masked_outer_alpha_min), 1e-6)).clamp(0.0, 1.0)
                    color_mask = (color_mask * alpha_gate).clamp(0.0, 1.0)
                if bool(outer_color_masked_exclude_hands) and hand_mask_curr is not None:
                    color_mask = (color_mask * (1.0 - hand_mask_curr.clamp(0.0, 1.0))).clamp(0.0, 1.0)
                if bool(outer_color_masked_exclude_face_box) and face_boxes is not None and face_boxes[fj] is not None:
                    fx0, fy0, fx1, fy1 = face_boxes[fj]
                    x0 = max(0, int(fx0))
                    y0 = max(0, int(fy0))
                    x1 = min(W - 1, int(fx1))
                    y1 = min(H - 1, int(fy1))
                    if x1 >= x0 and y1 >= y0:
                        color_mask = color_mask.clone()
                        color_mask[:, y0:y1 + 1, x0:x1 + 1] = 0.0
                if float(color_mask.sum().detach().cpu()) > 1.0:
                    mask3 = color_mask.expand_as(pred_rgb)
                    outer_color_masked_term = ((pred_rgb - gt).abs() * mask3).sum() / (mask3.sum() + 1e-6)
                    outer_color_masked_valid = True
            if bool(outer_color_masked_replace_global) and outer_color_masked_valid:
                photo = photo * 0.0
                l1 = l1 * 0.0

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

            face_local_term = pred_rgb.new_tensor(0.0)
            face_local_ok = True
            if face_view_scores is not None and 0 <= fj < len(face_view_scores):
                face_local_ok = float(face_view_scores[fj]) >= float(face_local_view_min)
            if (
                float(face_local_loss_weight_eff) > 0.0
                and face_local_ok
                and face_boxes is not None
                and face_boxes[fj] is not None
            ):
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                shrink_scale = 1.0 - float(face_local_box_shrink)
                cx = 0.5 * (float(fx0) + float(fx1))
                cy = 0.5 * (float(fy0) + float(fy1))
                half_w = 0.5 * max(float(fx1) - float(fx0), 1.0) * shrink_scale
                half_h = 0.5 * max(float(fy1) - float(fy0), 1.0) * shrink_scale
                x0 = max(0, int(math.floor(cx - half_w)))
                y0 = max(0, int(math.floor(cy - half_h)))
                x1 = min(W - 1, int(math.ceil(cx + half_w)))
                y1 = min(H - 1, int(math.ceil(cy + half_h)))
                if x1 >= x0 and y1 >= y0:
                    face_local_mask = torch.zeros((1, H, W), dtype=torch.float32, device=device)
                    if bool(face_local_core_enable):
                        yy = torch.arange(y0, y1 + 1, device=device, dtype=torch.float32).view(-1, 1)
                        xx = torch.arange(x0, x1 + 1, device=device, dtype=torch.float32).view(1, -1)
                        box_w = max(float(x1 - x0 + 1), 1.0)
                        box_h = max(float(y1 - y0 + 1), 1.0)
                        cx_local = 0.5 * (float(x0) + float(x1))
                        cy_local = 0.5 * (float(y0) + float(y1)) - 0.06 * box_h
                        rx = max(0.34 * box_w, 1.0)
                        ry = max(0.42 * box_h, 1.0)
                        norm = ((xx - cx_local) / rx).pow(2) + ((yy - cy_local) / ry).pow(2)
                        face_local_mask[:, y0:y1 + 1, x0:x1 + 1] = (norm <= 1.0).to(torch.float32).unsqueeze(0)
                    else:
                        face_local_mask[:, y0:y1 + 1, x0:x1 + 1] = 1.0
                    local_support_mask = inner_supervision_mask
                    if local_support_mask is None:
                        local_support_mask = person_mask_curr
                    if local_support_mask is None:
                        local_support_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
                    if local_support_mask is not None:
                        face_local_mask = (face_local_mask * local_support_mask).clamp(0.0, 1.0)
                    face_local_mask3 = face_local_mask.expand_as(l1_map)
                    face_local_term = (l1_map * face_local_mask3).sum() / (face_local_mask3.sum() + 1e-6)

            face_surface_term = pred_rgb.new_tensor(0.0)
            face_patch_term = pred_rgb.new_tensor(0.0)
            face_patch_hf_term = pred_rgb.new_tensor(0.0)
            face_patch_edge_term = pred_rgb.new_tensor(0.0)
            face_feature_term = pred_rgb.new_tensor(0.0)
            face_feature_hf_term = pred_rgb.new_tensor(0.0)
            face_feature_edge_term = pred_rgb.new_tensor(0.0)
            canonical_face_prior_term = pred_rgb.new_tensor(0.0)
            canonical_face_prior_hf_term = pred_rgb.new_tensor(0.0)
            canonical_face_prior_edge_term = pred_rgb.new_tensor(0.0)
            canonical_face_prior_symmetry_term = pred_rgb.new_tensor(0.0)
            face_patch_ok = True
            if face_view_scores is not None and 0 <= fj < len(face_view_scores):
                face_patch_ok = float(face_view_scores[fj]) >= float(face_patch_view_min)
            face_feature_ok = True
            if face_view_scores is not None and 0 <= fj < len(face_view_scores):
                face_feature_ok = float(face_view_scores[fj]) >= float(face_feature_view_min)
            strict_face_teacher_surface_mode = (
                int(nF) <= 2
                and bool(face_selected_active)
                and bool(face_supervision_selected_only)
                and bool(face_detail_selected_only)
                and face_boxes is not None
                and face_boxes[fj] is not None
                and g.effective_face_idx is not None
            )
            if strict_face_teacher_surface_mode:
                teacher_face_gaussian_mask = (g.shell_layer == 0) & (g.prune_state != 2)
                if face_semantic_focus_ids_t is not None and face_semantic_raw_t is not None:
                    face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
                    raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
                    face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
                    semantic_face_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                    if bool(face_valid_all.any()):
                        semantic_face_gauss[face_valid_all] = torch.isin(
                            raw_sem_all[face_ids_all[face_valid_all]],
                            face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                        )
                    teacher_face_gaussian_mask = teacher_face_gaussian_mask & semantic_face_gauss
                if bool(teacher_face_gaussian_mask.any()):
                    fx0, fy0, fx1, fy1 = face_boxes[fj]
                    face_surface_term = _compute_face_surface_color_term(
                        g=g,
                        xyz_render=xyz_render,
                        K=K_curr,
                        R=R,
                        t=t,
                        gt_rgb=gt,
                        visibility=visibility if visibility is not None else torch.ones((n_gauss,), dtype=torch.float32, device=device),
                        face_box_xyxy=(int(fx0), int(fy0), int(fx1), int(fy1)),
                        feature_points=face_feature_points_frames[fj] if (face_feature_points_frames is not None and 0 <= fj < len(face_feature_points_frames)) else None,
                        face_gaussian_mask=teacher_face_gaussian_mask,
                        visibility_min=float(face_texture_bake_visibility_min),
                        feature_sigma_frac=float(face_feature_sigma_frac_eff),
                        sharpen_amount=float(max(face_feature_sharpen_amount_eff, face_texture_bake_sharpen_amount)),
                    )
            if (
                (
                    float(face_patch_loss_weight_eff) > 0.0
                    or float(face_patch_hf_weight_eff) > 0.0
                    or float(face_patch_edge_weight_eff) > 0.0
                    or float(face_feature_loss_weight_eff) > 0.0
                    or float(face_feature_hf_weight_eff) > 0.0
                    or float(face_feature_edge_weight_eff) > 0.0
                )
                and (face_patch_ok or face_feature_ok)
                and face_boxes is not None
                and face_boxes[fj] is not None
            ):
                fx0, fy0, fx1, fy1 = face_boxes[fj]
                shrink_scale = 1.0 - float(face_local_box_shrink)
                cx = 0.5 * (float(fx0) + float(fx1))
                cy = 0.5 * (float(fy0) + float(fy1))
                half_w = 0.5 * max(float(fx1) - float(fx0), 1.0) * shrink_scale
                half_h = 0.5 * max(float(fy1) - float(fy0), 1.0) * shrink_scale
                x0 = max(0, int(math.floor(cx - half_w)))
                y0 = max(0, int(math.floor(cy - half_h)))
                x1 = min(W - 1, int(math.ceil(cx + half_w)))
                y1 = min(H - 1, int(math.ceil(cy + half_h)))
                if x1 >= x0 and y1 >= y0:
                    patch_size_hw = (int(face_patch_size_eff), int(face_patch_size_eff))
                    pred_patch_up, _ = _render_box_resized(
                        g=g_render,
                        K=K_curr,
                        R=R,
                        t=t,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        out_h=patch_size_hw[0],
                        out_w=patch_size_hw[1],
                        bg_rgb=bg_rgb,
                        gaussian_weights=visibility.detach() if visibility is not None else None,
                    )
                    gt_patch = gt_phase_target[:, y0:y1 + 1, x0:x1 + 1]
                    gt_patch_up = F.interpolate(gt_patch.unsqueeze(0), size=patch_size_hw, mode='bilinear', align_corners=False).squeeze(0)
                    gt_patch_raw_up = None
                    if bool(face_patch_use_raw_gt) or bool(face_feature_use_raw_gt):
                        gt_raw_patch = gt[:, y0:y1 + 1, x0:x1 + 1]
                        gt_patch_raw_up = F.interpolate(gt_raw_patch.unsqueeze(0), size=patch_size_hw, mode='bilinear', align_corners=False).squeeze(0)
                    patch_mask = None
                    local_support_mask = inner_supervision_mask
                    if local_support_mask is None:
                        local_support_mask = person_mask_curr
                    if local_support_mask is None:
                        local_support_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
                    if local_support_mask is not None:
                        patch_mask = local_support_mask[:, y0:y1 + 1, x0:x1 + 1].clamp(0.0, 1.0)
                    patch_mask_up = None
                    if patch_mask is not None:
                        patch_mask_up = F.interpolate(patch_mask.unsqueeze(0), size=patch_size_hw, mode='bilinear', align_corners=False).squeeze(0).clamp(0.0, 1.0)
                    if bool(face_local_core_enable):
                        ph = int(pred_patch_up.shape[-2])
                        pw = int(pred_patch_up.shape[-1])
                        yy = torch.arange(ph, device=device, dtype=torch.float32).view(-1, 1)
                        xx = torch.arange(pw, device=device, dtype=torch.float32).view(1, -1)
                        cx_patch = 0.5 * float(max(pw - 1, 1))
                        cy_patch = 0.5 * float(max(ph - 1, 1)) - 0.06 * float(ph)
                        rx = max(0.34 * float(pw), 1.0)
                        ry = max(0.42 * float(ph), 1.0)
                        core_mask = ((((xx - cx_patch) / rx).pow(2) + ((yy - cy_patch) / ry).pow(2)) <= 1.0).to(torch.float32).unsqueeze(0)
                        patch_mask_up = core_mask if patch_mask_up is None else (patch_mask_up * core_mask).clamp(0.0, 1.0)
                    face_feature_mask_up = None
                    if (
                        face_feature_points_frames is not None
                        and 0 <= fj < len(face_feature_points_frames)
                        and face_feature_points_frames[fj] is not None
                    ):
                        face_feature_mask_up = _build_face_feature_mask_from_points(
                            patch_h=int(pred_patch_up.shape[-2]),
                            patch_w=int(pred_patch_up.shape[-1]),
                            box_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                            feature_points=face_feature_points_frames[fj],
                            sigma_frac=float(face_feature_sigma_frac_eff),
                            device=device,
                        )
                        if face_feature_mask_up is not None:
                            face_feature_mask_up = face_feature_mask_up if patch_mask_up is None else (face_feature_mask_up * patch_mask_up).clamp(0.0, 1.0)
                    gt_patch_supervise = gt_patch_raw_up if (bool(face_patch_use_raw_gt) and gt_patch_raw_up is not None) else gt_patch_up
                    face_patch_term = _masked_multiscale_l1_chw(
                        pred_patch_up,
                        gt_patch_supervise,
                        patch_mask_up,
                        levels=2,
                    )
                    if float(face_patch_hf_weight_eff) > 0.0:
                        face_patch_pred_hf = _image_laplacian_chw(pred_patch_up)
                        face_patch_gt_hf = _image_laplacian_chw(gt_patch_supervise)
                        face_patch_hf_err = (face_patch_pred_hf - face_patch_gt_hf).abs()
                        if patch_mask_up is not None:
                            patch_mask_up3 = patch_mask_up.expand_as(face_patch_hf_err)
                            face_patch_hf_term = (face_patch_hf_err * patch_mask_up3).sum() / (patch_mask_up3.sum() + 1e-6)
                        else:
                            face_patch_hf_term = face_patch_hf_err.mean()
                    if float(face_patch_edge_weight_eff) > 0.0:
                        face_patch_pred_edge = _compute_image_edge_strength_torch(pred_patch_up)
                        face_patch_gt_edge = _compute_image_edge_strength_torch(gt_patch_supervise)
                        face_patch_edge_err = (face_patch_pred_edge - face_patch_gt_edge).abs()
                        edge_mask = patch_mask_up
                        if edge_mask is None:
                            edge_mask = torch.ones_like(face_patch_edge_err)
                        edge_focus = (0.35 + 0.65 * face_patch_gt_edge).clamp(0.0, 1.5)
                        edge_weight = (edge_mask * edge_focus).clamp(min=0.0)
                        face_patch_edge_term = (face_patch_edge_err * edge_weight).sum() / (edge_weight.sum() + 1e-6)
                    if face_feature_mask_up is not None:
                        gt_patch_feature = gt_patch_raw_up if (bool(face_feature_use_raw_gt) and gt_patch_raw_up is not None) else gt_patch_up
                        if float(face_feature_sharpen_amount_eff) > 0.0:
                            blur = F.avg_pool2d(gt_patch_up.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
                            gt_patch_feature = (gt_patch_up + float(face_feature_sharpen_amount_eff) * (gt_patch_up - blur)).clamp(0.0, 1.0)
                        if float(face_feature_loss_weight_eff) > 0.0 and face_feature_ok:
                            face_feature_term = _masked_multiscale_l1_chw(
                                pred_patch_up,
                                gt_patch_feature,
                                face_feature_mask_up,
                                levels=2,
                            )
                        if float(face_feature_hf_weight_eff) > 0.0 and face_feature_ok:
                            feat_pred_hf = _image_laplacian_chw(pred_patch_up)
                            feat_gt_hf = _image_laplacian_chw(gt_patch_feature)
                            feat_hf_err = (feat_pred_hf - feat_gt_hf).abs()
                            feat_mask3 = face_feature_mask_up.expand_as(feat_hf_err)
                            face_feature_hf_term = (feat_hf_err * feat_mask3).sum() / (feat_mask3.sum() + 1e-6)
                        if float(face_feature_edge_weight_eff) > 0.0 and face_feature_ok:
                            feat_pred_edge = _compute_image_edge_strength_torch(pred_patch_up)
                            feat_gt_edge = _compute_image_edge_strength_torch(gt_patch_feature)
                            feat_edge_err = (feat_pred_edge - feat_gt_edge).abs()
                            feat_focus = (face_feature_mask_up * (0.35 + 0.65 * feat_gt_edge).clamp(0.0, 1.5)).clamp(min=0.0)
                            face_feature_edge_term = (feat_edge_err * feat_focus).sum() / (feat_focus.sum() + 1e-6)
                    if (
                        canonical_face_prior_template is not None
                        and float(canonical_face_prior_weight_eff) > 0.0
                        and (not bool(canonical_face_prior_selected_only) or bool(face_selected_active))
                        and ((not bool(canonical_face_prior_refine_only)) or bool(face_refine_phase))
                        and float(face_view_scores[fj]) >= float(canonical_face_prior_view_min) if (face_view_scores is not None and 0 <= fj < len(face_view_scores)) else True
                        and face_feature_points_frames is not None
                        and 0 <= fj < len(face_feature_points_frames)
                        and face_feature_points_frames[fj] is not None
                    ):
                        canonical_pred, canonical_valid = _warp_face_patch_to_canonical(
                            pred_patch_up,
                            box_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                            feature_points=face_feature_points_frames[fj],
                            out_size=int(canonical_face_prior_size),
                        )
                        if canonical_pred is not None and canonical_valid is not None:
                            canonical_mask = canonical_face_prior_mask if canonical_face_prior_mask is not None else canonical_valid
                            canonical_mask = (canonical_mask * canonical_valid).clamp(0.0, 1.0)
                            canonical_feat_mask = canonical_face_prior_feature_mask if canonical_face_prior_feature_mask is not None else canonical_mask
                            canonical_feat_mask = (canonical_feat_mask * canonical_mask).clamp(0.0, 1.0)
                            canonical_face_prior_term = _masked_multiscale_l1_chw(
                                canonical_pred,
                                canonical_face_prior_template,
                                canonical_feat_mask,
                                levels=2,
                            )
                            if float(canonical_face_prior_hf_weight_eff) > 0.0:
                                cp_pred_hf = _image_laplacian_chw(canonical_pred)
                                cp_gt_hf = _image_laplacian_chw(canonical_face_prior_template)
                                cp_hf_err = (cp_pred_hf - cp_gt_hf).abs()
                                cp_mask3 = canonical_feat_mask.expand_as(cp_hf_err)
                                canonical_face_prior_hf_term = (cp_hf_err * cp_mask3).sum() / (cp_mask3.sum() + 1e-6)
                            if float(canonical_face_prior_edge_weight_eff) > 0.0:
                                cp_pred_edge = _compute_image_edge_strength_torch(canonical_pred)
                                cp_gt_edge = _compute_image_edge_strength_torch(canonical_face_prior_template)
                                cp_edge_err = (cp_pred_edge - cp_gt_edge).abs()
                                cp_focus = (canonical_feat_mask * (0.35 + 0.65 * cp_gt_edge).clamp(0.0, 1.5)).clamp(min=0.0)
                                canonical_face_prior_edge_term = (cp_edge_err * cp_focus).sum() / (cp_focus.sum() + 1e-6)
                            if float(canonical_face_prior_symmetry_weight) > 0.0:
                                sym_pred = torch.flip(canonical_pred, dims=[2])
                                sym_mask = canonical_feat_mask
                                sym_err = (canonical_pred - sym_pred).abs()
                                sym_mask3 = sym_mask.expand_as(sym_err)
                                canonical_face_prior_symmetry_term = (sym_err * sym_mask3).sum() / (sym_mask3.sum() + 1e-6)
                    if (
                        bool(face_texture_bake_enable)
                        and bool(face_refine_phase)
                        and (int(it + 1) >= int(face_texture_bake_start_iter))
                        and (int(face_texture_bake_every) > 0)
                        and (((int(it) + 1) % int(face_texture_bake_every)) == 0)
                    ):
                        face_bake_score = float(face_view_scores[fj]) if (face_view_scores is not None and 0 <= fj < len(face_view_scores)) else 0.0
                        face_bake_ok = face_bake_score >= float(face_texture_bake_view_min)
                        if bool(face_texture_bake_selected_only) and (not bool(face_selected_active)):
                            face_bake_ok = False
                        if face_bake_ok:
                            face_bake_gt = gt if bool(face_texture_bake_use_raw_gt) else gt_phase_target
                            if (best_face_bake_state is None) or (face_bake_score > float(best_face_bake_state['score'])):
                                best_face_bake_state = {
                                    'score': float(face_bake_score),
                                    'xyz_render': xyz_render.detach().clone(),
                                    'visibility': visibility.detach().clone(),
                                    'K': K_curr.detach().clone(),
                                    'R': R.detach().clone(),
                                    't': t.detach().clone(),
                                    'gt_rgb': face_bake_gt.detach().clone(),
                                    'face_box_xyxy': (int(x0), int(y0), int(x1), int(y1)),
                                    'feature_points': dict(face_feature_points_frames[fj]) if (face_feature_points_frames is not None and 0 <= fj < len(face_feature_points_frames) and face_feature_points_frames[fj] is not None) else None,
                                }

            hand_local_term = pred_rgb.new_tensor(0.0)
            if float(hand_local_loss_weight) > 0.0 and hand_local_mask_curr is not None:
                hand_local_mask = hand_local_mask_curr.clamp(0.0, 1.0)
                local_support_mask = inner_supervision_mask
                if local_support_mask is None:
                    local_support_mask = person_mask_curr
                if local_support_mask is not None:
                    hand_local_mask = (hand_local_mask * local_support_mask).clamp(0.0, 1.0)
                hand_local_mask3 = hand_local_mask.expand_as(l1_map)
                hand_local_term = (l1_map * hand_local_mask3).sum() / (hand_local_mask3.sum() + 1e-6)

            face_refine_body_preserve_term = pred_rgb.new_tensor(0.0)
            if (
                face_refine_phase
                and bool(face_refine_body_preserve_enable)
                and float(face_refine_body_preserve_weight) > 0.0
            ):
                body_preserve_mask = person_mask_curr
                if body_preserve_mask is None:
                    body_preserve_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
                if body_preserve_mask is None:
                    body_preserve_mask = outer_supervision_mask
                if body_preserve_mask is not None:
                    body_preserve_mask = body_preserve_mask.clamp(0.0, 1.0)
                    if bool(face_refine_body_preserve_exclude_face) and face_boxes is not None and face_boxes[fj] is not None:
                        fx0, fy0, fx1, fy1 = face_boxes[fj]
                        x0 = max(0, int(fx0))
                        y0 = max(0, int(fy0))
                        x1 = min(W - 1, int(fx1))
                        y1 = min(H - 1, int(fy1))
                        body_preserve_mask = body_preserve_mask.clone()
                        body_preserve_mask[:, y0:y1 + 1, x0:x1 + 1] = 0.0
                    body_pts = torch.nonzero(body_preserve_mask[0] > 0.5, as_tuple=False)
                    if int(body_pts.shape[0]) > 0:
                        y_min = int(body_pts[:, 0].min().item())
                        y_max = int(body_pts[:, 0].max().item())
                        x_min = int(body_pts[:, 1].min().item())
                        x_max = int(body_pts[:, 1].max().item())
                        band_y0 = int(np.clip(y_min + int(round(0.18 * max(y_max - y_min + 1, 1))), 0, H - 1))
                        band_y1 = int(np.clip(y_min + int(round(0.76 * max(y_max - y_min + 1, 1))), 0, H - 1))
                        band_x0 = int(np.clip(x_min + int(round(0.06 * max(x_max - x_min + 1, 1))), 0, W - 1))
                        band_x1 = int(np.clip(x_max - int(round(0.06 * max(x_max - x_min + 1, 1))), 0, W - 1))
                        if band_y1 >= band_y0 and band_x1 >= band_x0:
                            torso_band = torch.zeros_like(body_preserve_mask)
                            torso_band[:, band_y0:band_y1 + 1, band_x0:band_x1 + 1] = 1.0
                            body_preserve_mask = (body_preserve_mask * torso_band).clamp(0.0, 1.0)
                    face_refine_body_preserve_term = _masked_multiscale_l1_chw(
                        pred_rgb,
                        gt_phase_target,
                        body_preserve_mask,
                        levels=2,
                    )

            base_frame_loss = (
                photo
                + float(l1_weight) * l1
                + float(dq_outer_color_band_weight) * outer_band_term
                + float(outer_phase_residual_weight) * outer_residual_term
                + float(outer_color_masked_weight) * outer_color_masked_term
            )
            if bool(ctx_face_only):
                base_frame_loss = base_frame_loss * 0.0
            strict_face_surface_fit_loss_only = bool(strict_face_teacher_surface_mode) and bool(ctx_face_only)
            if strict_face_surface_fit_loss_only:
                local_frame_loss = (
                    max(float(face_patch_loss_weight_eff), 1.0) * face_surface_term
                    + float(face_patch_loss_weight_eff) * face_patch_term
                    + float(face_patch_hf_weight_eff) * face_patch_hf_term
                    + float(face_patch_edge_weight_eff) * face_patch_edge_term
                    + float(face_feature_loss_weight_eff) * face_feature_term
                    + float(face_feature_hf_weight_eff) * face_feature_hf_term
                    + float(face_feature_edge_weight_eff) * face_feature_edge_term
                )
            else:
                local_frame_loss = (
                    float(face_local_loss_weight_eff) * face_local_term
                    + float(face_patch_loss_weight_eff) * face_patch_term
                    + float(face_patch_hf_weight_eff) * face_patch_hf_term
                    + float(face_patch_edge_weight_eff) * face_patch_edge_term
                    + float(face_feature_loss_weight_eff) * face_feature_term
                    + float(face_feature_hf_weight_eff) * face_feature_hf_term
                    + float(face_feature_edge_weight_eff) * face_feature_edge_term
                    + float(canonical_face_prior_weight_eff) * canonical_face_prior_term
                    + float(canonical_face_prior_hf_weight_eff) * canonical_face_prior_hf_term
                    + float(canonical_face_prior_edge_weight_eff) * canonical_face_prior_edge_term
                    + float(canonical_face_prior_symmetry_weight) * canonical_face_prior_symmetry_term
                    + float(hand_local_loss_weight) * hand_local_term
                    + float(face_refine_body_preserve_weight) * face_refine_body_preserve_term
                )
            if face_refine_phase:
                base_frame_loss = base_frame_loss * float(face_refine_global_loss_scale)
            frame_loss = base_frame_loss + local_frame_loss

            if (not bool(ctx_face_only)) and outer_supervision_mask is not None and float(outer_silhouette_weight) > 0.0:
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
                sil_term = float(outer_silhouette_weight) * sil_loss
                if face_refine_phase:
                    sil_term = sil_term * float(face_refine_global_loss_scale)
                frame_loss = frame_loss + sil_term

            if (not bool(ctx_face_only)) and float(detail_hf_weight) > 0.0:
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
                hf_term = float(detail_hf_weight) * hf_loss
                if face_refine_phase:
                    hf_term = hf_term * float(face_refine_global_loss_scale)
                frame_loss = frame_loss + hf_term

            if (not bool(ctx_face_only)) and float(detail_ms_weight) > 0.0:
                ms_w = inner_supervision_mask
                if ms_w is None:
                    ms_w = person_mask_curr
                if ms_w is None:
                    ms_w = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
                if ms_w is None:
                    ms_w = outer_supervision_mask
                if ms_w is not None and bool(detail_ms_exclude_face) and face_boxes is not None and face_boxes[fj] is not None:
                    fx0, fy0, fx1, fy1 = face_boxes[fj]
                    x0 = max(0, int(fx0))
                    y0 = max(0, int(fy0))
                    x1 = min(W - 1, int(fx1))
                    y1 = min(H - 1, int(fy1))
                    ms_w = ms_w.clone()
                    ms_w[:, y0:y1 + 1, x0:x1 + 1] = 0.0
                ms_loss = _masked_multiscale_l1_chw(
                    pred_rgb,
                    gt_phase_target,
                    ms_w,
                    levels=int(detail_ms_levels),
                )
                ms_term = float(detail_ms_weight) * ms_loss
                if face_refine_phase:
                    ms_term = ms_term * float(face_refine_global_loss_scale)
                frame_loss = frame_loss + ms_term

            leak_mask = inner_supervision_mask
            if leak_mask is None:
                leak_mask = person_mask_curr
            if leak_mask is None:
                leak_mask = smpl_mask_for_color if smpl_mask_for_color is not None else smpl_mask
            if leak_mask is None:
                leak_mask = outer_supervision_mask
            if (not bool(ctx_face_only)) and leak_mask is not None and float(smpl_alpha_leak_weight) > 0.0:
                outside = (1.0 - leak_mask).clamp(0.0, 1.0)
                leak = (pred_a * outside).sum() / (outside.sum() + 1e-6)
                leak_term = float(smpl_alpha_leak_weight) * leak
                if face_refine_phase:
                    leak_term = leak_term * float(face_refine_global_loss_scale)
                frame_loss = frame_loss + leak_term
            if (
                (not bool(ctx_face_only))
                and person_contour_band_frames is not None
                and person_mask_curr is not None
                and person_contour_band_frames[fj] is not None
                and float(person_mask_contour_leak_weight) > 0.0
            ):
                contour_band_curr = person_contour_band_frames[fj]
                outside_band = (contour_band_curr * (1.0 - person_mask_curr)).clamp(0.0, 1.0)
                leak_band = (pred_a * outside_band).sum() / (outside_band.sum() + 1e-6)
                leak_band_term = float(person_mask_contour_leak_weight) * leak_band
                if face_refine_phase:
                    leak_band_term = leak_band_term * float(face_refine_global_loss_scale)
                frame_loss = frame_loss + leak_band_term

            if frame_loss_weights_t is not None:
                frame_loss = frame_loss * frame_loss_weights_t[fj]
            frame_loss = float(wj) * frame_w_scalar_t * frame_loss

            frame_loss_detached = frame_loss.detach()
            loss_value_t = frame_loss_detached if loss_value_t is None else (loss_value_t + frame_loss_detached)
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
                aniso_loss_detached = aniso_loss.detach()
                loss_value_t = aniso_loss_detached if loss_value_t is None else (loss_value_t + aniso_loss_detached)
                aniso_loss.backward()

            if (
                binding_enabled
                and frame_data is not None
                and bind_mask_curr is not None
                and bool(bind_mask_curr.any())
                and inner_local_scale_cap_weight > 0.0
                and g.shell_layer is not None
                and g.effective_face_idx is not None
                and frame_data.face_s1_t is not None
                and frame_data.face_s2_t is not None
            ):
                inner_face_mask = (
                    bind_mask_curr
                    & (g.shell_layer == 0)
                    & (g.effective_face_idx >= 0)
                    & (g.effective_face_idx < int(frame_data.face_s1_t.shape[0]))
                )
                if bool(inner_face_mask.any()):
                    face_ids = g.effective_face_idx[inner_face_mask].to(device=device, dtype=torch.long)
                    scales_inner = g.scales()[inner_face_mask]
                    tang_scales = torch.sort(scales_inner[:, :2], dim=1, descending=True).values
                    local_cap = torch.stack(
                        [
                            float(inner_local_scale_cap_ratio) * frame_data.face_s1_t.index_select(0, face_ids),
                            float(inner_local_scale_cap_ratio) * frame_data.face_s2_t.index_select(0, face_ids),
                        ],
                        dim=1,
                    )
                    local_cap = torch.sort(local_cap, dim=1, descending=True).values.clamp(min=1e-6)
                    normal_cap = (
                        float(inner_local_scale_cap_normal_ratio)
                        * torch.minimum(local_cap[:, 0], local_cap[:, 1])
                    ).clamp(min=1e-6)
                    pen_t = F.relu(tang_scales - local_cap)
                    pen_n = F.relu(scales_inner[:, 2] - normal_cap)
                    inner_local_scale_loss = float(inner_local_scale_cap_weight) * float(wj) * (
                        (pen_t * pen_t).mean() + (pen_n * pen_n).mean()
                    )
                    inner_local_scale_loss_detached = inner_local_scale_loss.detach()
                    loss_value_t = (
                        inner_local_scale_loss_detached
                        if loss_value_t is None
                        else (loss_value_t + inner_local_scale_loss_detached)
                    )
                    inner_local_scale_loss.backward()

        vis_mean = torch.stack(vis_accum, dim=0).mean(dim=0) if len(vis_accum) > 0 else torch.ones((n_gauss,), device=device)
        has_inner_color_support = len(inner_color_support_accum) > 0
        inner_color_support_mean = (
            torch.stack(inner_color_support_accum, dim=0).mean(dim=0)
            if has_inner_color_support
            else torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        )
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
        outer_face_contrib_count = None
        outer_face_support_prob = None
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
                if outer_face_contrib_frames is not None:
                    outer_face_contrib_count = torch.as_tensor(
                        [len(v) for v in outer_face_contrib_frames],
                        device=device,
                        dtype=torch.float32,
                    )
                    outer_face_support_prob = outer_face_contrib_count / float(max(nF, 1))
        # Keep all policy vectors strictly 1D to avoid accidental broadcasting (NxN).
        vis_mean = vis_mean.view(-1)
        inner_color_support_mean = inner_color_support_mean.view(-1)
        edge_mean = edge_mean.view(-1)
        outer_head_support_mean = outer_head_support_mean.view(-1)
        outer_inside_support_mean = outer_inside_support_mean.view(-1)
        outer_person_support_mean = outer_person_support_mean.view(-1)
        outer_hand_support_mean = outer_hand_support_mean.view(-1)
        outer_align_peak_mean = torch.zeros((n_gauss,), dtype=torch.float32, device=device)
        outer_align_protect_mask = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
        outer_persist_protect_mask = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
        if (
            bool(shell_enable)
            and bool(outer_align_prune_protect_enable)
            and g.bind_face_idx is not None
            and outer_face_align_peak is not None
        ):
            face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
            face_valid = (
                (face_idx_all >= 0)
                & (face_idx_all < int(outer_face_align_peak.shape[0]))
            )
            if bool(face_valid.any()):
                outer_align_peak_mean[face_valid] = outer_face_align_peak.index_select(
                    0,
                    face_idx_all[face_valid],
                )
            align_candidate = (
                (g.shell_layer == 1)
                & (g.prune_state != 2)
                & (outer_align_peak_mean >= float(outer_align_prune_protect_min_iou))
            )
            severe_bad = (
                (vis_mean.detach() <= float(outer_align_prune_protect_min_vis))
                & (outer_drive_support_mean.detach() <= float(outer_align_prune_protect_min_support))
            )
            outer_align_protect_mask = align_candidate & (~severe_bad)
        if (
            bool(shell_enable)
            and bool(outer_prune_persist_protect_enable)
            and g.bind_face_idx is not None
            and outer_face_weight_sum is not None
            and outer_face_contrib_count is not None
            and outer_face_support_prob is not None
        ):
            face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
            face_valid = (
                (face_idx_all >= 0)
                & (face_idx_all < int(outer_face_weight_sum.shape[0]))
            )
            if bool(face_valid.any()):
                face_count = outer_face_contrib_count.index_select(0, face_idx_all[face_valid])
                face_prob = outer_face_support_prob.index_select(0, face_idx_all[face_valid])
                face_weight_local = outer_face_weight_sum.detach().index_select(0, face_idx_all[face_valid])
                persist_local = (
                    (face_count >= float(max(int(outer_prune_persist_min_contrib_frames), 1)))
                    & (face_prob >= float(max(outer_prune_persist_min_frame_ratio, 0.0)))
                    & (face_weight_local >= float(max(outer_prune_persist_min_weight_sum, 0.0)))
                )
                if bool(persist_local.any()):
                    outer_persist_protect_mask[face_valid] = persist_local

        color_conf_global_now: float | None = None
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
            color_conf_global_now = float(color_conf_now)
            g.geom_conf_ema = 0.9 * g.geom_conf_ema + 0.1 * float(geom_conf_now)

            if dq_fail_active:
                outer_fail_m = (g.shell_layer == 1) & (g.prune_state != 2)
                g.quality_fail_counter[outer_fail_m] = g.quality_fail_counter[outer_fail_m] + 1.0

        with torch.no_grad():
            color_conf_prev = g.color_conf_ema
            color_conf_next = (
                0.9 * color_conf_prev + 0.1 * float(color_conf_global_now)
                if color_conf_global_now is not None
                else color_conf_prev.clone()
            )
            if (
                bool(inner_local_color_prior_enable)
                and has_inner_color_support
                and face_semantic_raw_t is not None
                and face_semantic_conf_t is not None
                and g.effective_face_idx is not None
                and g.shell_layer is not None
            ):
                face_idx_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
                valid_inner_sem = (
                    (g.shell_layer == 0)
                    & (g.prune_state != 2)
                    & (face_idx_all >= 0)
                    & (face_idx_all < int(face_semantic_raw_t.shape[0]))
                )
                if bool(valid_inner_sem.any()):
                    face_idx_valid = face_idx_all[valid_inner_sem]
                    face_raw_valid = face_semantic_raw_t.index_select(0, face_idx_valid)
                    face_conf_valid = face_semantic_conf_t.index_select(0, face_idx_valid)
                    semantic_update_mask = valid_inner_sem.clone()
                    semantic_update_mask[valid_inner_sem] = (
                        (face_raw_valid >= 0)
                        & (face_conf_valid >= float(inner_local_color_prior_semantic_conf_min))
                    )
                    if bool(semantic_update_mask.any()):
                        local_color_conf = inner_color_support_mean.detach().clamp(0.0, 1.0)
                        color_conf_next[semantic_update_mask] = (
                            0.9 * color_conf_prev[semantic_update_mask]
                            + 0.1 * local_color_conf[semantic_update_mask]
                        )
            g.color_conf_ema = color_conf_next

        # Shell/occlusion policy updates (state-based, non-destructive).
        if bool(shell_enable):
            inner_mask = (g.shell_layer == 0)
            outer_mask = (g.shell_layer == 1)
            active_mask = (g.prune_state != 2)
            opacity_detached = g.opacity().detach().view(-1)
            if (
                bool(outer_block_exposed_faces)
                and outer_blocked_face_mask_t is not None
                and g.bind_face_idx is not None
            ):
                face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
                blocked_outer_valid = (
                    outer_mask
                    & active_mask
                    & (face_idx_all >= 0)
                    & (face_idx_all < int(outer_blocked_face_mask_t.shape[0]))
                )
                blocked_outer = torch.zeros_like(blocked_outer_valid)
                if bool(blocked_outer_valid.any()):
                    valid_outer_idx = torch.nonzero(blocked_outer_valid, as_tuple=False).view(-1)
                    valid_outer_face_idx = face_idx_all.index_select(0, valid_outer_idx)
                    blocked_local = outer_blocked_face_mask_t.index_select(0, valid_outer_face_idx)
                    if bool(blocked_local.any()):
                        with torch.no_grad():
                            blocked_outer[valid_outer_idx] = blocked_local
                            g.prune_state[blocked_outer] = 2
                            g.opacity_logit[blocked_outer] = g.opacity_logit[blocked_outer].clamp(
                                max=float(outer_pruned_opacity_logit_max)
                            )
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

            face_lock_protect_mask = None
            if (
                bool(face_lock_active)
                and bool(shell_enable)
                and g.shell_layer is not None
                and g.effective_face_idx is not None
                and face_semantic_focus_ids_t is not None
                and face_semantic_raw_t is not None
            ):
                face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
                raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
                face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
                face_lock_semantic_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                if bool(face_valid_all.any()):
                    face_lock_semantic_gauss[face_valid_all] = torch.isin(
                        raw_sem_all[face_ids_all[face_valid_all]],
                        face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                    )
                face_lock_protect_mask = (g.shell_layer == 0) & (g.prune_state != 2) & face_lock_semantic_gauss

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
                if face_lock_protect_mask is not None and int(to_hard.numel()) > 0:
                    keep_mask = ~face_lock_protect_mask.index_select(0, to_hard)
                    to_hard = to_hard[keep_mask]
                g.prune_state[to_hard] = 2

            if bool(inner_sparse_cleanup_enable):
                sparse_cleanup_phase = (phase_tag == "phase_inner")
                sparse_bad = (
                    inner_mask
                    & active_mask
                    & (vis_mean.detach() <= float(inner_sparse_cleanup_vis_th))
                    & (opacity_detached <= float(inner_sparse_cleanup_opacity_th))
                )
                if sparse_cleanup_phase:
                    inner_sparse_counter = torch.where(
                        sparse_bad,
                        inner_sparse_counter + 1.0,
                        torch.zeros_like(inner_sparse_counter),
                    )
                else:
                    inner_sparse_counter = torch.zeros_like(inner_sparse_counter)
                if (
                    sparse_cleanup_phase
                    and (it + 1) >= int(inner_sparse_cleanup_warmup_iters)
                    and int(inner_sparse_cleanup_every) > 0
                    and ((it + 1) % int(inner_sparse_cleanup_every) == 0)
                ):
                    to_sparse_prune = (
                        inner_mask
                        & active_mask
                        & (inner_sparse_counter >= float(inner_sparse_cleanup_patience))
                    )
                    if face_lock_protect_mask is not None and bool(to_sparse_prune.any()):
                        to_sparse_prune = to_sparse_prune & (~face_lock_protect_mask)
                    if bool(to_sparse_prune.any()):
                        pruned_now = int(to_sparse_prune.sum().item())
                        g.prune_state[to_sparse_prune] = 2
                        inner_sparse_pruned_total += pruned_now

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

            def _filter_protected_outer(idx: torch.Tensor) -> torch.Tensor:
                if int(idx.numel()) <= 0:
                    return idx
                protected_mask = torch.zeros((int(idx.numel()),), dtype=torch.bool, device=idx.device)
                if bool(outer_align_prune_protect_enable):
                    protected_mask = protected_mask | outer_align_protect_mask.index_select(0, idx)
                if bool(outer_prune_persist_protect_enable):
                    protected_mask = protected_mask | outer_persist_protect_mask.index_select(0, idx)
                keep_idx = ~protected_mask
                return idx[keep_idx]

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
                to_prune_outer = _filter_protected_outer(to_prune_outer)
                if float(outer_prune_support_th) > 0.0 and (it + 1) > int(phase_inner_end):
                    sup_th = float(max(float(outer_prune_support_th), float(outer_prune_dual_min_support)))
                    low_sup = outer_mask & active_mask & (outer_drive_support_mean.detach() < sup_th)
                    if bool(outer_prune_dual_enable):
                        low_sup = low_sup & (vis_mean.detach() <= float(max(outer_prune_dual_max_vis, 0.0)))
                    low_sup_idx = torch.nonzero(low_sup, as_tuple=False).view(-1)
                    low_sup_idx = _filter_protected_outer(low_sup_idx)
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
                    force_low_idx = _filter_protected_outer(force_low_idx)
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
                    overlap_idx = _filter_protected_outer(overlap_idx)
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
                        and outer_face_contrib_count is not None
                        and outer_face_support_prob is not None
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
                        persist_count = outer_face_contrib_count.view(-1)
                        persist_prob = outer_face_support_prob.view(-1)
                        face_score = outer_face_spawn_weight.view(-1)
                        if bool(outer_spawn_score_use_persistence):
                            face_score = face_score * persist_prob.clamp(0.0, 1.0)
                        face_candidate_mask = (
                            (face_score > 0.0)
                            & (outer_face_spawn_weight.view(-1) >= float(max(outer_spawn_min_weight_sum, 0.0)))
                            & (persist_count >= float(max(int(outer_spawn_min_contrib_frames), 1)))
                            & (persist_prob >= float(max(outer_spawn_min_frame_ratio, 0.0)))
                            & (outer_face_gap_half_avg.view(-1) > 0.0)
                            & (outer_face_gap_scale_avg.view(-1) > 0.0)
                            & (~outer_face_spawned_mask.view(-1))
                        )
                        if (
                            bool(outer_block_exposed_faces)
                            and outer_blocked_face_mask_t is not None
                            and int(outer_blocked_face_mask_t.shape[0]) == int(face_candidate_mask.shape[0])
                        ):
                            face_candidate_mask = face_candidate_mask & (~outer_blocked_face_mask_t.view(-1))
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

        face_reserve_last_spawned = 0
        face_repurpose_last_spawned = 0
        face_rebalance_target_count = max(int(face_reserve_target_count), int(face_repurpose_target_count))
        if (
            bool(shell_enable)
            and phase_tag == "phase_inner"
            and trigger_frame_data is not None
            and g.shell_layer is not None
            and g.prune_state is not None
            and g.effective_face_idx is not None
            and face_semantic_raw_t is not None
            and face_semantic_focus_ids_t is not None
            and (bool(face_reserve_enable) or bool(face_repurpose_enable))
            and (not bool(face_lock_active))
        ):
            face_idx_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
            face_valid_all = (face_idx_all >= 0) & (face_idx_all < int(face_semantic_raw_t.shape[0]))
            raw_face_all = torch.full((int(face_idx_all.shape[0]),), -1, dtype=torch.long, device=device)
            raw_conf_all = torch.zeros((int(face_idx_all.shape[0]),), dtype=torch.float32, device=device)
            if bool(face_valid_all.any()):
                face_ids_valid = face_idx_all[face_valid_all]
                raw_face_all[face_valid_all] = face_semantic_raw_t.index_select(0, face_ids_valid)
                if face_semantic_conf_t is not None:
                    raw_conf_all[face_valid_all] = face_semantic_conf_t.index_select(0, face_ids_valid)
                else:
                    raw_conf_all[face_valid_all] = 1.0
            active_inner_face = (
                (g.shell_layer == 0)
                & (g.prune_state != 2)
                & face_valid_all
                & torch.isin(raw_face_all, face_semantic_focus_ids_t)
                & (raw_conf_all >= float(face_reserve_semantic_conf_min))
            )
            active_face_count = int(active_inner_face.sum().item())
            face_need = max(int(face_rebalance_target_count) - active_face_count, 0)
            if face_need > 0:
                reserve_tick = (it == 0) or ((it + 1) % int(face_reserve_bootstrap_every) == 0)
                repurpose_tick = (it == 0) or ((it + 1) % int(face_repurpose_every) == 0)
                if bool(face_reserve_enable) and reserve_tick and ((it + 1) <= int(face_reserve_bootstrap_stop_iter)):
                    spawn_slots = torch.nonzero(
                        (g.shell_layer == 1) & (g.prune_state == 2),
                        as_tuple=False,
                    ).view(-1)
                    max_spawn = min(int(face_reserve_spawn_topk), int(face_need), int(spawn_slots.numel()))
                    if max_spawn > 0:
                        candidate_faces, candidate_scores = _select_face_reserve_candidates(
                            g=g,
                            effective_face_idx=g.effective_face_idx,
                            frame_data=trigger_frame_data,
                            face_semantic_raw_t=face_semantic_raw_t,
                            face_semantic_conf_t=face_semantic_conf_t,
                            target_raw_ids=face_semantic_focus_ids_t,
                            topk=max_spawn,
                            semantic_conf_min=float(face_reserve_semantic_conf_min),
                            face_reserve_min_active_per_face=float(face_reserve_min_active_per_face),
                        )
                        if int(candidate_faces.numel()) > 0:
                            if int(candidate_faces.numel()) > max_spawn:
                                top = torch.topk(candidate_scores, k=max_spawn, largest=True).indices
                                candidate_faces = candidate_faces.index_select(0, top)
                            face_reserve_last_spawned = _spawn_inner_face_gaussians_from_slots(
                                g=g,
                                frame_data=trigger_frame_data,
                                effective_face_idx=g.effective_face_idx,
                                face_ids=candidate_faces,
                                spawn_slots=spawn_slots,
                                inner_local_scale_cap_ratio=float(inner_local_scale_cap_ratio),
                                inner_local_scale_cap_normal_ratio=float(inner_local_scale_cap_normal_ratio),
                                face_semantic_raw_t=face_semantic_raw_t,
                                spawn_scale_mult=float(face_reserve_scale_mult),
                                spawn_opacity_logit_init=float(face_reserve_opacity_logit_init),
                                spawn_opacity_logit_cap=float(face_reserve_opacity_logit_cap),
                            )
                            if face_reserve_last_spawned > 0:
                                _log_info(
                                    train_logger,
                                    "[Iter %d/%d] face_reserve_spawn=%d active_face=%d target=%d active_inner=%d",
                                    it + 1,
                                    iters,
                                    int(face_reserve_last_spawned),
                                    int(active_face_count + face_reserve_last_spawned),
                                    int(face_rebalance_target_count),
                                    int(((g.shell_layer == 0) & (g.prune_state != 2)).sum().item()),
                                )
                face_need = max(int(face_rebalance_target_count) - (active_face_count + face_reserve_last_spawned), 0)
                if bool(face_repurpose_enable) and repurpose_tick and ((it + 1) <= int(face_repurpose_stop_iter)) and face_need > 0:
                    max_repurpose = min(int(face_repurpose_topk), int(face_need))
                    if max_repurpose > 0:
                        candidate_faces, candidate_scores = _select_face_reserve_candidates(
                            g=g,
                            effective_face_idx=g.effective_face_idx,
                            frame_data=trigger_frame_data,
                            face_semantic_raw_t=face_semantic_raw_t,
                            face_semantic_conf_t=face_semantic_conf_t,
                            target_raw_ids=face_semantic_focus_ids_t,
                            topk=max_repurpose,
                            semantic_conf_min=float(face_reserve_semantic_conf_min),
                            face_reserve_min_active_per_face=float(face_reserve_min_active_per_face),
                        )
                        repurpose_slots, repurpose_scores = _select_face_repurpose_slots(
                            g=g,
                            effective_face_idx=g.effective_face_idx,
                            face_semantic_raw_t=face_semantic_raw_t,
                            target_raw_ids=face_semantic_focus_ids_t,
                            topk=max_repurpose,
                            nonface_conf_max=float(face_repurpose_nonface_conf_max),
                        )
                        n_repurpose = min(int(candidate_faces.numel()), int(repurpose_slots.numel()), int(max_repurpose))
                        if n_repurpose > 0:
                            if int(candidate_faces.numel()) > n_repurpose:
                                topf = torch.topk(candidate_scores, k=n_repurpose, largest=True).indices
                                candidate_faces = candidate_faces.index_select(0, topf)
                            else:
                                candidate_faces = candidate_faces[:n_repurpose]
                            if int(repurpose_slots.numel()) > n_repurpose:
                                tops = torch.topk(-repurpose_scores, k=n_repurpose, largest=True).indices
                                repurpose_slots = repurpose_slots.index_select(0, tops)
                            else:
                                repurpose_slots = repurpose_slots[:n_repurpose]
                            face_repurpose_last_spawned = _spawn_inner_face_gaussians_from_slots(
                                g=g,
                                frame_data=trigger_frame_data,
                                effective_face_idx=g.effective_face_idx,
                                face_ids=candidate_faces,
                                spawn_slots=repurpose_slots,
                                inner_local_scale_cap_ratio=float(inner_local_scale_cap_ratio),
                                inner_local_scale_cap_normal_ratio=float(inner_local_scale_cap_normal_ratio),
                                face_semantic_raw_t=face_semantic_raw_t,
                                spawn_scale_mult=float(face_reserve_scale_mult),
                                spawn_opacity_logit_init=float(face_reserve_opacity_logit_init),
                                spawn_opacity_logit_cap=float(face_reserve_opacity_logit_cap),
                            )
                            if face_repurpose_last_spawned > 0:
                                _log_info(
                                    train_logger,
                                    "[Iter %d/%d] face_repurpose_spawn=%d active_face=%d target=%d active_inner=%d",
                                    it + 1,
                                    iters,
                                    int(face_repurpose_last_spawned),
                                    int(active_face_count + face_reserve_last_spawned + face_repurpose_last_spawned),
                                    int(face_rebalance_target_count),
                                    int(((g.shell_layer == 0) & (g.prune_state != 2)).sum().item()),
                                )

        inner_coverage_last_spawned = 0
        if (
            bool(shell_enable)
            and bool(inner_coverage_densify_enable)
            and phase_tag == "phase_inner"
            and ((it + 1) % int(inner_coverage_eval_every) == 0)
            and ((it + 1) <= max(int(round(float(iters) * float(inner_coverage_stop_ratio))), int(inner_coverage_eval_every)))
            and trigger_frame_data is not None
            and g.shell_layer is not None
            and g.prune_state is not None
            and g.effective_face_idx is not None
        ):
            active_inner_count = int(((g.shell_layer == 0) & (g.prune_state != 2)).sum().item())
            inner_room = max(int(inner_count_max) - active_inner_count, 0)
            spawn_slots = torch.nonzero(
                (g.shell_layer == 1) & (g.prune_state == 2),
                as_tuple=False,
            ).view(-1)
            max_spawn = min(int(inner_coverage_spawn_topk), int(inner_room), int(spawn_slots.numel()))
            if max_spawn > 0:
                candidate_faces, candidate_scores = _select_inner_coverage_face_candidates(
                    g=g,
                    effective_face_idx=g.effective_face_idx,
                    frame_data=trigger_frame_data,
                    vis_mean=vis_mean,
                    inner_color_support_mean=inner_color_support_mean,
                    residual_ema=densify_stats.residual_ema,
                    topk=max_spawn,
                )
                if int(candidate_faces.numel()) > 0:
                    if int(candidate_faces.numel()) > max_spawn:
                        top = torch.topk(candidate_scores, k=max_spawn, largest=True).indices
                        candidate_faces = candidate_faces.index_select(0, top)
                    inner_coverage_last_spawned = _spawn_inner_face_gaussians_from_slots(
                        g=g,
                        frame_data=trigger_frame_data,
                        effective_face_idx=g.effective_face_idx,
                        face_ids=candidate_faces,
                        spawn_slots=spawn_slots,
                        inner_local_scale_cap_ratio=float(inner_local_scale_cap_ratio),
                        inner_local_scale_cap_normal_ratio=float(inner_local_scale_cap_normal_ratio),
                        face_semantic_raw_t=face_semantic_raw_t,
                        spawn_scale_mult=float(inner_coverage_spawn_scale_mult),
                        spawn_opacity_logit_init=float(inner_coverage_spawn_opacity_logit_init),
                        spawn_opacity_logit_cap=float(inner_coverage_spawn_opacity_logit_cap),
                    )
                    # Keep coverage densify as a silent training event.
                    # The resulting active-count change is reflected by the regular
                    # progress summary, so emitting an extra console line here only
                    # doubles the noise without adding real decision value.

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
        reg_inner_local_color = opacity_all.new_tensor(0.0)
        inner_color_prior_stats = {
            "targets": 0.0,
            "assisted": 0.0,
            "donors": 0.0,
            "mean_target_conf": 0.0,
        }
        if (
            bool(inner_local_color_prior_enable)
            and float(inner_local_color_prior_weight) > 0.0
            and (it + 1) >= int(inner_local_color_prior_start_iter)
            and face_semantic_raw_t is not None
            and face_semantic_conf_t is not None
            and g.shell_layer is not None
            and g.prune_state is not None
            and g.effective_face_idx is not None
        ):
            color_prior_term, inner_color_prior_stats = _compute_inner_local_color_prior(
                rgb=g.rgb(),
                rgb_logit_param=g.rgb_logit,
                shell_layer=g.shell_layer,
                prune_state=g.prune_state,
                effective_face_idx=g.effective_face_idx,
                color_conf=g.color_conf_ema.detach(),
                vis_mean=vis_mean.detach(),
                support_mean=inner_color_support_mean.detach(),
                opacity=opacity_all.detach().view(-1),
                face_raw_semantic=face_semantic_raw_t,
                face_semantic_conf=face_semantic_conf_t,
                neighbor_idx=bind_face_neighbor_idx_t,
                neighbor_w=bind_face_neighbor_w_t,
                target_conf_max=float(inner_local_color_prior_target_conf_max),
                target_vis_max=float(inner_local_color_prior_target_vis_max),
                target_support_max=float(inner_local_color_prior_target_support_max),
                donor_conf_min=float(inner_local_color_prior_donor_conf_min),
                donor_opacity_min=float(inner_local_color_prior_donor_opacity_min),
                donor_vis_min=float(inner_local_color_prior_donor_vis_min),
                semantic_conf_min=float(inner_local_color_prior_semantic_conf_min),
                min_donor_weight=float(inner_local_color_prior_min_donor_weight),
                ignore_raw_semantic_ids=face_semantic_color_prior_ignore_ids_t,
                repair_blend=float(inner_local_color_repair_blend) if bool(inner_local_color_repair_enable) else 0.0,
                repair_min_weight=float(inner_local_color_repair_min_weight),
            )
            reg_inner_local_color = float(inner_local_color_prior_weight) * color_prior_term
        inner_color_prior_last_loss = float(reg_inner_local_color.detach().cpu())
        inner_color_prior_last_targets = int(inner_color_prior_stats["targets"])
        inner_color_prior_last_assisted = int(inner_color_prior_stats["assisted"])
        inner_color_prior_last_donors = int(inner_color_prior_stats["donors"])
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
            + reg_inner_local_color
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
        reg_total_detached = reg_total.detach()
        loss_value_t = reg_total_detached if loss_value_t is None else (loss_value_t + reg_total_detached)
        reg_total.backward()

        if learn_cam and cam_reg_weight > 0:
            # Keep camera near prior (prevents drift)
            reg = (cam_w ** 2).mean() + ((cam_t - cam_t_init) ** 2).mean()
            reg_loss = float(cam_reg_weight) * reg
            reg_loss_detached = reg_loss.detach()
            loss_value_t = reg_loss_detached if loss_value_t is None else (loss_value_t + reg_loss_detached)
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

        strict_twoframe_face_fit_grad_mode = False
        strict_face_inner_grad_mask_only = None
        if (
            int(nF) <= 2
            and bool(shell_enable)
            and g.shell_layer is not None
            and g.prune_state is not None
            and g.effective_face_idx is not None
            and face_semantic_focus_ids_t is not None
            and face_semantic_raw_t is not None
            and bool(face_supervision_selected_only)
            and bool(face_detail_selected_only)
        ):
            face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
            raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
            face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
            semantic_face_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
            if bool(face_valid_all.any()):
                semantic_face_gauss[face_valid_all] = torch.isin(
                    raw_sem_all[face_ids_all[face_valid_all]],
                    face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                )
            strict_face_inner_grad_mask_only = (g.shell_layer == 0) & (g.prune_state != 2) & semantic_face_gauss
            strict_twoframe_face_fit_grad_mode = bool(strict_face_inner_grad_mask_only.any())
        if strict_twoframe_face_fit_grad_mode and strict_face_inner_grad_mask_only is not None:
            strict_keep = strict_face_inner_grad_mask_only
            strict_drop = ~strict_keep
            with torch.no_grad():
                if g.rgb_logit.grad is not None:
                    g.rgb_logit.grad[strict_drop] = 0.0
                if g.opacity_logit.grad is not None:
                    g.opacity_logit.grad[strict_drop] = 0.0
                if g.log_scales.grad is not None:
                    g.log_scales.grad[strict_drop] = 0.0
                if g.quat.grad is not None:
                    g.quat.grad[strict_drop] = 0.0
                for p in (g.bind_face_offset_local, g.bind_mix_weight):
                    if p is not None and p.grad is not None:
                        p.grad.zero_()

        if bool(face_refine_phase) and bool(shell_enable) and g.shell_layer is not None:
            face_inner_grad_mask = None
            nonface_inner_grad_mask = None
            if g.effective_face_idx is not None and face_semantic_focus_ids_t is not None and face_semantic_raw_t is not None:
                face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
                raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
                face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
                semantic_face_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                if bool(face_valid_all.any()):
                    semantic_face_gauss[face_valid_all] = torch.isin(
                        raw_sem_all[face_ids_all[face_valid_all]],
                        face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                    )
                active_inner_mask = (g.shell_layer == 0) & (g.prune_state != 2)
                face_inner_grad_mask = active_inner_mask & semantic_face_gauss
                nonface_inner_grad_mask = active_inner_mask & (~semantic_face_gauss)
            if face_inner_grad_mask is not None and nonface_inner_grad_mask is not None:
                with torch.no_grad():
                    for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit, g.bind_face_offset_local):
                        if p.grad is None:
                            continue
                        if bool(nonface_inner_grad_mask.any()) and float(face_refine_nonface_grad_scale) < 1.0:
                            p.grad[nonface_inner_grad_mask] *= float(face_refine_nonface_grad_scale)
                        if bool(face_inner_grad_mask.any()) and float(face_refine_face_grad_boost) > 1.0:
                            p.grad[face_inner_grad_mask] *= float(face_refine_face_grad_boost)

        if bool(face_lock_active) and bool(shell_enable) and g.shell_layer is not None and g.effective_face_idx is not None and face_semantic_focus_ids_t is not None and face_semantic_raw_t is not None:
            face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
            raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
            face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
            face_lock_semantic_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
            if bool(face_valid_all.any()):
                face_lock_semantic_gauss[face_valid_all] = torch.isin(
                    raw_sem_all[face_ids_all[face_valid_all]],
                    face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                )
            face_lock_inner_mask = (g.shell_layer == 0) & (g.prune_state != 2) & face_lock_semantic_gauss
            if bool(face_lock_inner_mask.any()):
                with torch.no_grad():
                    for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit, g.bind_face_offset_local, g.bind_mix_weight):
                        if p is None or p.grad is None:
                            continue
                        p.grad[face_lock_inner_mask] = 0.0

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
                        if g.opacity_logit.grad is not None:
                            if bool(outer_color_refine_rgb_only):
                                g.opacity_logit.grad[outer_layer] = 0.0
                            elif (
                                bool(outer_color_train_opacity_parts_enable)
                                and outer_color_train_opacity_raw_ids_t is not None
                                and g.bind_face_idx is not None
                                and face_semantic_raw_t is not None
                            ):
                                face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
                                face_valid = (face_idx_all >= 0) & (face_idx_all < int(face_semantic_raw_t.shape[0]))
                                allowed_outer_opacity = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                                candidate_idx = torch.nonzero(outer_layer & face_valid, as_tuple=False).view(-1)
                                if int(candidate_idx.numel()) > 0:
                                    candidate_faces = face_idx_all.index_select(0, candidate_idx)
                                    raw_ids = face_semantic_raw_t.to(device=device, dtype=torch.long).index_select(0, candidate_faces)
                                    keep = torch.isin(raw_ids, outer_color_train_opacity_raw_ids_t.to(device=device, dtype=torch.long))
                                    if face_semantic_conf_t is not None and float(outer_color_train_opacity_conf_min) > 0.0:
                                        conf = face_semantic_conf_t.to(device=device, dtype=torch.float32).index_select(0, candidate_faces)
                                        keep = keep & (conf >= float(outer_color_train_opacity_conf_min))
                                    if bool(keep.any()):
                                        allowed_outer_opacity[candidate_idx.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))] = True
                                block_outer_opacity = outer_layer & (~allowed_outer_opacity)
                                if bool(block_outer_opacity.any()):
                                    g.opacity_logit.grad[block_outer_opacity] = 0.0
                        if (
                            bool(outer_color_train_parts_enable)
                            and outer_color_train_raw_ids_t is not None
                            and g.rgb_logit.grad is not None
                            and g.bind_face_idx is not None
                            and face_semantic_raw_t is not None
                        ):
                            face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
                            face_valid = (face_idx_all >= 0) & (face_idx_all < int(face_semantic_raw_t.shape[0]))
                            allowed_outer_rgb = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                            candidate_idx = torch.nonzero(outer_layer & face_valid, as_tuple=False).view(-1)
                            if int(candidate_idx.numel()) > 0:
                                candidate_faces = face_idx_all.index_select(0, candidate_idx)
                                raw_ids = face_semantic_raw_t.to(device=device, dtype=torch.long).index_select(0, candidate_faces)
                                keep = torch.isin(raw_ids, outer_color_train_raw_ids_t.to(device=device, dtype=torch.long))
                                if face_semantic_conf_t is not None and float(outer_color_train_conf_min) > 0.0:
                                    conf = face_semantic_conf_t.to(device=device, dtype=torch.float32).index_select(0, candidate_faces)
                                    keep = keep & (conf >= float(outer_color_train_conf_min))
                                if bool(keep.any()):
                                    allowed_outer_rgb[candidate_idx.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))] = True
                            block_outer_rgb = outer_layer & (~allowed_outer_rgb)
                            if bool(block_outer_rgb.any()):
                                g.rgb_logit.grad[block_outer_rgb] = 0.0
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
            best_face_bake_state is not None
            and (not bool(strict_twoframe_face_fit_grad_mode))
            and bool(face_texture_bake_enable)
            and bool(shell_enable)
            and g.shell_layer is not None
            and g.effective_face_idx is not None
            and face_semantic_focus_ids_t is not None
            and face_semantic_raw_t is not None
        ):
            with torch.no_grad():
                face_ids_all = g.effective_face_idx.to(device=device, dtype=torch.long).view(-1)
                raw_sem_all = face_semantic_raw_t.to(device=device, dtype=torch.long).view(-1)
                face_valid_all = (face_ids_all >= 0) & (face_ids_all < raw_sem_all.shape[0])
                semantic_face_gauss = torch.zeros((n_gauss,), dtype=torch.bool, device=device)
                if bool(face_valid_all.any()):
                    semantic_face_gauss[face_valid_all] = torch.isin(
                        raw_sem_all[face_ids_all[face_valid_all]],
                        face_semantic_focus_ids_t.to(device=device, dtype=torch.long).view(-1),
                    )
                face_gaussian_mask = (g.shell_layer == 0) & (g.prune_state != 2) & semantic_face_gauss
                face_bake_updates = _apply_face_texture_bake(
                    g=g,
                    xyz_render=best_face_bake_state['xyz_render'],
                    K=best_face_bake_state['K'],
                    R=best_face_bake_state['R'],
                    t=best_face_bake_state['t'],
                    gt_rgb=best_face_bake_state['gt_rgb'],
                    visibility=best_face_bake_state['visibility'],
                    face_box_xyxy=best_face_bake_state['face_box_xyxy'],
                    feature_points=best_face_bake_state['feature_points'],
                    face_gaussian_mask=face_gaussian_mask,
                    view_score=float(best_face_bake_state['score']),
                    base_blend=float(face_texture_bake_blend),
                    max_blend=float(face_texture_bake_max_blend),
                    visibility_min=float(face_texture_bake_visibility_min),
                    feature_boost=float(face_texture_bake_feature_boost),
                    sharpen_amount=float(face_texture_bake_sharpen_amount),
                    feature_sigma_frac=float(face_texture_bake_feature_sigma_frac),
                )
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
                    if g.shell_layer is not None:
                        outer_b_fg = bind_mask & (g.shell_layer == 1)
                        inner_b_fg = bind_mask & (g.shell_layer == 0)
                        inner_local_applied = torch.zeros_like(bind_mask, dtype=torch.bool, device=device)
                        if bool(inner_local_scale_clamp_enable) and bool(inner_b_fg.any()):
                            clamp_frame_data = trigger_frame_data if trigger_frame_data is not None else frame_data
                            inner_local_applied = _clamp_inner_log_scales_with_local_face_caps(
                                g=g,
                                bind_mask=inner_b_fg,
                                frame_data=clamp_frame_data,
                                fg_log_scale_min=fg_log_scale_min,
                                fg_log_scale_max=fg_log_scale_max,
                                local_ratio=inner_local_scale_clamp_ratio,
                                local_normal_ratio=inner_local_scale_clamp_normal_ratio,
                            )
                        face_semantic_local_applied = torch.zeros_like(bind_mask, dtype=torch.bool, device=device)
                        if (
                            bool(face_semantic_scale_clamp_enable)
                            and bool(inner_b_fg.any())
                            and face_semantic_focus_ids_t is not None
                            and face_semantic_raw_t is not None
                            and ((not bool(face_semantic_scale_clamp_refine_only)) or bool(face_refine_phase))
                        ):
                            clamp_frame_data = trigger_frame_data if trigger_frame_data is not None else frame_data
                            face_semantic_local_applied = _clamp_face_semantic_inner_log_scales(
                                g=g,
                                bind_mask=inner_b_fg,
                                frame_data=clamp_frame_data,
                                face_ids_focus=face_semantic_focus_ids_t,
                                face_semantic_raw_t=face_semantic_raw_t,
                                fg_log_scale_min=fg_log_scale_min,
                                ratio=face_semantic_scale_clamp_ratio,
                                normal_ratio=face_semantic_scale_clamp_normal_ratio,
                            )
                        fg_non_outer = bind_mask & (~outer_b_fg)
                        fg_non_outer_global = fg_non_outer & (~inner_local_applied) & (~face_semantic_local_applied)
                        if bool(fg_non_outer_global.any()):
                            g.log_scales[fg_non_outer_global] = g.log_scales[fg_non_outer_global].clamp(
                                min=min_v, max=max_v
                            )
                        if bool(outer_b_fg.any()):
                            g.log_scales[outer_b_fg, :2] = g.log_scales[outer_b_fg, :2].clamp(
                                min=min_v, max=max_v
                            )
                    else:
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
                        g.log_scales[outer_b, :2] = g.log_scales[outer_b, :2].clamp(min=min_v, max=max_v)
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
                if (
                    bool(outer_semantic_cap_enable)
                    and outer_semantic_cap_raw_ids_t is not None
                    and g.bind_face_idx is not None
                    and face_semantic_raw_t is not None
                ):
                    face_idx_all = g.bind_face_idx.to(device=device, dtype=torch.long).view(-1)
                    face_valid = (face_idx_all >= 0) & (face_idx_all < int(face_semantic_raw_t.shape[0]))
                    semantic_cap_mask = (
                        (g.shell_layer == 1)
                        & (g.prune_state != 2)
                        & face_valid
                    )
                    if bool(semantic_cap_mask.any()):
                        cap_idx = torch.nonzero(semantic_cap_mask, as_tuple=False).view(-1)
                        cap_faces = face_idx_all.index_select(0, cap_idx)
                        raw_ids = face_semantic_raw_t.to(device=device, dtype=torch.long).index_select(0, cap_faces)
                        keep = torch.isin(raw_ids, outer_semantic_cap_raw_ids_t.to(device=device, dtype=torch.long))
                        if face_semantic_conf_t is not None and float(outer_semantic_cap_conf_min) > 0.0:
                            conf = face_semantic_conf_t.to(device=device, dtype=torch.float32).index_select(0, cap_faces)
                            keep = keep & (conf >= float(outer_semantic_cap_conf_min))
                        if bool(keep.any()):
                            capped_idx = cap_idx.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))
                            if float(outer_semantic_cap_opacity) < 1.0:
                                cap = float(np.clip(float(outer_semantic_cap_opacity), 1e-4, 0.999))
                                cap_logit = float(np.log(cap / (1.0 - cap)))
                                g.opacity_logit[capped_idx] = g.opacity_logit[capped_idx].clamp(max=cap_logit)
                            if outer_semantic_cap_scale_xy_max is not None:
                                scale_cap = float(max(float(outer_semantic_cap_scale_xy_max), 1e-6))
                                log_scale_cap = float(np.log(scale_cap))
                                g.log_scales[capped_idx, :2] = g.log_scales[capped_idx, :2].clamp(max=log_scale_cap)
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
                align_protected_q = outer_active_q & outer_align_protect_mask
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
                    "hard_pruned_inner_sparse_count": int(inner_sparse_pruned_total),
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
                    "outer_align_protected_count": int(align_protected_q.sum().item()),
                    "inner_local_color_prior_loss": float(inner_color_prior_last_loss),
                    "inner_local_color_prior_target_count": int(inner_color_prior_last_targets),
                    "inner_local_color_prior_assisted_count": int(inner_color_prior_last_assisted),
                    "inner_local_color_prior_donor_count": int(inner_color_prior_last_donors),
                    "face_reserve_spawn_count": int(face_reserve_last_spawned),
                    "inner_coverage_spawn_count": int(inner_coverage_last_spawned),
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
                    _log_info(
                        train_logger,
                        "[Iter %d/%d] phase=%s inner_active=%d outer_active=%d outer_pruned=%d",
                        it + 1,
                        iters,
                        phase_label,
                        inner_active_iter,
                        outer_active_iter,
                        outer_pruned_iter,
                    )
                if outer_zero_stop_enable:
                    phase_ok = (not bool(outer_zero_stop_after_phase_inner)) or (phase_tag != "phase_inner")
                    if phase_ok and (it + 1) >= int(outer_zero_stop_warmup_iters):
                        if outer_active_iter <= 0:
                            outer_zero_consecutive += 1
                        else:
                            outer_zero_consecutive = 0
                        if outer_zero_consecutive >= int(outer_zero_stop_patience):
                            _log_warning(
                                train_logger,
                                "Early stop at iter %d/%d: outer_active_count stayed 0 for %d consecutive iterations.",
                                it + 1,
                                iters,
                                outer_zero_consecutive,
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
            inner_checkpoint_payload = _build_stage2_checkpoint_payload(
                g=g,
                K=K,
                frame_Ks=frame_K_tensors,
                cam_w=cam_w,
                cam_t=cam_t,
                checkpoint_metadata={
                    **checkpoint_metadata_base,
                    "dual_quality_gate_last": _build_dual_quality_gate_last(
                        color_error=dq_last_color_error,
                        color_var=dq_last_color_var,
                        geometry_score=dq_last_geom_score,
                        fail_active=dq_fail_active,
                    ),
                },
                quality_pass_now=False,
            )
            _save_stage2_checkpoint(
                path=save_inner_checkpoint_path,
                payload=inner_checkpoint_payload,
                logger=train_logger,
                label="inner",
            )
            inner_ckpt_saved = True

        loss_value = float(loss_value_t.item()) if loss_value_t is not None else 0.0

        if es_enable:
            if loss_value + es_min_delta < es_best:
                es_best = loss_value
                es_bad = 0
            elif it >= es_warmup:
                es_bad += 1
                if es_bad >= es_patience:
                    _log_warning(
                        train_logger,
                        "Early stop at iter %d/%d: best_loss=%.6f, last_loss=%.6f, patience=%d, min_delta=%.6f",
                        it + 1,
                        iters,
                        es_best,
                        loss_value,
                        es_patience,
                        es_min_delta,
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
        inner_checkpoint_payload = _build_stage2_checkpoint_payload(
            g=g,
            K=K,
            frame_Ks=frame_K_tensors,
            cam_w=cam_w,
            cam_t=cam_t,
            checkpoint_metadata={
                **checkpoint_metadata_base,
                "dual_quality_gate_last": _build_dual_quality_gate_last(
                    color_error=dq_last_color_error,
                    color_var=dq_last_color_var,
                    geometry_score=dq_last_geom_score,
                    fail_active=dq_fail_active,
                ),
            },
            quality_pass_now=bool(quality_pass),
        )
        _save_stage2_checkpoint(
            path=save_inner_checkpoint_path,
            payload=inner_checkpoint_payload,
            logger=train_logger,
            label="inner",
        )
        inner_ckpt_saved = True

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
    final_checkpoint_payload = _build_stage2_checkpoint_payload(
        g=g,
        K=K,
        frame_Ks=frame_K_tensors,
        cam_w=cam_w,
        cam_t=cam_t,
        checkpoint_metadata={
            **checkpoint_metadata_base,
            "dual_quality_gate_last": _build_dual_quality_gate_last(
                color_error=dq_last_color_error,
                color_var=dq_last_color_var,
                geometry_score=dq_last_geom_score,
                fail_active=dq_fail_active,
            ),
        },
        quality_pass_now=bool(quality_pass),
    )
    _save_stage2_checkpoint(
        path=out_ckpt,
        payload=final_checkpoint_payload,
        logger=train_logger,
        label="final",
    )











