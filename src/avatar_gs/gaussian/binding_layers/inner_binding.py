from __future__ import annotations

import torch

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from .types import BindingFrameData, BindingModeConfig
from .binding_update import (
    _blend_quat_linear,
    _normalize_torch,
    _safe_barycentric,
    build_vertex_binding_targets,
    quat_from_rotmat_torch,
)


def compute_inner_bind_mask(
    g: Gaussian3D,
    bind_mask: torch.Tensor,
) -> torch.Tensor:
    mask = bind_mask.bool()
    if g.shell_layer is None:
        return mask
    shell = g.shell_layer.to(device=mask.device, dtype=torch.int64)
    return mask & (shell == 0)


def apply_inner_binding_per_frame(
    *,
    g: Gaussian3D,
    frame: BindingFrameData,
    bind_mask: torch.Tensor,
    bind_vertex_idx: torch.Tensor,
    bind_vertex_offset: torch.Tensor | None,
    bind_face_idx: torch.Tensor,
    bind_bary: torch.Tensor,
    cfg: BindingModeConfig,
    quat_blend: float = 1.0,
    prev_vertex_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    inner_mask = compute_inner_bind_mask(g=g, bind_mask=bind_mask)
    if not bool(inner_mask.any()):
        return prev_vertex_dirs, torch.zeros_like(bind_mask, dtype=torch.bool)
    return update_inner_gaussians_with_binding(
        g=g,
        frame=frame,
        bind_mask=inner_mask,
        bind_vertex_idx=bind_vertex_idx,
        bind_vertex_offset=bind_vertex_offset,
        bind_face_idx=bind_face_idx,
        bind_bary=bind_bary,
        cfg=cfg,
        quat_blend=quat_blend,
        prev_vertex_dirs=prev_vertex_dirs,
    )


def update_inner_gaussians_with_binding(
    g: Gaussian3D,
    frame: BindingFrameData,
    bind_mask: torch.Tensor,
    bind_vertex_idx: torch.Tensor,
    bind_vertex_offset: torch.Tensor | None,
    cfg: BindingModeConfig,
    *,
    bind_face_idx: torch.Tensor | None = None,
    bind_bary: torch.Tensor | None = None,
    quat_blend: float = 1.0,
    prev_vertex_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Inner-layer binding: vertex-anchored primary, face fallback, no local outer offset."""
    if bind_mask.ndim != 1:
        raise ValueError(f"bind_mask must be (N,), got {bind_mask.shape}")
    if not bool(bind_mask.any()):
        return prev_vertex_dirs, bind_mask

    mode = str(cfg.mode).strip().lower()
    if mode == "legacy":
        return prev_vertex_dirs, bind_mask

    bound_ids = torch.nonzero(bind_mask, as_tuple=False).view(-1)
    n_bound = int(bound_ids.shape[0])
    if n_bound <= 0:
        return prev_vertex_dirs, bind_mask

    scales_old = g.scales().index_select(0, bound_ids)
    quat_old = g.quat.index_select(0, bound_ids)
    mu_old = g.xyz.index_select(0, bound_ids)

    # Inner logic: strictly vertex-driven geometry/orientation.
    mu_vert, quat_vert, scale_vert, v_valid, prev_out = build_vertex_binding_targets(
        frame=frame,
        bind_vertex_idx=bind_vertex_idx,
        bind_vertex_offset=bind_vertex_offset,
        bound_ids=bound_ids,
        scales_old=scales_old,
        quat_old=quat_old,
        mu_old=mu_old,
        cfg=cfg,
        prev_vertex_dirs=prev_vertex_dirs,
    )
    target_mu = mu_old.clone()
    target_quat = _normalize_torch(quat_old.clone(), eps=1e-8)
    target_scale = scales_old.clone()

    if bool(v_valid.any()):
        local = torch.nonzero(v_valid, as_tuple=False).view(-1)
        target_mu[local] = mu_vert.index_select(0, local)
        target_quat[local] = quat_vert.index_select(0, local)
        target_scale[local] = scale_vert.index_select(0, local)

    # Optional face fallback only when vertex index is invalid.
    have_face_meta = (
        bind_face_idx is not None
        and bind_bary is not None
        and frame.faces_t is not None
        and frame.face_t1_t is not None
        and frame.face_t2_t is not None
        and frame.face_n_t is not None
    )
    f_valid = torch.zeros((n_bound,), dtype=torch.bool, device=bind_mask.device)
    if have_face_meta:
        f_idx = bind_face_idx.index_select(0, bound_ids)
        n_faces = int(frame.faces_t.shape[0])
        f_valid = (f_idx >= 0) & (f_idx < n_faces)
        use_face_fallback = f_valid & (~v_valid)
        if bool(use_face_fallback.any()):
            local = torch.nonzero(use_face_fallback, as_tuple=False).view(-1)
            fs = f_idx.index_select(0, local)
            bary = _safe_barycentric(bind_bary.index_select(0, bound_ids.index_select(0, local)))
            tri_idx = frame.faces_t.index_select(0, fs)
            tri = frame.verts_t.index_select(0, tri_idx.reshape(-1)).reshape(-1, 3, 3)
            mu_f = (bary.unsqueeze(-1) * tri).sum(dim=1)
            target_mu[local] = mu_f

            t1 = frame.face_t1_t.index_select(0, fs)
            t2 = frame.face_t2_t.index_select(0, fs)
            nf = frame.face_n_t.index_select(0, fs)
            R_f = torch.stack([t1, t2, nf], dim=-1)
            target_quat[local] = quat_from_rotmat_torch(R_f)

    target_scale = target_scale.clamp(
        min=float(cfg.scale_clamp_min),
        max=float(cfg.scale_clamp_max),
    )
    target_log_scale = torch.log(target_scale.clamp(min=1e-8))
    target_quat = _normalize_torch(target_quat, eps=1e-8)

    with torch.no_grad():
        g.xyz.index_copy_(0, bound_ids, target_mu.detach())
        q_old_local = g.quat.index_select(0, bound_ids)
        q_apply = _blend_quat_linear(q_old=q_old_local, q_target=target_quat, blend=float(quat_blend))
        g.quat.index_copy_(0, bound_ids, q_apply.detach())
        g.log_scales.index_copy_(0, bound_ids, target_log_scale.detach())

    valid_local = v_valid | f_valid
    valid_mask = torch.zeros_like(bind_mask)
    valid_mask[bound_ids] = valid_local
    return prev_out, valid_mask
