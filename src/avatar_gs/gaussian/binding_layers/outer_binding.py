from __future__ import annotations

import numpy as np
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


def compute_outer_anchor_mask(
    g: Gaussian3D,
    bind_mask: torch.Tensor,
) -> torch.Tensor:
    mask = bind_mask.bool()
    if g.shell_layer is None:
        return torch.zeros_like(mask, dtype=torch.bool)
    shell = g.shell_layer.to(device=mask.device, dtype=torch.int64)
    return mask & (shell == 1)


def apply_outer_anchor_per_frame(
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
    outer_mask = compute_outer_anchor_mask(g=g, bind_mask=bind_mask)
    if not bool(outer_mask.any()):
        return prev_vertex_dirs, torch.zeros_like(bind_mask, dtype=torch.bool)
    return update_outer_gaussians_with_binding(
        g=g,
        frame=frame,
        bind_mask=outer_mask,
        bind_vertex_idx=bind_vertex_idx,
        bind_vertex_offset=bind_vertex_offset,
        bind_face_idx=bind_face_idx,
        bind_bary=bind_bary,
        cfg=cfg,
        quat_blend=quat_blend,
        prev_vertex_dirs=prev_vertex_dirs,
    )


def update_outer_gaussians_with_binding(
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
    """Outer-layer binding: face-anchored primary with tangent-frame offset, vertex fallback."""
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

    # Vertex fallback components.
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

    # Outer primary logic: face binding.
    target_mu = mu_vert.clone()
    target_quat = quat_vert.clone()
    target_scale = scale_vert.clone()
    f_valid = torch.zeros((n_bound,), dtype=torch.bool, device=bind_mask.device)
    have_face_meta = (
        bind_face_idx is not None
        and bind_bary is not None
        and frame.faces_t is not None
        and frame.face_t1_t is not None
        and frame.face_t2_t is not None
        and frame.face_n_t is not None
    )
    if have_face_meta:
        f_idx = bind_face_idx.index_select(0, bound_ids)
        n_faces = int(frame.faces_t.shape[0])
        f_valid = (f_idx >= 0) & (f_idx < n_faces)
        if bool(f_valid.any()):
            local = torch.nonzero(f_valid, as_tuple=False).view(-1)
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

            s_geom_1 = frame.face_s1_t.index_select(0, fs) if frame.face_s1_t is not None else None
            s_geom_2 = frame.face_s2_t.index_select(0, fs) if frame.face_s2_t is not None else None
            s_old = scales_old.index_select(0, local)
            if s_geom_1 is None or s_geom_2 is None:
                s_t1 = s_old[:, 0]
                s_t2 = s_old[:, 1]
            else:
                ob = float(np.clip(float(cfg.outer_geom_scale_blend), 0.0, 1.0))
                s_t1 = (1.0 - ob) * s_old[:, 0] + ob * s_geom_1
                s_t2 = (1.0 - ob) * s_old[:, 1] + ob * s_geom_2
            s_n_local = torch.full_like(s_t1, float(cfg.eps_normal))
            target_scale[local] = torch.stack([s_t1, s_t2, s_n_local], dim=1)

            # Outer-only local offset in face tangent frame.
            if g.bind_face_offset_local is not None:
                off_local = g.bind_face_offset_local.index_select(0, bound_ids.index_select(0, local))
                dxyz = (
                    off_local[:, 0:1] * t1
                    + off_local[:, 1:2] * t2
                    + off_local[:, 2:3] * nf
                )
                target_mu[local] = target_mu.index_select(0, local) + dxyz

    target_scale = target_scale.clamp(
        min=float(cfg.scale_clamp_min),
        max=float(cfg.scale_clamp_max),
    )
    outer_s_max = float(max(float(cfg.scale_clamp_min), float(cfg.outer_scale_clamp_max)))
    target_scale[:, 0] = target_scale[:, 0].clamp(min=float(cfg.scale_clamp_min), max=outer_s_max)
    target_scale[:, 1] = target_scale[:, 1].clamp(min=float(cfg.scale_clamp_min), max=outer_s_max)
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
