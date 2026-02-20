from __future__ import annotations

import torch

from avatar_gs.gaussian.binding import BindingFrameData, BindingModeConfig, update_gaussians_with_binding
from avatar_gs.gaussian.structured_gaussian import Gaussian3D


def init_barycentric_binding(g: Gaussian3D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if g.bind_idx is None:
        raise RuntimeError("Gaussian bind_idx is missing; cannot initialize mesh binding.")
    if g.bind_face_idx is None or g.bind_bary is None:
        raise RuntimeError(
            "Barycentric binding requires bind_face_idx and bind_bary. "
            "Reinitialize gaussians with surface sampling enabled."
        )
    bind_idx = g.bind_idx.long()
    bind_face_idx = g.bind_face_idx.long()
    bind_bary = g.bind_bary.float()
    return bind_idx, bind_face_idx, bind_bary


def apply_barycentric_binding_per_frame(
    *,
    g: Gaussian3D,
    frame_data: BindingFrameData,
    bind_mask: torch.Tensor,
    bind_vertex_idx: torch.Tensor,
    bind_vertex_offset: torch.Tensor | None,
    bind_face_idx: torch.Tensor,
    bind_bary: torch.Tensor,
    quat_blend: float,
    prev_vertex_dirs: torch.Tensor | None,
    cfg: BindingModeConfig,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if bind_face_idx is None or bind_bary is None:
        raise RuntimeError("Barycentric binding is required but bind_face_idx/bind_bary is missing.")
    return update_gaussians_with_binding(
        g=g,
        frame=frame_data,
        bind_mask=bind_mask,
        bind_vertex_idx=bind_vertex_idx,
        bind_vertex_offset=bind_vertex_offset,
        bind_face_idx=bind_face_idx,
        bind_bary=bind_bary,
        quat_blend=quat_blend,
        prev_vertex_dirs=prev_vertex_dirs,
        cfg=cfg,
    )
