from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BindingModeConfig:
    mode: str = "games_face"
    hybrid_weight: float = 0.0
    hybrid_dynamic: bool = False
    mix_w_min: float = 0.05
    mix_w_max: float = 0.95
    eps_normal: float = 1e-3
    curv_conf_tau: float = 0.2
    curv_degen_tau: float = 1e-4
    temporal_frame: bool = True
    scale_clamp_min: float = 1e-4
    scale_clamp_max: float = 1.0
    outer_geom_scale_blend: float = 0.0
    outer_scale_clamp_max: float = 0.01
    max_temporal_angle_deg: float = 75.0
    temporal_alpha: float = 0.35
    debug_assertions: bool = False


@dataclass
class BindingFrameData:
    verts_t: torch.Tensor
    faces_t: torch.Tensor | None
    face_t1_t: torch.Tensor | None
    face_t2_t: torch.Tensor | None
    face_n_t: torch.Tensor | None
    face_s1_t: torch.Tensor | None
    face_s2_t: torch.Tensor | None
    vertex_n_t: torch.Tensor
    vertex_e_min_t: torch.Tensor
    vertex_e_max_t: torch.Tensor
    curv_conf_t: torch.Tensor
    curv_reliable_t: torch.Tensor

