from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Gaussian3D:
    # (N,3) xyz in world/canonical space
    xyz: torch.Tensor
    # (N,3) log-scales (anisotropic), exp -> scales
    log_scales: torch.Tensor
    # (N,4) unit quaternion (w,x,y,z)
    quat: torch.Tensor
    # (N,1) opacity in logit space
    opacity_logit: torch.Tensor
    # (N,3) RGB in logit space (or SH later)
    rgb_logit: torch.Tensor
    # (N,) optional binding to mesh vertex indices (for dynamic attachment)
    bind_idx: torch.Tensor | None = None
    # (N,3) optional static offset from bound vertex (for dense surface sampling)
    bind_offset: torch.Tensor | None = None
    # (N,) optional bound face index for face-pinned binding
    bind_face_idx: torch.Tensor | None = None
    # (N,3) optional barycentric coordinates on bound face
    bind_bary: torch.Tensor | None = None
    # (N,1) optional per-gaussian hybrid blend weight in [0,1]
    bind_hybrid_w: torch.Tensor | None = None
    # (N,3) optional local offset in bound face tangent frame [t1, t2, n]
    bind_face_offset_local: torch.Tensor | None = None
    # (N,) optional shell layer id: 0=inner, 1=outer
    shell_layer: torch.Tensor | None = None
    # (N,1) optional adaptive mix weight in [0,1] for hybrid binding
    bind_mix_weight: torch.Tensor | None = None
    # (N,) optional prune/freezing state: 0=active, 1=soft-frozen, 2=hard-pruned
    prune_state: torch.Tensor | None = None
    # (N,) optional per-gaussian quality gate fail counter
    quality_fail_counter: torch.Tensor | None = None
    # (N,) optional EMA confidence for color gate (higher is better)
    color_conf_ema: torch.Tensor | None = None
    # (N,) optional EMA confidence for geometry gate (higher is better)
    geom_conf_ema: torch.Tensor | None = None

    def to(self, device: str):
        return Gaussian3D(
            xyz=self.xyz.to(device),
            log_scales=self.log_scales.to(device),
            quat=self.quat.to(device),
            opacity_logit=self.opacity_logit.to(device),
            rgb_logit=self.rgb_logit.to(device),
            bind_idx=self.bind_idx.to(device) if self.bind_idx is not None else None,
            bind_offset=self.bind_offset.to(device) if self.bind_offset is not None else None,
            bind_face_idx=self.bind_face_idx.to(device) if self.bind_face_idx is not None else None,
            bind_bary=self.bind_bary.to(device) if self.bind_bary is not None else None,
            bind_hybrid_w=self.bind_hybrid_w.to(device) if self.bind_hybrid_w is not None else None,
            bind_face_offset_local=self.bind_face_offset_local.to(device) if self.bind_face_offset_local is not None else None,
            shell_layer=self.shell_layer.to(device) if self.shell_layer is not None else None,
            bind_mix_weight=self.bind_mix_weight.to(device) if self.bind_mix_weight is not None else None,
            prune_state=self.prune_state.to(device) if self.prune_state is not None else None,
            quality_fail_counter=self.quality_fail_counter.to(device) if self.quality_fail_counter is not None else None,
            color_conf_ema=self.color_conf_ema.to(device) if self.color_conf_ema is not None else None,
            geom_conf_ema=self.geom_conf_ema.to(device) if self.geom_conf_ema is not None else None,
        )

    def state_dict(self):
        sd = {
            "xyz": self.xyz,
            "log_scales": self.log_scales,
            "quat": self.quat,
            "opacity_logit": self.opacity_logit,
            "rgb_logit": self.rgb_logit,
        }
        if self.bind_idx is not None:
            sd["bind_idx"] = self.bind_idx
        if self.bind_offset is not None:
            sd["bind_offset"] = self.bind_offset
        if self.bind_face_idx is not None:
            sd["bind_face_idx"] = self.bind_face_idx
        if self.bind_bary is not None:
            sd["bind_bary"] = self.bind_bary
        if self.bind_hybrid_w is not None:
            sd["bind_hybrid_w"] = self.bind_hybrid_w
        if self.bind_face_offset_local is not None:
            sd["bind_face_offset_local"] = self.bind_face_offset_local
        if self.shell_layer is not None:
            sd["shell_layer"] = self.shell_layer
        if self.bind_mix_weight is not None:
            sd["bind_mix_weight"] = self.bind_mix_weight
        if self.prune_state is not None:
            sd["prune_state"] = self.prune_state
        if self.quality_fail_counter is not None:
            sd["quality_fail_counter"] = self.quality_fail_counter
        if self.color_conf_ema is not None:
            sd["color_conf_ema"] = self.color_conf_ema
        if self.geom_conf_ema is not None:
            sd["geom_conf_ema"] = self.geom_conf_ema
        return sd

    @staticmethod
    def load_state_dict(sd: dict) -> "Gaussian3D":
        return Gaussian3D(
            xyz=sd["xyz"],
            log_scales=sd["log_scales"],
            quat=sd["quat"],
            opacity_logit=sd["opacity_logit"],
            rgb_logit=sd["rgb_logit"],
            bind_idx=sd.get("bind_idx", None),
            bind_offset=sd.get("bind_offset", None),
            bind_face_idx=sd.get("bind_face_idx", None),
            bind_bary=sd.get("bind_bary", None),
            bind_hybrid_w=sd.get("bind_hybrid_w", None),
            bind_face_offset_local=sd.get("bind_face_offset_local", None),
            shell_layer=sd.get("shell_layer", None),
            bind_mix_weight=sd.get("bind_mix_weight", None),
            prune_state=sd.get("prune_state", None),
            quality_fail_counter=sd.get("quality_fail_counter", None),
            color_conf_ema=sd.get("color_conf_ema", None),
            geom_conf_ema=sd.get("geom_conf_ema", None),
        )

    def rgb(self):
        return torch.sigmoid(self.rgb_logit)

    def opacity(self):
        return torch.sigmoid(self.opacity_logit)

    def scales(self):
        return torch.exp(self.log_scales)

    def normalized_quat(self):
        q = self.quat
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
