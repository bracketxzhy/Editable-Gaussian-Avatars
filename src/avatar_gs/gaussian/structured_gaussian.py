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

    def to(self, device: str):
        return Gaussian3D(
            xyz=self.xyz.to(device),
            log_scales=self.log_scales.to(device),
            quat=self.quat.to(device),
            opacity_logit=self.opacity_logit.to(device),
            rgb_logit=self.rgb_logit.to(device),
        )

    def state_dict(self):
        return {
            "xyz": self.xyz,
            "log_scales": self.log_scales,
            "quat": self.quat,
            "opacity_logit": self.opacity_logit,
            "rgb_logit": self.rgb_logit,
        }

    @staticmethod
    def load_state_dict(sd: dict) -> "Gaussian3D":
        return Gaussian3D(
            xyz=sd["xyz"],
            log_scales=sd["log_scales"],
            quat=sd["quat"],
            opacity_logit=sd["opacity_logit"],
            rgb_logit=sd["rgb_logit"],
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
