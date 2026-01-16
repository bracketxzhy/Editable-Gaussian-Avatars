from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class Gaussian2D:
    # 2D center in pixel coords (N,2)
    mu: torch.Tensor
    # isotropic sigma (N,1)
    sigma: torch.Tensor
    # opacity (N,1)
    alpha: torch.Tensor
    # color (N,3) in [0,1]
    rgb: torch.Tensor

    def to(self, device: str):
        return Gaussian2D(
            mu=self.mu.to(device),
            sigma=self.sigma.to(device),
            alpha=self.alpha.to(device),
            rgb=self.rgb.to(device),
        )

    def state_dict(self):
        return {"mu": self.mu, "sigma": self.sigma, "alpha": self.alpha, "rgb": self.rgb}

    @staticmethod
    def load_state_dict(sd: dict) -> "Gaussian2D":
        return Gaussian2D(mu=sd["mu"], sigma=sd["sigma"], alpha=sd["alpha"], rgb=sd["rgb"])
