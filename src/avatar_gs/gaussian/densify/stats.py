from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianDensifyStats:
    vis_count: torch.Tensor
    residual_ema: torch.Tensor

    @staticmethod
    def create(n_gaussians: int, device: str | torch.device) -> "GaussianDensifyStats":
        n = int(n_gaussians)
        return GaussianDensifyStats(
            vis_count=torch.zeros((n,), dtype=torch.float32, device=device),
            residual_ema=torch.zeros((n,), dtype=torch.float32, device=device),
        )
