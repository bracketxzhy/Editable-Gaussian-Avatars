from __future__ import annotations

import torch


def opacity_prior_loss(opacity: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    t = float(target)
    return ((opacity - t) ** 2).mean()


def covariance_penalty(scales: torch.Tensor, cov_min: float, cov_max: float) -> torch.Tensor:
    s = scales
    lo = torch.relu(float(cov_min) - s)
    hi = torch.relu(s - float(cov_max))
    return (lo * lo + hi * hi).mean()


def canonical_offset_l2(bind_offset: torch.Tensor | None) -> torch.Tensor:
    if bind_offset is None:
        return torch.tensor(0.0)
    return (bind_offset * bind_offset).mean()
