from __future__ import annotations

import torch

from .stats import GaussianDensifyStats


def update_densify_stats(
    stats: GaussianDensifyStats,
    visibility: torch.Tensor,
    residual_scalar: float,
    tau_vis: float,
    ema_alpha: float = 0.9,
) -> None:
    vis = visibility.detach().float().clamp(0.0, 1.0)
    visible = vis > float(tau_vis)
    stats.vis_count[visible] += 1.0

    r = float(residual_scalar)
    a = min(max(float(ema_alpha), 0.0), 1.0)
    if bool(visible.any()):
        idx = torch.nonzero(visible, as_tuple=False).view(-1)
        old = stats.residual_ema[idx]
        stats.residual_ema[idx] = a * old + (1.0 - a) * r


def select_densify_candidates(
    stats: GaussianDensifyStats,
    visibility: torch.Tensor,
    V_min: int,
    densify_top_percent: float,
) -> torch.Tensor:
    """Return candidate indices based on vis_count + residual top-percent among visible gaussians."""
    vis = visibility.detach().float().clamp(0.0, 1.0)
    eligible = (stats.vis_count >= float(max(int(V_min), 0))) & (vis > 0.0)
    idx = torch.nonzero(eligible, as_tuple=False).view(-1)
    if int(idx.numel()) == 0:
        return idx

    scores = stats.residual_ema[idx]
    p = float(max(min(densify_top_percent, 100.0), 0.0))
    if p <= 0.0:
        return torch.empty((0,), dtype=torch.long, device=idx.device)

    k = max(int(round((p / 100.0) * int(idx.numel()))), 1)
    if k >= int(idx.numel()):
        return idx
    topk = torch.topk(scores, k=k, largest=True).indices
    return idx[topk]
