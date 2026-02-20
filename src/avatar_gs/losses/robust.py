from __future__ import annotations

import torch


def huber_loss(diff: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    d = abs(float(delta))
    if d <= 0.0:
        return diff.abs()
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(d, device=diff.device, dtype=diff.dtype))
    lin = abs_diff - quad
    return 0.5 * quad * quad + d * lin


def charbonnier_loss(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    e = max(float(eps), 1e-12)
    return torch.sqrt(diff * diff + e * e)


def compute_robust_photo_loss(
    diff: torch.Tensor,
    loss_type: str = "charbonnier",
    huber_delta: float = 0.01,
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    lt = str(loss_type).strip().lower()
    if lt in ("charbonnier", "charb"):
        return charbonnier_loss(diff, eps=charbonnier_eps)
    if lt in ("huber", "smooth_l1"):
        return huber_loss(diff, delta=huber_delta)
    if lt in ("l1",):
        return diff.abs()
    if lt in ("l2", "mse"):
        return diff * diff
    raise ValueError(f"Unsupported robust loss type: {loss_type}")
