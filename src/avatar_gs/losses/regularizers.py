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


def outer_color_band_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    band_mask: torch.Tensor | None,
) -> torch.Tensor:
    if band_mask is None:
        return pred_rgb.new_tensor(0.0)
    if band_mask.ndim == 2:
        band = band_mask.unsqueeze(0)
    else:
        band = band_mask
    if band.shape[0] != 1:
        band = band[:1]
    band = band.clamp(0.0, 1.0)
    w = band.expand_as(pred_rgb)
    return ((pred_rgb - gt_rgb).abs() * w).sum() / (w.sum() + 1e-6)


def layer_alpha_balance_loss(
    opacity: torch.Tensor,
    shell_layer: torch.Tensor | None,
    *,
    outer_alpha_cap: float = 0.30,
    inner_alpha_floor: float = 0.03,
) -> torch.Tensor:
    if shell_layer is None:
        return opacity.new_tensor(0.0)
    layer = shell_layer.view(-1).to(dtype=torch.int64, device=opacity.device)
    op = opacity.view(-1)
    out_m = layer == 1
    in_m = layer == 0
    loss = opacity.new_tensor(0.0)
    if bool(out_m.any()):
        out_mean = op[out_m].mean()
        loss = loss + torch.relu(out_mean - float(outer_alpha_cap)) ** 2
    if bool(in_m.any()):
        in_mean = op[in_m].mean()
        loss = loss + 0.25 * (torch.relu(float(inner_alpha_floor) - in_mean) ** 2)
    return loss
