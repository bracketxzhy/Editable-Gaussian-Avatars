from __future__ import annotations
import torch
import torch.nn.functional as F
from avatar_gs.gaussian.structured_gaussian import Gaussian2D

def render_gaussians_2d(g: Gaussian2D, H: int, W: int) -> torch.Tensor:
    """
    返回 (3,H,W)，简单 alpha-blend 的 2D 高斯泼溅，占位渲染器（可微）。
    """
    device = g.mu.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    img = torch.zeros((3, H, W), device=device, dtype=torch.float32)

    # (N,1,1)
    mu_x = g.mu[:, 0].view(-1, 1, 1)
    mu_y = g.mu[:, 1].view(-1, 1, 1)
    sig = g.sigma.clamp(min=0.5, max=50.0).view(-1, 1, 1)
    a = g.alpha.clamp(0.0, 1.0).view(-1, 1, 1)
    c = g.rgb.clamp(0.0, 1.0).view(-1, 3, 1, 1)

    # (N,H,W) gaussian weights
    dx2 = (xx[None, :, :] - mu_x) ** 2
    dy2 = (yy[None, :, :] - mu_y) ** 2
    w = torch.exp(-(dx2 + dy2) / (2.0 * sig ** 2)) * a

    # naive over blending: sum(w * color) / (sum(w) + eps)
    num = (w[:, None, :, :] * c).sum(dim=0)           # (3,H,W)
    den = (w.sum(dim=0, keepdim=False) + 1e-6)        # (H,W)
    img = num / den[None, :, :]
    return img
