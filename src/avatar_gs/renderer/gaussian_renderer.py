from __future__ import annotations
import torch
from avatar_gs.gaussian.structured_gaussian import Gaussian3D

try:
    import gsplat  # type: ignore
except Exception as e:  # pragma: no cover
    gsplat = None
    _gsplat_err = e


def _viewmat_world_to_cam(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Build a 4x4 world->camera matrix:
        Xc = R * Xw + t
    """
    V = torch.eye(4, device=R.device, dtype=R.dtype)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def render_gaussians_3d(
    g: Gaussian3D,
    K: torch.Tensor,   # (3,3)
    R: torch.Tensor,   # (3,3) world->camera
    t: torch.Tensor,   # (3,)  world->camera
    H: int,
    W: int,
    bg_color=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    Render 3D Gaussians using gsplat.rasterization (common legacy signature).
    Returns: (3,H,W) float in [0,1]
    """
    if gsplat is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "gsplat is not installed or failed to import. "
            f"Original error: {_gsplat_err}"
        )

    device = g.xyz.device
    dtype = g.xyz.dtype

    # ---- Gaussian params ----
    means = g.xyz.contiguous()                         # [N,3]
    quats = g.normalized_quat().contiguous()           # [N,4]
    scales = g.scales().contiguous()                   # [N,3]
    opacities = g.opacity().squeeze(-1).contiguous()   # [N]
    colors = g.rgb().contiguous()                      # [N,3] or [N,D] depending on render_mode

    # ---- Camera batch (C=1) ----
    viewmats = _viewmat_world_to_cam(R, t)[None, :].contiguous()  # [1,4,4]
    Ks = K[None, :].contiguous()                                  # [1,3,3]

    # ---- Background ----
    # IMPORTANT:
    # Your current gsplat build expects per-camera background colors (C, 3) for this rasterization API.
    # With C=1, shape must be (1, 3). Do NOT expand to (H, W, 3).
    backgrounds = torch.tensor(bg_color, device=device, dtype=dtype).view(1, 3).contiguous()  # [1,3]

    # ---- Rasterize ----
    out = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(W),
        height=int(H),
        backgrounds=backgrounds,
        packed=False,          # <- key: avoid packed path shape assertion mismatch
        render_mode="RGB",
    )

    # ---- Output compatibility ----
    # Depending on version, return could be:
    # (render_colors, render_alphas, meta) OR ((render_colors, render_alphas), meta) etc.
    render_colors = None

    if isinstance(out, tuple):
        # case A: (colors, alphas, meta)
        if len(out) >= 1 and torch.is_tensor(out[0]):
            render_colors = out[0]
        # case B: ((colors, alphas), meta)
        elif (
            len(out) >= 1
            and isinstance(out[0], (tuple, list))
            and len(out[0]) >= 1
            and torch.is_tensor(out[0][0])
        ):
            render_colors = out[0][0]

    if render_colors is None:
        raise RuntimeError(
            "Unexpected gsplat.rasterization() return format; please paste the full traceback/output."
        )

    # render_colors usually: [C, H, W, 3]
    if render_colors.ndim == 4:
        img = render_colors[0]  # [H,W,3]
        img = img.permute(2, 0, 1).contiguous()  # [3,H,W]
    elif render_colors.ndim == 3 and render_colors.shape[-1] == 3:
        img = render_colors.permute(2, 0, 1).contiguous()
    else:
        raise RuntimeError(f"Unexpected rendered colors shape: {tuple(render_colors.shape)}")

    return img.clamp(0.0, 1.0)
