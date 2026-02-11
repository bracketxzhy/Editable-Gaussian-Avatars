from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import gsplat  # pip package "gsplat"

from avatar_gs.gaussian.structured_gaussian import Gaussian3D


def _attach_cached_gsplat_cuda_extension() -> None:
    """
    Windows workaround:
    - If `gsplat` package does not expose `gsplat.csrc`, gsplat falls back to
      JIT compile in `gsplat.cuda._backend`, which requires `cl.exe`.
    - Reuse already-built torch extension (`gsplat_cuda.pyd`) from
      torch_extensions cache when available, avoiding recompile.
    """
    if getattr(gsplat, "csrc", None) is not None:
        return

    try:
        from torch.utils.cpp_extension import _get_build_directory, _import_module_from_library
    except Exception:
        return

    try:
        build_dir = Path(_get_build_directory("gsplat_cuda", verbose=False))
    except Exception:
        return

    if not build_dir.exists():
        return
    if not ((build_dir / "gsplat_cuda.pyd").exists() or (build_dir / "gsplat_cuda.so").exists()):
        return

    try:
        mod = _import_module_from_library("gsplat_cuda", str(build_dir), True)
    except Exception:
        return

    sys.modules.setdefault("gsplat.csrc", mod)
    setattr(gsplat, "csrc", mod)


_attach_cached_gsplat_cuda_extension()


def _ensure_hwc_background(
    bg: Union[None, Tuple[float, float, float], torch.Tensor],
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns background tensor of shape (H, W, 3) on device.
    - bg is None: black background
    - bg is (3,) tuple/list/tensor: broadcast to (H,W,3)
    - bg is (H,W,3) tensor: used as-is
    """
    if bg is None:
        return torch.zeros((H, W, 3), device=device, dtype=dtype)

    if torch.is_tensor(bg):
        bgt = bg.to(device=device, dtype=dtype)
        if bgt.ndim == 1 and bgt.numel() == 3:
            return bgt.view(1, 1, 3).expand(H, W, 3).contiguous()
        if bgt.ndim == 3 and tuple(bgt.shape) == (H, W, 3):
            return bgt.contiguous()
        raise ValueError(f"bg tensor must be (3,) or (H,W,3). Got {tuple(bgt.shape)}")

    if isinstance(bg, (tuple, list)) and len(bg) == 3:
        bgt = torch.tensor(bg, device=device, dtype=dtype).view(1, 1, 3)
        return bgt.expand(H, W, 3).contiguous()

    raise ValueError(f"Unsupported bg type: {type(bg)}")


def _ensure_R(R: torch.Tensor) -> torch.Tensor:
    """Accept (3,3) or (B,3,3) -> return (B,3,3)."""
    if R.ndim == 2 and R.shape == (3, 3):
        return R[None, ...].contiguous()
    if R.ndim == 3 and R.shape[1:] == (3, 3):
        return R.contiguous()
    raise ValueError(f"R must be (3,3) or (B,3,3), got {tuple(R.shape)}")


def _ensure_t(t: torch.Tensor) -> torch.Tensor:
    """Accept (3,) or (B,3) -> return (B,3)."""
    if t.ndim == 1 and t.numel() == 3:
        return t[None, :].contiguous()
    if t.ndim == 2 and t.shape[1] == 3:
        return t.contiguous()
    raise ValueError(f"t must be (3,) or (B,3), got {tuple(t.shape)}")


def _ensure_K(K: torch.Tensor) -> torch.Tensor:
    """Accept (3,3) or (B,3,3) -> return (B,3,3)."""
    if K.ndim == 2 and K.shape == (3, 3):
        return K[None, ...].contiguous()
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        return K.contiguous()
    raise ValueError(f"K must be (3,3) or (B,3,3), got {tuple(K.shape)}")


def _make_viewmats(Rs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """
    Build viewmats (world->cam) of shape (B, 4, 4) from:
      Rs: (B,3,3)
      ts: (B,3)
    """
    B = Rs.shape[0]
    view = torch.eye(4, device=Rs.device, dtype=Rs.dtype).unsqueeze(0).repeat(B, 1, 1)
    view[:, :3, :3] = Rs
    view[:, :3, 3] = ts
    return view


def _squeeze_bc(x: torch.Tensor) -> torch.Tensor:
    """
    Squeeze leading batch/cam dims if they are 1.
    Typical gsplat outputs:
      (B, C, H, W, 3) or (B, C, H, W, 1)
    We want:
      (H, W, 3) or (H, W, 1)
    """
    while x.ndim >= 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    while x.ndim >= 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    return x


def _infer_hw_from_ref(
    ref_image: Optional[torch.Tensor],
    bg: Optional[Union[Tuple[float, float, float], torch.Tensor]],
) -> Tuple[int, int]:
    """
    Infer (H,W) from:
    - ref_image: (3,H,W) or (H,W,3)
    - bg tensor: (H,W,3)
    """
    if ref_image is not None:
        if not torch.is_tensor(ref_image):
            raise ValueError("ref_image must be a torch.Tensor when provided.")
        if ref_image.ndim == 3:
            if ref_image.shape[0] == 3:
                return int(ref_image.shape[1]), int(ref_image.shape[2])  # (H,W) from (3,H,W)
            if ref_image.shape[2] == 3:
                return int(ref_image.shape[0]), int(ref_image.shape[1])  # (H,W) from (H,W,3)
        raise ValueError(f"ref_image must be (3,H,W) or (H,W,3). Got {tuple(ref_image.shape)}")

    if torch.is_tensor(bg):
        bgt = bg
        if bgt.ndim == 3 and bgt.shape[2] == 3:
            return int(bgt.shape[0]), int(bgt.shape[1])

    raise ValueError(
        "H/W were not provided and could not be inferred. "
        "Pass H and W explicitly, or pass ref_image (3,H,W)/(H,W,3), or pass bg tensor (H,W,3)."
    )


def render_gaussians_3d(
    g: Gaussian3D,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: Optional[int],
    W: Optional[int],
    bg: Optional[Union[Tuple[float, float, float], torch.Tensor]] = (0.0, 0.0, 0.0),
    near_plane: float = 0.01,
    far_plane: float = 1e5,
    return_alpha: bool = False,
    packed: bool = True,
    ref_image: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Render Gaussian3D with gsplat (v1.x compatible).

    If H/W are None, infer them from ref_image (recommended) or bg tensor shape.
    This enables "render output size equals input frame size" without hard-coding H/W.
    """
    device = g.xyz.device
    dtype = g.xyz.dtype

    if H is None or W is None:
        H, W = _infer_hw_from_ref(ref_image=ref_image, bg=bg)

    if not isinstance(H, int) or not isinstance(W, int) or H <= 0 or W <= 0:
        raise ValueError(f"Invalid H/W: H={H}, W={W}")

    Ks_b = _ensure_K(K.to(device=device, dtype=dtype))  # (B,3,3)
    Rs_b = _ensure_R(R.to(device=device, dtype=dtype))  # (B,3,3)
    ts_b = _ensure_t(t.to(device=device, dtype=dtype))  # (B,3)

    viewmats_b = _make_viewmats(Rs_b, ts_b)             # (B,4,4)

    # gsplat expects explicit camera dimension C
    viewmats = viewmats_b[:, None, :, :]  # (B,1,4,4)
    Ks = Ks_b[:, None, :, :]              # (B,1,3,3)

    # Prepare gaussians
    means = g.xyz[None, ...].contiguous()                      # (1,N,3)
    scales = g.scales()[None, ...].contiguous()                # (1,N,3)
    quats = g.normalized_quat()[None, ...].contiguous()        # (1,N,4)
    opacities = g.opacity().view(-1)[None, ...].contiguous()   # (1,N)
    colors = g.rgb()[None, ...].contiguous()                   # (1,N,3)

    render_colors, render_alphas, _meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(W),
        height=int(H),
        near_plane=near_plane,
        far_plane=far_plane,
        packed=packed,
        backgrounds=None,     # avoid strict asserts
        render_mode="RGB",
    )

    render_colors = _squeeze_bc(render_colors)
    render_alphas = _squeeze_bc(render_alphas)

    if render_colors.ndim != 3 or render_colors.shape[-1] != 3:
        raise RuntimeError(f"Unexpected render_colors shape: {tuple(render_colors.shape)}")
    if render_alphas.ndim != 3 or render_alphas.shape[-1] != 1:
        raise RuntimeError(f"Unexpected render_alphas shape: {tuple(render_alphas.shape)}")

    if bg is not None:
        bg_hwc = _ensure_hwc_background(bg, int(H), int(W), device=device, dtype=dtype)
        render_colors = render_colors + (1.0 - render_alphas) * bg_hwc
        render_colors = render_colors.clamp(0.0, 1.0)

    rgb = render_colors.permute(2, 0, 1).contiguous()   # (3,H,W)
    alpha = render_alphas.permute(2, 0, 1).contiguous() # (1,H,W)

    if return_alpha:
        return rgb, alpha
    return rgb
