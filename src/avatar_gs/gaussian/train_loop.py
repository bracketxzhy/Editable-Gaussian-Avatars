from __future__ import annotations

from pathlib import Path
import cv2
import torch
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read an image and return float tensor in [0,1], shape (3,H,W)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (3,) axis-angle.
    Fully differentiable. Avoid torch.tensor([...]) with tensor entries (causes warnings).
    """
    theta = torch.norm(w) + 1e-8
    k = w / theta  # (3,)

    kx, ky, kz = k[0], k[1], k[2]
    zeros = torch.zeros((), device=w.device, dtype=w.dtype)

    # Skew-symmetric matrix of k
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky]),
            torch.stack([kz, zeros, -kx]),
            torch.stack([-ky, kx, zeros]),
        ],
        dim=0,
    )

    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    R = I + s * K + (1.0 - c) * (K @ K)
    return R


def train_gaussians_3d_learnable_camera(
    g: Gaussian3D,
    frames_dir: str | Path,
    frames_list: list[str],
    K_list: list[float],
    out_ckpt: str | Path,
    iters: int = 1000,
    lr_gauss: float = 1e-2,
    lr_cam: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # ---- New knobs (safe defaults) ----
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),  # match your white-ish dataset background
    cam_warmup_iters: int = 5000,  # freeze camera for first N iterations
    l1_weight: float = 0.25,       # add L1 to reduce blur
    use_alpha_mask: bool = True,   # suppress background learning
    alpha_threshold: float = 0.15, # mask threshold
):
    """
    Stage2: optimize 3D Gaussians + (optional) per-frame learnable camera.

    Key fixes vs the original:
    - Use white background compositing to match GT background (reduces colored speckles).
    - Freeze camera for warmup_iters, then enable camera learning (stabilizes convergence).
    - Add L1 term to reduce MSE-induced blur.
    - Random frame sampling (remove periodic bias).
    - Optional alpha-based masking to avoid fitting background.
    """

    frames_dir = Path(frames_dir)
    if len(frames_list) == 0:
        raise ValueError("frames_list is empty.")

    # Intrinsics
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)

    # Load first frame for H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    # Move gaussian to device and make trainable
    g = g.to(device)
    g.xyz = torch.nn.Parameter(g.xyz)
    g.log_scales = torch.nn.Parameter(g.log_scales)
    g.quat = torch.nn.Parameter(g.quat)
    g.opacity_logit = torch.nn.Parameter(g.opacity_logit)
    g.rgb_logit = torch.nn.Parameter(g.rgb_logit)

    # Learnable camera per-frame (axis-angle + translation)
    nF = len(frames_list)
    cam_w = torch.nn.Parameter(torch.zeros((nF, 3), device=device))
    cam_t = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 2.5]] * nF, device=device))

    # Optimizer (we will dynamically freeze/unfreeze camera by toggling requires_grad)
    opt = torch.optim.Adam(
        [
            {"params": [g.xyz, g.log_scales, g.quat, g.opacity_logit, g.rgb_logit], "lr": lr_gauss},
            {"params": [cam_w, cam_t], "lr": lr_cam},
        ]
    )

    # Preload GT frames (ok for small clips; for long clips stream from disk)
    gts = [read_image_rgb(frames_dir / fn).to(device) for fn in frames_list]

    # Helper to freeze/unfreeze cam params
    def _set_cam_trainable(flag: bool):
        cam_w.requires_grad_(flag)
        cam_t.requires_grad_(flag)

    # Warmup: freeze camera first
    _set_cam_trainable(False)

    # Optional: small robustness—randomize frame order sampling
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    for it in tqdm(range(iters), desc="Stage2 3DGS (learnable camera)"):
        opt.zero_grad(set_to_none=True)

        # Enable camera learning after warmup
        if lr_cam > 0 and it == cam_warmup_iters:
            _set_cam_trainable(True)

        # Sample a random frame index (better than it % nF)
        fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        gt = gts[fi]  # (3,H,W)

        # Camera pose
        R = axis_angle_to_rotmat(cam_w[fi])
        t = cam_t[fi]

        # Render
        if use_alpha_mask:
            pred_rgb, pred_a = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=True)
            # pred_rgb: (3,H,W), pred_a: (1,H,W)
            # Build a mask to suppress background. Use detached alpha to avoid degenerate solutions.
            mask = (pred_a.detach() > alpha_threshold).float()  # (1,H,W)
        else:
            pred_rgb = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=False)
            mask = None

        # Loss: MSE + L1 (reduces blur)
        diff = pred_rgb - gt
        l2 = (diff * diff)
        l1 = diff.abs()

        if mask is not None:
            # broadcast mask (1,H,W) -> (3,H,W)
            m = mask.expand_as(pred_rgb)
            l2 = (l2 * m).sum() / (m.sum() + 1e-6)
            l1 = (l1 * m).sum() / (m.sum() + 1e-6)
        else:
            l2 = l2.mean()
            l1 = l1.mean()

        loss = l2 + l1_weight * l1

        loss.backward()
        opt.step()

        # Mild regularization: keep parameters in a reasonable range
        with torch.no_grad():
            g.log_scales.clamp_(min=-7.0, max=0.0)      # exp range ~ [0.0009, 1]
            g.opacity_logit.clamp_(min=-8.0, max=8.0)
            # Optional: keep translations in a sane range to prevent explosion
            cam_t[:, 2].clamp_(min=0.5, max=10.0)

    # Save checkpoint
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gaussians": g.state_dict(),
            "K": K.detach().cpu(),
            "cam_w": cam_w.detach().cpu(),
            "cam_t": cam_t.detach().cpu(),
            "frames": frames_list,
            "H": H,
            "W": W,
            "bg_rgb": bg_rgb,
            "cam_warmup_iters": cam_warmup_iters,
            "l1_weight": l1_weight,
            "use_alpha_mask": use_alpha_mask,
            "alpha_threshold": alpha_threshold,
        },
        out_ckpt,
    )
