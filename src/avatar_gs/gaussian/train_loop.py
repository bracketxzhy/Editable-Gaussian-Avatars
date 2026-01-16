from __future__ import annotations
from pathlib import Path
import cv2
import torch
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian2D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_2d

def read_image_rgb(path: str | Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0  # (H,W,3)
    return x.permute(2, 0, 1).contiguous()     # (3,H,W)

def train_gaussians_on_first_frame(
    gauss: Gaussian2D,
    frame_path: str | Path,
    out_ckpt: str | Path,
    iters: int = 200,
    lr: float = 1e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    gt = read_image_rgb(frame_path).to(device)
    _, H, W = gt.shape

    g = gauss.to(device)
    # 让部分参数可训练
    g.mu = torch.nn.Parameter(g.mu)
    g.sigma = torch.nn.Parameter(g.sigma)
    g.alpha = torch.nn.Parameter(g.alpha)
    g.rgb = torch.nn.Parameter(g.rgb)

    opt = torch.optim.Adam([g.mu, g.sigma, g.alpha, g.rgb], lr=lr)

    for _ in tqdm(range(iters), desc="Stage2 (dummy) optimize"):
        opt.zero_grad(set_to_none=True)
        pred = render_gaussians_2d(g, H, W)
        loss = torch.mean((pred - gt) ** 2)
        loss.backward()
        opt.step()

    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g.state_dict(), out_ckpt)
