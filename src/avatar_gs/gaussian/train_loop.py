from __future__ import annotations
from pathlib import Path
import json
import cv2
import torch
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def read_image_rgb(path: str | Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    # w: (3,) axis-angle
    theta = torch.norm(w) + 1e-8
    k = w / theta
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], device=w.device, dtype=w.dtype)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
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
):
    frames_dir = Path(frames_dir)
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)

    # load first frame to get H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    g = g.to(device)
    # make params trainable
    g.xyz = torch.nn.Parameter(g.xyz)
    g.log_scales = torch.nn.Parameter(g.log_scales)
    g.quat = torch.nn.Parameter(g.quat)
    g.opacity_logit = torch.nn.Parameter(g.opacity_logit)
    g.rgb_logit = torch.nn.Parameter(g.rgb_logit)

    # learnable camera per-frame (axis-angle + translation)
    nF = len(frames_list)
    cam_w = torch.nn.Parameter(torch.zeros((nF, 3), device=device))
    cam_t = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 2.5]] * nF, device=device))

    opt = torch.optim.Adam(
        [
            {"params": [g.xyz, g.log_scales, g.quat, g.opacity_logit, g.rgb_logit], "lr": lr_gauss},
            {"params": [cam_w, cam_t], "lr": lr_cam},
        ]
    )

    # pre-load GT (small clips only; for longer, stream)
    gts = [read_image_rgb(frames_dir / fn).to(device) for fn in frames_list]

    for it in tqdm(range(iters), desc="Stage2 3DGS (learnable camera)"):
        opt.zero_grad(set_to_none=True)

        # sample a frame
        fi = it % nF
        gt = gts[fi]

        R = axis_angle_to_rotmat(cam_w[fi])
        t = cam_t[fi]

        pred = render_gaussians_3d(g, K, R, t, H, W)
        loss = torch.mean((pred - gt) ** 2)
        loss.backward()
        opt.step()

        # mild regularization: keep scales reasonable
        with torch.no_grad():
            g.log_scales.clamp_(min=-7.0, max=0.0)  # exp range ~ [0.0009,1]
            g.opacity_logit.clamp_(min=-8.0, max=8.0)

    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gaussians": g.state_dict(),
            "K": K.detach().cpu(),
            "cam_w": cam_w.detach().cpu(),
            "cam_t": cam_t.detach().cpu(),
            "frames": frames_list,
        },
        out_ckpt,
    )
