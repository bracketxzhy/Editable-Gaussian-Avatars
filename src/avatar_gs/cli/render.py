from __future__ import annotations
from pathlib import Path
import cv2
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.gaussian.train_loop import axis_angle_to_rotmat


def main():
    parser = parse_args_base("Render 3D gaussians checkpoint -> image")
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)

    root = Path(cfg.get("project_root", ".")).resolve()
    logger = setup_logger("render3d", str(root / cfg.get("log_dir", "outputs/logs") / "render3d.log"))

    stage2_dir = root / cfg["render"]["stage2_dir"]
    out_dir = ensure_dir(root / cfg["render"]["out_dir"])
    ckpt = torch.load(stage2_dir / "gaussians3d.pt", map_location="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = ckpt["gaussians"]
    g = Gaussian3D.load_state_dict(sd).to(device)

    K = ckpt["K"].to(device)
    cam_w = ckpt["cam_w"].to(device)
    cam_t = ckpt["cam_t"].to(device)

    H = int(cfg["render"]["H"])
    W = int(cfg["render"]["W"])

    # render first trained frame
    fi = 0
    R = axis_angle_to_rotmat(cam_w[fi])
    t = cam_t[fi]

    img = render_gaussians_3d(g, K, R, t, H, W).detach().clamp(0, 1)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    out_path = out_dir / "render3d_000000.png"
    cv2.imwrite(str(out_path), bgr)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
