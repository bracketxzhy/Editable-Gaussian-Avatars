from __future__ import annotations
from pathlib import Path
import cv2
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base
from avatar_gs.utils.io import ensure_dir, write_json, read_json
from avatar_gs.utils.log import setup_logger

from avatar_gs.gaussian.structured_gaussian import Gaussian2D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_2d


def main():
    parser = parse_args_base("Render gaussians (dummy 2D splat) -> image")
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)

    project_root = Path(cfg.get("project_root", ".")).resolve()
    logger = setup_logger("render", str(project_root / cfg.get("log_dir", "outputs/logs") / "render.log"))

    stage2_dir = project_root / cfg["render"]["stage2_dir"]
    stage1_dir = project_root / cfg["render"]["stage1_dir"]
    out_dir = ensure_dir(project_root / cfg["render"]["out_dir"])

    cam = read_json(stage1_dir / "cameras.json")
    W = int(cfg["render"]["W"])
    H = int(cfg["render"]["H"])

    sd = torch.load(stage2_dir / "gaussians.pt", map_location="cpu")
    g = Gaussian2D.load_state_dict(sd)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = g.to(device)

    img = render_gaussians_2d(g, H, W).detach().clamp(0, 1)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    out_path = out_dir / "render_000000.png"
    cv2.imwrite(str(out_path), bgr)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
