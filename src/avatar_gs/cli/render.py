from __future__ import annotations

from pathlib import Path
import cv2
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, resolve_path
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read image to (3,H,W) float in [0,1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle -> R."""
    theta = torch.norm(w) + 1e-8
    k = w / theta
    kx, ky, kz = k[0], k[1], k[2]
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    K = torch.stack(
        [
            torch.stack([z, -kz, ky]),
            torch.stack([kz, z, -kx]),
            torch.stack([-ky, kx, z]),
        ],
        dim=0,
    )
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return I + s * K + (1.0 - c) * (K @ K)


def save_rgb(path: str | Path, rgb_chw: torch.Tensor) -> None:
    """Save (3,H,W) RGB float [0,1] as png/jpg, preserving exact H/W."""
    img = (rgb_chw.detach().clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
    bgr = img[..., ::-1]  # BGR
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _resolve_stage2_ckpt(stage2_dir: Path) -> Path:
    """
    Find a checkpoint file inside stage2_dir.
    Tries common filenames first, then falls back to any .pt found.
    """
    common = [
        stage2_dir / "gaussians.pt",
        stage2_dir / "gaussians3d.pt",
        stage2_dir / "ckpt.pt",
    ]
    for p in common:
        if p.exists():
            return p

    pts = sorted(stage2_dir.glob("*.pt"))
    if pts:
        return pts[-1]

    ckpt_dir = stage2_dir / "checkpoints"
    if ckpt_dir.exists():
        pts2 = sorted(ckpt_dir.glob("*.pt"))
        if pts2:
            return pts2[-1]

    raise FileNotFoundError(f"No checkpoint (*.pt) found under: {stage2_dir}")


def main():
    parser = parse_args_base("Render images from Stage2 gaussian checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("render", str(log_dir / "render.log"))

    render_cfg = cfg["render"]
    stage2_dir = project_root / Path(render_cfg.get("stage2_dir", "outputs/stage2"))
    out_dir = ensure_dir(project_root / Path(render_cfg.get("out_dir", "outputs/renders")))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoint path: optional explicit, otherwise auto-resolve
    ckpt_path = render_cfg.get("ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).strip():
        ckpt_path = Path(resolve_path(str(ckpt_path)))
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
    else:
        ckpt_path = _resolve_stage2_ckpt(stage2_dir)

    logger.info(f"Stage2 dir: {stage2_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Out dir: {out_dir}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Load gaussians
    #g = Gaussian3D()
    #g.load_state_dict(ckpt["gaussians"])
    #g = g.to(device)

    # Load gaussians (Gaussian3D requires tensors in __init__)
    gsd = ckpt["gaussians"]  # this is g.state_dict() saved in stage2

    required = ["xyz", "log_scales", "quat", "opacity_logit", "rgb_logit"]
    missing = [k for k in required if k not in gsd]
    if missing:
      raise KeyError(f"Checkpoint gaussians missing keys: {missing}. Available keys: {list(gsd.keys())}")

    g = Gaussian3D(
      xyz=gsd["xyz"].to(device),
      log_scales=gsd["log_scales"].to(device),
      quat=gsd["quat"].to(device),
      opacity_logit=gsd["opacity_logit"].to(device),
      rgb_logit=gsd["rgb_logit"].to(device),
    )

    # Intrinsics
    K = torch.tensor(ckpt["K"], dtype=torch.float32, device=device).view(3, 3)

    # Cameras and frames
    frames = ckpt.get("frames", None)
    if frames is None or len(frames) == 0:
        raise KeyError("Checkpoint missing 'frames'. Cannot render without frame list.")

    cam_w = ckpt.get("cam_w", None)
    cam_t = ckpt.get("cam_t", None)
    cam_w = torch.tensor(cam_w, dtype=torch.float32, device=device) if cam_w is not None else None
    cam_t = torch.tensor(cam_t, dtype=torch.float32, device=device) if cam_t is not None else None

    # frames_dir: optional in config, otherwise default to data/frames/input
    frames_dir = render_cfg.get("frames_dir", None)
    if frames_dir is not None and str(frames_dir).strip():
        frames_dir = Path(resolve_path(str(frames_dir)))
        if not frames_dir.is_absolute():
            frames_dir = project_root / frames_dir
    else:
        frames_dir = project_root / Path("data/frames/input")

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    # How many frames to render
    max_frames = int(render_cfg.get("max_frames", -1))
    to_render = len(frames) if max_frames <= 0 else min(len(frames), max_frames)

    # Background (default white)
    bg = render_cfg.get("bg", [1.0, 1.0, 1.0])
    if isinstance(bg, (list, tuple)) and len(bg) == 3:
        bg = (float(bg[0]), float(bg[1]), float(bg[2]))
    else:
        bg = (1.0, 1.0, 1.0)

    logger.info(f"Frames dir: {frames_dir}")
    logger.info(f"Will render: {to_render} frames; bg={bg}")

    for i in range(to_render):
        frame_path = frames_dir / frames[i]
        gt = read_image_rgb(frame_path).to(device)  # (3,H,W)

        # Camera for this frame
        if cam_w is None or cam_t is None:
            R = torch.eye(3, device=device, dtype=torch.float32)
            t = torch.tensor([0.0, 0.0, 2.5], device=device, dtype=torch.float32)
        else:
            R = axis_angle_to_rotmat(cam_w[i])
            t = cam_t[i]

        # IMPORTANT: infer output size from gt (do NOT hardcode H/W)
        pred = render_gaussians_3d(
            g=g,
            K=K,
            R=R,
            t=t,
            H=None,
            W=None,
            bg=bg,
            ref_image=gt,  # << output size == gt size
        )

        out_path = out_dir / f"{i:06d}.png"
        save_rgb(out_path, pred)

        if (i + 1) % 50 == 0:
            logger.info(f"Rendered {i+1}/{to_render} ...")

    logger.info("Done.")


if __name__ == "__main__":
    main()
