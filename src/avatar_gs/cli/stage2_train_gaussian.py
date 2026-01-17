from __future__ import annotations
from pathlib import Path
import json

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.init_from_smpl import init_gaussians_from_mesh_3d
from avatar_gs.gaussian.train_loop import train_gaussians_3d_learnable_camera


def main():
    parser = parse_args_base("Stage2: 3D Gaussian Splatting training with learnable camera (no COLMAP)")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage2_train_gaussian", str(log_dir / "stage2_train_gaussian.log"))

    stage1_dir = project_root / cfg["stage2"]["stage1_dir"]
    out_dir = ensure_dir(project_root / cfg["stage2"]["out_dir"])

    mesh_obj = stage1_dir / "smpl_mesh.obj"
    cam = read_json(stage1_dir / "cameras.json")

    frames_dir = project_root / cfg["data"]["frames_dir"]
    n_frames = int(cfg["stage2"].get("n_frames", 20))

    frames = sorted([p.name for p in frames_dir.glob("*.jpg")])[:n_frames]
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    logger.info(f"Mesh: {mesh_obj}")
    logger.info(f"Frames: {len(frames)} used for training")
    logger.info(f"Out dir: {out_dir}")

    g = init_gaussians_from_mesh_3d(
        mesh_obj=mesh_obj,
        n_keep=int(cfg["stage2"].get("n_gaussians", 3000)),
        seed=int(cfg.get("seed", 42)),
    )

    ckpt_path = out_dir / "gaussians3d.pt"
    train_gaussians_3d_learnable_camera(
        g=g,
        frames_dir=frames_dir,
        frames_list=frames,
        K_list=cam["K"],
        out_ckpt=ckpt_path,
        iters=int(cfg["stage2"].get("iters", 2000)),
        lr_gauss=float(cfg["stage2"].get("lr_gauss", 1e-2)),
        lr_cam=float(cfg["stage2"].get("lr_cam", 1e-3)),
    )

    logger.info(f"Saved checkpoint: {ckpt_path}")
    save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")


if __name__ == "__main__":
    main()
