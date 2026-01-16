from __future__ import annotations
from pathlib import Path

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.init_from_smpl import init_gaussians_from_mesh_2d
from avatar_gs.gaussian.train_loop import train_gaussians_on_first_frame


def main():
    parser = parse_args_base("Stage2: init structured gaussians + optimize (dummy 2D splat)")
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
    first_frame = frames_dir / cfg["stage2"]["first_frame_name"]

    logger.info(f"Mesh: {mesh_obj}")
    logger.info(f"First frame: {first_frame}")
    logger.info(f"Out dir: {out_dir}")

    gauss = init_gaussians_from_mesh_2d(
        mesh_obj=mesh_obj,
        K_list=cam["K"],
        t_list=cam["t"],
        n_keep=int(cfg["stage2"].get("n_gaussians", 800)),
        seed=int(cfg.get("seed", 42)),
    )

    ckpt_path = out_dir / "gaussians.pt"
    train_gaussians_on_first_frame(
        gauss=gauss,
        frame_path=first_frame,
        out_ckpt=ckpt_path,
        iters=int(cfg["stage2"].get("iters", 200)),
        lr=float(cfg["stage2"].get("lr", 1e-2)),
    )

    logger.info(f"Saved gaussians checkpoint: {ckpt_path}")
    save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")


if __name__ == "__main__":
    main()
