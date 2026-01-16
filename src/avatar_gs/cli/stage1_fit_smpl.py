from __future__ import annotations
from pathlib import Path

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything
from avatar_gs.smpl.fit_from_video import dummy_fit_smpl_from_frames


def main():
    parser = parse_args_base("Stage1: fit SMPL from monocular frames (dummy placeholder)")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage1_fit_smpl", str(log_dir / "stage1_fit_smpl.log"))

    frames_dir = project_root / cfg["data"]["frames_dir"]
    out_dir = ensure_dir(project_root / cfg["stage1"]["out_dir"])

    logger.info(f"Frames dir: {frames_dir}")
    logger.info(f"Stage1 out: {out_dir}")

    res = dummy_fit_smpl_from_frames(
        frames_dir=frames_dir,
        out_dir=out_dir,
        n_verts=int(cfg["stage1"].get("dummy_n_verts", 1200)),
        seed=int(cfg.get("seed", 42)),
    )

    # 伪相机：单目先用固定相机（后续你可以替换成 SLAM/colmap）
    cam = {
        "type": "pinhole",
        "K": cfg["camera"]["K"],
        "R": [1,0,0, 0,1,0, 0,0,1],
        "t": [0,0,2.5],   # 相机在 +z
    }
    write_json(cam, out_dir / "cameras.json")

    logger.info(f"Saved: {res}")
    save_yaml(cfg, log_dir / "stage1_fit_smpl_resolved.yaml")


if __name__ == "__main__":
    main()
