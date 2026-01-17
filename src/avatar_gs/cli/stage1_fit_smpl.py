from __future__ import annotations

from pathlib import Path

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.data.keypoints_mediapipe import extract_keypoints_from_frames
from avatar_gs.smpl.fit_from_video import fit_smpl_from_keypoints


def main():
    parser = parse_args_base("Stage1: fit SMPL from monocular frames (MediaPipe 2D keypoints + lite fitting)")
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

    # 1) Extract MediaPipe keypoints from frames (use per-frame jpgs)
    kpt_json = out_dir / "keypoints.json"
    extract_keypoints_from_frames(frames_dir, kpt_json)
    logger.info(f"Saved keypoints: {kpt_json}")

    # 2) Fit SMPL (lite) using the extracted keypoints (currently a minimal placeholder fit)
    fit_smpl_from_keypoints(
        keypoints_json=kpt_json,
        out_dir=out_dir,
        K_list=cfg["camera"]["K"],
        n_verts=int(cfg["stage1"].get("dummy_n_verts", 1200)),
    )
    logger.info("Saved smpl_params.npz and smpl_mesh.obj")

    # 3) Save a simple (single) camera definition (placeholder)
    cam = {
        "type": "pinhole",
        "K": cfg["camera"]["K"],
        "R": [1, 0, 0,
              0, 1, 0,
              0, 0, 1],
        # NOTE: In this placeholder setup, we treat this as translating the object along +z
        # to be in front of the camera (camera assumed at origin, looking towards +z).
        "t": [0, 0, 2.5],
    }
    write_json(cam, out_dir / "cameras.json")

    # Save resolved config snapshot (reproducibility)
    save_yaml(cfg, log_dir / "stage1_fit_smpl_resolved.yaml")
    logger.info("Done.")


if __name__ == "__main__":
    main()
