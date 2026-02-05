from __future__ import annotations

from pathlib import Path

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.data.keypoints_mediapipe import extract_keypoints_from_frames
from avatar_gs.smpl.fit_from_video import (
    fit_smpl_from_keypoints,
    fit_smpl_sequence_from_keypoints,
)


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

    # 1) Extract MediaPipe keypoints from frames (single fixed camera sequence)
    mp_cfg = cfg.get("mediapipe", {})
    kpt_json = out_dir / "keypoints.json"
    extract_keypoints_from_frames(
        frames_dir=frames_dir,
        out_json=kpt_json,
        static_image_mode=bool(mp_cfg.get("static_image_mode", False)),
        model_complexity=int(mp_cfg.get("model_complexity", 1)),
        min_detection_confidence=float(mp_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )
    logger.info(f"Saved keypoints: {kpt_json}")

    # 2) Fit SMPL using extracted keypoints (requires smplx model files)
    stage1_cfg = cfg["stage1"]
    fit_mode = str(stage1_cfg.get("fit_mode", "sequence")).lower()
    common_kwargs = dict(
        keypoints_json=kpt_json,
        out_dir=out_dir,
        K_list=cfg["camera"]["K"],
        smpl_model_path=stage1_cfg["smpl_model_path"],
        smpl_model_type=str(stage1_cfg.get("smpl_model_type", "smpl")),
        smpl_gender=str(stage1_cfg.get("smpl_gender", "neutral")),
        fit_frames=int(stage1_cfg.get("fit_frames", 120)),
        fit_iters=int(stage1_cfg.get("fit_iters", 2000)),
        fit_lr=float(stage1_cfg.get("fit_lr", 3e-2)),
        export_smooth_iters=int(stage1_cfg.get("export_smooth_iters", 0)),
        export_smooth_alpha=float(stage1_cfg.get("export_smooth_alpha", 0.1)),
    )
    if fit_mode == "static":
        fit_smpl_from_keypoints(**common_kwargs)
    else:
        fit_smpl_sequence_from_keypoints(
            **common_kwargs,
            frame_batch=int(stage1_cfg.get("frame_batch", 16)),
            temporal_pose_weight=float(stage1_cfg.get("temporal_pose_weight", 5e-4)),
            temporal_orient_weight=float(stage1_cfg.get("temporal_orient_weight", 1e-3)),
            temporal_trans_weight=float(stage1_cfg.get("temporal_trans_weight", 5e-3)),
            optimize_betas=bool(stage1_cfg.get("optimize_betas", False)),
            unfreeze_betas_iter=int(stage1_cfg.get("unfreeze_betas_iter", -1)),
            betas_l2_weight=float(stage1_cfg.get("betas_l2_weight", 5e-2)),
            betas_abs_max=float(stage1_cfg.get("betas_abs_max", 3.0)),
            shape_from_previous_frames=bool(stage1_cfg.get("shape_from_previous_frames", True)),
            final_pose_from_last_frame=bool(stage1_cfg.get("final_pose_from_last_frame", True)),
            last_frame_refine_iters=int(stage1_cfg.get("last_frame_refine_iters", 300)),
            upright_prior_weight=float(stage1_cfg.get("upright_prior_weight", 1e-2)),
            knee_hinge_weight=float(stage1_cfg.get("knee_hinge_weight", 2e-3)),
            knee_direction_weight=float(stage1_cfg.get("knee_direction_weight", 5e-3)),
            knee_min_flex=float(stage1_cfg.get("knee_min_flex", 0.02)),
            left_knee_flex_sign=float(stage1_cfg.get("left_knee_flex_sign", 1.0)),
            right_knee_flex_sign=float(stage1_cfg.get("right_knee_flex_sign", 1.0)),
            enforce_export_upright=bool(stage1_cfg.get("enforce_export_upright", True)),
            export_center_xy=bool(stage1_cfg.get("export_center_xy", True)),
            export_on_floor=bool(stage1_cfg.get("export_on_floor", True)),
            export_meshes=bool(stage1_cfg.get("export_meshes", False)),
        )
    logger.info(f"Saved SMPL outputs (fit_mode={fit_mode})")

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
