from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from avatar_gs.data.h36m import resolve_h36m_image_paths
except Exception:
    resolve_h36m_image_paths = None
from avatar_gs.data.keypoints_mediapipe import extract_keypoints_from_frames
from avatar_gs.smpl.fit_from_video import fit_smpl_sequence_stage1
from avatar_gs.utils.cfg import apply_overrides, load_yaml, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything


def _get_first_frame_size(frames_dir: Path, frame_paths: list[Path] | None) -> tuple[int, int] | None:
    if frame_paths is not None and len(frame_paths) > 0:
        cands = [Path(frame_paths[0])]
    else:
        cands = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    for p in cands:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        return int(w), int(h)
    return None


def _parse_optional_int_list(value, name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"{name} must be a list/tuple of integers or null.")


def main():
    parser = parse_args_base("Stage1: fit SMPL/SMPL-X with VIBE-only backend")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage1_fit_smpl", str(log_dir / "stage1_fit_smpl.log"))
    # Save resolved config at run start so interrupted runs are still traceable.
    save_yaml(cfg, log_dir / "stage1_fit_smpl_resolved.yaml")

    stage1_cfg = cfg["stage1"]
    data_cfg = cfg["data"]
    frames_dir = project_root / data_cfg["frames_dir"]
    out_dir = ensure_dir(project_root / stage1_cfg.get("out_dir", "outputs/stage1"))

    logger.info(f"Frames dir: {frames_dir}")
    logger.info(f"Stage1 out: {out_dir}")
    logger.info("Stage1 backend: VIBE-only (integrated src/lib)")

    mp_cfg = cfg.get("mediapipe", {})
    kpt_smooth_cfg = stage1_cfg.get("keypoint_smooth", {})
    if not isinstance(kpt_smooth_cfg, dict):
        kpt_smooth_cfg = {}
    translation_cfg = stage1_cfg.get("translation", {})
    if not isinstance(translation_cfg, dict):
        translation_cfg = {}
    kpt_json = out_dir / "keypoints.json"
    seg_npz = out_dir / "person_masks.npz"
    crop_meta_npz = out_dir / "crop_metadata.npz"

    frame_paths: list[Path] | None = None
    if "h36m" in data_cfg:
        if resolve_h36m_image_paths is None:
            raise RuntimeError(
                "data.h36m config is provided, but avatar_gs.data.h36m is unavailable."
            )
        resolved = resolve_h36m_image_paths(project_root=project_root, data_cfg=data_cfg)
        if len(resolved) <= 0:
            raise RuntimeError("No H36M images resolved from data.h36m config.")
        frame_paths = [Path(p) for p in resolved]
        frames_dir = frame_paths[0].parent
        logger.info(f"H36M resolved frames_dir: {frames_dir}")
        logger.info(f"H36M selected frames: {len(frame_paths)}")

    K_use = [float(x) for x in cfg["camera"]["K"]]
    if len(K_use) != 9:
        raise ValueError(f"camera.K must have 9 values, got {len(K_use)}")
    if bool(stage1_cfg.get("auto_camera_from_frames", True)):
        sz = _get_first_frame_size(frames_dir=frames_dir, frame_paths=frame_paths)
        if sz is not None:
            w, h = sz
            if bool(stage1_cfg.get("auto_camera_center", True)):
                K_use[2] = 0.5 * float(w)
                K_use[5] = 0.5 * float(h)
            if bool(stage1_cfg.get("auto_camera_scale_focal", True)):
                ref_w = max(2.0 * float(cfg["camera"]["K"][2]), 1.0)
                ref_h = max(2.0 * float(cfg["camera"]["K"][5]), 1.0)
                s = 0.5 * (float(w) / ref_w + float(h) / ref_h)
                K_use[0] = float(K_use[0]) * s
                K_use[4] = float(K_use[4]) * s
            logger.info(f"Auto camera K from frames ({w}x{h}): {K_use}")

    extract_keypoints_from_frames(
        frames_dir=frames_dir,
        out_json=kpt_json,
        frame_paths=frame_paths,
        static_image_mode=bool(mp_cfg.get("static_image_mode", False)),
        model_complexity=int(mp_cfg.get("model_complexity", 1)),
        min_detection_confidence=float(mp_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
        enable_segmentation=bool(mp_cfg.get("enable_segmentation", True)),
        segmentation_threshold=float(mp_cfg.get("segmentation_threshold", 0.5)),
        two_stage_crop=bool(mp_cfg.get("two_stage_crop", True)),
        two_stage_crop_scale=float(mp_cfg.get("two_stage_crop_scale", 2.0)),
        two_stage_min_size_px=float(mp_cfg.get("two_stage_min_size_px", 160.0)),
        two_stage_min_visibility=float(mp_cfg.get("two_stage_min_visibility", 0.1)),
        two_stage_use_prev_bbox=bool(mp_cfg.get("two_stage_use_prev_bbox", True)),
        two_stage_center_on_mask=bool(mp_cfg.get("two_stage_center_on_mask", True)),
        two_stage_crop_stable_momentum=float(mp_cfg.get("two_stage_crop_stable_momentum", 0.65)),
        save_segmentation_npz=seg_npz,
        save_crop_metadata_npz=crop_meta_npz,
        enable_hand_landmarks=bool(mp_cfg.get("enable_hand_landmarks", True)),
        hand_max_num_hands=int(mp_cfg.get("hand_max_num_hands", 2)),
        hand_model_complexity=int(mp_cfg.get("hand_model_complexity", 1)),
        hand_min_detection_confidence=float(mp_cfg.get("hand_min_detection_confidence", 0.35)),
        hand_min_tracking_confidence=float(mp_cfg.get("hand_min_tracking_confidence", 0.45)),
        keypoint_smooth_enable=bool(kpt_smooth_cfg.get("enable", False)),
        keypoint_smooth_method=str(kpt_smooth_cfg.get("method", "one_euro")),
        keypoint_smooth_fps=float(kpt_smooth_cfg.get("fps", 30.0)),
        keypoint_smooth_min_cutoff=float(kpt_smooth_cfg.get("min_cutoff", 1.0)),
        keypoint_smooth_beta=float(kpt_smooth_cfg.get("beta", 0.3)),
        keypoint_smooth_gaussian_kernel_size=int(kpt_smooth_cfg.get("gaussian_kernel_size", 7)),
        keypoint_smooth_gaussian_sigma=float(kpt_smooth_cfg.get("gaussian_sigma", 1.5)),
    )
    logger.info(f"Saved keypoints: {kpt_json}")
    if seg_npz.exists():
        logger.info(f"Saved person masks: {seg_npz}")

    if torch.cuda.is_available() and bool(stage1_cfg.get("enable_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(stage1_cfg.get("cudnn_benchmark", True))

    smpl_model_type = str(stage1_cfg.get("smpl_model_type", "smplx")).strip().lower()
    if smpl_model_type not in ("smpl", "smplx"):
        raise ValueError(
            f"stage1.smpl_model_type must be 'smpl' or 'smplx', got: {smpl_model_type}"
        )

    vibe_batch_size = int(stage1_cfg.get("vibe_batch_size", 64))
    vibe_bbox_scale = float(stage1_cfg.get("vibe_bbox_scale", 1.1))
    vibe_use_3dpw_ckpt = bool(stage1_cfg.get("vibe_use_3dpw_ckpt", False))
    vibe_use_smooth_pose = bool(stage1_cfg.get("vibe_use_smooth_pose", True))
    vibe_smooth_min_cutoff = float(stage1_cfg.get("vibe_smooth_min_cutoff", 0.004))
    vibe_smooth_beta = float(stage1_cfg.get("vibe_smooth_beta", 0.7))
    vibe_translation_smooth_alpha = float(stage1_cfg.get("vibe_translation_smooth_alpha", 0.65))

    smpl_gender_cfg = str(stage1_cfg.get("smpl_gender", "neutral")).strip().lower()
    if smpl_gender_cfg in ("", "auto"):
        resolved_gender = str(stage1_cfg.get("vibe_default_gender", "neutral")).strip().lower()
        if resolved_gender == "":
            resolved_gender = "neutral"
    else:
        resolved_gender = smpl_gender_cfg

    export_mesh_sequence_indices = _parse_optional_int_list(
        stage1_cfg.get("export_mesh_sequence_indices", None),
        "stage1.export_mesh_sequence_indices",
    )

    fit_smpl_sequence_stage1(
        keypoints_json=kpt_json,
        frames_dir=frames_dir,
        out_dir=out_dir,
        K_list=K_use,
        smpl_model_path=stage1_cfg["smpl_model_path"],
        smpl_model_type=smpl_model_type,
        smpl_gender=resolved_gender,
        smpl_num_betas=int(stage1_cfg.get("smpl_num_betas", 10)),
        fit_frames=int(stage1_cfg.get("fit_frames", -1)),
        frame_batch=int(stage1_cfg.get("frame_batch", 32)),
        keypoint_conf_thresh=float(stage1_cfg.get("keypoint_conf_thresh", 0.02)),
        export_pose_mode=str(stage1_cfg.get("export_pose_mode", "last_frame")),
        export_mesh_tpose=bool(stage1_cfg.get("export_mesh_tpose", True)),
        export_frame_index=int(stage1_cfg.get("export_frame_index", -1)),
        export_mesh_sequence_indices=export_mesh_sequence_indices,
        export_mesh_sequence_last_only=bool(stage1_cfg.get("export_mesh_sequence_last_only", False)),
        export_legacy_mesh_objs=bool(stage1_cfg.get("export_legacy_mesh_objs", False)),
        export_meshes=bool(stage1_cfg.get("export_meshes", True)),
        vibe_batch_size=vibe_batch_size,
        vibe_bbox_scale=vibe_bbox_scale,
        vibe_use_3dpw_ckpt=vibe_use_3dpw_ckpt,
        vibe_use_smooth_pose=vibe_use_smooth_pose,
        vibe_smooth_min_cutoff=vibe_smooth_min_cutoff,
        vibe_smooth_beta=vibe_smooth_beta,
        vibe_translation_smooth_alpha=vibe_translation_smooth_alpha,
        show_progress=bool(stage1_cfg.get("show_progress", True)),
        quality_report_enable=bool(stage1_cfg.get("quality_report_enable", True)),
        quality_report_conf_thresh=float(stage1_cfg.get("quality_report_conf_thresh", 0.08)),
        quality_report_score_tau_px=float(stage1_cfg.get("quality_report_score_tau_px", 18.0)),
        translation_min_required_points=int(translation_cfg.get("min_required_points", 4)),
        translation_min_projected_height_px=float(translation_cfg.get("min_projected_height_px", 8.0)),
        translation_fallback_projected_height_px=float(
            translation_cfg.get("fallback_projected_height_px", 64.0)
        ),
        translation_fallback_body_height_m=float(translation_cfg.get("fallback_body_height_m", 1.7)),
        translation_tz_min=float(translation_cfg.get("tz_min", 1.0)),
        translation_tz_max=float(translation_cfg.get("tz_max", 12.0)),
        translation_prev_tz_momentum=float(translation_cfg.get("prev_tz_momentum", 0.75)),
        translation_prev_xy_momentum=float(translation_cfg.get("prev_xy_momentum", 0.7)),
    )
    logger.info("Saved SMPL outputs (backend=stage1_vibe)")
    logger.info(f"Stage1 model_type: {smpl_model_type}")

    quality_report_path = out_dir / "fit_quality_report.json"
    if quality_report_path.exists():
        try:
            with open(quality_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            g = report.get("global", {})
            logger.info(
                "Fit score global: %.3f, last_frame: %.3f",
                float(g.get("score_0_1", 0.0)),
                float(g.get("last_frame_score_0_1", 0.0)),
            )
            parts = report.get("parts", {})
            if isinstance(parts, dict):
                for part_name in sorted(parts.keys()):
                    pi = parts.get(part_name, {})
                    if not isinstance(pi, dict):
                        continue
                    if str(pi.get("status", "")) != "ok":
                        logger.info("Part score %-10s: unavailable", str(part_name))
                        continue
                    logger.info(
                        "Part score %-10s: %.3f, last: %.3f, err=%.2fpx",
                        str(part_name),
                        float(pi.get("score_0_1", 0.0)),
                        float(pi.get("last_frame_score_0_1", 0.0)),
                        float(pi.get("mean_reproj_error_px", 0.0)),
                    )
        except Exception as e:
            logger.warning(f"Failed to read fit quality report: {e}")

    cam_K = list(K_use)
    params_seq_path = out_dir / "smpl_params_sequence.npz"
    if params_seq_path.exists():
        try:
            seq = np.load(str(params_seq_path), allow_pickle=True)
            if "camera_K_refined" in seq.files:
                K_ref = np.asarray(seq["camera_K_refined"], dtype=np.float32).reshape(3, 3)
                if np.isfinite(K_ref).all():
                    cam_K = K_ref.reshape(-1).astype(float).tolist()
                    logger.info(f"Using refined camera K from smpl_params_sequence.npz: {cam_K}")
        except Exception as e:
            logger.warning(f"Failed to load refined camera K, fallback to initial K: {e}")

    cam = {
        "type": "pinhole",
        "K": cam_K,
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "t": [0, 0, 2.5],
    }
    write_json(cam, out_dir / "cameras.json")

    logger.info("Done.")


if __name__ == "__main__":
    main()



