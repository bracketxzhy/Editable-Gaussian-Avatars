from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.data.keypoints_mediapipe import extract_keypoints_from_frames
from avatar_gs.data.h36m import resolve_h36m_image_paths
from avatar_gs.smpl.fit_from_video import (
    fit_smpl_from_keypoints,
    fit_smpl_sequence_from_keypoints,
    select_smpl_gender_from_keypoints,
)


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

    stage1_cfg = cfg["stage1"]

    # 1) Extract MediaPipe keypoints from frames.
    # For Human3.6M-style multi-view exports, you can provide data.h36m.* to select a subset.
    mp_cfg = cfg.get("mediapipe", {})
    kpt_smooth_cfg = stage1_cfg.get("keypoint_smooth", {})
    if not isinstance(kpt_smooth_cfg, dict):
        kpt_smooth_cfg = {}
    kpt_json = out_dir / "keypoints.json"
    seg_npz = out_dir / "person_masks.npz"
    crop_meta_npz = out_dir / "crop_metadata.npz"
    frame_paths = None
    if "h36m" in cfg.get("data", {}):
        frame_paths = resolve_h36m_image_paths(project_root=project_root, data_cfg=cfg["data"])
        frames_dir = frame_paths[0].parent
        logger.info(f"H36M resolved frames_dir: {frames_dir}")
        logger.info(f"H36M selected frames: {len(frame_paths)}")

    K_use = [float(x) for x in cfg["camera"]["K"]]
    if len(K_use) != 9:
        raise ValueError(f"camera.K must have 9 values, got {len(K_use)}")
    if bool(stage1_cfg.get("auto_camera_from_frames", True)):
        sz = _get_first_frame_size(frames_dir=frames_dir, frame_paths=([Path(p) for p in frame_paths] if frame_paths is not None else None))
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

    # 2) Fit SMPL/SMPL-X using extracted keypoints.
    fit_mode = str(stage1_cfg.get("fit_mode", "sequence")).lower()
    smpl_model_type = str(stage1_cfg.get("smpl_model_type", "smplx"))
    smpl_gender_cfg = str(stage1_cfg.get("smpl_gender", "neutral"))
    smpl_num_betas = int(stage1_cfg.get("smpl_num_betas", 10))
    use_auto_gender = bool(stage1_cfg.get("auto_gender", False)) or smpl_gender_cfg.strip().lower() == "auto"
    resolved_gender = smpl_gender_cfg
    if use_auto_gender:
        probe_fit_frames = int(stage1_cfg.get("auto_gender_probe_frames", 32))
        probe_iters = int(stage1_cfg.get("auto_gender_probe_iters", 80))
        probe_lr = float(stage1_cfg.get("auto_gender_probe_lr", 2e-2))
        candidates_raw = stage1_cfg.get("auto_gender_candidates", ["female", "male", "neutral"])
        if not isinstance(candidates_raw, (list, tuple)):
            candidates_raw = ["female", "male", "neutral"]
        candidates = [str(x) for x in candidates_raw]
        if "neutral" not in [c.lower() for c in candidates]:
            candidates.append("neutral")
        resolved_gender, gender_scores = select_smpl_gender_from_keypoints(
            keypoints_json=kpt_json,
            K_list=K_use,
            smpl_model_path=stage1_cfg["smpl_model_path"],
            smpl_model_type=smpl_model_type,
            fit_frames=probe_fit_frames,
            num_betas=smpl_num_betas,
            keypoint_loss_type=str(stage1_cfg.get("keypoint_loss_type", "gmof")),
            keypoint_gmof_sigma_px=float(stage1_cfg.get("keypoint_gmof_sigma_px", 100.0)),
            keypoint_conf_thresh=float(stage1_cfg.get("keypoint_conf_thresh", 0.02)),
            keypoint_bbox_norm=bool(stage1_cfg.get("keypoint_bbox_norm", True)),
            keypoint_bbox_norm_min_px=float(stage1_cfg.get("keypoint_bbox_norm_min_px", 96.0)),
            probe_iters=probe_iters,
            probe_lr=probe_lr,
            candidates=candidates,
        )
        finite_scores = {
            str(k): float(v) for k, v in gender_scores.items()
            if np.isfinite(float(v))
        }
        sorted_items = sorted(finite_scores.items(), key=lambda kv: kv[1])
        probe_best_gender = resolved_gender
        probe_best_score = float(finite_scores.get(probe_best_gender, float("inf")))
        probe_second_score = float(sorted_items[1][1]) if len(sorted_items) > 1 else float("inf")
        neutral_score = float(finite_scores.get("neutral", float("inf")))

        auto_gender_confident_only = bool(stage1_cfg.get("auto_gender_confident_only", True))
        min_rel_gain_vs_neutral = float(stage1_cfg.get("auto_gender_min_rel_gain_vs_neutral", 0.04))
        min_rel_margin_vs_second = float(stage1_cfg.get("auto_gender_min_rel_margin_vs_second", 0.02))
        min_abs_gap_vs_neutral = float(stage1_cfg.get("auto_gender_min_abs_gap_vs_neutral", 0.0))

        decision = "probe_best"
        rel_gain_vs_neutral = 0.0
        rel_margin_vs_second = 0.0
        abs_gap_vs_neutral = 0.0
        if np.isfinite(probe_best_score) and np.isfinite(neutral_score):
            abs_gap_vs_neutral = neutral_score - probe_best_score
            rel_gain_vs_neutral = abs_gap_vs_neutral / max(abs(neutral_score), 1e-6)
        if np.isfinite(probe_best_score) and np.isfinite(probe_second_score):
            rel_margin_vs_second = (probe_second_score - probe_best_score) / max(abs(probe_second_score), 1e-6)

        if auto_gender_confident_only:
            confident_non_neutral = (
                probe_best_gender in ("female", "male")
                and np.isfinite(neutral_score)
                and rel_gain_vs_neutral >= float(min_rel_gain_vs_neutral)
                and rel_margin_vs_second >= float(min_rel_margin_vs_second)
                and abs_gap_vs_neutral >= float(min_abs_gap_vs_neutral)
            )
            if confident_non_neutral:
                resolved_gender = probe_best_gender
                decision = "confident_non_neutral"
            else:
                resolved_gender = "neutral" if ("neutral" in finite_scores) else probe_best_gender
                decision = "fallback_neutral_uncertain"

        write_json(
            {
                "strategy": "conservative_probe",
                "probe_best_gender": probe_best_gender,
                "probe_best_score": probe_best_score,
                "probe_second_score": probe_second_score,
                "neutral_score": neutral_score,
                "rel_gain_vs_neutral": float(rel_gain_vs_neutral),
                "rel_margin_vs_second": float(rel_margin_vs_second),
                "abs_gap_vs_neutral": float(abs_gap_vs_neutral),
                "auto_gender_confident_only": auto_gender_confident_only,
                "auto_gender_min_rel_gain_vs_neutral": float(min_rel_gain_vs_neutral),
                "auto_gender_min_rel_margin_vs_second": float(min_rel_margin_vs_second),
                "auto_gender_min_abs_gap_vs_neutral": float(min_abs_gap_vs_neutral),
                "decision": decision,
                "selected_gender": resolved_gender,
                "scores": {str(k): float(v) for k, v in gender_scores.items()},
                "model_type": smpl_model_type,
            },
            out_dir / "auto_gender_selection.json",
        )
        logger.info(
            "Auto gender selected: "
            f"{resolved_gender}; decision={decision}; "
            f"probe_best={probe_best_gender}; scores={gender_scores}"
        )

    export_mesh_sequence_indices = _parse_optional_int_list(
        stage1_cfg.get("export_mesh_sequence_indices", None),
        "stage1.export_mesh_sequence_indices",
    )
    common_kwargs = dict(
        keypoints_json=kpt_json,
        out_dir=out_dir,
        K_list=K_use,
        smpl_model_path=stage1_cfg["smpl_model_path"],
        smpl_model_type=smpl_model_type,
        smpl_gender=resolved_gender,
        smpl_num_betas=smpl_num_betas,
        fit_frames=int(stage1_cfg.get("fit_frames", 120)),
        fit_iters=int(stage1_cfg.get("fit_iters", 2000)),
        fit_lr=float(stage1_cfg.get("fit_lr", 3e-2)),
        keypoint_loss_type=str(stage1_cfg.get("keypoint_loss_type", "gmof")),
        keypoint_gmof_sigma_px=float(stage1_cfg.get("keypoint_gmof_sigma_px", 100.0)),
        keypoint_conf_thresh=float(stage1_cfg.get("keypoint_conf_thresh", 0.02)),
        keypoint_bbox_norm=bool(stage1_cfg.get("keypoint_bbox_norm", True)),
        keypoint_bbox_norm_min_px=float(stage1_cfg.get("keypoint_bbox_norm_min_px", 96.0)),
        global_init_iters=int(stage1_cfg.get("global_init_iters", 120)),
        optimize_focal_scale=bool(stage1_cfg.get("optimize_focal_scale", False)),
        focal_scale_l2_weight=float(stage1_cfg.get("focal_scale_l2_weight", 1e-3)),
        focal_scale_min=float(stage1_cfg.get("focal_scale_min", 0.5)),
        focal_scale_max=float(stage1_cfg.get("focal_scale_max", 2.0)),
        monocular_depth_weight=float(stage1_cfg.get("monocular_depth_weight", 0.0)),
        monocular_depth_pair_margin=float(stage1_cfg.get("monocular_depth_pair_margin", 0.01)),
        monocular_depth_auto_sign=bool(stage1_cfg.get("monocular_depth_auto_sign", True)),
        world_depth_weight=float(stage1_cfg.get("world_depth_weight", 0.0)),
        world_depth_pair_margin=float(stage1_cfg.get("world_depth_pair_margin", 0.01)),
        world_depth_auto_sign=bool(stage1_cfg.get("world_depth_auto_sign", True)),
        world_min_visibility=float(stage1_cfg.get("world_min_visibility", 0.35)),
        world_pair_distance_weight=float(stage1_cfg.get("world_pair_distance_weight", 0.0)),
        silhouette_npz=(str(seg_npz) if seg_npz.exists() else None),
        silhouette_loss_weight=float(stage1_cfg.get("silhouette_loss_weight", 0.0)),
        silhouette_loss_final_ratio=float(stage1_cfg.get("silhouette_loss_final_ratio", 1.0)),
        silhouette_dist_clip_px=float(stage1_cfg.get("silhouette_dist_clip_px", 24.0)),
        silhouette_sample_verts=int(stage1_cfg.get("silhouette_sample_verts", 2048)),
        silhouette_side_view_boost=float(stage1_cfg.get("silhouette_side_view_boost", 1.5)),
        silhouette_bbox_weight=float(stage1_cfg.get("silhouette_bbox_weight", 0.0)),
        limb_vector_weight=float(stage1_cfg.get("limb_vector_weight", 0.0)),
        limb_vector_len_weight=float(stage1_cfg.get("limb_vector_len_weight", 0.2)),
        pose_angle_prior_weight=float(stage1_cfg.get("pose_angle_prior_weight", 0.0)),
        scale_l2_weight=float(stage1_cfg.get("scale_l2_weight", 1e-3)),
        init_translation_ls=bool(stage1_cfg.get("init_translation_ls", False)),
        loss_schedule=str(stage1_cfg.get("loss_schedule", "none")),
        keypoint_smooth_enable=bool(kpt_smooth_cfg.get("enable", False)),
        keypoint_smooth_method=str(kpt_smooth_cfg.get("method", "one_euro")),
        keypoint_smooth_fps=float(kpt_smooth_cfg.get("fps", 30.0)),
        keypoint_smooth_min_cutoff=float(kpt_smooth_cfg.get("min_cutoff", 1.0)),
        keypoint_smooth_beta=float(kpt_smooth_cfg.get("beta", 0.3)),
        keypoint_smooth_gaussian_kernel_size=int(kpt_smooth_cfg.get("gaussian_kernel_size", 7)),
        keypoint_smooth_gaussian_sigma=float(kpt_smooth_cfg.get("gaussian_sigma", 1.5)),
        export_pose_mode=str(stage1_cfg.get("export_pose_mode", "last_frame")),
        export_ensure_head_up=bool(stage1_cfg.get("export_ensure_head_up", False)),
        export_head_up_margin=float(stage1_cfg.get("export_head_up_margin", 0.06)),
        export_center_xyz=bool(stage1_cfg.get("export_center_xyz", False)),
        export_on_floor=bool(stage1_cfg.get("export_on_floor", True)),
        export_smooth_iters=int(stage1_cfg.get("export_smooth_iters", 0)),
        export_smooth_alpha=float(stage1_cfg.get("export_smooth_alpha", 0.1)),
        export_mesh_tpose=bool(stage1_cfg.get("export_mesh_tpose", False)),
        export_frame_index=int(stage1_cfg.get("export_frame_index", -1)),
        enforce_export_upright=bool(stage1_cfg.get("enforce_export_upright", False)),
        export_center_xy=bool(stage1_cfg.get("export_center_xy", False)),
        export_center_mode=str(stage1_cfg.get("export_center_mode", "pelvis")),
        export_mesh_sequence_indices=export_mesh_sequence_indices,
        export_legacy_mesh_objs=bool(stage1_cfg.get("export_legacy_mesh_objs", False)),
        export_transform_reference=str(stage1_cfg.get("export_transform_reference", "tpose")),
        export_meshes=bool(stage1_cfg.get("export_meshes", True)),
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
            temporal_window=int(stage1_cfg.get("temporal_window", 1)),
            temporal_window_decay=float(stage1_cfg.get("temporal_window_decay", 0.7)),
            temporal_pose_accel_weight=float(stage1_cfg.get("temporal_pose_accel_weight", 0.0)),
            temporal_orient_accel_weight=float(stage1_cfg.get("temporal_orient_accel_weight", 0.0)),
            temporal_trans_accel_weight=float(stage1_cfg.get("temporal_trans_accel_weight", 0.0)),
            last_frame_depth_weight=float(stage1_cfg.get("last_frame_depth_weight", -1.0)),
            last_frame_world_depth_weight=float(stage1_cfg.get("last_frame_world_depth_weight", -1.0)),
            last_frame_world_pair_weight=float(stage1_cfg.get("last_frame_world_pair_weight", -1.0)),
            last_frame_silhouette_weight=float(stage1_cfg.get("last_frame_silhouette_weight", -1.0)),
            last_frame_silhouette_bbox_weight=float(stage1_cfg.get("last_frame_silhouette_bbox_weight", -1.0)),
            last_frame_limb_vector_weight=float(stage1_cfg.get("last_frame_limb_vector_weight", -1.0)),
            optimize_hand_pose=bool(stage1_cfg.get("optimize_hand_pose", True)),
            optimize_face_pose=bool(stage1_cfg.get("optimize_face_pose", True)),
            optimize_expression=bool(stage1_cfg.get("optimize_expression", False)),
            hand_pose_reg_weight=float(stage1_cfg.get("hand_pose_reg_weight", 2e-4)),
            face_pose_reg_weight=float(stage1_cfg.get("face_pose_reg_weight", 2e-4)),
            expression_reg_weight=float(stage1_cfg.get("expression_reg_weight", 1e-3)),
            hand_pose_temporal_weight=float(stage1_cfg.get("hand_pose_temporal_weight", 5e-4)),
            face_pose_temporal_weight=float(stage1_cfg.get("face_pose_temporal_weight", 5e-4)),
            optimize_betas=bool(stage1_cfg.get("optimize_betas", False)),
            unfreeze_betas_iter=int(stage1_cfg.get("unfreeze_betas_iter", -1)),
            betas_l2_weight=float(stage1_cfg.get("betas_l2_weight", 5e-2)),
            betas_l2_final_ratio=float(stage1_cfg.get("betas_l2_final_ratio", 1.0)),
            betas_abs_max=float(stage1_cfg.get("betas_abs_max", 3.0)),
            shape_from_previous_frames=bool(stage1_cfg.get("shape_from_previous_frames", True)),
            final_pose_from_last_frame=bool(stage1_cfg.get("final_pose_from_last_frame", True)),
            last_frame_refine_iters=int(stage1_cfg.get("last_frame_refine_iters", 300)),
            last_frame_temporal_window=int(stage1_cfg.get("last_frame_temporal_window", 0)),
            last_frame_temporal_decay=float(stage1_cfg.get("last_frame_temporal_decay", 0.8)),
            last_frame_temporal_pose_weight=float(stage1_cfg.get("last_frame_temporal_pose_weight", 0.0)),
            last_frame_temporal_orient_weight=float(stage1_cfg.get("last_frame_temporal_orient_weight", 0.0)),
            last_frame_temporal_trans_weight=float(stage1_cfg.get("last_frame_temporal_trans_weight", 0.0)),
            upright_prior_weight=float(stage1_cfg.get("upright_prior_weight", 1e-2)),
            knee_hinge_weight=float(stage1_cfg.get("knee_hinge_weight", 2e-3)),
            knee_direction_weight=float(stage1_cfg.get("knee_direction_weight", 5e-3)),
            knee_hinge_axis=int(stage1_cfg.get("knee_hinge_axis", 1)),
            knee_min_flex=float(stage1_cfg.get("knee_min_flex", 0.12)),
            left_knee_flex_sign=float(stage1_cfg.get("left_knee_flex_sign", -1.0)),
            right_knee_flex_sign=float(stage1_cfg.get("right_knee_flex_sign", -1.0)),
            auto_knee_sign=bool(stage1_cfg.get("auto_knee_sign", False)),
            knee_sign_update_every=int(stage1_cfg.get("knee_sign_update_every", 200)),
            ankle_twist_weight=float(stage1_cfg.get("ankle_twist_weight", 3e-3)),
            back_stability_weight=float(stage1_cfg.get("back_stability_weight", 6e-3)),
            export_mesh_sequence_last_only=bool(stage1_cfg.get("export_mesh_sequence_last_only", False)),
            last_frame_upper_body_boost=float(stage1_cfg.get("last_frame_upper_body_boost", 1.0)),
            full_batch_every=int(stage1_cfg.get("full_batch_every", 200)),
        )
    logger.info(f"Saved SMPL outputs (fit_mode={fit_mode})")

    # 3) Save a simple (single) camera definition (placeholder)
    cam = {
        "type": "pinhole",
        "K": K_use,
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
