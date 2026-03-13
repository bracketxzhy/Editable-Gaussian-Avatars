from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Stage2FrameSelectionConfig:
    n_frames: int
    frame_select: str
    frame_indices: list[int] = field(default_factory=list)
    frame_weights: list[float] | None = None
    frame_loss_weights: list[float] | None = None
    last_frame_sample_weight: float | None = None
    last_frame_loss_weight: float | None = None
    stage1_align_filter_enable: bool = True
    stage1_align_filter_metric: str = "camera_align_iou"
    stage1_align_filter_min: float = 0.83

    @classmethod
    def from_stage2_cfg(cls, stage2_cfg: dict[str, Any]) -> "Stage2FrameSelectionConfig":
        frame_indices_raw = stage2_cfg.get("frame_indices", [])
        if isinstance(frame_indices_raw, (int, float, str)):
            frame_indices = [int(frame_indices_raw)]
        else:
            frame_indices = [int(x) for x in frame_indices_raw] if isinstance(frame_indices_raw, (list, tuple)) else []
        frame_weights = stage2_cfg.get("frame_weights", None)
        frame_loss_weights = stage2_cfg.get("frame_loss_weights", None)
        return cls(
            n_frames=int(stage2_cfg.get("n_frames", 20)),
            frame_select=str(stage2_cfg.get("frame_select", "first")).strip().lower(),
            frame_indices=frame_indices,
            frame_weights=None if frame_weights is None else [float(x) for x in frame_weights],
            frame_loss_weights=None if frame_loss_weights is None else [float(x) for x in frame_loss_weights],
            last_frame_sample_weight=(
                None
                if stage2_cfg.get("last_frame_sample_weight", None) is None
                else float(stage2_cfg.get("last_frame_sample_weight"))
            ),
            last_frame_loss_weight=(
                None
                if stage2_cfg.get("last_frame_loss_weight", None) is None
                else float(stage2_cfg.get("last_frame_loss_weight"))
            ),
            stage1_align_filter_enable=bool(stage2_cfg.get("stage1_align_filter_enable", True)),
            stage1_align_filter_metric=(
                str(stage2_cfg.get("stage1_align_filter_metric", "camera_align_iou")).strip() or "camera_align_iou"
            ),
            stage1_align_filter_min=float(stage2_cfg.get("stage1_align_filter_min", 0.83)),
        )


@dataclass(slots=True)
class Stage2BindingConfig:
    bind_to_mesh: bool
    bind_mesh_sequence_dir: str | None
    bind_sequence_npz: str | None
    bind_vertices_npz: str | None
    bind_deform_obj: str | None
    bind_cache_size: int
    bind_mesh_obj_space: str
    bind_mesh_to_camera: bool
    bind_inverse_export_upright: bool
    bind_mesh_offset: tuple[float, float, float]
    bind_align_quat_to_mesh_normals: bool
    bind_quat_normal_blend: float
    bind_rigid_align: dict | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stage2_cfg(cls, stage2_cfg: dict[str, Any]) -> "Stage2BindingConfig":
        offset = stage2_cfg.get("bind_mesh_offset", (0.0, 0.0, 0.0))
        if not (isinstance(offset, (list, tuple)) and len(offset) == 3):
            offset = (0.0, 0.0, 0.0)
        return cls(
            bind_to_mesh=bool(stage2_cfg.get("bind_to_mesh", False)),
            bind_mesh_sequence_dir=(
                None if stage2_cfg.get("bind_mesh_sequence_dir", None) is None else str(stage2_cfg.get("bind_mesh_sequence_dir"))
            ),
            bind_sequence_npz=(
                None if stage2_cfg.get("bind_sequence_npz", None) is None else str(stage2_cfg.get("bind_sequence_npz"))
            ),
            bind_vertices_npz=(
                None if stage2_cfg.get("bind_vertices_npz", None) is None else str(stage2_cfg.get("bind_vertices_npz"))
            ),
            bind_deform_obj=(
                None if stage2_cfg.get("bind_deform_obj", None) is None else str(stage2_cfg.get("bind_deform_obj"))
            ),
            bind_cache_size=int(stage2_cfg.get("bind_cache_size", 8)),
            bind_mesh_obj_space=str(stage2_cfg.get("bind_mesh_obj_space", "auto")).strip().lower(),
            bind_mesh_to_camera=bool(stage2_cfg.get("bind_mesh_to_camera", False)),
            bind_inverse_export_upright=bool(stage2_cfg.get("bind_inverse_export_upright", True)),
            bind_mesh_offset=(float(offset[0]), float(offset[1]), float(offset[2])),
            bind_align_quat_to_mesh_normals=bool(stage2_cfg.get("bind_align_quat_to_mesh_normals", True)),
            bind_quat_normal_blend=float(stage2_cfg.get("bind_quat_normal_blend", 1.0)),
            bind_rigid_align=stage2_cfg.get("bind_rigid_align", None),
            raw=dict(stage2_cfg),
        )


@dataclass(slots=True)
class Stage2SupervisionConfig:
    use_smpl_silhouette_mask: bool
    use_person_mask: bool
    person_masks_npz: str | None
    prepared_supervision_cache_path: str | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stage2_cfg(cls, stage2_cfg: dict[str, Any], *, stage1_dir: Path | None = None) -> "Stage2SupervisionConfig":
        default_pm = None if stage1_dir is None else str(stage1_dir / "person_masks.npz")
        return cls(
            use_smpl_silhouette_mask=bool(stage2_cfg.get("use_smpl_silhouette_mask", False)),
            use_person_mask=bool(stage2_cfg.get("use_person_mask", False)),
            person_masks_npz=(
                None if stage2_cfg.get("person_masks_npz", default_pm) is None else str(stage2_cfg.get("person_masks_npz", default_pm))
            ),
            prepared_supervision_cache_path=(
                None
                if stage2_cfg.get("prepared_supervision_cache_path", None) is None
                else str(stage2_cfg.get("prepared_supervision_cache_path"))
            ),
            raw=dict(stage2_cfg),
        )


@dataclass(slots=True)
class Stage2TrainConfig:
    project_root: Path
    stage1_dir: Path
    out_dir: Path
    log_dir: Path
    stage2_raw: dict[str, Any]
    data_raw: dict[str, Any]
    frame_selection: Stage2FrameSelectionConfig
    binding: Stage2BindingConfig
    supervision: Stage2SupervisionConfig

    @classmethod
    def from_cfg(
        cls,
        *,
        cfg: dict[str, Any],
        project_root: Path,
        log_dir: Path,
    ) -> "Stage2TrainConfig":
        stage2_cfg = dict(cfg["stage2"])
        stage1_dir = project_root / stage2_cfg["stage1_dir"]
        out_dir = project_root / stage2_cfg["out_dir"]
        return cls(
            project_root=project_root,
            stage1_dir=stage1_dir,
            out_dir=out_dir,
            log_dir=log_dir,
            stage2_raw=stage2_cfg,
            data_raw=dict(cfg.get("data", {})),
            frame_selection=Stage2FrameSelectionConfig.from_stage2_cfg(stage2_cfg),
            binding=Stage2BindingConfig.from_stage2_cfg(stage2_cfg),
            supervision=Stage2SupervisionConfig.from_stage2_cfg(stage2_cfg, stage1_dir=stage1_dir),
        )

    @property
    def frames_dir(self) -> Path:
        return self.project_root / self.data_raw["frames_dir"]



@dataclass(slots=True)
class Stage2TrainLoopConfig:
    frames_dir: Path
    frames_list: list[str]
    K_list: list[float]
    frame_Ks: list[list[float]] | None
    out_ckpt: Path
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stage2_setup(
        cls,
        *,
        train_config: Stage2TrainConfig,
        frames: list[str],
        cam: dict[str, Any],
        out_ckpt: str | Path,
        runtime: dict[str, Any],
    ) -> "Stage2TrainLoopConfig":
        stage2_cfg = train_config.stage2_raw
        stage1_dir = train_config.stage1_dir
        smpl_face_semantics_npz = stage2_cfg.get(
            "smpl_face_semantics_npz",
            str(stage1_dir / "fashn_face_semantics" / "smpl_face_semantics.npz"),
        )
        if smpl_face_semantics_npz is not None:
            smpl_face_semantics_npz = Path(str(smpl_face_semantics_npz))
            if not smpl_face_semantics_npz.is_absolute():
                smpl_face_semantics_npz = train_config.project_root / smpl_face_semantics_npz
        return cls(
            frames_dir=train_config.frames_dir,
            frames_list=[str(x) for x in frames],
            K_list=[float(x) for x in cam["K"]],
            frame_Ks=(
                None
                if runtime.get("frame_Ks", None) is None
                else [[float(v) for v in row] for row in runtime.get("frame_Ks")]
            ),
            out_ckpt=Path(out_ckpt),
            options={
                "iters": int(stage2_cfg.get("iters", 2000)),
                "lr_gauss": float(stage2_cfg.get("lr_gauss", 1e-2)),
                "lr_cam": float(runtime["lr_cam"]),
                "init_cam_w": runtime.get("init_cam_w"),
                "init_cam_t": runtime.get("init_cam_t"),
                "device": runtime.get("device", "cuda"),
                "bg_rgb": tuple(stage2_cfg.get("bg_rgb", (1.0, 1.0, 1.0))),
                "cam_warmup_iters": int(stage2_cfg.get("cam_warmup_iters", 5000)),
                "l1_weight": float(stage2_cfg.get("l1_weight", 0.25)),
                "use_alpha_mask": bool(stage2_cfg.get("use_alpha_mask", True)),
                "alpha_threshold": float(stage2_cfg.get("alpha_threshold", 0.15)),
                "camera_mode": str(runtime.get("camera_mode", "learnable")),
                "cam_t_prior": tuple(runtime.get("cam_t_prior", (0.0, 0.0, 0.0))),
                "cam_reg_weight": float(runtime.get("cam_reg_weight", 0.0)),
                "keypoints_json": stage2_cfg.get("keypoints_json", str(stage1_dir / "keypoints.json")),
                "kpt_conf_min": float(stage2_cfg.get("kpt_conf_min", 0.2)),
                "kpt_box_pad": float(stage2_cfg.get("kpt_box_pad", 0.15)),
                "use_kpt_bbox_mask": bool(stage2_cfg.get("use_kpt_bbox_mask", False)),
                "face_weight": float(stage2_cfg.get("face_weight", 1.0)),
                "face_conf_min": float(stage2_cfg.get("face_conf_min", 0.3)),
                "face_box_pad": float(stage2_cfg.get("face_box_pad", 0.2)),
                "face_use_ears": bool(stage2_cfg.get("face_use_ears", False)),
                "hand_weight": float(stage2_cfg.get("hand_weight", 1.0)),
                "hand_conf_min": float(stage2_cfg.get("hand_conf_min", 0.25)),
                "hand_radius_scale": float(stage2_cfg.get("hand_radius_scale", 0.45)),
                "hand_radius_min_px": int(stage2_cfg.get("hand_radius_min_px", 10)),
                "hand_extension_scale": float(stage2_cfg.get("hand_extension_scale", 0.35)),
                "outer_exclude_hands": bool(stage2_cfg.get("outer_exclude_hands", False)),
                "outer_hand_expand_px": int(stage2_cfg.get("outer_hand_expand_px", 3)),
                "frame_weights": runtime.get("frame_weights"),
                "frame_loss_weights": runtime.get("frame_loss_weights"),
                "frame_align_scores": runtime.get("frame_align_scores"),
                "temporal_ref_window": int(stage2_cfg.get("temporal_ref_window", 0)),
                "temporal_ref_decay": float(stage2_cfg.get("temporal_ref_decay", 0.85)),
                "temporal_ref_anchor": str(stage2_cfg.get("temporal_ref_anchor", "sample")),
                "bind_vertex_indices": runtime.get("bind_vertex_indices"),
                "bind_cache_size": int(stage2_cfg.get("bind_cache_size", 8)),
                "bind_align_quat_to_mesh_normals": bool(stage2_cfg.get("bind_align_quat_to_mesh_normals", True)),
                "bind_quat_normal_blend": float(stage2_cfg.get("bind_quat_normal_blend", 1.0)),
                "bind_surface_aniso_weight": float(stage2_cfg.get("bind_surface_aniso_weight", 0.0)),
                "bind_surface_aniso_margin": float(stage2_cfg.get("bind_surface_aniso_margin", 0.15)),
                "inner_local_scale_cap_weight": float(stage2_cfg.get("inner_local_scale_cap_weight", 0.0)),
                "inner_local_scale_cap_ratio": float(stage2_cfg.get("inner_local_scale_cap_ratio", 1.2)),
                "inner_local_scale_cap_normal_ratio": float(stage2_cfg.get("inner_local_scale_cap_normal_ratio", 0.7)),
                "smpl_face_semantics_npz": None if smpl_face_semantics_npz is None else str(smpl_face_semantics_npz),
                "outer_block_exposed_faces": bool(stage2_cfg.get("outer_block_exposed_faces", True)),
                "inner_local_color_prior_enable": bool(stage2_cfg.get("inner_local_color_prior_enable", False)),
                "inner_local_color_prior_weight": float(stage2_cfg.get("inner_local_color_prior_weight", 0.03)),
                "inner_local_color_prior_start_iter": int(stage2_cfg.get("inner_local_color_prior_start_iter", 200)),
                "inner_local_color_prior_target_conf_max": float(stage2_cfg.get("inner_local_color_prior_target_conf_max", 0.30)),
                "inner_local_color_prior_donor_conf_min": float(stage2_cfg.get("inner_local_color_prior_donor_conf_min", 0.65)),
                "inner_local_color_prior_donor_opacity_min": float(stage2_cfg.get("inner_local_color_prior_donor_opacity_min", 0.08)),
                "inner_local_color_prior_donor_vis_min": float(stage2_cfg.get("inner_local_color_prior_donor_vis_min", 0.05)),
                "inner_local_color_prior_semantic_conf_min": float(stage2_cfg.get("inner_local_color_prior_semantic_conf_min", 0.55)),
                "inner_local_color_prior_min_donor_weight": float(stage2_cfg.get("inner_local_color_prior_min_donor_weight", 0.20)),
                "use_smpl_silhouette_mask": bool(stage2_cfg.get("use_smpl_silhouette_mask", False)),
                "smpl_mask_dilate_px": int(stage2_cfg.get("smpl_mask_dilate_px", 0)),
                "smpl_mask_erode_px": int(stage2_cfg.get("smpl_mask_erode_px", 0)),
                "smpl_color_mask_erode_px": int(stage2_cfg.get("smpl_color_mask_erode_px", 0)),
                "inner_use_strict_smpl_mask": bool(stage2_cfg.get("inner_use_strict_smpl_mask", True)),
                "smpl_alpha_leak_weight": float(stage2_cfg.get("smpl_alpha_leak_weight", 0.0)),
                "smpl_mask_override_alpha": bool(stage2_cfg.get("smpl_mask_override_alpha", True)),
                "smpl_mask_precompute": bool(stage2_cfg.get("smpl_mask_precompute", True)),
                "smpl_mask_precompute_max_frames": int(stage2_cfg.get("smpl_mask_precompute_max_frames", 64)),
                "person_masks_npz": stage2_cfg.get("person_masks_npz", str(stage1_dir / "person_masks.npz")),
                "prepared_supervision": runtime.get("prepared_supervision"),
                "use_person_mask": bool(stage2_cfg.get("use_person_mask", False)),
                "person_mask_dilate_px": int(stage2_cfg.get("person_mask_dilate_px", 0)),
                "person_mask_erode_px": int(stage2_cfg.get("person_mask_erode_px", 0)),
                "person_mask_fuse_mode": str(stage2_cfg.get("person_mask_fuse_mode", "intersect")),
                "person_mask_eval_clean": bool(stage2_cfg.get("person_mask_eval_clean", False)),
                "person_mask_eval_refine_modnet": bool(stage2_cfg.get("person_mask_eval_refine_modnet", False)),
                "person_mask_contour_width_px": int(stage2_cfg.get("person_mask_contour_width_px", 2)),
                "person_mask_contour_leak_weight": float(stage2_cfg.get("person_mask_contour_leak_weight", 0.0)),
                "outer_use_gt_contour": bool(stage2_cfg.get("outer_use_gt_contour", True)),
                "outer_gt_contour_fuse_mode": str(stage2_cfg.get("outer_gt_contour_fuse_mode", "union")),
                "outer_gt_contour_modnet_ref_size": int(stage2_cfg.get("outer_gt_contour_modnet_ref_size", 512)),
                "outer_gt_contour_modnet_threshold": float(stage2_cfg.get("outer_gt_contour_modnet_threshold", 0.5)),
                "modnet_ckpt_path": None if str(stage2_cfg.get("modnet_ckpt_path", "")).strip() == "" else str(stage2_cfg.get("modnet_ckpt_path")).strip(),
                "outer_head_priority_enable": bool(stage2_cfg.get("outer_head_priority_enable", False)),
                "outer_head_priority_weight": float(stage2_cfg.get("outer_head_priority_weight", 1.5)),
                "outer_head_priority_face_pad": float(stage2_cfg.get("outer_head_priority_face_pad", 0.45)),
                "outer_head_priority_up_scale": float(stage2_cfg.get("outer_head_priority_up_scale", 2.4)),
                "outer_head_priority_down_scale": float(stage2_cfg.get("outer_head_priority_down_scale", 0.35)),
                "outer_head_priority_person_top_ratio": float(stage2_cfg.get("outer_head_priority_person_top_ratio", 0.16)),
                "outer_head_support_th": float(stage2_cfg.get("outer_head_support_th", 0.08)),
                "outer_trigger_smpl_gap_px": int(stage2_cfg.get("outer_trigger_smpl_gap_px", 3)),
                "outer_trigger_edge_th": float(stage2_cfg.get("outer_trigger_edge_th", 0.06)),
                "outer_trigger_min_pixels": int(stage2_cfg.get("outer_trigger_min_pixels", 32)),
                "outer_trigger_ring_width_px": int(stage2_cfg.get("outer_trigger_ring_width_px", 2)),
                "outer_trigger_use_edge_gate": bool(stage2_cfg.get("outer_trigger_use_edge_gate", True)),
                "outer_contour_inside_only": bool(stage2_cfg.get("outer_contour_inside_only", False)),
                "outer_detached_solid_enable": bool(stage2_cfg.get("outer_detached_solid_enable", False)),
                "outer_detached_solid_dilate_px": int(stage2_cfg.get("outer_detached_solid_dilate_px", 0)),
                "outer_detached_solid_close_px": int(stage2_cfg.get("outer_detached_solid_close_px", 0)),
                "outer_detached_solid_mix_start": float(stage2_cfg.get("outer_detached_solid_mix_start", 0.0)),
                "outer_detached_solid_mix_end": float(stage2_cfg.get("outer_detached_solid_mix_end", 0.0)),
                "outer_head_support_sample_dilate_px": int(stage2_cfg.get("outer_head_support_sample_dilate_px", 0)),
                "outer_exclude_face": bool(stage2_cfg.get("outer_exclude_face", False)),
                "outer_face_expand_px": int(stage2_cfg.get("outer_face_expand_px", 0)),
                "outer_face_exclude_lower_ratio": float(stage2_cfg.get("outer_face_exclude_lower_ratio", 1.0)),
                "outer_silhouette_weight": float(stage2_cfg.get("outer_silhouette_weight", 0.0)),
                "outer_silhouette_band_px": int(stage2_cfg.get("outer_silhouette_band_px", 2)),
                "outer_silhouette_band_weight": float(stage2_cfg.get("outer_silhouette_band_weight", 1.0)),
                "detail_hf_weight": float(stage2_cfg.get("detail_hf_weight", 0.0)),
                "detail_hf_exclude_face": bool(stage2_cfg.get("detail_hf_exclude_face", True)),
                "outer_inside_support_th": float(stage2_cfg.get("outer_inside_support_th", 0.05)),
                "outer_outside_support_th": float(stage2_cfg.get("outer_outside_support_th", 0.5)),
                "outer_outside_opacity_cap": float(stage2_cfg.get("outer_outside_opacity_cap", 0.02)),
                "outer_outside_offset_decay": float(stage2_cfg.get("outer_outside_offset_decay", 0.92)),
                "outer_outside_opacity_weight": float(stage2_cfg.get("outer_outside_opacity_weight", 0.10)),
                "outer_outside_offset_weight": float(stage2_cfg.get("outer_outside_offset_weight", 0.02)),
                "outer_offset_suppress_low_support": bool(stage2_cfg.get("outer_offset_suppress_low_support", False)),
                "outer_offset_suppress_support_th": float(stage2_cfg.get("outer_offset_suppress_support_th", 0.0)),
                "outer_offset_suppress_decay": float(stage2_cfg.get("outer_offset_suppress_decay", 1.0)),
                "outer_offset_suppress_opacity_cap": float(stage2_cfg.get("outer_offset_suppress_opacity_cap", 1.0)),
                "separate_fg_bg_training": bool(stage2_cfg.get("separate_fg_bg_training", False)),
                "fg_region_weight": float(stage2_cfg.get("fg_region_weight", 1.0)),
                "bg_region_weight": float(stage2_cfg.get("bg_region_weight", 1.0)),
                "inner_phase_use_raw_gt_target": bool(stage2_cfg.get("inner_phase_use_raw_gt_target", False)),
                "inner_phase_keep_bg_weight": bool(stage2_cfg.get("inner_phase_keep_bg_weight", False)),
                "bg_grad_scale": float(stage2_cfg.get("bg_grad_scale", 1.0)),
                "bg_opacity_logit_max": stage2_cfg.get("bg_opacity_logit_max", None),
                "bg_log_scale_min": stage2_cfg.get("bg_log_scale_min", None),
                "bg_log_scale_max": stage2_cfg.get("bg_log_scale_max", None),
                "fg_opacity_logit_max": stage2_cfg.get("fg_opacity_logit_max", None),
                "fg_log_scale_min": stage2_cfg.get("fg_log_scale_min", None),
                "fg_log_scale_max": stage2_cfg.get("fg_log_scale_max", None),
                "outer_log_scale_min": stage2_cfg.get("outer_log_scale_min", None),
                "outer_log_scale_max": stage2_cfg.get("outer_log_scale_max", None),
                "outer_prune_support_th": float(stage2_cfg.get("outer_prune_support_th", 0.0)),
                "outer_prune_dual_enable": bool(stage2_cfg.get("outer_prune_dual_enable", False)),
                "outer_prune_dual_max_vis": float(stage2_cfg.get("outer_prune_dual_max_vis", 0.06)),
                "outer_prune_dual_min_support": float(stage2_cfg.get("outer_prune_dual_min_support", 0.001)),
                "outer_align_prune_protect_enable": bool(stage2_cfg.get("outer_align_prune_protect_enable", True)),
                "outer_align_prune_protect_min_iou": float(stage2_cfg.get("outer_align_prune_protect_min_iou", 0.998)),
                "outer_align_prune_protect_min_vis": float(stage2_cfg.get("outer_align_prune_protect_min_vis", 0.006)),
                "outer_align_prune_protect_min_support": float(stage2_cfg.get("outer_align_prune_protect_min_support", 0.006)),
                "early_stop_enable": bool(stage2_cfg.get("early_stop_enable", False)),
                "early_stop_patience": int(stage2_cfg.get("early_stop_patience", 0)),
                "early_stop_min_delta": float(stage2_cfg.get("early_stop_min_delta", 0.0)),
                "early_stop_warmup_iters": int(stage2_cfg.get("early_stop_warmup_iters", 0)),
                "binding": stage2_cfg.get("binding", None),
                "tau_vis": float(stage2_cfg.get("tau_vis", 0.15)),
                "frame_weight_k": float(stage2_cfg.get("frame_weight_k", 4.0)),
                "use_robust_loss": bool(stage2_cfg.get("use_robust_loss", True)),
                "robust_loss_type": str(stage2_cfg.get("robust_loss_type", "charbonnier")),
                "alpha_prior_weight": float(stage2_cfg.get("alpha_prior_weight", 0.1)),
                "inner_opacity_prior_target": float(stage2_cfg.get("inner_opacity_prior_target", 0.50)),
                "outer_opacity_prior_target": float(stage2_cfg.get("outer_opacity_prior_target", 0.12)),
                "cov_min": float(stage2_cfg.get("cov_min", 5e-4)),
                "cov_max": float(stage2_cfg.get("cov_max", 2.5e-1)),
                "cov_penalty_weight": float(stage2_cfg.get("cov_penalty_weight", 0.05)),
                "use_border_attenuation": bool(stage2_cfg.get("use_border_attenuation", True)),
                "border_attenuation_width": int(stage2_cfg.get("border_attenuation_width", 8)),
                "V_min": int(stage2_cfg.get("V_min", 8)),
                "densify_top_percent": float(stage2_cfg.get("densify_top_percent", 10.0)),
                "phase_ratio_inner_only": float(stage2_cfg.get("phase_ratio_inner_only", 0.20)),
                "phase_ratio_outer_only": float(stage2_cfg.get("phase_ratio_outer_only", 0.20)),
                "phase_ratio_occlusion_simplify": float(stage2_cfg.get("phase_ratio_occlusion_simplify", 0.30)),
                "phase_ratio_joint_finetune": float(stage2_cfg.get("phase_ratio_joint_finetune", 0.30)),
                "shell_enable": bool(stage2_cfg.get("shell_enable", False)),
                "shell_max_ratio": float(stage2_cfg.get("shell_max_ratio", 0.40)),
                "shell_spawn_top_percent": float(stage2_cfg.get("shell_spawn_top_percent", 15.0)),
                "shell_spawn_every": int(stage2_cfg.get("shell_spawn_every", 200)),
                "shell_prune_every": int(stage2_cfg.get("shell_prune_every", 400)),
                "shell_offset_max_n": float(stage2_cfg.get("shell_offset_max_n", 0.04)),
                "shell_offset_max_t": float(stage2_cfg.get("shell_offset_max_t", 0.015)),
                "outer_offset_unbounded": bool(stage2_cfg.get("outer_offset_unbounded", True)),
                "outer_offset_auto_range_enable": bool(stage2_cfg.get("outer_offset_auto_range_enable", True)),
                "outer_offset_auto_range_min_scale": float(stage2_cfg.get("outer_offset_auto_range_min_scale", 0.35)),
                "outer_offset_auto_range_edge_mix": float(stage2_cfg.get("outer_offset_auto_range_edge_mix", 0.75)),
                "outer_offset_edge_guided_reg": bool(stage2_cfg.get("outer_offset_edge_guided_reg", True)),
                "outer_offset_reg_relax_support": bool(stage2_cfg.get("outer_offset_reg_relax_support", True)),
                "outer_offset_min_active_weight": float(stage2_cfg.get("outer_offset_min_active_weight", 0.08)),
                "outer_offset_min_target_scale": float(stage2_cfg.get("outer_offset_min_target_scale", 0.20)),
                "outer_offset_min_support_th": float(stage2_cfg.get("outer_offset_min_support_th", 0.02)),
                "shell_offset_reg_weight": float(stage2_cfg.get("shell_offset_reg_weight", 0.02)),
                "shell_offset_reg_phase0_scale": float(stage2_cfg.get("shell_offset_reg_phase0_scale", 0.25)),
                "shell_offset_freeze_phase0": bool(stage2_cfg.get("shell_offset_freeze_phase0", False)),
                "outer_offset_grad_scale": float(stage2_cfg.get("outer_offset_grad_scale", 3.0)),
                "outer_offset_edge_grad_scale": float(stage2_cfg.get("outer_offset_edge_grad_scale", 2.0)),
                "outer_offset_grad_scale_max": float(stage2_cfg.get("outer_offset_grad_scale_max", 8.0)),
                "outer_support_weight": float(stage2_cfg.get("outer_support_weight", 1.2)),
                "outer_spawn_support_th": float(stage2_cfg.get("outer_spawn_support_th", 0.02)),
                "outer_spawn_inside_only": bool(stage2_cfg.get("outer_spawn_inside_only", True)),
                "outer_phase_inner_vis_scale": float(stage2_cfg.get("outer_phase_inner_vis_scale", 0.0)),
                "outer_phase_inner_full_mask": bool(stage2_cfg.get("outer_phase_inner_full_mask", False)),
                "outer_low_support_patience": int(stage2_cfg.get("outer_low_support_patience", 3)),
                "bind_face_offset_straight_through": bool(stage2_cfg.get("bind_face_offset_straight_through", True)),
                "inner_bind_mode_default": str(stage2_cfg.get("inner_bind_mode_default", "vertex")),
                "outer_init_from_inner": bool(stage2_cfg.get("outer_init_from_inner", False)),
                "outer_dormant_slot_ratio": float(stage2_cfg.get("outer_dormant_slot_ratio", stage2_cfg.get("shell_init_ratio", 0.15))),
                "outer_trigger_face_activate_enable": bool(stage2_cfg.get("outer_trigger_face_activate_enable", True)),
                "outer_trigger_face_sample_dilate_px": int(stage2_cfg.get("outer_trigger_face_sample_dilate_px", 0)),
                "outer_viewplane_init_enable": bool(stage2_cfg.get("outer_viewplane_init_enable", True)),
                "outer_center_init_rule": str(stage2_cfg.get("outer_center_init_rule", "segment_midpoint")),
                "outer_center_phase1_normal_tune_enable": bool(stage2_cfg.get("outer_center_phase1_normal_tune_enable", True)),
                "outer_center_phase1_normal_tune_cap": float(stage2_cfg.get("outer_center_phase1_normal_tune_cap", 0.04)),
                "outer_center_freeze_after_phase1": bool(stage2_cfg.get("outer_center_freeze_after_phase1", True)),
                "outer_contact_alpha_edge_th": float(stage2_cfg.get("outer_contact_alpha_edge_th", 0.05)),
                "outer_detached_support_boost": float(stage2_cfg.get("outer_detached_support_boost", 1.8)),
                "outer_detached_vis_boost": float(stage2_cfg.get("outer_detached_vis_boost", 1.5)),
                "outer_overlap_prune_offset_th": float(stage2_cfg.get("outer_overlap_prune_offset_th", 0.004)),
                "outer_overlap_prune_detached_th": float(stage2_cfg.get("outer_overlap_prune_detached_th", 0.02)),
                "outer_detached_support_th": float(stage2_cfg.get("outer_detached_support_th", 0.05)),
                "outer_detached_opacity_floor": float(stage2_cfg.get("outer_detached_opacity_floor", 0.08)),
                "outer_detached_log_scale_floor": float(stage2_cfg.get("outer_detached_log_scale_floor", -5.8)),
                "outer_gap_search_px": int(stage2_cfg.get("outer_gap_search_px", 24)),
                "outer_gap_step_px": float(stage2_cfg.get("outer_gap_step_px", 1.0)),
                "outer_gap_probe_world": float(stage2_cfg.get("outer_gap_probe_world", 0.01)),
                "outer_gap_target_support_th": float(stage2_cfg.get("outer_gap_target_support_th", 0.05)),
                "outer_gap_target_weight": float(stage2_cfg.get("outer_gap_target_weight", 0.20)),
                "outer_gap_scale_weight": float(stage2_cfg.get("outer_gap_scale_weight", 0.10)),
                "outer_gap_tangent_scale_ratio": float(stage2_cfg.get("outer_gap_tangent_scale_ratio", 0.45)),
                "outer_gap_tangent_scale_weight": float(stage2_cfg.get("outer_gap_tangent_scale_weight", 0.08)),
                "outer_gap_hard_target_enable": bool(stage2_cfg.get("outer_gap_hard_target_enable", False)),
                "outer_gap_hard_target_blend": float(stage2_cfg.get("outer_gap_hard_target_blend", 1.0)),
                "outer_gap_refine_steps": int(stage2_cfg.get("outer_gap_refine_steps", 4)),
                "outer_gap_fallback_enable": bool(stage2_cfg.get("outer_gap_fallback_enable", True)),
                "outer_gap_fallback_support_th": float(stage2_cfg.get("outer_gap_fallback_support_th", 0.03)),
                "outer_gap_fallback_half_min_n": float(stage2_cfg.get("outer_gap_fallback_half_min_n", 0.010)),
                "outer_gap_fallback_half_max_n": float(stage2_cfg.get("outer_gap_fallback_half_max_n", 0.045)),
                "outer_geom_fraction": float(stage2_cfg.get("outer_geom_fraction", 0.65)),
                "outer_phase_residual_weight": float(stage2_cfg.get("outer_phase_residual_weight", 0.15)),
                "outer_trigger_face_topk": int(stage2_cfg.get("outer_trigger_face_topk", 0)),
                "outer_trigger_face_ema_decay": float(stage2_cfg.get("outer_trigger_face_ema_decay", 0.85)),
                "outer_trigger_face_ema_weight": float(stage2_cfg.get("outer_trigger_face_ema_weight", 0.60)),
                "outer_trigger_face_topk_min_support": float(stage2_cfg.get("outer_trigger_face_topk_min_support", 0.02)),
                "outer_airspace_penalty_weight": float(stage2_cfg.get("outer_airspace_penalty_weight", 0.04)),
                "outer_airspace_main_support_scale": float(stage2_cfg.get("outer_airspace_main_support_scale", 1.5)),
                "outer_airspace_sharp_allow_dot": float(stage2_cfg.get("outer_airspace_sharp_allow_dot", 0.15)),
                "outer_airspace_neighbor_rings": int(stage2_cfg.get("outer_airspace_neighbor_rings", 1)),
                "outer_edge_trigger_enable": bool(stage2_cfg.get("outer_edge_trigger_enable", True)),
                "outer_edge_trigger_weight": float(stage2_cfg.get("outer_edge_trigger_weight", 0.35)),
                "outer_edge_prune_protect_quantile": float(stage2_cfg.get("outer_edge_prune_protect_quantile", 0.90)),
                "inner_occlusion_soft_enable": bool(stage2_cfg.get("inner_occlusion_soft_enable", True)),
                "inner_occlusion_hard_enable": bool(stage2_cfg.get("inner_occlusion_hard_enable", True)),
                "inner_soft_vis_th": float(stage2_cfg.get("inner_soft_vis_th", 0.05)),
                "inner_soft_contrib_th": float(stage2_cfg.get("inner_soft_contrib_th", 0.01)),
                "outer_cover_vis_th": float(stage2_cfg.get("outer_cover_vis_th", 0.20)),
                "outer_cover_contrib_th": float(stage2_cfg.get("outer_cover_contrib_th", 0.03)),
                "inner_soft_patience": int(stage2_cfg.get("inner_soft_patience", 300)),
                "inner_hard_patience": int(stage2_cfg.get("inner_hard_patience", 800)),
                "inner_soft_lr_scale": float(stage2_cfg.get("inner_soft_lr_scale", 0.2)),
                "inner_prune_every": int(stage2_cfg.get("inner_prune_every", 400)),
                "inner_prune_protect_parts": stage2_cfg.get("inner_prune_protect_parts", ["face", "hands"]),
                "inner_sparse_cleanup_enable": bool(stage2_cfg.get("inner_sparse_cleanup_enable", True)),
                "inner_sparse_cleanup_vis_th": float(stage2_cfg.get("inner_sparse_cleanup_vis_th", 0.02)),
                "inner_sparse_cleanup_opacity_th": float(stage2_cfg.get("inner_sparse_cleanup_opacity_th", 0.08)),
                "inner_sparse_cleanup_warmup_iters": int(stage2_cfg.get("inner_sparse_cleanup_warmup_iters", 800)),
                "inner_sparse_cleanup_patience": int(stage2_cfg.get("inner_sparse_cleanup_patience", 2)),
                "inner_sparse_cleanup_every": int(stage2_cfg.get("inner_sparse_cleanup_every", 200)),
                "outer_pruned_opacity_logit_max": float(stage2_cfg.get("outer_pruned_opacity_logit_max", -12.0)),
                "outer_quality_log_every": int(stage2_cfg.get("outer_quality_log_every", 100)),
                "outer_progress_log_every": int(stage2_cfg.get("outer_progress_log_every", 20)),
                "outer_zero_stop_enable": bool(stage2_cfg.get("outer_zero_stop_enable", False)),
                "outer_zero_stop_patience": int(stage2_cfg.get("outer_zero_stop_patience", 120)),
                "outer_zero_stop_warmup_iters": int(stage2_cfg.get("outer_zero_stop_warmup_iters", 200)),
                "outer_zero_stop_after_phase_inner": bool(stage2_cfg.get("outer_zero_stop_after_phase_inner", True)),
                "outer_boundary_enforce": str(stage2_cfg.get("outer_boundary_enforce", "hard")),
                "outer_boundary_tol_px": int(stage2_cfg.get("outer_boundary_tol_px", 1)),
                "outer_boundary_check_every": int(stage2_cfg.get("outer_boundary_check_every", 50)),
                "outer_boundary_alpha_cap": float(stage2_cfg.get("outer_boundary_alpha_cap", 0.01)),
                "outer_boundary_scale_decay": float(stage2_cfg.get("outer_boundary_scale_decay", 0.85)),
                "outer_boundary_offset_decay": float(stage2_cfg.get("outer_boundary_offset_decay", 0.90)),
                "outer_boundary_patience": int(stage2_cfg.get("outer_boundary_patience", 6)),
                "outer_boundary_geom_start_frac": float(stage2_cfg.get("outer_boundary_geom_start_frac", 0.45)),
                "outer_offset_cap_geom_start_frac": float(stage2_cfg.get("outer_offset_cap_geom_start_frac", 0.55)),
                "dual_quality_gate": stage2_cfg.get("dual_quality_gate", None),
                "stage2_frame_assets": runtime.get("stage2_frame_assets"),
                "stage2_mesh_runtime": runtime.get("stage2_mesh_runtime"),
                "training_mode": str(runtime.get("training_mode", "full")),
                "save_inner_checkpoint_path": runtime.get("save_inner_checkpoint_path"),
                "logger": runtime.get("logger"),
            },
        )

    def to_train_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.options)
        kwargs.update(
            {
                "frames_dir": self.frames_dir,
                "frames_list": list(self.frames_list),
                "K_list": list(self.K_list),
                "frame_Ks": None if self.frame_Ks is None else [list(row) for row in self.frame_Ks],
                "out_ckpt": self.out_ckpt,
            }
        )
        return kwargs

