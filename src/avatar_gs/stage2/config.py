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

