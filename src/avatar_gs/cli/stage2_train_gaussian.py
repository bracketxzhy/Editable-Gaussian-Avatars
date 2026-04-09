from __future__ import annotations
from pathlib import Path
import json
import shutil
import subprocess
import sys
import cv2
import numpy as np
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.init_from_smpl import init_gaussians_from_mesh_3d
from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.train_loop import train_gaussians_3d_learnable_camera
from avatar_gs.data.sequence_mesh import SequenceMeshProvider
from avatar_gs.stage2 import (
    Stage2TrainConfig,
    Stage2TrainLoopConfig,
    Stage2MeshRuntime,
    normalize_rigid_align_config,
    build_stage2_frame_assets,
    load_stage2_supervision_cache,
    prepare_stage2_supervision_artifacts,
    run_stage2_rigid_align_search,
)
try:
    from avatar_gs.data.h36m import resolve_h36m_image_paths
except Exception:
    resolve_h36m_image_paths = None



def _cfg_string_list(value, default: list[str]) -> list[str]:
    if value is None:
        items = default
    elif isinstance(value, str):
        stripped = value.strip()
        if len(stripped) <= 0:
            items = []
        elif "," in stripped:
            items = [x.strip() for x in stripped.split(",")]
        else:
            items = [stripped]
    else:
        items = list(value)
    return [str(x).strip() for x in items if str(x).strip()]


def _cfg_list_is_auto(value) -> bool:
    items = _cfg_string_list(value, [])
    if len(items) <= 0:
        return True
    return len(items) == 1 and items[0].strip().lower() == 'auto'


def _run_part_teacher_bank(*, project_root: Path, stage1_dir: Path, out_dir: Path, stage2_cfg: dict, logger) -> Path | None:
    if not (bool(stage2_cfg.get('face_postfit_enable', False)) or bool(stage2_cfg.get('part_postfit_enable', False))):
        return None
    select_script = project_root / 'src' / 'avatar_gs' / 'cli' / 'stage2_select_part_teachers.py'
    if not select_script.exists():
        raise FileNotFoundError(f"part teacher selection script missing: {select_script}")
    frames_dir = _resolve_project_path(project_root, stage2_cfg.get('part_postfit_frames_dir', project_root / 'data' / 'frames' / 'input'))
    supervision_cache = _resolve_project_path(project_root, stage2_cfg.get('prepared_supervision_cache_path', out_dir / 'supervision_cache.npz'))
    semantics_npz = _resolve_project_path(project_root, stage2_cfg.get('smpl_face_semantics_npz', stage1_dir / 'fashn_face_semantics' / 'smpl_face_semantics.npz'))
    if frames_dir is None or not frames_dir.exists():
        raise FileNotFoundError(f"part teacher selection requires frames_dir: {frames_dir}")
    if supervision_cache is None or not supervision_cache.exists():
        raise FileNotFoundError(f"part teacher selection requires supervision_cache: {supervision_cache}")
    if semantics_npz is None or not semantics_npz.exists():
        raise FileNotFoundError(f"part teacher selection requires semantics_npz: {semantics_npz}")
    bank_tag = str(stage2_cfg.get('part_teacher_bank_tag', '')).strip() or out_dir.name
    bank_dir = project_root / 'outputs' / 'stage2' / 'teacher_bank' / bank_tag
    select_cmd = [
        sys.executable, str(select_script),
        '--project_root', str(project_root),
        '--stage1_dir', str(stage1_dir),
        '--frames_dir', str(frames_dir),
        '--supervision_cache', str(supervision_cache),
        '--semantics_npz', str(semantics_npz),
        '--out_dir', str(bank_dir),
    ]
    for key, cli_name in [
        ('part_teacher_min_pixels', '--min_pixels'),
        ('part_teacher_min_score', '--min_score'),
        ('part_teacher_cluster_gap', '--cluster_gap'),
        ('part_teacher_max_keep_per_part', '--max_keep_per_part'),
        ('part_teacher_body_bucket_count', '--body_bucket_count'),
        ('part_teacher_body_min_keep', '--body_min_keep'),
        ('part_teacher_extra_score_margin', '--extra_score_margin'),
        ('part_teacher_extra_min_iou', '--extra_min_iou'),
        ('part_teacher_extra_min_precision', '--extra_min_precision'),
        ('part_teacher_extra_min_recall', '--extra_min_recall'),
    ]:
        if key in stage2_cfg:
            select_cmd.extend([cli_name, str(stage2_cfg[key])])
    view_bucket_parts = str(stage2_cfg.get('part_teacher_view_bucket_parts', '')).strip()
    if view_bucket_parts:
        select_cmd.extend(['--view_bucket_parts', view_bucket_parts])
    logger.info('Running part teacher bank: %s', ' '.join(select_cmd))
    subprocess.run(select_cmd, cwd=str(project_root), check=True)
    teacher_bank_json = bank_dir / 'teacher_bank.json'
    if not teacher_bank_json.exists():
        raise FileNotFoundError(f"part teacher bank missing: {teacher_bank_json}")
    return teacher_bank_json


def _load_teacher_bank(teacher_bank_json: Path | None) -> dict | None:
    if teacher_bank_json is None or (not teacher_bank_json.exists()):
        return None
    return json.loads(teacher_bank_json.read_text(encoding='utf-8'))


def _select_auto_face_teachers(*, stage2_cfg: dict, teacher_bank_json: Path | None) -> list[str]:
    manual = _cfg_string_list(stage2_cfg.get('face_postfit_teacher_frames', []), [])
    if len(manual) > 0 and not _cfg_list_is_auto(stage2_cfg.get('face_postfit_teacher_frames', [])):
        return manual
    bank = _load_teacher_bank(teacher_bank_json)
    if bank is None:
        raise RuntimeError('face_postfit_enable=true but no teacher bank is available for automatic face teacher selection')
    face_payload = (bank.get('teacher_bank', {}) or {}).get('face', {})
    teachers = [str(x.get('frame_name', '')).strip() for x in list(face_payload.get('teachers', [])) if str(x.get('frame_name', '')).strip()]
    if len(teachers) <= 0:
        raise RuntimeError('automatic face teacher selection found no usable face teacher in teacher bank')
    return [teachers[0]]


def _select_auto_train_parts(*, stage2_cfg: dict, teacher_bank_json: Path | None, fixed_parts: list[str]) -> list[str]:
    manual = _cfg_string_list(stage2_cfg.get('part_postfit_train_parts', []), [])
    if len(manual) > 0 and not _cfg_list_is_auto(stage2_cfg.get('part_postfit_train_parts', [])):
        return manual
    bank = _load_teacher_bank(teacher_bank_json)
    if bank is None:
        raise RuntimeError('part_postfit_enable=true but no teacher bank is available for automatic train_parts selection')
    fixed = {str(x).strip() for x in fixed_parts if str(x).strip()}
    parts = []
    for part_name, payload in (bank.get('teacher_bank', {}) or {}).items():
        part_name = str(part_name).strip()
        if (not part_name) or (part_name in fixed) or part_name == 'background':
            continue
        teachers = list((payload or {}).get('teachers', []))
        if len(teachers) > 0:
            parts.append(part_name)
    if len(parts) <= 0:
        raise RuntimeError('automatic train_parts selection found no usable non-fixed parts in teacher bank')
    return parts


def _maybe_run_face_postfit(*, project_root: Path, stage1_dir: Path, ckpt_path: Path, out_dir: Path, stage2_cfg: dict, teacher_bank_json: Path | None, logger) -> None:
    if not bool(stage2_cfg.get('face_postfit_enable', False)):
        return
    teacher_frames = _select_auto_face_teachers(stage2_cfg=stage2_cfg, teacher_bank_json=teacher_bank_json)
    iters = int(max(int(stage2_cfg.get('face_postfit_iters', 1000)), 1))
    target_source = str(stage2_cfg.get('face_postfit_target_source', 'inner_rgb')).strip() or 'inner_rgb'
    tag = str(stage2_cfg.get('face_postfit_tag', '')).strip() or f"{out_dir.name}_facepostfit"
    fit_script = project_root / 'src' / 'avatar_gs' / 'cli' / 'stage2_face_fit.py'
    if not fit_script.exists():
        raise FileNotFoundError(f"face postfit script missing: {fit_script}")
    cmd = [sys.executable, str(fit_script), '--project_root', str(project_root), '--stage1_dir', str(stage1_dir), '--stage2_ckpt', str(ckpt_path), '--teacher_frames', *teacher_frames, '--target_source', str(target_source), '--iters', str(iters), '--tag', tag]
    logger.info('Running face postfit: %s', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)
    fit_train_dir = project_root / 'outputs' / 'stage2' / 'face_fit' / tag / 'train'
    fit_eval_dir = project_root / 'outputs' / 'stage2' / 'face_fit' / tag
    preview_ckpt = fit_train_dir / 'gaussians3d_facefit_preview.pt'
    if not preview_ckpt.exists():
        raise FileNotFoundError(f"face postfit preview checkpoint missing: {preview_ckpt}")
    backup_ckpt = out_dir / 'gaussians3d_pre_facepostfit.pt'
    shutil.copy2(ckpt_path, backup_ckpt)
    shutil.copy2(preview_ckpt, ckpt_path)
    summary_src = fit_eval_dir / 'summary.json'
    if summary_src.exists():
        shutil.copy2(summary_src, out_dir / 'face_postfit_summary.json')
    logger.info('Applied face postfit to main checkpoint: teachers=%s | target_source=%s | iters=%d | backup=%s', teacher_frames, target_source, iters, backup_ckpt)


def _maybe_run_part_teacher_postfit(*, project_root: Path, stage1_dir: Path, ckpt_path: Path, out_dir: Path, stage2_cfg: dict, teacher_bank_json: Path | None, logger) -> None:
    if not bool(stage2_cfg.get('part_postfit_enable', False)):
        return
    fit_script = project_root / 'src' / 'avatar_gs' / 'cli' / 'stage2_part_postfit.py'
    if not fit_script.exists():
        raise FileNotFoundError(f"part postfit script missing: {fit_script}")
    if teacher_bank_json is None or not teacher_bank_json.exists():
        raise FileNotFoundError(f"part teacher bank missing: {teacher_bank_json}")
    frames_dir = _resolve_project_path(project_root, stage2_cfg.get('part_postfit_frames_dir', project_root / 'data' / 'frames' / 'input'))
    supervision_cache = _resolve_project_path(project_root, stage2_cfg.get('prepared_supervision_cache_path', out_dir / 'supervision_cache.npz'))
    semantics_npz = _resolve_project_path(project_root, stage2_cfg.get('smpl_face_semantics_npz', stage1_dir / 'fashn_face_semantics' / 'smpl_face_semantics.npz'))
    if frames_dir is None or not frames_dir.exists():
        raise FileNotFoundError(f"part postfit requires frames_dir: {frames_dir}")
    if supervision_cache is None or not supervision_cache.exists():
        raise FileNotFoundError(f"part postfit requires supervision_cache: {supervision_cache}")
    if semantics_npz is None or not semantics_npz.exists():
        raise FileNotFoundError(f"part postfit requires semantics_npz: {semantics_npz}")
    fixed_parts = _cfg_string_list(stage2_cfg.get('part_postfit_fixed_parts', ['face', 'hair']), ['face', 'hair'])
    train_parts = _select_auto_train_parts(stage2_cfg=stage2_cfg, teacher_bank_json=teacher_bank_json, fixed_parts=fixed_parts)
    iters = int(max(int(stage2_cfg.get('part_postfit_iters', 300)), 1))
    target_source = str(stage2_cfg.get('part_postfit_target_source', 'inner_rgb')).strip() or 'inner_rgb'
    tag = str(stage2_cfg.get('part_postfit_tag', '')).strip() or f"{out_dir.name}_partpostfit"
    fixed_source_ckpt = _resolve_project_path(project_root, stage2_cfg.get('part_postfit_fixed_source_ckpt', str(ckpt_path))) or ckpt_path
    auto_face_teacher = _select_auto_face_teachers(stage2_cfg=stage2_cfg, teacher_bank_json=teacher_bank_json)[0]
    fixed_box_frame = str(stage2_cfg.get('part_postfit_fixed_box_frame', '')).strip() or auto_face_teacher
    fit_cmd = [sys.executable, str(fit_script), '--project_root', str(project_root), '--stage1_dir', str(stage1_dir), '--stage2_ckpt', str(ckpt_path), '--teacher_bank_json', str(teacher_bank_json), '--supervision_cache', str(supervision_cache), '--frames_dir', str(frames_dir), '--smpl_face_semantics_npz', str(semantics_npz), '--target_source', target_source, '--iters', str(iters), '--cloth_min_support', str(int(stage2_cfg.get('part_postfit_cloth_min_support', 2))), '--cloth_core_support', str(int(stage2_cfg.get('part_postfit_cloth_core_support', 3))), '--fixed_source_ckpt', str(fixed_source_ckpt), '--fixed_box_frame', fixed_box_frame, '--face_teacher_override', auto_face_teacher, '--tag', tag, '--face_force_single_teacher', '--fixed_parts', *fixed_parts, '--train_parts', *train_parts]
    if 'part_postfit_cloth_core_erode_px' in stage2_cfg:
        fit_cmd.extend(['--cloth_core_erode_px', str(int(stage2_cfg['part_postfit_cloth_core_erode_px']))])
    logger.info('Running part postfit: %s', ' '.join(fit_cmd))
    subprocess.run(fit_cmd, cwd=str(project_root), check=True)
    fit_train_dir = project_root / 'outputs' / 'stage2' / 'part_fit' / tag / 'train'
    preview_ckpt = fit_train_dir / 'gaussians3d_partfit_preview.pt'
    if not preview_ckpt.exists():
        raise FileNotFoundError(f"part postfit preview checkpoint missing: {preview_ckpt}")
    backup_ckpt = out_dir / 'gaussians3d_pre_partpostfit.pt'
    shutil.copy2(ckpt_path, backup_ckpt)
    shutil.copy2(preview_ckpt, ckpt_path)
    logger.info('Applied part postfit to main checkpoint: teacher_bank=%s | fixed_source=%s | train_parts=%s | fixed_parts=%s | backup=%s', teacher_bank_json, fixed_source_ckpt, train_parts, fixed_parts, backup_ckpt)


def _resolve_project_path(project_root: Path, path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path_str = str(path_like).strip()
    if len(path_str) <= 0:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _filter_frames_by_stage1_align(
    *,
    project_root: Path,
    stage1_dir: Path,
    stage2_cfg: dict,
    frames: list[str],
    logger,
) -> list[str]:
    if not bool(stage2_cfg.get("stage1_align_filter_enable", True)):
        return frames
    if len(frames) <= 1:
        return frames
    seq_npz_path = _resolve_project_path(
        project_root,
        stage2_cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz")),
    )
    if seq_npz_path is None or (not seq_npz_path.exists()):
        raise FileNotFoundError(f"stage1 align filter requires smpl_params_sequence.npz: {seq_npz_path}")
    seq = np.load(seq_npz_path, allow_pickle=True)
    if "frame_names" not in seq.files:
        raise RuntimeError(f"Bad smpl_params_sequence format, missing frame_names: {seq_npz_path}")
    metric_key = str(stage2_cfg.get("stage1_align_filter_metric", "camera_align_iou")).strip() or "camera_align_iou"
    if metric_key not in seq.files:
        raise RuntimeError(f"stage1 align filter metric not found in {seq_npz_path}: {metric_key}")
    seq_names = [str(x) for x in np.asarray(seq["frame_names"]).reshape(-1).tolist()]
    scores = np.asarray(seq[metric_key], dtype=np.float32).reshape(-1)
    n = min(len(seq_names), int(scores.shape[0]))
    score_map = {seq_names[i]: float(scores[i]) for i in range(n)}
    last_frame = str(frames[-1])
    keep_ratio_cfg = stage2_cfg.get("stage1_align_filter_keep_ratio", None)
    keep_ratio = None if keep_ratio_cfg is None else float(keep_ratio_cfg)
    if keep_ratio is not None and keep_ratio > 0.0:
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        keep_n = int(np.clip(int(np.ceil(keep_ratio * float(len(frames)))), 1, len(frames)))
        ranked = sorted(
            [(float(score_map.get(str(fn), -1.0)), idx, str(fn)) for idx, fn in enumerate(frames)],
            key=lambda x: (-x[0], x[1]),
        )
        selected = {name for _, _, name in ranked[:keep_n]}
        if last_frame not in selected:
            selected.add(last_frame)
            if len(selected) > keep_n:
                removable = [
                    (float(score_map.get(name, -1.0)), idx, name)
                    for idx, name in enumerate(frames)
                    if (name in selected) and (name != last_frame)
                ]
                removable.sort(key=lambda x: (x[0], -x[1]))
                if len(removable) > 0:
                    selected.remove(removable[0][2])
        kept = [fn for fn in frames if str(fn) in selected]
        score_floor = min(float(score_map.get(str(fn), -1.0)) for fn in kept) if len(kept) > 0 else -1.0
        logger.info(
            f"Stage1 align frame filter: kept {len(kept)}/{len(frames)} frames "
            f"with top {keep_ratio * 100.0:.1f}% by {metric_key}; "
            f"effective_floor={score_floor:.5f}; forced_keep_last={last_frame}"
        )
    else:
        min_metric = float(stage2_cfg.get("stage1_align_filter_min", 0.83))
        kept = [fn for fn in frames if (float(score_map.get(fn, -1.0)) >= min_metric) or (str(fn) == last_frame)]
        if len(kept) <= 0:
            kept = [last_frame]
        logger.info(
            f"Stage1 align frame filter: kept {len(kept)}/{len(frames)} frames "
            f"with {metric_key}>={min_metric:.2f}; forced_keep_last={last_frame}"
        )
    dropped = int(len(frames) - len(kept))
    if dropped > 0:
        logger.info(f"Stage1 align frame filter: dropped {dropped} low-quality frame(s)")
    return kept


def _load_stage1_align_scores(
    *,
    project_root: Path,
    stage1_dir: Path,
    stage2_cfg: dict,
    frames: list[str],
    logger,
) -> list[float] | None:
    if len(frames) <= 0:
        return None
    seq_npz_path = _resolve_project_path(
        project_root,
        stage2_cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz")),
    )
    if seq_npz_path is None or (not seq_npz_path.exists()):
        logger.warning(
            "Stage1 align scores unavailable: missing smpl_params_sequence.npz at %s; "
            "outer geometric accumulation will fall back to equal weights.",
            seq_npz_path,
        )
        return None
    seq = np.load(seq_npz_path, allow_pickle=True)
    if "frame_names" not in seq.files:
        logger.warning(
            "Stage1 align scores unavailable: frame_names missing in %s; "
            "outer geometric accumulation will fall back to equal weights.",
            seq_npz_path,
        )
        return None
    metric_key = str(stage2_cfg.get("stage1_align_filter_metric", "camera_align_iou")).strip() or "camera_align_iou"
    if metric_key not in seq.files:
        logger.warning(
            "Stage1 align scores unavailable: metric %s missing in %s; "
            "outer geometric accumulation will fall back to equal weights.",
            metric_key,
            seq_npz_path,
        )
        return None
    seq_names = [str(x) for x in np.asarray(seq["frame_names"]).reshape(-1).tolist()]
    scores = np.asarray(seq[metric_key], dtype=np.float32).reshape(-1)
    n = min(len(seq_names), int(scores.shape[0]))
    score_map = {seq_names[i]: float(scores[i]) for i in range(n)}
    align_scores = [float(score_map.get(str(fn), 1.0)) for fn in frames]
    logger.info(
        "Loaded %d stage1 align score(s) from %s for outer geometric accumulation.",
        len(align_scores),
        seq_npz_path,
    )
    return align_scores


def _load_stage1_frame_Ks(
    *,
    project_root: Path,
    stage1_dir: Path,
    stage2_cfg: dict,
    frames: list[str],
    logger,
) -> list[list[float]] | None:
    if len(frames) <= 0:
        return None
    seq_npz_path = _resolve_project_path(
        project_root,
        stage2_cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz")),
    )
    if seq_npz_path is None or (not seq_npz_path.exists()):
        logger.warning(
            "Stage1 frame intrinsics unavailable: missing smpl_params_sequence.npz at %s; "
            "stage2 will fall back to cameras.json K.",
            seq_npz_path,
        )
        return None
    seq = np.load(seq_npz_path, allow_pickle=True)
    required = ("frame_names", "camera_focal_xy", "camera_princpt_xy")
    if any(key not in seq.files for key in required):
        logger.warning(
            "Stage1 frame intrinsics unavailable: missing one of %s in %s; "
            "stage2 will fall back to cameras.json K.",
            required,
            seq_npz_path,
        )
        return None
    seq_names = [str(x) for x in np.asarray(seq["frame_names"]).reshape(-1).tolist()]
    focal_xy = np.asarray(seq["camera_focal_xy"], dtype=np.float32).reshape(-1, 2)
    princpt_xy = np.asarray(seq["camera_princpt_xy"], dtype=np.float32).reshape(-1, 2)
    n = min(len(seq_names), int(focal_xy.shape[0]), int(princpt_xy.shape[0]))
    K_map: dict[str, list[float]] = {}
    for i in range(n):
        fx = float(focal_xy[i, 0])
        fy = float(focal_xy[i, 1])
        cx = float(princpt_xy[i, 0])
        cy = float(princpt_xy[i, 1])
        K_map[seq_names[i]] = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    missing = [str(fn) for fn in frames if str(fn) not in K_map]
    if len(missing) > 0:
        logger.warning(
            "Stage1 frame intrinsics missing for %d selected frame(s); "
            "stage2 will fall back to cameras.json K for those frames. Example: %s",
            len(missing),
            missing[:3],
        )
        return None
    frame_Ks = [list(K_map[str(fn)]) for fn in frames]
    logger.info(
        "Loaded %d stage1 frame intrinsic matrix/matrices from %s for stage2 frame-wise rendering.",
        len(frame_Ks),
        seq_npz_path,
    )
    return frame_Ks


def _align_checkpoint_camera_tensor(
    *,
    tensor: torch.Tensor | np.ndarray | list | None,
    ckpt_frames: list[str] | None,
    frames: list[str],
    tensor_name: str,
    logger,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    t = torch.as_tensor(tensor, dtype=torch.float32)
    if t.ndim != 2 or int(t.shape[1]) != 3:
        logger.warning(
            "Ignoring %s from init checkpoint: expected shape [N,3], got %s.",
            tensor_name,
            tuple(t.shape),
        )
        return None
    if ckpt_frames is None or len(ckpt_frames) <= 0:
        if int(t.shape[0]) == len(frames):
            return t.clone()
        logger.warning(
            "Ignoring %s from init checkpoint: missing frame names and length mismatch (%d vs %d).",
            tensor_name,
            int(t.shape[0]),
            len(frames),
        )
        return None
    frame_to_idx = {str(fn): int(i) for i, fn in enumerate(ckpt_frames)}
    missing = [str(fn) for fn in frames if str(fn) not in frame_to_idx]
    if len(missing) > 0:
        logger.warning(
            "Ignoring %s from init checkpoint: %d selected frame(s) missing in checkpoint frame list.",
            tensor_name,
            len(missing),
        )
        return None
    idx = torch.as_tensor([frame_to_idx[str(fn)] for fn in frames], dtype=torch.long)
    if int(idx.max().item()) >= int(t.shape[0]):
        logger.warning(
            "Ignoring %s from init checkpoint: frame index exceeds tensor length (%d).",
            tensor_name,
            int(t.shape[0]),
        )
        return None
    return t.index_select(0, idx).clone()


def _load_stage2_init_checkpoint(
    *,
    project_root: Path,
    init_ckpt_cfg: str | Path | None,
    frames: list[str],
    logger,
) -> tuple[Gaussian3D | None, torch.Tensor | None, torch.Tensor | None, Path | None]:
    if init_ckpt_cfg is None or len(str(init_ckpt_cfg).strip()) <= 0:
        return None, None, None, None
    init_ckpt_path = _resolve_project_path(project_root, init_ckpt_cfg)
    if init_ckpt_path is None:
        return None, None, None, None
    if not init_ckpt_path.exists():
        raise FileNotFoundError(f"init_ckpt not found: {init_ckpt_path}")
    payload = torch.load(str(init_ckpt_path), map_location="cpu")
    if not isinstance(payload, dict) or "gaussians" not in payload:
        raise RuntimeError(f"Bad init_ckpt format, missing 'gaussians': {init_ckpt_path}")
    g = Gaussian3D.load_state_dict(payload["gaussians"])
    ckpt_frames_raw = payload.get("frames", None)
    ckpt_frames = [str(x) for x in ckpt_frames_raw] if isinstance(ckpt_frames_raw, (list, tuple)) else None
    init_cam_w = _align_checkpoint_camera_tensor(
        tensor=payload.get("cam_w", None),
        ckpt_frames=ckpt_frames,
        frames=frames,
        tensor_name="cam_w",
        logger=logger,
    )
    init_cam_t = _align_checkpoint_camera_tensor(
        tensor=payload.get("cam_t", None),
        ckpt_frames=ckpt_frames,
        frames=frames,
        tensor_name="cam_t",
        logger=logger,
    )
    logger.info("Loaded stage2 init checkpoint: %s", init_ckpt_path)
    return g, init_cam_w, init_cam_t, init_ckpt_path

def main():
    parser = parse_args_base("Stage2: 3D Gaussian Splatting training with learnable camera (no COLMAP)")
    parser.add_argument(
        "--prepare_supervision_only",
        action="store_true",
        help="Only generate stage2 supervision artifacts/previews and exit.",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default=None,
        choices=("full", "inner_only", "outer_only"),
        help="Stage2 training mode. full=default schedule, inner_only=cache inner result, outer_only=resume from init_ckpt and train outer only.",
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Optional stage2 checkpoint used to initialize gaussians/camera state, typically an inner cache for outer_only runs.",
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage2_train_gaussian", str(log_dir / "stage2_train_gaussian.log"))

    train_cfg = Stage2TrainConfig.from_cfg(cfg=cfg, project_root=project_root, log_dir=log_dir)
    stage1_dir = train_cfg.stage1_dir
    out_dir = ensure_dir(train_cfg.out_dir)
    stage2_cfg = train_cfg.stage2_raw
    deprecated_trigger_keys = [
        k
        for k in (
            "outer_supervision_mode",
            "outer_residual_only",
            "outer_vis_floor",
            "outer_vis_edge_floor",
            "outer_edge_visibility_boost",
            "outer_support_sample_dilate_px",
            "outer_reassign_enable",
            "outer_reassign_every",
            "outer_reassign_ratio",
            "outer_reassign_seed_n",
            "outer_reassign_inside_only",
            "outer_reassign_warmup_iters",
            "outer_head_min_ratio",
            "outer_contour_dilate_px",
            "outer_contour_band_px",
            "outer_spawn_allow_outer_source",
            "outer_reassign_allow_outer_source",
            "outer_reassign_preserve_source_offset",
        )
        if k in stage2_cfg
    ]
    if len(deprecated_trigger_keys) > 0:
        raise ValueError(
            "Deprecated stage2 keys detected: "
            f"{deprecated_trigger_keys}. "
            "Remove them and use only trigger-based keys: "
            "outer_trigger_smpl_gap_px / outer_trigger_edge_th / "
            "outer_trigger_min_pixels / outer_trigger_ring_width_px / "
            "outer_trigger_use_edge_gate."
        )

    if torch.cuda.is_available() and bool(stage2_cfg.get("enable_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(stage2_cfg.get("cudnn_benchmark", True))

    init_mesh_cfg = str(stage2_cfg.get("init_mesh_obj", ""))
    mesh_obj = Path(init_mesh_cfg)
    if not mesh_obj.is_absolute():
        cand_paths = [
            Path(init_mesh_cfg),
            project_root / mesh_obj,
            stage1_dir / mesh_obj,
            stage1_dir / mesh_obj.name,
        ]
        picked = None
        for cp in cand_paths:
            try:
                if cp.exists():
                    picked = cp
                    break
            except Exception:
                continue
        if picked is not None:
            mesh_obj = picked
        else:
            mesh_obj = project_root / mesh_obj
    if (not str(init_mesh_cfg).strip()) or (not mesh_obj.exists()):
        fallback_mesh = None
        deform_cands = sorted(stage1_dir.glob("*_deform.obj"))
        if len(deform_cands) > 0:
            fallback_mesh = deform_cands[-1]
        else:
            for cand in (stage1_dir / "smpl_mesh_canonical.obj", stage1_dir / "smpl_mesh.obj"):
                if cand.exists():
                    fallback_mesh = cand
                    break
            # Also support stage1 numeric sequence exports: outputs/stage1/sequence/*.obj.
            if fallback_mesh is None:
                seq_dir = stage1_dir / "sequence"
                if seq_dir.exists():
                    try:
                        seq_provider = SequenceMeshProvider(seq_dir)
                        seq_provider.scan()
                        seq_ids = seq_provider.get_frame_ids()
                        fallback_mesh = seq_provider.get_path(seq_ids[-1])
                    except Exception:
                        fallback_mesh = None
        if fallback_mesh is None:
            raise FileNotFoundError(
                f"init mesh not found: {mesh_obj}. "
                f"Checked fallback: {stage1_dir / '*_deform.obj'}, {stage1_dir / 'smpl_mesh_canonical.obj'}, {stage1_dir / 'smpl_mesh.obj'}, "
                f"{stage1_dir / 'sequence' / '*.obj'}."
            )
        mesh_obj = fallback_mesh

    cam = read_json(stage1_dir / "cameras.json")

    frames_dir = project_root / cfg["data"]["frames_dir"]
    n_frames = int(stage2_cfg.get("n_frames", 20))

    if "h36m" in cfg.get("data", {}):
        if resolve_h36m_image_paths is None:
            raise ModuleNotFoundError(
                "data.h36m config is provided, but avatar_gs.data.h36m is unavailable."
            )
        paths = resolve_h36m_image_paths(project_root=project_root, data_cfg=cfg["data"])
        # Use the actual image root to load files by name.
        frames_dir = paths[0].parent
        frames = [p.name for p in paths]
    else:
        frames = sorted([p.name for p in frames_dir.glob("*.jpg")])
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Frame selection strategy
    frame_select = str(stage2_cfg.get("frame_select", "first")).lower()
    if frame_select == "stage1_sequence":
        seq_npz_cfg = stage2_cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz"))
        seq_npz_path = _resolve_project_path(project_root, seq_npz_cfg)
        if seq_npz_path is None or (not seq_npz_path.exists()):
            raise FileNotFoundError(
                f"frame_select=stage1_sequence requires bind_sequence_npz, missing: {seq_npz_path}"
            )
        seq = np.load(seq_npz_path, allow_pickle=True)
        if "frame_names" not in seq.files:
            raise RuntimeError(f"bind_sequence_npz has no frame_names: {seq_npz_path}")
        seq_names = [str(x) for x in seq["frame_names"].tolist()]
        frame_set = set(frames)
        picked = [n for n in seq_names if n in frame_set]
        if len(picked) == 0:
            # basename fallback
            by_name = {Path(f).name: f for f in frames}
            picked = [by_name[Path(n).name] for n in seq_names if Path(n).name in by_name]
        if len(picked) == 0:
            raise RuntimeError(
                "frame_select=stage1_sequence matched zero frames between data/frames and stage1 sequence frame_names."
            )
        frames = picked
    elif frame_select == "last":
        frames = [frames[-1]]
    elif frame_select == "last_k":
        k = max(1, int(n_frames))
        frames = frames[-k:]
    elif frame_select == "uniform_k":
        k = max(1, int(n_frames))
        if k >= len(frames):
            picked_idx = np.arange(len(frames), dtype=np.int64)
        else:
            idx = np.linspace(0, len(frames) - 1, num=k, dtype=np.float64)
            idx = np.clip(np.round(idx).astype(np.int64), 0, len(frames) - 1)
            picked_idx = np.unique(idx)
        frames = [frames[int(i)] for i in picked_idx.tolist()]
    elif frame_select == "indices":
        indices = stage2_cfg.get("frame_indices", [])
        if isinstance(indices, (int, np.integer, float, np.floating, str)):
            indices = [indices]
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise ValueError("frame_select=indices but frame_indices is empty.")
        picked = []
        for idx in indices:
            i = int(idx)
            if i < 0:
                i = len(frames) + i
            if i < 0 or i >= len(frames):
                raise IndexError(f"frame index out of range: {idx}")
            picked.append(frames[i])
        frames = picked
    elif frame_select == "first":
        # default: first N frames
        if n_frames > 0:
            frames = frames[:n_frames]
    else:
        raise ValueError(f"Unsupported frame_select: {frame_select}. Use first|last|last_k|uniform_k|indices|stage1_sequence")

    frames = _filter_frames_by_stage1_align(
        project_root=project_root,
        stage1_dir=stage1_dir,
        stage2_cfg=stage2_cfg,
        frames=frames,
        logger=logger,
    )
    frame_align_scores = _load_stage1_align_scores(
        project_root=project_root,
        stage1_dir=stage1_dir,
        stage2_cfg=stage2_cfg,
        frames=frames,
        logger=logger,
    )
    frame_Ks = _load_stage1_frame_Ks(
        project_root=project_root,
        stage1_dir=stage1_dir,
        stage2_cfg=stage2_cfg,
        frames=frames,
        logger=logger,
    )

    logger.info(f"Mesh: {mesh_obj}")
    logger.info(f"Frames: {len(frames)} used for training")
    logger.info(f"Out dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_mode = str(cfg["stage2"].get("camera_mode", "learnable"))
    lr_cam = float(cfg["stage2"].get("lr_cam", 1e-3))
    cam_reg_weight = float(cfg["stage2"].get("cam_reg_weight", 0.0))
    training_mode = str(
        args.training_mode
        if args.training_mode is not None
        else cfg["stage2"].get("training_mode", "full")
    ).strip().lower()
    if training_mode not in ("full", "inner_only", "outer_only"):
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    init_ckpt_cfg = args.init_ckpt if args.init_ckpt is not None else cfg["stage2"].get("init_ckpt", None)
    init_ckpt_cfg_str = None if init_ckpt_cfg is None else str(init_ckpt_cfg).strip()
    if init_ckpt_cfg_str in ("", ".", "./", ".\\"):
        init_ckpt_cfg = None
    else:
        init_ckpt_cfg = init_ckpt_cfg_str
    inner_ckpt_cfg = cfg["stage2"].get("inner_ckpt_path", None)
    inner_ckpt_path = _resolve_project_path(project_root, inner_ckpt_cfg)
    if inner_ckpt_path is None:
        inner_ckpt_path = out_dir / "gaussians3d_inner.pt"
    cfg["stage2"]["training_mode"] = training_mode
    cfg["stage2"]["init_ckpt"] = None if init_ckpt_cfg is None else str(init_ckpt_cfg)
    cfg["stage2"]["inner_ckpt_path"] = str(inner_ckpt_path)
    logger.info("Stage2 training mode: %s", training_mode)
    if len(frames) == 1 and camera_mode.lower() != "fixed" and cam_reg_weight <= 0.0:
        logger.warning(
            "Single-frame training with learnable camera and cam_reg_weight<=0 "
            "is ill-posed; switching to fixed camera."
        )
        camera_mode = "fixed"
        lr_cam = 0.0

    # Frame sampling / loss weights (emphasize last frame)
    frame_weights = stage2_cfg.get("frame_weights", None)
    frame_loss_weights = stage2_cfg.get("frame_loss_weights", None)
    last_sample_w = stage2_cfg.get("last_frame_sample_weight", None)
    last_loss_w = stage2_cfg.get("last_frame_loss_weight", None)
    if frame_weights is None and last_sample_w is not None:
        frame_weights = [1.0] * len(frames)
        frame_weights[-1] = float(last_sample_w)
    if frame_loss_weights is None and last_loss_w is not None:
        frame_loss_weights = [1.0] * len(frames)
        frame_loss_weights[-1] = float(last_loss_w)

    init_g, init_cam_w, init_cam_t, init_ckpt_path = _load_stage2_init_checkpoint(
        project_root=project_root,
        init_ckpt_cfg=init_ckpt_cfg,
        frames=frames,
        logger=logger,
    )
    if training_mode == "outer_only" and init_g is None:
        raise ValueError("training_mode=outer_only requires init_ckpt.")
    if init_g is not None:
        g = init_g
        logger.info(f"Loaded gaussians from checkpoint: N={int(g.xyz.shape[0])}")
    else:
        init_cam_w = None
        init_cam_t = None
        g = init_gaussians_from_mesh_3d(
            mesh_obj=mesh_obj,
            n_keep=int(cfg["stage2"].get("n_gaussians", 3000)),
            seed=int(cfg.get("seed", 42)),
            n_bg=int(cfg["stage2"].get("n_gaussians_bg", 0)),
            bg_extent=tuple(cfg["stage2"].get("bg_extent", (3.0, 3.0, 0.6))),
            bg_center_z=float(cfg["stage2"].get("bg_center_z", 2.5)),
            bg_log_scale=float(cfg["stage2"].get("bg_log_scale", -3.5)),
            bg_opacity_logit=float(cfg["stage2"].get("bg_opacity_logit", -1.5)),
            bg_exclude_bbox_pad=float(cfg["stage2"].get("bg_exclude_bbox_pad", 0.0)),
            sample_on_surface=bool(cfg["stage2"].get("sample_on_surface", True)),
            inner_count_auto=bool(cfg["stage2"].get("inner_count_auto", False)),
            inner_count_kV=float(cfg["stage2"].get("inner_count_kV", 1.2)),
            inner_count_kF=float(cfg["stage2"].get("inner_count_kF", 0.35)),
            inner_count_min=int(cfg["stage2"].get("inner_count_min", 80000)),
            inner_count_max=int(cfg["stage2"].get("inner_count_max", 220000)),
            shell_enable=bool(cfg["stage2"].get("shell_enable", False)),
            shell_init_ratio=float(cfg["stage2"].get("shell_init_ratio", 0.15)),
            shell_offset_n=float(cfg["stage2"].get("shell_offset_max_n", 0.015)),
            inner_dual_candidate_enable=bool(cfg["stage2"].get("inner_dual_candidate_enable", True)),
            inner_high_curv_dual_activate_th=float(cfg["stage2"].get("inner_high_curv_dual_activate_th", 0.2)),
            inner_face_candidate_ratio=float(cfg["stage2"].get("inner_face_candidate_ratio", 0.35)),
            inner_vertex_candidate_ratio=float(cfg["stage2"].get("inner_vertex_candidate_ratio", 0.65)),
            inner_bind_mode_default=str(cfg["stage2"].get("inner_bind_mode_default", "vertex")),
            inner_local_scale_init_enable=bool(cfg["stage2"].get("inner_local_scale_init_enable", True)),
            inner_local_scale_init_vertex_ratio=float(cfg["stage2"].get("inner_local_scale_init_vertex_ratio", 0.40)),
            inner_local_scale_init_face_ratio=float(cfg["stage2"].get("inner_local_scale_init_face_ratio", 0.75)),
            inner_local_scale_init_normal_ratio=float(cfg["stage2"].get("inner_local_scale_init_normal_ratio", 0.45)),
            inner_local_scale_init_min=float(cfg["stage2"].get("inner_local_scale_init_min", 8e-4)),
            inner_local_scale_init_max=float(cfg["stage2"].get("inner_local_scale_init_max", 1.2e-2)),
            smpl_face_semantics_npz=cfg["stage2"].get("smpl_face_semantics_npz", None),
            inner_face_semantic_sampling_boost=float(cfg["stage2"].get("inner_face_semantic_sampling_boost", 1.0)),
            inner_face_semantic_scale_mult=float(cfg["stage2"].get("inner_face_semantic_scale_mult", 1.0)),
            outer_init_from_inner=bool(cfg["stage2"].get("outer_init_from_inner", False)),
            outer_dormant_slot_ratio=float(cfg["stage2"].get("outer_dormant_slot_ratio", cfg["stage2"].get("shell_init_ratio", 0.15))),
            outer_dormant_face_ratio=float(cfg["stage2"].get("outer_dormant_face_ratio", 0.0)),
            outer_face_anchor_rule=str(cfg["stage2"].get("outer_face_anchor_rule", "centroid")),
            outer_pruned_opacity_logit=float(cfg["stage2"].get("outer_pruned_opacity_logit_max", -12.0)),
        )
        logger.info(f"Initialized gaussians: N={int(g.xyz.shape[0])}")

    # ---- Optional dynamic binding to SMPL mesh sequence ----
    bind_to_mesh = bool(cfg["stage2"].get("bind_to_mesh", False))
    if bind_to_mesh and g.bind_idx is None:
        raise RuntimeError("bind_to_mesh=true but gaussians have no bind_idx.")
    mesh_runtime = Stage2MeshRuntime.from_stage2_config(
        project_root=project_root,
        stage1_dir=stage1_dir,
        cfg=cfg["stage2"],
        frames=frames,
        logger=logger,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        bind_to_mesh=bind_to_mesh,
    )
    bind_mesh_dir = mesh_runtime.context.bind_mesh_dir
    bind_mesh_indices = mesh_runtime.context.bind_mesh_indices
    bind_seq_npz_path = mesh_runtime.context.bind_seq_npz_path
    bind_vertices_npz_path = mesh_runtime.context.bind_vertices_npz_path
    bind_deform_obj_path = mesh_runtime.context.bind_deform_obj_path
    if bind_to_mesh and bind_mesh_dir is not None and not bind_mesh_dir.exists():
        raise FileNotFoundError(
            f"bind_mesh_sequence_dir not found: {bind_mesh_dir}. "
            "Run stage1 to generate numeric OBJ meshes under outputs/stage1."
        )
    if bind_to_mesh and (bind_seq_npz_path is None or (not bind_seq_npz_path.exists())):
        raise FileNotFoundError(
            f"Missing bind sequence npz: {bind_seq_npz_path}. "
            "Required to map frame names to mesh indices for binding."
        )
    if bind_to_mesh and bind_vertices_npz_path is not None:
        logger.info(f"Using dynamic bind vertices from: {bind_vertices_npz_path}")
    elif bind_to_mesh:
        logger.warning("Missing bind_vertices_npz; binding will fall back to OBJ sequence only.")
    if bind_to_mesh and bind_deform_obj_path is None:
        logger.warning("bind_deform_obj not found; render will use fallback.")

    cam_t_prior = tuple(cfg["stage2"].get("cam_t_prior", (0.0, 0.0, 0.0)))
    if (
        bind_to_mesh
        and bool(cfg["stage2"].get("bind_mesh_to_camera", False))
        and camera_mode.lower() == "fixed"
        and len(cam_t_prior) == 3
        and abs(float(cam_t_prior[0])) < 1e-8
        and abs(float(cam_t_prior[1])) < 1e-8
        and abs(float(cam_t_prior[2]) - 2.5) < 1e-8
    ):
        logger.warning(
            "cam_t_prior is [0,0,2.5] with bind_mesh_to_camera=true and fixed camera; "
            "this causes projection shrink/shift. Auto-correcting cam_t_prior to [0,0,0]."
        )
        cam_t_prior = (0.0, 0.0, 0.0)

    bind_rigid_align_for_train = cfg["stage2"].get("bind_rigid_align", None)
    _bind_cfg_norm = normalize_rigid_align_config(bind_rigid_align_for_train, default_enabled=False)
    require_pre_stage2_align = bool((_bind_cfg_norm or {}).get("require_pre_stage2", True)) if bool((_bind_cfg_norm or {}).get("enabled", False)) else False
    try:
        if (
            bind_to_mesh
            and _bind_cfg_norm is not None
            and bool(_bind_cfg_norm.get("enabled", False))
            and bool(_bind_cfg_norm.get("auto", False))
            and bool(_bind_cfg_norm.get("pre_stage2", True))
            and mesh_runtime.context.bind_faces_np is not None
            and bind_mesh_indices is not None
            and len(bind_mesh_indices) > 0
        ):
            first_img = cv2.imread(str(frames_dir / frames[0]), cv2.IMREAD_COLOR)
            if first_img is None:
                raise FileNotFoundError(f"Cannot read frame for pre-stage2 rigid align: {frames_dir / frames[0]}")
            pm_cfg = _bind_cfg_norm.get("person_masks_npz", None)
            if pm_cfg is None:
                pm_cfg = cfg["stage2"].get("person_masks_npz", None)
            pm_path = _resolve_project_path(project_root, pm_cfg)
            if pm_path is None:
                raise FileNotFoundError("bind_rigid_align pre-stage2 requires person mask file.")
            align_result = run_stage2_rigid_align_search(
                cfg=_bind_cfg_norm,
                frames=frames,
                bind_mesh_indices=bind_mesh_indices,
                K_np=np.asarray(cam["K"], dtype=np.float32).reshape(3, 3),
                image_hw=(int(first_img.shape[0]), int(first_img.shape[1])),
                bind_faces_np=mesh_runtime.context.bind_faces_np,
                mesh_loader=lambda mesh_idx: mesh_runtime.load_mesh_np_raw(int(mesh_idx))[0],
                person_masks_npz_path=pm_path,
                logger=logger,
                t_eval=np.asarray(cam_t_prior, dtype=np.float32).reshape(3,),
                bind_mesh_offset_np=mesh_runtime.context.bind_mesh_offset_np,
                cache_report_path=out_dir / "bind_rigid_align_pre_stage2.json",
            )
            bind_rigid_align_for_train = align_result.effective_cfg
            if bind_rigid_align_for_train is not None:
                report_path = out_dir / "bind_rigid_align_pre_stage2.json"
                report_payload = dict(bind_rigid_align_for_train)
                if align_result.metrics is not None:
                    report_payload.update(align_result.metrics)
                report_payload["person_masks_npz"] = str(pm_path)
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(report_payload, f, ensure_ascii=False, indent=2)
        else:
            bind_rigid_align_for_train = cfg["stage2"].get("bind_rigid_align", None)
    except Exception as e:
        if require_pre_stage2_align:
            raise RuntimeError(f"bind_rigid_align pre-stage2 failed and require_pre_stage2=true: {e}") from e
        logger.warning(
            "bind_rigid_align pre-stage2 failed (%s); fallback to in-train align.",
            str(e),
        )
        bind_rigid_align_for_train = cfg["stage2"].get("bind_rigid_align", None)

    _eff_norm = normalize_rigid_align_config(bind_rigid_align_for_train, default_enabled=False)
    if require_pre_stage2_align and bool((_eff_norm or {}).get("enabled", False)) and bool((_eff_norm or {}).get("auto", False)):
        raise RuntimeError("require_pre_stage2=true but pre-stage2 rigid alignment did not complete; refusing to start stage2 training.")

    if bind_rigid_align_for_train is not None:
        cfg["stage2"]["bind_rigid_align_effective"] = bind_rigid_align_for_train
    mesh_runtime.set_effective_rigid_align_config(bind_rigid_align_for_train)

    if bool(args.prepare_supervision_only):
        prepare_stage2_supervision_artifacts(
            project_root=project_root,
            stage1_dir=stage1_dir,
            out_dir=out_dir,
            stage2_cfg=stage2_cfg,
            frames_dir=frames_dir,
            frames=frames,
            K_list=cam["K"],
            bind_mesh_indices=bind_mesh_indices,
            bind_vertices_npz_path=bind_vertices_npz_path,
            mesh_faces_np=mesh_runtime.context.bind_faces_np,
            mesh_runtime=mesh_runtime,
            logger=logger,
        )
        save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")
        logger.info("prepare_supervision_only=true; skip stage2 training.")
        return

    prepared_supervision_cache = None
    prepared_supervision_cache_cfg = stage2_cfg.get("prepared_supervision_cache_path", None)
    prepared_supervision_cache_path = (
        _resolve_project_path(project_root, prepared_supervision_cache_cfg)
        if prepared_supervision_cache_cfg is not None
        else (out_dir / "supervision_cache.npz")
    )
    prepared_supervision_cache = load_stage2_supervision_cache(
        cache_path=prepared_supervision_cache_path,
        frames=frames,
        logger=logger,
    )

    ckpt_path = out_dir / "gaussians3d.pt"
    stage2_frame_assets = build_stage2_frame_assets(
        frames_dir=frames_dir,
        frames_list=frames,
        device=device,
        prepared_supervision=prepared_supervision_cache,
        frame_align_scores=frame_align_scores,
    )
    train_loop_cfg = Stage2TrainLoopConfig.from_stage2_setup(
        train_config=train_cfg,
        frames=frames,
        cam=cam,
        out_ckpt=ckpt_path,
        runtime={
            "lr_cam": lr_cam,
            "init_cam_w": init_cam_w,
            "init_cam_t": init_cam_t,
            "device": device,
            "camera_mode": camera_mode,
            "cam_t_prior": cam_t_prior,
            "cam_reg_weight": cam_reg_weight,
            "frame_weights": frame_weights,
            "frame_loss_weights": frame_loss_weights,
            "frame_align_scores": frame_align_scores,
            "frame_Ks": frame_Ks,
            "bind_vertex_indices": g.bind_idx,
            "prepared_supervision": prepared_supervision_cache,
            "stage2_frame_assets": stage2_frame_assets,
            "stage2_mesh_runtime": mesh_runtime,
            "training_mode": training_mode,
            "save_inner_checkpoint_path": str(inner_ckpt_path),
            "logger": logger,
        },
    )
    train_loop_cfg.options.update(
        {
            "face_align_quality_enable": bool(
                stage2_cfg.get("face_align_quality_enable", False)
            ),
            "face_align_weight_low": float(
                stage2_cfg.get("face_align_weight_low", 0.25)
            ),
            "face_align_weight_high": float(
                stage2_cfg.get("face_align_weight_high", 3.0)
            ),
            "face_align_score_min": float(
                stage2_cfg.get("face_align_score_min", 0.86)
            ),
            "face_align_score_max": float(
                stage2_cfg.get("face_align_score_max", 0.92)
            ),
            "face_align_score_power": float(
                stage2_cfg.get("face_align_score_power", 2.0)
            ),
            "face_frontal_quality_enable": bool(
                stage2_cfg.get("face_frontal_quality_enable", False)
            ),
            "face_frontal_weight_low": float(
                stage2_cfg.get("face_frontal_weight_low", 0.85)
            ),
            "face_frontal_weight_high": float(
                stage2_cfg.get("face_frontal_weight_high", 1.5)
            ),
            "face_frontal_power": float(
                stage2_cfg.get("face_frontal_power", 1.5)
            ),
            "inner_legacy_e526_compat_enable": bool(
                stage2_cfg.get("inner_legacy_e526_compat_enable", False)
            ),
            "inner_local_scale_clamp_enable": bool(
                stage2_cfg.get("inner_local_scale_clamp_enable", False)
            ),
            "inner_local_scale_clamp_ratio": float(
                stage2_cfg.get("inner_local_scale_clamp_ratio", 1.2)
            ),
            "inner_local_scale_clamp_normal_ratio": float(
                stage2_cfg.get("inner_local_scale_clamp_normal_ratio", 0.7)
            ),
            "face_hard_frame_select_enable": bool(
                stage2_cfg.get("face_hard_frame_select_enable", False)
            ),
            "face_hard_frame_keep_topk": int(
                stage2_cfg.get("face_hard_frame_keep_topk", 0)
            ),
            "face_hard_align_min": float(
                stage2_cfg.get("face_hard_align_min", 0.90)
            ),
            "face_hard_frontal_min": float(
                stage2_cfg.get("face_hard_frontal_min", 0.75)
            ),
            "face_hard_reject_weight": float(
                stage2_cfg.get("face_hard_reject_weight", 0.0)
            ),
            "face_hard_exclusive_enable": bool(
                stage2_cfg.get("face_hard_exclusive_enable", False)
            ),
            "face_reference_frame_mode": str(
                stage2_cfg.get("face_reference_frame_mode", "last")
            ),
            "face_reference_frame_name": str(
                stage2_cfg.get("face_reference_frame_name", "")
            ),
            "face_reference_frame_index": int(
                stage2_cfg.get("face_reference_frame_index", -1)
            ),
            "face_reference_weight": float(
                stage2_cfg.get("face_reference_weight", 1.0)
            ),
            "inner_local_color_prior_target_vis_max": float(
                stage2_cfg.get("inner_local_color_prior_target_vis_max", 0.08)
            ),
            "inner_local_color_prior_target_support_max": float(
                stage2_cfg.get("inner_local_color_prior_target_support_max", 0.15)
            ),
            "inner_local_color_repair_enable": bool(
                stage2_cfg.get("inner_local_color_repair_enable", False)
            ),
            "inner_local_color_repair_blend": float(
                stage2_cfg.get("inner_local_color_repair_blend", 0.20)
            ),
            "inner_local_color_repair_min_weight": float(
                stage2_cfg.get("inner_local_color_repair_min_weight", 0.35)
            ),
            "inner_coverage_densify_enable": bool(
                stage2_cfg.get("inner_coverage_densify_enable", True)
            ),
            "inner_coverage_eval_every": int(
                stage2_cfg.get("inner_coverage_eval_every", 200)
            ),
            "inner_coverage_spawn_topk": int(
                stage2_cfg.get("inner_coverage_spawn_topk", 256)
            ),
            "inner_coverage_stop_ratio": float(
                stage2_cfg.get("inner_coverage_stop_ratio", 0.82)
            ),
            "inner_coverage_spawn_scale_mult": float(
                stage2_cfg.get("inner_coverage_spawn_scale_mult", 0.82)
            ),
            "inner_coverage_spawn_opacity_logit_init": float(
                stage2_cfg.get("inner_coverage_spawn_opacity_logit_init", -2.6)
            ),
            "inner_coverage_spawn_opacity_logit_cap": float(
                stage2_cfg.get("inner_coverage_spawn_opacity_logit_cap", -2.0)
            ),
            "inner_count_max": int(stage2_cfg.get("inner_count_max", 260000)),
        }
    )
    train_gaussians_3d_learnable_camera(g=g, train_cfg=train_loop_cfg)

    teacher_bank_json = _run_part_teacher_bank(
        project_root=project_root,
        stage1_dir=stage1_dir,
        out_dir=out_dir,
        stage2_cfg=stage2_cfg,
        logger=logger,
    )

    _maybe_run_face_postfit(
        project_root=project_root,
        stage1_dir=stage1_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        stage2_cfg=stage2_cfg,
        teacher_bank_json=teacher_bank_json,
        logger=logger,
    )

    _maybe_run_part_teacher_postfit(
        project_root=project_root,
        stage1_dir=stage1_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        stage2_cfg=stage2_cfg,
        teacher_bank_json=teacher_bank_json,
        logger=logger,
    )

    logger.info(f"Saved checkpoint: {ckpt_path}")
    if training_mode in ("full", "inner_only"):
        logger.info(f"Inner checkpoint path: {inner_ckpt_path}")
    if init_ckpt_path is not None:
        logger.info(f"Init checkpoint: {init_ckpt_path}")
    save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")


if __name__ == "__main__":
    main()



