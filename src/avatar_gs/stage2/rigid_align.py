from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from avatar_gs.gaussian.rigid_align import apply_rigid_transform_np, coordinate_descent_rigid_search
from .supervision import rasterize_mesh_mask


def _as_float3(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return default
    return default


def normalize_rigid_align_config(cfg, default_enabled: bool = False) -> dict | None:
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        return {
            "enabled": bool(cfg),
            "auto": False,
            "pre_stage2": True,
            "require_pre_stage2": True,
            "require_quality": True,
            "min_iou_after": 0.70,
            "max_outside_after": 0.20,
            "init_from_bbox": True,
            "use_best_if_quality_fail": True,
            "rx_deg": 0.0,
            "ry_deg": 0.0,
            "rz_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "sample_frames": 8,
            "eval_downsample": 0.5,
            "outside_weight": 2.0,
            "levels": 3,
            "iters_per_level": 2,
            "rot_step_deg": 3.0,
            "trans_step": (0.04, 0.04, 0.06),
            "max_rot_deg": 20.0,
            "max_trans": (0.35, 0.35, 0.45),
            "person_masks_npz": None,
        }
    if not isinstance(cfg, dict):
        return None
    base = dict(cfg)
    nested_params = base.get("params", None)
    if isinstance(nested_params, dict):
        merged = dict(nested_params)
        for k, v in base.items():
            if k != "params":
                merged[k] = v
        base = merged
    rot_default = (
        float(base.get("rx_deg", 0.0)),
        float(base.get("ry_deg", 0.0)),
        float(base.get("rz_deg", 0.0)),
    )
    trans_default = (
        float(base.get("tx", 0.0)),
        float(base.get("ty", 0.0)),
        float(base.get("tz", 0.0)),
    )
    rx, ry, rz = _as_float3(base.get("rot_deg", None), rot_default)
    tx, ty, tz = _as_float3(base.get("trans", None), trans_default)
    return {
        "enabled": bool(base.get("enabled", base.get("enable", default_enabled))),
        "auto": bool(base.get("auto", base.get("auto_enable", False))),
        "pre_stage2": bool(base.get("pre_stage2", True)),
        "require_pre_stage2": bool(base.get("require_pre_stage2", True)),
        "require_quality": bool(base.get("require_quality", True)),
        "min_iou_after": float(base.get("min_iou_after", 0.70)),
        "max_outside_after": float(base.get("max_outside_after", 0.20)),
        "init_from_bbox": bool(base.get("init_from_bbox", True)),
        "use_best_if_quality_fail": bool(base.get("use_best_if_quality_fail", True)),
        "rx_deg": float(rx),
        "ry_deg": float(ry),
        "rz_deg": float(rz),
        "tx": float(tx),
        "ty": float(ty),
        "tz": float(tz),
        "sample_frames": max(1, int(base.get("sample_frames", 8))),
        "eval_downsample": float(np.clip(float(base.get("eval_downsample", 0.5)), 0.2, 1.0)),
        "outside_weight": float(max(0.0, float(base.get("outside_weight", 2.0)))),
        "levels": max(1, int(base.get("levels", 3))),
        "iters_per_level": max(1, int(base.get("iters_per_level", 2))),
        "rot_step_deg": float(max(0.05, float(base.get("rot_step_deg", 3.0)))),
        "trans_step": _as_float3(base.get("trans_step", None), (0.04, 0.04, 0.06)),
        "max_rot_deg": float(max(1.0, float(base.get("max_rot_deg", 20.0)))),
        "max_trans": _as_float3(base.get("max_trans", None), (0.35, 0.35, 0.45)),
        "person_masks_npz": None if base.get("person_masks_npz", None) is None else str(base.get("person_masks_npz")),
    }



@dataclass(slots=True)
class Stage2RigidAlignResult:
    effective_cfg: dict | None
    metrics: dict[str, float] | None
    report_path: Path | None = None


def _norm_frame_name(name: str) -> str:
    s = str(name).strip().replace("\\", "/")
    return Path(s).name.lower()


def _build_mask_lookup(frame_names: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    name_to_exact: dict[str, int] = {}
    name_to_stem: dict[str, int] = {}
    for i, name in enumerate(frame_names):
        nn = _norm_frame_name(name)
        name_to_exact[nn] = int(i)
        stem = Path(nn).stem
        if stem and stem not in name_to_stem:
            name_to_stem[stem] = int(i)
    return name_to_exact, name_to_stem


def _lookup_mask_idx(frame_name: str, name_to_exact: dict[str, int], name_to_stem: dict[str, int]) -> int:
    nn = _norm_frame_name(frame_name)
    if nn in name_to_exact:
        return int(name_to_exact[nn])
    return int(name_to_stem.get(Path(nn).stem, -1))


def run_stage2_rigid_align_search(
    *,
    cfg: dict,
    frames: list[str],
    bind_mesh_indices: list[int],
    K_np: np.ndarray,
    image_hw: tuple[int, int],
    bind_faces_np: np.ndarray,
    mesh_loader,
    person_masks_npz_path: Path,
    logger,
    R_eval: np.ndarray | None = None,
    t_eval: np.ndarray | None = None,
    bind_mesh_offset_np: np.ndarray | None = None,
    cache_report_path: Path | None = None,
    camera_resolver=None,
) -> Stage2RigidAlignResult:
    if not person_masks_npz_path.exists():
        logger.warning("bind_rigid_align skipped: person mask file missing.")
        return Stage2RigidAlignResult(effective_cfg=cfg, metrics=None, report_path=cache_report_path)
    pdata = np.load(person_masks_npz_path, allow_pickle=True)
    if "frame_names" not in pdata.files or "masks" not in pdata.files:
        logger.warning("bind_rigid_align skipped: person mask NPZ missing frame_names/masks.")
        return Stage2RigidAlignResult(effective_cfg=cfg, metrics=None, report_path=cache_report_path)
    p_names = [str(x) for x in pdata["frame_names"].tolist()]
    p_masks = np.asarray(pdata["masks"], dtype=np.uint8)
    if p_masks.ndim != 3 or len(p_names) != int(p_masks.shape[0]):
        logger.warning("bind_rigid_align skipped: invalid person mask tensor shape.")
        return Stage2RigidAlignResult(effective_cfg=cfg, metrics=None, report_path=cache_report_path)
    name_to_exact, name_to_stem = _build_mask_lookup(p_names)
    valid_frame_ids = [i for i, fn in enumerate(frames) if _lookup_mask_idx(fn, name_to_exact, name_to_stem) >= 0]
    if len(valid_frame_ids) == 0:
        logger.warning("bind_rigid_align skipped: no overlapping frame names with person masks.")
        return Stage2RigidAlignResult(effective_cfg=cfg, metrics=None, report_path=cache_report_path)
    sample_n = min(int(cfg.get("sample_frames", 8)), len(valid_frame_ids))
    pick = np.linspace(0, len(valid_frame_ids) - 1, num=sample_n, dtype=np.float64)
    pick = np.clip(np.round(pick).astype(np.int64), 0, len(valid_frame_ids) - 1)
    sample_ids = [valid_frame_ids[int(k)] for k in np.unique(pick).tolist()]
    h, w = int(image_hw[0]), int(image_hw[1])
    ds = float(cfg.get("eval_downsample", 0.5))
    hs = max(8, int(round(float(h) * ds)))
    ws = max(8, int(round(float(w) * ds)))
    sx = float(ws) / float(w)
    sy = float(hs) / float(h)
    k_eval = np.asarray(K_np, dtype=np.float32).copy()
    k_eval[0, 0] *= sx
    k_eval[0, 2] *= sx
    k_eval[1, 1] *= sy
    k_eval[1, 2] *= sy
    outside_w = float(cfg.get("outside_weight", 2.0))
    gt_masks_eval: dict[int, np.ndarray] = {}
    for fj in sample_ids:
        pm_i = int(_lookup_mask_idx(frames[fj], name_to_exact, name_to_stem))
        if pm_i < 0:
            continue
        gt_m = (p_masks[pm_i] > 0).astype(np.uint8)
        if hs != h or ws != w:
            gt_m = cv2.resize(gt_m, (ws, hs), interpolation=cv2.INTER_NEAREST)
        gt_masks_eval[fj] = gt_m > 0
    if len(gt_masks_eval) == 0:
        logger.warning("bind_rigid_align skipped: sampled masks are empty after frame-name matching.")
        return Stage2RigidAlignResult(effective_cfg=cfg, metrics=None, report_path=cache_report_path)
    r_eval = np.eye(3, dtype=np.float32) if R_eval is None else np.asarray(R_eval, dtype=np.float32)
    t_eval_np = np.zeros((3,), dtype=np.float32) if t_eval is None else np.asarray(t_eval, dtype=np.float32).reshape(3,)
    mesh_offset = np.zeros((1, 3), dtype=np.float32) if bind_mesh_offset_np is None else np.asarray(bind_mesh_offset_np, dtype=np.float32).reshape(1, 3)

    def _resolve_camera(frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if camera_resolver is None:
            return k_eval, r_eval, t_eval_np
        K_frame, R_frame, t_frame = camera_resolver(frame_idx)
        K_frame_eval = np.asarray(K_frame, dtype=np.float32).copy()
        K_frame_eval[0, 0] *= sx
        K_frame_eval[0, 2] *= sx
        K_frame_eval[1, 1] *= sy
        K_frame_eval[1, 2] *= sy
        return (
            K_frame_eval,
            np.asarray(R_frame, dtype=np.float32),
            np.asarray(t_frame, dtype=np.float32).reshape(3,),
        )

    def _eval_score(param_dict: dict) -> float:
        acc = 0.0
        used = 0
        for fj in sample_ids:
            verts_raw = np.asarray(mesh_loader(int(bind_mesh_indices[fj])), dtype=np.float32)
            verts_eval = apply_rigid_transform_np(verts_raw, {"enabled": True, **param_dict}) + mesh_offset
            K_eval_frame, R_eval_frame, t_eval_frame = _resolve_camera(fj)
            pred = rasterize_mesh_mask(verts=verts_eval, faces=bind_faces_np, K=K_eval_frame, R=R_eval_frame, t=t_eval_frame, H=hs, W=ws) > 0
            gt_m = gt_masks_eval[fj]
            inter = float(np.logical_and(pred, gt_m).sum())
            union = float(np.logical_or(pred, gt_m).sum())
            iou = inter / (union + 1e-6)
            pred_area = float(pred.sum())
            outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
            acc += iou - outside_w * outside
            used += 1
        return -1e9 if used <= 0 else float(acc / float(used))

    def _eval_metrics(param_dict: dict) -> tuple[float, float, float]:
        iou_acc = 0.0
        outside_acc = 0.0
        score_acc = 0.0
        used = 0
        for fj in sample_ids:
            verts_raw = np.asarray(mesh_loader(int(bind_mesh_indices[fj])), dtype=np.float32)
            verts_eval = apply_rigid_transform_np(verts_raw, {"enabled": True, **param_dict}) + mesh_offset
            K_eval_frame, R_eval_frame, t_eval_frame = _resolve_camera(fj)
            pred = rasterize_mesh_mask(verts=verts_eval, faces=bind_faces_np, K=K_eval_frame, R=R_eval_frame, t=t_eval_frame, H=hs, W=ws) > 0
            gt_m = gt_masks_eval[fj]
            inter = float(np.logical_and(pred, gt_m).sum())
            union = float(np.logical_or(pred, gt_m).sum())
            iou = inter / (union + 1e-6)
            pred_area = float(pred.sum())
            outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
            score = iou - outside_w * outside
            iou_acc += iou
            outside_acc += outside
            score_acc += score
            used += 1
        if used <= 0:
            return 0.0, 1.0, -1e9
        n = float(used)
        return float(iou_acc / n), float(outside_acc / n), float(score_acc / n)

    init_params = {
        "rx_deg": float(cfg.get("rx_deg", 0.0)),
        "ry_deg": float(cfg.get("ry_deg", 0.0)),
        "rz_deg": float(cfg.get("rz_deg", 0.0)),
        "tx": float(cfg.get("tx", 0.0)),
        "ty": float(cfg.get("ty", 0.0)),
        "tz": float(cfg.get("tz", 0.0)),
    }
    best_params, score_before, score_after = coordinate_descent_rigid_search(
        score_fn=_eval_score,
        init_params=init_params,
        rot_step_deg=float(cfg.get("rot_step_deg", 3.0)),
        trans_step=tuple(cfg.get("trans_step", (0.04, 0.04, 0.06))),
        levels=int(cfg.get("levels", 3)),
        iters_per_level=int(cfg.get("iters_per_level", 2)),
        max_rot_deg=float(cfg.get("max_rot_deg", 20.0)),
        max_trans=tuple(cfg.get("max_trans", (0.35, 0.35, 0.45))),
    )
    iou_before, outside_before, _ = _eval_metrics(init_params)
    iou_after, outside_after, _ = _eval_metrics(best_params)
    effective_cfg = {"enabled": True, **cfg, **best_params}
    metrics = {
        "score_before": float(score_before),
        "score_after": float(score_after),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "outside_before": float(outside_before),
        "outside_after": float(outside_after),
        "sample_frames": float(len(sample_ids)),
    }
    logger.info(
        "bind_rigid_align auto: "
        f"score {score_before:.4f}->{score_after:.4f}, "
        f"iou {iou_before:.4f}->{iou_after:.4f}, "
        f"outside {outside_before:.4f}->{outside_after:.4f}"
    )
    return Stage2RigidAlignResult(effective_cfg=effective_cfg, metrics=metrics, report_path=cache_report_path)
