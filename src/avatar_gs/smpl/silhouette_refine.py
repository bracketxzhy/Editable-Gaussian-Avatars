from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None

from avatar_gs.alignment.mask_sources import estimate_person_mask_modnet_eval_style_np


@dataclass
class SilhouetteRefineConfig:
    enabled: bool = True
    mask_source: str = "modnet"
    refit_iters: int = 55
    lr: float = 0.01
    y_bin_count: int = 96
    x_bin_count: int = 48
    inside_margin_px: float = 1.5
    outside_loss_weight: float = 9.0
    inside_loss_weight: float = 0.08
    outside_peak_loss_weight: float = 0.0
    inside_peak_loss_weight: float = 0.0
    peak_fraction: float = 0.12
    span_loss_weight: float = 0.0
    gap_loss_weight: float = 0.0
    span_peak_loss_weight: float = 0.0
    gap_peak_loss_weight: float = 0.0
    bbox_loss_weight: float = 0.0
    joint2d_loss_weight: float = 0.0
    arm_local_loss_weight: float = 0.0
    arm_local_peak_loss_weight: float = 0.0
    arm_joint2d_loss_weight: float = 0.0
    arm_box_pad_scale: float = 0.45
    arm_box_min_pad_px: float = 22.0
    pose_reg_weight: float = 0.015
    orient_reg_weight: float = 0.02
    transl_reg_weight: float = 0.08
    optimize_betas: bool = False
    betas_reg_weight: float = 0.05
    modnet_ref_size: int = 512
    modnet_threshold: float = 0.5
    modnet_ckpt_path: str = ""
    accept_outside_weight: float = 4.5
    accept_oversize_weight: float = 0.45
    accept_undersize_weight: float = 0.03
    accept_boundary_mean_weight: float = 0.0
    accept_boundary_p95_weight: float = 0.0
    enable_multi_candidate: bool = True
    enable_temporal_candidate: bool = True
    temporal_max_frame_gap: int = 2
    temporal_pose_reg_weight: float = 0.0
    temporal_orient_reg_weight: float = 0.0
    temporal_transl_reg_weight: float = 0.0
    temporal_betas_reg_weight: float = 0.0
    hard_frame_iou_threshold: float = 0.84
    hard_frame_outside_threshold: float = 0.06
    hard_frame_area_ratio_threshold: float = 0.93
    hard_frame_max_extra_candidates: int = 2
    hard_frame_enable_beta_nudge: bool = False
    arm_accept_global_outside_slack: float = 0.010
    arm_accept_local_outside_slack: float = 0.015
    arm_accept_boundary_gain_px: float = 1.25
    arm_accept_iou_gain: float = 0.015
    arm_accept_score_slack: float = 0.03
    device: str = "auto"


@dataclass
class SilhouetteRefineResult:
    records: list[dict[str, Any]]
    meta: dict[str, Any]


def _resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    raw = str(device).strip().lower()
    if raw in ("", "auto"):
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if raw == "cuda" and (not bool(torch.cuda.is_available())):
        return "cpu"
    return raw


def _create_model(assets_dir: Path, gender: str, device: str):
    if smplx is None or torch is None:
        raise RuntimeError("smplx and torch are required for silhouette refinement.")
    gender_use = str(gender).strip().lower()
    if gender_use not in {"neutral", "male", "female"}:
        gender_use = "neutral"
    return smplx.create(
        model_path=str(assets_dir),
        model_type="smplx",
        gender=gender_use,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to(device=device)


def _reshape_like(flat: np.ndarray, ref: np.ndarray | None, fallback_dim: int) -> np.ndarray:
    x = np.asarray(flat, dtype=np.float32).reshape(-1)
    if ref is not None:
        ref_arr = np.asarray(ref)
        if ref_arr.size == x.size:
            return x.reshape(ref_arr.shape).astype(np.float32)
    return x.reshape(1, fallback_dim).astype(np.float32)


def _read_param(params: dict[str, Any], key: str, fallback_dim: int) -> np.ndarray:
    arr = np.asarray(params.get(key, np.zeros((1, fallback_dim), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if arr.size < fallback_dim:
        arr = np.pad(arr, (0, fallback_dim - arr.size))
    return arr[:fallback_dim].astype(np.float32)


def _sanitize_joints2d_np(value: object) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return np.asarray(arr[:, :3], dtype=np.float32)
    flat = arr.reshape(-1)
    usable = (int(flat.size) // 3) * 3
    if usable <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(flat[:usable], dtype=np.float32).reshape(-1, 3)


def _extract_temporal_seed(rec_in: dict[str, Any], *, body_dim: int, num_betas: int) -> dict[str, np.ndarray]:
    params_seed = dict(rec_in.get("smplestx_params", {}))
    transl_seed = np.asarray(rec_in.get("cam_t", params_seed.get("transl", np.zeros((1, 3), dtype=np.float32))), dtype=np.float32).reshape(-1)
    if transl_seed.size < 3:
        transl_seed = np.pad(transl_seed, (0, 3 - transl_seed.size))
    betas_seed = np.asarray(params_seed.get("betas", np.zeros((1, num_betas), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if betas_seed.size < num_betas:
        betas_seed = np.pad(betas_seed, (0, num_betas - betas_seed.size))
    return {
        "global_orient": _read_param(params_seed, "global_orient", 3),
        "body_pose": _read_param(params_seed, "body_pose", body_dim),
        "transl": transl_seed[:3].astype(np.float32),
        "betas": betas_seed[:num_betas].astype(np.float32),
    }


def _build_temporal_inits(
    records: list[dict[str, Any]],
    rec_idx: int,
    *,
    body_dim: int,
    num_betas: int,
    max_frame_gap: int,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    cur_frame_idx = int(records[rec_idx].get("frame_idx", rec_idx))
    prev_seed = None
    next_seed = None
    if rec_idx > 0:
        prev_frame_idx = int(records[rec_idx - 1].get("frame_idx", rec_idx - 1))
        if abs(cur_frame_idx - prev_frame_idx) <= int(max_frame_gap):
            prev_seed = _extract_temporal_seed(records[rec_idx - 1], body_dim=body_dim, num_betas=num_betas)
    if rec_idx + 1 < len(records):
        next_frame_idx = int(records[rec_idx + 1].get("frame_idx", rec_idx + 1))
        if abs(next_frame_idx - cur_frame_idx) <= int(max_frame_gap):
            next_seed = _extract_temporal_seed(records[rec_idx + 1], body_dim=body_dim, num_betas=num_betas)
    out: list[tuple[str, dict[str, np.ndarray]]] = []
    if prev_seed is not None:
        out.append(("temporal_prev", prev_seed))
    if next_seed is not None:
        out.append(("temporal_next", next_seed))
    if prev_seed is not None and next_seed is not None:
        blend = {
            key: ((np.asarray(prev_seed[key], dtype=np.float32) + np.asarray(next_seed[key], dtype=np.float32)) * 0.5).astype(np.float32)
            for key in ("global_orient", "body_pose", "transl", "betas")
        }
        out.append(("temporal_blend", blend))
    return out


def _project_points_torch(
    pts: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = pts[:, 2]
    valid = torch.isfinite(pts).all(dim=1) & (z.abs() > float(eps))
    z_safe = torch.where(valid, z, torch.ones_like(z))
    u = (pts[:, 0] / z_safe) * float(fx) + float(cx)
    v = (pts[:, 1] / z_safe) * float(fy) + float(cy)
    return torch.stack((u, v), dim=1), valid


def _project_points_np(
    pts: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    z = arr[:, 2]
    valid = np.isfinite(arr).all(axis=1) & (np.abs(z) > float(eps))
    z_safe = np.where(valid, z, 1.0).astype(np.float32)
    u = (arr[:, 0] / z_safe) * float(fx) + float(cx)
    v = (arr[:, 1] / z_safe) * float(fy) + float(cy)
    return np.stack((u, v), axis=1).astype(np.float32), valid


def _bbox_from_mask_np(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    mm = np.asarray(mask, dtype=np.uint8)
    if mm.ndim < 2:
        return None
    ys, xs = np.nonzero(mm > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return np.asarray([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _rasterize_mask_np(uv: np.ndarray, valid_v: np.ndarray, faces: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in np.asarray(faces, dtype=np.int32):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a >= len(valid_v) or b >= len(valid_v) or c >= len(valid_v):
            continue
        if (not bool(valid_v[a])) or (not bool(valid_v[b])) or (not bool(valid_v[c])):
            continue
        pts = np.round(uv[[a, b, c], :]).astype(np.int32)
        if int(pts[:, 0].max()) < 0 or int(pts[:, 1].max()) < 0 or int(pts[:, 0].min()) >= w or int(pts[:, 1].min()) >= h:
            continue
        cv2.fillConvexPoly(mask, pts, color=255, lineType=cv2.LINE_8)
    return mask


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=np.uint8) > 0
    b = np.asarray(mask_b, dtype=np.uint8) > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def _mask_outside_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=np.uint8) > 0
    b = np.asarray(mask_b, dtype=np.uint8) > 0
    area = float(max(1, a.sum()))
    outside = float(np.logical_and(a, np.logical_not(b)).sum())
    return outside / area


def _mask_area_ratio(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    area_pred = float(max(1, (np.asarray(mask_pred, dtype=np.uint8) > 0).sum()))
    area_gt = float(max(1, (np.asarray(mask_gt, dtype=np.uint8) > 0).sum()))
    return area_pred / area_gt


def _boundary_mask_np(mask_u8: np.ndarray) -> np.ndarray:
    m = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if m.ndim != 2 or m.size <= 0:
        return np.zeros_like(m, dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(out, contours, -1, color=255, thickness=1, lineType=cv2.LINE_8)
    return out


def _score_mask_fit(
    *,
    iou: float,
    outside: float,
    area_ratio: float,
    outside_weight: float,
    oversize_weight: float,
    undersize_weight: float,
    boundary_mean_px: float = 0.0,
    boundary_p95_px: float = 0.0,
    boundary_mean_weight: float = 0.0,
    boundary_p95_weight: float = 0.0,
) -> float:
    oversize = max(0.0, float(area_ratio) - 1.0)
    undersize = max(0.0, 1.0 - float(area_ratio))
    boundary_mean_penalty = float(boundary_mean_weight) * float(boundary_mean_px) if np.isfinite(float(boundary_mean_px)) else 0.0
    boundary_p95_penalty = float(boundary_p95_weight) * float(boundary_p95_px) if np.isfinite(float(boundary_p95_px)) else 0.0
    return float(iou - outside_weight * outside - oversize_weight * oversize - undersize_weight * undersize - boundary_mean_penalty - boundary_p95_penalty)


def _boundary_distance_stats_np(mask_src: np.ndarray, mask_tgt: np.ndarray) -> tuple[float, float]:
    src = (_boundary_mask_np(mask_src) > 0)
    tgt = (_boundary_mask_np(mask_tgt) > 0)
    if int(src.sum()) <= 0:
        return float("inf"), float("inf")
    inv_tgt = np.logical_not(tgt).astype(np.uint8)
    dist = cv2.distanceTransform(inv_tgt, cv2.DIST_L2, 3)
    vals = np.asarray(dist[src], dtype=np.float32).reshape(-1)
    if vals.size <= 0:
        return float("inf"), float("inf")
    return float(vals.mean()), float(np.percentile(vals, 95))


def _union_box_from_boxes_np(boxes: np.ndarray, *, image_hw: tuple[int, int]) -> np.ndarray | None:
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if arr.shape[0] <= 0:
        return None
    h, w = int(image_hw[0]), int(image_hw[1])
    x0 = max(0.0, float(arr[:, 0].min()))
    y0 = max(0.0, float(arr[:, 1].min()))
    x1 = min(float(max(0, w - 1)), float(arr[:, 2].max()))
    y1 = min(float(max(0, h - 1)), float(arr[:, 3].max()))
    if (x1 - x0) < 4.0 or (y1 - y0) < 4.0:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _crop_mask_to_box_np(mask_u8: np.ndarray, box_xyxy: np.ndarray | None) -> np.ndarray:
    mm = np.asarray(mask_u8, dtype=np.uint8)
    if mm.ndim != 2 or box_xyxy is None:
        return np.zeros((0, 0), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in np.round(np.asarray(box_xyxy, dtype=np.float32).reshape(4)).tolist()]
    x0 = max(0, min(int(mm.shape[1]), x0))
    y0 = max(0, min(int(mm.shape[0]), y0))
    x1 = max(x0 + 1, min(int(mm.shape[1]), x1 + 1))
    y1 = max(y0 + 1, min(int(mm.shape[0]), y1 + 1))
    return np.asarray(mm[y0:y1, x0:x1], dtype=np.uint8)


def _arm_region_stats_np(mask_pred: np.ndarray, mask_gt: np.ndarray, arm_boxes: np.ndarray, *, image_hw: tuple[int, int]) -> dict[str, float]:
    union_box = _union_box_from_boxes_np(arm_boxes, image_hw=image_hw)
    if union_box is None:
        return {
            "iou": 0.0,
            "outside": 1.0,
            "area_ratio": 0.0,
            "boundary_mean_px": float("inf"),
            "boundary_p95_px": float("inf"),
        }
    pred_crop = _crop_mask_to_box_np(mask_pred, union_box)
    gt_crop = _crop_mask_to_box_np(mask_gt, union_box)
    if pred_crop.size <= 0 or gt_crop.size <= 0:
        return {
            "iou": 0.0,
            "outside": 1.0,
            "area_ratio": 0.0,
            "boundary_mean_px": float("inf"),
            "boundary_p95_px": float("inf"),
        }
    boundary_mean_px, boundary_p95_px = _boundary_distance_stats_np(pred_crop, gt_crop)
    return {
        "iou": float(_mask_iou(pred_crop, gt_crop)),
        "outside": float(_mask_outside_ratio(pred_crop, gt_crop)),
        "area_ratio": float(_mask_area_ratio(pred_crop, gt_crop)),
        "boundary_mean_px": float(boundary_mean_px),
        "boundary_p95_px": float(boundary_p95_px),
    }


def _tail_mean_torch(values: torch.Tensor, frac: float) -> torch.Tensor:
    vv = values.reshape(-1)
    if int(vv.numel()) <= 0:
        return torch.tensor(0.0, device=values.device)
    keep = max(1, int(np.ceil(float(vv.numel()) * max(0.0, float(frac)))))
    tail_vals, _ = torch.topk(vv, k=min(int(keep), int(vv.numel())), largest=True, sorted=False)
    return tail_vals.mean()


def _extract_mask_span_targets(mask_u8: np.ndarray, *, y_bin_count: int) -> dict[str, np.ndarray]:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0)
    h, w = m.shape[:2]
    if h <= 0 or w <= 0:
        zeros = np.zeros((0,), dtype=np.float32)
        return {
            "y_edges": zeros,
            "x_mid": np.float32(0.0),
            "outer_valid": np.zeros((0,), dtype=np.uint8),
            "outer_left": zeros,
            "outer_right": zeros,
            "gap_valid": np.zeros((0,), dtype=np.uint8),
            "gap_left": zeros,
            "gap_right": zeros,
        }
    ys, xs = np.nonzero(m)
    if xs.size <= 0 or ys.size <= 0:
        zeros = np.zeros((int(max(1, y_bin_count)),), dtype=np.float32)
        return {
            "y_edges": np.linspace(0.0, float(h), num=max(2, int(y_bin_count) + 1), dtype=np.float32),
            "x_mid": np.float32(0.5 * float(w)),
            "outer_valid": np.zeros_like(zeros, dtype=np.uint8),
            "outer_left": zeros,
            "outer_right": zeros,
            "gap_valid": np.zeros_like(zeros, dtype=np.uint8),
            "gap_left": zeros,
            "gap_right": zeros,
        }
    xmin = float(xs.min())
    xmax = float(xs.max())
    x_mid = 0.5 * (xmin + xmax)
    y_edges = np.linspace(float(ys.min()), float(ys.max()) + 1.0e-4, num=max(2, int(y_bin_count) + 1), dtype=np.float32)
    outer_valid = np.zeros((int(max(1, y_bin_count)),), dtype=np.uint8)
    outer_left = np.zeros_like(outer_valid, dtype=np.float32)
    outer_right = np.zeros_like(outer_valid, dtype=np.float32)
    gap_valid = np.zeros_like(outer_valid, dtype=np.uint8)
    gap_left = np.zeros_like(outer_valid, dtype=np.float32)
    gap_right = np.zeros_like(outer_valid, dtype=np.float32)
    for i_bin in range(int(max(1, y_bin_count))):
        y0 = int(np.floor(y_edges[i_bin]))
        y1 = int(np.ceil(y_edges[i_bin + 1]))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))
        band = m[y0:y1, :]
        col_any = np.any(band, axis=0)
        xs_band = np.flatnonzero(col_any)
        if xs_band.size <= 0:
            continue
        outer_valid[i_bin] = np.uint8(1)
        outer_left[i_bin] = float(xs_band.min())
        outer_right[i_bin] = float(xs_band.max())
        left_cluster = xs_band[xs_band <= x_mid]
        right_cluster = xs_band[xs_band >= x_mid]
        if left_cluster.size <= 0 or right_cluster.size <= 0:
            continue
        inner_left = int(left_cluster.max())
        inner_right = int(right_cluster.min())
        if inner_right - inner_left > 2:
            gap_valid[i_bin] = np.uint8(1)
            gap_left[i_bin] = float(inner_left)
            gap_right[i_bin] = float(inner_right)
    return {
        "y_edges": y_edges.astype(np.float32),
        "x_mid": np.float32(x_mid),
        "outer_valid": outer_valid,
        "outer_left": outer_left,
        "outer_right": outer_right,
        "gap_valid": gap_valid,
        "gap_left": gap_left,
        "gap_right": gap_right,
    }


def _span_edge_loss_torch(
    *,
    uv: torch.Tensor,
    valid_v: torch.Tensor,
    span_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if uv.ndim != 2 or uv.shape[1] != 2:
        z = torch.tensor(0.0, dtype=torch.float32, device=uv.device)
        return z, z
    valid = valid_v & torch.isfinite(uv).all(dim=1)
    if not bool(valid.any().item()):
        z = torch.tensor(0.0, dtype=torch.float32, device=uv.device)
        return z, z
    x = uv[:, 0]
    y = uv[:, 1]
    y_edges = span_targets["y_edges"]
    x_mid = span_targets["x_mid"]
    outer_valid = span_targets["outer_valid"] > 0.5
    outer_left_t = span_targets["outer_left"]
    outer_right_t = span_targets["outer_right"]
    gap_valid = span_targets["gap_valid"] > 0.5
    gap_left_t = span_targets["gap_left"]
    gap_right_t = span_targets["gap_right"]
    big = torch.full_like(x, 1.0e9)
    small = torch.full_like(x, -1.0e9)
    outer_terms: list[torch.Tensor] = []
    gap_terms: list[torch.Tensor] = []
    num_bins = max(0, int(y_edges.shape[0]) - 1)
    for i_bin in range(num_bins):
        in_bin = valid & (y >= y_edges[i_bin]) & (y < y_edges[i_bin + 1])
        if not bool(in_bin.any().item()):
            continue
        if bool(outer_valid[i_bin].item()):
            idx_l = torch.argmin(torch.where(in_bin, x, big))
            idx_r = torch.argmax(torch.where(in_bin, x, small))
            outer_terms.append(F.smooth_l1_loss(x[idx_l], outer_left_t[i_bin], reduction="mean"))
            outer_terms.append(F.smooth_l1_loss(x[idx_r], outer_right_t[i_bin], reduction="mean"))
        if bool(gap_valid[i_bin].item()):
            in_left = in_bin & (x <= x_mid)
            in_right = in_bin & (x >= x_mid)
            if bool(in_left.any().item()) and bool(in_right.any().item()):
                idx_inner_l = torch.argmax(torch.where(in_left, x, small))
                idx_inner_r = torch.argmin(torch.where(in_right, x, big))
                gap_terms.append(F.smooth_l1_loss(x[idx_inner_l], gap_left_t[i_bin], reduction="mean"))
                gap_terms.append(F.smooth_l1_loss(x[idx_inner_r], gap_right_t[i_bin], reduction="mean"))
    z = torch.tensor(0.0, dtype=torch.float32, device=uv.device)
    outer_loss = torch.stack(outer_terms).mean() if len(outer_terms) > 0 else z
    gap_loss = torch.stack(gap_terms).mean() if len(gap_terms) > 0 else z
    return outer_loss, gap_loss


def _build_arm_boxes_np(
    joints2d: np.ndarray,
    *,
    image_hw: tuple[int, int],
    conf_min: float,
    pad_scale: float,
    min_pad_px: float,
) -> np.ndarray:
    arr = _sanitize_joints2d_np(joints2d)
    h, w = int(image_hw[0]), int(image_hw[1])
    if arr.shape[0] <= 10:
        return np.zeros((0, 4), dtype=np.float32)
    boxes: list[np.ndarray] = []
    for s_idx, e_idx, w_idx in ((5, 7, 9), (6, 8, 10)):
        pts: list[np.ndarray] = []
        for idx in (s_idx, e_idx, w_idx):
            if idx >= arr.shape[0]:
                continue
            x, y, c = float(arr[idx, 0]), float(arr[idx, 1]), float(arr[idx, 2])
            if c < float(conf_min) or (not np.isfinite(x)) or (not np.isfinite(y)):
                continue
            pts.append(np.asarray([x, y], dtype=np.float32))
        if len(pts) < 2:
            continue
        pts_arr = np.asarray(pts, dtype=np.float32)
        seg_len = 0.0
        for i_pt in range(1, pts_arr.shape[0]):
            seg_len += float(np.linalg.norm(pts_arr[i_pt] - pts_arr[i_pt - 1]))
        pad = max(float(min_pad_px), float(pad_scale) * max(seg_len, 1.0))
        x0 = max(0.0, float(pts_arr[:, 0].min()) - pad)
        y0 = max(0.0, float(pts_arr[:, 1].min()) - pad)
        x1 = min(float(max(0, w - 1)), float(pts_arr[:, 0].max()) + pad)
        y1 = min(float(max(0, h - 1)), float(pts_arr[:, 1].max()) + pad)
        if (x1 - x0) >= 4.0 and (y1 - y0) >= 4.0:
            boxes.append(np.asarray([x0, y0, x1, y1], dtype=np.float32))
    if len(boxes) <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def _select_points_in_boxes_torch(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 2 or boxes.ndim != 2 or boxes.shape[1] != 4 or int(points.shape[0]) <= 0 or int(boxes.shape[0]) <= 0:
        return torch.zeros((int(points.shape[0]),), dtype=torch.bool, device=points.device)
    px = points[:, 0:1]
    py = points[:, 1:2]
    inside = (
        (px >= boxes[:, 0].view(1, -1))
        & (px <= boxes[:, 2].view(1, -1))
        & (py >= boxes[:, 1].view(1, -1))
        & (py <= boxes[:, 3].view(1, -1))
    )
    return torch.any(inside, dim=1)


def _select_candidate_configs(
    cfg: SilhouetteRefineConfig,
    *,
    iou_before: float,
    outside_before: float,
    area_ratio_before: float,
) -> list[tuple[str, SilhouetteRefineConfig]]:
    candidates: list[tuple[str, SilhouetteRefineConfig]] = [("primary", cfg)]
    if not bool(cfg.enable_multi_candidate):
        return candidates
    hard_frame = bool(
        (float(iou_before) < float(cfg.hard_frame_iou_threshold))
        or (float(outside_before) > float(cfg.hard_frame_outside_threshold))
        or (float(area_ratio_before) < float(cfg.hard_frame_area_ratio_threshold))
    )
    if not hard_frame:
        return candidates
    extra_candidates: list[tuple[str, SilhouetteRefineConfig]] = []
    severe_outside = float(outside_before) > max(float(cfg.hard_frame_outside_threshold) + 0.03, 0.09)
    severe_undersize = float(area_ratio_before) < min(float(cfg.hard_frame_area_ratio_threshold) - 0.03, 0.90)
    severe_iou = float(iou_before) < min(float(cfg.hard_frame_iou_threshold) - 0.06, 0.78)
    extra_candidates.append(
        (
            "arm_focus",
            replace(
                cfg,
                refit_iters=max(int(cfg.refit_iters), 88),
                lr=min(float(cfg.lr), 0.0045),
                y_bin_count=max(int(cfg.y_bin_count), 120),
                x_bin_count=max(int(cfg.x_bin_count), 60),
                outside_loss_weight=max(float(cfg.outside_loss_weight), 10.5),
                outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 5.5),
                inside_loss_weight=min(float(cfg.inside_loss_weight), 0.06),
                span_loss_weight=max(float(cfg.span_loss_weight), 0.024),
                gap_loss_weight=max(float(cfg.gap_loss_weight), 0.032),
                span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.012),
                gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.016),
                arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.060),
                arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.035),
                arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.045),
                arm_box_pad_scale=max(float(cfg.arm_box_pad_scale), 0.60),
                arm_box_min_pad_px=max(float(cfg.arm_box_min_pad_px), 28.0),
                bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.035),
                joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.018),
                pose_reg_weight=max(float(cfg.pose_reg_weight), 0.02),
                orient_reg_weight=max(float(cfg.orient_reg_weight), 0.025),
                transl_reg_weight=max(float(cfg.transl_reg_weight), 0.08),
                accept_boundary_mean_weight=max(float(cfg.accept_boundary_mean_weight), 0.001),
                accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.002),
            ),
        )
    )
    if severe_outside:
        extra_candidates.append(
            (
                "limb_clamp",
                replace(
                    cfg,
                    refit_iters=max(int(cfg.refit_iters), 108),
                    lr=min(float(cfg.lr), 0.0045),
                    y_bin_count=max(int(cfg.y_bin_count), 128),
                    x_bin_count=max(int(cfg.x_bin_count), 64),
                    outside_loss_weight=max(float(cfg.outside_loss_weight), 14.0),
                    outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 12.0),
                    inside_loss_weight=min(float(cfg.inside_loss_weight), 0.05),
                    inside_peak_loss_weight=max(float(cfg.inside_peak_loss_weight), 0.10),
                    span_loss_weight=max(float(cfg.span_loss_weight), 0.040),
                    gap_loss_weight=max(float(cfg.gap_loss_weight), 0.055),
                    span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.025),
                    gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.030),
                    arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.040),
                    arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.022),
                    arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.032),
                    bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.025),
                    joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.015),
                    peak_fraction=min(max(float(cfg.peak_fraction), 0.16), 0.22),
                    accept_boundary_mean_weight=max(float(cfg.accept_boundary_mean_weight), 0.0015),
                    accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.0040),
                ),
            )
        )
    if severe_undersize:
        extra_candidates.append(
            (
                "limb_recover",
                replace(
                    cfg,
                    refit_iters=max(int(cfg.refit_iters), 116),
                    lr=min(float(cfg.lr), 0.0045),
                    y_bin_count=max(int(cfg.y_bin_count), 132),
                    x_bin_count=max(int(cfg.x_bin_count), 64),
                    outside_loss_weight=max(float(cfg.outside_loss_weight), 8.5),
                    outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 4.5),
                    inside_loss_weight=max(float(cfg.inside_loss_weight), 0.16),
                    inside_peak_loss_weight=max(float(cfg.inside_peak_loss_weight), 0.18),
                    span_loss_weight=max(float(cfg.span_loss_weight), 0.060),
                    gap_loss_weight=max(float(cfg.gap_loss_weight), 0.045),
                    span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.035),
                    gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.020),
                    arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.055),
                    arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.030),
                    arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.040),
                    bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.030),
                    joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.015),
                    pose_reg_weight=min(float(cfg.pose_reg_weight), 0.012),
                    orient_reg_weight=max(float(cfg.orient_reg_weight), 0.025),
                    transl_reg_weight=max(float(cfg.transl_reg_weight), 0.08),
                    accept_boundary_mean_weight=max(float(cfg.accept_boundary_mean_weight), 0.001),
                    accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.0025),
                ),
            )
        )
    if severe_iou or (not severe_outside and not severe_undersize):
        extra_candidates.append(
            (
                "pose_bbox_b",
                replace(
                    cfg,
                    refit_iters=max(int(cfg.refit_iters), 96),
                    lr=min(float(cfg.lr), 0.005),
                    y_bin_count=max(int(cfg.y_bin_count), 120),
                    x_bin_count=max(int(cfg.x_bin_count), 60),
                    outside_loss_weight=max(float(cfg.outside_loss_weight), 11.0),
                    outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 5.0),
                    inside_loss_weight=min(float(cfg.inside_loss_weight), 0.055),
                    span_loss_weight=max(float(cfg.span_loss_weight), 0.022),
                    gap_loss_weight=max(float(cfg.gap_loss_weight), 0.028),
                    span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.010),
                    gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.014),
                    arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.024),
                    arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.012),
                    arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.026),
                    bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.04),
                    joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.02),
                    accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.0025),
                ),
            )
        )
    if bool(cfg.hard_frame_enable_beta_nudge) and float(area_ratio_before) < 0.95 and float(outside_before) < 0.085:
        extra_candidates.append(
            (
                "beta_nudge",
                replace(
                    cfg,
                    refit_iters=max(int(cfg.refit_iters), 96),
                    lr=min(float(cfg.lr), 0.0045),
                    y_bin_count=max(int(cfg.y_bin_count), 120),
                    x_bin_count=max(int(cfg.x_bin_count), 60),
                    outside_loss_weight=max(float(cfg.outside_loss_weight), 12.0),
                    outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 9.0),
                    inside_loss_weight=max(float(cfg.inside_loss_weight), 0.10),
                    inside_peak_loss_weight=max(float(cfg.inside_peak_loss_weight), 0.12),
                    span_loss_weight=max(float(cfg.span_loss_weight), 0.028),
                    gap_loss_weight=max(float(cfg.gap_loss_weight), 0.040),
                    span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.014),
                    gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.020),
                    arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.028),
                    arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.015),
                    arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.026),
                    bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.03),
                    joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.015),
                    optimize_betas=True,
                    betas_reg_weight=max(float(cfg.betas_reg_weight), 0.20),
                    pose_reg_weight=max(float(cfg.pose_reg_weight), 0.06),
                    orient_reg_weight=max(float(cfg.orient_reg_weight), 0.04),
                    transl_reg_weight=max(float(cfg.transl_reg_weight), 0.12),
                    accept_boundary_mean_weight=max(float(cfg.accept_boundary_mean_weight), 0.001),
                    accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.003),
                ),
            )
        )
    max_extra = max(0, int(cfg.hard_frame_max_extra_candidates))
    if max_extra > 0:
        candidates.extend(extra_candidates[:max_extra])
    return candidates


def _make_signed_distance_map(mask_u8: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = outside.astype(np.float32)
    sdf[m > 0] = -inside[m > 0].astype(np.float32)
    return sdf


def _sample_map_torch(map_t: torch.Tensor, uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[1] != 2:
        return torch.zeros((0,), dtype=map_t.dtype, device=map_t.device)
    h, w = int(image_hw[0]), int(image_hw[1])
    if h <= 1 or w <= 1 or int(uv.shape[0]) <= 0:
        return torch.zeros((0,), dtype=map_t.dtype, device=map_t.device)
    gx = (uv[:, 0] / float(max(w - 1, 1))) * 2.0 - 1.0
    gy = (uv[:, 1] / float(max(h - 1, 1))) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(map_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(-1)


def _collect_contour_points(
    uv: torch.Tensor,
    valid_v: torch.Tensor,
    target_bbox: torch.Tensor,
    *,
    y_bin_count: int,
    x_bin_count: int,
) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[1] != 2:
        return torch.zeros((0, 2), dtype=torch.float32, device=uv.device)
    valid = valid_v & torch.isfinite(uv).all(dim=1)
    if not bool(valid.any().item()):
        return torch.zeros((0, 2), dtype=torch.float32, device=uv.device)
    x = uv[:, 0]
    y = uv[:, 1]
    xmin, ymin, xmax, ymax = [float(v) for v in target_bbox.reshape(-1).tolist()]
    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)
    pts: list[torch.Tensor] = []
    big = torch.full_like(x, 1.0e9)
    small = torch.full_like(x, -1.0e9)
    y_edges = torch.linspace(ymin, ymax + 1.0e-4, steps=max(2, int(y_bin_count) + 1), device=uv.device)
    for i_bin in range(max(1, int(y_bin_count))):
        in_bin = valid & (y >= y_edges[i_bin]) & (y < y_edges[i_bin + 1])
        if not bool(in_bin.any().item()):
            continue
        idx_l = torch.argmin(torch.where(in_bin, x, big))
        idx_r = torch.argmax(torch.where(in_bin, x, small))
        pts.append(uv[idx_l : idx_l + 1])
        pts.append(uv[idx_r : idx_r + 1])

        in_left = in_bin & (x <= x_mid)
        in_right = in_bin & (x >= x_mid)
        if bool(in_left.any().item()) and bool(in_right.any().item()):
            idx_inner_l = torch.argmax(torch.where(in_left, x, small))
            idx_inner_r = torch.argmin(torch.where(in_right, x, big))
            if float((x[idx_inner_r] - x[idx_inner_l]).detach().cpu().item()) > 1.5:
                pts.append(uv[idx_inner_l : idx_inner_l + 1])
                pts.append(uv[idx_inner_r : idx_inner_r + 1])

    x_edges = torch.linspace(xmin, xmax + 1.0e-4, steps=max(2, int(x_bin_count) + 1), device=uv.device)
    for i_bin in range(max(1, int(x_bin_count))):
        in_bin = valid & (x >= x_edges[i_bin]) & (x < x_edges[i_bin + 1])
        if not bool(in_bin.any().item()):
            continue
        idx_t = torch.argmin(torch.where(in_bin, y, big))
        idx_b = torch.argmax(torch.where(in_bin, y, small))
        pts.append(uv[idx_t : idx_t + 1])
        pts.append(uv[idx_b : idx_b + 1])

        in_top = in_bin & (y <= y_mid)
        in_bottom = in_bin & (y >= y_mid)
        if bool(in_top.any().item()) and bool(in_bottom.any().item()):
            idx_inner_t = torch.argmax(torch.where(in_top, y, small))
            idx_inner_b = torch.argmin(torch.where(in_bottom, y, big))
            if float((y[idx_inner_b] - y[idx_inner_t]).detach().cpu().item()) > 1.5:
                pts.append(uv[idx_inner_t : idx_inner_t + 1])
                pts.append(uv[idx_inner_b : idx_inner_b + 1])
    if len(pts) <= 0:
        return uv[valid]
    return torch.cat(pts, dim=0)


def _resolve_person_mask(
    *,
    frame_name: str,
    frames_dir: Path | None,
    person_masks_by_name: dict[str, np.ndarray] | None,
    cfg: SilhouetteRefineConfig,
) -> tuple[np.ndarray | None, str]:
    cached = None
    if isinstance(person_masks_by_name, dict):
        cached = person_masks_by_name.get(str(frame_name), None)
        if cached is not None:
            cached = (np.asarray(cached, dtype=np.uint8) > 0).astype(np.uint8) * 255
    mode = str(cfg.mask_source or "modnet").strip().lower()
    if mode == "cache" and cached is not None and int((cached > 0).sum()) > 0:
        return cached, "cache"
    if frames_dir is not None:
        img_path = frames_dir / str(frame_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            try:
                mask_modnet = estimate_person_mask_modnet_eval_style_np(
                    img,
                    hint_mask_u8=None,
                    modnet_ref_size=int(cfg.modnet_ref_size),
                    modnet_threshold=float(cfg.modnet_threshold),
                    modnet_ckpt_path=(str(cfg.modnet_ckpt_path).strip() or None),
                )
                if int((mask_modnet > 0).sum()) > 0:
                    return (np.asarray(mask_modnet, dtype=np.uint8) > 0).astype(np.uint8) * 255, "modnet"
            except Exception:
                pass
    if cached is not None and int((cached > 0).sum()) > 0:
        return cached, "cache_fallback"
    return None, "missing"


def apply_silhouette_refine(
    *,
    records: list[dict[str, Any]],
    assets_dir: Path,
    config: SilhouetteRefineConfig | None = None,
    person_masks_by_name: dict[str, np.ndarray] | None = None,
    frames_dir: Path | None = None,
) -> SilhouetteRefineResult:
    cfg = SilhouetteRefineConfig() if config is None else config
    meta: dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "mask_source": str(cfg.mask_source),
    }
    if (not bool(cfg.enabled)) or len(records) <= 0:
        meta["reason"] = "disabled_or_empty"
        return SilhouetteRefineResult(records=records, meta=meta)
    if torch is None or F is None or smplx is None:
        meta["reason"] = "missing_torch_or_smplx"
        return SilhouetteRefineResult(records=records, meta=meta)

    device = _resolve_device(cfg.device)
    meta["device"] = device

    out_records = [dict(rec) for rec in records]
    model_cache: dict[tuple[str, str], Any] = {}

    n_frames = len(records)
    iou_before_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    iou_after_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    outside_before_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    outside_after_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    score_before_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    score_after_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    total_loss_arr = np.full((n_frames,), np.nan, dtype=np.float32)
    accepted_arr = np.zeros((n_frames,), dtype=np.uint8)
    skipped_arr = np.zeros((n_frames,), dtype=np.uint8)
    mask_source_arr: list[str] = []
    candidate_name_arr = [""] * n_frames

    for rec_idx, rec in enumerate(records):
        frame_name = str(rec.get("frame_name", ""))
        person_mask, mask_source = _resolve_person_mask(
            frame_name=frame_name,
            frames_dir=frames_dir,
            person_masks_by_name=person_masks_by_name,
            cfg=cfg,
        )
        mask_source_arr.append(str(mask_source))
        if person_mask is None or int((np.asarray(person_mask, dtype=np.uint8) > 0).sum()) <= 0:
            skipped_arr[rec_idx] = np.uint8(1)
            continue

        target_bbox_np = _bbox_from_mask_np(person_mask)
        if target_bbox_np is None:
            skipped_arr[rec_idx] = np.uint8(1)
            continue

        focal_xy = np.asarray(rec.get("camera_focal", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        princpt_xy = np.asarray(rec.get("camera_princpt", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if focal_xy.size < 2 or princpt_xy.size < 2 or (not np.all(np.isfinite(focal_xy[:2]))) or (not np.all(np.isfinite(princpt_xy[:2]))):
            skipped_arr[rec_idx] = np.uint8(1)
            continue

        gender = str(rec.get("gender", "neutral")).strip().lower()
        model_key = ("smplx", gender)
        if model_key not in model_cache:
            model_cache[model_key] = _create_model(assets_dir=assets_dir, gender=gender, device=device)
        model = model_cache[model_key]

        params = dict(rec.get("smplestx_params", {}))
        num_betas = int(getattr(model, "num_betas", 10))
        num_expr = int(getattr(model, "num_expression_coeffs", 10))
        body_dim = int(getattr(model, "NUM_BODY_JOINTS", 21) * 3)

        go0_np = _read_param(params, "global_orient", 3)
        bp0_np = _read_param(params, "body_pose", body_dim)
        tr0_np = np.asarray(rec.get("cam_t", params.get("transl", np.zeros((1, 3), dtype=np.float32))), dtype=np.float32).reshape(-1)
        if tr0_np.size < 3:
            tr0_np = np.pad(tr0_np, (0, 3 - tr0_np.size))
        tr0_np = tr0_np[:3].astype(np.float32)
        betas0_np = np.asarray(params.get("betas", np.zeros((1, num_betas), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if betas0_np.size < num_betas:
            betas0_np = np.pad(betas0_np, (0, num_betas - betas0_np.size))
        betas0_np = betas0_np[:num_betas].astype(np.float32)

        lh_np = _read_param(params, "left_hand_pose", 45)
        rh_np = _read_param(params, "right_hand_pose", 45)
        jaw_np = _read_param(params, "jaw_pose", 3)
        le_np = _read_param(params, "leye_pose", 3)
        re_np = _read_param(params, "reye_pose", 3)
        expr_np = _read_param(params, "expression", num_expr)

        go = torch.nn.Parameter(torch.from_numpy(go0_np.reshape(1, 3)).to(device=device, dtype=torch.float32))
        bp = torch.nn.Parameter(torch.from_numpy(bp0_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32))
        tr = torch.nn.Parameter(torch.from_numpy(tr0_np.reshape(1, 3)).to(device=device, dtype=torch.float32))
        betas_t = torch.from_numpy(betas0_np.reshape(1, num_betas)).to(device=device, dtype=torch.float32)
        beta_params: list[torch.nn.Parameter] = []
        if bool(cfg.optimize_betas):
            betas_param = torch.nn.Parameter(betas_t.clone())
            betas_t = betas_param
            beta_params = [betas_param]

        go0_t = torch.from_numpy(go0_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
        bp0_t = torch.from_numpy(bp0_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32)
        tr0_t = torch.from_numpy(tr0_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
        betas0_t = torch.from_numpy(betas0_np.reshape(1, num_betas)).to(device=device, dtype=torch.float32)

        sdf_t = torch.from_numpy(_make_signed_distance_map(person_mask)).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target_bbox_t = torch.from_numpy(target_bbox_np).to(device=device, dtype=torch.float32)
        image_hw = tuple(int(x) for x in person_mask.shape[:2])
        span_targets_cache: dict[int, dict[str, torch.Tensor]] = {}

        target_joints2d_np = _sanitize_joints2d_np(rec.get("joints2d", None))
        target_joints2d = (
            torch.from_numpy(target_joints2d_np[:, :2]).to(device=device, dtype=torch.float32)
            if target_joints2d_np.shape[0] > 0
            else None
        )
        target_joints2d_conf = (
            torch.from_numpy(target_joints2d_np[:, 2]).to(device=device, dtype=torch.float32)
            if target_joints2d_np.shape[0] > 0 and target_joints2d_np.shape[1] >= 3
            else None
        )
        arm_boxes_np = _build_arm_boxes_np(
            target_joints2d_np,
            image_hw=image_hw,
            conf_min=0.05,
            pad_scale=float(cfg.arm_box_pad_scale),
            min_pad_px=float(cfg.arm_box_min_pad_px),
        )
        arm_boxes_t = (
            torch.from_numpy(arm_boxes_np).to(device=device, dtype=torch.float32)
            if arm_boxes_np.shape[0] > 0
            else torch.zeros((0, 4), device=device, dtype=torch.float32)
        )

        fixed_kwargs = {
            "left_hand_pose": torch.from_numpy(lh_np.reshape(1, 45)).to(device=device, dtype=torch.float32),
            "right_hand_pose": torch.from_numpy(rh_np.reshape(1, 45)).to(device=device, dtype=torch.float32),
            "jaw_pose": torch.from_numpy(jaw_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "leye_pose": torch.from_numpy(le_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "reye_pose": torch.from_numpy(re_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "expression": torch.from_numpy(expr_np.reshape(1, num_expr)).to(device=device, dtype=torch.float32),
            "return_verts": True,
        }

        def _forward_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out = model(
                betas=betas_t,
                global_orient=go,
                body_pose=bp,
                transl=tr,
                **fixed_kwargs,
            )
            verts = out.vertices[0]
            joints = out.joints[0]
            uv_v, valid_v = _project_points_torch(
                verts,
                fx=float(focal_xy[0]),
                fy=float(focal_xy[1]),
                cx=float(princpt_xy[0]),
                cy=float(princpt_xy[1]),
            )
            return verts, joints, uv_v, valid_v

        with torch.no_grad():
            verts_init, joints_init, uv_init, valid_init = _forward_state()
        uv_init_np = uv_init.detach().cpu().numpy().astype(np.float32)
        valid_init_np = valid_init.detach().cpu().numpy().astype(bool)
        mask_init = _rasterize_mask_np(uv_init_np, valid_init_np, np.asarray(rec["faces"], dtype=np.int32), image_hw=image_hw)
        iou_before = _mask_iou(mask_init, person_mask)
        outside_before = _mask_outside_ratio(mask_init, person_mask)
        area_ratio_before = _mask_area_ratio(mask_init, person_mask)
        boundary_mean_before, boundary_p95_before = _boundary_distance_stats_np(mask_init, person_mask)
        arm_stats_before = _arm_region_stats_np(mask_init, person_mask, arm_boxes_np, image_hw=image_hw) if arm_boxes_np.shape[0] > 0 else None
        arm_iou_before = float(arm_stats_before["iou"]) if isinstance(arm_stats_before, dict) else 0.0
        arm_outside_before = float(arm_stats_before["outside"]) if isinstance(arm_stats_before, dict) else 1.0
        arm_boundary_mean_before = float(arm_stats_before["boundary_mean_px"]) if isinstance(arm_stats_before, dict) else float("inf")
        arm_boundary_p95_before = float(arm_stats_before["boundary_p95_px"]) if isinstance(arm_stats_before, dict) else float("inf")
        score_before = _score_mask_fit(
            iou=iou_before,
            outside=outside_before,
            area_ratio=area_ratio_before,
            outside_weight=float(cfg.accept_outside_weight),
            oversize_weight=float(cfg.accept_oversize_weight),
            undersize_weight=float(cfg.accept_undersize_weight),
            boundary_mean_px=float(boundary_mean_before),
            boundary_p95_px=float(boundary_p95_before),
            boundary_mean_weight=float(cfg.accept_boundary_mean_weight),
            boundary_p95_weight=float(cfg.accept_boundary_p95_weight),
        )

        iou_before_arr[rec_idx] = float(iou_before)
        outside_before_arr[rec_idx] = float(outside_before)
        score_before_arr[rec_idx] = float(score_before)
        def _run_candidate(candidate_cfg: SilhouetteRefineConfig, *, init_state: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
            go_init_np = go0_np if not isinstance(init_state, dict) else np.asarray(init_state.get("global_orient", go0_np), dtype=np.float32).reshape(-1)[:3]
            bp_init_np = bp0_np if not isinstance(init_state, dict) else np.asarray(init_state.get("body_pose", bp0_np), dtype=np.float32).reshape(-1)[:body_dim]
            tr_init_np = tr0_np if not isinstance(init_state, dict) else np.asarray(init_state.get("transl", tr0_np), dtype=np.float32).reshape(-1)[:3]
            betas_init_np = betas0_np if not isinstance(init_state, dict) else np.asarray(init_state.get("betas", betas0_np), dtype=np.float32).reshape(-1)[:num_betas]
            temporal_anchor = isinstance(init_state, dict)
            go = torch.nn.Parameter(torch.from_numpy(go_init_np.reshape(1, 3)).to(device=device, dtype=torch.float32))
            bp = torch.nn.Parameter(torch.from_numpy(bp_init_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32))
            tr = torch.nn.Parameter(torch.from_numpy(tr_init_np.reshape(1, 3)).to(device=device, dtype=torch.float32))
            betas_t = torch.from_numpy(betas_init_np.reshape(1, num_betas)).to(device=device, dtype=torch.float32)
            beta_params: list[torch.nn.Parameter] = []
            if bool(candidate_cfg.optimize_betas):
                betas_param = torch.nn.Parameter(betas_t.clone())
                betas_t = betas_param
                beta_params = [betas_param]
            go_anchor_t = torch.from_numpy(go_init_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
            bp_anchor_t = torch.from_numpy(bp_init_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32)
            tr_anchor_t = torch.from_numpy(tr_init_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
            betas_anchor_t = torch.from_numpy(betas_init_np.reshape(1, num_betas)).to(device=device, dtype=torch.float32)
            cand_arm_boxes_np = _build_arm_boxes_np(
                target_joints2d_np,
                image_hw=image_hw,
                conf_min=0.05,
                pad_scale=float(candidate_cfg.arm_box_pad_scale),
                min_pad_px=float(candidate_cfg.arm_box_min_pad_px),
            )
            cand_arm_boxes_t = (
                torch.from_numpy(cand_arm_boxes_np).to(device=device, dtype=torch.float32)
                if cand_arm_boxes_np.shape[0] > 0
                else torch.zeros((0, 4), device=device, dtype=torch.float32)
            )

            def _forward_state_local() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                out = model(
                    betas=betas_t,
                    global_orient=go,
                    body_pose=bp,
                    transl=tr,
                    **fixed_kwargs,
                )
                verts = out.vertices[0]
                joints = out.joints[0]
                uv_v, valid_v = _project_points_torch(
                    verts,
                    fx=float(focal_xy[0]),
                    fy=float(focal_xy[1]),
                    cx=float(princpt_xy[0]),
                    cy=float(princpt_xy[1]),
                )
                return verts, joints, uv_v, valid_v

            optimizer = torch.optim.Adam([go, bp, tr] + beta_params, lr=float(candidate_cfg.lr))
            last_loss = float("nan")
            span_bin_count = int(max(1, candidate_cfg.y_bin_count))
            span_targets_t = span_targets_cache.get(span_bin_count, None)
            if span_targets_t is None:
                span_targets_np = _extract_mask_span_targets(person_mask, y_bin_count=span_bin_count)
                span_targets_t = {
                    "y_edges": torch.from_numpy(span_targets_np["y_edges"]).to(device=device, dtype=torch.float32),
                    "x_mid": torch.as_tensor(span_targets_np["x_mid"], device=device, dtype=torch.float32),
                    "outer_valid": torch.from_numpy(span_targets_np["outer_valid"].astype(np.float32)).to(device=device, dtype=torch.float32),
                    "outer_left": torch.from_numpy(span_targets_np["outer_left"]).to(device=device, dtype=torch.float32),
                    "outer_right": torch.from_numpy(span_targets_np["outer_right"]).to(device=device, dtype=torch.float32),
                    "gap_valid": torch.from_numpy(span_targets_np["gap_valid"].astype(np.float32)).to(device=device, dtype=torch.float32),
                    "gap_left": torch.from_numpy(span_targets_np["gap_left"]).to(device=device, dtype=torch.float32),
                    "gap_right": torch.from_numpy(span_targets_np["gap_right"]).to(device=device, dtype=torch.float32),
                }
                span_targets_cache[span_bin_count] = span_targets_t
            for it in range(max(1, int(candidate_cfg.refit_iters))):
                optimizer.zero_grad(set_to_none=True)
                verts, joints, uv_v, valid_v = _forward_state_local()
                contour_uv = _collect_contour_points(
                    uv=uv_v,
                    valid_v=valid_v,
                    target_bbox=target_bbox_t,
                    y_bin_count=int(candidate_cfg.y_bin_count),
                    x_bin_count=int(candidate_cfg.x_bin_count),
                )
                if int(contour_uv.shape[0]) <= 0:
                    break
                sdf_samples = _sample_map_torch(sdf_t, contour_uv, image_hw=image_hw)
                outside_term = torch.relu(sdf_samples)
                inside_term = torch.relu((-sdf_samples) - float(candidate_cfg.inside_margin_px))
                loss = float(candidate_cfg.outside_loss_weight) * (
                    outside_term.pow(2).mean() if int(outside_term.numel()) > 0 else torch.tensor(0.0, device=device)
                )
                if float(candidate_cfg.outside_peak_loss_weight) > 0.0 and int(outside_term.numel()) > 0:
                    loss = loss + float(candidate_cfg.outside_peak_loss_weight) * _tail_mean_torch(
                        outside_term.pow(2), float(candidate_cfg.peak_fraction)
                    )
                loss = loss + float(candidate_cfg.inside_loss_weight) * (
                    inside_term.pow(2).mean() if int(inside_term.numel()) > 0 else torch.tensor(0.0, device=device)
                )
                if float(candidate_cfg.inside_peak_loss_weight) > 0.0 and int(inside_term.numel()) > 0:
                    loss = loss + float(candidate_cfg.inside_peak_loss_weight) * _tail_mean_torch(
                        inside_term.pow(2), float(candidate_cfg.peak_fraction)
                    )
                if (
                    float(candidate_cfg.span_loss_weight) > 0.0
                    or float(candidate_cfg.gap_loss_weight) > 0.0
                    or float(candidate_cfg.span_peak_loss_weight) > 0.0
                    or float(candidate_cfg.gap_peak_loss_weight) > 0.0
                ):
                    span_loss, gap_loss = _span_edge_loss_torch(
                        uv=uv_v,
                        valid_v=valid_v,
                        span_targets=span_targets_t,
                    )
                    if float(candidate_cfg.span_loss_weight) > 0.0:
                        loss = loss + float(candidate_cfg.span_loss_weight) * span_loss
                    if float(candidate_cfg.gap_loss_weight) > 0.0:
                        loss = loss + float(candidate_cfg.gap_loss_weight) * gap_loss
                    if float(candidate_cfg.span_peak_loss_weight) > 0.0 and torch.isfinite(span_loss):
                        loss = loss + float(candidate_cfg.span_peak_loss_weight) * span_loss.pow(2)
                    if float(candidate_cfg.gap_peak_loss_weight) > 0.0 and torch.isfinite(gap_loss):
                        loss = loss + float(candidate_cfg.gap_peak_loss_weight) * gap_loss.pow(2)
                if (
                    int(cand_arm_boxes_t.shape[0]) > 0
                    and (
                        float(candidate_cfg.arm_local_loss_weight) > 0.0
                        or float(candidate_cfg.arm_local_peak_loss_weight) > 0.0
                    )
                ):
                    arm_sel = _select_points_in_boxes_torch(contour_uv, cand_arm_boxes_t)
                    arm_uv = contour_uv[arm_sel]
                    if int(arm_uv.shape[0]) > 0:
                        arm_sdf = _sample_map_torch(sdf_t, arm_uv, image_hw=image_hw)
                        arm_outside = torch.relu(arm_sdf)
                        arm_inside = torch.relu((-arm_sdf) - float(candidate_cfg.inside_margin_px))
                        arm_term = arm_outside.pow(2).mean() + 0.35 * arm_inside.pow(2).mean()
                        if float(candidate_cfg.arm_local_loss_weight) > 0.0:
                            loss = loss + float(candidate_cfg.arm_local_loss_weight) * arm_term
                        if float(candidate_cfg.arm_local_peak_loss_weight) > 0.0 and int(arm_outside.numel()) > 0:
                            loss = loss + float(candidate_cfg.arm_local_peak_loss_weight) * _tail_mean_torch(
                                arm_outside.pow(2), float(candidate_cfg.peak_fraction)
                            )

                valid_uv = uv_v[valid_v]
                if int(valid_uv.shape[0]) > 0 and float(candidate_cfg.bbox_loss_weight) > 0.0:
                    bbox_pred = torch.stack(
                        (
                            valid_uv[:, 0].amin(),
                            valid_uv[:, 1].amin(),
                            valid_uv[:, 0].amax(),
                            valid_uv[:, 1].amax(),
                        ),
                        dim=0,
                    )
                    loss = loss + float(candidate_cfg.bbox_loss_weight) * F.smooth_l1_loss(
                        bbox_pred, target_bbox_t, reduction="mean"
                    )

                if target_joints2d is not None and float(candidate_cfg.joint2d_loss_weight) > 0.0:
                    uv_j, valid_j = _project_points_torch(
                        joints,
                        fx=float(focal_xy[0]),
                        fy=float(focal_xy[1]),
                        cx=float(princpt_xy[0]),
                        cy=float(princpt_xy[1]),
                    )
                    joint_n = min(int(uv_j.shape[0]), int(target_joints2d.shape[0]))
                    if joint_n > 0:
                        valid2 = valid_j[:joint_n]
                        if target_joints2d_conf is not None and int(target_joints2d_conf.shape[0]) >= joint_n:
                            valid2 = valid2 & (target_joints2d_conf[:joint_n] > 0.05)
                        if bool(valid2.any().item()):
                            pred2 = uv_j[:joint_n][valid2]
                            tgt2 = target_joints2d[:joint_n][valid2]
                            if target_joints2d_conf is not None and int(target_joints2d_conf.shape[0]) >= joint_n:
                                conf2 = target_joints2d_conf[:joint_n][valid2].clamp(0.0, 1.0).unsqueeze(1)
                                loss_j2 = (F.smooth_l1_loss(pred2, tgt2, reduction="none") * conf2).mean()
                            else:
                                loss_j2 = F.smooth_l1_loss(pred2, tgt2, reduction="mean")
                            loss = loss + float(candidate_cfg.joint2d_loss_weight) * loss_j2
                            if float(candidate_cfg.arm_joint2d_loss_weight) > 0.0 and joint_n > 10:
                                arm_ids = torch.as_tensor([5, 6, 7, 8, 9, 10], device=device, dtype=torch.long)
                                arm_ids = arm_ids[arm_ids < joint_n]
                                if int(arm_ids.numel()) > 0:
                                    arm_valid = valid2[arm_ids]
                                    if bool(arm_valid.any().item()):
                                        pred_arm = uv_j[:joint_n][arm_ids][arm_valid]
                                        tgt_arm = target_joints2d[:joint_n][arm_ids][arm_valid]
                                        if target_joints2d_conf is not None and int(target_joints2d_conf.shape[0]) >= joint_n:
                                            conf_arm = target_joints2d_conf[:joint_n][arm_ids][arm_valid].clamp(0.0, 1.0).unsqueeze(1)
                                            arm_j2 = (F.smooth_l1_loss(pred_arm, tgt_arm, reduction="none") * conf_arm).mean()
                                        else:
                                            arm_j2 = F.smooth_l1_loss(pred_arm, tgt_arm, reduction="mean")
                                        loss = loss + float(candidate_cfg.arm_joint2d_loss_weight) * arm_j2

                loss = loss + float(candidate_cfg.orient_reg_weight) * F.mse_loss(go, go0_t, reduction="mean")
                loss = loss + float(candidate_cfg.pose_reg_weight) * F.mse_loss(bp, bp0_t, reduction="mean")
                loss = loss + float(candidate_cfg.transl_reg_weight) * F.mse_loss(tr, tr0_t, reduction="mean")
                if bool(candidate_cfg.optimize_betas):
                    loss = loss + float(candidate_cfg.betas_reg_weight) * F.mse_loss(betas_t, betas0_t, reduction="mean")
                if temporal_anchor:
                    if float(candidate_cfg.temporal_orient_reg_weight) > 0.0:
                        loss = loss + float(candidate_cfg.temporal_orient_reg_weight) * F.mse_loss(
                            go, go_anchor_t, reduction="mean"
                        )
                    if float(candidate_cfg.temporal_pose_reg_weight) > 0.0:
                        loss = loss + float(candidate_cfg.temporal_pose_reg_weight) * F.mse_loss(
                            bp, bp_anchor_t, reduction="mean"
                        )
                    if float(candidate_cfg.temporal_transl_reg_weight) > 0.0:
                        loss = loss + float(candidate_cfg.temporal_transl_reg_weight) * F.mse_loss(
                            tr, tr_anchor_t, reduction="mean"
                        )
                    if bool(candidate_cfg.optimize_betas) and float(candidate_cfg.temporal_betas_reg_weight) > 0.0:
                        loss = loss + float(candidate_cfg.temporal_betas_reg_weight) * F.mse_loss(
                            betas_t, betas_anchor_t, reduction="mean"
                        )
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                if it == max(1, int(candidate_cfg.refit_iters)) // 2:
                    for group in optimizer.param_groups:
                        group["lr"] = float(group["lr"]) * 0.35

            with torch.no_grad():
                verts_final, joints_final, uv_final, valid_final = _forward_state_local()
            uv_final_np = uv_final.detach().cpu().numpy().astype(np.float32)
            valid_final_np = valid_final.detach().cpu().numpy().astype(bool)
            mask_final = _rasterize_mask_np(uv_final_np, valid_final_np, np.asarray(rec["faces"], dtype=np.int32), image_hw=image_hw)
            iou_after = _mask_iou(mask_final, person_mask)
            outside_after = _mask_outside_ratio(mask_final, person_mask)
            area_ratio_after = _mask_area_ratio(mask_final, person_mask)
            boundary_mean_after, boundary_p95_after = _boundary_distance_stats_np(mask_final, person_mask)
            arm_stats_after = _arm_region_stats_np(mask_final, person_mask, cand_arm_boxes_np, image_hw=image_hw) if cand_arm_boxes_np.shape[0] > 0 else None
            arm_iou_after = float(arm_stats_after["iou"]) if isinstance(arm_stats_after, dict) else 0.0
            arm_outside_after = float(arm_stats_after["outside"]) if isinstance(arm_stats_after, dict) else 1.0
            arm_boundary_mean_after = float(arm_stats_after["boundary_mean_px"]) if isinstance(arm_stats_after, dict) else float("inf")
            arm_boundary_p95_after = float(arm_stats_after["boundary_p95_px"]) if isinstance(arm_stats_after, dict) else float("inf")
            arm_score_after = (
                _score_mask_fit(
                    iou=arm_iou_after,
                    outside=arm_outside_after,
                    area_ratio=float(arm_stats_after["area_ratio"]),
                    outside_weight=float(candidate_cfg.accept_outside_weight),
                    oversize_weight=float(candidate_cfg.accept_oversize_weight),
                    undersize_weight=float(candidate_cfg.accept_undersize_weight),
                    boundary_mean_px=float(arm_boundary_mean_after),
                    boundary_p95_px=float(arm_boundary_p95_after),
                    boundary_mean_weight=float(candidate_cfg.accept_boundary_mean_weight),
                    boundary_p95_weight=float(candidate_cfg.accept_boundary_p95_weight),
                )
                if isinstance(arm_stats_after, dict)
                else float("-inf")
            )
            score_after = _score_mask_fit(
                iou=iou_after,
                outside=outside_after,
                area_ratio=area_ratio_after,
                outside_weight=float(candidate_cfg.accept_outside_weight),
                oversize_weight=float(candidate_cfg.accept_oversize_weight),
                undersize_weight=float(candidate_cfg.accept_undersize_weight),
                boundary_mean_px=float(boundary_mean_after),
                boundary_p95_px=float(boundary_p95_after),
                boundary_mean_weight=float(candidate_cfg.accept_boundary_mean_weight),
                boundary_p95_weight=float(candidate_cfg.accept_boundary_p95_weight),
            )
            outside_not_worse = bool(outside_after <= (outside_before + 1e-4))
            arm_region_accept = bool(
                isinstance(arm_stats_after, dict)
                and outside_after <= (outside_before + float(candidate_cfg.arm_accept_global_outside_slack))
                and score_after >= (score_before - float(candidate_cfg.arm_accept_score_slack))
                and arm_outside_after <= (arm_outside_before + float(candidate_cfg.arm_accept_local_outside_slack))
                and (
                    (
                        arm_boundary_mean_after <= (arm_boundary_mean_before - float(candidate_cfg.arm_accept_boundary_gain_px))
                        and arm_boundary_p95_after <= (arm_boundary_p95_before + 1.5)
                    )
                    or (
                        arm_iou_after >= (arm_iou_before + float(candidate_cfg.arm_accept_iou_gain))
                        and arm_boundary_mean_after <= (arm_boundary_mean_before + 0.5)
                    )
                )
            )
            accepted = bool(
                (
                    outside_not_worse
                    and (
                        (score_after > score_before + 1e-4)
                        or ((outside_after + 1e-4) < outside_before and iou_after >= (iou_before - 0.01))
                        or (
                            outside_after <= (outside_before + 1e-4)
                            and boundary_p95_after <= (boundary_p95_before - 1.0)
                            and iou_after >= (iou_before - 0.02)
                        )
                        or (
                            outside_after <= (outside_before + 0.003)
                            and area_ratio_before < 0.95
                            and area_ratio_after >= (area_ratio_before + 0.02)
                            and boundary_mean_after <= (boundary_mean_before - 0.5)
                        )
                    )
                )
                or arm_region_accept
            )
            transl_np = tr.detach().cpu().numpy().reshape(3).astype(np.float32)
            params_new = dict(params)
            params_new["global_orient"] = _reshape_like(go.detach().cpu().numpy(), params.get("global_orient", None), 3)
            params_new["body_pose"] = _reshape_like(bp.detach().cpu().numpy(), params.get("body_pose", None), body_dim)
            params_new["transl"] = _reshape_like(transl_np, params.get("transl", None), 3)
            if bool(candidate_cfg.optimize_betas):
                params_new["betas"] = _reshape_like(betas_t.detach().cpu().numpy(), params.get("betas", None), num_betas)
            verts_np = verts_final.detach().cpu().numpy().astype(np.float32)
            joints_np = joints_final.detach().cpu().numpy().astype(np.float32)
            joints2d_np, joints2d_valid = _project_points_np(
                joints_np,
                fx=float(focal_xy[0]),
                fy=float(focal_xy[1]),
                cx=float(princpt_xy[0]),
                cy=float(princpt_xy[1]),
            )
            return {
                "accepted": accepted,
                "score_after": float(score_after),
                "iou_after": float(iou_after),
                "outside_after": float(outside_after),
                "area_ratio_after": float(area_ratio_after),
                "boundary_mean_after": float(boundary_mean_after),
                "boundary_p95_after": float(boundary_p95_after),
                "arm_iou_after": float(arm_iou_after),
                "arm_outside_after": float(arm_outside_after),
                "arm_boundary_mean_after": float(arm_boundary_mean_after),
                "arm_boundary_p95_after": float(arm_boundary_p95_after),
                "arm_score_after": float(arm_score_after),
                "arm_region_accept": bool(arm_region_accept),
                "last_loss": float(last_loss),
                "params_new": params_new,
                "transl_np": transl_np,
                "verts_np": verts_np,
                "joints_np": joints_np,
                "joints2d_np": np.concatenate(
                    [joints2d_np, joints2d_valid.astype(np.float32).reshape(-1, 1)],
                    axis=1,
                ).astype(np.float32),
            }

        candidate_results: list[tuple[str, dict[str, Any]]] = []
        hard_frame = bool(
            (float(iou_before) < float(cfg.hard_frame_iou_threshold))
            or (float(outside_before) > float(cfg.hard_frame_outside_threshold))
            or (float(area_ratio_before) < float(cfg.hard_frame_area_ratio_threshold))
        )
        for candidate_name, candidate_cfg in _select_candidate_configs(
            cfg,
            iou_before=float(iou_before),
            outside_before=float(outside_before),
            area_ratio_before=float(area_ratio_before),
        ):
            candidate_results.append((str(candidate_name), _run_candidate(candidate_cfg)))
        if bool(cfg.enable_temporal_candidate) and hard_frame and len(candidate_results) <= 2:
            temporal_cfg = replace(
                cfg,
                refit_iters=max(int(cfg.refit_iters), 88),
                lr=min(float(cfg.lr), 0.0045),
                y_bin_count=max(int(cfg.y_bin_count), 112),
                x_bin_count=max(int(cfg.x_bin_count), 56),
                outside_loss_weight=max(float(cfg.outside_loss_weight), 11.0),
                outside_peak_loss_weight=max(float(cfg.outside_peak_loss_weight), 7.0),
                inside_loss_weight=max(float(cfg.inside_loss_weight), 0.09),
                inside_peak_loss_weight=max(float(cfg.inside_peak_loss_weight), 0.08),
                span_loss_weight=max(float(cfg.span_loss_weight), 0.034),
                gap_loss_weight=max(float(cfg.gap_loss_weight), 0.048),
                span_peak_loss_weight=max(float(cfg.span_peak_loss_weight), 0.018),
                gap_peak_loss_weight=max(float(cfg.gap_peak_loss_weight), 0.024),
                arm_local_loss_weight=max(float(cfg.arm_local_loss_weight), 0.038),
                arm_local_peak_loss_weight=max(float(cfg.arm_local_peak_loss_weight), 0.020),
                arm_joint2d_loss_weight=max(float(cfg.arm_joint2d_loss_weight), 0.032),
                bbox_loss_weight=max(float(cfg.bbox_loss_weight), 0.025),
                joint2d_loss_weight=max(float(cfg.joint2d_loss_weight), 0.015),
                pose_reg_weight=max(float(cfg.pose_reg_weight), 0.03),
                orient_reg_weight=max(float(cfg.orient_reg_weight), 0.03),
                transl_reg_weight=max(float(cfg.transl_reg_weight), 0.10),
                temporal_pose_reg_weight=max(float(cfg.temporal_pose_reg_weight), 0.06),
                temporal_orient_reg_weight=max(float(cfg.temporal_orient_reg_weight), 0.06),
                temporal_transl_reg_weight=max(float(cfg.temporal_transl_reg_weight), 0.16),
                temporal_betas_reg_weight=max(float(cfg.temporal_betas_reg_weight), 0.08),
                accept_boundary_mean_weight=max(float(cfg.accept_boundary_mean_weight), 0.001),
                accept_boundary_p95_weight=max(float(cfg.accept_boundary_p95_weight), 0.0025),
            )
            for temp_name, temp_state in _build_temporal_inits(
                records,
                rec_idx,
                body_dim=body_dim,
                num_betas=num_betas,
                max_frame_gap=int(cfg.temporal_max_frame_gap),
            ):
                candidate_results.append((str(temp_name), _run_candidate(temporal_cfg, init_state=temp_state)))
        candidate_name_arr[rec_idx] = "|".join(name for name, _ in candidate_results)

        accepted_candidates = [
            (name, result)
            for name, result in candidate_results
            if bool(result.get("accepted", False))
        ]
        if len(accepted_candidates) > 0:
            best_name, best_result = max(
                accepted_candidates,
                key=lambda item: (
                    float(item[1]["score_after"]),
                    float(item[1].get("arm_score_after", float("-inf"))),
                    -float(item[1]["outside_after"]),
                    -float(item[1].get("boundary_p95_after", float("inf"))),
                    float(item[1]["iou_after"]),
                ),
            )
            for cand_name, cand_result in accepted_candidates:
                cand_score = float(cand_result["score_after"])
                cand_outside = float(cand_result["outside_after"])
                cand_boundary_p95 = float(cand_result.get("boundary_p95_after", float("inf")))
                cand_area = float(cand_result.get("area_ratio_after", 1.0))
                cand_arm_score = float(cand_result.get("arm_score_after", float("-inf")))
                cand_arm_boundary = float(cand_result.get("arm_boundary_mean_after", float("inf")))
                cand_arm_outside = float(cand_result.get("arm_outside_after", float("inf")))
                best_score = float(best_result["score_after"])
                best_outside = float(best_result["outside_after"])
                best_boundary_p95 = float(best_result.get("boundary_p95_after", float("inf")))
                best_area = float(best_result.get("area_ratio_after", 1.0))
                best_arm_score = float(best_result.get("arm_score_after", float("-inf")))
                best_arm_boundary = float(best_result.get("arm_boundary_mean_after", float("inf")))
                best_arm_outside = float(best_result.get("arm_outside_after", float("inf")))
                if (cand_outside + 1e-3) < best_outside and cand_score >= (best_score - 0.01):
                    best_name, best_result = cand_name, cand_result
                    continue
                if (cand_boundary_p95 + 1.0) < best_boundary_p95 and cand_outside <= (best_outside + 0.003) and cand_score >= (best_score - 0.015):
                    best_name, best_result = cand_name, cand_result
                    continue
                if cand_arm_score >= (best_arm_score + 0.025) and cand_outside <= (best_outside + 0.010):
                    best_name, best_result = cand_name, cand_result
                    continue
                if (cand_arm_boundary + 1.0) < best_arm_boundary and cand_arm_outside <= (best_arm_outside + 0.015) and cand_score >= (best_score - 0.03):
                    best_name, best_result = cand_name, cand_result
                    continue
                if (
                    best_area < 0.91
                    and cand_area >= (best_area + 0.018)
                    and cand_outside <= min(best_outside + 0.010, 0.080)
                    and cand_score >= (best_score - 0.020)
                    and cand_boundary_p95 <= (best_boundary_p95 + 1.5)
                ):
                    best_name, best_result = cand_name, cand_result
        else:
            best_name, best_result = candidate_results[0]

        iou_after = float(best_result["iou_after"])
        outside_after = float(best_result["outside_after"])
        score_after = float(best_result["score_after"])
        last_loss = float(best_result["last_loss"])

        iou_after_arr[rec_idx] = iou_after
        outside_after_arr[rec_idx] = outside_after
        score_after_arr[rec_idx] = score_after
        total_loss_arr[rec_idx] = last_loss

        accepted = bool(best_result.get("accepted", False))
        accepted_arr[rec_idx] = np.uint8(1 if accepted else 0)
        if not accepted:
            continue

        rec_out = out_records[rec_idx]
        rec_out["silhouette_refine_pre_smplestx_params"] = dict(params)
        rec_out["silhouette_refine_pre_cam_t"] = np.asarray(rec.get("cam_t", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
        rec_out["silhouette_refine_pre_verts_cam"] = np.asarray(rec.get("verts_cam", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        rec_out["silhouette_refine_pre_verts_local"] = np.asarray(rec.get("verts_local", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        rec_out["silhouette_refine_pre_joints_cam"] = np.asarray(rec.get("joints_cam", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        rec_out["silhouette_refine_pre_joints2d"] = np.asarray(_sanitize_joints2d_np(rec.get("joints2d", None)), dtype=np.float32)
        rec_out["smplestx_params"] = dict(best_result["params_new"])
        rec_out["cam_t"] = np.asarray(best_result["transl_np"], dtype=np.float32)
        rec_out["verts_cam"] = np.asarray(best_result["verts_np"], dtype=np.float32)
        rec_out["verts_local"] = np.asarray(best_result["verts_np"] - np.asarray(best_result["transl_np"], dtype=np.float32).reshape(1, 3), dtype=np.float32)
        rec_out["joints_cam"] = np.asarray(best_result["joints_np"], dtype=np.float32)
        rec_out["joints2d"] = np.asarray(best_result["joints2d_np"], dtype=np.float32)
        rec_out["silhouette_refine_total_loss"] = last_loss
        rec_out["silhouette_refine_iou_before"] = float(iou_before)
        rec_out["silhouette_refine_iou_after"] = iou_after
        rec_out["silhouette_refine_outside_before"] = float(outside_before)
        rec_out["silhouette_refine_outside_after"] = outside_after
        rec_out["silhouette_refine_variant"] = str(best_name)

    meta["mask_sources"] = np.asarray(mask_source_arr, dtype=object)
    meta["candidate_names"] = np.asarray(candidate_name_arr, dtype=object)
    meta["iou_before"] = iou_before_arr
    meta["iou_after"] = iou_after_arr
    meta["outside_before"] = outside_before_arr
    meta["outside_after"] = outside_after_arr
    meta["score_before"] = score_before_arr
    meta["score_after"] = score_after_arr
    meta["refit_total_loss"] = total_loss_arr
    meta["accepted"] = accepted_arr
    meta["skipped"] = skipped_arr
    return SilhouetteRefineResult(records=out_records, meta=meta)
