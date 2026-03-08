from __future__ import annotations

import cv2
import numpy as np


def _resize_mask_nearest(mask_u8: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid out_hw: {out_hw}")
    return cv2.resize(np.asarray(mask_u8, dtype=np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def mask_bbox(mask_u8: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask_u8, dtype=np.uint8) > 0)
    if ys.size <= 0 or xs.size <= 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_outside_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=np.uint8) > 0
    b = np.asarray(mask_b, dtype=np.uint8) > 0
    area = float(max(1, a.sum()))
    outside = float(np.logical_and(a, np.logical_not(b)).sum())
    return outside / area


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=np.uint8) > 0
    b = np.asarray(mask_b, dtype=np.uint8) > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def apply_uv_similarity(uv: np.ndarray, scale: float, tx: float, ty: float) -> np.ndarray:
    uv2 = np.asarray(uv, dtype=np.float32).copy()
    uv2[:, 0] = float(scale) * uv2[:, 0] + float(tx)
    uv2[:, 1] = float(scale) * uv2[:, 1] + float(ty)
    return uv2


def apply_similarity_to_camera_points(
    points_cam: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
    tx_px: float,
    ty_px: float,
) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3:
        return pts
    z = pts[:, 2]
    dx = (float(tx_px) + (float(scale) - 1.0) * float(cx)) / float(max(abs(fx), 1e-6))
    dy = (float(ty_px) + (float(scale) - 1.0) * float(cy)) / float(max(abs(fy), 1e-6))
    pts[:, 0] = float(scale) * pts[:, 0] + dx * z
    pts[:, 1] = float(scale) * pts[:, 1] + dy * z
    return pts


def _rasterize_mesh_mask(uv: np.ndarray, valid_v: np.ndarray, faces: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a >= len(valid_v) or b >= len(valid_v) or c >= len(valid_v):
            continue
        if (not valid_v[a]) or (not valid_v[b]) or (not valid_v[c]):
            continue
        pts = np.round(uv[[a, b, c], :]).astype(np.int32)
        xmin = int(np.min(pts[:, 0]))
        xmax = int(np.max(pts[:, 0]))
        ymin = int(np.min(pts[:, 1]))
        ymax = int(np.max(pts[:, 1]))
        if xmax < 0 or ymax < 0 or xmin >= w or ymin >= h:
            continue
        cv2.fillConvexPoly(mask, pts, color=255, lineType=cv2.LINE_8)
    return mask


def _metrics(mask_pred: np.ndarray, mask_gt: np.ndarray) -> tuple[float, float, float]:
    iou = mask_iou(mask_pred, mask_gt)
    outside = mask_outside_ratio(mask_pred, mask_gt)
    area_ratio = float((np.asarray(mask_pred, dtype=np.uint8) > 0).sum()) / float(
        max(1, (np.asarray(mask_gt, dtype=np.uint8) > 0).sum())
    )
    return float(iou), float(outside), float(area_ratio)


def _score(
    *,
    iou: float,
    outside: float,
    area_ratio: float,
    scale: float,
    scale_init: float,
    outside_weight: float,
    oversize_weight: float,
    undersize_weight: float,
    expand_weight: float,
) -> float:
    oversize = max(0.0, float(area_ratio) - 1.0)
    undersize = max(0.0, 1.0 - float(area_ratio))
    expand = max(0.0, float(scale) / max(float(scale_init), 1e-6) - 1.0)
    return float(iou - outside_weight * outside - oversize_weight * oversize - undersize_weight * undersize - expand_weight * expand)


def estimate_mask_similarity_transform(
    *,
    uv: np.ndarray,
    valid_v: np.ndarray,
    faces: np.ndarray,
    image_hw: tuple[int, int],
    smpl_mask: np.ndarray,
    person_mask: np.ndarray,
    scale_min: float = 0.75,
    scale_max: float = 1.25,
    scale_delta: float = 0.08,
    scale_steps: int = 9,
    trans_ratio: float = 0.06,
    trans_min_px: float = 8.0,
    trans_max_px: float = 48.0,
    trans_steps: int = 9,
    outside_weight: float = 1.35,
    oversize_weight: float = 0.22,
    undersize_weight: float = 0.04,
    expand_weight: float = 0.08,
    accept_outside_slack: float = 0.01,
    search_max_side: int = 384,
) -> tuple[np.ndarray, dict]:
    b_s = mask_bbox(smpl_mask)
    b_p = mask_bbox(person_mask)
    if b_s is None or b_p is None:
        return uv, {"applied": False, "accepted": False, "reason": "empty_bbox"}

    ws = float(max(1.0, b_s[2] - b_s[0] + 1.0))
    hs = float(max(1.0, b_s[3] - b_s[1] + 1.0))
    wp = float(max(1.0, b_p[2] - b_p[0] + 1.0))
    hp = float(max(1.0, b_p[3] - b_p[1] + 1.0))
    csx = float(0.5 * (b_s[0] + b_s[2]))
    csy = float(0.5 * (b_s[1] + b_s[3]))
    cpx = float(0.5 * (b_p[0] + b_p[2]))
    cpy = float(0.5 * (b_p[1] + b_p[3]))
    s0 = float(np.clip(0.5 * (wp / ws + hp / hs), scale_min, scale_max))
    tx0 = float(cpx - s0 * csx)
    ty0 = float(cpy - s0 * csy)
    iou_before, outside_before, area_ratio_before = _metrics(smpl_mask, person_mask)
    score_before = _score(
        iou=iou_before,
        outside=outside_before,
        area_ratio=area_ratio_before,
        scale=1.0,
        scale_init=max(s0, 1e-6),
        outside_weight=outside_weight,
        oversize_weight=oversize_weight,
        undersize_weight=undersize_weight,
        expand_weight=expand_weight,
    )
    base_extent = float(max(wp, hp, ws, hs))
    trans_span = float(np.clip(float(trans_ratio) * base_extent, trans_min_px, trans_max_px))
    h, w = int(image_hw[0]), int(image_hw[1])
    long_side = max(h, w)
    ds_ratio = min(1.0, float(max(int(search_max_side), 32)) / float(max(long_side, 1)))
    hs_ = max(32, int(round(float(h) * ds_ratio)))
    ws_ = max(32, int(round(float(w) * ds_ratio)))
    if hs_ != h or ws_ != w:
        person_mask_search = _resize_mask_nearest(person_mask, (hs_, ws_))
        uv_search = np.asarray(uv, dtype=np.float32).copy()
        uv_search[:, 0] *= float(ws_) / float(max(w, 1))
        uv_search[:, 1] *= float(hs_) / float(max(h, 1))
    else:
        person_mask_search = np.asarray(person_mask, dtype=np.uint8)
        uv_search = np.asarray(uv, dtype=np.float32)

    best = None
    best_score = -1e18
    scale_center = float(s0)
    tx_center = float(tx0) * (float(ws_) / float(max(w, 1)))
    ty_center = float(ty0) * (float(hs_) / float(max(h, 1)))
    scale_window = float(scale_delta)
    trans_window = float(trans_span) * (float(ws_) / float(max(w, 1)))
    levels = (
        (float(max(scale_window, 0.01)), float(max(trans_window, 2.0)), max(5, int(min(scale_steps, 7)))),
        (float(max(scale_window * 0.35, 0.006)), float(max(trans_window * 0.35, 1.0)), 5),
        (float(max(scale_window * 0.15, 0.003)), float(max(trans_window * 0.15, 0.5)), 5),
    )
    for lvl_scale_delta, lvl_trans_span, lvl_steps in levels:
        ds_vals = np.linspace(-lvl_scale_delta, lvl_scale_delta, int(max(3, lvl_steps)))
        dt_vals = np.linspace(-lvl_trans_span, lvl_trans_span, int(max(3, lvl_steps)))
        lvl_best = None
        lvl_best_score = -1e18
        for ds in ds_vals:
            s = float(np.clip(scale_center * (1.0 + float(ds)), scale_min, scale_max))
            for dtx in dt_vals:
                for dty in dt_vals:
                    tx = float(tx_center + float(dtx))
                    ty = float(ty_center + float(dty))
                    uv2 = apply_uv_similarity(uv_search, scale=s, tx=tx, ty=ty)
                    mask2 = _rasterize_mesh_mask(uv=uv2, valid_v=valid_v, faces=faces, image_hw=(hs_, ws_))
                    iou, outside, area_ratio = _metrics(mask2, person_mask_search)
                    score = _score(
                        iou=iou,
                        outside=outside,
                        area_ratio=area_ratio,
                        scale=s,
                        scale_init=s0,
                        outside_weight=outside_weight,
                        oversize_weight=oversize_weight,
                        undersize_weight=undersize_weight,
                        expand_weight=expand_weight,
                    )
                    if score > lvl_best_score:
                        lvl_best_score = score
                        lvl_best = (s, tx, ty, iou, outside, area_ratio, score)
        if lvl_best is None:
            continue
        scale_center, tx_center, ty_center, iou_lvl, outside_lvl, area_ratio_lvl, score_lvl = lvl_best
        lvl_best_full = (
            scale_center,
            tx_center * (float(w) / float(max(ws_, 1))),
            ty_center * (float(h) / float(max(hs_, 1))),
            iou_lvl,
            outside_lvl,
            area_ratio_lvl,
            score_lvl,
        )
        if score_lvl > best_score:
            best_score = score_lvl
            best = lvl_best_full
    if best is None:
        return uv, {"applied": False, "accepted": False, "reason": "search_failed"}

    s_best, tx_best, ty_best, _, _, _, score_after = best
    uv_best = apply_uv_similarity(uv, scale=s_best, tx=tx_best, ty=ty_best)
    mask_best = _rasterize_mesh_mask(uv=uv_best, valid_v=valid_v, faces=faces, image_hw=image_hw)
    iou_after, outside_after, area_ratio_after = _metrics(mask_best, person_mask)
    accepted = bool((score_after > score_before + 1e-6) and (outside_after <= outside_before + float(accept_outside_slack)))
    return uv_best, {
        "applied": True,
        "accepted": accepted,
        "scale": float(s_best),
        "tx_px": float(tx_best),
        "ty_px": float(ty_best),
        "center_src": [float(csx), float(csy)],
        "center_tgt": [float(cpx), float(cpy)],
        "bbox_src": b_s.tolist(),
        "bbox_tgt": b_p.tolist(),
        "search_downsample_ratio": float(ds_ratio),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "outside_before": float(outside_before),
        "outside_after": float(outside_after),
        "area_ratio_before": float(area_ratio_before),
        "area_ratio_after": float(area_ratio_after),
        "score_before": float(score_before),
        "score_after": float(score_after),
    }
