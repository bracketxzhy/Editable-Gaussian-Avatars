from __future__ import annotations

import cv2
import numpy as np


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


def _resize_mask_nn(mask_u8: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    src = np.asarray(mask_u8, dtype=np.uint8)
    if src.shape[0] == h and src.shape[1] == w:
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_NEAREST)


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
    search_max_side: int = 256,
    local_scale_delta: float = 0.05,
    local_scale_steps: int = 3,
    local_trans_ratio: float = 0.03,
    local_trans_min_px: float = 4.0,
    local_trans_max_px: float = 24.0,
    local_trans_steps: int = 5,
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
    s_best = float(np.clip(0.5 * (wp / ws + hp / hs), scale_min, scale_max))
    tx_best = float(cpx - s_best * csx)
    ty_best = float(cpy - s_best * csy)

    iou_before, outside_before, area_ratio_before = _metrics(smpl_mask, person_mask)
    score_before = _score(
        iou=iou_before,
        outside=outside_before,
        area_ratio=area_ratio_before,
        scale=1.0,
        scale_init=max(s_best, 1e-6),
        outside_weight=outside_weight,
        oversize_weight=oversize_weight,
        undersize_weight=undersize_weight,
        expand_weight=expand_weight,
    )

    h0, w0 = int(image_hw[0]), int(image_hw[1])
    ds = min(1.0, float(max(search_max_side, 8)) / float(max(h0, w0, 1)))
    hs = max(8, int(round(float(h0) * ds)))
    ws = max(8, int(round(float(w0) * ds)))
    small_hw = (hs, ws)
    smpl_small = _resize_mask_nn(smpl_mask, small_hw)
    person_small = _resize_mask_nn(person_mask, small_hw)
    uv_small = np.asarray(uv, dtype=np.float32).copy()
    uv_small[:, 0] *= float(ws) / float(max(w0, 1))
    uv_small[:, 1] *= float(hs) / float(max(h0, 1))

    trans_base = max(
        float(local_trans_min_px),
        min(
            float(local_trans_max_px),
            float(local_trans_ratio) * float(max(wp, hp)),
        ),
    )
    tx_grid = np.linspace(tx_best - trans_base, tx_best + trans_base, num=max(int(local_trans_steps), 1), dtype=np.float32)
    ty_grid = np.linspace(ty_best - trans_base, ty_best + trans_base, num=max(int(local_trans_steps), 1), dtype=np.float32)
    if int(local_scale_steps) <= 1 or float(local_scale_delta) <= 0.0:
        scale_grid = np.asarray([s_best], dtype=np.float32)
    else:
        scale_grid = np.linspace(
            max(scale_min, s_best - float(local_scale_delta)),
            min(scale_max, s_best + float(local_scale_delta)),
            num=max(int(local_scale_steps), 2),
            dtype=np.float32,
        )

    best = {
        "scale": float(s_best),
        "tx": float(tx_best),
        "ty": float(ty_best),
        "iou": float(iou_before),
        "outside": float(outside_before),
        "area_ratio": float(area_ratio_before),
        "score": float(score_before),
    }
    for s_c in scale_grid.tolist():
        scale_ratio_x = float(ws) / float(max(w0, 1))
        scale_ratio_y = float(hs) / float(max(h0, 1))
        for tx_c in tx_grid.tolist():
            for ty_c in ty_grid.tolist():
                uv_c = apply_uv_similarity(
                    uv_small,
                    scale=float(s_c),
                    tx=float(tx_c) * scale_ratio_x,
                    ty=float(ty_c) * scale_ratio_y,
                )
                mask_c = _rasterize_mesh_mask(uv=uv_c, valid_v=valid_v, faces=faces, image_hw=small_hw)
                iou_c, outside_c, area_ratio_c = _metrics(mask_c, person_small)
                score_c = _score(
                    iou=iou_c,
                    outside=outside_c,
                    area_ratio=area_ratio_c,
                    scale=float(s_c),
                    scale_init=max(s_best, 1e-6),
                    outside_weight=outside_weight,
                    oversize_weight=oversize_weight,
                    undersize_weight=undersize_weight,
                    expand_weight=expand_weight,
                )
                if (
                    (score_c > float(best["score"]) + 1e-6)
                    or (
                        abs(score_c - float(best["score"])) <= 1e-6
                        and outside_c < float(best["outside"]) - 1e-6
                    )
                ):
                    best = {
                        "scale": float(s_c),
                        "tx": float(tx_c),
                        "ty": float(ty_c),
                        "iou": float(iou_c),
                        "outside": float(outside_c),
                        "area_ratio": float(area_ratio_c),
                        "score": float(score_c),
                    }

    uv_best = apply_uv_similarity(uv, scale=float(best["scale"]), tx=float(best["tx"]), ty=float(best["ty"]))
    mask_best = _rasterize_mesh_mask(uv=uv_best, valid_v=valid_v, faces=faces, image_hw=image_hw)
    iou_after, outside_after, area_ratio_after = _metrics(mask_best, person_mask)
    score_after = _score(
        iou=iou_after,
        outside=outside_after,
        area_ratio=area_ratio_after,
        scale=float(best["scale"]),
        scale_init=max(s_best, 1e-6),
        outside_weight=outside_weight,
        oversize_weight=oversize_weight,
        undersize_weight=undersize_weight,
        expand_weight=expand_weight,
    )
    accepted = bool(
        (
            (score_after > score_before + 1e-4)
            and (outside_after <= outside_before + float(accept_outside_slack))
        )
        or (
            (iou_after > iou_before + 0.01)
            and (outside_after <= outside_before + float(accept_outside_slack) + 0.01)
        )
    )
    return uv_best, {
        "applied": True,
        "accepted": accepted,
        "scale": float(best["scale"]),
        "tx_px": float(best["tx"]),
        "ty_px": float(best["ty"]),
        "center_src": [float(csx), float(csy)],
        "center_tgt": [float(cpx), float(cpy)],
        "bbox_src": b_s.tolist(),
        "bbox_tgt": b_p.tolist(),
        "search_downsample_ratio": float(ds),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "outside_before": float(outside_before),
        "outside_after": float(outside_after),
        "area_ratio_before": float(area_ratio_before),
        "area_ratio_after": float(area_ratio_after),
        "score_before": float(score_before),
        "score_after": float(score_after),
    }
