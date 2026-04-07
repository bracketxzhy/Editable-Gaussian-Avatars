
from __future__ import annotations

import argparse
import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / 'src', THIS_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from avatar_gs.alignment.mask_sources import read_stage1_frame_camera_K
from avatar_gs.segmentation.fashn import get_fashn_label_maps, load_fashn_parser, predict_fashn_segmentation
from avatar_gs.segmentation.smpl_face_semantics import (
    load_stage1_vertices_sequence,
    rasterize_visible_face_index_map,
    resize_rgb_and_intrinsics,
    resolve_frame_image_path,
)


def iou_precision_recall(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    pa = float(a.sum())
    pb = float(b.sum())
    iou = inter / union if union > 0.0 else float('nan')
    precision = inter / pa if pa > 0.0 else float('nan')
    recall = inter / pb if pb > 0.0 else float('nan')
    return iou, precision, recall


def centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size <= 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_offset(a: np.ndarray, b: np.ndarray) -> float | None:
    ca = centroid_xy(a)
    cb = centroid_xy(b)
    if ca is None or cb is None:
        return None
    return float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def score_part(iou: float, precision: float, recall: float, centroid_px: float | None) -> float:
    iou_v = 0.0 if not math.isfinite(iou) else iou
    p_v = 0.0 if not math.isfinite(precision) else precision
    r_v = 0.0 if not math.isfinite(recall) else recall
    c_pen = min(float(centroid_px or 0.0), 24.0) / 24.0
    return float(0.60 * iou_v + 0.20 * p_v + 0.20 * r_v - 0.08 * c_pen)


def cluster_select(items: list[dict], cluster_gap: int, max_keep: int) -> list[dict]:
    if not items:
        return []
    items = sorted(items, key=lambda x: int(x['frame_index']))
    clusters: list[list[dict]] = [[items[0]]]
    for item in items[1:]:
        if int(item['frame_index']) - int(clusters[-1][-1]['frame_index']) <= int(cluster_gap):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    picked = []
    for cluster in clusters:
        picked.append(max(cluster, key=lambda x: float(x['score'])))
    picked.sort(key=lambda x: float(x['score']), reverse=True)
    return picked[: max_keep]


def sequence_bucket_select(items: list[dict], *, frame_count: int, bucket_count: int, max_keep: int, cluster_gap: int) -> list[dict]:
    if not items:
        return []
    items = sorted(items, key=lambda x: int(x['frame_index']))
    bucket_count = max(int(bucket_count), 1)
    selected: list[dict] = []
    used = set()
    for b in range(bucket_count):
        lo = int((b * frame_count) / bucket_count)
        hi = int(((b + 1) * frame_count) / bucket_count)
        bucket_items = [x for x in items if lo <= int(x['frame_index']) < hi]
        if not bucket_items:
            continue
        pick = max(bucket_items, key=lambda x: float(x['score']))
        key = str(pick['frame_name'])
        if key not in used:
            pick = dict(pick)
            pick.setdefault('view_bucket', 'sequence')
            selected.append(pick)
            used.add(key)
    rest = [x for x in cluster_select(items, cluster_gap=cluster_gap, max_keep=max_keep * 2) if str(x['frame_name']) not in used]
    for item in rest:
        clone = dict(item)
        clone.setdefault('view_bucket', 'sequence')
        selected.append(clone)
    selected.sort(key=lambda x: float(x['score']), reverse=True)
    return selected[:max_keep]


SIX_VIEW_BUCKETS = ('front', 'right', 'back', 'left', 'top', 'bottom')

def face_view_value(item: dict) -> int:
    return int(max(item.get('face_fashn_pixels', 0) or 0, item.get('face_projected_pixels', 0) or 0))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-8, None)
    return x / n


def infer_body_axes(verts_canonical: np.ndarray, faces: np.ndarray, raw_label_ids: np.ndarray, raw_label_names: list[str]) -> dict[str, int | float]:
    face_verts = verts_canonical[np.asarray(faces, dtype=np.int64)]
    face_centers = face_verts.mean(axis=1)
    face_normals = _normalize_rows(np.cross(face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0]))
    spans = verts_canonical.max(axis=0) - verts_canonical.min(axis=0)
    vertical_axis = int(np.argmax(spans))
    face_part_id = int(raw_label_names.index('face')) if 'face' in raw_label_names else 0
    face_mask = np.asarray(raw_label_ids == face_part_id, dtype=bool)
    rem_axes = [ax for ax in range(3) if ax != vertical_axis]
    face_mean_normal = face_normals[face_mask].mean(axis=0) if bool(face_mask.any()) else face_normals.mean(axis=0)
    face_mean_center = face_centers[face_mask].mean(axis=0) if bool(face_mask.any()) else face_centers.mean(axis=0)
    front_axis = int(max(rem_axes, key=lambda ax: abs(float(face_mean_normal[ax]))))
    lateral_axis = int([ax for ax in rem_axes if ax != front_axis][0])
    front_sign = 1.0 if float(face_mean_normal[front_axis]) >= 0.0 else -1.0
    top_sign = 1.0 if float(face_mean_center[vertical_axis]) >= 0.0 else -1.0
    lateral_sign = 1.0
    part_center_means: dict[str, list[float]] = {}
    for part_id, part_name in enumerate(raw_label_names):
        part_mask = np.asarray(raw_label_ids == int(part_id), dtype=bool)
        if not bool(part_mask.any()):
            continue
        part_center_means[str(part_name)] = face_centers[part_mask].mean(axis=0).astype(np.float64).tolist()
    return {
        'vertical_axis': vertical_axis,
        'front_axis': front_axis,
        'lateral_axis': lateral_axis,
        'front_sign': front_sign,
        'top_sign': top_sign,
        'lateral_sign': lateral_sign,
        'face_centers': face_centers,
        'face_normals': face_normals,
        'part_center_means': part_center_means,
    }


def classify_part_view_bucket_from_faces(face_ids: np.ndarray, face_weights: np.ndarray, axes: dict[str, int | float | np.ndarray], part_name: str | None = None) -> str:
    if face_ids.size <= 0:
        return 'front'
    centers = np.asarray(axes['face_centers'])[face_ids]
    normals = np.asarray(axes['face_normals'])[face_ids]
    w = np.asarray(face_weights, dtype=np.float64).reshape(-1)
    if w.size != face_ids.size or float(w.sum()) <= 0.0:
        w = np.ones((face_ids.size,), dtype=np.float64)
    w = w / np.clip(w.sum(), 1e-8, None)
    mean_center = (centers * w[:, None]).sum(axis=0)
    mean_normal = (normals * w[:, None]).sum(axis=0)
    front_axis = int(axes['front_axis'])
    lateral_axis = int(axes['lateral_axis'])
    vertical_axis = int(axes['vertical_axis'])
    part_center_means = axes.get('part_center_means', {})
    part_center = mean_center
    if part_name is not None and isinstance(part_center_means, dict) and part_name in part_center_means:
        part_center = np.asarray(part_center_means[part_name], dtype=np.float64)
    rel_center = mean_center - part_center
    front_val = float(rel_center[front_axis]) * float(axes['front_sign'])
    side_val = float(rel_center[lateral_axis]) * float(axes['lateral_sign'])
    top_val = float(rel_center[vertical_axis]) * float(axes['top_sign'])
    abs_front = abs(front_val)
    abs_side = abs(side_val)
    abs_top = abs(top_val)
    cloth_like = {'top', 'pants', 'torso', 'dress', 'skirt'}
    top_gate = 1.80 if (part_name in cloth_like) else 1.25
    if abs_top > max(abs_front, abs_side) * top_gate and abs_top > 1e-6:
        return 'top' if top_val >= 0.0 else 'bottom'
    if abs_side > abs_front * 1.15 and abs_side > 1e-6:
        return 'right' if side_val >= 0.0 else 'left'
    if abs_front > 1e-6:
        return 'front' if front_val >= 0.0 else 'back'
    normal_front = float(mean_normal[front_axis]) * float(axes['front_sign'])
    normal_side = float(mean_normal[lateral_axis]) * float(axes['lateral_sign'])
    normal_top = float(mean_normal[vertical_axis]) * float(axes['top_sign'])
    if abs(normal_top) > max(abs(normal_front), abs(normal_side)) * top_gate:
        return 'top' if normal_top >= 0.0 else 'bottom'
    if abs(normal_side) > abs(normal_front) * 1.10:
        return 'right' if normal_side >= 0.0 else 'left'
    return 'front' if normal_front >= 0.0 else 'back'


def projected_area_value(item: dict) -> int:
    return int(item.get('projected_pixels', 0) or 0)


def texture_density_rgb(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if int(mask_bool.sum()) <= 16:
        return 0.0, 0.0
    rgb_f = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = (0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[mask_bool]
    if vals.size <= 0:
        return 0.0, 0.0
    return float(vals.mean()), float(np.percentile(vals, 90.0))


def part_region_quality(seg_mask: np.ndarray, proj_mask: np.ndarray, fg_mask: np.ndarray) -> tuple[float, float, float]:
    seg_bool = np.asarray(seg_mask, dtype=bool)
    proj_bool = np.asarray(proj_mask, dtype=bool)
    support = seg_bool | proj_bool
    if int(support.sum()) <= 16:
        return 0.0, 0.0, 1.0
    ys, xs = np.where(support)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    support_crop = support[y0:y1, x0:x1]
    fg_crop = np.asarray(fg_mask, dtype=bool)[y0:y1, x0:x1]
    bbox_area = max(int((y1 - y0) * (x1 - x0)), 1)
    inter = seg_bool & proj_bool
    support_fill_ratio = float(support_crop.sum()) / float(bbox_area)
    agreement_ratio = float(inter.sum()) / float(max(int(support.sum()), 1))
    contamination_ratio = float((fg_crop & (~support_crop)).sum()) / float(bbox_area)
    return support_fill_ratio, agreement_ratio, contamination_ratio


def _bucket_rank_key(bucket_name: str, item: dict):
    iou_v = float(item.get('iou', 0.0) or 0.0)
    precision_v = float(item.get('precision', 0.0) or 0.0)
    recall_v = float(item.get('recall', 0.0) or 0.0)
    centroid_penalty = float(item.get('centroid_offset_px', 9999.0) or 9999.0)
    area_v = projected_area_value(item)
    texture_mean = float(item.get('texture_density', 0.0) or 0.0)
    texture_p90 = float(item.get('texture_p90', 0.0) or 0.0)
    support_fill_ratio = float(item.get('support_fill_ratio', 0.0) or 0.0)
    agreement_ratio = float(item.get('agreement_ratio', 0.0) or 0.0)
    contamination_ratio = float(item.get('contamination_ratio', 1.0) or 1.0)
    part_name = str(item.get('part_name', ''))
    cloth_like = {'top', 'pants', 'torso', 'dress', 'skirt'}
    if part_name in cloth_like:
        return (agreement_ratio, -contamination_ratio, float(item['score']), texture_p90, texture_mean, support_fill_ratio, iou_v, precision_v, recall_v, -centroid_penalty, area_v)
    return (float(item['score']), iou_v, precision_v, recall_v, -centroid_penalty, area_v)


def bucket_max_projected_areas(items: list[dict]) -> dict[str, int]:
    non_empty_max: dict[str, int] = {}
    for bucket_name in SIX_VIEW_BUCKETS:
        bucket_items = [x for x in items if x.get('view_bucket') == bucket_name]
        if not bucket_items:
            continue
        non_empty_max[bucket_name] = int(max(max(projected_area_value(x), 1) for x in bucket_items))
    return non_empty_max


def proportional_bucket_minimums(items: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    non_empty_max = bucket_max_projected_areas(items)
    if not non_empty_max:
        return {}, {}
    min_area = max(min(non_empty_max.values()), 1)
    minimums = {bucket_name: int(math.ceil(float(area) / float(min_area))) for bucket_name, area in non_empty_max.items()}
    return minimums, non_empty_max


def _metric_ok(value, threshold: float) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return math.isfinite(v) and v >= float(threshold)


def six_view_bucket_select(
    items: list[dict],
    *,
    extra_score_margin: float,
    extra_min_iou: float,
    extra_min_precision: float,
    extra_min_recall: float,
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, int]]:
    if not items:
        return [], {}, {}, {}
    tagged = [dict(x) for x in items]
    minimums, bucket_max_areas = proportional_bucket_minimums(tagged)
    bucket_best_score: dict[str, float] = {}
    for bucket_name in SIX_VIEW_BUCKETS:
        bucket_items = [x for x in tagged if x.get('view_bucket') == bucket_name]
        if not bucket_items:
            continue
        bucket_best_score[bucket_name] = float(max(float(x['score']) for x in bucket_items))
    selected: list[dict] = []
    used = set()
    teacher_counts: dict[str, int] = {}
    for bucket_name in SIX_VIEW_BUCKETS:
        bucket_items = [x for x in tagged if x.get('view_bucket') == bucket_name]
        if not bucket_items:
            continue
        quota = int(minimums.get(bucket_name, 0))
        if quota <= 0:
            continue
        candidates = sorted(bucket_items, key=lambda x: _bucket_rank_key(bucket_name, x), reverse=True)
        taken = 0
        for item in candidates:
            key = str(item['frame_name'])
            if key in used:
                continue
            clone = dict(item)
            clone['teacher_role'] = 'base'
            selected.append(clone)
            used.add(key)
            taken += 1
            if taken >= quota:
                break
        teacher_counts[bucket_name] = int(taken)
    rest = [x for x in tagged if str(x['frame_name']) not in used]
    rest.sort(key=lambda x: _bucket_rank_key(str(x.get('view_bucket', 'unknown')), x), reverse=True)
    for item in rest:
        bucket_name = str(item.get('view_bucket', 'unknown'))
        best_score = float(bucket_best_score.get(bucket_name, float(item['score'])))
        score_floor = best_score - float(extra_score_margin)
        if float(item['score']) < score_floor:
            continue
        if not _metric_ok(item.get('iou', None), extra_min_iou):
            continue
        if not _metric_ok(item.get('precision', None), extra_min_precision):
            continue
        if not _metric_ok(item.get('recall', None), extra_min_recall):
            continue
        clone = dict(item)
        clone['teacher_role'] = 'extra'
        selected.append(clone)
        teacher_counts[bucket_name] = int(teacher_counts.get(bucket_name, 0)) + 1
    bucket_rank = {name: i for i, name in enumerate(SIX_VIEW_BUCKETS)}
    selected.sort(key=lambda x: (bucket_rank.get(str(x.get('view_bucket')), 99), -float(x['score']), -projected_area_value(x)))
    return selected, minimums, bucket_max_areas, teacher_counts


def main() -> None:
    ap = argparse.ArgumentParser('Select per-part teacher frames by FASHN-vs-SMPL agreement')
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--frames_dir', default='data/frames/input')
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    ap.add_argument('--out_dir', default='outputs/stage2_teacher_bank/part_teacher_bank_20260406')
    ap.add_argument('--max_edge', type=int, default=768)
    ap.add_argument('--min_pixels', type=int, default=128)
    ap.add_argument('--min_score', type=float, default=0.35)
    ap.add_argument('--cluster_gap', type=int, default=3)
    ap.add_argument('--max_keep_per_part', type=int, default=6)
    ap.add_argument('--body_bucket_count', type=int, default=4)
    ap.add_argument('--body_min_keep', type=int, default=4)
    ap.add_argument('--view_bucket_parts', default='hair,top,dress,skirt,pants,arms,hands,torso,feet')
    ap.add_argument('--extra_score_margin', type=float, default=0.015)
    ap.add_argument('--extra_min_iou', type=float, default=0.86)
    ap.add_argument('--extra_min_precision', type=float, default=0.94)
    ap.add_argument('--extra_min_recall', type=float, default=0.94)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve() if not Path(args.stage1_dir).is_absolute() else Path(args.stage1_dir).resolve()
    frames_dir = (project_root / args.frames_dir).resolve() if not Path(args.frames_dir).is_absolute() else Path(args.frames_dir).resolve()
    supervision_cache = (project_root / args.supervision_cache).resolve() if not Path(args.supervision_cache).is_absolute() else Path(args.supervision_cache).resolve()
    semantics_npz = (project_root / args.semantics_npz).resolve() if not Path(args.semantics_npz).is_absolute() else Path(args.semantics_npz).resolve()
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sup = np.load(str(supervision_cache), allow_pickle=True)
    frame_names = [str(x) for x in np.asarray(sup['frame_names']).reshape(-1).tolist()]
    frame_index_map = {name: idx for idx, name in enumerate(frame_names)}

    sem = np.load(str(semantics_npz), allow_pickle=True)
    faces = np.asarray(sem['faces'], dtype=np.int32)
    raw_label_ids = np.asarray(sem['raw_label_ids'], dtype=np.int32)
    raw_label_names = [str(x) for x in np.asarray(sem['raw_label_names']).reshape(-1).tolist()]
    smpl_seq = np.load(str(stage1_dir / 'smpl_params_sequence.npz'), allow_pickle=True)
    verts_canonical = np.asarray(smpl_seq['verts_canonical_cam'], dtype=np.float32)
    body_axes = infer_body_axes(verts_canonical, faces, raw_label_ids, raw_label_names)

    verts_frame_names, verts_cam_all = load_stage1_vertices_sequence(stage1_dir)
    verts_idx = {str(n): i for i, n in enumerate(verts_frame_names)}

    ids_to_labels, labels_to_ids = get_fashn_label_maps()
    parser = load_fashn_parser(device='cuda')

    per_part_candidates: dict[str, list[dict]] = {name: [] for name in raw_label_names if name != 'background'}
    per_frame: list[dict] = []

    for frame_name in frame_names:
        if frame_name not in verts_idx:
            continue
        frame_i = int(frame_index_map[frame_name])
        image_path = resolve_frame_image_path(frames_dir, frame_name)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = read_stage1_frame_camera_K(
            stage1_dir=stage1_dir,
            frame_index=frame_i,
            frame_name=frame_name,
            cam_json_path=stage1_dir / 'cameras.json',
            image_hw=rgb.shape[:2],
        )
        rgb_small, K_small = resize_rgb_and_intrinsics(rgb, np.asarray(K, dtype=np.float32), max_edge=int(args.max_edge))
        seg = predict_fashn_segmentation(rgb_small, parser=parser)
        face_index_map_2d, _ = rasterize_visible_face_index_map(
            verts_cam=np.asarray(verts_cam_all[int(verts_idx[frame_name])], dtype=np.float32),
            faces=faces,
            K=np.asarray(K_small, dtype=np.float32),
            image_hw=rgb_small.shape[:2],
        )
        valid = face_index_map_2d >= 0
        projected_labels = np.full(face_index_map_2d.shape, fill_value=-1, dtype=np.int32)
        if bool(valid.any()):
            fids = face_index_map_2d[valid].astype(np.int64)
            projected_labels[valid] = raw_label_ids[fids]

        frame_report = {'frame_name': frame_name, 'frame_index': frame_i, 'parts': {}}
        face_part_id = labels_to_ids.get('face', None)
        face_proj_px = 0
        face_seg_px = 0
        if face_part_id is not None:
            face_proj_px = int((projected_labels == int(face_part_id)).sum())
            face_seg_px = int((np.asarray(seg == int(face_part_id), dtype=bool)).sum())

        for part_name in raw_label_names:
            if part_name == 'background':
                continue
            if part_name not in labels_to_ids:
                continue
            part_id = int(labels_to_ids[part_name])
            proj_mask = projected_labels == part_id
            seg_mask = np.asarray(seg == part_id, dtype=bool)
            proj_px = int(proj_mask.sum())
            seg_px = int(seg_mask.sum())
            iou, precision, recall = iou_precision_recall(proj_mask, seg_mask)
            c_off = centroid_offset(proj_mask, seg_mask)
            score = score_part(iou, precision, recall, c_off)
            texture_mask = np.asarray(seg_mask & proj_mask, dtype=np.uint8)
            if int(texture_mask.sum()) <= 32:
                texture_mask = np.asarray(seg_mask, dtype=np.uint8)
            texture_density, texture_p90 = texture_density_rgb(rgb_small, texture_mask)
            support_fill_ratio, agreement_ratio, contamination_ratio = part_region_quality(seg_mask, proj_mask, np.asarray(seg > 0, dtype=bool))
            view_bucket = 'front'
            if int(proj_px) > 0 and bool(proj_mask.any()):
                part_face_ids = face_index_map_2d[proj_mask].astype(np.int64)
                if part_face_ids.size > 0:
                    unique_fids, unique_counts = np.unique(part_face_ids, return_counts=True)
                    view_bucket = classify_part_view_bucket_from_faces(unique_fids, unique_counts, body_axes, part_name=part_name)
            info = {
                'frame_name': frame_name,
                'frame_index': frame_i,
                'part_name': part_name,
                'part_id': part_id,
                'projected_pixels': proj_px,
                'fashn_pixels': seg_px,
                'iou': None if not math.isfinite(iou) else float(iou),
                'precision': None if not math.isfinite(precision) else float(precision),
                'recall': None if not math.isfinite(recall) else float(recall),
                'centroid_offset_px': None if c_off is None else float(c_off),
                'score': float(score),
                'texture_density': float(texture_density),
                'texture_p90': float(texture_p90),
                'support_fill_ratio': float(support_fill_ratio),
                'agreement_ratio': float(agreement_ratio),
                'contamination_ratio': float(contamination_ratio),
                'face_projected_pixels': int(face_proj_px),
                'face_fashn_pixels': int(face_seg_px),
                'view_bucket': str(view_bucket),
                'usable': bool(max(proj_px, seg_px) >= int(args.min_pixels) and score >= float(args.min_score)),
            }
            frame_report['parts'][part_name] = info
            if info['usable']:
                per_part_candidates[part_name].append(info)
        per_frame.append(frame_report)

    front_anchor_index = 0
    face_candidates = per_part_candidates.get('face', [])
    if face_candidates:
        front_anchor_index = int(max(face_candidates, key=lambda x: (float(x['score']), face_view_value(x)))['frame_index'])

    body_parts = {'hair', 'top', 'dress', 'skirt', 'pants', 'arms', 'hands', 'legs', 'feet', 'torso'}
    view_bucket_parts = {x.strip() for x in str(args.view_bucket_parts).split(',') if x.strip()}
    teacher_bank = {}
    for part_name, items in per_part_candidates.items():
        if part_name == 'face':
            bucket_minimums = {}
            bucket_max_projected_pixels = {}
            bucket_teacher_counts = {}
            picked = []
            if items:
                picked = [max(items, key=lambda x: (float(x['score']), projected_area_value(x), -abs(int(x['frame_index']) - int(front_anchor_index))))]
            selection_mode = 'single_teacher_best_match'
        elif part_name in view_bucket_parts:
            picked, bucket_minimums, bucket_max_projected_pixels, bucket_teacher_counts = six_view_bucket_select(
                items,
                extra_score_margin=float(args.extra_score_margin),
                extra_min_iou=float(args.extra_min_iou),
                extra_min_precision=float(args.extra_min_precision),
                extra_min_recall=float(args.extra_min_recall),
            )
            selection_mode = 'three_d_bucket_two_d_consistency'
        elif part_name in body_parts:
            bucket_minimums = {}
            bucket_max_projected_pixels = {}
            bucket_teacher_counts = {}
            picked = sequence_bucket_select(
                items,
                frame_count=len(frame_names),
                bucket_count=int(args.body_bucket_count),
                max_keep=max(int(args.max_keep_per_part), int(args.body_min_keep)),
                cluster_gap=int(args.cluster_gap),
            )
            selection_mode = 'sequence_buckets'
        else:
            bucket_minimums = {}
            bucket_max_projected_pixels = {}
            bucket_teacher_counts = {}
            picked = cluster_select(items, cluster_gap=int(args.cluster_gap), max_keep=int(args.max_keep_per_part))
            selection_mode = 'cluster_topk'
        teacher_bank[part_name] = {
            'candidate_count': len(items),
            'selection_mode': selection_mode,
            'bucket_minimums': bucket_minimums,
            'bucket_max_projected_pixels': bucket_max_projected_pixels,
            'bucket_teacher_counts': bucket_teacher_counts,
            'teachers': picked,
        }

    report_lines = ['# Part Teacher Bank', '', 'Teacher selection from FASHN-2D vs SMPL-projected agreement.', '']
    for part_name, payload in teacher_bank.items():
        report_lines.append(f'## {part_name}')
        report_lines.append(f"candidate_count: {payload['candidate_count']}")
        report_lines.append(f"selection_mode: {payload['selection_mode']}")
        if payload.get('bucket_minimums'):
            report_lines.append(f"bucket_minimums: {payload['bucket_minimums']}")
        if payload.get('bucket_max_projected_pixels'):
            report_lines.append(f"bucket_max_projected_pixels: {payload['bucket_max_projected_pixels']}")
        if payload.get('bucket_teacher_counts'):
            report_lines.append(f"bucket_teacher_counts: {payload['bucket_teacher_counts']}")
        if not payload['teachers']:
            report_lines.append('no usable teacher')
            report_lines.append('')
            continue
        for item in payload['teachers']:
            report_lines.append(
                f"- {item['frame_name']} | score={item['score']:.4f} | iou={item['iou']:.4f} | precision={item['precision']:.4f} | recall={item['recall']:.4f} | centroid_px={(item['centroid_offset_px'] if item['centroid_offset_px'] is not None else float('nan')):.2f}"
            )
        report_lines.append('')

    summary = {
        'frame_count': len(frame_names),
        'raw_label_names': raw_label_names,
        'min_pixels': int(args.min_pixels),
        'min_score': float(args.min_score),
        'cluster_gap': int(args.cluster_gap),
        'max_keep_per_part': int(args.max_keep_per_part),
        'body_bucket_count': int(args.body_bucket_count),
        'body_min_keep': int(args.body_min_keep),
        'extra_score_margin': float(args.extra_score_margin),
        'extra_min_iou': float(args.extra_min_iou),
        'extra_min_precision': float(args.extra_min_precision),
        'extra_min_recall': float(args.extra_min_recall),
        'front_anchor_index': int(front_anchor_index),
        'body_axes': {k: v for k, v in body_axes.items() if k not in {'face_centers', 'face_normals'}},
        'teacher_bank': teacher_bank,
    }
    (out_dir / 'teacher_bank.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'per_frame_metrics.json').write_text(json.dumps(per_frame, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'report.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print(json.dumps({'out_dir': str(out_dir), 'parts': list(teacher_bank.keys())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
