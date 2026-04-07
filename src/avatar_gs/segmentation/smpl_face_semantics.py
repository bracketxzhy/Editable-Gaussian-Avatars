from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from avatar_gs.alignment.mask_sources import (
    project_vertices_to_image,
    read_stage1_frame_camera_K,
)
from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from avatar_gs.segmentation.fashn import (
    FASHN_COARSE_GROUPS,
    FASHN_COARSE_TO_ID,
    build_fashn_raw_to_coarse_id,
    get_fashn_label_maps,
    load_fashn_parser,
    normalize_fashn_device,
    predict_fashn_segmentation,
    resolve_fashn_model_dir,
    resolve_fashn_runtime_dir,
)


def load_stage1_vertices_sequence(stage1_dir: str | Path) -> tuple[list[str], np.ndarray]:
    stage1_path = Path(stage1_dir)
    seq_path = stage1_path / "smpl_vertices_sequence.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing stage1 vertices sequence: {seq_path}")
    seq = np.load(str(seq_path), allow_pickle=True)
    if "frame_names" not in seq.files or "verts_cam" not in seq.files:
        raise RuntimeError(f"Bad stage1 vertices format: {seq_path}")
    frame_names = [str(x) for x in np.asarray(seq["frame_names"]).reshape(-1).tolist()]
    verts_cam_all = np.asarray(seq["verts_cam"], dtype=np.float32)
    if verts_cam_all.ndim != 3 or verts_cam_all.shape[-1] != 3:
        raise RuntimeError(f"Unexpected verts_cam shape in {seq_path}: {verts_cam_all.shape}")
    if int(verts_cam_all.shape[0]) != len(frame_names):
        raise RuntimeError(
            f"Stage1 frame count mismatch in {seq_path}: "
            f"{verts_cam_all.shape[0]} verts frames vs {len(frame_names)} names"
        )
    return frame_names, verts_cam_all


def load_stage1_face_topology(stage1_dir: str | Path, frame_names: list[str]) -> np.ndarray:
    stage1_path = Path(stage1_dir)
    frame_stem = Path(frame_names[0]).stem if len(frame_names) > 0 else "000000"
    mesh_candidates = [
        stage1_path / "smpl_mesh_tpose.obj",
        stage1_path / "sequence" / f"{frame_stem}.obj",
        stage1_path / "sequence" / f"frame_{frame_stem}.obj",
        stage1_path / "final" / "last_frame_smplx_cam.obj",
        stage1_path / "final" / "last_frame_smplx.obj",
        stage1_path / "final" / "last_frame_pose_local.obj",
    ]
    mesh_path = next((p for p in mesh_candidates if p.exists()), None)
    if mesh_path is None:
        raise FileNotFoundError("Missing mesh topology under stage1 output.")
    _, faces = load_obj_mesh(mesh_path)
    if faces is None or len(faces) <= 0:
        raise RuntimeError(f"No faces found in mesh topology: {mesh_path}")
    return np.asarray(faces, dtype=np.int32)


def resolve_frame_image_path(frames_dir: str | Path, frame_name: str) -> Path:
    frames_path = Path(frames_dir)
    direct = frames_path / frame_name
    if direct.exists():
        return direct
    stem = Path(frame_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = frames_path / f"{stem}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Frame image not found: {frames_path / frame_name}")


def build_binary_support_mask(image_bgr_or_gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_bgr_or_gray)
    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return (np.any(arr[:, :, :3] < 250, axis=2)).astype(np.uint8)
    raise ValueError(f"Unsupported support image shape: {arr.shape}")


def estimate_frame_skin_reference_lab(
    seg: np.ndarray,
    rgb: np.ndarray,
    *,
    labels_to_ids: dict[str, int],
) -> np.ndarray | None:
    seg_i = np.asarray(seg, dtype=np.int32)
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    candidate_names = ("face", "hands", "arms")
    mask = np.zeros(seg_i.shape[:2], dtype=bool)
    for name in candidate_names:
        lid = labels_to_ids.get(name, None)
        if lid is None:
            continue
        mask |= seg_i == int(lid)
        if int(mask.sum()) >= 64:
            break
    if int(mask.sum()) < 32:
        return None
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.median(lab[mask], axis=0).astype(np.float32)


def classify_group_ids_with_color_support(
    raw_label_ids: np.ndarray,
    rgb_pixels: np.ndarray,
    *,
    raw_to_group_lut: np.ndarray,
    ids_to_labels: dict[int, str],
    skin_ref_lab: np.ndarray | None,
    skin_lab_distance_th: float = 26.0,
    ambiguous_label_names: tuple[str, ...] = ("legs", "feet"),
) -> np.ndarray:
    raw_ids = np.asarray(raw_label_ids, dtype=np.int64).reshape(-1)
    base_group_ids = np.full((raw_ids.shape[0],), fill_value=int(FASHN_COARSE_TO_ID["unknown"]), dtype=np.int64)
    valid_raw = (raw_ids >= 0) & (raw_ids < int(raw_to_group_lut.shape[0]))
    if bool(valid_raw.any()):
        base_group_ids[valid_raw] = raw_to_group_lut[raw_ids[valid_raw]].astype(np.int64)
    if skin_ref_lab is None or raw_ids.size <= 0:
        return base_group_ids
    amb_names = {str(name) for name in ambiguous_label_names}
    if len(amb_names) <= 0:
        return base_group_ids
    per_label_th = {
        "legs": max(float(skin_lab_distance_th) - 2.0, 24.0),
        "feet": max(float(skin_lab_distance_th) - 8.0, 18.0),
        "torso": max(float(skin_lab_distance_th) - 4.0, 22.0),
    }
    raw_names = [ids_to_labels.get(int(rid), "") for rid in raw_ids]
    amb_mask = np.asarray([name in amb_names for name in raw_names], dtype=bool)
    if not bool(amb_mask.any()):
        return base_group_ids
    rgb_u8 = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    amb_idx = np.flatnonzero(amb_mask)
    amb_names_list = [raw_names[int(i)] for i in amb_idx.tolist()]
    dist = np.linalg.norm(lab[amb_mask] - np.asarray(skin_ref_lab, dtype=np.float32).reshape(1, 3), axis=1)
    th = np.asarray([per_label_th.get(name, float(skin_lab_distance_th)) for name in amb_names_list], dtype=np.float32)
    covered = dist > th
    if bool(covered.any()):
        base_group_ids[amb_idx[covered]] = int(FASHN_COARSE_TO_ID["covered"])
    return base_group_ids


def resize_rgb_and_intrinsics(
    image_rgb: np.ndarray,
    K: np.ndarray,
    *,
    max_edge: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    K_scaled = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
    if int(max_edge) <= 0:
        return rgb, K_scaled
    h, w = rgb.shape[:2]
    src_max = max(int(h), int(w))
    if src_max <= int(max_edge):
        return rgb, K_scaled
    scale = float(max_edge) / float(src_max)
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    sx = float(new_w) / float(max(w, 1))
    sy = float(new_h) / float(max(h, 1))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return resized, K_scaled


def rasterize_visible_face_index_map(
    *,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    depth_eps: float = 1e-4,
    inside_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(image_hw[0]), int(image_hw[1])
    face_index_map = np.full((h, w), fill_value=-1, dtype=np.int32)
    depth_map = np.full((h, w), fill_value=np.inf, dtype=np.float32)

    uv, valid_v = project_vertices_to_image(verts_cam=np.asarray(verts_cam, dtype=np.float32), K=np.asarray(K, dtype=np.float32))
    faces_i = np.asarray(faces, dtype=np.int32)
    tri_uv = uv[faces_i]
    tri_z = np.asarray(verts_cam, dtype=np.float32)[faces_i, 2]
    valid_face = valid_v[faces_i].all(axis=1) & np.isfinite(tri_uv).all(axis=(1, 2)) & (tri_z > 1e-5).all(axis=1)
    if not bool(valid_face.any()):
        return face_index_map, depth_map

    x0 = tri_uv[:, 0, 0]
    y0 = tri_uv[:, 0, 1]
    x1 = tri_uv[:, 1, 0]
    y1 = tri_uv[:, 1, 1]
    x2 = tri_uv[:, 2, 0]
    y2 = tri_uv[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    valid_face &= np.abs(denom) > 1e-8

    xmin_all = np.floor(np.min(tri_uv[:, :, 0], axis=1)).astype(np.int32)
    xmax_all = np.ceil(np.max(tri_uv[:, :, 0], axis=1)).astype(np.int32)
    ymin_all = np.floor(np.min(tri_uv[:, :, 1], axis=1)).astype(np.int32)
    ymax_all = np.ceil(np.max(tri_uv[:, :, 1], axis=1)).astype(np.int32)
    valid_face &= (xmax_all >= 0) & (ymax_all >= 0) & (xmin_all < w) & (ymin_all < h)

    for face_idx in np.flatnonzero(valid_face):
        xmin = max(int(xmin_all[face_idx]), 0)
        xmax = min(int(xmax_all[face_idx]), w - 1)
        ymin = max(int(ymin_all[face_idx]), 0)
        ymax = min(int(ymax_all[face_idx]), h - 1)
        if xmax < xmin or ymax < ymin:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)

        den = float(denom[face_idx])
        w0 = ((y1[face_idx] - y2[face_idx]) * (xx - x2[face_idx]) + (x2[face_idx] - x1[face_idx]) * (yy - y2[face_idx])) / den
        w1 = ((y2[face_idx] - y0[face_idx]) * (xx - x2[face_idx]) + (x0[face_idx] - x2[face_idx]) * (yy - y2[face_idx])) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -float(inside_eps)) & (w1 >= -float(inside_eps)) & (w2 >= -float(inside_eps))
        if not bool(inside.any()):
            continue

        z0, z1, z2 = tri_z[face_idx]
        depth = (w0 * z0 + w1 * z1 + w2 * z2).astype(np.float32)
        region_depth = depth_map[ymin : ymax + 1, xmin : xmax + 1]
        region_face = face_index_map[ymin : ymax + 1, xmin : xmax + 1]
        update = inside & (depth < (region_depth - float(depth_eps)))
        if not bool(update.any()):
            continue
        region_depth[update] = depth[update]
        region_face[update] = int(face_idx)

    return face_index_map, depth_map


def _build_group_vote_counts(raw_vote_counts: np.ndarray, raw_to_group_lut: np.ndarray) -> np.ndarray:
    raw_counts = np.asarray(raw_vote_counts, dtype=np.int64)
    group_counts = np.zeros((raw_counts.shape[0], len(FASHN_COARSE_GROUPS)), dtype=np.int64)
    for raw_id in range(raw_counts.shape[1]):
        group_id = int(raw_to_group_lut[raw_id]) if raw_id < int(raw_to_group_lut.shape[0]) else int(len(FASHN_COARSE_GROUPS) - 1)
        group_counts[:, group_id] += raw_counts[:, raw_id]
    return group_counts


def _dominant_non_background_labels(
    raw_vote_counts: np.ndarray,
    group_vote_counts: np.ndarray,
    *,
    min_non_bg_votes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_counts = np.asarray(raw_vote_counts, dtype=np.int64)
    group_counts = np.asarray(group_vote_counts, dtype=np.int64)
    n_faces = int(raw_counts.shape[0])

    raw_label_id = np.full((n_faces,), fill_value=-1, dtype=np.int16)
    raw_conf = np.zeros((n_faces,), dtype=np.float32)
    group_id = np.full((n_faces,), fill_value=3, dtype=np.int16)
    group_conf = np.zeros((n_faces,), dtype=np.float32)

    if raw_counts.shape[1] > 1:
        raw_non_bg = raw_counts[:, 1:]
        raw_non_bg_sum = raw_non_bg.sum(axis=1)
        raw_arg = raw_non_bg.argmax(axis=1) + 1
        valid_raw = raw_non_bg_sum >= int(max(1, min_non_bg_votes))
        raw_label_id[valid_raw] = raw_arg[valid_raw].astype(np.int16)
        raw_conf[valid_raw] = (
            raw_counts[np.nonzero(valid_raw)[0], raw_arg[valid_raw]].astype(np.float32)
            / np.clip(raw_non_bg_sum[valid_raw].astype(np.float32), 1.0, None)
        )

    if group_counts.shape[1] > 2:
        group_non_bg = group_counts[:, 1:3]
        group_non_bg_sum = group_non_bg.sum(axis=1)
        group_arg = group_non_bg.argmax(axis=1) + 1
        valid_group = group_non_bg_sum >= int(max(1, min_non_bg_votes))
        group_id[valid_group] = group_arg[valid_group].astype(np.int16)
        group_conf[valid_group] = (
            group_counts[np.nonzero(valid_group)[0], group_arg[valid_group]].astype(np.float32)
            / np.clip(group_non_bg_sum[valid_group].astype(np.float32), 1.0, None)
        )

    return raw_label_id, raw_conf, group_id, group_conf


def build_face_adjacency_lists(faces: np.ndarray) -> list[np.ndarray]:
    faces_i = np.asarray(faces, dtype=np.int32)
    n_faces = int(faces_i.shape[0])
    vert_to_faces: dict[int, list[int]] = {}
    for face_idx, tri in enumerate(faces_i.tolist()):
        for vid in tri:
            vert_to_faces.setdefault(int(vid), []).append(int(face_idx))
    neigh_sets: list[set[int]] = [set() for _ in range(n_faces)]
    for face_idx, tri in enumerate(faces_i.tolist()):
        for vid in tri:
            neigh_sets[face_idx].update(vert_to_faces.get(int(vid), []))
        neigh_sets[face_idx].discard(int(face_idx))
    return [np.asarray(sorted(neigh), dtype=np.int32) for neigh in neigh_sets]


def propagate_group_labels_from_observed_raw_classes(
    raw_label_ids: np.ndarray,
    raw_confidences: np.ndarray,
    group_ids_observed: np.ndarray,
    group_confidences_observed: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    group_unknown_id: int,
    raw_to_group_lut: np.ndarray,
    conf_decay: float = 0.96,
    min_propagated_conf: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_ids = np.asarray(raw_label_ids, dtype=np.int16)
    raw_conf = np.asarray(raw_confidences, dtype=np.float32)
    group_ids = np.asarray(group_ids_observed, dtype=np.int16).copy()
    group_conf = np.asarray(group_confidences_observed, dtype=np.float32).copy()
    n_faces = int(raw_ids.shape[0])
    filled = 0

    valid_raw = np.flatnonzero(raw_ids >= 0)
    if valid_raw.size <= 0:
        return group_ids, group_conf, 0
    for rid in np.unique(raw_ids[valid_raw].astype(np.int32)):
        class_mask = raw_ids == int(rid)
        unknown_idx = np.flatnonzero(class_mask & (group_ids == int(group_unknown_id)))
        if unknown_idx.size <= 0:
            continue
        seed_idx = np.flatnonzero(class_mask & (group_ids != int(group_unknown_id)) & (group_conf > 0.0))
        default_gid = int(group_unknown_id)
        if int(rid) >= 0 and int(rid) < int(raw_to_group_lut.shape[0]):
            default_gid = int(raw_to_group_lut[int(rid)])
        if seed_idx.size <= 0:
            if default_gid != int(group_unknown_id):
                group_ids[unknown_idx] = np.int16(default_gid)
                group_conf[unknown_idx] = np.clip(raw_conf[unknown_idx], float(min_propagated_conf), 0.95)
                filled += int(unknown_idx.size)
            continue

        class_mask_bool = np.asarray(class_mask, dtype=bool)
        dist = np.full((n_faces,), fill_value=-1, dtype=np.int32)
        owner = np.full((n_faces,), fill_value=-1, dtype=np.int32)
        q: deque[int] = deque()
        for seed in seed_idx.tolist():
            seed_i = int(seed)
            dist[seed_i] = 0
            owner[seed_i] = seed_i
            q.append(seed_i)

        while q:
            cur = int(q.popleft())
            cur_owner = int(owner[cur])
            cur_dist = int(dist[cur])
            neigh = adjacency[cur] if cur < len(adjacency) else np.zeros((0,), dtype=np.int32)
            for nb in np.asarray(neigh, dtype=np.int32).tolist():
                nb_i = int(nb)
                if nb_i < 0 or nb_i >= n_faces or not bool(class_mask_bool[nb_i]):
                    continue
                cand_dist = int(cur_dist + 1)
                if dist[nb_i] < 0:
                    dist[nb_i] = cand_dist
                    owner[nb_i] = cur_owner
                    q.append(nb_i)
                elif cand_dist == int(dist[nb_i]) and float(group_conf[cur_owner]) > float(group_conf[int(owner[nb_i])]):
                    owner[nb_i] = cur_owner

        for face_idx in unknown_idx.tolist():
            fi = int(face_idx)
            src = int(owner[fi])
            if src >= 0:
                gid = int(group_ids[src])
                if gid != int(group_unknown_id):
                    hop = max(int(dist[fi]), 1)
                    group_ids[fi] = np.int16(gid)
                    group_conf[fi] = max(float(min_propagated_conf), float(group_conf[src]) * (float(conf_decay) ** float(max(hop - 1, 0))))
                    filled += 1
                    continue
            if default_gid != int(group_unknown_id):
                group_ids[fi] = np.int16(default_gid)
                group_conf[fi] = np.clip(float(raw_conf[fi]), float(min_propagated_conf), 0.95)
                filled += 1

    return group_ids, group_conf, int(filled)


def refine_raw_label_pair_by_local_votes(
    raw_label_ids: np.ndarray,
    raw_confidences: np.ndarray,
    raw_vote_counts: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    ids_to_labels: dict[int, str],
    pair_names: tuple[str, str] = ("face", "hair"),
    vote_ratio_th: float = 0.82,
    local_majority_ratio: float = 0.68,
    max_keep_conf: float = 0.72,
    min_neighbor_votes: int = 3,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.asarray(raw_label_ids, dtype=np.int16).copy()
    conf = np.asarray(raw_confidences, dtype=np.float32).copy()
    vote_counts = np.asarray(raw_vote_counts, dtype=np.int64)
    name_to_id = {str(v): int(k) for k, v in ids_to_labels.items()}
    if pair_names[0] not in name_to_id or pair_names[1] not in name_to_id:
        return labels, conf, 0
    a_id = int(name_to_id[pair_names[0]])
    b_id = int(name_to_id[pair_names[1]])
    pair_set = {a_id, b_id}
    changed_total = 0
    for _ in range(max(int(passes), 1)):
        changed = 0
        for face_idx in range(int(labels.shape[0])):
            cur = int(labels[face_idx])
            if cur not in pair_set:
                continue
            neigh = adjacency[face_idx] if face_idx < len(adjacency) else np.zeros((0,), dtype=np.int32)
            neigh = np.asarray(neigh, dtype=np.int32)
            if neigh.size <= 0:
                continue
            neigh = neigh[(neigh >= 0) & (neigh < int(labels.shape[0]))]
            if neigh.size <= 0:
                continue
            neigh_pair = neigh[np.isin(labels[neigh], list(pair_set))]
            if int(neigh_pair.size) < int(min_neighbor_votes):
                continue
            neigh_labels = labels[neigh_pair].astype(np.int32)
            neigh_conf = np.clip(conf[neigh_pair].astype(np.float32), 0.05, 1.0)
            w_a = float(neigh_conf[neigh_labels == a_id].sum())
            w_b = float(neigh_conf[neigh_labels == b_id].sum())
            total = float(w_a + w_b)
            if total <= 0.0:
                continue
            maj = a_id if w_a >= w_b else b_id
            maj_ratio = max(w_a, w_b) / total
            if maj == cur or maj_ratio < float(local_majority_ratio) or float(conf[face_idx]) > float(max_keep_conf):
                continue
            cur_votes = int(vote_counts[face_idx, cur]) if 0 <= cur < int(vote_counts.shape[1]) else 0
            alt_votes = int(vote_counts[face_idx, maj]) if 0 <= maj < int(vote_counts.shape[1]) else 0
            if alt_votes < float(vote_ratio_th) * float(max(cur_votes, 1)):
                continue
            labels[face_idx] = np.int16(maj)
            conf[face_idx] = max(float(conf[face_idx]), float(min(0.95, max(maj_ratio, alt_votes / float(max(cur_votes + alt_votes, 1))))))
            changed += 1
        changed_total += int(changed)
        if changed <= 0:
            break
    return labels, conf, int(changed_total)


def refine_face_hair_with_head_geometry(
    raw_label_ids: np.ndarray,
    raw_confidences: np.ndarray,
    raw_vote_counts: np.ndarray,
    face_centroids_ref: np.ndarray,
    *,
    ids_to_labels: dict[int, str],
    vote_ratio_th: float = 0.60,
    max_keep_conf: float = 0.82,
    x_pad: float = 0.020,
    y_pad_top: float = 0.020,
    y_pad_bottom: float = 0.018,
) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.asarray(raw_label_ids, dtype=np.int16).copy()
    conf = np.asarray(raw_confidences, dtype=np.float32).copy()
    votes = np.asarray(raw_vote_counts, dtype=np.int64)
    cent = np.asarray(face_centroids_ref, dtype=np.float32)
    if cent.ndim != 2 or cent.shape[0] != labels.shape[0] or cent.shape[1] != 3:
        return labels, conf, 0
    name_to_id = {str(v): int(k) for k, v in ids_to_labels.items()}
    if 'face' not in name_to_id or 'hair' not in name_to_id:
        return labels, conf, 0
    face_id = int(name_to_id['face'])
    hair_id = int(name_to_id['hair'])
    face_mask = labels == face_id
    if int(face_mask.sum()) < 64:
        return labels, conf, 0
    face_x = cent[face_mask, 0]
    face_y = cent[face_mask, 1]
    x_lo = float(np.quantile(face_x, 0.03) - float(x_pad))
    x_hi = float(np.quantile(face_x, 0.97) + float(x_pad))
    # In this camera convention, smaller y is higher on the head. Target the upper face/forehead band.
    y_lo = float(np.min(face_y) - float(y_pad_top))
    y_hi = float(np.quantile(face_y, 0.52) + float(y_pad_bottom))
    changed = 0
    hair_idx = np.flatnonzero(labels == hair_id)
    for face_idx in hair_idx.tolist():
        x = float(cent[face_idx, 0])
        y = float(cent[face_idx, 1])
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            continue
        hv = int(votes[face_idx, hair_id]) if hair_id < int(votes.shape[1]) else 0
        fv = int(votes[face_idx, face_id]) if face_id < int(votes.shape[1]) else 0
        if fv <= 0:
            continue
        if hv > 0 and float(fv) < float(vote_ratio_th) * float(hv):
            continue
        if float(conf[face_idx]) > float(max_keep_conf) and hv > fv:
            continue
        labels[face_idx] = np.int16(face_id)
        conf[face_idx] = max(float(conf[face_idx]), float(min(0.95, fv / float(max(fv + hv, 1)))))
        changed += 1
    return labels, conf, int(changed)


def smooth_binary_group_labels_within_raw_classes(
    raw_label_ids: np.ndarray,
    group_ids: np.ndarray,
    group_confidences: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    exposed_id: int,
    covered_id: int,
    majority_ratio: float = 0.74,
    max_keep_conf: float = 0.78,
    min_neighbor_votes: int = 3,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_ids = np.asarray(raw_label_ids, dtype=np.int16)
    groups = np.asarray(group_ids, dtype=np.int16).copy()
    conf = np.asarray(group_confidences, dtype=np.float32).copy()
    valid_group = {int(exposed_id), int(covered_id)}
    changed_total = 0
    for _ in range(max(int(passes), 1)):
        changed = 0
        for face_idx in range(int(groups.shape[0])):
            cur_g = int(groups[face_idx])
            if cur_g not in valid_group:
                continue
            cur_raw = int(raw_ids[face_idx])
            if cur_raw < 0:
                continue
            neigh = adjacency[face_idx] if face_idx < len(adjacency) else np.zeros((0,), dtype=np.int32)
            neigh = np.asarray(neigh, dtype=np.int32)
            if neigh.size <= 0:
                continue
            neigh = neigh[(neigh >= 0) & (neigh < int(groups.shape[0]))]
            if neigh.size <= 0:
                continue
            same_raw = neigh[raw_ids[neigh] == cur_raw]
            same_raw = same_raw[np.isin(groups[same_raw], list(valid_group))]
            if int(same_raw.size) < int(min_neighbor_votes):
                continue
            neigh_groups = groups[same_raw].astype(np.int32)
            neigh_conf = np.clip(conf[same_raw].astype(np.float32), 0.05, 1.0)
            w_exp = float(neigh_conf[neigh_groups == int(exposed_id)].sum())
            w_cov = float(neigh_conf[neigh_groups == int(covered_id)].sum())
            total = float(w_exp + w_cov)
            if total <= 0.0:
                continue
            maj = int(exposed_id) if w_exp >= w_cov else int(covered_id)
            maj_ratio = max(w_exp, w_cov) / total
            if maj == cur_g or maj_ratio < float(majority_ratio) or float(conf[face_idx]) > float(max_keep_conf):
                continue
            groups[face_idx] = np.int16(maj)
            conf[face_idx] = max(float(conf[face_idx]), float(min(0.95, maj_ratio)))
            changed += 1
        changed_total += int(changed)
        if changed <= 0:
            break
    return groups, conf, int(changed_total)


def propagate_unknown_face_labels(
    label_ids: np.ndarray,
    confidences: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    unknown_ids: tuple[int, ...],
    visible_frame_counts: np.ndarray | None = None,
    min_seed_conf: float = 1e-4,
    conf_decay: float = 0.90,
    min_propagated_conf: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.asarray(label_ids).astype(np.int16, copy=True)
    conf = np.asarray(confidences, dtype=np.float32).copy()
    unknown_set = {int(x) for x in unknown_ids}
    unknown_mask = np.asarray([int(v) in unknown_set for v in labels.tolist()], dtype=bool)
    if visible_frame_counts is not None:
        unknown_mask &= np.asarray(visible_frame_counts, dtype=np.int32).reshape(-1) <= 0
    if not bool(unknown_mask.any()):
        return labels, conf, 0

    n_faces = int(labels.shape[0])
    visited = np.zeros((n_faces,), dtype=bool)
    filled_total = 0

    for start in np.flatnonzero(unknown_mask):
        if visited[int(start)]:
            continue
        comp: list[int] = []
        stack = [int(start)]
        visited[int(start)] = True
        while stack:
            cur = int(stack.pop())
            comp.append(cur)
            neigh = adjacency[cur] if cur < len(adjacency) else np.zeros((0,), dtype=np.int32)
            for nb in np.asarray(neigh, dtype=np.int32).tolist():
                nb_i = int(nb)
                if nb_i < 0 or nb_i >= n_faces or visited[nb_i] or not bool(unknown_mask[nb_i]):
                    continue
                visited[nb_i] = True
                stack.append(nb_i)
        if len(comp) <= 0:
            continue

        comp_set = set(comp)
        seed_faces: set[int] = set()
        for face_idx in comp:
            neigh = adjacency[face_idx] if face_idx < len(adjacency) else np.zeros((0,), dtype=np.int32)
            for nb in np.asarray(neigh, dtype=np.int32).tolist():
                nb_i = int(nb)
                if nb_i in comp_set or nb_i < 0 or nb_i >= n_faces:
                    continue
                if int(labels[nb_i]) in unknown_set:
                    continue
                if float(conf[nb_i]) < float(min_seed_conf):
                    continue
                seed_faces.add(nb_i)
        if len(seed_faces) <= 0:
            continue

        seeds_sorted = sorted(seed_faces, key=lambda fi: (-float(conf[int(fi)]), int(fi)))
        comp_mask = np.zeros((n_faces,), dtype=bool)
        comp_mask[np.asarray(comp, dtype=np.int32)] = True
        owner = np.full((n_faces,), fill_value=-1, dtype=np.int32)
        dist = np.full((n_faces,), fill_value=-1, dtype=np.int32)
        q: deque[int] = deque()

        for seed in seeds_sorted:
            neigh = adjacency[int(seed)] if int(seed) < len(adjacency) else np.zeros((0,), dtype=np.int32)
            for nb in np.asarray(neigh, dtype=np.int32).tolist():
                nb_i = int(nb)
                if nb_i < 0 or nb_i >= n_faces or not bool(comp_mask[nb_i]):
                    continue
                if dist[nb_i] < 0:
                    dist[nb_i] = 1
                    owner[nb_i] = int(seed)
                    q.append(nb_i)
                elif dist[nb_i] == 1 and float(conf[int(seed)]) > float(conf[int(owner[nb_i])]):
                    owner[nb_i] = int(seed)

        while q:
            cur = int(q.popleft())
            cur_owner = int(owner[cur])
            cur_dist = int(dist[cur])
            neigh = adjacency[cur] if cur < len(adjacency) else np.zeros((0,), dtype=np.int32)
            for nb in np.asarray(neigh, dtype=np.int32).tolist():
                nb_i = int(nb)
                if nb_i < 0 or nb_i >= n_faces or not bool(comp_mask[nb_i]):
                    continue
                cand_dist = int(cur_dist + 1)
                if dist[nb_i] < 0:
                    dist[nb_i] = cand_dist
                    owner[nb_i] = cur_owner
                    q.append(nb_i)
                elif cand_dist == int(dist[nb_i]) and float(conf[cur_owner]) > float(conf[int(owner[nb_i])]):
                    owner[nb_i] = cur_owner

        for face_idx in comp:
            src = int(owner[face_idx])
            if src < 0:
                continue
            hop = max(int(dist[face_idx]), 1)
            labels[face_idx] = labels[src]
            conf[face_idx] = max(float(min_propagated_conf), float(conf[src]) * (float(conf_decay) ** float(max(hop - 1, 0))))
            filled_total += 1

    return labels, conf, int(filled_total)


def _label_name_from_id(label_id: int, ids_to_names: dict[int, str], unknown_name: str = "unknown") -> str:
    if int(label_id) < 0:
        return unknown_name
    return str(ids_to_names.get(int(label_id), unknown_name))


def _write_face_semantics_csv(
    path: Path,
    *,
    raw_label_id: np.ndarray,
    raw_conf: np.ndarray,
    group_id: np.ndarray,
    group_conf: np.ndarray,
    visible_px_counts: np.ndarray,
    non_bg_px_counts: np.ndarray,
    visible_frame_counts: np.ndarray,
    exclude_outer_mask: np.ndarray,
    ids_to_labels: dict[int, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "face_idx",
                "raw_label_id",
                "raw_label_name",
                "raw_confidence",
                "group_id",
                "group_name",
                "group_confidence",
                "visible_px_count",
                "non_bg_px_count",
                "visible_frame_count",
                "exclude_outer",
            ]
        )
        for face_idx in range(int(raw_label_id.shape[0])):
            gid = int(group_id[face_idx])
            group_name = FASHN_COARSE_GROUPS[gid] if 0 <= gid < len(FASHN_COARSE_GROUPS) else "unknown"
            writer.writerow(
                [
                    int(face_idx),
                    int(raw_label_id[face_idx]),
                    _label_name_from_id(int(raw_label_id[face_idx]), ids_to_labels),
                    float(raw_conf[face_idx]),
                    gid,
                    group_name,
                    float(group_conf[face_idx]),
                    int(visible_px_counts[face_idx]),
                    int(non_bg_px_counts[face_idx]),
                    int(visible_frame_counts[face_idx]),
                    int(exclude_outer_mask[face_idx]),
                ]
            )


def _face_count_summary_by_name(label_ids: np.ndarray, id_to_name: dict[int, str], *, unknown_name: str = "unknown") -> dict[str, int]:
    counts: dict[str, int] = {}
    for label_id in np.asarray(label_ids, dtype=np.int16).tolist():
        name = _label_name_from_id(int(label_id), id_to_name, unknown_name=unknown_name)
        counts[name] = int(counts.get(name, 0) + 1)
    return counts


def build_stage1_fashn_face_semantics(
    *,
    stage1_dir: str | Path,
    frames_dir: str | Path,
    out_dir: str | Path,
    support_frames_dir: str | Path | None = None,
    device: str = "cuda",
    model_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    max_edge: int = 640,
    frame_stride: int = 1,
    max_frames: int = 0,
    min_non_bg_votes: int = 8,
) -> dict[str, object]:
    stage1_path = Path(stage1_dir).resolve()
    frames_path = Path(frames_dir).resolve()
    out_path = Path(out_dir).resolve()
    support_frames_path = None if support_frames_dir is None else Path(support_frames_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    eval_out_dir = stage1_path.parent.parent / "evaluation" / "output_images" / "fashn_face_semantics"

    from evaluation.fashn_face_semantics.export_artifacts import (
        save_face_semantics_previews as export_face_semantics_previews,
        save_fashn_2d_preview,
    )

    frame_names, verts_cam_all = load_stage1_vertices_sequence(stage1_path)
    faces = load_stage1_face_topology(stage1_path, frame_names)
    n_faces = int(faces.shape[0])

    ids_to_labels, labels_to_ids = get_fashn_label_maps(runtime_dir=runtime_dir)
    raw_to_group_lut = build_fashn_raw_to_coarse_id(ids_to_labels)
    device_norm = normalize_fashn_device(device)
    parser = load_fashn_parser(
        model_dir=model_dir,
        runtime_dir=runtime_dir,
        device=device_norm,
    )

    max_raw_id = max(ids_to_labels.keys()) if ids_to_labels else -1
    n_raw = int(max_raw_id + 1)
    raw_vote_counts = np.zeros((n_faces, n_raw), dtype=np.int64)
    group_vote_counts = np.zeros((n_faces, len(FASHN_COARSE_GROUPS)), dtype=np.int64)
    visible_px_counts = np.zeros((n_faces,), dtype=np.int64)
    non_bg_px_counts = np.zeros((n_faces,), dtype=np.int64)
    visible_frame_counts = np.zeros((n_faces,), dtype=np.int32)

    frame_step = max(int(frame_stride), 1)
    selected_indices = list(range(0, int(len(frame_names)), frame_step))
    if int(max_frames) > 0:
        selected_indices = selected_indices[: int(max_frames)]
    if len(selected_indices) <= 0:
        raise RuntimeError("No frames selected for FASHN->SMPL face semantics.")

    print(
        f"[FASHN->SMPL] stage1={stage1_path} | requested_frames={len(selected_indices)}/{len(frame_names)} | "
        f"faces={n_faces} | max_edge={int(max_edge)} | device={device_norm}"
    )
    processed_indices: list[int] = []
    skipped_missing_frames = 0
    skipped_missing_support_frames = 0
    fashn_2d_preview_paths: list[str] = []
    for rank, frame_idx in enumerate(selected_indices, start=1):
        frame_name = str(frame_names[int(frame_idx)])
        try:
            image_path = resolve_frame_image_path(frames_path, frame_name)
        except FileNotFoundError:
            skipped_missing_frames += 1
            print(
                f"[FASHN->SMPL] skip missing frame image | "
                f"idx={int(frame_idx):03d} | name={frame_name}"
            )
            continue
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read frame image: {image_path}")
        support_mask_small = None
        if support_frames_path is not None:
            try:
                support_image_path = resolve_frame_image_path(support_frames_path, frame_name)
            except FileNotFoundError:
                skipped_missing_support_frames += 1
                print(
                    f"[FASHN->SMPL] skip missing support frame image | "
                    f"idx={int(frame_idx):03d} | name={frame_name}"
                )
                continue
            support_img = cv2.imread(str(support_image_path), cv2.IMREAD_UNCHANGED)
            if support_img is None:
                skipped_missing_support_frames += 1
                print(
                    f"[FASHN->SMPL] skip unreadable support frame image | "
                    f"idx={int(frame_idx):03d} | name={frame_name}"
                )
                continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = read_stage1_frame_camera_K(
            stage1_dir=stage1_path,
            frame_index=int(frame_idx),
            frame_name=frame_name,
            cam_json_path=stage1_path / "cameras.json",
            image_hw=rgb.shape[:2],
        )
        rgb_small, K_small = resize_rgb_and_intrinsics(rgb, K, max_edge=int(max_edge))
        if support_frames_path is not None:
            support_mask = build_binary_support_mask(support_img)
            support_mask_small, _ = resize_rgb_and_intrinsics((support_mask * 255).astype(np.uint8), K, max_edge=int(max_edge))
            support_mask_small = (np.asarray(support_mask_small, dtype=np.uint8) > 0).astype(np.uint8)
        seg = predict_fashn_segmentation(
            rgb_small,
            parser=parser,
            model_dir=model_dir,
            runtime_dir=runtime_dir,
            device=device_norm,
        )
        skin_ref_lab = estimate_frame_skin_reference_lab(seg, rgb_small, labels_to_ids=labels_to_ids)
        if seg.shape[:2] != rgb_small.shape[:2]:
            seg = cv2.resize(seg.astype(np.uint8), (rgb_small.shape[1], rgb_small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        if support_mask_small is not None:
            if support_mask_small.shape[:2] != seg.shape[:2]:
                support_mask_small = cv2.resize(
                    (support_mask_small.astype(np.uint8) * 255),
                    (seg.shape[1], seg.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                support_mask_small = (np.asarray(support_mask_small, dtype=np.uint8) > 0).astype(np.uint8)
            seg = np.where(support_mask_small > 0, seg, 0).astype(np.int32)
        fashn_2d_preview_paths.append(
            save_fashn_2d_preview(
                out_dir=eval_out_dir / "fashn_2d_labels",
                frame_name=frame_name,
                rgb=rgb_small,
                seg=seg,
            )
        )
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=np.asarray(verts_cam_all[int(frame_idx)], dtype=np.float32),
            faces=faces,
            K=K_small,
            image_hw=seg.shape[:2],
        )
        valid_pix = face_index_map >= 0
        if bool(valid_pix.any()):
            face_ids = face_index_map[valid_pix].astype(np.int64)
            label_ids = np.asarray(seg[valid_pix], dtype=np.int64)
            keep = (label_ids >= 0) & (label_ids < n_raw)
            if bool(keep.any()):
                face_ids = face_ids[keep]
                label_ids = label_ids[keep]
                combined = face_ids * int(n_raw) + label_ids
                binc = np.bincount(combined, minlength=int(n_faces * n_raw)).reshape(n_faces, n_raw)
                raw_vote_counts += binc
                group_pixel_ids = classify_group_ids_with_color_support(
                    label_ids,
                    rgb_small[valid_pix][keep],
                    raw_to_group_lut=raw_to_group_lut,
                    ids_to_labels=ids_to_labels,
                    skin_ref_lab=skin_ref_lab,
                    skin_lab_distance_th=26.0,
                    ambiguous_label_names=("legs", "feet"),
                ).astype(np.int64)
                group_combined = face_ids * int(len(FASHN_COARSE_GROUPS)) + group_pixel_ids.astype(np.int64)
                group_binc = np.bincount(group_combined, minlength=int(n_faces * len(FASHN_COARSE_GROUPS))).reshape(n_faces, len(FASHN_COARSE_GROUPS))
                group_vote_counts += group_binc
                visible_px_counts += np.bincount(face_ids, minlength=n_faces).astype(np.int64)
                non_bg_face_ids = face_ids[label_ids > 0]
                if int(non_bg_face_ids.size) > 0:
                    non_bg_px_counts += np.bincount(non_bg_face_ids, minlength=n_faces).astype(np.int64)
                visible_frame_counts[np.unique(face_ids)] += 1
        processed_indices.append(int(frame_idx))
        print(
            f"[FASHN->SMPL] frame {rank:03d}/{len(selected_indices):03d} | "
            f"idx={int(frame_idx):03d} | name={frame_name}"
        )

    if len(processed_indices) <= 0:
        raise RuntimeError(
            f"No usable frame images found under {frames_path} for FASHN->SMPL face semantics."
        )

    face_centroids_ref = np.asarray(
        verts_cam_all[np.asarray(processed_indices, dtype=np.int32)][:, faces].mean(axis=2).mean(axis=0),
        dtype=np.float32,
    )

    raw_label_id, raw_conf, group_id_observed, group_conf_observed = _dominant_non_background_labels(
        raw_vote_counts,
        group_vote_counts,
        min_non_bg_votes=int(min_non_bg_votes),
    )
    raw_unknown_before = int(np.sum(np.asarray(raw_label_id, dtype=np.int16) < 0))
    group_unknown_id = int(FASHN_COARSE_TO_ID["unknown"])
    group_unknown_before = int(np.sum(np.asarray(group_id_observed, dtype=np.int16) == group_unknown_id))
    face_adjacency = build_face_adjacency_lists(faces)
    raw_label_id, raw_conf, raw_face_hair_refined_count = refine_raw_label_pair_by_local_votes(
        raw_label_id,
        raw_conf,
        raw_vote_counts,
        face_adjacency,
        ids_to_labels=ids_to_labels,
        pair_names=("face", "hair"),
        vote_ratio_th=0.72,
        local_majority_ratio=0.60,
        max_keep_conf=0.82,
        min_neighbor_votes=2,
        passes=3,
    )
    raw_label_id, raw_conf, raw_face_hair_geometry_refined_count = refine_face_hair_with_head_geometry(
        raw_label_id,
        raw_conf,
        raw_vote_counts,
        face_centroids_ref,
        ids_to_labels=ids_to_labels,
        vote_ratio_th=0.55,
        max_keep_conf=0.82,
    )
    raw_label_id, raw_conf, raw_filled_by_propagation = propagate_unknown_face_labels(
        raw_label_id,
        raw_conf,
        face_adjacency,
        unknown_ids=(-1,),
        visible_frame_counts=visible_frame_counts,
        conf_decay=0.92,
        min_propagated_conf=0.08,
    )
    raw_unknown_after = int(np.sum(np.asarray(raw_label_id, dtype=np.int16) < 0))

    group_id, group_conf, group_filled_by_propagation = propagate_group_labels_from_observed_raw_classes(
        raw_label_id,
        raw_conf,
        group_id_observed,
        group_conf_observed,
        face_adjacency,
        group_unknown_id=group_unknown_id,
        raw_to_group_lut=raw_to_group_lut,
        conf_decay=0.97,
        min_propagated_conf=0.08,
    )
    group_id, group_conf, group_binary_smooth_count = smooth_binary_group_labels_within_raw_classes(
        raw_label_id,
        group_id,
        group_conf,
        face_adjacency,
        exposed_id=int(FASHN_COARSE_TO_ID["exposed"]),
        covered_id=int(FASHN_COARSE_TO_ID["covered"]),
        majority_ratio=0.74,
        max_keep_conf=0.78,
        min_neighbor_votes=2,
        passes=3,
    )
    group_unknown_after = int(np.sum(np.asarray(group_id, dtype=np.int16) == group_unknown_id))
    exclude_outer_mask = (group_id == int(FASHN_COARSE_TO_ID["exposed"])).astype(np.uint8)

    raw_name_array = np.asarray([ids_to_labels.get(i, f"id_{i}") for i in range(n_raw)])
    group_name_array = np.asarray(list(FASHN_COARSE_GROUPS))
    selected_frame_names = np.asarray([str(frame_names[i]) for i in processed_indices])

    out_npz = out_path / "smpl_face_semantics.npz"
    np.savez_compressed(
        out_npz,
        faces=np.asarray(faces, dtype=np.int32),
        raw_label_ids=np.asarray(raw_label_id, dtype=np.int16),
        raw_label_confidence=np.asarray(raw_conf, dtype=np.float32),
        raw_vote_counts=np.asarray(raw_vote_counts, dtype=np.int64),
        group_ids=np.asarray(group_id, dtype=np.int16),
        group_confidence=np.asarray(group_conf, dtype=np.float32),
        group_vote_counts=np.asarray(group_vote_counts, dtype=np.int64),
        visible_px_counts=np.asarray(visible_px_counts, dtype=np.int64),
        non_bg_px_counts=np.asarray(non_bg_px_counts, dtype=np.int64),
        visible_frame_counts=np.asarray(visible_frame_counts, dtype=np.int32),
        exclude_outer_mask=np.asarray(exclude_outer_mask, dtype=np.uint8),
        raw_label_names=raw_name_array,
        group_names=group_name_array,
        frame_names=selected_frame_names,
    )

    csv_path = out_path / "smpl_face_semantics.csv"
    _write_face_semantics_csv(
        csv_path,
        raw_label_id=raw_label_id,
        raw_conf=raw_conf,
        group_id=group_id,
        group_conf=group_conf,
        visible_px_counts=visible_px_counts,
        non_bg_px_counts=non_bg_px_counts,
        visible_frame_counts=visible_frame_counts,
        exclude_outer_mask=exclude_outer_mask,
        ids_to_labels=ids_to_labels,
    )

    preview_indices = sorted(set(int(x) for x in processed_indices))
    preview_paths = export_face_semantics_previews(
        out_dir=eval_out_dir,
        preview_indices=preview_indices,
        frame_names=frame_names,
        verts_cam_all=verts_cam_all,
        faces=faces,
        frames_dir=frames_path,
        stage1_dir=stage1_path,
        raw_label_id=raw_label_id,
        group_id=group_id,
        max_edge=int(max_edge),
    )

    group_id_to_name = {idx: name for idx, name in enumerate(FASHN_COARSE_GROUPS)}
    summary = {
        "stage1_dir": str(stage1_path),
        "frames_dir": str(frames_path),
        "support_frames_dir": None if support_frames_path is None else str(support_frames_path),
        "out_npz": str(out_npz),
        "out_csv": str(csv_path),
        "evaluation_output_dir": str(eval_out_dir),
        "evaluation_preview_images": preview_paths,
        "evaluation_fashn_2d_dir": str(eval_out_dir / "fashn_2d_labels"),
        "evaluation_fashn_2d_images": fashn_2d_preview_paths,
        "evaluation_raw_dir": str(eval_out_dir / "raw_face_labels"),
        "evaluation_group_dir": str(eval_out_dir / "group_face_labels"),
        "n_faces": int(n_faces),
        "n_frames_total": int(len(frame_names)),
        "n_frames_requested": int(len(selected_indices)),
        "n_frames_used": int(len(processed_indices)),
        "n_frames_skipped_missing": int(skipped_missing_frames),
        "n_frames_skipped_missing_support": int(skipped_missing_support_frames),
        "frame_stride": int(frame_step),
        "max_edge": int(max_edge),
        "device": str(device_norm),
        "min_non_bg_votes": int(min_non_bg_votes),
        "fashn_model_dir": str(resolve_fashn_model_dir(model_dir)),
        "fashn_runtime_dir": str(resolve_fashn_runtime_dir(runtime_dir)),
        "raw_face_counts": _face_count_summary_by_name(raw_label_id, ids_to_labels),
        "group_face_counts": _face_count_summary_by_name(group_id, group_id_to_name),
        "exclude_outer_faces": int(exclude_outer_mask.sum()),
        "observed_faces": int((visible_frame_counts > 0).sum()),
        "non_bg_observed_faces": int((non_bg_px_counts >= int(max(1, min_non_bg_votes))).sum()),
        "raw_unknown_before_propagation": int(raw_unknown_before),
        "raw_unknown_after_propagation": int(raw_unknown_after),
        "raw_face_hair_refined_count": int(raw_face_hair_refined_count),
        "raw_face_hair_geometry_refined_count": int(raw_face_hair_geometry_refined_count),
        "raw_filled_by_propagation": int(raw_filled_by_propagation),
        "group_unknown_before_propagation": int(group_unknown_before),
        "group_unknown_after_propagation": int(group_unknown_after),
        "group_filled_by_propagation": int(group_filled_by_propagation),
        "group_binary_smooth_count": int(group_binary_smooth_count),
    }
    summary_path = out_path / "smpl_face_semantics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
