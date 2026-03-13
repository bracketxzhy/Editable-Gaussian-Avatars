from __future__ import annotations

import csv
import json
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
    build_fashn_raw_to_coarse_id,
    get_fashn_label_maps,
    load_fashn_parser,
    normalize_fashn_device,
    predict_fashn_segmentation,
    resolve_fashn_model_dir,
    resolve_fashn_runtime_dir,
)


RAW_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [220, 20, 60],
        [255, 140, 0],
        [65, 105, 225],
        [186, 85, 211],
        [255, 215, 0],
        [46, 139, 87],
        [255, 99, 71],
        [112, 128, 144],
        [160, 82, 45],
        [72, 61, 139],
        [70, 130, 180],
        [154, 205, 50],
        [233, 150, 122],
        [255, 182, 193],
        [127, 255, 212],
        [210, 180, 140],
        [255, 250, 205],
    ],
    dtype=np.uint8,
)

GROUP_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [244, 92, 92],
        [52, 101, 164],
        [160, 160, 160],
    ],
    dtype=np.uint8,
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


def _label_name_from_id(label_id: int, ids_to_names: dict[int, str], unknown_name: str = "unknown") -> str:
    if int(label_id) < 0:
        return unknown_name
    return str(ids_to_names.get(int(label_id), unknown_name))


def _save_overlay(path: Path, rgb: np.ndarray, color_map: np.ndarray) -> None:
    overlay = (0.45 * rgb.astype(np.float32) + 0.55 * color_map.astype(np.float32)).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _render_face_label_color_map(
    face_index_map: np.ndarray,
    label_ids: np.ndarray,
    palette: np.ndarray,
    *,
    unknown_color: tuple[int, int, int] = (160, 160, 160),
) -> np.ndarray:
    h, w = face_index_map.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = face_index_map >= 0
    if not bool(valid.any()):
        return color
    face_ids = face_index_map[valid].astype(np.int64)
    labels = np.asarray(label_ids, dtype=np.int16)[face_ids]
    colors = np.full((labels.shape[0], 3), fill_value=np.asarray(unknown_color, dtype=np.uint8), dtype=np.uint8)
    known = (labels >= 0) & (labels < int(palette.shape[0]))
    if bool(known.any()):
        colors[known] = palette[labels[known].astype(np.int64)]
    color[valid] = colors
    return color


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


def save_face_semantics_previews(
    *,
    out_dir: str | Path,
    preview_indices: list[int],
    frame_names: list[str],
    verts_cam_all: np.ndarray,
    faces: np.ndarray,
    frames_dir: str | Path,
    stage1_dir: str | Path,
    raw_label_id: np.ndarray,
    group_id: np.ndarray,
    max_edge: int,
) -> list[str]:
    out_path = Path(out_dir)
    preview_paths: list[str] = []
    for idx in preview_indices:
        frame_name = str(frame_names[int(idx)])
        image_path = resolve_frame_image_path(frames_dir, frame_name)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = read_stage1_frame_camera_K(
            stage1_dir=Path(stage1_dir),
            frame_index=int(idx),
            frame_name=frame_name,
            cam_json_path=Path(stage1_dir) / "cameras.json",
            image_hw=rgb.shape[:2],
        )
        rgb_small, K_small = resize_rgb_and_intrinsics(rgb, K, max_edge=int(max_edge))
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=np.asarray(verts_cam_all[int(idx)], dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            K=np.asarray(K_small, dtype=np.float32),
            image_hw=rgb_small.shape[:2],
        )
        raw_color = _render_face_label_color_map(face_index_map, raw_label_id, RAW_PALETTE)
        group_color = _render_face_label_color_map(face_index_map, group_id, GROUP_PALETTE)
        stem = Path(frame_name).stem
        raw_path = out_path / f"{stem}_raw_face_labels.png"
        group_path = out_path / f"{stem}_group_face_labels.png"
        _save_overlay(raw_path, rgb_small, raw_color)
        _save_overlay(group_path, rgb_small, group_color)
        preview_paths.extend([str(raw_path), str(group_path)])
    return preview_paths


def build_stage1_fashn_face_semantics(
    *,
    stage1_dir: str | Path,
    frames_dir: str | Path,
    out_dir: str | Path,
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
    out_path.mkdir(parents=True, exist_ok=True)

    frame_names, verts_cam_all = load_stage1_vertices_sequence(stage1_path)
    faces = load_stage1_face_topology(stage1_path, frame_names)
    n_faces = int(faces.shape[0])

    ids_to_labels, _ = get_fashn_label_maps(runtime_dir=runtime_dir)
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
        f"[FASHN->SMPL] stage1={stage1_path} | frames={len(selected_indices)}/{len(frame_names)} | "
        f"faces={n_faces} | max_edge={int(max_edge)} | device={device_norm}"
    )
    for rank, frame_idx in enumerate(selected_indices, start=1):
        frame_name = str(frame_names[int(frame_idx)])
        image_path = resolve_frame_image_path(frames_path, frame_name)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read frame image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = read_stage1_frame_camera_K(
            stage1_dir=stage1_path,
            frame_index=int(frame_idx),
            frame_name=frame_name,
            cam_json_path=stage1_path / "cameras.json",
            image_hw=rgb.shape[:2],
        )
        rgb_small, K_small = resize_rgb_and_intrinsics(rgb, K, max_edge=int(max_edge))
        seg = predict_fashn_segmentation(
            rgb_small,
            parser=parser,
            model_dir=model_dir,
            runtime_dir=runtime_dir,
            device=device_norm,
        )
        if seg.shape[:2] != rgb_small.shape[:2]:
            seg = cv2.resize(seg.astype(np.uint8), (rgb_small.shape[1], rgb_small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
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
                visible_px_counts += np.bincount(face_ids, minlength=n_faces).astype(np.int64)
                non_bg_face_ids = face_ids[label_ids > 0]
                if int(non_bg_face_ids.size) > 0:
                    non_bg_px_counts += np.bincount(non_bg_face_ids, minlength=n_faces).astype(np.int64)
                visible_frame_counts[np.unique(face_ids)] += 1
        print(
            f"[FASHN->SMPL] frame {rank:03d}/{len(selected_indices):03d} | "
            f"idx={int(frame_idx):03d} | name={frame_name}"
        )

    group_vote_counts = _build_group_vote_counts(raw_vote_counts, raw_to_group_lut)
    raw_label_id, raw_conf, group_id, group_conf = _dominant_non_background_labels(
        raw_vote_counts,
        group_vote_counts,
        min_non_bg_votes=int(min_non_bg_votes),
    )
    exclude_outer_mask = (group_id == 1).astype(np.uint8)

    raw_name_array = np.asarray([ids_to_labels.get(i, f"id_{i}") for i in range(n_raw)])
    group_name_array = np.asarray(list(FASHN_COARSE_GROUPS))
    selected_frame_names = np.asarray([str(frame_names[i]) for i in selected_indices])

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

    preview_indices = sorted(set([selected_indices[0], selected_indices[-1]]))
    preview_paths = save_face_semantics_previews(
        out_dir=out_path,
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
        "out_npz": str(out_npz),
        "out_csv": str(csv_path),
        "preview_images": preview_paths,
        "n_faces": int(n_faces),
        "n_frames_total": int(len(frame_names)),
        "n_frames_used": int(len(selected_indices)),
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
    }
    summary_path = out_path / "smpl_face_semantics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
