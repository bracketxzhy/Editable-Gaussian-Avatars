from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avatar_gs.alignment.mask_sources import read_stage1_frame_camera_K
from avatar_gs.segmentation.smpl_face_semantics import (
    rasterize_visible_face_index_map,
    resize_rgb_and_intrinsics,
    resolve_frame_image_path,
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


def _save_overlay(path: Path, rgb: np.ndarray, color_map: np.ndarray) -> None:
    overlay = (0.45 * rgb.astype(np.float32) + 0.55 * color_map.astype(np.float32)).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _render_pixel_label_color_map(
    label_map: np.ndarray,
    palette: np.ndarray,
    *,
    unknown_color: tuple[int, int, int] = (160, 160, 160),
) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    color = np.full(labels.shape + (3,), fill_value=np.asarray(unknown_color, dtype=np.uint8), dtype=np.uint8)
    valid = (labels >= 0) & (labels < int(palette.shape[0]))
    if bool(valid.any()):
        color[valid] = palette[labels[valid].astype(np.int64)]
    return color


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


def save_fashn_2d_preview(
    *,
    out_dir: str | Path,
    frame_name: str,
    rgb: np.ndarray,
    seg: np.ndarray,
) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(frame_name).stem
    seg_color = _render_pixel_label_color_map(seg, RAW_PALETTE)
    save_path = out_path / f"{stem}_fashn_2d_labels.png"
    _save_overlay(save_path, np.asarray(rgb, dtype=np.uint8), seg_color)
    return str(save_path)


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
    raw_dir = out_path / "raw_face_labels"
    group_dir = out_path / "group_face_labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
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
        raw_path = raw_dir / f"{stem}_raw_face_labels.png"
        group_path = group_dir / f"{stem}_group_face_labels.png"
        _save_overlay(raw_path, rgb_small, raw_color)
        _save_overlay(group_path, rgb_small, group_color)
        preview_paths.extend([str(raw_path), str(group_path)])
    return preview_paths
