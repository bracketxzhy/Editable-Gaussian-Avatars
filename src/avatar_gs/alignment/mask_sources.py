from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from avatar_gs.alignment import mask_iou
from avatar_gs.segmentation.modnet import estimate_person_mask_modnet


def project_vertices_to_image(verts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(np.asarray(verts_cam[:, 2], dtype=np.float32), 1e-6, None)
    u = float(K[0, 0]) * (verts_cam[:, 0] / z) + float(K[0, 2])
    v = float(K[1, 1]) * (verts_cam[:, 1] / z) + float(K[1, 2])
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & (np.asarray(verts_cam[:, 2], dtype=np.float32) > 1e-5)
    return uv, valid


def rasterize_uv_mask(
    uv: np.ndarray,
    valid_v: np.ndarray,
    faces: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
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
        cv2.fillConvexPoly(mask, pts, color=1, lineType=cv2.LINE_8)
    return mask


def rasterize_mesh_mask(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    *,
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    uv, valid_v = project_vertices_to_image(verts_cam=verts_cam, K=K)
    mask = rasterize_uv_mask(uv=uv, valid_v=valid_v, faces=faces, image_hw=image_hw)
    if int(dilate_px) > 0:
        k = np.ones((2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    if int(erode_px) > 0:
        k = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), dtype=np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    return (mask > 0).astype(np.uint8)


def read_stage1_frame_camera_K(
    *,
    stage1_dir: Path,
    frame_index: int,
    frame_name: str,
    fallback_K: np.ndarray | None = None,
    cam_json_path: Path | None = None,
    image_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    params_npz = stage1_dir / "smpl_params_sequence.npz"
    if params_npz.exists():
        try:
            z = np.load(str(params_npz), allow_pickle=True)
            if "camera_focal_xy" in z.files and "camera_princpt_xy" in z.files:
                foc = np.asarray(z["camera_focal_xy"], dtype=np.float32)
                pp = np.asarray(z["camera_princpt_xy"], dtype=np.float32)
                names = [str(x) for x in z["frame_names"].tolist()] if "frame_names" in z.files else []
                row = int(frame_index)
                if len(names) == int(foc.shape[0]) and frame_name in names:
                    row = int(names.index(frame_name))
                if 0 <= row < int(foc.shape[0]) and 0 <= row < int(pp.shape[0]):
                    fx, fy = float(foc[row, 0]), float(foc[row, 1])
                    cx, cy = float(pp[row, 0]), float(pp[row, 1])
                    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        except Exception:
            pass

    meta_path = stage1_dir / "smplestx" / "selected_frames" / "meta" / f"{int(frame_index) + 1:05d}_0.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            focal = np.asarray(meta.get("focal", []), dtype=np.float32).reshape(-1)
            princpt = np.asarray(meta.get("princpt", []), dtype=np.float32).reshape(-1)
            if focal.size >= 2 and princpt.size >= 2:
                return np.asarray(
                    [[float(focal[0]), 0.0, float(princpt[0])], [0.0, float(focal[1]), float(princpt[1])], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
        except Exception:
            pass

    if cam_json_path is not None and cam_json_path.exists():
        try:
            cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
            K = np.asarray(cam.get("K", []), dtype=np.float32)
            if K.size == 9:
                return K.reshape(3, 3)
        except Exception:
            pass

    if fallback_K is not None:
        return np.asarray(fallback_K, dtype=np.float32).reshape(3, 3)

    if image_hw is None:
        raise ValueError("image_hw is required when fallback_K and cam_json_path are unavailable.")
    h, w = int(image_hw[0]), int(image_hw[1])
    f = float(max(h, w))
    return np.asarray([[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float32)


def _largest_component_by_overlap(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None, min_area: int = 128) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    best_i = -1
    best_score = -1.0
    ref = None if ref_mask_u8 is None else (np.asarray(ref_mask_u8, dtype=np.uint8) > 0)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(max(1, min_area)):
            continue
        comp = labels == i
        if ref is not None:
            inter = float(np.logical_and(comp, ref).sum())
            cover = inter / float(max(area, 1))
            score = inter + 0.2 * cover
        else:
            score = float(area)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i < 0:
        return np.zeros_like(m, dtype=np.uint8)
    out = np.zeros_like(m, dtype=np.uint8)
    out[labels == best_i] = 1
    return out


def _fill_small_sealed_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if m.ndim != 2 or m.size <= 0:
        return (m > 0).astype(np.uint8)
    h, w = m.shape[:2]
    inv = (1 - m).astype(np.uint8)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (flood == 1).astype(np.uint8)
    if int(holes.sum()) <= 0:
        return (m > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    out = m.copy()
    for comp_i in range(1, int(n)):
        area = int(stats[comp_i, cv2.CC_STAT_AREA])
        if area <= int(max_hole_area):
            out[labels == comp_i] = 1
    return (out > 0).astype(np.uint8)


def clean_person_mask_eval_style_np(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None = None) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = _fill_small_sealed_holes(m, max_hole_area=96)
    return _largest_component_by_overlap(m, ref_mask_u8, min_area=128)


def load_person_mask_from_npz(
    *,
    npz_path: Path | None,
    frame_name: str,
    image_hw: tuple[int, int],
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray | None:
    if npz_path is None or (not npz_path.exists()):
        return None
    seq = np.load(str(npz_path), allow_pickle=True)
    if "frame_names" not in seq.files or "masks" not in seq.files:
        return None
    names = [str(x) for x in seq["frame_names"].tolist()]
    masks = np.asarray(seq["masks"], dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[0] != len(names):
        return None
    name_to_i = {n: i for i, n in enumerate(names)}
    if frame_name not in name_to_i:
        return None
    h, w = int(image_hw[0]), int(image_hw[1])
    m = masks[int(name_to_i[frame_name])]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8)
    if int(dilate_px) > 0:
        k = np.ones((2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1), dtype=np.uint8)
        m = cv2.dilate(m, k, iterations=1)
    if int(erode_px) > 0:
        k = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), dtype=np.uint8)
        m = cv2.erode(m, k, iterations=1)
    return clean_person_mask_eval_style_np(m, ref_mask_u8=None)


def estimate_person_mask_chromakey(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 35, 25], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    fg = (green == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return _pick_main_component(fg, prefer_non_border=True, min_area=128)


def _pick_main_component(mask_u8: np.ndarray, prefer_non_border: bool = True, min_area: int = 128) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    h, w = m.shape[:2]
    best_i = -1
    best_area = -1
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if int(area) < int(min_area):
            continue
        touch_border = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if prefer_non_border and touch_border:
            continue
        if int(area) > best_area:
            best_area = int(area)
            best_i = int(i)
    if best_i < 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) <= 0:
            return np.zeros_like(m, dtype=np.uint8)
        best_i = 1 + int(np.argmax(areas))
    out = np.zeros_like(m, dtype=np.uint8)
    out[labels == best_i] = 1
    return out


def estimate_person_mask_modnet_eval_style_np(
    image_bgr: np.ndarray,
    *,
    hint_mask_u8: np.ndarray | None = None,
    modnet_ref_size: int = 512,
    modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
) -> np.ndarray:
    fg = estimate_person_mask_modnet(
        image_bgr,
        hint_mask_u8=hint_mask_u8,
        ref_size=int(modnet_ref_size),
        threshold=float(modnet_threshold),
        ckpt_path=modnet_ckpt_path,
    )
    return clean_person_mask_eval_style_np(fg, ref_mask_u8=hint_mask_u8)


def _mask_border_touch_ratio(mask_u8: np.ndarray) -> float:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if h <= 2 or w <= 2:
        return 1.0
    border = np.zeros_like(m, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    inter = float((m * border).sum())
    area = float(max(1, m.sum()))
    return inter / area


def is_person_mask_plausible(person_mask: np.ndarray, smpl_mask: np.ndarray) -> bool:
    pm = (np.asarray(person_mask, dtype=np.uint8) > 0).astype(np.uint8)
    sm = (np.asarray(smpl_mask, dtype=np.uint8) > 0).astype(np.uint8)
    area_pm = float(pm.sum())
    area_sm = float(max(1, sm.sum()))
    ratio = area_pm / area_sm
    iou = mask_iou(pm, sm)
    border_ratio = _mask_border_touch_ratio(pm)
    if ratio < 0.45 or ratio > 2.20:
        return False
    if iou < 0.22:
        return False
    if border_ratio > 0.08:
        return False
    return True


def resolve_person_mask_source_np(
    *,
    gt_mask_mode: str,
    pm_npz: Path | None,
    frame_name: str,
    image_hw: tuple[int, int],
    img_bgr: np.ndarray,
    smpl_mask_raw: np.ndarray,
    modnet_ref_size: int = 512,
    modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
    npz_dilate_px: int = 0,
    npz_erode_px: int = 0,
    modnet_use_smpl_hint: bool = False,
) -> tuple[np.ndarray, str]:
    mode = str(gt_mask_mode or "auto").strip().lower()
    if mode not in ("auto", "npz", "chromakey", "modnet"):
        raise ValueError(f"Unsupported gt_mask_mode: {gt_mask_mode}")

    if mode == "npz":
        person_mask = load_person_mask_from_npz(
            npz_path=pm_npz,
            frame_name=frame_name,
            image_hw=image_hw,
            dilate_px=npz_dilate_px,
            erode_px=npz_erode_px,
        )
        if person_mask is None or int(person_mask.sum()) <= 0:
            raise RuntimeError(f"gt_mask_mode=npz but no valid person mask for frame={frame_name} in {pm_npz}")
        return person_mask, "npz"

    if mode == "chromakey":
        return estimate_person_mask_chromakey(img_bgr), "chromakey"

    if mode == "modnet":
        return (
            estimate_person_mask_modnet_eval_style_np(
                img_bgr,
                hint_mask_u8=(smpl_mask_raw if bool(modnet_use_smpl_hint) else None),
                modnet_ref_size=modnet_ref_size,
                modnet_threshold=modnet_threshold,
                modnet_ckpt_path=modnet_ckpt_path,
            ),
            "modnet",
        )

    person_mask = load_person_mask_from_npz(
        npz_path=pm_npz,
        frame_name=frame_name,
        image_hw=image_hw,
        dilate_px=npz_dilate_px,
        erode_px=npz_erode_px,
    )
    if person_mask is not None and int(person_mask.sum()) > 0:
        if is_person_mask_plausible(person_mask=person_mask, smpl_mask=smpl_mask_raw):
            return person_mask, "npz"

    ck_mask = estimate_person_mask_chromakey(img_bgr)
    if int((ck_mask > 0).sum()) > 0:
        return ck_mask, "chromakey"

    return (
        estimate_person_mask_modnet_eval_style_np(
            img_bgr,
            hint_mask_u8=(smpl_mask_raw if bool(modnet_use_smpl_hint) else None),
            modnet_ref_size=modnet_ref_size,
            modnet_threshold=modnet_threshold,
            modnet_ckpt_path=modnet_ckpt_path,
        ),
        "modnet",
    )
