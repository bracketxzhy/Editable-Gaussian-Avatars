from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from avatar_gs.segmentation import estimate_person_mask_modnet


def _infer_frame_name_from_image_name(image_path: Path) -> str | None:
    stem = image_path.stem
    m = re.search(r"(\d{6})", stem)
    if not m:
        return None
    fid = m.group(1)
    return f"{fid}.jpg"


def _largest_component(mask_u8: np.ndarray, min_area: int = 64) -> np.ndarray:
    if mask_u8.ndim != 2:
        raise ValueError(f"mask_u8 must be HxW, got {mask_u8.shape}")
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if int(areas[best - 1]) < int(max(1, min_area)):
        return np.zeros_like(mask_u8, dtype=np.uint8)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[labels == best] = 255
    return out


def _clean_mask(mask_u8: np.ndarray) -> np.ndarray:
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = _fill_small_sealed_holes(mask, max_hole_area=96)
    mask = _largest_component(mask, min_area=128)
    return mask


def _fill_small_sealed_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if m.ndim != 2 or m.size <= 0:
        return (m > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    inv = (1 - m).astype(np.uint8)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (flood == 1).astype(np.uint8)
    if int(holes.sum()) <= 0:
        return (m > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    out = m.copy()
    for comp_i in range(1, int(n)):
        area = int(stats[comp_i, cv2.CC_STAT_AREA])
        if area <= int(max_hole_area):
            out[labels == comp_i] = 1
    return (out > 0).astype(np.uint8) * 255


def load_person_mask_from_npz(npz_path: Path, frame_name: str, image_hw: tuple[int, int]) -> np.ndarray | None:
    if not npz_path.exists():
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
    m = masks[int(name_to_i[frame_name])]
    if m.shape != image_hw:
        m = cv2.resize(m, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
    return _clean_mask((m > 0).astype(np.uint8) * 255)


def estimate_person_mask_modnet_eval(
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
    return _clean_mask(fg)


def extract_person_contours(mask_u8: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: abs(float(cv2.contourArea(c))), reverse=True)
    return contours
