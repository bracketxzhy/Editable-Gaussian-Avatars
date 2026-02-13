from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ContourResult:
    method: str
    contour_count: int
    contour_points: int
    mask_foreground_pixels: int
    output_path: Path
    mask_path: Path | None


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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = _largest_component(mask, min_area=128)
    return mask


def load_person_mask_from_npz(
    npz_path: Path,
    frame_name: str,
    image_hw: tuple[int, int],
) -> np.ndarray | None:
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


def estimate_person_mask_grabcut(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    rect_x = int(0.08 * w)
    rect_y = int(0.04 * h)
    rect_w = int(0.84 * w)
    rect_h = int(0.92 * h)
    rect = (rect_x, rect_y, rect_w, rect_h)

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        image_bgr,
        gc_mask,
        rect,
        bg_model,
        fg_model,
        5,
        mode=cv2.GC_INIT_WITH_RECT,
    )
    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return _clean_mask(fg)


def extract_person_contours(mask_u8: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def overlay_contours(
    image_bgr: np.ndarray,
    contours: list[np.ndarray],
    color_bgr: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = image_bgr.copy()
    if len(contours) > 0:
        cv2.drawContours(out, contours, contourIdx=-1, color=color_bgr, thickness=int(max(1, thickness)))
    return out


def annotate_image_with_person_contour(
    image_path: Path,
    output_path: Path,
    person_masks_npz: Path | None = None,
    frame_name: str | None = None,
    save_mask_path: Path | None = None,
    contour_color_bgr: tuple[int, int, int] = (0, 255, 255),
    contour_thickness: int = 2,
) -> ContourResult:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image_bgr.shape[:2]

    frame_name_resolved = frame_name
    if frame_name_resolved is None:
        frame_name_resolved = _infer_frame_name_from_image_name(image_path)

    mask = None
    method = "grabcut"
    if person_masks_npz is not None and frame_name_resolved is not None:
        mask = load_person_mask_from_npz(
            npz_path=person_masks_npz,
            frame_name=frame_name_resolved,
            image_hw=(h, w),
        )
        if mask is not None and int(mask.sum()) > 0:
            method = "npz_mask"

    if mask is None or int(mask.sum()) <= 0:
        mask = estimate_person_mask_grabcut(image_bgr)
        method = "grabcut"

    contours = extract_person_contours(mask)
    out_img = overlay_contours(
        image_bgr=image_bgr,
        contours=contours[:1],
        color_bgr=contour_color_bgr,
        thickness=contour_thickness,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), out_img)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    saved_mask = None
    if save_mask_path is not None:
        save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_mask_path), mask)
        saved_mask = save_mask_path

    contour_points = int(sum(len(c) for c in contours))
    return ContourResult(
        method=method,
        contour_count=int(len(contours)),
        contour_points=contour_points,
        mask_foreground_pixels=int((mask > 0).sum()),
        output_path=output_path,
        mask_path=saved_mask,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Person contour annotator")
    p.add_argument("--image", type=str, required=True, help="Input image path")
    p.add_argument("--output", type=str, required=True, help="Output image path")
    p.add_argument("--person-masks-npz", type=str, default="", help="Optional stage1 person_masks.npz")
    p.add_argument("--frame-name", type=str, default="", help="Frame name in npz, e.g. 000756.jpg")
    p.add_argument("--save-mask", type=str, default="", help="Optional output path for binary person mask")
    p.add_argument("--contour-bgr", type=int, nargs=3, default=[0, 255, 255], help="Contour color in BGR")
    p.add_argument("--thickness", type=int, default=2, help="Contour thickness")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    image = Path(args.image)
    output = Path(args.output)
    npz = Path(args.person_masks_npz) if str(args.person_masks_npz).strip() else None
    frame_name = str(args.frame_name).strip() or None
    save_mask = Path(args.save_mask) if str(args.save_mask).strip() else None
    color = (int(args.contour_bgr[0]), int(args.contour_bgr[1]), int(args.contour_bgr[2]))
    res = annotate_image_with_person_contour(
        image_path=image,
        output_path=output,
        person_masks_npz=npz,
        frame_name=frame_name,
        save_mask_path=save_mask,
        contour_color_bgr=color,
        contour_thickness=int(args.thickness),
    )
    print(
        "Person contour done:",
        f"method={res.method}, contours={res.contour_count},",
        f"points={res.contour_points}, fg_pixels={res.mask_foreground_pixels},",
        f"output={res.output_path}",
    )
    if res.mask_path is not None:
        print(f"Saved mask: {res.mask_path}")


if __name__ == "__main__":
    main()

