from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from person_contour import extract_person_contours, load_person_mask_from_npz


def _resolve_pick_index(mode: str, n: int, index: int = -1, frame_name: str | None = None, names: list[str] | None = None) -> int:
    if n <= 0:
        return 0
    m = str(mode or "last").strip().lower()
    if m == "first":
        return 0
    if m == "name":
        if not frame_name or names is None:
            return n - 1
        q = str(frame_name).strip()
        if q in names:
            return int(names.index(q))
        q2 = q if (q.lower().endswith(".jpg") or q.lower().endswith(".png")) else f"{q}.jpg"
        if q2 in names:
            return int(names.index(q2))
        return n - 1
    if m == "index":
        i = int(index)
        if i < 0:
            i = n + i
        return int(np.clip(i, 0, n - 1))
    return n - 1


def _project_vertices_to_image(verts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(verts_cam[:, 2], 1e-6, None)
    u = float(K[0, 0]) * (verts_cam[:, 0] / z) + float(K[0, 2])
    v = float(K[1, 1]) * (verts_cam[:, 1] / z) + float(K[1, 2])
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & (verts_cam[:, 2] > 1e-5)
    return uv, valid


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


def _mask_bbox(mask_u8: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size <= 0 or xs.size <= 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def _align_uv_by_bbox(
    uv: np.ndarray,
    smpl_mask: np.ndarray,
    person_mask: np.ndarray,
    scale_min: float = 0.4,
    scale_max: float = 2.5,
) -> tuple[np.ndarray, dict]:
    b_s = _mask_bbox(smpl_mask)
    b_p = _mask_bbox(person_mask)
    if b_s is None or b_p is None:
        return uv, {"applied": False, "reason": "empty_bbox"}

    ws = float(max(1.0, b_s[2] - b_s[0] + 1.0))
    hs = float(max(1.0, b_s[3] - b_s[1] + 1.0))
    wp = float(max(1.0, b_p[2] - b_p[0] + 1.0))
    hp = float(max(1.0, b_p[3] - b_p[1] + 1.0))

    csx = float(0.5 * (b_s[0] + b_s[2]))
    csy = float(0.5 * (b_s[1] + b_s[3]))
    cpx = float(0.5 * (b_p[0] + b_p[2]))
    cpy = float(0.5 * (b_p[1] + b_p[3]))

    s = float(0.5 * (wp / ws + hp / hs))
    s = float(np.clip(s, scale_min, scale_max))

    uv2 = uv.copy()
    uv2[:, 0] = (uv2[:, 0] - csx) * s + cpx
    uv2[:, 1] = (uv2[:, 1] - csy) * s + cpy

    return uv2, {
        "applied": True,
        "scale": s,
        "center_src": [csx, csy],
        "center_tgt": [cpx, cpy],
        "bbox_src": b_s.tolist(),
        "bbox_tgt": b_p.tolist(),
    }


def _read_camera_K(cam_json_path: Path, w: int, h: int) -> np.ndarray:
    if cam_json_path.exists():
        try:
            cam = json.loads(cam_json_path.read_text(encoding="utf-8"))
            K = np.asarray(cam.get("K", []), dtype=np.float32)
            if K.size == 9:
                return K.reshape(3, 3)
        except Exception:
            pass
    f = float(max(h, w))
    return np.asarray([[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float32)


def _largest_component_by_overlap(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None, min_area: int = 128) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)

    best_i = -1
    best_score = -1.0
    ref = (ref_mask_u8 > 0) if ref_mask_u8 is not None else None

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comp = (labels == i)
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
        return np.zeros_like(mask_u8, dtype=np.uint8)

    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[labels == best_i] = 255
    return out


def _grabcut_person_mask(img_bgr: np.ndarray, smpl_mask_hint: np.ndarray | None = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    if smpl_mask_hint is None or int((smpl_mask_hint > 0).sum()) <= 0:
        rect = (int(0.08 * w), int(0.04 * h), int(0.84 * w), int(0.92 * h))
        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        cv2.grabCut(img_bgr, gc_mask, rect, bg_model, fg_model, 5, mode=cv2.GC_INIT_WITH_RECT)
    else:
        hint = (smpl_mask_hint > 0).astype(np.uint8)
        if hint.shape[0] != h or hint.shape[1] != w:
            hint = cv2.resize(hint, (w, h), interpolation=cv2.INTER_NEAREST)

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Strong foreground prior from SMPL silhouette center region.
        sure_fg = cv2.erode(hint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        pr_fg = cv2.dilate(hint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)

        gc_mask[pr_fg > 0] = cv2.GC_PR_FGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD

        # Image borders are very likely background.
        bd = max(8, int(0.02 * min(h, w)))
        gc_mask[:bd, :] = cv2.GC_BGD
        gc_mask[-bd:, :] = cv2.GC_BGD
        gc_mask[:, :bd] = cv2.GC_BGD
        gc_mask[:, -bd:] = cv2.GC_BGD

        cv2.grabCut(img_bgr, gc_mask, None, bg_model, fg_model, 5, mode=cv2.GC_INIT_WITH_MASK)

    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    # Keep the component that overlaps SMPL the most, not the largest green area.
    fg = _largest_component_by_overlap(fg, smpl_mask_hint, min_area=256)
    return fg


def main() -> None:
    parser = argparse.ArgumentParser("Stage1 contour evaluation (GT vs Stage1 SMPL)")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--stage1_dir", type=str, default="outputs/stage1")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input")
    parser.add_argument("--out_dir", type=str, default="evaluation/output_images/stage1")
    parser.add_argument("--frame_select", type=str, default="last", choices=["first", "last", "index", "name"])
    parser.add_argument("--frame_index", type=int, default=-1)
    parser.add_argument("--frame_name", type=str, default="")
    parser.add_argument("--person_masks_npz", type=str, default="")
    parser.add_argument("--auto_align", action="store_true")
    parser.add_argument("--no_auto_align", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    verts_npz = stage1_dir / "smpl_vertices_sequence.npz"
    if not verts_npz.exists():
        raise FileNotFoundError(f"Missing stage1 vertices sequence: {verts_npz}")

    seq = np.load(str(verts_npz), allow_pickle=True)
    if "frame_names" not in seq.files or "verts_cam" not in seq.files:
        raise RuntimeError(f"Bad vertices npz format: {verts_npz}")
    frame_names = [str(x) for x in seq["frame_names"].tolist()]
    verts_cam_all = np.asarray(seq["verts_cam"], dtype=np.float32)
    n_frames = int(verts_cam_all.shape[0])
    idx = _resolve_pick_index(
        mode=str(args.frame_select),
        n=n_frames,
        index=int(args.frame_index),
        frame_name=(None if str(args.frame_name).strip() == "" else str(args.frame_name).strip()),
        names=frame_names,
    )
    frame_name = frame_names[idx]

    img_path = frames_dir / frame_name
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read GT image: {img_path}")
    h, w = img.shape[:2]

    mesh_obj = stage1_dir / "smpl_mesh_tpose.obj"
    if not mesh_obj.exists():
        raise FileNotFoundError(f"Missing mesh for face topology: {mesh_obj}")
    _, faces = load_obj_mesh(mesh_obj)
    if faces is None or len(faces) <= 0:
        raise RuntimeError(f"No faces in mesh topology: {mesh_obj}")

    K = _read_camera_K(stage1_dir / "cameras.json", w=w, h=h)
    verts_cam = verts_cam_all[idx]
    uv, valid_v = _project_vertices_to_image(verts_cam=verts_cam, K=K)
    smpl_mask_raw = _rasterize_mesh_mask(uv=uv, valid_v=valid_v, faces=faces, image_hw=(h, w))

    pm_npz = (stage1_dir / "person_masks.npz") if str(args.person_masks_npz).strip() == "" else Path(args.person_masks_npz)
    if not pm_npz.is_absolute():
        pm_npz = (project_root / pm_npz).resolve()

    person_mask = None
    if pm_npz.exists():
        person_mask = load_person_mask_from_npz(npz_path=pm_npz, frame_name=frame_name, image_hw=(h, w))

    need_recompute = (person_mask is None) or (int(person_mask.sum()) <= 0)
    if not need_recompute:
        iou_npz = _iou(person_mask, smpl_mask_raw)
        # If mask is far from SMPL body area, treat it as unreliable and recompute.
        if iou_npz < 0.10:
            need_recompute = True

    if need_recompute:
        person_mask = _grabcut_person_mask(img, smpl_mask_hint=smpl_mask_raw)

    use_auto_align = (not bool(args.no_auto_align))
    if bool(args.auto_align):
        use_auto_align = True

    smpl_mask = smpl_mask_raw
    align_report = {"applied": False, "reason": "disabled_or_missing_person_mask"}
    iou_before = _iou(smpl_mask_raw, person_mask)
    iou_after = iou_before
    if use_auto_align and person_mask is not None and int(person_mask.sum()) > 0:
        uv_aligned, align_report = _align_uv_by_bbox(uv=uv, smpl_mask=smpl_mask_raw, person_mask=person_mask)
        smpl_mask_aligned = _rasterize_mesh_mask(uv=uv_aligned, valid_v=valid_v, faces=faces, image_hw=(h, w))
        iou_after = _iou(smpl_mask_aligned, person_mask)
        if iou_after >= iou_before:
            smpl_mask = smpl_mask_aligned
            align_report["accepted"] = True
        else:
            align_report["accepted"] = False

    out = img.copy()
    smpl_cnt = extract_person_contours((smpl_mask > 0).astype(np.uint8) * 255)
    gt_cnt = extract_person_contours((person_mask > 0).astype(np.uint8) * 255)
    if len(gt_cnt) > 0:
        cv2.drawContours(out, gt_cnt[:1], -1, color=(0, 255, 255), thickness=2)
    if len(smpl_cnt) > 0:
        cv2.drawContours(out, smpl_cnt[:1], -1, color=(0, 0, 255), thickness=2)

    out_img = out_dir / f"{Path(frame_name).stem}_gt_stage1_contour.png"
    if not cv2.imwrite(str(out_img), out):
        raise RuntimeError(f"Failed to write image: {out_img}")

    report = {
        "frame_name": frame_name,
        "frame_index": int(idx),
        "output_image": str(out_img),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "align": align_report,
        "stage1_dir": str(stage1_dir),
        "frames_dir": str(frames_dir),
    }
    report_path = out_dir / f"{Path(frame_name).stem}_stage1_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {out_img}")
    print(f"Report: {report_path}")
    print(f"Frame: idx={idx}, name={frame_name}")
    print(f"IoU: {iou_before:.4f} -> {iou_after:.4f}")


if __name__ == "__main__":
    main()

