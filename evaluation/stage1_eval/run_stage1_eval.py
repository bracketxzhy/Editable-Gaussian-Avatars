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

from avatar_gs.alignment import estimate_mask_similarity_transform, mask_iou
from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from person_contour import extract_person_contours, load_person_mask_from_npz, estimate_person_mask_modnet_eval


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


def _read_frame_camera_K(
    *,
    stage1_dir: Path,
    cam_json_path: Path,
    frame_index: int,
    frame_name: str,
    w: int,
    h: int,
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

    meta_path = stage1_dir / "smplerx" / "selected_frames" / "meta" / f"{int(frame_index) + 1:05d}_0.json"
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

    return _read_camera_K(cam_json_path, w=w, h=h)


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


def _pick_main_component(mask_u8: np.ndarray, prefer_non_border: bool = True, min_area: int = 128) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)
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
            return np.zeros_like(mask_u8, dtype=np.uint8)
        best_i = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[labels == best_i] = 255
    return out


def _estimate_person_mask_chromakey(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Same green-screen range used in stage1 mask generation.
    lower = np.array([35, 35, 25], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    fg = (green == 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = _pick_main_component(fg, prefer_non_border=True, min_area=128)
    return fg




def _mask_border_touch_ratio(mask_u8: np.ndarray) -> float:
    m = (mask_u8 > 0).astype(np.uint8)
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


def _is_person_mask_plausible(person_mask: np.ndarray, smpl_mask: np.ndarray) -> bool:
    pm = (person_mask > 0).astype(np.uint8)
    sm = (smpl_mask > 0).astype(np.uint8)
    area_pm = float(pm.sum())
    area_sm = float(max(1, sm.sum()))
    ratio = area_pm / area_sm
    iou = mask_iou(pm, sm)
    border_ratio = _mask_border_touch_ratio(pm)

    # Too large/small vs SMPL, too little overlap, or too much border touching -> unreliable.
    if ratio < 0.45 or ratio > 2.20:
        return False
    if iou < 0.22:
        return False
    if border_ratio > 0.08:
        return False
    return True

def _resolve_person_mask(
    *,
    gt_mask_mode: str,
    pm_npz: Path,
    frame_name: str,
    image_hw: tuple[int, int],
    img_bgr: np.ndarray,
    smpl_mask_raw: np.ndarray,
    modnet_ckpt_path: str | None = None,
) -> tuple[np.ndarray, str]:
    mode = str(gt_mask_mode).strip().lower()

    if mode == "npz":
        person_mask = None
        if pm_npz.exists():
            person_mask = load_person_mask_from_npz(npz_path=pm_npz, frame_name=frame_name, image_hw=image_hw)
        if person_mask is None or int(person_mask.sum()) <= 0:
            raise RuntimeError(f"gt_mask_mode=npz but no valid person mask for frame={frame_name} in {pm_npz}")
        return person_mask, "npz"

    if mode == "chromakey":
        return _estimate_person_mask_chromakey(img_bgr), "chromakey"

    if mode == "modnet":
        return estimate_person_mask_modnet_eval(
            img_bgr,
            hint_mask_u8=smpl_mask_raw,
            modnet_ckpt_path=modnet_ckpt_path,
        ), "modnet"

    # auto: npz first, then chromakey, then modnet.
    person_mask = None
    if pm_npz.exists():
        person_mask = load_person_mask_from_npz(npz_path=pm_npz, frame_name=frame_name, image_hw=image_hw)
    if person_mask is not None and int(person_mask.sum()) > 0:
        if _is_person_mask_plausible(person_mask=person_mask, smpl_mask=smpl_mask_raw):
            return person_mask, "npz"

    ck_mask = _estimate_person_mask_chromakey(img_bgr)
    if int((ck_mask > 0).sum()) > 0:
        return ck_mask, "chromakey"

    return estimate_person_mask_modnet_eval(
        img_bgr,
        hint_mask_u8=smpl_mask_raw,
        modnet_ckpt_path=modnet_ckpt_path,
    ), "modnet"


def _evaluate_one_frame(
    *,
    idx: int,
    frame_name: str,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    cam_json_path: Path,
    frames_dir: Path,
    out_dir: Path,
    pm_npz: Path,
    use_auto_align: bool,
    gt_mask_mode: str,
    modnet_ckpt_path: str | None,
    stage1_dir: Path,
) -> dict:
    img_path = frames_dir / frame_name
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read GT image: {img_path}")
    h, w = img.shape[:2]

    K = _read_frame_camera_K(stage1_dir=stage1_dir, cam_json_path=cam_json_path, frame_index=int(idx), frame_name=frame_name, w=w, h=h)
    uv, valid_v = _project_vertices_to_image(verts_cam=verts_cam, K=K)
    smpl_mask_raw = _rasterize_mesh_mask(uv=uv, valid_v=valid_v, faces=faces, image_hw=(h, w))

    person_mask, person_mask_source = _resolve_person_mask(
        gt_mask_mode=gt_mask_mode,
        pm_npz=pm_npz,
        frame_name=frame_name,
        image_hw=(h, w),
        img_bgr=img,
        smpl_mask_raw=smpl_mask_raw,
        modnet_ckpt_path=modnet_ckpt_path,
    )

    smpl_mask = smpl_mask_raw
    align_report = {"applied": False, "reason": "disabled_or_missing_person_mask"}
    iou_before = mask_iou(smpl_mask_raw, person_mask)
    iou_after = iou_before
    if use_auto_align and person_mask is not None and int(person_mask.sum()) > 0:
        uv_aligned, align_report = estimate_mask_similarity_transform(
            uv=uv,
            valid_v=valid_v,
            faces=faces,
            image_hw=(h, w),
            smpl_mask=smpl_mask_raw,
            person_mask=person_mask,
            scale_min=0.4,
            scale_max=2.5,
        )
        smpl_mask_aligned = _rasterize_mesh_mask(uv=uv_aligned, valid_v=valid_v, faces=faces, image_hw=(h, w))
        iou_after = mask_iou(smpl_mask_aligned, person_mask)
        if bool(align_report.get("accepted", False)):
            smpl_mask = smpl_mask_aligned
            align_report["accepted"] = True
        else:
            align_report["accepted"] = False

    out = img.copy()
    smpl_cnt = extract_person_contours((smpl_mask > 0).astype(np.uint8) * 255)
    gt_cnt = extract_person_contours((person_mask > 0).astype(np.uint8) * 255)
    contour_thickness = 2
    if len(gt_cnt) > 0:
        cv2.drawContours(out, gt_cnt, -1, color=(0, 255, 255), thickness=contour_thickness, lineType=cv2.LINE_AA)
    if len(smpl_cnt) > 0:
        cv2.drawContours(out, smpl_cnt, -1, color=(0, 0, 255), thickness=contour_thickness, lineType=cv2.LINE_AA)

    out_img = out_dir / f"{Path(frame_name).stem}_gt_stage1_contour.png"
    if not cv2.imwrite(str(out_img), out):
        raise RuntimeError(f"Failed to write image: {out_img}")

    return {
        "frame_name": frame_name,
        "frame_index": int(idx),
        "output_image": str(out_img),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "align": align_report,
        "gt_mask_source": str(person_mask_source),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Stage1 contour evaluation (GT vs Stage1 SMPL)")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--stage1_dir", type=str, default="outputs/stage1")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input")
    parser.add_argument("--out_dir", type=str, default="evaluation/output_images/stage1")
    parser.add_argument("--frame_select", type=str, default="last", choices=["all", "first", "last", "index", "name"])
    parser.add_argument("--frame_index", type=int, default=-1)
    parser.add_argument("--frame_name", type=str, default="")
    parser.add_argument("--person_masks_npz", type=str, default="")
    parser.add_argument("--modnet_ckpt", type=str, default="")
    parser.add_argument(
        "--gt_mask_mode",
        type=str,
        default="auto",
        choices=["auto", "npz", "chromakey", "modnet"],
        help="GT person mask source: auto(npz->chromakey->modnet), npz, chromakey, or modnet.",
    )
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
    mode = str(args.frame_select).strip().lower()
    if mode == "all":
        eval_indices = list(range(n_frames))
    else:
        idx = _resolve_pick_index(
            mode=mode,
            n=n_frames,
            index=int(args.frame_index),
            frame_name=(None if str(args.frame_name).strip() == "" else str(args.frame_name).strip()),
            names=frame_names,
        )
        eval_indices = [int(idx)]

    frame_name_ref = frame_names[int(eval_indices[-1])]
    frame_stem = Path(frame_name_ref).stem
    mesh_candidates = [
        stage1_dir / "smpl_mesh_tpose.obj",
        stage1_dir / "sequence" / f"{frame_stem}.obj",
        stage1_dir / "sequence" / f"frame_{frame_stem}.obj",
        stage1_dir / "final" / "last_frame_smplx_cam.obj",
        stage1_dir / "final" / "last_frame_smplx.obj",
        stage1_dir / "final" / "last_frame_pose_local.obj",
    ]
    mesh_obj = next((m for m in mesh_candidates if m.exists()), None)
    if mesh_obj is None:
        raise FileNotFoundError(
            "Missing mesh for face topology. Tried: " + ", ".join(str(x) for x in mesh_candidates)
        )
    _, faces = load_obj_mesh(mesh_obj)
    if faces is None or len(faces) <= 0:
        raise RuntimeError(f"No faces in mesh topology: {mesh_obj}")

    pm_npz = (stage1_dir / "person_masks.npz") if str(args.person_masks_npz).strip() == "" else Path(args.person_masks_npz)
    if not pm_npz.is_absolute():
        pm_npz = (project_root / pm_npz).resolve()
    modnet_ckpt_path = str(args.modnet_ckpt).strip() or None

    use_auto_align = (not bool(args.no_auto_align))
    if bool(args.auto_align):
        use_auto_align = True

    reports: list[dict] = []
    for idx in eval_indices:
        frame_name = frame_names[idx]
        report_i = _evaluate_one_frame(
            idx=int(idx),
            frame_name=frame_name,
            verts_cam=verts_cam_all[idx],
            faces=faces,
            cam_json_path=stage1_dir / "cameras.json",
            frames_dir=frames_dir,
            out_dir=out_dir,
            pm_npz=pm_npz,
            use_auto_align=use_auto_align,
            gt_mask_mode=str(args.gt_mask_mode),
            modnet_ckpt_path=modnet_ckpt_path,
            stage1_dir=stage1_dir,
        )
        reports.append(report_i)
        print(
            f"Saved: {report_i['output_image']} | "
            f"Frame: idx={report_i['frame_index']}, name={report_i['frame_name']} | "
            f"IoU: {report_i['iou_before']:.4f} -> {report_i['iou_after']:.4f}"
        )

    if len(reports) == 1:
        report = dict(reports[0])
        report["stage1_dir"] = str(stage1_dir)
        report["frames_dir"] = str(frames_dir)
        report_path = out_dir / f"{Path(str(report['frame_name'])).stem}_stage1_eval_report.json"
    else:
        report = {
            "n_frames": len(reports),
            "frame_reports": reports,
            "stage1_dir": str(stage1_dir),
            "frames_dir": str(frames_dir),
        }
        report_path = out_dir / "stage1_eval_report_all.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

