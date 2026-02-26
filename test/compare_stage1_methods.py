from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from avatar_gs.io.mesh_export import export_obj


@dataclass
class MethodResult:
    name: str
    out_dir: Path
    obj_path: Path
    model_type: str
    iou: float
    precision: float
    recall: float
    f1: float
    overlay_path: Path
    align_report: dict


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _find_last_sequence_obj(stage1_out_dir: Path) -> Path:
    seq_dir = stage1_out_dir / "sequence"
    cands = sorted(seq_dir.glob("frame_*.obj"))
    if not cands:
        cands = sorted(seq_dir.glob("*.obj"))
    if not cands:
        raise FileNotFoundError(f"No sequence obj found under: {seq_dir}")
    return cands[-1]


def _load_faces_from_npz(npz_path: Path) -> np.ndarray:
    arr = np.load(str(npz_path), allow_pickle=True)
    if "f" in arr.files:
        return np.asarray(arr["f"], dtype=np.int32)
    if "faces" in arr.files:
        return np.asarray(arr["faces"], dtype=np.int32)
    raise KeyError(f"No face topology key found in {npz_path} (expected 'f' or 'faces').")


def _resolve_smpl_face_npz(assets_dir: Path) -> Path:
    cands = [
        assets_dir / "smpl" / "SMPL_N_model.npz",
        assets_dir / "smpl" / "SMPL_NEUTRAL.npz",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find SMPL topology npz for WHAM-only export. "
        f"Checked: {[str(x) for x in cands]}"
    )


def _export_wham_last_obj(stage1_out_dir: Path, assets_dir: Path, dst_obj: Path) -> str:
    wham_pth = stage1_out_dir / "wham" / "wham_results.pth"
    if not wham_pth.exists():
        raise FileNotFoundError(f"WHAM output missing: {wham_pth}")
    wham_res = joblib.load(str(wham_pth))
    if not isinstance(wham_res, dict) or len(wham_res) <= 0:
        raise RuntimeError(f"Invalid WHAM result format: {type(wham_res)}")

    first_key = sorted(wham_res.keys())[0]
    track = wham_res[first_key]
    if not isinstance(track, dict):
        raise RuntimeError("WHAM track payload is not a dict.")
    if "verts_cam" not in track:
        raise RuntimeError("WHAM track has no verts_cam field.")

    verts_cam = np.asarray(track["verts_cam"], dtype=np.float32)
    if verts_cam.ndim != 3 or verts_cam.shape[2] != 3:
        raise RuntimeError(f"Unexpected WHAM verts_cam shape: {verts_cam.shape}")
    frame_ids = np.asarray(track.get("frame_id", np.arange(verts_cam.shape[0])), dtype=np.int64).reshape(-1)
    last_i = int(np.argmax(frame_ids)) if frame_ids.size > 0 else int(verts_cam.shape[0] - 1)
    last_i = max(0, min(last_i, int(verts_cam.shape[0] - 1)))

    verts_last = verts_cam[last_i]
    faces = _load_faces_from_npz(_resolve_smpl_face_npz(assets_dir))
    export_obj(dst_obj, verts_last, faces)
    return "smpl"


def _run_stage1_mode(
    project_root: Path,
    frames_dir: Path,
    assets_dir: Path,
    wham_repo_dir: Path,
    pymaf_repo_dir: Path,
    out_dir: Path,
    mode: str,
    single_person: bool,
    pymaf_frame_indices: str,
    pymaf_stride: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "avatar_gs.cli.stage1_fit_wham_pymaf",
        "--project_root",
        str(project_root),
        "--frames_dir",
        str(frames_dir),
        "--assets_dir",
        str(assets_dir),
        "--out_dir",
        str(out_dir),
        "--wham_repo_dir",
        str(wham_repo_dir),
        "--pymaf_repo_dir",
        str(pymaf_repo_dir),
        "--pymaf_frame_indices",
        str(pymaf_frame_indices),
        "--pymaf_stride",
        str(int(pymaf_stride)),
    ]
    if single_person:
        cmd.append("--single_person")
    if mode == "wham_only":
        cmd.append("--skip_pymaf")
    elif mode == "pymaf_only":
        cmd.append("--skip_wham")

    subprocess.run(cmd, cwd=str(project_root), check=True)


def _project_verts_to_uv(verts_cam: np.ndarray, width: int, height: int, focal: float = 5000.0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(verts_cam, dtype=np.float32).reshape(-1, 3)
    z = pts[:, 2]
    valid = z > 1e-6
    u = np.zeros((pts.shape[0],), dtype=np.float32)
    v = np.zeros((pts.shape[0],), dtype=np.float32)
    fx = float(focal)
    fy = float(focal)
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    u[valid] = fx * (pts[valid, 0] / z[valid]) + cx
    v[valid] = fy * (pts[valid, 1] / z[valid]) + cy
    return np.stack([u, v], axis=1), valid


def _rasterize_mesh_mask(uv: np.ndarray, valid_v: np.ndarray, faces: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for tri in f:
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


def _largest_component_by_area(mask_u8: np.ndarray, min_area: int = 128) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if int(m.sum()) <= 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    best = 0
    best_area = 0
    for i in range(1, n):
        a = int(areas[i])
        if a > best_area:
            best = i
            best_area = a
    if best_area < int(min_area):
        return np.zeros_like(mask_u8, dtype=np.uint8)
    out = (labels == best).astype(np.uint8) * 255
    return out


def _estimate_gt_person_mask(image_bgr: np.ndarray, border_ratio: float = 0.04, bg_percentile: float = 97.0, min_delta: float = 6.0) -> np.ndarray:
    h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if h <= 2 or w <= 2:
        return np.zeros((h, w), dtype=np.uint8)
    rgb_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    band = max(2, int(round(float(min(h, w)) * float(max(border_ratio, 0.01)))))
    border = np.zeros((h, w), dtype=np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_lab = lab[border > 0]
    if border_lab.shape[0] < 16:
        return np.zeros((h, w), dtype=np.uint8)

    bg_med = np.median(border_lab, axis=0).astype(np.float32)
    d_all = np.linalg.norm(lab - bg_med.reshape(1, 1, 3), axis=2)
    d_border = np.linalg.norm(border_lab - bg_med.reshape(1, 3), axis=1)
    th_border = float(np.percentile(d_border, float(np.clip(bg_percentile, 50.0, 100.0)))) + float(max(min_delta, 0.0))
    th_global = float(np.percentile(d_all, 55.0))
    th = max(th_border, th_global)

    fg = (d_all > th).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k5, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5, iterations=2)
    fg = _largest_component_by_area(fg, min_area=max(128, int(round(0.01 * float(h * w)))))
    if int((fg > 0).sum()) > 0:
        fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return fg


def _binary_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray) -> tuple[float, float, float, float]:
    p = (pred_u8 > 0)
    g = (gt_u8 > 0)
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    p_sum = float(p.sum())
    g_sum = float(g.sum())
    iou = inter / (union + 1e-8)
    precision = inter / (p_sum + 1e-8)
    recall = inter / (g_sum + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return iou, precision, recall, f1


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


def _draw_overlay(
    image_bgr: np.ndarray,
    gt_mask_u8: np.ndarray,
    mesh_mask_u8: np.ndarray,
    title: str,
    metric_text: str,
    out_path: Path,
) -> None:
    out = image_bgr.copy()
    gt_contours, _ = cv2.findContours((gt_mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mesh_contours, _ = cv2.findContours((mesh_mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, gt_contours, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)   # yellow
    cv2.drawContours(out, mesh_contours, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)     # red
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, metric_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def _method_label(method: str) -> str:
    if method == "wham_pymaf":
        return "WHAM+PYMAF"
    if method == "wham_only":
        return "WHAM-ONLY"
    if method == "pymaf_only":
        return "PYMAF-ONLY"
    return method


def main() -> None:
    parser = argparse.ArgumentParser("Compare stage1 modes: WHAM+PYMAF vs WHAM-only vs PYMAF-only")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input")
    parser.add_argument("--assets_dir", type=str, default="assets")
    parser.add_argument("--wham_repo_dir", type=str, default="third_party/WHAM")
    parser.add_argument("--pymaf_repo_dir", type=str, default="third_party/PyMAF-X")
    parser.add_argument("--work_root", type=str, default="test/stage1_compare")
    parser.add_argument("--output_dir", type=str, default="test/stage1_method_compare_output")
    parser.add_argument("--single_person", action="store_true")
    parser.add_argument("--pymaf_frame_indices", type=str, default="-1")
    parser.add_argument("--pymaf_stride", type=int, default=1)
    parser.add_argument("--focal", type=float, default=5000.0)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    wham_repo_dir = (project_root / args.wham_repo_dir).resolve()
    pymaf_repo_dir = (project_root / args.pymaf_repo_dir).resolve()
    work_root = (project_root / args.work_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    obj_out_dir = output_dir / "objs"
    vis_out_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    gt_last_path = frames[-1]
    gt_bgr = cv2.imread(str(gt_last_path), cv2.IMREAD_COLOR)
    if gt_bgr is None:
        raise RuntimeError(f"Cannot read GT frame: {gt_last_path}")
    h, w = int(gt_bgr.shape[0]), int(gt_bgr.shape[1])
    gt_mask = _estimate_gt_person_mask(gt_bgr)

    methods = ["wham_pymaf", "wham_only", "pymaf_only"]
    results: list[MethodResult] = []

    for method in methods:
        stage1_out = work_root / method
        if (not bool(args.reuse_existing)) or (not stage1_out.exists()):
            if stage1_out.exists():
                shutil.rmtree(stage1_out, ignore_errors=True)
            _run_stage1_mode(
                project_root=project_root,
                frames_dir=frames_dir,
                assets_dir=assets_dir,
                wham_repo_dir=wham_repo_dir,
                pymaf_repo_dir=pymaf_repo_dir,
                out_dir=stage1_out,
                mode=method,
                single_person=bool(args.single_person),
                pymaf_frame_indices=str(args.pymaf_frame_indices),
                pymaf_stride=int(args.pymaf_stride),
            )

        dst_obj = obj_out_dir / f"{method}_last.obj"
        model_type = "smplx"
        if method == "wham_only":
            model_type = _export_wham_last_obj(stage1_out_dir=stage1_out, assets_dir=assets_dir, dst_obj=dst_obj)
        else:
            last_obj = _find_last_sequence_obj(stage1_out)
            shutil.copy2(last_obj, dst_obj)
            model_type = "smplx"

        verts, faces = load_obj_mesh(dst_obj)
        uv, valid = _project_verts_to_uv(verts_cam=verts, width=w, height=h, focal=float(args.focal))
        mesh_mask_raw = _rasterize_mesh_mask(uv=uv, valid_v=valid, faces=faces, h=h, w=w)
        iou_before = _iou(mesh_mask_raw, gt_mask)
        uv_aligned, align_report = _align_uv_by_bbox(uv=uv, smpl_mask=mesh_mask_raw, person_mask=gt_mask)
        mesh_mask_aligned = _rasterize_mesh_mask(uv=uv_aligned, valid_v=valid, faces=faces, h=h, w=w)
        iou_after = _iou(mesh_mask_aligned, gt_mask)
        if iou_after > iou_before:
            mesh_mask = mesh_mask_aligned
            align_report["accepted"] = True
            align_report["iou_before"] = float(iou_before)
            align_report["iou_after"] = float(iou_after)
        else:
            mesh_mask = mesh_mask_raw
            align_report["accepted"] = False
            align_report["iou_before"] = float(iou_before)
            align_report["iou_after"] = float(iou_after)
        iou, precision, recall, f1 = _binary_metrics(pred_u8=mesh_mask, gt_u8=gt_mask)

        overlay_path = vis_out_dir / f"{method}_contour_compare.png"
        metric_txt = f"IoU={iou:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
        title = f"{_method_label(method)}  GT(yellow) vs Mesh(red)"
        _draw_overlay(
            image_bgr=gt_bgr,
            gt_mask_u8=gt_mask,
            mesh_mask_u8=mesh_mask,
            title=title,
            metric_text=metric_txt,
            out_path=overlay_path,
        )

        results.append(
            MethodResult(
                name=method,
                out_dir=stage1_out,
                obj_path=dst_obj,
                model_type=model_type,
                iou=iou,
                precision=precision,
                recall=recall,
                f1=f1,
                overlay_path=overlay_path,
                align_report=align_report,
            )
        )

    ranked = sorted(results, key=lambda x: (x.f1, x.iou), reverse=True)
    best = ranked[0]
    summary = {
        "gt_last_frame": str(gt_last_path),
        "note": (
            "WHAM-only export is SMPL topology (6890 verts) when WHAM output lacks SMPL-X vertices; "
            "other two methods are SMPL-X from PyMAF."
        ),
        "best_method_by_f1_then_iou": best.name,
        "methods": [
            {
                "name": r.name,
                "label": _method_label(r.name),
                "model_type": r.model_type,
                "stage1_out_dir": str(r.out_dir),
                "obj_path": str(r.obj_path),
                "overlay_path": str(r.overlay_path),
                "iou": float(r.iou),
                "precision": float(r.precision),
                "recall": float(r.recall),
                "f1": float(r.f1),
                "align_report": r.align_report,
            }
            for r in results
        ],
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"GT last frame: {gt_last_path}")
    for r in results:
        print(
            f"[{_method_label(r.name)}] model={r.model_type} "
            f"IoU={r.iou:.4f} F1={r.f1:.4f} "
            f"OBJ={r.obj_path} OVERLAY={r.overlay_path}"
        )
    print(f"Best (F1, IoU): {_method_label(best.name)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
