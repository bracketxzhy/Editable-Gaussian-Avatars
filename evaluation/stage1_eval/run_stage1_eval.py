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

from avatar_gs.cli.stage1_fit_smplestx import (
    _build_smplestx_frame_records as _build_smplestx_frame_records_raw,
    _find_any_file as _find_any_file_stage1,
    _load_faces_from_npz as _load_faces_from_npz_stage1,
)
from avatar_gs.alignment import estimate_mask_similarity_transform, mask_iou
from avatar_gs.alignment.mask_sources import (
    project_vertices_to_image as _shared_project_vertices_to_image,
    rasterize_uv_mask as _shared_rasterize_uv_mask,
    read_stage1_frame_camera_K as _shared_read_stage1_frame_camera_K,
    resolve_person_mask_source_np as _shared_resolve_person_mask_source_np,
)
from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from person_contour import extract_person_contours


def _mask_outside_ratio(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    pred = np.asarray(mask_pred, dtype=np.uint8) > 0
    gt = np.asarray(mask_gt, dtype=np.uint8) > 0
    pred_area = float(max(1, pred.sum()))
    outside = float(np.logical_and(pred, np.logical_not(gt)).sum())
    return outside / pred_area


def _mask_inside_coverage(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    pred = np.asarray(mask_pred, dtype=np.uint8) > 0
    gt = np.asarray(mask_gt, dtype=np.uint8) > 0
    gt_area = float(max(1, gt.sum()))
    covered = float(np.logical_and(pred, gt).sum())
    return covered / gt_area


def _mask_area_ratio(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    pred_area = float(max(1, (np.asarray(mask_pred, dtype=np.uint8) > 0).sum()))
    gt_area = float(max(1, (np.asarray(mask_gt, dtype=np.uint8) > 0).sum()))
    return pred_area / gt_area


def _boundary_mask(mask_u8: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask.ndim != 2:
        raise ValueError(f"mask_u8 must be HxW, got {mask.shape}")
    contours = extract_person_contours(mask)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(boundary, contours, -1, color=255, thickness=1, lineType=cv2.LINE_8)
    return boundary


def _boundary_distance_stats(boundary_src: np.ndarray, boundary_tgt: np.ndarray) -> dict[str, float]:
    src = np.asarray(boundary_src, dtype=np.uint8) > 0
    tgt = np.asarray(boundary_tgt, dtype=np.uint8) > 0
    src_count = int(src.sum())
    if src_count <= 0:
        return {"mean_px": float("nan"), "p95_px": float("nan"), "within_2px": float("nan")}
    inv_tgt = np.logical_not(tgt).astype(np.uint8)
    dist = cv2.distanceTransform(inv_tgt, cv2.DIST_L2, 3)
    vals = dist[src].astype(np.float32).reshape(-1)
    return {
        "mean_px": float(vals.mean()),
        "p95_px": float(np.percentile(vals, 95)),
        "within_2px": float((vals <= 2.0).mean()),
    }


def _compute_mask_metrics(mask_pred: np.ndarray, mask_gt: np.ndarray) -> dict[str, float]:
    boundary_pred = _boundary_mask(mask_pred)
    boundary_gt = _boundary_mask(mask_gt)
    pred_to_gt = _boundary_distance_stats(boundary_pred, boundary_gt)
    gt_to_pred = _boundary_distance_stats(boundary_gt, boundary_pred)
    return {
        "iou": float(mask_iou(mask_pred, mask_gt)),
        "outside_ratio": float(_mask_outside_ratio(mask_pred, mask_gt)),
        "inside_coverage": float(_mask_inside_coverage(mask_pred, mask_gt)),
        "area_ratio": float(_mask_area_ratio(mask_pred, mask_gt)),
        "pred_to_gt_boundary_mean_px": float(pred_to_gt["mean_px"]),
        "pred_to_gt_boundary_p95_px": float(pred_to_gt["p95_px"]),
        "pred_boundary_within_2px": float(pred_to_gt["within_2px"]),
        "gt_to_pred_boundary_mean_px": float(gt_to_pred["mean_px"]),
        "gt_to_pred_boundary_p95_px": float(gt_to_pred["p95_px"]),
        "gt_boundary_within_2px": float(gt_to_pred["within_2px"]),
    }


def _summarize_reports(reports: list[dict]) -> dict[str, float]:
    if len(reports) <= 0:
        return {}
    metric_keys = sorted({
        str(k)
        for rep in reports
        for bucket_name in ("metrics_before", "metrics_after")
        for k in dict(rep.get(bucket_name, {})).keys()
    })
    out: dict[str, float] = {}
    for stage_name in ("before", "after"):
        bucket_name = f"metrics_{stage_name}"
        for key in metric_keys:
            vals = [
                float(dict(rep.get(bucket_name, {})).get(key))
                for rep in reports
                if key in dict(rep.get(bucket_name, {}))
            ]
            if len(vals) <= 0:
                continue
            arr = np.asarray(vals, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                continue
            out[f"{key}_{stage_name}_mean"] = float(arr.mean())
    return out


def _resolve_pred_source(stage1_dir: Path, pred_source: str) -> str:
    raw = str(pred_source or "auto").strip().lower()
    if raw in {"stage1", "smplestx_raw"}:
        return raw
    smplestx_meta = stage1_dir / "smplestx" / "selected_frames" / "selected_frames_meta.json"
    if smplestx_meta.exists():
        return "smplestx_raw"
    return "stage1"


def _load_stage1_mesh_faces(stage1_dir: Path, frame_name_ref: str) -> np.ndarray:
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
    return np.asarray(faces, dtype=np.int32)


def _load_eval_sequence(
    *,
    project_root: Path,
    stage1_dir: Path,
    assets_dir: Path,
    pred_source: str,
) -> tuple[str, list[str], np.ndarray, np.ndarray]:
    pred_use = _resolve_pred_source(stage1_dir=stage1_dir, pred_source=pred_source)
    if pred_use == "smplestx_raw":
        selected_result_dir = stage1_dir / "smplestx" / "selected_frames"
        meta_json = selected_result_dir / "selected_frames_meta.json"
        if not meta_json.exists():
            raise FileNotFoundError(f"Missing SMPLest-X selected-frame metadata: {meta_json}")
        selected_meta = json.loads(meta_json.read_text(encoding="utf-8"))
        ref_name = str(selected_meta.get("selected_files", ["000000.jpg"])[-1])
        faces = _load_stage1_mesh_faces(stage1_dir=stage1_dir, frame_name_ref=ref_name)
        records = _build_smplestx_frame_records_raw(
            selected_result_dir=selected_result_dir,
            selected_meta=selected_meta,
            faces_smplx=faces,
            assets_dir=assets_dir,
        )
        if len(records) <= 0:
            raise RuntimeError(f"No SMPLest-X records found under: {selected_result_dir}")
        frame_names = [str(rec["frame_name"]) for rec in records]
        verts_cam_all = np.asarray([np.asarray(rec["verts_cam"], dtype=np.float32) for rec in records], dtype=np.float32)
        return pred_use, frame_names, verts_cam_all, faces

    verts_npz = stage1_dir / "smpl_vertices_sequence.npz"
    if not verts_npz.exists():
        raise FileNotFoundError(f"Missing stage1 vertices sequence: {verts_npz}")
    seq = np.load(str(verts_npz), allow_pickle=True)
    if "frame_names" not in seq.files or "verts_cam" not in seq.files:
        raise RuntimeError(f"Bad vertices npz format: {verts_npz}")
    frame_names = [str(x) for x in seq["frame_names"].tolist()]
    verts_cam_all = np.asarray(seq["verts_cam"], dtype=np.float32)
    faces = _load_stage1_mesh_faces(stage1_dir=stage1_dir, frame_name_ref=frame_names[-1])
    return pred_use, frame_names, verts_cam_all, faces


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

    K = _shared_read_stage1_frame_camera_K(
        stage1_dir=stage1_dir,
        frame_index=int(idx),
        frame_name=frame_name,
        cam_json_path=cam_json_path,
        image_hw=(h, w),
    )
    uv, valid_v = _shared_project_vertices_to_image(verts_cam=verts_cam, K=K)
    smpl_mask_raw = _shared_rasterize_uv_mask(uv=uv, valid_v=valid_v, faces=faces, image_hw=(h, w))

    person_mask, person_mask_source = _shared_resolve_person_mask_source_np(
        gt_mask_mode=gt_mask_mode,
        pm_npz=pm_npz,
        frame_name=frame_name,
        image_hw=(h, w),
        img_bgr=img,
        smpl_mask_raw=smpl_mask_raw,
        modnet_ckpt_path=modnet_ckpt_path,
    )

    smpl_mask = smpl_mask_raw
    metrics_before = _compute_mask_metrics(smpl_mask_raw, person_mask)
    metrics_after = dict(metrics_before)
    align_report = {"applied": False, "accepted": False, "reason": "native_projection"}
    if bool(use_auto_align):
        uv_aligned, align_report = estimate_mask_similarity_transform(
            uv=uv,
            valid_v=valid_v,
            faces=faces,
            image_hw=(h, w),
            smpl_mask=smpl_mask_raw,
            person_mask=person_mask,
        )
        if bool(align_report.get("accepted", False)):
            smpl_mask = _shared_rasterize_uv_mask(uv=uv_aligned, valid_v=valid_v, faces=faces, image_hw=(h, w))
            metrics_after = _compute_mask_metrics(smpl_mask, person_mask)

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
        "iou_before": float(metrics_before["iou"]),
        "iou_after": float(metrics_after["iou"]),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "align": align_report,
        "gt_mask_source": str(person_mask_source),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Stage1 contour evaluation (GT vs Stage1 SMPL)")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--stage1_dir", type=str, default="outputs/stage1")
    parser.add_argument("--assets_dir", type=str, default="assets")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input")
    parser.add_argument("--out_dir", type=str, default="evaluation/eval_outputs/stage1")
    parser.add_argument("--pred_source", type=str, default="smplestx_raw", choices=["auto", "stage1", "smplestx_raw"])
    parser.add_argument("--frame_select", type=str, default="last", choices=["all", "first", "last", "index", "name"])
    parser.add_argument("--frame_index", type=int, default=-1)
    parser.add_argument("--frame_name", type=str, default="")
    parser.add_argument("--person_masks_npz", type=str, default="")
    parser.add_argument("--modnet_ckpt", type=str, default="")
    parser.add_argument(
        "--gt_mask_mode",
        type=str,
        default="modnet",
        choices=["auto", "npz", "chromakey", "modnet"],
        help="GT person mask source: auto(npz->chromakey->modnet), npz, chromakey, or modnet.",
    )
    parser.add_argument("--auto_align", action="store_true")
    parser.add_argument("--no_auto_align", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_use, frame_names, verts_cam_all, faces = _load_eval_sequence(
        project_root=project_root,
        stage1_dir=stage1_dir,
        assets_dir=assets_dir,
        pred_source=str(args.pred_source),
    )
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

    pm_npz = (stage1_dir / "person_masks.npz") if str(args.person_masks_npz).strip() == "" else Path(args.person_masks_npz)
    if not pm_npz.is_absolute():
        pm_npz = (project_root / pm_npz).resolve()
    modnet_ckpt_path = str(args.modnet_ckpt).strip() or None

    use_auto_align = True
    if bool(args.no_auto_align):
        use_auto_align = False
    elif bool(args.auto_align):
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
            f"IoU: {report_i['iou_before']:.4f} -> {report_i['iou_after']:.4f} | "
            f"Outside: {report_i['metrics_before']['outside_ratio']:.4f} -> {report_i['metrics_after']['outside_ratio']:.4f}"
        )

    if len(reports) == 1:
        report = dict(reports[0])
        report["stage1_dir"] = str(stage1_dir)
        report["frames_dir"] = str(frames_dir)
        report["pred_source"] = str(pred_use)
        report_path = out_dir / f"{Path(str(report['frame_name'])).stem}_stage1_eval_report.json"
    else:
        report = {
            "n_frames": len(reports),
            "frame_reports": reports,
            "summary": _summarize_reports(reports),
            "stage1_dir": str(stage1_dir),
            "frames_dir": str(frames_dir),
            "pred_source": str(pred_use),
        }
        report_path = out_dir / "stage1_eval_report_all.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

