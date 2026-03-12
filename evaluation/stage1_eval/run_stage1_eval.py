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
from avatar_gs.alignment.mask_sources import (
    project_vertices_to_image as _shared_project_vertices_to_image,
    rasterize_uv_mask as _shared_rasterize_uv_mask,
    read_stage1_frame_camera_K as _shared_read_stage1_frame_camera_K,
    resolve_person_mask_source_np as _shared_resolve_person_mask_source_np,
)
from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from person_contour import extract_person_contours


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
    iou_before = mask_iou(smpl_mask_raw, person_mask)
    iou_after = iou_before
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
            iou_after = mask_iou(smpl_mask, person_mask)

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

