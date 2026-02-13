from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from avatar_gs.gaussian.init_from_smpl import load_obj_mesh
from avatar_gs.utils.cfg import apply_overrides, load_yaml, parse_args_base
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.person_contour import extract_person_contours, load_person_mask_from_npz


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
        q2 = q if q.lower().endswith(".jpg") or q.lower().endswith(".png") else f"{q}.jpg"
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


def _rasterize_mesh_mask(
    uv: np.ndarray,
    valid_v: np.ndarray,
    faces: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
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


def _compute_contour_diagnostics(smpl_mask: np.ndarray, person_mask: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {}
    sm = (smpl_mask > 0)
    out["smpl_area_px"] = float(sm.sum())
    if person_mask is None:
        return out
    pm = (person_mask > 0)
    out["person_area_px"] = float(pm.sum())
    inter = np.logical_and(sm, pm).sum()
    union = np.logical_or(sm, pm).sum()
    out["iou"] = float(inter / max(union, 1))
    sm_out = np.logical_and(sm, ~pm).sum()
    out["smpl_outside_ratio"] = float(sm_out / max(sm.sum(), 1))

    b_s = _mask_bbox(smpl_mask)
    b_p = _mask_bbox(person_mask)
    if b_s is not None and b_p is not None:
        w_s = float(b_s[2] - b_s[0] + 1.0)
        h_s = float(b_s[3] - b_s[1] + 1.0)
        w_p = float(b_p[2] - b_p[0] + 1.0)
        h_p = float(b_p[3] - b_p[1] + 1.0)
        out["bbox_width_ratio_smpl_over_person"] = float(w_s / max(w_p, 1.0))
        out["bbox_height_ratio_smpl_over_person"] = float(h_s / max(h_p, 1.0))
        out["bbox_center_dx_px"] = float(0.5 * ((b_s[0] + b_s[2]) - (b_p[0] + b_p[2])))
        out["bbox_center_dy_px"] = float(0.5 * ((b_s[1] + b_s[3]) - (b_p[1] + b_p[3])))
    return out


def main():
    parser = parse_args_base("Stage1-only GT/SMPL contour alignment check")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage1_align_check", str(log_dir / "stage1_align_check.log"))

    viz_cfg = cfg.get("stage1_align_check", {})
    if not isinstance(viz_cfg, dict):
        viz_cfg = {}

    stage1_dir = project_root / Path(viz_cfg.get("stage1_dir", "outputs/stage1"))
    frames_dir = project_root / Path(viz_cfg.get("frames_dir", "data/frames/input"))
    out_image = project_root / Path(viz_cfg.get("out_image", "outputs/renders_align_check/000756_gt_smpl_contour.png"))
    save_smpl_mask = bool(viz_cfg.get("save_smpl_mask", False))
    save_person_mask = bool(viz_cfg.get("save_person_mask", False))

    verts_npz = stage1_dir / "smpl_vertices_sequence.npz"
    if not verts_npz.exists():
        raise FileNotFoundError(f"Missing stage1 vertices sequence: {verts_npz}")
    params_seq_npz = stage1_dir / "smpl_params_sequence.npz"
    keypoints_json = stage1_dir / "keypoints.json"
    person_masks_npz = stage1_dir / "person_masks.npz"
    stale_reasons: list[str] = []
    if params_seq_npz.exists():
        t_fit = params_seq_npz.stat().st_mtime
        if keypoints_json.exists() and keypoints_json.stat().st_mtime > t_fit + 1.0:
            stale_reasons.append("keypoints.json is newer than smpl_params_sequence.npz")
        if person_masks_npz.exists() and person_masks_npz.stat().st_mtime > t_fit + 1.0:
            stale_reasons.append("person_masks.npz is newer than smpl_params_sequence.npz")
    if len(stale_reasons) > 0:
        warn_msg = (
            "Stage1 artifacts look stale. Re-run stage1 fitting before alignment check. "
            + "; ".join(stale_reasons)
        )
        logger.warning(warn_msg)
        print(f"Warning: {warn_msg}")

    seq = np.load(str(verts_npz), allow_pickle=True)
    if "frame_names" not in seq.files or "verts_cam" not in seq.files:
        raise RuntimeError(f"Bad vertices npz format: {verts_npz}")
    frame_names = [str(x) for x in seq["frame_names"].tolist()]
    verts_cam_all = np.asarray(seq["verts_cam"], dtype=np.float32)
    if verts_cam_all.ndim != 3:
        raise RuntimeError(f"Bad verts_cam shape: {verts_cam_all.shape}")
    n_frames = int(verts_cam_all.shape[0])

    pick_mode = str(viz_cfg.get("frame_select", "last"))
    pick_index = int(viz_cfg.get("frame_index", -1))
    pick_name = viz_cfg.get("frame_name", None)
    idx = _resolve_pick_index(
        mode=pick_mode,
        n=n_frames,
        index=pick_index,
        frame_name=(None if pick_name is None else str(pick_name)),
        names=frame_names,
    )
    frame_name = frame_names[idx]

    img_path = frames_dir / frame_name
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read GT image: {img_path}")
    h, w = img.shape[:2]

    cam_json = stage1_dir / "cameras.json"
    cam = read_json(cam_json)
    K = np.asarray(cam.get("K", []), dtype=np.float32).reshape(3, 3)

    mesh_obj = stage1_dir / "smpl_mesh_tpose.obj"
    if not mesh_obj.exists():
        raise FileNotFoundError(f"Missing mesh for face topology: {mesh_obj}")
    _, faces = load_obj_mesh(mesh_obj)
    if faces is None or len(faces) <= 0:
        raise RuntimeError(f"No faces in mesh topology: {mesh_obj}")

    verts_cam = verts_cam_all[idx]
    uv, valid_v = _project_vertices_to_image(verts_cam=verts_cam, K=K)
    smpl_mask_raw = _rasterize_mesh_mask(uv=uv, valid_v=valid_v, faces=faces, image_hw=(h, w))

    person_mask_path_cfg = viz_cfg.get("person_masks_npz", None)
    if person_mask_path_cfg is None or str(person_mask_path_cfg).strip() == "":
        person_mask_npz = stage1_dir / "person_masks.npz"
    else:
        p = Path(str(person_mask_path_cfg))
        person_mask_npz = p if p.is_absolute() else (project_root / p)

    draw_person = bool(viz_cfg.get("draw_person_contour", True))
    person_mask = None
    if draw_person and person_mask_npz.exists():
        person_mask = load_person_mask_from_npz(
            npz_path=person_mask_npz,
            frame_name=frame_name,
            image_hw=(h, w),
        )
        if person_mask is not None and int(person_mask.sum()) > 0:
            person_cnt = extract_person_contours(person_mask)
            if len(person_cnt) > 0:
                pass

    smpl_mask = smpl_mask_raw
    smpl_cnt = extract_person_contours((smpl_mask > 0).astype(np.uint8) * 255)
    out = img.copy()
    if len(smpl_cnt) > 0:
        cv2.drawContours(
            out,
            smpl_cnt[:1],
            -1,
            color=(0, 0, 255),
            thickness=int(max(1, int(viz_cfg.get("smpl_thickness", 2)))),
        )
    if draw_person and person_mask is not None and int(person_mask.sum()) > 0:
        person_cnt = extract_person_contours(person_mask)
        if len(person_cnt) > 0:
            cv2.drawContours(
                out,
                person_cnt[:1],
                -1,
                color=(0, 255, 255),
                thickness=int(max(1, int(viz_cfg.get("person_thickness", 2)))),
            )

    out_image.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_image), out)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_image}")

    if save_smpl_mask:
        cv2.imwrite(str(out_image.with_name(out_image.stem + "_smpl_mask.png")), smpl_mask)
    if save_person_mask and person_mask is not None:
        cv2.imwrite(str(out_image.with_name(out_image.stem + "_person_mask.png")), person_mask)

    diag = _compute_contour_diagnostics(smpl_mask=smpl_mask, person_mask=person_mask)
    logger.info(f"Stage1 align check saved: {out_image}")
    logger.info(f"Frame: idx={idx}, name={frame_name}")
    logger.info(f"SMPL contour count: {len(smpl_cnt)}")
    if len(diag) > 0:
        logger.info(f"Alignment diagnostics: {diag}")
    print(f"Saved: {out_image}")
    print(f"Frame: idx={idx}, name={frame_name}")
    if len(diag) > 0:
        print(f"Diagnostics: {diag}")


if __name__ == "__main__":
    main()
