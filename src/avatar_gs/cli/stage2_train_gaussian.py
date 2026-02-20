from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.init_from_smpl import (
    init_gaussians_from_mesh_3d,
    load_obj_mesh,
    load_obj_vertices,
)
from avatar_gs.gaussian.rigid_align import (
    apply_rigid_transform_np,
    coordinate_descent_rigid_search,
    normalize_rigid_align_config,
)
from avatar_gs.gaussian.train_loop import train_gaussians_3d_learnable_camera
from avatar_gs.data.sequence_mesh import SequenceMeshProvider
try:
    from avatar_gs.data.h36m import resolve_h36m_image_paths
except Exception:
    resolve_h36m_image_paths = None


def _resolve_project_path(project_root: Path, path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(str(path_like))
    if not p.is_absolute():
        p = project_root / p
    return p


def _project_points_numpy(
    xyz: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc = xyz @ R.T + t.reshape(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = np.zeros((xyz.shape[0],), dtype=np.float32)
    v = np.zeros((xyz.shape[0],), dtype=np.float32)
    if np.any(valid):
        Xcv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (Xcv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (Xcv[:, 1] / zv) + K[1, 2]
    return u, v, valid


def _rasterize_mesh_mask(
    verts: np.ndarray,
    faces: np.ndarray | None,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if faces is None or len(faces) == 0 or verts.shape[0] == 0:
        return mask

    u, v, valid = _project_points_numpy(verts, K, R, t)
    if not np.any(valid):
        return mask

    uvi = np.stack([u, v], axis=1).astype(np.float32)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if i0 < 0 or i1 < 0 or i2 < 0:
            continue
        if i0 >= len(valid) or i1 >= len(valid) or i2 >= len(valid):
            continue
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        pts = uvi[[i0, i1, i2]]
        if (
            (pts[:, 0] < -2).all()
            or (pts[:, 0] > W + 1).all()
            or (pts[:, 1] < -2).all()
            or (pts[:, 1] > H + 1).all()
        ):
            continue
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(mask, poly, 1)
    return mask


def _maybe_pre_stage2_rigid_align(
    *,
    project_root: Path,
    stage2_cfg: dict,
    frames_dir: Path,
    frames: list[str],
    K_list: list[float],
    bind_mesh_dir: Path | None,
    bind_mesh_indices: list[int] | None,
    bind_seq_npz_path: Path | None,
    bind_vertices_npz_path: Path | None,
    cam_t_prior: tuple[float, float, float],
    out_dir: Path,
    logger,
) -> dict | None:
    raw_cfg = stage2_cfg.get("bind_rigid_align", None)
    cfg = normalize_rigid_align_config(raw_cfg, default_enabled=False)
    if cfg is None:
        return raw_cfg
    if not bool(cfg.get("enabled", False)):
        return cfg
    if not bool(cfg.get("auto", False)):
        return cfg
    if not bool(cfg.get("pre_stage2", True)):
        logger.info("bind_rigid_align.pre_stage2=false; skip pre-stage2 rigid search.")
        return cfg
    if bind_mesh_dir is None or bind_mesh_indices is None or len(bind_mesh_indices) == 0:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: binding inputs are unavailable. "
            "Will fallback to in-train auto align."
        )
        return cfg

    pm_cfg = cfg.get("person_masks_npz", None)
    if pm_cfg is None:
        pm_cfg = stage2_cfg.get("person_masks_npz", None)
    pm_path = _resolve_project_path(project_root, pm_cfg)
    if pm_path is None or (not pm_path.exists()):
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: person mask file missing. "
            "Will fallback to in-train auto align."
        )
        return cfg

    first_img = cv2.imread(str(frames_dir / frames[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: cannot read frame image. "
            "Will fallback to in-train auto align."
        )
        return cfg
    H, W = int(first_img.shape[0]), int(first_img.shape[1])
    if H <= 0 or W <= 0:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: invalid frame size. "
            "Will fallback to in-train auto align."
        )
        return cfg

    K_np = np.asarray(K_list, dtype=np.float32).reshape(3, 3)
    R_eval = np.eye(3, dtype=np.float32)
    t_eval = np.asarray(cam_t_prior, dtype=np.float32).reshape(3,)

    try:
        seq_provider = SequenceMeshProvider(bind_mesh_dir)
        seq_provider.scan()
    except Exception:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: no readable sequence OBJ meshes found. "
            "Will fallback to in-train auto align."
        )
        return cfg
    mesh_ids = seq_provider.get_frame_ids()
    mesh_ids_set = set(mesh_ids)
    fallback_mesh_idx = int(mesh_ids[-1])

    first_mesh_idx = int(bind_mesh_indices[0])
    if first_mesh_idx not in mesh_ids_set:
        first_mesh_idx = fallback_mesh_idx
    try:
        first_obj = seq_provider.get_path(first_mesh_idx)
    except Exception:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: first mesh OBJ missing. "
            "Will fallback to in-train auto align."
        )
        return cfg
    _, bind_faces_np = load_obj_mesh(first_obj)
    if bind_faces_np is None or len(bind_faces_np) == 0:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: mesh faces unavailable. "
            "Will fallback to in-train auto align."
        )
        return cfg

    verts_seq_np = None
    verts_seq_ids_set: set[int] = set()
    verts_seq_fallback_idx = 0
    if bind_vertices_npz_path is not None and bind_vertices_npz_path.exists():
        vseq = np.load(bind_vertices_npz_path, allow_pickle=True)
        if "verts_cam" in vseq.files:
            verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)
            if verts_seq_np.ndim == 3 and verts_seq_np.shape[-1] == 3:
                verts_seq_ids_set = set(range(int(verts_seq_np.shape[0])))
                verts_seq_fallback_idx = int(max(int(verts_seq_np.shape[0]) - 1, 0))
            else:
                verts_seq_np = None

    bind_mesh_obj_space_mode = str(stage2_cfg.get("bind_mesh_obj_space", "auto")).strip().lower()
    if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
        bind_mesh_obj_space_mode = "auto"
    mesh_obj_in_camera_space = bool(bind_mesh_obj_space_mode == "camera")
    bind_mesh_to_camera = bool(stage2_cfg.get("bind_mesh_to_camera", False))
    bind_inverse_export_upright = bool(stage2_cfg.get("bind_inverse_export_upright", True))
    exp_upright_rot = np.eye(3, dtype=np.float32)
    exp_center_xy = np.zeros((2,), dtype=np.float32)
    exp_floor_z = 0.0
    exp_has_center_xy = False
    exp_has_floor = False
    if bind_seq_npz_path is not None and bind_seq_npz_path.exists():
        seq = np.load(bind_seq_npz_path, allow_pickle=True)
        if bind_mesh_obj_space_mode == "auto" and "fit_strategy" in seq.files:
            fs_arr = np.asarray(seq["fit_strategy"]).reshape(-1)
            if fs_arr.size > 0 and str(fs_arr[0]).strip().lower() in ("stage1_motion_prior", "vibe"):
                mesh_obj_in_camera_space = True
        if "export_upright_rot" in seq.files:
            exp_upright_rot = np.asarray(seq["export_upright_rot"], dtype=np.float32).reshape(3, 3)
        if "export_center_xy" in seq.files:
            exp_has_center_xy = bool(int(np.asarray(seq["export_center_xy"]).reshape(-1)[0]))
        if "export_center_xy_value" in seq.files:
            exp_center_xy = np.asarray(seq["export_center_xy_value"], dtype=np.float32).reshape(2,)
        if "export_on_floor" in seq.files:
            exp_has_floor = bool(int(np.asarray(seq["export_on_floor"]).reshape(-1)[0]))
        if "export_floor_z" in seq.files:
            exp_floor_z = float(np.asarray(seq["export_floor_z"], dtype=np.float32).reshape(-1)[0])

    bind_mesh_offset_cfg = stage2_cfg.get("bind_mesh_offset", (0.0, 0.0, 0.0))
    if not (isinstance(bind_mesh_offset_cfg, (list, tuple)) and len(bind_mesh_offset_cfg) == 3):
        bind_mesh_offset_cfg = (0.0, 0.0, 0.0)
    bind_mesh_offset_np = np.asarray(
        [[
            float(bind_mesh_offset_cfg[0]),
            float(bind_mesh_offset_cfg[1]),
            float(bind_mesh_offset_cfg[2]),
        ]],
        dtype=np.float32,
    )

    mesh_raw_cache: dict[int, np.ndarray] = {}

    def _resolve_mesh_idx(mesh_idx: int) -> int:
        mesh_idx_use = int(mesh_idx)
        if verts_seq_np is not None:
            if mesh_idx_use not in verts_seq_ids_set:
                mesh_idx_use = verts_seq_fallback_idx
        else:
            if mesh_idx_use not in mesh_ids_set:
                mesh_idx_use = fallback_mesh_idx
        return mesh_idx_use

    def _load_mesh_np_raw(mesh_idx: int) -> np.ndarray:
        mesh_idx_use = _resolve_mesh_idx(mesh_idx)
        if mesh_idx_use in mesh_raw_cache:
            return mesh_raw_cache[mesh_idx_use]
        if verts_seq_np is not None:
            verts_np = np.asarray(verts_seq_np[mesh_idx_use], dtype=np.float32)
        else:
            obj_path = seq_provider.get_path(mesh_idx_use)
            verts_np = load_obj_vertices(obj_path)
            if bind_mesh_to_camera and (not bool(mesh_obj_in_camera_space)):
                if bind_inverse_export_upright:
                    if exp_has_center_xy:
                        verts_np[:, 0] += float(exp_center_xy[0])
                        verts_np[:, 1] += float(exp_center_xy[1])
                    if exp_has_floor:
                        verts_np[:, 2] += float(exp_floor_z)
                    verts_np = (verts_np @ exp_upright_rot.astype(np.float32))
                verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
        mesh_raw_cache[mesh_idx_use] = np.asarray(verts_np, dtype=np.float32)
        return mesh_raw_cache[mesh_idx_use]

    pdata = np.load(pm_path, allow_pickle=True)
    if "frame_names" not in pdata.files or "masks" not in pdata.files:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: person mask NPZ missing frame_names/masks. "
            "Will fallback to in-train auto align."
        )
        return cfg
    p_names = [str(x) for x in pdata["frame_names"].tolist()]
    p_masks = np.asarray(pdata["masks"], dtype=np.uint8)
    if p_masks.ndim != 3 or len(p_names) != int(p_masks.shape[0]):
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: invalid mask tensor shape. "
            "Will fallback to in-train auto align."
        )
        return cfg
    def _norm_frame_name(name: str) -> str:
        s = str(name).strip().replace("\\", "/")
        return Path(s).name.lower()

    # Robust matching: support with/without extension and case differences.
    name_to_pm_exact: dict[str, int] = {}
    name_to_pm_stem: dict[str, int] = {}
    for i, n in enumerate(p_names):
        nn = _norm_frame_name(n)
        name_to_pm_exact[nn] = int(i)
        stem = Path(nn).stem
        if stem and stem not in name_to_pm_stem:
            name_to_pm_stem[stem] = int(i)

    def _lookup_pm_idx(frame_name: str) -> int:
        nn = _norm_frame_name(frame_name)
        if nn in name_to_pm_exact:
            return int(name_to_pm_exact[nn])
        stem = Path(nn).stem
        if stem in name_to_pm_stem:
            return int(name_to_pm_stem[stem])
        return -1

    valid_frame_ids = [i for i, fn in enumerate(frames) if _lookup_pm_idx(fn) >= 0]
    if len(valid_frame_ids) == 0:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: no overlapping frame names with person masks. "
            "Will fallback to in-train auto align."
        )
        return cfg

    sample_n = min(int(cfg.get("sample_frames", 8)), len(valid_frame_ids))
    pick = np.linspace(0, len(valid_frame_ids) - 1, num=sample_n, dtype=np.float64)
    pick = np.clip(np.round(pick).astype(np.int64), 0, len(valid_frame_ids) - 1)
    pick = np.unique(pick)
    sample_ids = [valid_frame_ids[int(k)] for k in pick.tolist()]

    ds = float(cfg.get("eval_downsample", 0.5))
    Hs = max(8, int(round(float(H) * ds)))
    Ws = max(8, int(round(float(W) * ds)))
    sx = float(Ws) / float(W)
    sy = float(Hs) / float(H)
    K_eval = np.asarray(K_np, dtype=np.float32).copy()
    K_eval[0, 0] *= sx
    K_eval[0, 2] *= sx
    K_eval[1, 1] *= sy
    K_eval[1, 2] *= sy
    outside_w = float(cfg.get("outside_weight", 2.0))
    gt_masks_eval: dict[int, np.ndarray] = {}
    for fj in sample_ids:
        pm_i = int(_lookup_pm_idx(frames[fj]))
        if pm_i < 0:
            continue
        gt_m = (p_masks[pm_i] > 0).astype(np.uint8)
        if Hs != H or Ws != W:
            gt_m = cv2.resize(gt_m, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        gt_masks_eval[fj] = (gt_m > 0)
    if len(gt_masks_eval) == 0:
        logger.warning(
            "bind_rigid_align pre-stage2 skipped: sampled masks are empty after frame-name matching. "
            "Will fallback to in-train auto align."
        )
        return cfg

    def _eval_score(param_dict: dict) -> float:
        cfg_eval = {"enabled": True}
        cfg_eval.update(param_dict)
        acc = 0.0
        used = 0
        for fj in sample_ids:
            mesh_i = int(bind_mesh_indices[fj])
            verts_raw = _load_mesh_np_raw(mesh_i)
            verts_eval = apply_rigid_transform_np(verts_raw, cfg_eval) + bind_mesh_offset_np
            pred = _rasterize_mesh_mask(
                verts=verts_eval,
                faces=bind_faces_np,
                K=K_eval,
                R=R_eval,
                t=t_eval,
                H=Hs,
                W=Ws,
            ) > 0
            gt_m = gt_masks_eval[fj]
            inter = float(np.logical_and(pred, gt_m).sum())
            union = float(np.logical_or(pred, gt_m).sum())
            iou = inter / (union + 1e-6)
            pred_area = float(pred.sum())
            outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
            acc += (iou - outside_w * outside)
            used += 1
        if used <= 0:
            return -1e9
        return float(acc / float(used))

    def _eval_metrics(param_dict: dict) -> tuple[float, float, float]:
        cfg_eval = {"enabled": True}
        cfg_eval.update(param_dict)
        iou_acc = 0.0
        outside_acc = 0.0
        score_acc = 0.0
        used = 0
        for fj in sample_ids:
            mesh_i = int(bind_mesh_indices[fj])
            verts_raw = _load_mesh_np_raw(mesh_i)
            verts_eval = apply_rigid_transform_np(verts_raw, cfg_eval) + bind_mesh_offset_np
            pred = _rasterize_mesh_mask(
                verts=verts_eval,
                faces=bind_faces_np,
                K=K_eval,
                R=R_eval,
                t=t_eval,
                H=Hs,
                W=Ws,
            ) > 0
            gt_m = gt_masks_eval[fj]
            inter = float(np.logical_and(pred, gt_m).sum())
            union = float(np.logical_or(pred, gt_m).sum())
            iou = inter / (union + 1e-6)
            pred_area = float(pred.sum())
            outside = float(np.logical_and(pred, ~gt_m).sum()) / (pred_area + 1e-6)
            score = iou - outside_w * outside
            iou_acc += iou
            outside_acc += outside
            score_acc += score
            used += 1
        if used <= 0:
            return 0.0, 1.0, -1e9
        n = float(used)
        return float(iou_acc / n), float(outside_acc / n), float(score_acc / n)

    def _mask_bbox(mask_bool: np.ndarray) -> tuple[float, float, float, float] | None:
        ys, xs = np.where(mask_bool > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    init_params = {
        "rx_deg": float(cfg.get("rx_deg", 0.0)),
        "ry_deg": float(cfg.get("ry_deg", 0.0)),
        "rz_deg": float(cfg.get("rz_deg", 0.0)),
        "tx": float(cfg.get("tx", 0.0)),
        "ty": float(cfg.get("ty", 0.0)),
        "tz": float(cfg.get("tz", 0.0)),
    }

    # Stage1-style bbox alignment prior (2D contour center/scale) to improve rigid-search init.
    bbox_init_applied = False
    if bool(cfg.get("init_from_bbox", True)) and len(sample_ids) > 0:
        fj0 = int(sample_ids[0])
        mesh_i0 = int(bind_mesh_indices[fj0])
        verts_raw0 = _load_mesh_np_raw(mesh_i0)
        verts_eval0 = apply_rigid_transform_np(verts_raw0, {"enabled": True, **init_params}) + bind_mesh_offset_np
        pred0 = _rasterize_mesh_mask(
            verts=verts_eval0,
            faces=bind_faces_np,
            K=K_eval,
            R=R_eval,
            t=t_eval,
            H=Hs,
            W=Ws,
        ) > 0
        gt0 = gt_masks_eval.get(fj0, None)
        if gt0 is not None:
            b_s = _mask_bbox(pred0)
            b_p = _mask_bbox(gt0)
            if b_s is not None and b_p is not None:
                ws = max(1.0, b_s[2] - b_s[0] + 1.0)
                hs = max(1.0, b_s[3] - b_s[1] + 1.0)
                wp = max(1.0, b_p[2] - b_p[0] + 1.0)
                hp = max(1.0, b_p[3] - b_p[1] + 1.0)
                csx = 0.5 * (b_s[0] + b_s[2])
                csy = 0.5 * (b_s[1] + b_s[3])
                cpx = 0.5 * (b_p[0] + b_p[2])
                cpy = 0.5 * (b_p[1] + b_p[3])
                s = float(np.clip(0.5 * ((wp / ws) + (hp / hs)), 0.5, 2.0))

                verts_cam0 = (verts_eval0 @ R_eval.T) + t_eval.reshape(1, 3)
                z_ref = float(np.median(np.clip(verts_cam0[:, 2], 1e-3, None)))
                fx = max(abs(float(K_eval[0, 0])), 1e-6)
                fy = max(abs(float(K_eval[1, 1])), 1e-6)
                dx_px = float(cpx - csx)
                dy_px = float(cpy - csy)

                init_params["tx"] += (dx_px / fx) * z_ref
                init_params["ty"] += (dy_px / fy) * z_ref
                init_params["tz"] += (1.0 / max(s, 1e-6) - 1.0) * z_ref
                bbox_init_applied = True

    max_trans_cfg = tuple(cfg.get("max_trans", (0.35, 0.35, 0.45)))
    if len(max_trans_cfg) != 3:
        max_trans_cfg = (0.35, 0.35, 0.45)
    init_params["tx"] = float(np.clip(init_params["tx"], -float(max_trans_cfg[0]), float(max_trans_cfg[0])))
    init_params["ty"] = float(np.clip(init_params["ty"], -float(max_trans_cfg[1]), float(max_trans_cfg[1])))
    init_params["tz"] = float(np.clip(init_params["tz"], -float(max_trans_cfg[2]), float(max_trans_cfg[2])))

    best_params, score_before, score_after = coordinate_descent_rigid_search(
        score_fn=_eval_score,
        init_params=init_params,
        rot_step_deg=float(cfg.get("rot_step_deg", 3.0)),
        trans_step=tuple(cfg.get("trans_step", (0.04, 0.04, 0.06))),
        levels=int(cfg.get("levels", 3)),
        iters_per_level=int(cfg.get("iters_per_level", 2)),
        max_rot_deg=float(cfg.get("max_rot_deg", 20.0)),
        max_trans=max_trans_cfg,
    )
    iou_before, outside_before, _ = _eval_metrics(init_params)
    iou_after, outside_after, _ = _eval_metrics(best_params)

    min_iou_after = float(cfg.get("min_iou_after", 0.70))
    max_outside_after = float(cfg.get("max_outside_after", 0.20))
    require_quality = bool(cfg.get("require_quality", True))
    quality_ok = bool((iou_after >= min_iou_after) and (outside_after <= max_outside_after))

    logger.info(
        "bind_rigid_align pre-stage2 auto: "
        f"score {score_before:.4f}->{score_after:.4f}, "
        f"iou {iou_before:.4f}->{iou_after:.4f}, "
        f"outside {outside_before:.4f}->{outside_after:.4f}, "
        f"rot=({best_params['rx_deg']:.3f},{best_params['ry_deg']:.3f},{best_params['rz_deg']:.3f})deg, "
        f"trans=({best_params['tx']:.4f},{best_params['ty']:.4f},{best_params['tz']:.4f}), "
        f"quality_ok={quality_ok} (iou>={min_iou_after:.2f}, outside<={max_outside_after:.2f})"
    )
    use_best_if_quality_fail = bool(cfg.get("use_best_if_quality_fail", True))
    if require_quality and (not quality_ok):
        if not use_best_if_quality_fail:
            raise RuntimeError(
                "pre-stage2 rigid align quality check failed: "
                f"iou_after={iou_after:.4f} (min {min_iou_after:.4f}), "
                f"outside_after={outside_after:.4f} (max {max_outside_after:.4f})"
            )
        logger.warning(
            "pre-stage2 rigid align quality below threshold, but use_best_if_quality_fail=true; "
            "continue training with best-found rigid params. "
            f"iou_after={iou_after:.4f} (min {min_iou_after:.4f}), "
            f"outside_after={outside_after:.4f} (max {max_outside_after:.4f})"
        )

    report = {
        "enabled": True,
        "auto": True,
        "pre_stage2": True,
        "params": {k: float(v) for k, v in best_params.items()},
        "score_before": float(score_before),
        "score_after": float(score_after),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "outside_before": float(outside_before),
        "outside_after": float(outside_after),
        "bbox_init_applied": bool(bbox_init_applied),
        "require_quality": bool(require_quality),
        "use_best_if_quality_fail": bool(use_best_if_quality_fail),
        "quality_ok": bool(quality_ok),
        "min_iou_after": float(min_iou_after),
        "max_outside_after": float(max_outside_after),
        "sample_frames": [str(frames[i]) for i in sample_ids],
        "person_masks_npz": str(pm_path),
    }
    report_path = out_dir / "bind_rigid_align_pre_stage2.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved pre-stage2 rigid align report: {report_path}")

    out_cfg = dict(cfg)
    out_cfg.update(best_params)
    out_cfg["enabled"] = True
    out_cfg["auto"] = False  # already solved before training starts
    return out_cfg


def main():
    parser = parse_args_base("Stage2: 3D Gaussian Splatting training with learnable camera (no COLMAP)")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage2_train_gaussian", str(log_dir / "stage2_train_gaussian.log"))

    stage1_dir = project_root / cfg["stage2"]["stage1_dir"]
    out_dir = ensure_dir(project_root / cfg["stage2"]["out_dir"])
    stage2_cfg = cfg["stage2"]

    if torch.cuda.is_available() and bool(stage2_cfg.get("enable_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(stage2_cfg.get("cudnn_benchmark", True))

    init_mesh_cfg = str(stage2_cfg.get("init_mesh_obj", ""))
    mesh_obj = Path(init_mesh_cfg)
    if not mesh_obj.is_absolute():
        cand_paths = [
            Path(init_mesh_cfg),
            project_root / mesh_obj,
            stage1_dir / mesh_obj,
            stage1_dir / mesh_obj.name,
        ]
        picked = None
        for cp in cand_paths:
            try:
                if cp.exists():
                    picked = cp
                    break
            except Exception:
                continue
        if picked is not None:
            mesh_obj = picked
        else:
            mesh_obj = project_root / mesh_obj
    if (not str(init_mesh_cfg).strip()) or (not mesh_obj.exists()):
        fallback_mesh = None
        tpose_cands = sorted(stage1_dir.glob("*_t-pose.obj"))
        if len(tpose_cands) > 0:
            fallback_mesh = tpose_cands[-1]
        else:
            for cand in (stage1_dir / "smpl_mesh_tpose.obj", stage1_dir / "smpl_mesh.obj"):
                if cand.exists():
                    fallback_mesh = cand
                    break
            # Also support stage1 numeric sequence exports: outputs/stage1/sequence/*.obj.
            if fallback_mesh is None:
                seq_dir = stage1_dir / "sequence"
                if seq_dir.exists():
                    try:
                        seq_provider = SequenceMeshProvider(seq_dir)
                        seq_provider.scan()
                        seq_ids = seq_provider.get_frame_ids()
                        fallback_mesh = seq_provider.get_path(seq_ids[-1])
                    except Exception:
                        fallback_mesh = None
        if fallback_mesh is None:
            raise FileNotFoundError(
                f"init mesh not found: {mesh_obj}. "
                f"Checked fallback: {stage1_dir / '*_t-pose.obj'}, {stage1_dir / 'smpl_mesh_tpose.obj'}, {stage1_dir / 'smpl_mesh.obj'}, "
                f"{stage1_dir / 'sequence' / '*.obj'}."
            )
        mesh_obj = fallback_mesh

    cam = read_json(stage1_dir / "cameras.json")

    frames_dir = project_root / cfg["data"]["frames_dir"]
    n_frames = int(stage2_cfg.get("n_frames", 20))

    if "h36m" in cfg.get("data", {}):
        if resolve_h36m_image_paths is None:
            raise ModuleNotFoundError(
                "data.h36m config is provided, but avatar_gs.data.h36m is unavailable."
            )
        paths = resolve_h36m_image_paths(project_root=project_root, data_cfg=cfg["data"])
        # Use the actual image root to load files by name.
        frames_dir = paths[0].parent
        frames = [p.name for p in paths]
    else:
        frames = sorted([p.name for p in frames_dir.glob("*.jpg")])
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Frame selection strategy
    frame_select = str(stage2_cfg.get("frame_select", "first")).lower()
    if frame_select == "stage1_sequence":
        seq_npz_cfg = stage2_cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz"))
        seq_npz_path = _resolve_project_path(project_root, seq_npz_cfg)
        if seq_npz_path is None or (not seq_npz_path.exists()):
            raise FileNotFoundError(
                f"frame_select=stage1_sequence requires bind_sequence_npz, missing: {seq_npz_path}"
            )
        seq = np.load(seq_npz_path, allow_pickle=True)
        if "frame_names" not in seq.files:
            raise RuntimeError(f"bind_sequence_npz has no frame_names: {seq_npz_path}")
        seq_names = [str(x) for x in seq["frame_names"].tolist()]
        frame_set = set(frames)
        picked = [n for n in seq_names if n in frame_set]
        if len(picked) == 0:
            # basename fallback
            by_name = {Path(f).name: f for f in frames}
            picked = [by_name[Path(n).name] for n in seq_names if Path(n).name in by_name]
        if len(picked) == 0:
            raise RuntimeError(
                "frame_select=stage1_sequence matched zero frames between data/frames and stage1 sequence frame_names."
            )
        frames = picked
    elif frame_select == "last":
        frames = [frames[-1]]
    elif frame_select == "last_k":
        k = max(1, int(n_frames))
        frames = frames[-k:]
    elif frame_select == "uniform_k":
        k = max(1, int(n_frames))
        if k >= len(frames):
            picked_idx = np.arange(len(frames), dtype=np.int64)
        else:
            idx = np.linspace(0, len(frames) - 1, num=k, dtype=np.float64)
            idx = np.clip(np.round(idx).astype(np.int64), 0, len(frames) - 1)
            picked_idx = np.unique(idx)
        frames = [frames[int(i)] for i in picked_idx.tolist()]
    elif frame_select == "indices":
        indices = stage2_cfg.get("frame_indices", [])
        if isinstance(indices, (int, np.integer, float, np.floating, str)):
            indices = [indices]
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise ValueError("frame_select=indices but frame_indices is empty.")
        picked = []
        for idx in indices:
            i = int(idx)
            if i < 0:
                i = len(frames) + i
            if i < 0 or i >= len(frames):
                raise IndexError(f"frame index out of range: {idx}")
            picked.append(frames[i])
        frames = picked
    elif frame_select == "first":
        # default: first N frames
        if n_frames > 0:
            frames = frames[:n_frames]
    else:
        raise ValueError(f"Unsupported frame_select: {frame_select}. Use first|last|last_k|uniform_k|indices|stage1_sequence")

    logger.info(f"Mesh: {mesh_obj}")
    logger.info(f"Frames: {len(frames)} used for training")
    logger.info(f"Out dir: {out_dir}")

    camera_mode = str(cfg["stage2"].get("camera_mode", "learnable"))
    lr_cam = float(cfg["stage2"].get("lr_cam", 1e-3))
    cam_reg_weight = float(cfg["stage2"].get("cam_reg_weight", 0.0))
    if len(frames) == 1 and camera_mode.lower() != "fixed" and cam_reg_weight <= 0.0:
        logger.warning(
            "Single-frame training with learnable camera and cam_reg_weight<=0 "
            "is ill-posed; switching to fixed camera."
        )
        camera_mode = "fixed"
        lr_cam = 0.0

    # Frame sampling / loss weights (emphasize last frame)
    frame_weights = stage2_cfg.get("frame_weights", None)
    frame_loss_weights = stage2_cfg.get("frame_loss_weights", None)
    last_sample_w = stage2_cfg.get("last_frame_sample_weight", None)
    last_loss_w = stage2_cfg.get("last_frame_loss_weight", None)
    if frame_weights is None and last_sample_w is not None:
        frame_weights = [1.0] * len(frames)
        frame_weights[-1] = float(last_sample_w)
    if frame_loss_weights is None and last_loss_w is not None:
        frame_loss_weights = [1.0] * len(frames)
        frame_loss_weights[-1] = float(last_loss_w)

    g = init_gaussians_from_mesh_3d(
        mesh_obj=mesh_obj,
        n_keep=int(cfg["stage2"].get("n_gaussians", 3000)),
        seed=int(cfg.get("seed", 42)),
        n_bg=int(cfg["stage2"].get("n_gaussians_bg", 0)),
        bg_extent=tuple(cfg["stage2"].get("bg_extent", (3.0, 3.0, 0.6))),
        bg_center_z=float(cfg["stage2"].get("bg_center_z", 2.5)),
        bg_log_scale=float(cfg["stage2"].get("bg_log_scale", -3.5)),
        bg_opacity_logit=float(cfg["stage2"].get("bg_opacity_logit", -1.5)),
        bg_exclude_bbox_pad=float(cfg["stage2"].get("bg_exclude_bbox_pad", 0.0)),
        sample_on_surface=bool(cfg["stage2"].get("sample_on_surface", True)),
    )
    logger.info(f"Initialized gaussians: N={int(g.xyz.shape[0])}")

    # ---- Optional dynamic binding to SMPL mesh sequence ----
    bind_to_mesh = bool(cfg["stage2"].get("bind_to_mesh", False))
    bind_mesh_dir = None
    bind_mesh_indices = None
    bind_seq_npz_path = None
    bind_vertices_npz_path = None
    bind_tpose_obj_path = None
    if bind_to_mesh:
        if g.bind_idx is None:
            raise RuntimeError("bind_to_mesh=true but gaussians have no bind_idx.")
        bind_mesh_dir_cfg = cfg["stage2"].get("bind_mesh_sequence_dir", str(stage1_dir))
        bind_mesh_dir = project_root / Path(bind_mesh_dir_cfg)
        # If stage1 produced a `sequence` subfolder with numeric OBJs, prefer it.
        seq_dir_candidate = stage1_dir / "sequence"
        if seq_dir_candidate.exists():
            bind_mesh_dir = seq_dir_candidate
            logger.info(f"Using bind mesh sequence dir: {bind_mesh_dir}")
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(
                f"bind_mesh_sequence_dir not found: {bind_mesh_dir}. "
                "Run stage1 to generate numeric OBJ meshes under outputs/stage1."
            )

        seq_npz_cfg = cfg["stage2"].get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz"))
        bind_seq_npz_path = _resolve_project_path(project_root, seq_npz_cfg)
        seq_provider = SequenceMeshProvider(bind_mesh_dir)
        seq_provider.scan()
        seq_frame_ids = seq_provider.get_frame_ids()
        seq_avail_set = set(seq_frame_ids)
        seq_fallback_idx = int(seq_frame_ids[-1])
        if bind_seq_npz_path is None or (not bind_seq_npz_path.exists()):
            raise FileNotFoundError(
                f"Missing bind sequence npz: {bind_seq_npz_path}. "
                "Required to map frame names to mesh indices for binding."
            )
        verts_npz_cfg = cfg["stage2"].get("bind_vertices_npz", str(stage1_dir / "smpl_vertices_sequence.npz"))
        verts_npz = _resolve_project_path(project_root, verts_npz_cfg)
        if verts_npz is not None and verts_npz.exists():
            bind_vertices_npz_path = verts_npz
            logger.info(f"Using dynamic bind vertices from: {bind_vertices_npz_path}")
        else:
            logger.warning(
                f"Missing {verts_npz}; binding will fall back to OBJ sequence only."
            )
        tpose_obj_cfg = cfg["stage2"].get("bind_tpose_obj", str(mesh_obj))
        tpose_obj_path = _resolve_project_path(project_root, tpose_obj_cfg)
        if tpose_obj_path is not None and tpose_obj_path.exists():
            bind_tpose_obj_path = tpose_obj_path
        else:
            logger.warning(f"bind_tpose_obj not found: {tpose_obj_path}. Render will use fallback.")

        seq = np.load(bind_seq_npz_path, allow_pickle=True)
        seq_names = [str(x) for x in seq["frame_names"].tolist()]
        name_to_idx = {n: i for i, n in enumerate(seq_names)}

        missing = []
        bind_mesh_indices = []
        for fn in frames:
            if fn in name_to_idx:
                bind_mesh_indices.append(int(name_to_idx[fn]))
            else:
                missing.append(fn)

        if missing:
            # If there are numeric OBJ files in bind_mesh_dir (e.g. outputs/stage1/sequence),
            # prefer a sequential fallback mapping rather than failing hard.
            if len(seq_frame_ids) > 0:
                logger.warning(
                    "Some frame names not found in smpl_params_sequence.npz; "
                    "falling back to sequential mesh indices from bind_mesh_sequence_dir."
                )
                bind_mesh_indices = list(range(len(frames)))
            elif len(seq_names) == len(frames):
                logger.warning(
                    "Some frame names not found in smpl_params_sequence.npz; "
                    "falling back to sequential mesh indices."
                )
                bind_mesh_indices = list(range(len(frames)))
            else:
                raise RuntimeError(
                    f"Binding failed: {len(missing)} frames not found in smpl_params_sequence.npz. "
                    f"Example missing: {missing[0]}"
                )

        if bind_vertices_npz_path is not None:
            vseq = np.load(bind_vertices_npz_path, allow_pickle=True)
            if "verts_cam" in vseq.files:
                n_seq = int(np.asarray(vseq["verts_cam"]).shape[0])
                avail_set = set(range(n_seq))
                fallback_idx = max(n_seq - 1, 0)
                missing_seq = [mi for mi in bind_mesh_indices if mi not in avail_set]
                if len(missing_seq) > 0:
                    logger.warning(
                        "Some bind indices are out of range in smpl_vertices_sequence; "
                        f"falling back to frame {fallback_idx} for {len(missing_seq)} frames."
                    )
                    bind_mesh_indices = [mi if mi in avail_set else fallback_idx for mi in bind_mesh_indices]
            else:
                logger.warning(
                    f"{bind_vertices_npz_path} has no verts_cam; fallback to OBJ availability checks."
                )
                bind_vertices_npz_path = None
        if bind_vertices_npz_path is None:
            avail_set = set(seq_avail_set)
            fallback_idx = int(seq_fallback_idx)
            missing_mesh = [mi for mi in bind_mesh_indices if mi not in avail_set]
            if len(missing_mesh) > 0:
                logger.warning(
                    "Some bind mesh indices are missing from sequence; "
                    f"falling back to frame_{fallback_idx:06d}.obj for {len(missing_mesh)} frames."
                )
                bind_mesh_indices = [mi if mi in avail_set else fallback_idx for mi in bind_mesh_indices]

    cam_t_prior = tuple(cfg["stage2"].get("cam_t_prior", (0.0, 0.0, 0.0)))
    if (
        bind_to_mesh
        and bool(cfg["stage2"].get("bind_mesh_to_camera", False))
        and camera_mode.lower() == "fixed"
        and len(cam_t_prior) == 3
        and abs(float(cam_t_prior[0])) < 1e-8
        and abs(float(cam_t_prior[1])) < 1e-8
        and abs(float(cam_t_prior[2]) - 2.5) < 1e-8
    ):
        logger.warning(
            "cam_t_prior is [0,0,2.5] with bind_mesh_to_camera=true and fixed camera; "
            "this causes projection shrink/shift. Auto-correcting cam_t_prior to [0,0,0]."
        )
        cam_t_prior = (0.0, 0.0, 0.0)

    bind_rigid_align_for_train = cfg["stage2"].get("bind_rigid_align", None)
    _bind_cfg_norm = normalize_rigid_align_config(bind_rigid_align_for_train, default_enabled=False)
    require_pre_stage2_align = bool((_bind_cfg_norm or {}).get("require_pre_stage2", True)) if bool((_bind_cfg_norm or {}).get("enabled", False)) else False
    try:
        bind_rigid_align_for_train = _maybe_pre_stage2_rigid_align(
            project_root=project_root,
            stage2_cfg=stage2_cfg,
            frames_dir=frames_dir,
            frames=frames,
            K_list=cam["K"],
            bind_mesh_dir=bind_mesh_dir,
            bind_mesh_indices=bind_mesh_indices,
            bind_seq_npz_path=bind_seq_npz_path,
            bind_vertices_npz_path=bind_vertices_npz_path,
            cam_t_prior=cam_t_prior,
            out_dir=out_dir,
            logger=logger,
        )
    except Exception as e:
        if require_pre_stage2_align:
            raise RuntimeError(f"bind_rigid_align pre-stage2 failed and require_pre_stage2=true: {e}") from e
        logger.warning(
            "bind_rigid_align pre-stage2 failed (%s); fallback to in-train align.",
            str(e),
        )
        bind_rigid_align_for_train = cfg["stage2"].get("bind_rigid_align", None)

    _eff_norm = normalize_rigid_align_config(bind_rigid_align_for_train, default_enabled=False)
    if require_pre_stage2_align and bool((_eff_norm or {}).get("enabled", False)) and bool((_eff_norm or {}).get("auto", False)):
        raise RuntimeError("require_pre_stage2=true but pre-stage2 rigid alignment did not complete; refusing to start stage2 training.")

    if bind_rigid_align_for_train is not None:
        cfg["stage2"]["bind_rigid_align_effective"] = bind_rigid_align_for_train

    ckpt_path = out_dir / "gaussians3d.pt"
    train_gaussians_3d_learnable_camera(
        g=g,
        frames_dir=frames_dir,
        frames_list=frames,
        K_list=cam["K"],
        out_ckpt=ckpt_path,
        iters=int(cfg["stage2"].get("iters", 2000)),
        lr_gauss=float(cfg["stage2"].get("lr_gauss", 1e-2)),
        lr_cam=lr_cam,
        bg_rgb=tuple(cfg["stage2"].get("bg_rgb", (1.0, 1.0, 1.0))),
        cam_warmup_iters=int(cfg["stage2"].get("cam_warmup_iters", 5000)),
        l1_weight=float(cfg["stage2"].get("l1_weight", 0.25)),
        use_alpha_mask=bool(cfg["stage2"].get("use_alpha_mask", True)),
        alpha_threshold=float(cfg["stage2"].get("alpha_threshold", 0.15)),
        camera_mode=camera_mode,
        cam_t_prior=cam_t_prior,
        cam_reg_weight=cam_reg_weight,
        keypoints_json=cfg["stage2"].get("keypoints_json", str(stage1_dir / "keypoints.json")),
        kpt_conf_min=float(cfg["stage2"].get("kpt_conf_min", 0.2)),
        kpt_box_pad=float(cfg["stage2"].get("kpt_box_pad", 0.15)),
        use_kpt_bbox_mask=bool(cfg["stage2"].get("use_kpt_bbox_mask", True)),
        face_weight=float(cfg["stage2"].get("face_weight", 1.0)),
        face_conf_min=float(cfg["stage2"].get("face_conf_min", 0.3)),
        face_box_pad=float(cfg["stage2"].get("face_box_pad", 0.2)),
        face_use_ears=bool(cfg["stage2"].get("face_use_ears", False)),
        frame_weights=frame_weights,
        frame_loss_weights=frame_loss_weights,
        temporal_ref_window=int(cfg["stage2"].get("temporal_ref_window", 0)),
        temporal_ref_decay=float(cfg["stage2"].get("temporal_ref_decay", 0.85)),
        temporal_ref_anchor=str(cfg["stage2"].get("temporal_ref_anchor", "sample")),
        bind_mesh_dir=bind_mesh_dir,
        bind_vertex_indices=g.bind_idx,
        bind_mesh_indices=bind_mesh_indices,
        bind_cache_size=int(cfg["stage2"].get("bind_cache_size", 8)),
        bind_seq_npz=str(bind_seq_npz_path) if bind_seq_npz_path is not None else None,
        bind_vertices_npz=str(bind_vertices_npz_path) if bind_vertices_npz_path is not None else None,
        bind_tpose_obj=str(bind_tpose_obj_path) if bind_tpose_obj_path is not None else None,
        bind_mesh_to_camera=bool(cfg["stage2"].get("bind_mesh_to_camera", False)),
        bind_inverse_export_upright=bool(cfg["stage2"].get("bind_inverse_export_upright", True)),
        bind_mesh_obj_space=str(cfg["stage2"].get("bind_mesh_obj_space", "auto")),
        bind_align_quat_to_mesh_normals=bool(cfg["stage2"].get("bind_align_quat_to_mesh_normals", True)),
        bind_quat_normal_blend=float(cfg["stage2"].get("bind_quat_normal_blend", 1.0)),
        bind_surface_aniso_weight=float(cfg["stage2"].get("bind_surface_aniso_weight", 0.0)),
        bind_surface_aniso_margin=float(cfg["stage2"].get("bind_surface_aniso_margin", 0.15)),
        bind_mesh_offset=tuple(cfg["stage2"].get("bind_mesh_offset", (0.0, 0.0, 0.0))),
        bind_rigid_align=bind_rigid_align_for_train,
        use_smpl_silhouette_mask=bool(cfg["stage2"].get("use_smpl_silhouette_mask", False)),
        smpl_mask_dilate_px=int(cfg["stage2"].get("smpl_mask_dilate_px", 0)),
        smpl_mask_erode_px=int(cfg["stage2"].get("smpl_mask_erode_px", 0)),
        smpl_color_mask_erode_px=int(cfg["stage2"].get("smpl_color_mask_erode_px", 0)),
        smpl_alpha_leak_weight=float(cfg["stage2"].get("smpl_alpha_leak_weight", 0.0)),
        smpl_mask_override_alpha=bool(cfg["stage2"].get("smpl_mask_override_alpha", True)),
        smpl_mask_precompute=bool(cfg["stage2"].get("smpl_mask_precompute", True)),
        smpl_mask_precompute_max_frames=int(cfg["stage2"].get("smpl_mask_precompute_max_frames", 64)),
        person_masks_npz=cfg["stage2"].get("person_masks_npz", str(stage1_dir / "person_masks.npz")),
        use_person_mask=bool(cfg["stage2"].get("use_person_mask", False)),
        person_mask_dilate_px=int(cfg["stage2"].get("person_mask_dilate_px", 0)),
        person_mask_erode_px=int(cfg["stage2"].get("person_mask_erode_px", 0)),
        person_mask_fuse_mode=str(cfg["stage2"].get("person_mask_fuse_mode", "intersect")),
        separate_fg_bg_training=bool(cfg["stage2"].get("separate_fg_bg_training", False)),
        fg_region_weight=float(cfg["stage2"].get("fg_region_weight", 1.0)),
        bg_region_weight=float(cfg["stage2"].get("bg_region_weight", 1.0)),
        bg_grad_scale=float(cfg["stage2"].get("bg_grad_scale", 1.0)),
        bg_opacity_logit_max=cfg["stage2"].get("bg_opacity_logit_max", None),
        bg_log_scale_min=cfg["stage2"].get("bg_log_scale_min", None),
        bg_log_scale_max=cfg["stage2"].get("bg_log_scale_max", None),
        fg_opacity_logit_max=cfg["stage2"].get("fg_opacity_logit_max", None),
        fg_log_scale_min=cfg["stage2"].get("fg_log_scale_min", None),
        fg_log_scale_max=cfg["stage2"].get("fg_log_scale_max", None),
        early_stop_enable=bool(cfg["stage2"].get("early_stop_enable", False)),
        early_stop_patience=int(cfg["stage2"].get("early_stop_patience", 0)),
        early_stop_min_delta=float(cfg["stage2"].get("early_stop_min_delta", 0.0)),
        early_stop_warmup_iters=int(cfg["stage2"].get("early_stop_warmup_iters", 0)),
        binding=cfg["stage2"].get("binding", None),
    )

    logger.info(f"Saved checkpoint: {ckpt_path}")
    save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")


if __name__ == "__main__":
    main()


