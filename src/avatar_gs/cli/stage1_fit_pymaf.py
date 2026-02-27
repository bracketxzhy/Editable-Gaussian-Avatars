from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import joblib
import numpy as np

from avatar_gs.io.mesh_export import build_obj_face_text, export_obj

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _find_any_file(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        cands = list(root.rglob(pat))
        if len(cands) > 0:
            return cands[0]
    return None


def _ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def _resolve_pymaf_ckpt(assets_dir: Path, override: Path | None) -> Path:
    if override is not None and override.exists():
        return override

    p = _find_any_file(
        assets_dir,
        [
            "PyMAF-X_model_checkpoint_v1.1.pt",
            "PyMAF-X_model_checkpoint_v1.1.pth",
            "PyMAF-X_model_checkpoint_v1.1.pth.tar",
            "PyMAF*.pt",
            "PyMAF*.pth",
            "PyMAF*.pth.tar",
            "*pymaf*.pt",
            "*pymaf*.pth",
            "*pymaf*.pth.tar",
            "*.pt",
            "*.pth",
            "*.pth.tar",
            "*.ckpt",
        ],
    )
    if p is None:
        raise FileNotFoundError("Cannot find PyMAF checkpoint under assets. Please pass --pymaf_ckpt explicitly.")
    return p


def _extract_tar_if_needed(tar_path: Path, dst_dir: Path, marker: str) -> None:
    if (dst_dir / marker).exists():
        return
    if not tar_path.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = str(member.name).replace('\\', '/')
            if name.startswith('data/'):
                name = name[len('data/'):]
            if name == '' or name.endswith('/'):
                continue

            out_path = dst_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with out_path.open('wb') as f:
                f.write(src.read())


def _prepare_pymaf_runtime_data(assets_dir: Path, pymaf_dir: Path, pymaf_ckpt: Path) -> tuple[Path, Path]:
    data_dir = pymaf_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = assets_dir / "pymafx_data" / "data.tar.gz"
    _extract_tar_if_needed(tar_path, data_dir, marker="smpl_mean_params.npz")

    smpl_down_src = _find_any_file(assets_dir, ["smpl_downsampling.npz", "mesh_downsampling.npz"])
    mano_down_src = _find_any_file(assets_dir, ["mano_downsampling.npz"])
    if smpl_down_src is not None:
        _ensure_link_or_copy(smpl_down_src, data_dir / "smpl_downsampling.npz")
    if mano_down_src is not None:
        _ensure_link_or_copy(mano_down_src, data_dir / "mano_downsampling.npz")

    smpl_dir = data_dir / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    maps = [
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.npz", smpl_dir / "SMPLX_NEUTRAL.npz"),
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.npz", smpl_dir / "SMPLX_NEUTRAL_2020.npz"),
        (assets_dir / "smplx" / "SMPLX_MALE.npz", smpl_dir / "SMPLX_MALE.npz"),
        (assets_dir / "smplx" / "SMPLX_FEMALE.npz", smpl_dir / "SMPLX_FEMALE.npz"),
        (assets_dir / "smpl" / "SMPL_N_model.npz", smpl_dir / "SMPL_NEUTRAL.npz"),
        (assets_dir / "smpl" / "SMPL_M_model.npz", smpl_dir / "SMPL_MALE.npz"),
        (assets_dir / "smpl" / "SMPL_F_model.npz", smpl_dir / "SMPL_FEMALE.npz"),
    ]
    for src, dst in maps:
        if src.exists():
            _ensure_link_or_copy(src, dst)

    # Ensure checkpoint sits at PyMAF conventional location too.
    pretrain_dir = data_dir / "pretrained_model"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_copy(pymaf_ckpt, pretrain_dir / pymaf_ckpt.name)

    # PyMAF-X expects data/smpl/model_transfer/smplx_to_smpl.pkl
    transfer_dir = smpl_dir / "model_transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    transfer_dst = transfer_dir / "smplx_to_smpl.pkl"
    if not transfer_dst.exists():
        transfer_src = _find_any_file(assets_dir, ["smplx_to_smpl.pkl", "smplx2smpl.pkl"])
        if transfer_src is not None:
            _ensure_link_or_copy(transfer_src, transfer_dst)

    return data_dir, smpl_dir


def _pick_main_component(mask_u8: np.ndarray, prefer_non_border: bool = True) -> np.ndarray:
    mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if mask.size == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    h, w = mask.shape[:2]
    best_idx = -1
    best_area = -1
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area <= 0:
            continue
        touch_border = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if prefer_non_border and touch_border:
            continue
        if int(area) > best_area:
            best_area = int(area)
            best_idx = i
    if best_idx < 0:
        # Fallback: pick largest component regardless of border touching.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = 1 + int(np.argmax(areas))
    out = (labels == best_idx).astype(np.uint8)
    return out


def _estimate_person_mask_chromakey(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Typical green-screen range, slightly tolerant.
    lower = np.array([35, 35, 25], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    fg = (green == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = _pick_main_component((fg * 255).astype(np.uint8), prefer_non_border=True)
    return (fg * 255).astype(np.uint8)


def _mask_score(mask_u8: np.ndarray, mesh_hint_mask_u8: np.ndarray | None) -> float:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0)
    if m.size == 0:
        return -1e9
    area = float(m.mean())
    if area < 0.01:
        return -1e6

    h, w = m.shape[:2]
    border = np.zeros_like(m, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    border_touch = float(m[border > 0].mean())

    ys, xs = np.where(m)
    if len(xs) == 0:
        center_dist = 1.0
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
        dx = (cx - 0.5 * (w - 1)) / max(1.0, 0.5 * (w - 1))
        dy = (cy - 0.5 * (h - 1)) / max(1.0, 0.5 * (h - 1))
        center_dist = float(np.sqrt(dx * dx + dy * dy))

    # Prefer plausible person area ratio for full-body framing.
    area_target = 0.18
    area_term = -abs(area - area_target)

    iou_hint = 0.0
    if mesh_hint_mask_u8 is not None:
        g = (np.asarray(mesh_hint_mask_u8, dtype=np.uint8) > 0)
        inter = float(np.logical_and(m, g).sum())
        union = float(np.logical_or(m, g).sum())
        iou_hint = inter / max(union, 1.0)

    score = (3.0 * iou_hint) + (1.5 * area_term) - (2.0 * border_touch) - (0.5 * center_dist)
    return float(score)


def _estimate_person_mask_grabcut(image_bgr: np.ndarray, mesh_hint_mask: np.ndarray | None = None) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    # Candidate 1: GrabCut with loose rectangle.
    rect = (int(0.08 * w), int(0.04 * h), int(0.84 * w), int(0.92 * h))
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image_bgr, gc_mask, rect, bg_model, fg_model, 5, mode=cv2.GC_INIT_WITH_RECT)
    cand_gc = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand_gc = cv2.morphologyEx(cand_gc, cv2.MORPH_OPEN, k, iterations=1)
    cand_gc = cv2.morphologyEx(cand_gc, cv2.MORPH_CLOSE, k, iterations=2)
    cand_gc = (_pick_main_component(cand_gc, prefer_non_border=True) * 255).astype(np.uint8)

    # Candidate 2: Green-screen color keying.
    cand_ck = _estimate_person_mask_chromakey(image_bgr)

    cands = [cand_gc, cand_ck]
    if mesh_hint_mask is not None:
        mesh_hint_u8 = ((np.asarray(mesh_hint_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255)
        cands.append(mesh_hint_u8)
    else:
        mesh_hint_u8 = None

    best = None
    best_score = -1e12
    for c in cands:
        s = _mask_score(c, mesh_hint_u8)
        if s > best_score:
            best_score = s
            best = c

    if best is None:
        best = cand_gc
    return (np.asarray(best, dtype=np.uint8) > 0).astype(np.uint8) * 255


def _project_verts_to_uv(verts: np.ndarray, cam_t: np.ndarray, width: int, height: int, focal: float = 5000.0) -> tuple[np.ndarray, np.ndarray]:
    v_cam = verts + cam_t.reshape(1, 3)
    z = np.clip(v_cam[:, 2], 1e-6, None)
    u = float(focal) * (v_cam[:, 0] / z) + 0.5 * float(width)
    v = float(focal) * (v_cam[:, 1] / z) + 0.5 * float(height)
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & (v_cam[:, 2] > 1e-5)
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


def _load_faces_from_npz(npz_path: Path) -> np.ndarray:
    arr = np.load(str(npz_path), allow_pickle=True)
    if "f" in arr.files:
        return np.asarray(arr["f"], dtype=np.int32)
    if "faces" in arr.files:
        return np.asarray(arr["faces"], dtype=np.int32)
    raise KeyError(f"No face topology key found in {npz_path} (expected 'f' or 'faces').")


def _project_points_to_uv(points_cam: np.ndarray, width: int, height: int, focal: float = 5000.0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
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


def _to_numpy_cpu(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _smplx_param_batch_size(param_dict: dict) -> int:
    if not isinstance(param_dict, dict):
        return 1

    # Prefer shape/exp keys because pose keys can be [24, 3, 3] in per-frame form.
    for key in ("shape", "expression"):
        if key not in param_dict:
            continue
        arr = _to_numpy_cpu(param_dict[key])
        if arr.ndim >= 2:
            return int(max(1, arr.shape[0]))
        return 1

    for key in ("body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose"):
        if key not in param_dict:
            continue
        arr = _to_numpy_cpu(param_dict[key])
        if arr.ndim >= 4:
            return int(max(1, arr.shape[0]))
        return 1
    return 1


def _slice_smplx_param_dict(param_dict: dict, row_idx: int, batch_size: int) -> dict:
    out: dict = {}
    ridx = int(max(0, min(int(row_idx), int(max(1, batch_size) - 1))))
    for k, v in param_dict.items():
        try:
            if torch is not None and torch.is_tensor(v):
                if int(v.ndim) >= 1 and int(v.shape[0]) == int(batch_size):
                    out[k] = v[ridx]
                else:
                    out[k] = v
                continue
        except Exception:
            pass

        if isinstance(v, np.ndarray):
            if int(v.ndim) >= 1 and int(v.shape[0]) == int(batch_size):
                out[k] = v[ridx]
            else:
                out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) == int(batch_size):
            out[k] = v[ridx]
        else:
            out[k] = v
    return out


def _get_smplx_params_for_index(pred_results: dict, idx: int) -> dict | None:
    smplx_params_all = pred_results.get("smplx_params", None)
    if not isinstance(smplx_params_all, list) or len(smplx_params_all) <= 0:
        return None

    ii = int(idx)
    if ii < 0:
        return None

    # Fast path: already per-frame list.
    if ii < len(smplx_params_all):
        p_i = smplx_params_all[ii]
        if isinstance(p_i, dict):
            bs = _smplx_param_batch_size(p_i)
            if bs <= 1:
                return p_i

    # Batched list path: flatten by cumulative batch sizes.
    offset = 0
    for p_i in smplx_params_all:
        if not isinstance(p_i, dict):
            continue
        bs = _smplx_param_batch_size(p_i)
        if ii < offset + bs:
            return _slice_smplx_param_dict(p_i, ii - offset, bs)
        offset += bs

    return None


def _resolve_pred_index(
    pred_results: dict,
    selected_meta: dict,
    frame_name: str,
) -> int | None:
    sel_files = [str(x) for x in selected_meta.get("selected_files", [])]
    if len(sel_files) <= 0:
        return None

    if frame_name in sel_files:
        local_idx = int(sel_files.index(frame_name))
    else:
        local_idx = int(len(sel_files) - 1)

    frame_ids = np.asarray(pred_results.get("frame_ids", []), dtype=np.int64).reshape(-1)
    if frame_ids.size <= 0:
        return None
    cand = np.where(frame_ids == local_idx)[0]
    return int(cand[0]) if cand.size > 0 else int(np.argmax(frame_ids))


def _extract_tpose_cam_from_smplx(
    pred_results: dict,
    selected_meta: dict,
    frame_name: str,
    assets_dir: Path,
    cam_t: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if smplx is None or torch is None:
        return None, None

    idx = _resolve_pred_index(pred_results=pred_results, selected_meta=selected_meta, frame_name=frame_name)
    if idx is None:
        return None, None

    params_i = _get_smplx_params_for_index(pred_results=pred_results, idx=idx)

    gender = "neutral"
    if isinstance(params_i, dict):
        gender = str(params_i.get("gender", "neutral")).strip().lower()
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"

    betas_np = None
    if isinstance(params_i, dict):
        # shape may come as a CUDA tensor in PyMAF outputs
        betas_raw = params_i.get("shape", None)
        betas_np = None if betas_raw is None else _to_numpy_cpu(betas_raw)
    if betas_np is None and pred_results.get("betas", None) is not None:
        try:
            betas_all = pred_results["betas"]
            if idx < len(betas_all):
                betas_np = _to_numpy_cpu(betas_all[idx])
        except Exception:
            betas_np = None
    if betas_np is None:
        betas_np = np.zeros((10,), dtype=np.float32)
    betas_np = np.asarray(betas_np, dtype=np.float32).reshape(-1)

    transl_t = torch.from_numpy(np.asarray(cam_t, dtype=np.float32).reshape(1, 3))

    def _run_tpose_for_model(model_type: str, model_root: Path) -> tuple[np.ndarray, np.ndarray]:
        # Accept both root styles:
        # 1) assets/{smpl,smplx}
        # 2) direct folder with model files (e.g. assets/smplx)
        if model_type == "smplx":
            direct_candidates = [
                model_root / "SMPLX_NEUTRAL.npz",
                model_root / "SMPLX_NEUTRAL_2020.npz",
            ]
        else:
            direct_candidates = [
                model_root / "SMPL_NEUTRAL.npz",
                model_root / "SMPL_N_model.npz",
                model_root / "SMPL_M_model.npz",
                model_root / "SMPL_F_model.npz",
            ]
        model_root_for_create = model_root.parent if any(pp.exists() for pp in direct_candidates) else model_root

        model = smplx.create(
            model_path=str(model_root_for_create),
            model_type=model_type,
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to(device="cpu")

        num_betas = int(getattr(model, "num_betas", max(10, int(betas_np.shape[0]))))
        betas_use = betas_np
        if betas_use.shape[0] < num_betas:
            betas_use = np.pad(betas_use, (0, num_betas - betas_use.shape[0]))
        betas_t = torch.from_numpy(betas_use[:num_betas].reshape(1, num_betas)).to(dtype=torch.float32)

        body_j = int(getattr(model, "NUM_BODY_JOINTS", 21 if model_type == "smplx" else 23))
        body_pose_dim = int(max(3, body_j * 3))
        body_pose_base = torch.zeros((1, body_pose_dim), dtype=torch.float32)

        # try multiple index conventions; choose the pose with max X span (most "T-like")
        idx_sets = [
            (15, 16, 13, 14),
            (14, 15, 12, 13),
            (16, 17, 14, 15),
            (13, 14, 11, 12),
        ]

        candidates = [body_pose_base.clone()]
        for ls, rs, lc, rc in idx_sets:
            for shoulder_deg, collar_deg in ((90.0, 20.0), (80.0, 15.0), (70.0, 10.0)):
                bp = body_pose_base.clone()
                for j_idx, deg in ((ls, +shoulder_deg), (rs, -shoulder_deg), (lc, +collar_deg), (rc, -collar_deg)):
                    k = int(j_idx) * 3 + 2
                    if 0 <= k < body_pose_dim:
                        bp[0, k] = float(np.deg2rad(deg))
                candidates.append(bp)

        best_out = None
        best_span = -1.0
        for bp in candidates:
            kwargs = {
                "betas": betas_t,
                "global_orient": torch.zeros((1, 3), dtype=torch.float32),
                "body_pose": bp,
                "transl": transl_t,
                "return_verts": True,
            }
            if model_type == "smplx":
                num_expr = int(getattr(model, "num_expression_coeffs", 10))
                kwargs.update(
                    {
                        "left_hand_pose": torch.zeros((1, 45), dtype=torch.float32),
                        "right_hand_pose": torch.zeros((1, 45), dtype=torch.float32),
                        "jaw_pose": torch.zeros((1, 3), dtype=torch.float32),
                        "leye_pose": torch.zeros((1, 3), dtype=torch.float32),
                        "reye_pose": torch.zeros((1, 3), dtype=torch.float32),
                        "expression": torch.zeros((1, num_expr), dtype=torch.float32),
                    }
                )

            with torch.no_grad():
                out_i = model(**kwargs)
            verts_i = out_i.vertices[0].detach().cpu().numpy().astype(np.float32)
            span_x = float(np.percentile(verts_i[:, 0], 99.0) - np.percentile(verts_i[:, 0], 1.0))
            if span_x > best_span:
                best_span = span_x
                best_out = out_i

        if best_out is None:
            raise RuntimeError(f"No valid T-pose candidate for model_type={model_type}")

        verts_tpose_cam = best_out.vertices[0].detach().cpu().numpy().astype(np.float32)
        joints_tpose_cam = best_out.joints[0].detach().cpu().numpy().astype(np.float32)
        return verts_tpose_cam, joints_tpose_cam

    errors: list[str] = []
    try_order = [
        ("smplx", assets_dir / "smplx"),
        ("smpl", assets_dir / "smpl"),
    ]
    for model_type, model_root in try_order:
        if not model_root.exists():
            errors.append(f"{model_type}:missing_model_root:{model_root}")
            continue
        try:
            return _run_tpose_for_model(model_type=model_type, model_root=model_root)
        except Exception as e:
            errors.append(f"{model_type}:{e}")

    print("Warning: failed to compute true T-pose. attempts=" + " | ".join(errors))
    return None, None


def _frame_index_from_name(frame_name: str, fallback: int) -> int:
    stem = Path(str(frame_name)).stem
    if stem.isdigit():
        return int(stem)
    return int(fallback)


def _build_pymaf_frame_records(
    pred_results: dict,
    selected_meta: dict,
    faces_smplx: np.ndarray | None,
) -> list[dict]:
    selected_files = [str(x) for x in selected_meta.get("selected_files", [])]
    selected_global_idx = [int(x) for x in selected_meta.get("selected_indices_global", list(range(len(selected_files))))]
    if len(selected_files) == 0:
        raise RuntimeError("selected_frames_meta.json has empty selected_files.")
    if len(selected_global_idx) != len(selected_files):
        selected_global_idx = list(range(len(selected_files)))

    frame_ids = np.asarray(pred_results.get("frame_ids", []), dtype=np.int64).reshape(-1)
    if frame_ids.size == 0:
        frame_ids = np.arange(len(selected_files), dtype=np.int64)

    if "smplx_verts" not in pred_results or pred_results["smplx_verts"] is None:
        raise RuntimeError(
            "PyMAF output does not contain smplx_verts. "
            "Stage1 PyMAF pipeline requires SMPL-X results and will not downgrade to SMPL."
        )
    verts_all = np.asarray(pred_results["smplx_verts"], dtype=np.float32)
    if faces_smplx is None:
        raise RuntimeError("smplx_verts available but SMPL-X faces are missing.")
    faces = faces_smplx
    model_type = "smplx"

    cam_t_all = np.asarray(pred_results["orig_cam_t"], dtype=np.float32).reshape(-1, 3)
    joints2d_all = None
    if "joints2d" in pred_results and pred_results["joints2d"] is not None:
        arr = np.asarray(pred_results["joints2d"], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == verts_all.shape[0]:
            joints2d_all = arr
    joints3d_all = None
    if "joints3d" in pred_results and pred_results["joints3d"] is not None:
        arr = np.asarray(pred_results["joints3d"], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == verts_all.shape[0] and arr.shape[2] == 3:
            joints3d_all = arr

    local_to_pred: dict[int, int] = {}
    for pi, li in enumerate(frame_ids.tolist()):
        lii = int(li)
        if lii not in local_to_pred:
            local_to_pred[lii] = int(pi)

    records: list[dict] = []
    last_pred = int(max(0, verts_all.shape[0] - 1))
    for local_idx, frame_name in enumerate(selected_files):
        pred_idx = int(local_to_pred.get(int(local_idx), min(local_idx, last_pred)))
        pred_idx = int(np.clip(pred_idx, 0, last_pred))

        verts_local = np.asarray(verts_all[pred_idx], dtype=np.float32)
        cam_t = np.asarray(cam_t_all[pred_idx], dtype=np.float32).reshape(3)
        verts_cam = np.asarray(verts_local + cam_t.reshape(1, 3), dtype=np.float32)

        joints2d = None
        if joints2d_all is not None:
            joints2d = np.asarray(joints2d_all[pred_idx], dtype=np.float32)
        joints_cam = None
        if joints3d_all is not None:
            joints_cam = np.asarray(joints3d_all[pred_idx], dtype=np.float32) + cam_t.reshape(1, 3)

        global_idx = int(selected_global_idx[local_idx]) if local_idx < len(selected_global_idx) else int(local_idx)
        frame_idx = _frame_index_from_name(frame_name=frame_name, fallback=global_idx)

        records.append(
            {
                "frame_name": str(frame_name),
                "frame_idx": int(frame_idx),
                "global_idx": int(global_idx),
                "pred_idx": int(pred_idx),
                "verts_local": verts_local,
                "verts_cam": verts_cam,
                "cam_t": cam_t,
                "joints2d": joints2d,
                "joints_cam": joints_cam,
                "faces": faces,
                "model_type": model_type,
            }
        )
    return records


def _write_stage1_sequence_assets_full(
    *,
    out_dir: Path,
    frames_dir: Path,
    records: list[dict],
    focal: float = 5000.0,
    canonical_verts_cam: np.ndarray | None = None,
    canonical_joints_cam: np.ndarray | None = None,
) -> dict[str, str]:
    if len(records) == 0:
        raise RuntimeError("No records available for stage1 sequence export.")

    seq_dir = out_dir / "sequence"
    seq_dir.mkdir(parents=True, exist_ok=True)

    keypoints_json = out_dir / "keypoints.json"
    person_masks_npz = out_dir / "person_masks.npz"
    params_npz = out_dir / "smpl_params_sequence.npz"
    verts_npz = out_dir / "smpl_vertices_sequence.npz"
    joints_npz = out_dir / "smpl_joints_sequence.npz"
    skeleton_npz = out_dir / "smpl_skeleton_sequence.npz"
    cameras_json = out_dir / "cameras.json"

    # infer camera intrinsics from first frame shape
    first_name = str(records[0]["frame_name"])
    first_img = cv2.imread(str(frames_dir / first_name), cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read frame image: {frames_dir / first_name}")
    h0, w0 = int(first_img.shape[0]), int(first_img.shape[1])
    K_list = [float(focal), 0.0, float(w0) * 0.5, 0.0, float(focal), float(h0) * 0.5, 0.0, 0.0, 1.0]
    with cameras_json.open("w", encoding="utf-8") as f:
        json.dump({"K": K_list}, f, indent=2)

    frame_names: list[str] = []
    verts_cam_seq: list[np.ndarray] = []
    joints_cam_seq: list[np.ndarray] = []
    masks_seq: list[np.ndarray] = []
    kpt_map: dict[str, list[list[float]]] = {}
    shared_faces: np.ndarray | None = None
    shared_face_text: str | None = None
    first_faces = np.asarray(records[0]["faces"], dtype=np.int32)
    if first_faces.ndim == 2 and first_faces.shape[1] == 3:
        same_topology = True
        for rec_chk in records[1:]:
            chk_faces = np.asarray(rec_chk["faces"], dtype=np.int32)
            if chk_faces.shape != first_faces.shape or (not np.array_equal(chk_faces, first_faces)):
                same_topology = False
                break
        if same_topology:
            shared_faces = first_faces
            shared_face_text = build_obj_face_text(first_faces)

    for rec in records:
        frame_name = str(rec["frame_name"])
        frame_idx = int(rec["frame_idx"])
        verts_cam = np.asarray(rec["verts_cam"], dtype=np.float32)
        faces = shared_faces if shared_faces is not None else np.asarray(rec["faces"], dtype=np.int32)
        joints2d = rec.get("joints2d", None)
        joints_cam = rec.get("joints_cam", None)

        obj_name = f"frame_{frame_idx:06d}.obj"
        export_obj(seq_dir / obj_name, verts_cam, faces, face_text=shared_face_text)

        img = cv2.imread(str(frames_dir / frame_name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read frame image: {frames_dir / frame_name}")
        hh, ww = int(img.shape[0]), int(img.shape[1])

        uv_hint, valid_hint = _project_verts_to_uv(
            verts=np.asarray(rec["verts_local"], dtype=np.float32),
            cam_t=np.asarray(rec["cam_t"], dtype=np.float32),
            width=ww,
            height=hh,
            focal=float(focal),
        )
        mask_hint = _rasterize_mesh_mask(
            uv=uv_hint,
            valid_v=valid_hint,
            faces=faces,
            image_hw=(hh, ww),
        )
        mask = _estimate_person_mask_grabcut(img, mesh_hint_mask=mask_hint)

        if joints2d is not None and np.asarray(joints2d).ndim == 2 and np.asarray(joints2d).shape[1] >= 2:
            j2 = np.asarray(joints2d, dtype=np.float32)
            if j2.shape[1] >= 3:
                kpts = [[float(p[0]), float(p[1]), float(np.clip(p[2], 0.0, 1.0))] for p in j2]
            else:
                kpts = [[float(p[0]), float(p[1]), 1.0] for p in j2]
        elif joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
            uv, valid = _project_points_to_uv(np.asarray(joints_cam, dtype=np.float32), width=ww, height=hh, focal=float(focal))
            kpts = [[float(uv[i, 0]), float(uv[i, 1]), float(1.0 if bool(valid[i]) else 0.0)] for i in range(uv.shape[0])]
        else:
            kpts = []

        frame_names.append(frame_name)
        verts_cam_seq.append(verts_cam)
        if joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
            joints_cam_seq.append(np.asarray(joints_cam, dtype=np.float32))
        else:
            joints_cam_seq.append(np.zeros((0, 3), dtype=np.float32))
        masks_seq.append((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8))
        kpt_map[frame_name] = kpts

    with keypoints_json.open("w", encoding="utf-8") as f:
        json.dump(kpt_map, f, indent=2)

    np.savez(
        person_masks_npz,
        frame_names=np.asarray(frame_names, dtype=object),
        masks=np.asarray(masks_seq, dtype=np.uint8),
    )

    verts_cam_arr = np.asarray(verts_cam_seq, dtype=np.float32)
    np.savez(
        verts_npz,
        frame_names=np.asarray(frame_names, dtype=object),
        verts_cam=verts_cam_arr,
    )

    joints_valid = [j for j in joints_cam_seq if j.shape[0] > 0]
    if len(joints_valid) > 0:
        J = max(int(j.shape[0]) for j in joints_valid)
        joints_arr = np.zeros((len(joints_cam_seq), J, 3), dtype=np.float32)
        for i, j in enumerate(joints_cam_seq):
            if j.shape[0] > 0:
                n = min(J, int(j.shape[0]))
                joints_arr[i, :n, :] = j[:n, :]
        np.savez(joints_npz, frame_names=np.asarray(frame_names, dtype=object), joints_cam=joints_arr)
        np.savez(
            skeleton_npz,
            frame_names=np.asarray(frame_names, dtype=object),
            joints_cam=joints_arr,
            joints_cam_theta=joints_arr,
        )

    canonical_verts_ref = None
    if canonical_verts_cam is not None:
        vv = np.asarray(canonical_verts_cam, dtype=np.float32)
        if vv.ndim == 2 and vv.shape[1] == 3:
            canonical_verts_ref = vv
    if canonical_verts_ref is None and verts_cam_arr.ndim == 3 and verts_cam_arr.shape[0] > 0:
        canonical_verts_ref = np.asarray(verts_cam_arr[0], dtype=np.float32)

    canonical_joints_ref = None
    if canonical_joints_cam is not None:
        jj = np.asarray(canonical_joints_cam, dtype=np.float32)
        if jj.ndim == 2 and jj.shape[1] == 3:
            canonical_joints_ref = jj
    if canonical_joints_ref is None:
        for j in joints_cam_seq:
            jn = np.asarray(j, dtype=np.float32)
            if jn.ndim == 2 and jn.shape[1] == 3 and jn.shape[0] > 0:
                canonical_joints_ref = jn
                break

    params_kwargs = {
        "frame_names": np.asarray(frame_names, dtype=object),
        "fit_strategy": np.asarray(["stage1_pymaf"], dtype=object),
        "export_upright_rot": np.eye(3, dtype=np.float32),
        "export_center_xy": np.asarray([0], dtype=np.int32),
        "export_center_xy_value": np.asarray([0.0, 0.0], dtype=np.float32),
        "export_on_floor": np.asarray([0], dtype=np.int32),
        "export_floor_z": np.asarray([0.0], dtype=np.float32),
    }
    if canonical_verts_ref is not None:
        vv = np.asarray(canonical_verts_ref, dtype=np.float32)
        if vv.ndim == 2 and vv.shape[1] == 3:
            params_kwargs["verts_canonical_cam"] = vv
            params_kwargs["verts_tpose_cam"] = vv
    if canonical_joints_ref is not None:
        jj = np.asarray(canonical_joints_ref, dtype=np.float32)
        if jj.ndim == 2 and jj.shape[1] == 3:
            params_kwargs["joints_canonical_cam"] = jj
            params_kwargs["joints_tpose_cam"] = jj
    np.savez(params_npz, **params_kwargs)

    return {
        "cameras_json": str(cameras_json),
        "keypoints_json": str(keypoints_json),
        "person_masks_npz": str(person_masks_npz),
        "smpl_params_sequence_npz": str(params_npz),
        "smpl_vertices_sequence_npz": str(verts_npz),
        "sequence_dir": str(seq_dir),
    }

def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _check_stage1_requirements(assets_dir: Path, pymaf_dir: Path) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"pymaf": [], "python": []}

    if not pymaf_dir.exists():
        missing["pymaf"].append("third_party/PyMAF-X repo")

    if _find_any_file(assets_dir, ["PyMAF-X_model_checkpoint_v1.1.pt", "PyMAF*.pt", "*pymaf*.pt"]) is None:
        missing["pymaf"].append("PyMAF-X_model_checkpoint_v1.1.pt")

    checks = [
        "openpifpaf",
        "smplx",
        "joblib",
    ]
    for mod in checks:
        try:
            __import__(mod)
        except Exception:
            missing["python"].append(mod)

    return missing


def _remove_path_tree(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except Exception:
        return False




def main() -> None:
    parser = argparse.ArgumentParser("Stage1: PyMAF-X refine")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input", help="Video frame directory")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Assets root directory")
    parser.add_argument("--out_dir", type=str, default="outputs/stage1", help="All outputs are written here")
    parser.add_argument("--pymaf_frame_indices", type=str, default="-1", help="Frame indices passed to PyMAF refine")
    parser.add_argument("--pymaf_stride", type=int, default=1, help="Sample every N frames for PyMAF refine when pymaf_frame_indices is all/-1")
    parser.add_argument("--pymaf_ckpt", type=str, default="", help="Optional explicit PyMAF checkpoint path")
    parser.add_argument("--single_person", action="store_true", help="Set PyMAF single-person mode")
    parser.add_argument("--skip_pymaf", action="store_true", help="Skip PyMAF stage")
    parser.add_argument("--pymaf_repo_dir", type=str, default="third_party/PyMAF-X", help="PyMAF-X repo dir")
    parser.add_argument("--minimal_stage1_outputs", action="store_true", help="Keep only sequence assets required by stage2; delete intermediates")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    pymaf_dir = (project_root / args.pymaf_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets dir not found: {assets_dir}")
    if not pymaf_dir.exists():
        raise FileNotFoundError(f"PyMAF repo not found: {pymaf_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pymaf_out = out_dir / "pymaf"
    final_out = out_dir / "final"
    pymaf_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(frames_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    print(f"Frames: {len(frames)}")

    pymaf_indices_raw = str(args.pymaf_frame_indices).strip().lower()
    pymaf_stride = max(1, int(args.pymaf_stride))
    if pymaf_indices_raw in ("", "-1", "all", "*"):
        if pymaf_stride == 1:
            pymaf_indices_arg = "-1"
            pymaf_selected_count = len(frames)
        else:
            pymaf_idx = list(range(0, len(frames), pymaf_stride))
            pymaf_indices_arg = " ".join(str(i) for i in pymaf_idx)
            pymaf_selected_count = len(pymaf_idx)
    else:
        pymaf_indices_arg = str(args.pymaf_frame_indices).strip()
        pymaf_selected_count = -1

    if pymaf_selected_count > 0:
        print(
            f"PyMAF frame selection: {pymaf_selected_count}/{len(frames)} frames "
            f"(stride={pymaf_stride}, indices='{pymaf_indices_arg[:64]}{'...' if len(pymaf_indices_arg) > 64 else ''}')"
        )

    req = _check_stage1_requirements(assets_dir=assets_dir, pymaf_dir=pymaf_dir)
    pymaf_ckpt: Path | None = None
    if not bool(args.skip_pymaf):
        pymaf_ckpt = _resolve_pymaf_ckpt(
            assets_dir=assets_dir,
            override=(Path(args.pymaf_ckpt).resolve() if str(args.pymaf_ckpt).strip() else None),
        )
        _prepare_pymaf_runtime_data(assets_dir=assets_dir, pymaf_dir=pymaf_dir, pymaf_ckpt=pymaf_ckpt)

    env_base = os.environ.copy()

    if not bool(args.skip_pymaf):
        pymaf_cmd = [
            sys.executable,
            "-m",
            "avatar_gs.cli._run_pymaf_refine_frames",
            "--frames_dir",
            str(frames_dir),
            "--output_folder",
            str(pymaf_out),
            "--pymaf_repo_dir",
            str(pymaf_dir),
            "--pretrained_model",
            str(pymaf_ckpt),
            "--frame_indices",
            str(pymaf_indices_arg),
        ]
        if bool(args.single_person):
            pymaf_cmd.append("--single_person")
        _run_cmd(pymaf_cmd, cwd=project_root, env=env_base)
    else:
        print("Skipped PyMAF stage.")
        return

    pymaf_result_dir = pymaf_out / "selected_frames"
    pred_pkl = pymaf_result_dir / "output.pkl"
    meta_json = pymaf_result_dir / "selected_frames_meta.json"
    if not pred_pkl.exists():
        raise FileNotFoundError(f"PyMAF output missing: {pred_pkl}")
    if not meta_json.exists():
        raise FileNotFoundError(f"PyMAF selected-frame metadata missing: {meta_json}")

    pred_results = joblib.load(pred_pkl)
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    # _run_pymaf_refine_frames may bootstrap openpifpaf at runtime.
    if bool(selected_meta.get("openpifpaf_available", False)):
        req["python"] = [m for m in req.get("python", []) if m != "openpifpaf"]
    if "python" in req and "openpifpaf" in req["python"]:
        try:
            __import__("openpifpaf")
            req["python"] = [m for m in req["python"] if m != "openpifpaf"]
        except Exception:
            pass

    faces_smplx = None
    smplx_face_path = _find_any_file(assets_dir, ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"])
    if smplx_face_path is not None:
        faces_smplx = _load_faces_from_npz(smplx_face_path)

    records = _build_pymaf_frame_records(
        pred_results=pred_results,
        selected_meta=selected_meta,
        faces_smplx=faces_smplx,
    )
    if len(records) <= 0:
        raise RuntimeError("No frame records built from PyMAF output.")

    first_rec = records[0]
    last_rec = records[-1]

    canonical_verts_cam = None
    canonical_joints_cam = None
    try:
        canonical_verts_cam, canonical_joints_cam = _extract_tpose_cam_from_smplx(
            pred_results=pred_results,
            selected_meta=selected_meta,
            frame_name=str(last_rec["frame_name"]),
            assets_dir=assets_dir,
            cam_t=np.asarray(last_rec["cam_t"], dtype=np.float32),
        )
    except Exception as e:
        print(f"Warning: canonical pose extraction raised an exception; fallback to frame-space canonical. err={e}")

    if canonical_verts_cam is None:
        canonical_verts_cam = np.asarray(last_rec["verts_cam"], dtype=np.float32)
    if canonical_joints_cam is None:
        jc_last = last_rec.get("joints_cam", None)
        if jc_last is not None:
            canonical_joints_cam = np.asarray(jc_last, dtype=np.float32)

    compat_assets = _write_stage1_sequence_assets_full(
        out_dir=out_dir,
        frames_dir=frames_dir,
        records=records,
        focal=5000.0,
        canonical_verts_cam=canonical_verts_cam,
        canonical_joints_cam=canonical_joints_cam,
    )
    stage1_seq_obj = out_dir / "sequence" / f"frame_{int(last_rec['frame_idx']):06d}.obj"
    model_type_set = sorted({str(rec.get("model_type", "unknown")) for rec in records})
    summary = {
        "frames_dir": str(frames_dir),
        "assets_dir": str(assets_dir),
        "pymaf_repo_dir": str(pymaf_dir),
        "pymaf_output_dir": str(pymaf_result_dir),
        "final_output_dir": str(final_out),
        "sequence_frame_count": int(len(records)),
        "sequence_first_frame_name": str(first_rec["frame_name"]),
        "sequence_last_frame_name": str(last_rec["frame_name"]),
        "sequence_first_frame_idx": int(first_rec["frame_idx"]),
        "sequence_last_frame_idx": int(last_rec["frame_idx"]),
        "model_type": ("smplx" if model_type_set == ["smplx"] else ",".join(model_type_set)),
        "pymaf_checkpoint": (None if pymaf_ckpt is None else str(pymaf_ckpt)),
        "missing_requirements": req,
        "stage2_compat_assets": compat_assets,
        "stage1_sequence_obj": str(stage1_seq_obj),
    }
    with (final_out / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if bool(args.minimal_stage1_outputs):
        # Keep only canonical objs in stage1 root; delete bulky/intermediate outputs.
        for pp in (pymaf_out, final_out):
            _remove_path_tree(pp)


    print("Done.")
    print(f"Sequence frame count: {len(records)}")
    print(f"Sequence first frame: {first_rec['frame_name']} (idx={int(first_rec['frame_idx'])})")
    print(f"Sequence last frame: {last_rec['frame_name']} (idx={int(last_rec['frame_idx'])})")
    print(f"Sequence sample OBJ: {stage1_seq_obj}")
    print(f"Summary: {final_out / 'run_summary.json'}")


if __name__ == "__main__":
    main()
















