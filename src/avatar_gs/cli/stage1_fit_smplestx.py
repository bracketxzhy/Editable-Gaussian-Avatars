from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from avatar_gs.alignment import (
    apply_similarity_to_camera_points,
    estimate_mask_similarity_transform,
    mask_iou,
    mask_outside_ratio,
)
from avatar_gs.alignment.mask_sources import estimate_person_mask_modnet_eval_style_np
from avatar_gs.io.mesh_export import build_obj_face_text, export_obj
from avatar_gs.segmentation import estimate_person_mask_modnet, resolve_modnet_ckpt_path
from avatar_gs.segmentation.smpl_face_semantics import build_stage1_fashn_face_semantics
from avatar_gs.smpl import SilhouetteRefineConfig, SharedBetaRefitConfig, apply_shared_beta_refit, apply_silhouette_refine
from avatar_gs.utils.progress import tqdm

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
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink():
                if dst.resolve(strict=True) == src.resolve(strict=True):
                    return
            elif dst.is_file() and dst.stat().st_size > 0:
                return
        except Exception:
            pass
        try:
            dst.unlink()
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def _pick_best_existing_smplestx_ckpt(cands: list[Path]) -> Path | None:
    valid = [p for p in cands if p.exists() and p.is_file()]
    if len(valid) <= 0:
        return None
    valid.sort(key=lambda p: int(p.stat().st_size), reverse=True)
    return valid[0]


def _resolve_smplestx_model_name(path_like: str | Path) -> str:
    p = Path(str(path_like).strip())
    if p.is_dir():
        return p.name
    if p.suffixes[-2:] == [".pth", ".tar"]:
        return p.name[: -len(".pth.tar")]
    if p.suffix in (".pt", ".pth", ".ckpt"):
        return p.stem
    return p.name


def _resolve_smplestx_ckpt_file(path_like: str | Path) -> Path:
    p = Path(str(path_like).strip())
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"SMPLest-X checkpoint path not found: {p}")
    model_name = _resolve_smplestx_model_name(p)
    direct = p / f"{model_name}.pth.tar"
    if direct.exists():
        return direct
    for pat in ("*.pth.tar", "*.pth", "*.pt", "*.ckpt"):
        cands = sorted(p.glob(pat))
        if len(cands) > 0:
            return cands[0]
    raise FileNotFoundError(f"No SMPLest-X checkpoint file found under: {p}")


def _find_smplestx_config_candidate(smplestx_dir: Path, ckpt_file: Path, model_name: str) -> Path | None:
    candidates = [
        ckpt_file.parent / "config_base.py",
        ckpt_file.parent / "config.py",
        smplestx_dir / "configs" / f"config_{model_name}.py",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    alt_cfg = sorted((smplestx_dir / "configs").glob("config_*.py"))
    if len(alt_cfg) == 1:
        return alt_cfg[0]
    return None


def _resolve_smplestx_ckpt(assets_dir: Path, smplestx_dir: Path, override: Path | None) -> str:
    if override is not None and override.exists():
        return str(_resolve_smplestx_ckpt_file(override))

    repo_pretrained = smplestx_dir / "pretrained_models"
    preferred_repo = [
        repo_pretrained / "smplest_x_h" / "smplest_x_h.pth.tar",
        repo_pretrained / "smplest_x_huge" / "smplest_x_huge.pth.tar",
    ]
    best_repo = _pick_best_existing_smplestx_ckpt(preferred_repo)
    if best_repo is None:
        best_repo = _pick_best_existing_smplestx_ckpt(
            list(repo_pretrained.rglob("smplest_x*.pth.tar")) if repo_pretrained.exists() else []
        )
    if best_repo is not None:
        return str(best_repo)

    p = _find_any_file(
        assets_dir,
        [
            "smplest_x*.pth.tar",
            "smplest_x*.pth",
            "smplest_x*.pt",
            "SMPLest-X*.pth.tar",
            "SMPLestX*.pth.tar",
            "*smplest*x*.pth.tar",
        ],
    )
    if p is None:
        raise FileNotFoundError(
            "Cannot find SMPLest-X checkpoint under third_party/SMPLest-X/pretrained_models or assets. "
            "Please pass --smplestx_ckpt explicitly."
        )
    return str(_resolve_smplestx_ckpt_file(p))


def _prepare_smplestx_runtime_data(assets_dir: Path, smplestx_dir: Path, smplestx_ckpt: str) -> None:
    pretrained_dir = smplestx_dir / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    human_model_dir = smplestx_dir / "human_models" / "human_model_files"
    (human_model_dir / "smplx").mkdir(parents=True, exist_ok=True)
    (human_model_dir / "smpl").mkdir(parents=True, exist_ok=True)

    ckpt_file = _resolve_smplestx_ckpt_file(smplestx_ckpt)
    model_name = _resolve_smplestx_model_name(ckpt_file)
    model_dir = pretrained_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_copy(ckpt_file, model_dir / f"{model_name}.pth.tar")

    config_candidate = _find_smplestx_config_candidate(
        smplestx_dir=smplestx_dir,
        ckpt_file=ckpt_file,
        model_name=model_name,
    )
    if config_candidate is None:
        raise FileNotFoundError(
            f"Cannot find SMPLest-X config for model '{model_name}'. "
            f"Expected config_base.py next to {ckpt_file} or configs/config_{model_name}.py under {smplestx_dir}."
        )
    _ensure_link_or_copy(config_candidate, model_dir / "config_base.py")

    model_maps = [
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.npz", human_model_dir / "smplx" / "SMPLX_NEUTRAL.npz"),
        (assets_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz", human_model_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz"),
        (assets_dir / "smplx" / "SMPLX_MALE.npz", human_model_dir / "smplx" / "SMPLX_MALE.npz"),
        (assets_dir / "smplx" / "SMPLX_FEMALE.npz", human_model_dir / "smplx" / "SMPLX_FEMALE.npz"),
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.pkl", human_model_dir / "smplx" / "SMPLX_NEUTRAL.pkl"),
        (assets_dir / "smpl" / "SMPL_NEUTRAL.pkl", human_model_dir / "smpl" / "SMPL_NEUTRAL.pkl"),
        (assets_dir / "smpl" / "SMPL_MALE.pkl", human_model_dir / "smpl" / "SMPL_MALE.pkl"),
        (assets_dir / "smpl" / "SMPL_FEMALE.pkl", human_model_dir / "smpl" / "SMPL_FEMALE.pkl"),
        (assets_dir / "smpl" / "SMPL_N_model.npz", human_model_dir / "smpl" / "SMPL_NEUTRAL.npz"),
        (assets_dir / "smpl" / "SMPL_M_model.npz", human_model_dir / "smpl" / "SMPL_MALE.npz"),
        (assets_dir / "smpl" / "SMPL_F_model.npz", human_model_dir / "smpl" / "SMPL_FEMALE.npz"),
    ]
    for src, dst in model_maps:
        if src.exists():
            _ensure_link_or_copy(src, dst)

    for pat, dst_name in (
        ("MANO_SMPLX_vertex_ids.pkl", "MANO_SMPLX_vertex_ids.pkl"),
        ("SMPL-X__FLAME_vertex_ids.npy", "SMPL-X__FLAME_vertex_ids.npy"),
        ("SMPLX_to_J14.pkl", "SMPLX_to_J14.pkl"),
    ):
        src = _find_any_file(assets_dir, [pat])
        if src is not None:
            _ensure_link_or_copy(src, human_model_dir / "smplx" / dst_name)

    for pat, dst_name in (
        ("vitpose_huge.pth", "vitpose_huge.pth"),
        ("vitpose_large.pth", "vitpose_large.pth"),
        ("vitpose_base.pth", "vitpose_base.pth"),
        ("vitpose_small.pth", "vitpose_small.pth"),
    ):
        src = _find_any_file(assets_dir, [pat])
        if src is not None:
            _ensure_link_or_copy(src, pretrained_dir / dst_name)


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


def _boundary_mask_u8(mask_u8: np.ndarray) -> np.ndarray:
    m = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if m.ndim != 2 or m.size <= 0:
        return np.zeros_like(m, dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(out, contours, -1, color=255, thickness=1, lineType=cv2.LINE_8)
    return out


def _boundary_mean_distance_px(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    boundary_pred = (_boundary_mask_u8(mask_pred) > 0)
    boundary_gt = (_boundary_mask_u8(mask_gt) > 0)
    if int(boundary_pred.sum()) <= 0:
        return float("inf")
    inv_tgt = np.logical_not(boundary_gt).astype(np.uint8)
    dist = cv2.distanceTransform(inv_tgt, cv2.DIST_L2, 3)
    vals = np.asarray(dist[boundary_pred], dtype=np.float32).reshape(-1)
    if vals.size <= 0:
        return float("inf")
    return float(vals.mean())


def _boundary_p95_distance_px(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    boundary_pred = (_boundary_mask_u8(mask_pred) > 0)
    boundary_gt = (_boundary_mask_u8(mask_gt) > 0)
    if int(boundary_pred.sum()) <= 0:
        return float("inf")
    inv_tgt = np.logical_not(boundary_gt).astype(np.uint8)
    dist = cv2.distanceTransform(inv_tgt, cv2.DIST_L2, 3)
    vals = np.asarray(dist[boundary_pred], dtype=np.float32).reshape(-1)
    if vals.size <= 0:
        return float("inf")
    return float(np.percentile(vals, 95))


def _export_candidate_score(
    *,
    iou: float,
    outside: float,
    area_ratio: float,
    boundary_mean_px: float,
    boundary_p95_px: float,
) -> float:
    oversize = max(0.0, float(area_ratio) - 1.0)
    undersize = max(0.0, 1.0 - float(area_ratio))
    boundary_penalty = 0.015 * float(boundary_mean_px) if np.isfinite(float(boundary_mean_px)) else 1.0
    boundary_p95_penalty = 0.004 * float(boundary_p95_px) if np.isfinite(float(boundary_p95_px)) else 1.0
    outside_weight = 3.5 if float(outside) <= 0.05 else 6.0
    return float(iou - outside_weight * outside - 0.20 * oversize - 0.05 * undersize - boundary_penalty - boundary_p95_penalty)


def _estimate_person_mask_modnet_stage1(
    image_bgr: np.ndarray,
    mesh_hint_mask: np.ndarray | None = None,
    *,
    modnet_ckpt_path: str | None = None,
    ref_size: int = 512,
    threshold: float = 0.5,
) -> np.ndarray:
    mesh_hint_u8 = None
    if mesh_hint_mask is not None:
        mesh_hint_u8 = ((np.asarray(mesh_hint_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    try:
        out = estimate_person_mask_modnet_eval_style_np(
            image_bgr,
            hint_mask_u8=None,
            modnet_ref_size=int(ref_size),
            modnet_threshold=float(threshold),
            modnet_ckpt_path=(str(modnet_ckpt_path).strip() or None),
        )
        if int((np.asarray(out, dtype=np.uint8) > 0).sum()) > 0:
            return (np.asarray(out, dtype=np.uint8) > 0).astype(np.uint8) * 255
    except Exception:
        pass
    if mesh_hint_u8 is not None and int((mesh_hint_u8 > 0).sum()) > 0:
        return mesh_hint_u8
    return np.zeros(image_bgr.shape[:2], dtype=np.uint8)


def _camera_intrinsics_from_record(
    record: dict | None,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if isinstance(record, dict):
        focal_xy = record.get("camera_focal", None)
        princpt_xy = record.get("camera_princpt", None)
        if focal_xy is not None and princpt_xy is not None:
            focal_arr = np.asarray(focal_xy, dtype=np.float32).reshape(-1)
            princpt_arr = np.asarray(princpt_xy, dtype=np.float32).reshape(-1)
            if focal_arr.size >= 2 and princpt_arr.size >= 2:
                return float(focal_arr[0]), float(focal_arr[1]), float(princpt_arr[0]), float(princpt_arr[1])
    f = 5000.0
    return float(f), float(f), 0.5 * float(width), 0.5 * float(height)


def _project_verts_to_uv(
    verts: np.ndarray,
    cam_t: np.ndarray,
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    v_cam = verts + cam_t.reshape(1, 3)
    z = np.clip(v_cam[:, 2], 1e-6, None)
    u = float(fx) * (v_cam[:, 0] / z) + float(cx)
    v = float(fy) * (v_cam[:, 1] / z) + float(cy)
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


def _project_points_to_uv(
    points_cam: np.ndarray,
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = pts[:, 2]
    valid = z > 1e-6
    u = np.zeros((pts.shape[0],), dtype=np.float32)
    v = np.zeros((pts.shape[0],), dtype=np.float32)
    u[valid] = float(fx) * (pts[valid, 0] / z[valid]) + float(cx)
    v[valid] = float(fy) * (pts[valid, 1] / z[valid]) + float(cy)
    return np.stack([u, v], axis=1), valid


def _flatten_pose_param(arr: np.ndarray, fallback_dim: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.size <= 0:
        return np.zeros((1, fallback_dim), dtype=np.float32)
    return x.reshape(1, -1).astype(np.float32)


def _smplx_model_root(assets_dir: Path, model_type: str) -> Path:
    cand = assets_dir / model_type
    # smplx.create(model_path=..., model_type=...) appends model_type again when
    # model_path is a directory. Pass the assets root if it already contains the
    # expected subfolder.
    if cand.exists():
        return assets_dir
    return assets_dir


def _load_smplestx_npz_params(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(str(npz_path), allow_pickle=True)
    return {str(k): np.asarray(z[k]) for k in z.files}


def _rebuild_smplx_from_params(
    *,
    params: dict[str, np.ndarray],
    assets_dir: Path,
    gender: str = "neutral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if smplx is None or torch is None:
        raise RuntimeError("smplx or torch is not installed.")

    gender_use = str(gender).strip().lower()
    if gender_use not in {"neutral", "male", "female"}:
        gender_use = "neutral"

    model_root = _smplx_model_root(assets_dir=assets_dir, model_type="smplx")
    model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender=gender_use,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to(device="cpu")

    num_betas = int(getattr(model, "num_betas", 10))
    num_expr = int(getattr(model, "num_expression_coeffs", 10))
    body_dim = int(getattr(model, "NUM_BODY_JOINTS", 21) * 3)

    betas_np = np.asarray(params.get("betas", np.zeros((1, num_betas), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if betas_np.shape[0] < num_betas:
        betas_np = np.pad(betas_np, (0, num_betas - betas_np.shape[0]))
    expr_np = np.asarray(params.get("expression", np.zeros((1, num_expr), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if expr_np.shape[0] < num_expr:
        expr_np = np.pad(expr_np, (0, num_expr - expr_np.shape[0]))

    kwargs = {
        "betas": torch.from_numpy(betas_np[:num_betas].reshape(1, num_betas)).to(dtype=torch.float32),
        "global_orient": torch.from_numpy(_flatten_pose_param(params.get("global_orient", np.zeros((1, 3), dtype=np.float32)), 3)).to(dtype=torch.float32),
        "body_pose": torch.from_numpy(_flatten_pose_param(params.get("body_pose", np.zeros((1, body_dim), dtype=np.float32)), body_dim)).to(dtype=torch.float32),
        "left_hand_pose": torch.from_numpy(_flatten_pose_param(params.get("left_hand_pose", np.zeros((1, 45), dtype=np.float32)), 45)).to(dtype=torch.float32),
        "right_hand_pose": torch.from_numpy(_flatten_pose_param(params.get("right_hand_pose", np.zeros((1, 45), dtype=np.float32)), 45)).to(dtype=torch.float32),
        "jaw_pose": torch.from_numpy(_flatten_pose_param(params.get("jaw_pose", np.zeros((1, 3), dtype=np.float32)), 3)).to(dtype=torch.float32),
        "leye_pose": torch.from_numpy(_flatten_pose_param(params.get("leye_pose", np.zeros((1, 3), dtype=np.float32)), 3)).to(dtype=torch.float32),
        "reye_pose": torch.from_numpy(_flatten_pose_param(params.get("reye_pose", np.zeros((1, 3), dtype=np.float32)), 3)).to(dtype=torch.float32),
        "expression": torch.from_numpy(expr_np[:num_expr].reshape(1, num_expr)).to(dtype=torch.float32),
        "transl": torch.from_numpy(np.asarray(params.get("transl", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 3)).to(dtype=torch.float32),
        "return_verts": True,
    }
    with torch.no_grad():
        out = model(**kwargs)
    return (
        out.vertices[0].detach().cpu().numpy().astype(np.float32),
        out.joints[0].detach().cpu().numpy().astype(np.float32),
        np.asarray(kwargs["transl"][0].detach().cpu().numpy(), dtype=np.float32),
    )


def _project_points_with_meta(points_cam: np.ndarray, meta: dict | None) -> np.ndarray | None:
    if meta is None:
        return None
    focal = meta.get("focal", None)
    princpt = meta.get("princpt", None)
    if focal is None or princpt is None:
        return None
    focal_np = np.asarray(focal, dtype=np.float32).reshape(-1)
    princpt_np = np.asarray(princpt, dtype=np.float32).reshape(-1)
    if focal_np.size < 2 or princpt_np.size < 2:
        return None
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    u = focal_np[0] * (pts[:, 0] / z) + princpt_np[0]
    v = focal_np[1] * (pts[:, 1] / z) + princpt_np[1]
    return np.stack([u, v, np.ones_like(u, dtype=np.float32)], axis=1).astype(np.float32)


def _extract_tpose_cam_from_record(
    record: dict,
    assets_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if smplx is None or torch is None:
        return None, None

    params = dict(record.get("smplestx_params", {}))
    cam_t = np.asarray(record.get("cam_t", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
    gender = str(record.get("gender", "neutral")).strip().lower()
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"
    betas_np = np.asarray(params.get("betas", np.zeros((10,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    transl_np = np.asarray(
        params.get("transl", params.get("translation", params.get("cam_t", cam_t))),
        dtype=np.float32,
    ).reshape(3)
    transl_t = torch.from_numpy(transl_np.reshape(1, 3))
    global_orient_t = torch.from_numpy(
        _flatten_pose_param(
            params.get("global_orient", params.get("root_orient", np.zeros((1, 3), dtype=np.float32))),
            3,
        )
    ).to(dtype=torch.float32)

    def _run_tpose_for_model(model_type: str, assets_root: Path) -> tuple[np.ndarray, np.ndarray]:
        model = smplx.create(
            model_path=str(assets_root),
            model_type=model_type,
            gender=gender,
            use_pca=False,
            flat_hand_mean=False,
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
        candidates = [body_pose_base.clone()]
        for ls, rs, lc, rc in ((15, 16, 13, 14), (14, 15, 12, 13), (16, 17, 14, 15), (13, 14, 11, 12)):
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
                # Keep the frame root orientation so "T-pose" stays upright in camera space.
                "global_orient": global_orient_t,
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
        return (
            best_out.vertices[0].detach().cpu().numpy().astype(np.float32),
            best_out.joints[0].detach().cpu().numpy().astype(np.float32),
        )

    errors: list[str] = []
    for model_type in ("smplx", "smpl"):
        model_root = assets_dir / model_type
        if not model_root.exists():
            errors.append(f"{model_type}:missing_model_root:{model_root}")
            continue
        try:
            return _run_tpose_for_model(model_type=model_type, assets_root=assets_dir)
        except Exception as e:
            errors.append(f"{model_type}:{e}")

    print("Warning: failed to compute true T-pose. attempts=" + " | ".join(errors))
    return None, None


def _shared_beta_npz_kwargs(meta: dict[str, object] | None) -> dict[str, np.ndarray]:
    if not isinstance(meta, dict) or len(meta) <= 0:
        return {}
    out: dict[str, np.ndarray] = {
        "beta_refit_enabled": np.asarray([1 if bool(meta.get("enabled", False)) else 0], dtype=np.uint8),
    }
    if "shared_beta" in meta:
        out["beta_shared"] = np.asarray(meta["shared_beta"], dtype=np.float32).reshape(1, -1)
        out["beta_fit_strategy"] = np.asarray(["shared_beta_refit"], dtype=object)
    if "candidate_indices" in meta:
        out["beta_shared_candidate_indices"] = np.asarray(meta["candidate_indices"], dtype=np.int32).reshape(-1)
    if "candidate_weights" in meta:
        out["beta_shared_candidate_weights"] = np.asarray(meta["candidate_weights"], dtype=np.float32).reshape(-1)
    if "candidate_reason" in meta:
        out["beta_shared_candidate_reason"] = np.asarray([str(meta["candidate_reason"])], dtype=object)
    if "anchor_index" in meta:
        out["beta_shared_anchor_index"] = np.asarray([int(meta["anchor_index"])], dtype=np.int32)
    if "refit_vertex_loss" in meta:
        out["beta_refit_vertex_loss"] = np.asarray(meta["refit_vertex_loss"], dtype=np.float32).reshape(-1)
    if "refit_joint_loss" in meta:
        out["beta_refit_joint_loss"] = np.asarray(meta["refit_joint_loss"], dtype=np.float32).reshape(-1)
    if "refit_total_loss" in meta:
        out["beta_refit_total_loss"] = np.asarray(meta["refit_total_loss"], dtype=np.float32).reshape(-1)
    return out


def _shared_beta_summary(meta: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(meta, dict) or len(meta) <= 0:
        return None
    out: dict[str, object] = {"enabled": bool(meta.get("enabled", False))}
    if "reason" in meta:
        out["reason"] = str(meta["reason"])
    if "device" in meta:
        out["device"] = str(meta["device"])
    if "candidate_reason" in meta:
        out["candidate_reason"] = str(meta["candidate_reason"])
    if "anchor_index" in meta:
        out["anchor_index"] = int(meta["anchor_index"])
    if "shared_beta" in meta:
        out["shared_beta"] = np.asarray(meta["shared_beta"], dtype=np.float32).reshape(-1).round(6).tolist()
    if "candidate_indices" in meta:
        idx = np.asarray(meta["candidate_indices"], dtype=np.int32).reshape(-1)
        out["candidate_count"] = int(idx.shape[0])
        out["candidate_indices_head"] = idx[: min(8, int(idx.shape[0]))].tolist()
    if "refit_total_loss" in meta:
        loss = np.asarray(meta["refit_total_loss"], dtype=np.float32).reshape(-1)
        if loss.size > 0:
            out["refit_total_loss_mean"] = float(loss.mean())
            out["refit_total_loss_max"] = float(loss.max())
    if "canonical_available" in meta:
        out["canonical_available"] = bool(meta["canonical_available"])
    return out


def _silhouette_refine_npz_kwargs(meta: dict[str, object] | None) -> dict[str, np.ndarray]:
    if not isinstance(meta, dict) or len(meta) <= 0:
        return {}
    out: dict[str, np.ndarray] = {
        "silhouette_refine_enabled": np.asarray([1 if bool(meta.get("enabled", False)) else 0], dtype=np.uint8),
    }
    for src_key, dst_key, dtype in (
        ("iou_before", "silhouette_iou_before", np.float32),
        ("iou_after", "silhouette_iou_after", np.float32),
        ("outside_before", "silhouette_outside_before", np.float32),
        ("outside_after", "silhouette_outside_after", np.float32),
        ("score_before", "silhouette_score_before", np.float32),
        ("score_after", "silhouette_score_after", np.float32),
        ("refit_total_loss", "silhouette_refine_total_loss", np.float32),
        ("accepted", "silhouette_refine_accepted", np.uint8),
        ("skipped", "silhouette_refine_skipped", np.uint8),
    ):
        if src_key in meta:
            out[dst_key] = np.asarray(meta[src_key], dtype=dtype).reshape(-1)
    if "mask_sources" in meta:
        out["silhouette_mask_sources"] = np.asarray(meta["mask_sources"], dtype=object).reshape(-1)
    return out


def _silhouette_refine_summary(meta: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(meta, dict) or len(meta) <= 0:
        return None
    out: dict[str, object] = {"enabled": bool(meta.get("enabled", False))}
    if "device" in meta:
        out["device"] = str(meta["device"])
    if "mask_source" in meta:
        out["mask_source"] = str(meta["mask_source"])
    accepted = np.asarray(meta.get("accepted", np.zeros((0,), dtype=np.uint8)), dtype=np.uint8).reshape(-1)
    skipped = np.asarray(meta.get("skipped", np.zeros((0,), dtype=np.uint8)), dtype=np.uint8).reshape(-1)
    if accepted.size > 0:
        out["accepted_count"] = int(int(accepted.sum()))
        out["frame_count"] = int(accepted.shape[0])
    if skipped.size > 0:
        out["skipped_count"] = int(int(skipped.sum()))
    for key_in, key_before, key_after in (
        ("iou", "iou_before", "iou_after"),
        ("outside", "outside_before", "outside_after"),
        ("score", "score_before", "score_after"),
    ):
        before = np.asarray(meta.get(key_before, np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        after = np.asarray(meta.get(key_after, np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if before.size > 0 and after.size > 0:
            mask = np.isfinite(before) & np.isfinite(after)
            if bool(mask.any()):
                out[f"{key_in}_before_mean"] = float(before[mask].mean())
                out[f"{key_in}_after_mean"] = float(after[mask].mean())
    loss = np.asarray(meta.get("refit_total_loss", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if loss.size > 0:
        mask = np.isfinite(loss)
        if bool(mask.any()):
            out["refit_total_loss_mean"] = float(loss[mask].mean())
            out["refit_total_loss_max"] = float(loss[mask].max())
    return out


def _load_person_masks_by_name(person_masks_npz: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not person_masks_npz.exists():
        return out
    try:
        z = np.load(person_masks_npz, allow_pickle=True)
    except Exception:
        return out
    if "frame_names" not in z.files or "masks" not in z.files:
        return out
    names = [str(x) for x in np.asarray(z["frame_names"]).reshape(-1).tolist()]
    masks = np.asarray(z["masks"], dtype=np.uint8)
    n = min(len(names), int(masks.shape[0]))
    for i in range(n):
        out[names[i]] = (np.asarray(masks[i], dtype=np.uint8) > 0).astype(np.uint8) * 255
    return out


def _frame_index_from_name(frame_name: str, fallback: int) -> int:
    stem = Path(str(frame_name)).stem
    if stem.isdigit():
        return int(stem)
    return int(fallback)


def _build_smplestx_frame_records(
    selected_result_dir: Path,
    selected_meta: dict,
    faces_smplx: np.ndarray | None,
    assets_dir: Path,
) -> list[dict]:
    selected_files = [str(x) for x in selected_meta.get("selected_files", [])]
    selected_input_files = [str(x) for x in selected_meta.get("selected_input_files", [])]
    selected_global_idx = [int(x) for x in selected_meta.get("selected_indices_global", list(range(len(selected_files))))]
    if len(selected_files) == 0:
        raise RuntimeError("selected_frames_meta.json has empty selected_files.")
    if len(selected_global_idx) != len(selected_files):
        selected_global_idx = list(range(len(selected_files)))
    if faces_smplx is None:
        raise RuntimeError("SMPL-X faces are missing.")
    faces = faces_smplx
    model_type = "smplx"

    smplx_dir = selected_result_dir / "smplx"
    meta_dir = selected_result_dir / "meta"
    if not smplx_dir.exists():
        raise FileNotFoundError(f"SMPLest-X result folder missing: {smplx_dir}")

    available: dict[int, tuple[Path, Path | None]] = {}
    for local_idx in range(len(selected_files)):
        stem = f"{local_idx + 1:05d}_0"
        npz_path = smplx_dir / f"{stem}.npz"
        meta_path = meta_dir / f"{stem}.json"
        if npz_path.exists():
            available[int(local_idx)] = (npz_path, meta_path if meta_path.exists() else None)
    if len(available) <= 0:
        raise RuntimeError(f"No SMPLest-X frame predictions found under: {smplx_dir}")

    avail_local = np.asarray(sorted(available.keys()), dtype=np.int64)

    def _resolve_local_source(local_idx: int) -> tuple[str, int]:
        li = int(local_idx)
        if li in available:
            return "exact", li
        pos = int(np.searchsorted(avail_local, li))
        if pos <= 0:
            return "nearest_start", int(avail_local[0])
        if pos >= int(avail_local.shape[0]):
            return "nearest_end", int(avail_local[-1])
        l_local = int(avail_local[pos - 1])
        r_local = int(avail_local[pos])
        if abs(li - l_local) <= abs(r_local - li):
            return "nearest_left", l_local
        return "nearest_right", r_local

    records: list[dict] = []
    for local_idx, frame_name in enumerate(selected_files):
        source_mode, src_local = _resolve_local_source(int(local_idx))
        npz_path, meta_path = available[int(src_local)]
        params = _load_smplestx_npz_params(npz_path)
        meta = None
        if meta_path is not None and meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        verts_cam, joints_cam, cam_t = _rebuild_smplx_from_params(
            params=params,
            assets_dir=assets_dir,
            gender="neutral",
        )
        verts_local = np.asarray(verts_cam - cam_t.reshape(1, 3), dtype=np.float32)
        joints2d = _project_points_with_meta(joints_cam, meta)

        global_idx = int(selected_global_idx[local_idx]) if local_idx < len(selected_global_idx) else int(local_idx)
        frame_idx = _frame_index_from_name(frame_name=frame_name, fallback=global_idx)

        records.append(
            {
                "frame_name": str(frame_name),
                "frame_idx": int(frame_idx),
                "global_idx": int(global_idx),
                "pred_idx": int(src_local),
                "pred_source": str(source_mode),
                "pred_source_left_idx": int(src_local),
                "pred_source_right_idx": int(src_local),
                "pred_source_alpha": float(0.0),
                "verts_local": verts_local,
                "verts_cam": verts_cam,
                "cam_t": cam_t,
                "joints2d": joints2d,
                "joints_cam": joints_cam,
                "camera_focal": (
                    np.asarray(meta.get("focal"), dtype=np.float32).reshape(-1)[:2]
                    if isinstance(meta, dict) and meta.get("focal", None) is not None
                    else None
                ),
                "camera_princpt": (
                    np.asarray(meta.get("princpt"), dtype=np.float32).reshape(-1)[:2]
                    if isinstance(meta, dict) and meta.get("princpt", None) is not None
                    else None
                ),
                "faces": faces,
                "model_type": model_type,
                "gender": "neutral",
                "smplestx_params": params,
                "smplestx_input_file": (
                    str(selected_input_files[local_idx])
                    if local_idx < len(selected_input_files)
                    else f"{local_idx + 1:06d}.jpg"
                ),
            }
        )
    return records


def _write_stage1_sequence_assets_full(
    *,
    out_dir: Path,
    frames_dir: Path,
    records: list[dict],
    canonical_verts_cam: np.ndarray | None = None,
    canonical_joints_cam: np.ndarray | None = None,
    extra_params_kwargs: dict[str, np.ndarray] | None = None,
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

    # Persist a representative camera K for consumers that only support one camera.
    # Per-frame intrinsics are stored in smpl_params_sequence.npz and should take precedence.
    first_name = str(records[0]["frame_name"])
    first_img = cv2.imread(str(frames_dir / first_name), cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read frame image: {frames_dir / first_name}")
    h0, w0 = int(first_img.shape[0]), int(first_img.shape[1])
    fx0, fy0, cx0, cy0 = _camera_intrinsics_from_record(records[0], width=w0, height=h0)
    K_list = [float(fx0), 0.0, float(cx0), 0.0, float(fy0), float(cy0), 0.0, 0.0, 1.0]
    with cameras_json.open("w", encoding="utf-8") as f:
        json.dump({"K": K_list, "source": "smplestx_meta_or_fallback"}, f, indent=2)

    frame_names: list[str] = []
    verts_cam_seq: list[np.ndarray] = []
    verts_cam_raw_seq: list[np.ndarray] = []
    joints_cam_seq: list[np.ndarray] = []
    masks_seq: list[np.ndarray] = []
    cached_masks_by_name: dict[str, np.ndarray] = {}
    export_hint_cache: dict[str, dict[str, object]] = {}
    kpt_map: dict[str, list[list[float]]] = {}
    align_scale_seq: list[float] = []
    align_tx_seq: list[float] = []
    align_ty_seq: list[float] = []
    align_iou_seq: list[float] = []
    align_outside_seq: list[float] = []
    align_accepted_seq: list[np.uint8] = []
    export_variant_seq: list[str] = []
    export_score_seq: list[float] = []
    export_boundary_seq: list[float] = []
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

    if person_masks_npz.exists():
        try:
            pm_cache = np.load(person_masks_npz, allow_pickle=True)
            if "frame_names" in pm_cache.files and "masks" in pm_cache.files:
                pm_names = [str(x) for x in np.asarray(pm_cache["frame_names"]).reshape(-1).tolist()]
                pm_masks = np.asarray(pm_cache["masks"], dtype=np.uint8)
                n_pm = min(len(pm_names), int(pm_masks.shape[0]))
                for i_pm in range(n_pm):
                    cached_masks_by_name[pm_names[i_pm]] = (pm_masks[i_pm] > 0).astype(np.uint8) * 255
        except Exception:
            cached_masks_by_name = {}

    need_mask_precache = len(cached_masks_by_name) < len(records)
    if need_mask_precache:
        pre_iter = tqdm(records, desc="Stage1 precompute masks")
        for rec in pre_iter:
            frame_name = str(rec["frame_name"])
            if frame_name in export_hint_cache and frame_name in cached_masks_by_name:
                continue
            img_path = frames_dir / frame_name
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read frame image: {img_path}")
            hh, ww = int(img.shape[0]), int(img.shape[1])
            fx, fy, cx, cy = _camera_intrinsics_from_record(rec, width=ww, height=hh)
            faces_rec = shared_faces if shared_faces is not None else np.asarray(rec["faces"], dtype=np.int32)
            uv_raw, valid_hint = _project_verts_to_uv(
                verts=np.asarray(rec["verts_local"], dtype=np.float32),
                cam_t=np.asarray(rec["cam_t"], dtype=np.float32),
                width=ww,
                height=hh,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
            mask_hint_raw = _rasterize_mesh_mask(
                uv=uv_raw,
                valid_v=valid_hint,
                faces=faces_rec,
                image_hw=(hh, ww),
            )
            export_hint_cache[frame_name] = {
                "hh": hh,
                "ww": ww,
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "faces": faces_rec,
                "uv_raw": uv_raw,
                "valid_hint": valid_hint,
                "mask_hint_raw": mask_hint_raw,
            }
            if frame_name not in cached_masks_by_name:
                cached_masks_by_name[frame_name] = _estimate_person_mask_modnet_stage1(
                    img,
                    mesh_hint_mask=mask_hint_raw,
                )

    export_iter = tqdm(records, desc="Stage1 export sequence")
    for rec in export_iter:
        frame_name = str(rec["frame_name"])
        frame_idx = int(rec["frame_idx"])
        verts_cam_raw = np.asarray(rec["verts_cam"], dtype=np.float32)
        verts_cam = np.asarray(verts_cam_raw, dtype=np.float32)
        faces = shared_faces if shared_faces is not None else np.asarray(rec["faces"], dtype=np.int32)
        joints2d = rec.get("joints2d", None)
        joints_cam = rec.get("joints_cam", None)
        hint_cached = export_hint_cache.get(frame_name, None)
        if hint_cached is not None:
            hh = int(hint_cached["hh"])
            ww = int(hint_cached["ww"])
            fx = float(hint_cached["fx"])
            fy = float(hint_cached["fy"])
            cx = float(hint_cached["cx"])
            cy = float(hint_cached["cy"])
            faces = np.asarray(hint_cached["faces"], dtype=np.int32)
            uv_raw = np.asarray(hint_cached["uv_raw"], dtype=np.float32)
            valid_hint = np.asarray(hint_cached["valid_hint"], dtype=bool)
            mask_hint_raw = np.asarray(hint_cached["mask_hint_raw"], dtype=np.uint8)
            mask = np.asarray(cached_masks_by_name[frame_name], dtype=np.uint8)
        else:
            img = cv2.imread(str(frames_dir / frame_name), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read frame image: {frames_dir / frame_name}")
            hh, ww = int(img.shape[0]), int(img.shape[1])
            fx, fy, cx, cy = _camera_intrinsics_from_record(rec, width=ww, height=hh)
            uv_raw, valid_hint = _project_verts_to_uv(
                verts=np.asarray(rec["verts_local"], dtype=np.float32),
                cam_t=np.asarray(rec["cam_t"], dtype=np.float32),
                width=ww,
                height=hh,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
            mask_hint_raw = _rasterize_mesh_mask(
                uv=uv_raw,
                valid_v=valid_hint,
                faces=faces,
                image_hw=(hh, ww),
            )
            mask = cached_masks_by_name.get(frame_name, None)
            if mask is None or tuple(mask.shape[:2]) != (hh, ww):
                img = cv2.imread(str(frames_dir / frame_name), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Cannot read frame image: {frames_dir / frame_name}")
                mask = _estimate_person_mask_modnet_stage1(
                    img,
                    mesh_hint_mask=mask_hint_raw,
                )
        def _eval_mask_candidate(mask_candidate: np.ndarray) -> tuple[float, float, float, float, float, float]:
            iou_c = float(mask_iou(mask_candidate, mask))
            outside_c = float(mask_outside_ratio(mask_candidate, mask))
            area_ratio_c = float((np.asarray(mask_candidate, dtype=np.uint8) > 0).sum()) / float(max(1, (np.asarray(mask, dtype=np.uint8) > 0).sum()))
            boundary_c = float(_boundary_mean_distance_px(mask_candidate, mask))
            boundary_p95_c = float(_boundary_p95_distance_px(mask_candidate, mask))
            score_c = float(
                _export_candidate_score(
                    iou=iou_c,
                    outside=outside_c,
                    area_ratio=area_ratio_c,
                    boundary_mean_px=boundary_c,
                    boundary_p95_px=boundary_p95_c,
                )
            )
            return iou_c, outside_c, area_ratio_c, boundary_c, boundary_p95_c, score_c

        candidates: list[dict[str, object]] = []

        def _push_candidate(
            *,
            label: str,
            verts_cam_in: np.ndarray,
            cam_t_in: np.ndarray,
            joints_cam_in: np.ndarray | None,
            params_in: dict | None,
            joints2d_in: object,
            mask_in: np.ndarray,
            align_report_in: dict[str, object] | None,
        ) -> None:
            iou_c, outside_c, area_ratio_c, boundary_c, boundary_p95_c, score_c = _eval_mask_candidate(mask_in)
            candidates.append(
                {
                    "label": str(label),
                    "verts_cam": np.asarray(verts_cam_in, dtype=np.float32),
                    "cam_t": np.asarray(cam_t_in, dtype=np.float32).reshape(3),
                    "joints_cam": (None if joints_cam_in is None else np.asarray(joints_cam_in, dtype=np.float32)),
                    "params": (None if not isinstance(params_in, dict) else dict(params_in)),
                    "joints2d": None if joints2d_in is None else np.asarray(joints2d_in, dtype=np.float32),
                    "mask_hint": np.asarray(mask_in, dtype=np.uint8),
                    "align_report": ({} if align_report_in is None else dict(align_report_in)),
                    "iou": float(iou_c),
                    "outside": float(outside_c),
                    "area_ratio": float(area_ratio_c),
                    "boundary_mean_px": float(boundary_c),
                    "boundary_p95_px": float(boundary_p95_c),
                    "score": float(score_c),
                }
            )

        def _push_scaled_candidate(
            *,
            label: str,
            verts_cam_base: np.ndarray,
            cam_t_base: np.ndarray,
            joints_cam_base: np.ndarray | None,
            params_base: dict | None,
            scale_value: float,
        ) -> None:
            verts_scaled = apply_similarity_to_camera_points(
                np.asarray(verts_cam_base, dtype=np.float32),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                scale=float(scale_value),
                tx_px=0.0,
                ty_px=0.0,
            )
            joints_scaled = None
            if joints_cam_base is not None:
                joints_scaled = apply_similarity_to_camera_points(
                    np.asarray(joints_cam_base, dtype=np.float32),
                    fx=float(fx),
                    fy=float(fy),
                    cx=float(cx),
                    cy=float(cy),
                    scale=float(scale_value),
                    tx_px=0.0,
                    ty_px=0.0,
                )
            cam_t_scaled = apply_similarity_to_camera_points(
                np.asarray(cam_t_base, dtype=np.float32).reshape(1, 3),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                scale=float(scale_value),
                tx_px=0.0,
                ty_px=0.0,
            ).reshape(3)
            verts_local_scaled = np.asarray(verts_scaled, dtype=np.float32) - np.asarray(cam_t_scaled, dtype=np.float32).reshape(1, 3)
            uv_scaled, valid_scaled = _project_verts_to_uv(
                verts=verts_local_scaled,
                cam_t=np.asarray(cam_t_scaled, dtype=np.float32),
                width=ww,
                height=hh,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
            _push_candidate(
                label=str(label),
                verts_cam_in=verts_scaled,
                cam_t_in=cam_t_scaled,
                joints_cam_in=joints_scaled,
                params_in=params_base,
                joints2d_in=None,
                mask_in=_rasterize_mesh_mask(uv=uv_scaled, valid_v=valid_scaled, faces=faces, image_hw=(hh, ww)),
                align_report_in={
                    "applied": True,
                    "accepted": True,
                    "source": str(label),
                    "scale": float(scale_value),
                    "tx_px": 0.0,
                    "ty_px": 0.0,
                },
            )

        params_current = rec.get("smplestx_params", None)
        joints2d_current = rec.get("joints2d", None)
        _push_candidate(
            label="refined_native",
            verts_cam_in=np.asarray(rec["verts_cam"], dtype=np.float32),
            cam_t_in=np.asarray(rec["cam_t"], dtype=np.float32),
            joints_cam_in=(None if joints_cam is None else np.asarray(joints_cam, dtype=np.float32)),
            params_in=params_current,
            joints2d_in=joints2d_current,
            mask_in=mask_hint_raw,
            align_report_in={"applied": False, "accepted": False, "source": "refined_native"},
        )
        refined_native_outside = float(candidates[0]["outside"])
        refined_native_area = float(candidates[0]["area_ratio"])
        if refined_native_outside > 0.065:
            _push_scaled_candidate(
                label="refined_clamp_099",
                verts_cam_base=np.asarray(rec["verts_cam"], dtype=np.float32),
                cam_t_base=np.asarray(rec["cam_t"], dtype=np.float32),
                joints_cam_base=(None if joints_cam is None else np.asarray(joints_cam, dtype=np.float32)),
                params_base=params_current,
                scale_value=0.99,
            )
        if refined_native_outside > 0.090:
            _push_scaled_candidate(
                label="refined_clamp_0985",
                verts_cam_base=np.asarray(rec["verts_cam"], dtype=np.float32),
                cam_t_base=np.asarray(rec["cam_t"], dtype=np.float32),
                joints_cam_base=(None if joints_cam is None else np.asarray(joints_cam, dtype=np.float32)),
                params_base=params_current,
                scale_value=0.985,
            )
        if refined_native_outside < 0.055 and refined_native_area < 0.920:
            _push_scaled_candidate(
                label="refined_expand_1005",
                verts_cam_base=np.asarray(rec["verts_cam"], dtype=np.float32),
                cam_t_base=np.asarray(rec["cam_t"], dtype=np.float32),
                joints_cam_base=(None if joints_cam is None else np.asarray(joints_cam, dtype=np.float32)),
                params_base=params_current,
                scale_value=1.005,
            )

        pre_verts_cam = rec.get("silhouette_refine_pre_verts_cam", None)
        pre_cam_t = rec.get("silhouette_refine_pre_cam_t", None)
        if pre_verts_cam is not None and pre_cam_t is not None:
            pre_verts_cam_arr = np.asarray(pre_verts_cam, dtype=np.float32)
            pre_cam_t_arr = np.asarray(pre_cam_t, dtype=np.float32).reshape(3)
            pre_verts_local_arr = rec.get("silhouette_refine_pre_verts_local", None)
            if pre_verts_local_arr is None:
                pre_verts_local_arr = np.asarray(pre_verts_cam_arr - pre_cam_t_arr.reshape(1, 3), dtype=np.float32)
            pre_verts_local_arr = np.asarray(pre_verts_local_arr, dtype=np.float32)
            pre_joints_cam_arr = rec.get("silhouette_refine_pre_joints_cam", None)
            if pre_joints_cam_arr is not None:
                pre_joints_cam_arr = np.asarray(pre_joints_cam_arr, dtype=np.float32)
            pre_joints2d_arr = rec.get("silhouette_refine_pre_joints2d", None)
            pre_params = rec.get("silhouette_refine_pre_smplestx_params", None)
            uv_pre_raw, valid_pre = _project_verts_to_uv(
                verts=pre_verts_local_arr,
                cam_t=pre_cam_t_arr,
                width=ww,
                height=hh,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
            mask_pre_raw = _rasterize_mesh_mask(
                uv=uv_pre_raw,
                valid_v=valid_pre,
                faces=faces,
                image_hw=(hh, ww),
            )
            _push_candidate(
                label="raw_native",
                verts_cam_in=pre_verts_cam_arr,
                cam_t_in=pre_cam_t_arr,
                joints_cam_in=pre_joints_cam_arr,
                params_in=pre_params,
                joints2d_in=pre_joints2d_arr,
                mask_in=mask_pre_raw,
                align_report_in={"applied": False, "accepted": False, "source": "raw_native"},
            )
            raw_native_outside = float(candidates[-1]["outside"])
            raw_native_area = float(candidates[-1]["area_ratio"])
            if raw_native_outside < 0.060 and raw_native_area < 0.915:
                _push_scaled_candidate(
                    label="raw_expand_1005",
                    verts_cam_base=pre_verts_cam_arr,
                    cam_t_base=pre_cam_t_arr,
                    joints_cam_base=pre_joints_cam_arr,
                    params_base=pre_params,
                    scale_value=1.005,
                )
        best_candidate = candidates[0]
        for cand in candidates[1:]:
            cand_score = float(cand["score"])
            best_score = float(best_candidate["score"])
            cand_out = float(cand["outside"])
            best_out = float(best_candidate["outside"])
            cand_iou = float(cand["iou"])
            best_iou = float(best_candidate["iou"])
            cand_boundary = float(cand["boundary_mean_px"])
            best_boundary = float(best_candidate["boundary_mean_px"])
            cand_boundary_p95 = float(cand.get("boundary_p95_px", float("inf")))
            best_boundary_p95 = float(best_candidate.get("boundary_p95_px", float("inf")))
            if (cand_out + 0.01) < best_out and cand_iou >= (best_iou - 0.015):
                best_candidate = cand
                continue
            if (cand_out + 0.005) < best_out and cand_boundary <= (best_boundary + 0.75):
                best_candidate = cand
                continue
            if cand_out <= 0.05 and best_out <= 0.05 and cand_iou >= (best_iou + 0.02) and cand_boundary <= (best_boundary + 0.75):
                best_candidate = cand
                continue
            if cand_out <= 0.05 and best_out <= 0.05 and (cand_boundary_p95 + 1.0) < best_boundary_p95 and cand_iou >= (best_iou - 0.01):
                best_candidate = cand
                continue
            if (cand_boundary_p95 + 1.0) < best_boundary_p95 and cand_out <= (best_out + 0.003) and cand_iou >= (best_iou - 0.015):
                best_candidate = cand
                continue
            if cand_score > (best_score + 1e-6):
                best_candidate = cand
                continue
            if (cand_out + 1e-3) < best_out and cand_score >= (best_score - 0.02):
                best_candidate = cand
                continue
            if abs(cand_score - best_score) <= 1e-6 and cand_out < (best_out - 1e-6):
                best_candidate = cand
                continue
            if abs(cand_score - best_score) <= 1e-6 and abs(cand_out - best_out) <= 1e-6 and cand_iou > best_iou:
                best_candidate = cand

        verts_cam = np.asarray(best_candidate["verts_cam"], dtype=np.float32)
        rec["cam_t"] = np.asarray(best_candidate["cam_t"], dtype=np.float32).reshape(3)
        joints_cam = best_candidate["joints_cam"]
        if joints_cam is not None:
            joints_cam = np.asarray(joints_cam, dtype=np.float32)
        joints2d = best_candidate["joints2d"]
        align_report = dict(best_candidate.get("align_report", {}))
        align_report["selected_variant"] = str(best_candidate["label"])
        align_report["selected_score"] = float(best_candidate["score"])
        align_report["selected_boundary_mean_px"] = float(best_candidate["boundary_mean_px"])
        align_report["selected_boundary_p95_px"] = float(best_candidate.get("boundary_p95_px", float("inf")))
        mask_hint = np.asarray(best_candidate["mask_hint"], dtype=np.uint8)
        rec["verts_cam"] = np.asarray(verts_cam, dtype=np.float32)
        rec["verts_local"] = np.asarray(
            np.asarray(rec["verts_cam"], dtype=np.float32) - np.asarray(rec["cam_t"], dtype=np.float32).reshape(1, 3),
            dtype=np.float32,
        )
        params_sync = best_candidate.get("params", None)
        if isinstance(params_sync, dict):
            params_sync = dict(params_sync)
            params_sync["transl"] = np.asarray(rec["cam_t"], dtype=np.float32).reshape(1, 3)
            rec["smplestx_params"] = params_sync

        obj_name = f"frame_{frame_idx:06d}.obj"
        export_obj(seq_dir / obj_name, verts_cam, faces, face_text=shared_face_text)

        rec["verts_cam"] = np.asarray(verts_cam, dtype=np.float32)
        if joints_cam is not None:
            rec["joints_cam"] = np.asarray(joints_cam, dtype=np.float32)

        if joints2d is not None and np.asarray(joints2d).ndim == 2 and np.asarray(joints2d).shape[1] >= 2:
            j2 = np.asarray(joints2d, dtype=np.float32)
            if j2.shape[1] >= 3:
                kpts = [[float(p[0]), float(p[1]), float(np.clip(p[2], 0.0, 1.0))] for p in j2]
            else:
                kpts = [[float(p[0]), float(p[1]), 1.0] for p in j2]
        elif joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
            uv, valid = _project_points_to_uv(
                np.asarray(joints_cam, dtype=np.float32),
                width=ww,
                height=hh,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
            kpts = [[float(uv[i, 0]), float(uv[i, 1]), float(1.0 if bool(valid[i]) else 0.0)] for i in range(uv.shape[0])]
        else:
            kpts = []

        frame_names.append(frame_name)
        verts_cam_seq.append(verts_cam)
        verts_cam_raw_seq.append(verts_cam_raw)
        if joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
            joints_cam_seq.append(np.asarray(joints_cam, dtype=np.float32))
        else:
            joints_cam_seq.append(np.zeros((0, 3), dtype=np.float32))
        masks_seq.append((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8))
        kpt_map[frame_name] = kpts
        align_accepted = bool(str(best_candidate["label"]).endswith("_aligned"))
        export_variant_seq.append(str(best_candidate["label"]))
        export_score_seq.append(float(best_candidate["score"]))
        export_boundary_seq.append(float(best_candidate["boundary_mean_px"]))
        align_scale_seq.append(float(align_report.get("scale", 1.0)) if align_accepted else 1.0)
        align_tx_seq.append(float(align_report.get("tx_px", 0.0)) if align_accepted else 0.0)
        align_ty_seq.append(float(align_report.get("ty_px", 0.0)) if align_accepted else 0.0)
        align_iou_seq.append(float(best_candidate["iou"]))
        align_outside_seq.append(float(best_candidate["outside"]))
        align_accepted_seq.append(np.uint8(1 if align_accepted else 0))

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
        verts_cam_raw=np.asarray(verts_cam_raw_seq, dtype=np.float32),
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
        "fit_strategy": np.asarray(["stage1_smplestx"], dtype=object),
        "export_upright_rot": np.eye(3, dtype=np.float32),
        "export_center_xy": np.asarray([0], dtype=np.int32),
        "export_center_xy_value": np.asarray([0.0, 0.0], dtype=np.float32),
        "export_on_floor": np.asarray([0], dtype=np.int32),
        "export_floor_z": np.asarray([0.0], dtype=np.float32),
    }
    focal_seq: list[np.ndarray] = []
    princpt_seq: list[np.ndarray] = []
    for rec in records:
        focal_xy = rec.get("camera_focal", None)
        princpt_xy = rec.get("camera_princpt", None)
        if focal_xy is None or princpt_xy is None:
            focal_seq = []
            princpt_seq = []
            break
        focal_arr = np.asarray(focal_xy, dtype=np.float32).reshape(-1)
        princpt_arr = np.asarray(princpt_xy, dtype=np.float32).reshape(-1)
        if focal_arr.size < 2 or princpt_arr.size < 2:
            focal_seq = []
            princpt_seq = []
            break
        focal_seq.append(focal_arr[:2])
        princpt_seq.append(princpt_arr[:2])
    if len(focal_seq) == len(records) and len(princpt_seq) == len(records):
        params_kwargs["camera_focal_xy"] = np.asarray(focal_seq, dtype=np.float32)
        params_kwargs["camera_princpt_xy"] = np.asarray(princpt_seq, dtype=np.float32)
    if len(align_scale_seq) == len(records):
        params_kwargs["camera_align_scale"] = np.asarray(align_scale_seq, dtype=np.float32)
        params_kwargs["camera_align_tx_px"] = np.asarray(align_tx_seq, dtype=np.float32)
        params_kwargs["camera_align_ty_px"] = np.asarray(align_ty_seq, dtype=np.float32)
        params_kwargs["camera_align_iou"] = np.asarray(align_iou_seq, dtype=np.float32)
        params_kwargs["camera_align_outside_ratio"] = np.asarray(align_outside_seq, dtype=np.float32)
        params_kwargs["camera_align_accepted"] = np.asarray(align_accepted_seq, dtype=np.uint8)
    if len(export_variant_seq) == len(records):
        params_kwargs["export_selected_variant"] = np.asarray(export_variant_seq, dtype=object)
        params_kwargs["export_selected_score"] = np.asarray(export_score_seq, dtype=np.float32)
        params_kwargs["export_selected_boundary_mean_px"] = np.asarray(export_boundary_seq, dtype=np.float32)
    param_keys = (
        "betas",
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
        "transl",
    )
    smplestx_params_seq: dict[str, list[np.ndarray]] = {k: [] for k in param_keys}
    smplestx_param_shape: dict[str, tuple[int, ...]] = {}
    for rec in records:
        params = rec.get("smplestx_params", None)
        if not isinstance(params, dict):
            continue
        for key in param_keys:
            if key not in params:
                continue
            arr = np.asarray(params[key], dtype=np.float32)
            smplestx_params_seq[key].append(arr)
            smplestx_param_shape.setdefault(key, tuple(arr.shape))
    for key, seq in smplestx_params_seq.items():
        if len(seq) != len(records):
            continue
        ref_shape = smplestx_param_shape.get(key, ())
        if any(tuple(np.asarray(arr).shape) != ref_shape for arr in seq):
            continue
        params_kwargs[key] = np.asarray(seq, dtype=np.float32)
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
    if extra_params_kwargs:
        params_kwargs.update(extra_params_kwargs)
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


def _check_stage1_requirements(assets_dir: Path, smplestx_dir: Path) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"smplestx": [], "python": []}

    if not smplestx_dir.exists():
        missing["smplestx"].append("third_party/SMPLest-X repo")

    repo_has_ckpt = False
    repo_pretrained = smplestx_dir / "pretrained_models"
    if repo_pretrained.exists():
        repo_has_ckpt = _find_any_file(
            repo_pretrained,
            [
                "smplest_x*.pth.tar",
                "smplest_x*.pth",
                "smplest_x*.pt",
                "SMPLest-X*.pth.tar",
                "SMPLestX*.pth.tar",
                "*smplest*x*.pth.tar",
            ],
        ) is not None

    assets_has_ckpt = _find_any_file(
        assets_dir,
        [
            "smplest_x*.pth.tar",
            "smplest_x*.pth",
            "smplest_x*.pt",
            "SMPLest-X*.pth.tar",
            "SMPLestX*.pth.tar",
            "*smplest*x*.pth.tar",
        ],
    ) is not None

    if not assets_has_ckpt and not repo_has_ckpt:
        missing["smplestx"].append("SMPLest-X pretrained checkpoint")

    try:
        resolve_modnet_ckpt_path(None)
    except Exception:
        missing["smplestx"].append("MODNet checkpoint")

    checks = [
        "smplx",
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
    parser = argparse.ArgumentParser("Stage1: SMPLest-X refine")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input", help="Video frame directory")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Assets root directory")
    parser.add_argument("--out_dir", type=str, default="outputs/stage1", help="All outputs are written here")
    parser.add_argument("--smplestx_frame_indices", type=str, default="-1", help="Frame indices passed to SMPLestX refine")
    parser.add_argument("--smplestx_stride", type=int, default=1, help="Sample every N frames for SMPLestX refine when smplestx_frame_indices is all/-1")
    parser.add_argument("--smplestx_ckpt", type=str, default="smplest_x_h", help="Optional explicit SMPLestX checkpoint path")
    parser.add_argument("--single_person", dest="single_person", action="store_true", help="Set SMPLestX single-person mode")
    parser.add_argument("--multi_person", dest="single_person", action="store_false", help="Disable SMPLestX single-person mode")
    parser.add_argument("--skip_smplestx", action="store_true", help="Skip SMPLestX stage")
    parser.add_argument("--smplestx_repo_dir", type=str, default="third_party/SMPLest-X", help="SMPLest-X repo dir")
    parser.add_argument("--minimal_stage1_outputs", action="store_true", help="Keep only sequence assets required by stage2; delete intermediates")
    parser.add_argument(
        "--disable_shared_beta_refit",
        dest="disable_shared_beta_refit",
        action="store_true",
        help="Disable stage1 postprocess that shares beta and refits per-frame pose/transl",
    )
    parser.add_argument(
        "--enable_shared_beta_refit",
        dest="disable_shared_beta_refit",
        action="store_false",
        help="Enable shared-beta refit. Disabled by default because alignment-first stage1 prefers raw SMPLest-X shape plus silhouette refine.",
    )
    parser.add_argument("--shared_beta_min_align_iou", type=float, default=0.99, help="Use align_iou >= threshold when picking shared-beta frames if enough frames pass")
    parser.add_argument("--shared_beta_min_candidate_frames", type=int, default=8, help="Minimum number of frames required before applying exact/high-quality candidate filtering")
    parser.add_argument("--shared_beta_refit_iters", type=int, default=90, help="Optimization steps for each per-frame fixed-beta refit")
    parser.add_argument("--shared_beta_refit_lr", type=float, default=0.02, help="Learning rate for each per-frame fixed-beta refit")
    parser.add_argument("--shared_beta_refit_vertex_samples", type=int, default=4096, help="Number of mesh vertices sampled for each refit loss")
    parser.add_argument("--shared_beta_refit_device", type=str, default="auto", help="Device for shared-beta refit: auto/cpu/cuda")
    parser.add_argument("--disable_silhouette_refine", action="store_true", help="Disable stage1 silhouette-driven SMPL parameter refinement")
    parser.add_argument("--silhouette_refine_mask_source", type=str, default="modnet", choices=["modnet", "cache"], help="Person mask source for silhouette refine")
    parser.add_argument("--silhouette_refine_iters", type=int, default=55, help="Optimization steps for each silhouette-refine frame")
    parser.add_argument("--silhouette_refine_lr", type=float, default=0.01, help="Learning rate for silhouette refine")
    parser.add_argument("--silhouette_refine_device", type=str, default="auto", help="Device for silhouette refine: auto/cpu/cuda")
    parser.add_argument("--silhouette_refine_modnet_ckpt", type=str, default="", help="Optional MODNet checkpoint path used by silhouette refine when mask_source=modnet")
    parser.add_argument("--build_fashn_face_semantics", dest="build_fashn_face_semantics", action="store_true", help="Run local FASHN parsing after stage1 and export SMPL face-level exposed/covered labels")
    parser.add_argument("--skip_fashn_face_semantics", dest="build_fashn_face_semantics", action="store_false", help="Skip FASHN face-semantics postprocess after stage1")
    parser.add_argument("--fashn_face_semantics_out_dir", type=str, default="", help="Optional output dir for FASHN->SMPL face semantics; default is <out_dir>/fashn_face_semantics")
    parser.add_argument("--fashn_device", type=str, default="cuda", help="FASHN inference device: cuda/cpu/auto")
    parser.add_argument("--fashn_max_edge", type=int, default=384, help="Resize each frame so its longest edge is at most this size before FASHN inference")
    parser.add_argument("--fashn_min_non_bg_votes", type=int, default=8, help="Minimum non-background pixel votes required before assigning a non-unknown face label")
    parser.add_argument("--fashn_frame_stride", type=int, default=1, help="Use every Nth frame when building face semantics")
    parser.add_argument("--fashn_max_frames", type=int, default=0, help="Optional cap on number of frames used for FASHN face semantics; 0 means all selected frames")
    parser.set_defaults(build_fashn_face_semantics=True, single_person=True, disable_shared_beta_refit=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    smplestx_dir = (project_root / args.smplestx_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets dir not found: {assets_dir}")
    if not smplestx_dir.exists():
        raise FileNotFoundError(f"SMPLestX repo not found: {smplestx_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    smplestx_out = out_dir / "smplestx"
    final_out = out_dir / "final"
    smplestx_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(frames_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    print(f"Frames: {len(frames)}")

    smplestx_indices_raw = str(args.smplestx_frame_indices).strip().lower()
    smplestx_stride = max(1, int(args.smplestx_stride))
    if smplestx_indices_raw in ("", "-1", "all", "*"):
        if smplestx_stride == 1:
            smplestx_indices_arg = "-1"
            smplestx_selected_count = len(frames)
        else:
            smplestx_idx = list(range(0, len(frames), smplestx_stride))
            smplestx_indices_arg = " ".join(str(i) for i in smplestx_idx)
            smplestx_selected_count = len(smplestx_idx)
    else:
        smplestx_indices_arg = str(args.smplestx_frame_indices).strip()
        smplestx_selected_count = -1

    if smplestx_selected_count > 0:
        print(
            f"SMPLestX frame selection: {smplestx_selected_count}/{len(frames)} frames "
            f"(stride={smplestx_stride}, indices='{smplestx_indices_arg[:64]}{'...' if len(smplestx_indices_arg) > 64 else ''}')"
        )

    req = _check_stage1_requirements(assets_dir=assets_dir, smplestx_dir=smplestx_dir)
    if len(req["smplestx"]) > 0 or len(req["python"]) > 0:
        parts: list[str] = []
        if len(req["smplestx"]) > 0:
            parts.append("missing_smplestx=" + ",".join(req["smplestx"]))
        if len(req["python"]) > 0:
            parts.append("missing_python=" + ",".join(req["python"]))
        raise RuntimeError("Stage1 SMPLest-X requirements not satisfied: " + "; ".join(parts))

    smplestx_ckpt: str | None = None
    if not bool(args.skip_smplestx):
        smplestx_ckpt = _resolve_smplestx_ckpt(
            assets_dir=assets_dir,
            smplestx_dir=smplestx_dir,
            override=(Path(args.smplestx_ckpt).resolve() if str(args.smplestx_ckpt).strip() else None),
        )
        _prepare_smplestx_runtime_data(assets_dir=assets_dir, smplestx_dir=smplestx_dir, smplestx_ckpt=smplestx_ckpt)

    env_base = os.environ.copy()

    if not bool(args.skip_smplestx):
        smplestx_cmd = [
            sys.executable,
            "-m",
            "avatar_gs.cli._run_smplestx_refine_frames",
            "--frames_dir",
            str(frames_dir),
            "--output_folder",
            str(smplestx_out),
            "--smplestx_repo_dir",
            str(smplestx_dir),
            "--pretrained_model",
            str(smplestx_ckpt),
            "--frame_indices",
            str(smplestx_indices_arg),
        ]
        if bool(args.single_person):
            smplestx_cmd.append("--single_person")
        _run_cmd(smplestx_cmd, cwd=project_root, env=env_base)
    else:
        print("Skipped SMPLestX stage.")
        return

    smplestx_result_dir = smplestx_out / "selected_frames"
    meta_json = smplestx_result_dir / "selected_frames_meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"SMPLestX selected-frame metadata missing: {meta_json}")
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    faces_smplx = None
    smplx_face_path = _find_any_file(assets_dir, ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"])
    if smplx_face_path is not None:
        faces_smplx = _load_faces_from_npz(smplx_face_path)

    records = _build_smplestx_frame_records(
        selected_result_dir=smplestx_result_dir,
        selected_meta=selected_meta,
        faces_smplx=faces_smplx,
        assets_dir=assets_dir,
    )
    if len(records) <= 0:
        raise RuntimeError("No frame records built from SMPLestX output.")

    first_rec = records[0]
    last_rec = records[-1]
    pred_source_counts: dict[str, int] = {}
    for rec in records:
        src = str(rec.get("pred_source", "unknown"))
        pred_source_counts[src] = int(pred_source_counts.get(src, 0) + 1)
    non_exact_local_idx = [
        int(i)
        for i, rec in enumerate(records)
        if str(rec.get("pred_source", "exact")) != "exact"
    ]

    canonical_verts_cam = None
    canonical_joints_cam = None
    try:
        canonical_verts_cam, canonical_joints_cam = _extract_tpose_cam_from_record(
            record=last_rec,
            assets_dir=assets_dir,
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
        canonical_verts_cam=canonical_verts_cam,
        canonical_joints_cam=canonical_joints_cam,
    )
    align_iou = None
    params_npz_path = Path(str(compat_assets.get("smpl_params_sequence_npz", out_dir / "smpl_params_sequence.npz")))
    if params_npz_path.exists():
        with np.load(params_npz_path, allow_pickle=True) as params_first_pass:
            if "camera_align_iou" in params_first_pass.files:
                align_iou = np.asarray(params_first_pass["camera_align_iou"], dtype=np.float32).reshape(-1)
    person_masks_npz_path = Path(str(compat_assets.get("person_masks_npz", out_dir / "person_masks.npz")))
    person_masks_by_name = _load_person_masks_by_name(person_masks_npz_path)
    postprocess_npz_kwargs: dict[str, np.ndarray] = {}
    need_final_export = False

    shared_beta_refit_summary: dict[str, object] | None = None
    if not bool(args.disable_shared_beta_refit):
        try:
            refit_cfg = SharedBetaRefitConfig(
                enabled=True,
                min_align_iou=float(args.shared_beta_min_align_iou),
                min_candidate_frames=int(max(1, int(args.shared_beta_min_candidate_frames))),
                refit_iters=int(max(1, int(args.shared_beta_refit_iters))),
                lr=float(args.shared_beta_refit_lr),
                vertex_sample_count=int(max(0, int(args.shared_beta_refit_vertex_samples))),
                device=str(args.shared_beta_refit_device),
            )
            refit_result = apply_shared_beta_refit(
                records=records,
                assets_dir=assets_dir,
                align_iou=align_iou,
                config=refit_cfg,
                person_masks_by_name=person_masks_by_name,
            )
            records = refit_result.records
            shared_beta_refit_summary = _shared_beta_summary(refit_result.meta)
            if refit_result.canonical_verts_cam is not None:
                canonical_verts_cam = np.asarray(refit_result.canonical_verts_cam, dtype=np.float32)
            if refit_result.canonical_joints_cam is not None:
                canonical_joints_cam = np.asarray(refit_result.canonical_joints_cam, dtype=np.float32)
            postprocess_npz_kwargs.update(_shared_beta_npz_kwargs(refit_result.meta))
            need_final_export = True
        except Exception as e:
            shared_beta_refit_summary = {"enabled": False, "error": str(e)}
            print(f"Warning: shared-beta refit failed; keeping current stage1 records. err={e}")
    else:
        shared_beta_refit_summary = {"enabled": False, "reason": "disabled_by_flag"}

    silhouette_refine_summary: dict[str, object] | None = None
    if not bool(args.disable_silhouette_refine):
        try:
            sil_cfg = SilhouetteRefineConfig(
                enabled=True,
                mask_source=str(args.silhouette_refine_mask_source),
                refit_iters=int(max(1, int(args.silhouette_refine_iters))),
                lr=float(args.silhouette_refine_lr),
                device=str(args.silhouette_refine_device),
                modnet_ckpt_path=str(args.silhouette_refine_modnet_ckpt),
            )
            sil_result = apply_silhouette_refine(
                records=records,
                assets_dir=assets_dir,
                config=sil_cfg,
                person_masks_by_name=person_masks_by_name,
                frames_dir=frames_dir,
            )
            records = sil_result.records
            silhouette_refine_summary = _silhouette_refine_summary(sil_result.meta)
            postprocess_npz_kwargs.update(_silhouette_refine_npz_kwargs(sil_result.meta))
            last_rec = records[-1]
            try:
                canonical_verts_cam, canonical_joints_cam = _extract_tpose_cam_from_record(
                    record=last_rec,
                    assets_dir=assets_dir,
                )
            except Exception as e:
                print(f"Warning: silhouette refine canonical extraction failed; fallback to frame-space canonical. err={e}")
                canonical_verts_cam = None
                canonical_joints_cam = None
            if canonical_verts_cam is None:
                canonical_verts_cam = np.asarray(last_rec["verts_cam"], dtype=np.float32)
            if canonical_joints_cam is None:
                jc_last = last_rec.get("joints_cam", None)
                if jc_last is not None:
                    canonical_joints_cam = np.asarray(jc_last, dtype=np.float32)
            need_final_export = True
        except Exception as e:
            silhouette_refine_summary = {"enabled": False, "error": str(e)}
            print(f"Warning: silhouette refine failed; keeping current stage1 records. err={e}")
    else:
        silhouette_refine_summary = {"enabled": False, "reason": "disabled_by_flag"}

    if need_final_export or len(postprocess_npz_kwargs) > 0:
        compat_assets = _write_stage1_sequence_assets_full(
            out_dir=out_dir,
            frames_dir=frames_dir,
            records=records,
            canonical_verts_cam=canonical_verts_cam,
            canonical_joints_cam=canonical_joints_cam,
            extra_params_kwargs=(postprocess_npz_kwargs or None),
        )
    first_rec = records[0]
    last_rec = records[-1]
    stage1_seq_obj = out_dir / "sequence" / f"frame_{int(last_rec['frame_idx']):06d}.obj"
    model_type_set = sorted({str(rec.get("model_type", "unknown")) for rec in records})
    fashn_face_semantics_summary: dict[str, object] | None = None
    if bool(args.build_fashn_face_semantics):
        if str(args.fashn_face_semantics_out_dir).strip() == "":
            fashn_out_dir = (out_dir / "fashn_face_semantics").resolve()
        else:
            fashn_out_dir = Path(str(args.fashn_face_semantics_out_dir).strip())
            if not fashn_out_dir.is_absolute():
                fashn_out_dir = project_root / fashn_out_dir
            fashn_out_dir = fashn_out_dir.resolve()
        try:
            fashn_face_semantics_summary = build_stage1_fashn_face_semantics(
                stage1_dir=out_dir,
                frames_dir=frames_dir,
                out_dir=fashn_out_dir,
                device=str(args.fashn_device),
                max_edge=int(args.fashn_max_edge),
                frame_stride=int(args.fashn_frame_stride),
                max_frames=int(args.fashn_max_frames),
                min_non_bg_votes=int(args.fashn_min_non_bg_votes),
            )
        except Exception as e:
            fashn_face_semantics_summary = {"enabled": False, "error": str(e)}
            print(f"Warning: FASHN face semantics export failed. err={e}")
    summary = {
        "frames_dir": str(frames_dir),
        "assets_dir": str(assets_dir),
        "smplestx_repo_dir": str(smplestx_dir),
        "smplestx_output_dir": str(smplestx_result_dir),
        "final_output_dir": str(final_out),
        "sequence_frame_count": int(len(records)),
        "sequence_first_frame_name": str(first_rec["frame_name"]),
        "sequence_last_frame_name": str(last_rec["frame_name"]),
        "sequence_first_frame_idx": int(first_rec["frame_idx"]),
        "sequence_last_frame_idx": int(last_rec["frame_idx"]),
        "model_type": ("smplx" if model_type_set == ["smplx"] else ",".join(model_type_set)),
        "smplestx_checkpoint": (None if smplestx_ckpt is None else str(smplestx_ckpt)),
        "pred_source_counts": pred_source_counts,
        "pred_non_exact_local_indices": non_exact_local_idx,
        "missing_requirements": {
            "smplestx": list(req.get("smplestx", [])),
            "python": list(req.get("python", [])),
        },
        "shared_beta_refit": shared_beta_refit_summary,
        "silhouette_refine": silhouette_refine_summary,
        "fashn_face_semantics": fashn_face_semantics_summary,
        "stage2_compat_assets": compat_assets,
        "stage1_sequence_obj": str(stage1_seq_obj),
    }
    with (final_out / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if bool(args.minimal_stage1_outputs):
        # Keep only canonical objs in stage1 root; delete bulky/intermediate outputs.
        for pp in (smplestx_out, final_out):
            _remove_path_tree(pp)


    print("Done.")
    print(f"Sequence frame count: {len(records)}")
    print(f"Sequence first frame: {first_rec['frame_name']} (idx={int(first_rec['frame_idx'])})")
    print(f"Sequence last frame: {last_rec['frame_name']} (idx={int(last_rec['frame_idx'])})")
    print(f"Sequence sample OBJ: {stage1_seq_obj}")
    print(f"Summary: {final_out / 'run_summary.json'}")
    if fashn_face_semantics_summary is not None:
        print(f"FASHN face semantics: {fashn_face_semantics_summary.get('out_npz', fashn_face_semantics_summary)}")


if __name__ == "__main__":
    main()


















