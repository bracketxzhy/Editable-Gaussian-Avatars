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
from tqdm import tqdm

from avatar_gs.alignment import (
    apply_similarity_to_camera_points,
    apply_uv_similarity,
    estimate_mask_similarity_transform,
    mask_iou,
    mask_outside_ratio,
)
from avatar_gs.io.mesh_export import build_obj_face_text, export_obj
from avatar_gs.segmentation import estimate_person_mask_modnet, resolve_modnet_ckpt_path

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


def _pick_best_existing_smplerx_ckpt(cands: list[Path]) -> Path | None:
    valid = [p for p in cands if p.exists() and p.is_file()]
    if len(valid) <= 0:
        return None
    valid.sort(key=lambda p: int(p.stat().st_size), reverse=True)
    return valid[0]


def _resolve_smplerx_ckpt(assets_dir: Path, smplerx_dir: Path, override: Path | None) -> str:
    if override is not None and override.exists():
        override_name = override.name.lower()
        if ("smpler" not in override_name) and ("smpler_x" not in override_name):
            raise FileNotFoundError(
                f"Provided --smplerx_ckpt is not a SMPLer-X checkpoint: {override}"
            )
        return str(override)

    repo_pretrained = smplerx_dir / "pretrained_models"
    named_repo = [
        repo_pretrained / "smpler_x_h32_correct.pth.tar",
        repo_pretrained / "smpler_x_h32.pth.tar",
        repo_pretrained / "smpler_x_l32.pth.tar",
        repo_pretrained / "smpler_x_b32.pth.tar",
        repo_pretrained / "smpler_x_s32.pth.tar",
    ]
    best_repo = _pick_best_existing_smplerx_ckpt(named_repo)
    if best_repo is not None:
        return str(best_repo)

    p = _find_any_file(
        assets_dir,
        [
            "smpler_x_h32_correct.pth.tar",
            "smpler_x_h32.pth.tar",
            "smpler_x_l32.pth.tar",
            "smpler_x_b32.pth.tar",
            "smpler_x_s32.pth.tar",
            "SMPLer-X*.pth.tar",
            "SMPLerX*.pth.tar",
            "*smpler*x*.pth.tar",
        ],
    )
    if p is None:
        raise FileNotFoundError(
            "Cannot find SMPLer-X checkpoint under third_party/SMPLer-X/pretrained_models or assets. "
            "Please pass --smplerx_ckpt explicitly."
        )
    return str(p)


def _prepare_smplerx_runtime_data(assets_dir: Path, smplerx_dir: Path, smplerx_ckpt: str) -> None:
    pretrained_dir = smplerx_dir / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    human_model_dir = smplerx_dir / "common" / "utils" / "human_model_files"
    (human_model_dir / "smplx").mkdir(parents=True, exist_ok=True)
    (human_model_dir / "smpl").mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(str(smplerx_ckpt))
    if ckpt_path.exists():
        _ensure_link_or_copy(ckpt_path, pretrained_dir / ckpt_path.name)

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

    cands: list[np.ndarray] = []
    try:
        cand_modnet = estimate_person_mask_modnet(
            image_bgr,
            hint_mask_u8=mesh_hint_u8,
            ckpt_path=modnet_ckpt_path,
            ref_size=int(ref_size),
            threshold=float(threshold),
        )
        if int((cand_modnet > 0).sum()) > 0:
            cands.append(cand_modnet)
    except Exception:
        pass

    cand_ck = _estimate_person_mask_chromakey(image_bgr)
    if int((cand_ck > 0).sum()) > 0:
        cands.append(cand_ck)

    if mesh_hint_u8 is not None and int((mesh_hint_u8 > 0).sum()) > 0:
        cands.append(mesh_hint_u8)

    if len(cands) <= 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    best = None
    best_score = -1e12
    for c in cands:
        s = _mask_score(c, mesh_hint_u8)
        if s > best_score:
            best_score = s
            best = c

    return (np.asarray(best, dtype=np.uint8) > 0).astype(np.uint8) * 255


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


def _load_smplerx_npz_params(npz_path: Path) -> dict[str, np.ndarray]:
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

    params = dict(record.get("smplerx_params", {}))
    cam_t = np.asarray(record.get("cam_t", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
    gender = str(record.get("gender", "neutral")).strip().lower()
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"
    betas_np = np.asarray(params.get("betas", np.zeros((10,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    transl_t = torch.from_numpy(cam_t.reshape(1, 3))

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


def _frame_index_from_name(frame_name: str, fallback: int) -> int:
    stem = Path(str(frame_name)).stem
    if stem.isdigit():
        return int(stem)
    return int(fallback)


def _build_smplerx_frame_records(
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
        raise FileNotFoundError(f"SMPLer-X result folder missing: {smplx_dir}")

    available: dict[int, tuple[Path, Path | None]] = {}
    for local_idx in range(len(selected_files)):
        stem = f"{local_idx + 1:05d}_0"
        npz_path = smplx_dir / f"{stem}.npz"
        meta_path = meta_dir / f"{stem}.json"
        if npz_path.exists():
            available[int(local_idx)] = (npz_path, meta_path if meta_path.exists() else None)
    if len(available) <= 0:
        raise RuntimeError(f"No SMPLer-X frame predictions found under: {smplx_dir}")

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
        params = _load_smplerx_npz_params(npz_path)
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
                "smplerx_params": params,
                "smplerx_input_file": (
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
        json.dump({"K": K_list, "source": "smplerx_meta_or_fallback"}, f, indent=2)

    frame_names: list[str] = []
    verts_cam_seq: list[np.ndarray] = []
    joints_cam_seq: list[np.ndarray] = []
    masks_seq: list[np.ndarray] = []
    kpt_map: dict[str, list[list[float]]] = {}
    align_scale_seq: list[float] = []
    align_tx_seq: list[float] = []
    align_ty_seq: list[float] = []
    align_iou_seq: list[float] = []
    align_outside_seq: list[float] = []
    align_accepted_seq: list[np.uint8] = []
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

    export_iter = tqdm(records, desc="Stage1 export sequence", leave=False)
    for rec in export_iter:
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
        mask = _estimate_person_mask_modnet_stage1(
            img,
            mesh_hint_mask=mask_hint_raw,
        )
        uv_hint, align_report = estimate_mask_similarity_transform(
            uv=uv_raw,
            valid_v=valid_hint,
            faces=faces,
            image_hw=(hh, ww),
            smpl_mask=mask_hint_raw,
            person_mask=mask,
        )
        if bool(align_report.get("accepted", False)):
            verts_cam = apply_similarity_to_camera_points(
                verts_cam,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                scale=float(align_report.get("scale", 1.0)),
                tx_px=float(align_report.get("tx_px", 0.0)),
                ty_px=float(align_report.get("ty_px", 0.0)),
            )
            if joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
                joints_cam = apply_similarity_to_camera_points(
                    np.asarray(joints_cam, dtype=np.float32),
                    fx=float(fx),
                    fy=float(fy),
                    cx=float(cx),
                    cy=float(cy),
                    scale=float(align_report.get("scale", 1.0)),
                    tx_px=float(align_report.get("tx_px", 0.0)),
                    ty_px=float(align_report.get("ty_px", 0.0)),
                )
            cam_t_aligned = apply_similarity_to_camera_points(
                np.asarray(rec["cam_t"], dtype=np.float32).reshape(1, 3),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                scale=float(align_report.get("scale", 1.0)),
                tx_px=float(align_report.get("tx_px", 0.0)),
                ty_px=float(align_report.get("ty_px", 0.0)),
            )[0]
            rec["cam_t"] = cam_t_aligned.astype(np.float32)
            rec["verts_local"] = np.asarray(verts_cam - cam_t_aligned.reshape(1, 3), dtype=np.float32)
            rec["verts_cam"] = np.asarray(verts_cam, dtype=np.float32)
            if joints_cam is not None:
                rec["joints_cam"] = np.asarray(joints_cam, dtype=np.float32)
            uv_hint = apply_uv_similarity(
                uv_raw,
                scale=float(align_report.get("scale", 1.0)),
                tx=float(align_report.get("tx_px", 0.0)),
                ty=float(align_report.get("ty_px", 0.0)),
            )
        else:
            rec["verts_cam"] = np.asarray(verts_cam, dtype=np.float32)

        mask_hint = _rasterize_mesh_mask(
            uv=uv_hint,
            valid_v=valid_hint,
            faces=faces,
            image_hw=(hh, ww),
        )

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
        if joints_cam is not None and np.asarray(joints_cam).ndim == 2 and np.asarray(joints_cam).shape[1] == 3:
            joints_cam_seq.append(np.asarray(joints_cam, dtype=np.float32))
        else:
            joints_cam_seq.append(np.zeros((0, 3), dtype=np.float32))
        masks_seq.append((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8))
        kpt_map[frame_name] = kpts
        align_accepted = bool(align_report.get("accepted", False))
        align_scale_seq.append(float(align_report.get("scale", 1.0)) if align_accepted else 1.0)
        align_tx_seq.append(float(align_report.get("tx_px", 0.0)) if align_accepted else 0.0)
        align_ty_seq.append(float(align_report.get("ty_px", 0.0)) if align_accepted else 0.0)
        align_iou_seq.append(
            float(align_report.get("iou_after", mask_iou(mask_hint, mask)))
            if align_accepted
            else float(align_report.get("iou_before", mask_iou(mask_hint_raw, mask)))
        )
        align_outside_seq.append(
            float(align_report.get("outside_after", mask_outside_ratio(mask_hint, mask)))
            if align_accepted
            else float(align_report.get("outside_before", mask_outside_ratio(mask_hint_raw, mask)))
        )
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
        "fit_strategy": np.asarray(["stage1_smplerx"], dtype=object),
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
    smplerx_params_seq: dict[str, list[np.ndarray]] = {k: [] for k in param_keys}
    smplerx_param_shape: dict[str, tuple[int, ...]] = {}
    for rec in records:
        params = rec.get("smplerx_params", None)
        if not isinstance(params, dict):
            continue
        for key in param_keys:
            if key not in params:
                continue
            arr = np.asarray(params[key], dtype=np.float32)
            smplerx_params_seq[key].append(arr)
            smplerx_param_shape.setdefault(key, tuple(arr.shape))
    for key, seq in smplerx_params_seq.items():
        if len(seq) != len(records):
            continue
        ref_shape = smplerx_param_shape.get(key, ())
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


def _check_stage1_requirements(assets_dir: Path, smplerx_dir: Path) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"smplerx": [], "python": []}

    if not smplerx_dir.exists():
        missing["smplerx"].append("third_party/SMPLer-X repo")

    repo_has_ckpt = False
    repo_pretrained = smplerx_dir / "pretrained_models"
    if repo_pretrained.exists():
        repo_has_ckpt = _find_any_file(
            repo_pretrained,
            [
                "smpler_x_*.pth.tar",
                "SMPLer-X*.pth.tar",
                "SMPLerX*.pth.tar",
                "*smpler*x*.pth.tar",
            ],
        ) is not None

    assets_has_ckpt = _find_any_file(
        assets_dir,
        [
            "smpler_x_*.pth.tar",
            "SMPLer-X*.pth.tar",
            "SMPLerX*.pth.tar",
            "*smpler*x*.pth.tar",
        ],
    ) is not None

    if not assets_has_ckpt and not repo_has_ckpt:
        missing["smplerx"].append("SMPLer-X pretrained checkpoint")

    try:
        resolve_modnet_ckpt_path(None)
    except Exception:
        missing["smplerx"].append("MODNet checkpoint")

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
    parser = argparse.ArgumentParser("Stage1: SMPLer-X refine")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input", help="Video frame directory")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Assets root directory")
    parser.add_argument("--out_dir", type=str, default="outputs/stage1", help="All outputs are written here")
    parser.add_argument("--smplerx_frame_indices", type=str, default="-1", help="Frame indices passed to SMPLerX refine")
    parser.add_argument("--smplerx_stride", type=int, default=1, help="Sample every N frames for SMPLerX refine when smplerx_frame_indices is all/-1")
    parser.add_argument("--smplerx_ckpt", type=str, default="", help="Optional explicit SMPLerX checkpoint path")
    parser.add_argument("--single_person", action="store_true", help="Set SMPLerX single-person mode")
    parser.add_argument("--skip_smplerx", action="store_true", help="Skip SMPLerX stage")
    parser.add_argument("--smplerx_repo_dir", type=str, default="third_party/SMPLer-X", help="SMPLer-X repo dir")
    parser.add_argument("--minimal_stage1_outputs", action="store_true", help="Keep only sequence assets required by stage2; delete intermediates")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    smplerx_dir = (project_root / args.smplerx_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets dir not found: {assets_dir}")
    if not smplerx_dir.exists():
        raise FileNotFoundError(f"SMPLerX repo not found: {smplerx_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    smplerx_out = out_dir / "smplerx"
    final_out = out_dir / "final"
    smplerx_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(frames_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    print(f"Frames: {len(frames)}")

    smplerx_indices_raw = str(args.smplerx_frame_indices).strip().lower()
    smplerx_stride = max(1, int(args.smplerx_stride))
    if smplerx_indices_raw in ("", "-1", "all", "*"):
        if smplerx_stride == 1:
            smplerx_indices_arg = "-1"
            smplerx_selected_count = len(frames)
        else:
            smplerx_idx = list(range(0, len(frames), smplerx_stride))
            smplerx_indices_arg = " ".join(str(i) for i in smplerx_idx)
            smplerx_selected_count = len(smplerx_idx)
    else:
        smplerx_indices_arg = str(args.smplerx_frame_indices).strip()
        smplerx_selected_count = -1

    if smplerx_selected_count > 0:
        print(
            f"SMPLerX frame selection: {smplerx_selected_count}/{len(frames)} frames "
            f"(stride={smplerx_stride}, indices='{smplerx_indices_arg[:64]}{'...' if len(smplerx_indices_arg) > 64 else ''}')"
        )

    req = _check_stage1_requirements(assets_dir=assets_dir, smplerx_dir=smplerx_dir)
    if len(req["smplerx"]) > 0 or len(req["python"]) > 0:
        parts: list[str] = []
        if len(req["smplerx"]) > 0:
            parts.append("missing_smplerx=" + ",".join(req["smplerx"]))
        if len(req["python"]) > 0:
            parts.append("missing_python=" + ",".join(req["python"]))
        raise RuntimeError("Stage1 SMPLer-X requirements not satisfied: " + "; ".join(parts))

    smplerx_ckpt: str | None = None
    if not bool(args.skip_smplerx):
        smplerx_ckpt = _resolve_smplerx_ckpt(
            assets_dir=assets_dir,
            smplerx_dir=smplerx_dir,
            override=(Path(args.smplerx_ckpt).resolve() if str(args.smplerx_ckpt).strip() else None),
        )
        _prepare_smplerx_runtime_data(assets_dir=assets_dir, smplerx_dir=smplerx_dir, smplerx_ckpt=smplerx_ckpt)

    env_base = os.environ.copy()

    if not bool(args.skip_smplerx):
        smplerx_cmd = [
            sys.executable,
            "-m",
            "avatar_gs.cli._run_smplerx_refine_frames",
            "--frames_dir",
            str(frames_dir),
            "--output_folder",
            str(smplerx_out),
            "--smplerx_repo_dir",
            str(smplerx_dir),
            "--pretrained_model",
            str(smplerx_ckpt),
            "--frame_indices",
            str(smplerx_indices_arg),
        ]
        if bool(args.single_person):
            smplerx_cmd.append("--single_person")
        _run_cmd(smplerx_cmd, cwd=project_root, env=env_base)
    else:
        print("Skipped SMPLerX stage.")
        return

    smplerx_result_dir = smplerx_out / "selected_frames"
    meta_json = smplerx_result_dir / "selected_frames_meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"SMPLerX selected-frame metadata missing: {meta_json}")
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    faces_smplx = None
    smplx_face_path = _find_any_file(assets_dir, ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"])
    if smplx_face_path is not None:
        faces_smplx = _load_faces_from_npz(smplx_face_path)

    records = _build_smplerx_frame_records(
        selected_result_dir=smplerx_result_dir,
        selected_meta=selected_meta,
        faces_smplx=faces_smplx,
        assets_dir=assets_dir,
    )
    if len(records) <= 0:
        raise RuntimeError("No frame records built from SMPLerX output.")

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
    stage1_seq_obj = out_dir / "sequence" / f"frame_{int(last_rec['frame_idx']):06d}.obj"
    model_type_set = sorted({str(rec.get("model_type", "unknown")) for rec in records})
    summary = {
        "frames_dir": str(frames_dir),
        "assets_dir": str(assets_dir),
        "smplerx_repo_dir": str(smplerx_dir),
        "smplerx_output_dir": str(smplerx_result_dir),
        "final_output_dir": str(final_out),
        "sequence_frame_count": int(len(records)),
        "sequence_first_frame_name": str(first_rec["frame_name"]),
        "sequence_last_frame_name": str(last_rec["frame_name"]),
        "sequence_first_frame_idx": int(first_rec["frame_idx"]),
        "sequence_last_frame_idx": int(last_rec["frame_idx"]),
        "model_type": ("smplx" if model_type_set == ["smplx"] else ",".join(model_type_set)),
        "smplerx_checkpoint": (None if smplerx_ckpt is None else str(smplerx_ckpt)),
        "pred_source_counts": pred_source_counts,
        "pred_non_exact_local_indices": non_exact_local_idx,
        "missing_requirements": req,
        "stage2_compat_assets": compat_assets,
        "stage1_sequence_obj": str(stage1_seq_obj),
    }
    with (final_out / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if bool(args.minimal_stage1_outputs):
        # Keep only canonical objs in stage1 root; delete bulky/intermediate outputs.
        for pp in (smplerx_out, final_out):
            _remove_path_tree(pp)


    print("Done.")
    print(f"Sequence frame count: {len(records)}")
    print(f"Sequence first frame: {first_rec['frame_name']} (idx={int(first_rec['frame_idx'])})")
    print(f"Sequence last frame: {last_rec['frame_name']} (idx={int(last_rec['frame_idx'])})")
    print(f"Sequence sample OBJ: {stage1_seq_obj}")
    print(f"Summary: {final_out / 'run_summary.json'}")


if __name__ == "__main__":
    main()

















