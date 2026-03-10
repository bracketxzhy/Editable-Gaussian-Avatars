from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avatar_gs.alignment import estimate_mask_similarity_transform
from avatar_gs.segmentation import estimate_person_mask_modnet
from avatar_gs.training.inner_layer_training import resolve_inner_supervision_mask
from avatar_gs.utils.io import ensure_dir


def project_points_numpy(
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


def project_points_torch(
    xyz: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xc = (xyz @ R.transpose(0, 1)) + t.view(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = torch.zeros_like(z)
    v = torch.zeros_like(z)
    if bool(valid.any()):
        xv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (xv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (xv[:, 1] / zv) + K[1, 2]
    return u, v, valid


def rasterize_mesh_mask(
    verts: np.ndarray,
    faces: np.ndarray | None,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if faces is None or len(faces) == 0 or verts.shape[0] == 0:
        return mask
    u, v, valid = project_points_numpy(verts, K, R, t)
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
    if int(dilate_px) > 0:
        k = np.ones((2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    if int(erode_px) > 0:
        k = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), dtype=np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    return mask


def _largest_component_by_overlap_np(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None, min_area: int = 128) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)
    best_i = -1
    best_score = -1.0
    ref = None if ref_mask_u8 is None else (ref_mask_u8 > 0)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(max(1, min_area)):
            continue
        comp = labels == i
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


def mask_border_touch_ratio_np(mask_u8: np.ndarray) -> float:
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


def clean_person_mask_eval_style_np(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None = None) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return _largest_component_by_overlap_np(m, ref_mask_u8, min_area=128)


def _modnet_person_mask_eval_style_np(
    image_bgr: np.ndarray,
    *,
    hint_mask_u8: np.ndarray | None = None,
    modnet_ref_size: int = 512,
    modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
) -> np.ndarray:
    fg = estimate_person_mask_modnet(
        image_bgr,
        hint_mask_u8=hint_mask_u8,
        ref_size=int(modnet_ref_size),
        threshold=float(modnet_threshold),
        ckpt_path=modnet_ckpt_path,
    )
    return clean_person_mask_eval_style_np(fg, ref_mask_u8=hint_mask_u8)


def estimate_gt_person_contour_mask_np(
    rgb_chw: torch.Tensor,
    *,
    hint_mask_u8: np.ndarray | None = None,
    modnet_ref_size: int = 512,
    modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
) -> np.ndarray:
    rgb_np = rgb_chw.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
    bgr_u8 = (rgb_np[:, :, ::-1] * 255.0).astype(np.uint8)
    return (
        _modnet_person_mask_eval_style_np(
            bgr_u8,
            hint_mask_u8=hint_mask_u8,
            modnet_ref_size=int(max(32, modnet_ref_size)),
            modnet_threshold=float(np.clip(float(modnet_threshold), 0.0, 1.0)),
            modnet_ckpt_path=modnet_ckpt_path,
        )
        > 0
    ).astype(np.uint8)


def modnet_person_mask_eval_style_np(
    image_bgr: np.ndarray,
    *,
    hint_mask_u8: np.ndarray | None = None,
    modnet_ref_size: int = 512,
    modnet_threshold: float = 0.5,
    modnet_ckpt_path: str | None = None,
) -> np.ndarray:
    return _modnet_person_mask_eval_style_np(
        image_bgr,
        hint_mask_u8=hint_mask_u8,
        modnet_ref_size=modnet_ref_size,
        modnet_threshold=modnet_threshold,
        modnet_ckpt_path=modnet_ckpt_path,
    )


def _erode_binary_mask_torch(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    if int(erode_px) <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    k = int(erode_px) * 2 + 1
    inv = (1.0 - mask).clamp(0.0, 1.0).unsqueeze(0)
    dil_inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=int(erode_px))
    out = 1.0 - dil_inv.squeeze(0)
    return out.clamp(0.0, 1.0)


def _dilate_binary_mask_torch(mask: torch.Tensor, dilate_px: int) -> torch.Tensor:
    if int(dilate_px) <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    k = int(dilate_px) * 2 + 1
    x = mask.clamp(0.0, 1.0).unsqueeze(0)
    out = F.max_pool2d(x, kernel_size=k, stride=1, padding=int(dilate_px)).squeeze(0)
    return out.clamp(0.0, 1.0)


def compute_contour_band_torch(mask: torch.Tensor, width_px: int = 2) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    w = max(int(width_px), 1)
    m = (mask > 0.5).float()
    dil = _dilate_binary_mask_torch(m, w)
    ero = _erode_binary_mask_torch(m, w)
    return (dil - ero).clamp(0.0, 1.0)


def fuse_outer_contour_mask_np(
    gt_mask_np: np.ndarray,
    person_mask_np: np.ndarray | None,
    *,
    fuse_mode: str,
) -> np.ndarray | None:
    fuse_mode_norm = str(fuse_mode).strip().lower()
    if fuse_mode_norm not in ("intersect", "union", "replace", "gt_only"):
        raise ValueError(f"Unsupported outer contour fuse mode: {fuse_mode}")
    gt_mask = (np.asarray(gt_mask_np) > 0).astype(np.uint8)
    person_mask = None if person_mask_np is None else (np.asarray(person_mask_np) > 0).astype(np.uint8)
    if person_mask is not None:
        if fuse_mode_norm == "replace":
            gt_mask = person_mask
        elif fuse_mode_norm == "union":
            gt_mask = np.maximum(gt_mask, person_mask)
        elif fuse_mode_norm == "intersect":
            inter = (gt_mask * person_mask).astype(np.uint8)
            gt_area = int(np.sum(gt_mask > 0))
            inter_area = int(np.sum(inter > 0))
            keep_ratio = (float(inter_area) / float(max(gt_area, 1))) if gt_area > 0 else 0.0
            if keep_ratio >= 0.995:
                gt_mask = inter
    if int(np.sum(gt_mask > 0)) > 0:
        return gt_mask.astype(np.uint8)
    if person_mask is not None and int(np.sum(person_mask > 0)) > 0:
        return person_mask.astype(np.uint8)
    return None


def _masked_rgb_u8(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = np.full_like(rgb_u8, 255, dtype=np.uint8)
    keep = mask_u8 > 0
    out[keep] = rgb_u8[keep]
    return out


def _read_stage1_frame_camera_K(
    *,
    stage1_dir: Path,
    frame_index: int,
    frame_name: str,
    fallback_K: np.ndarray,
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
    meta_dir = stage1_dir / "smplerx" / "selected_frames" / "meta"
    meta_path = meta_dir / f"{int(frame_index) + 1:05d}_0.json"
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
    return np.asarray(fallback_K, dtype=np.float32).reshape(3, 3)


def _rasterize_uv_mask(uv: np.ndarray, valid_v: np.ndarray, faces: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
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
        cv2.fillConvexPoly(mask, pts, color=1, lineType=cv2.LINE_8)
    return mask


@dataclass(slots=True)
class Stage2FrameAssets:
    gts: list[torch.Tensor]
    prepared_supervision: dict | None
    frame_align_scores: list[float] | None = None


def build_stage2_frame_assets(
    *,
    frames_dir: Path,
    frames_list: list[str],
    device: str | torch.device,
    prepared_supervision: dict | None = None,
    frame_align_scores: list[float] | None = None,
) -> Stage2FrameAssets:
    gts: list[torch.Tensor] = []
    for fn in frames_list:
        bgr = cv2.imread(str(frames_dir / fn), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read selected frame: {frames_dir / fn}")
        rgb = torch.from_numpy((bgr[:, :, ::-1].astype(np.float32) / 255.0)).permute(2, 0, 1).to(device=device)
        gts.append(rgb)
    return Stage2FrameAssets(
        gts=gts,
        prepared_supervision=prepared_supervision,
        frame_align_scores=frame_align_scores,
    )


def load_stage2_supervision_cache(*, cache_path: Path, frames: list[str], logger) -> dict | None:
    if not cache_path.exists():
        logger.info(f"Stage2 supervision cache not found, fallback to online supervision: {cache_path}")
        return None
    z = np.load(str(cache_path), allow_pickle=True)
    required = (
        "frame_names",
        "inner_masks",
        "outer_masks",
        "smpl_masks",
        "gt_train_contour_masks",
        "inner_rgb_targets",
        "outer_rgb_targets",
        "combined_rgb_targets",
    )
    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f"Bad stage2 supervision cache, missing keys {missing}: {cache_path}")
    cache_frames = [str(x) for x in np.asarray(z["frame_names"]).reshape(-1).tolist()]
    if cache_frames != [str(x) for x in frames]:
        raise RuntimeError(
            "Stage2 supervision cache frame list does not match current selected frames. "
            "Regenerate supervision cache before training."
        )
    cache = {
        "frame_names": cache_frames,
        "inner_masks": np.asarray(z["inner_masks"], dtype=np.uint8),
        "outer_masks": np.asarray(z["outer_masks"], dtype=np.uint8),
        "smpl_masks": np.asarray(z["smpl_masks"], dtype=np.uint8),
        "gt_train_contour_masks": np.asarray(z["gt_train_contour_masks"], dtype=np.uint8),
        "inner_rgb_targets": np.asarray(z["inner_rgb_targets"], dtype=np.uint8),
        "outer_rgb_targets": np.asarray(z["outer_rgb_targets"], dtype=np.uint8),
        "combined_rgb_targets": np.asarray(z["combined_rgb_targets"], dtype=np.uint8),
    }
    logger.info(f"Using prepared stage2 supervision cache: {cache_path}")
    return cache




@dataclass(slots=True)
class Stage2TrainingSupervisionState:
    person_masks_frames: list[torch.Tensor | None] | None
    prepared_inner_masks_frames: list[torch.Tensor | None] | None
    prepared_outer_masks_frames: list[torch.Tensor | None] | None
    prepared_smpl_masks_frames: list[torch.Tensor | None] | None
    prepared_gt_train_masks_frames: list[torch.Tensor | None] | None
    prepared_inner_rgb_targets_frames: list[torch.Tensor] | None
    prepared_outer_rgb_targets_frames: list[torch.Tensor] | None
    prepared_combined_rgb_targets_frames: list[torch.Tensor] | None
    outer_gt_masks_frames: list[torch.Tensor | None] | None
    person_contour_band_frames: list[torch.Tensor | None] | None
    outer_head_eval_masks_frames: list[torch.Tensor | None] | None
    frame_weights_t: torch.Tensor | None
    frame_loss_weights_t: torch.Tensor | None
    frame_align_scores_t: torch.Tensor | None
    person_fuse: str
    outer_contour_fuse: str


def build_stage2_training_supervision_state(
    *,
    frames_dir: Path,
    frames_list: list[str],
    gts: list[torch.Tensor],
    device: str | torch.device,
    image_hw: tuple[int, int],
    prepared_supervision: dict | None,
    use_person_mask: bool,
    person_masks_npz: str | Path | None,
    person_mask_dilate_px: int,
    person_mask_erode_px: int,
    person_mask_eval_clean: bool,
    person_mask_eval_refine_modnet: bool,
    person_mask_fuse_mode: str,
    person_mask_contour_width_px: int,
    person_mask_contour_leak_weight: float,
    outer_use_gt_contour: bool,
    outer_gt_contour_fuse_mode: str,
    outer_gt_contour_modnet_ref_size: int,
    outer_gt_contour_modnet_threshold: float,
    modnet_ckpt_path: str | None,
    separate_fg_bg_training: bool,
    fg_region_weight: float,
    bg_region_weight: float,
    face_boxes: list[tuple[float, float, float, float] | None] | None,
    outer_head_priority_face_pad: float,
    outer_head_priority_up_scale: float,
    outer_head_priority_down_scale: float,
    outer_head_priority_person_top_ratio: float,
    frame_weights: list[float] | None,
    frame_loss_weights: list[float] | None,
    frame_align_scores: list[float] | None,
    out_ckpt_path: Path,
    logger,
) -> Stage2TrainingSupervisionState:
    nF = len(frames_list)
    H, W = int(image_hw[0]), int(image_hw[1])
    person_masks_frames: list[torch.Tensor | None] | None = None
    prepared_inner_masks_frames: list[torch.Tensor | None] | None = None
    prepared_outer_masks_frames: list[torch.Tensor | None] | None = None
    prepared_smpl_masks_frames: list[torch.Tensor | None] | None = None
    prepared_gt_train_masks_frames: list[torch.Tensor | None] | None = None
    person_fuse = str(person_mask_fuse_mode).strip().lower()
    if person_fuse not in ("intersect", "union", "replace"):
        raise ValueError(
            f"person_mask_fuse_mode must be intersect|union|replace. Got: {person_mask_fuse_mode}"
        )
    outer_contour_fuse = str(outer_gt_contour_fuse_mode).strip().lower()
    if outer_contour_fuse not in ("intersect", "union", "replace", "gt_only"):
        raise ValueError(
            "outer_gt_contour_fuse_mode must be intersect|union|replace|gt_only. "
            f"Got: {outer_gt_contour_fuse_mode}"
        )
    sep_fg_bg = bool(separate_fg_bg_training)
    fg_w = float(max(fg_region_weight, 0.0))
    bg_w = float(max(bg_region_weight, 0.0))
    if sep_fg_bg and fg_w <= 0.0 and bg_w <= 0.0:
        raise ValueError(
            "separate_fg_bg_training=true requires at least one positive "
            "weight in fg_region_weight/bg_region_weight."
        )

    if bool(use_person_mask):
        if person_masks_npz is None:
            raise ValueError("use_person_mask=true but person_masks_npz is not set.")
        p_path = Path(person_masks_npz)
        if not p_path.exists():
            raise FileNotFoundError(f"person_masks_npz not found: {p_path}")
        pdata = np.load(p_path, allow_pickle=True)
        if "frame_names" not in pdata.files or "masks" not in pdata.files:
            raise RuntimeError(f"{p_path} must contain frame_names and masks.")
        p_names = [str(x) for x in pdata["frame_names"].tolist()]
        p_masks = np.asarray(pdata["masks"])
        if p_masks.ndim != 3:
            raise RuntimeError(f"Bad masks shape in {p_path}: {p_masks.shape}")
        name_to_idx = {n: i for i, n in enumerate(p_names)}
        person_masks_frames = [None] * nF
        pd = max(int(person_mask_dilate_px), 0)
        pe = max(int(person_mask_erode_px), 0)
        k_d = np.ones((2 * pd + 1, 2 * pd + 1), dtype=np.uint8) if pd > 0 else None
        k_e = np.ones((2 * pe + 1, 2 * pe + 1), dtype=np.uint8) if pe > 0 else None
        for i, fn in enumerate(frames_list):
            if fn not in name_to_idx:
                continue
            m = p_masks[int(name_to_idx[fn])].astype(np.float32)
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.uint8)
            if k_d is not None:
                m = cv2.dilate(m, k_d, iterations=1)
            if k_e is not None:
                m = cv2.erode(m, k_e, iterations=1)
            if bool(person_mask_eval_clean):
                m_u8 = clean_person_mask_eval_style_np((m > 0).astype(np.uint8) * 255, ref_mask_u8=None)
                if bool(person_mask_eval_refine_modnet):
                    area_ratio = float((m_u8 > 0).mean())
                    border_ratio = float(mask_border_touch_ratio_np(m_u8))
                    if area_ratio < 0.03 or area_ratio > 0.75 or border_ratio > 0.08:
                        rgb_np = gts[i].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
                        bgr_u8 = (rgb_np[:, :, ::-1] * 255.0).astype(np.uint8)
                        m_u8 = modnet_person_mask_eval_style_np(
                            bgr_u8,
                            hint_mask_u8=m_u8,
                            modnet_ref_size=int(max(32, outer_gt_contour_modnet_ref_size)),
                            modnet_threshold=float(np.clip(float(outer_gt_contour_modnet_threshold), 0.0, 1.0)),
                            modnet_ckpt_path=modnet_ckpt_path,
                        )
                m = (m_u8 > 0).astype(np.uint8)
            person_masks_frames[i] = torch.from_numpy(m.astype(np.float32)).to(device=device).unsqueeze(0)

    prepared_inner_rgb_targets_frames: list[torch.Tensor] | None = None
    prepared_outer_rgb_targets_frames: list[torch.Tensor] | None = None
    prepared_combined_rgb_targets_frames: list[torch.Tensor] | None = None
    if prepared_supervision is not None:
        cache_frames = [str(x) for x in prepared_supervision.get("frame_names", [])]
        if cache_frames != [str(x) for x in frames_list]:
            raise RuntimeError(
                "prepared_supervision frame_names do not match current frames_list. "
                "Regenerate stage2 supervision cache before training."
            )

        def _prepared_mask_list(key: str) -> list[torch.Tensor | None]:
            arr = np.asarray(prepared_supervision[key], dtype=np.uint8)
            if arr.ndim != 3 or int(arr.shape[0]) != nF or int(arr.shape[1]) != H or int(arr.shape[2]) != W:
                raise RuntimeError(
                    f"prepared_supervision[{key}] has bad shape {arr.shape}, expected ({nF}, {H}, {W})"
                )
            out: list[torch.Tensor | None] = [None] * nF
            for ii in range(nF):
                m_np = (arr[ii] > 0).astype(np.uint8)
                if int(m_np.sum()) > 0:
                    out[ii] = torch.from_numpy(m_np.astype(np.float32)).to(device=device).unsqueeze(0)
            return out

        def _prepared_rgb_list(key: str) -> list[torch.Tensor]:
            arr = np.asarray(prepared_supervision[key], dtype=np.uint8)
            if arr.ndim != 4 or int(arr.shape[0]) != nF or int(arr.shape[1]) != H or int(arr.shape[2]) != W or int(arr.shape[3]) != 3:
                raise RuntimeError(
                    f"prepared_supervision[{key}] has bad shape {arr.shape}, expected ({nF}, {H}, {W}, 3)"
                )
            out: list[torch.Tensor] = []
            for ii in range(nF):
                rgb_t = (
                    torch.from_numpy(arr[ii].astype(np.float32) / 255.0)
                    .to(device=device)
                    .permute(2, 0, 1)
                    .contiguous()
                )
                out.append(rgb_t)
            return out

        prepared_inner_masks_frames = _prepared_mask_list("inner_masks")
        prepared_outer_masks_frames = _prepared_mask_list("outer_masks")
        prepared_smpl_masks_frames = _prepared_mask_list("smpl_masks")
        prepared_gt_train_masks_frames = _prepared_mask_list("gt_train_contour_masks")
        prepared_inner_rgb_targets_frames = _prepared_rgb_list("inner_rgb_targets")
        prepared_outer_rgb_targets_frames = _prepared_rgb_list("outer_rgb_targets")
        prepared_combined_rgb_targets_frames = _prepared_rgb_list("combined_rgb_targets")
        logger.info("stage2 supervision: using prepared supervision cache for smpl/inner/outer masks and rgb targets")

    outer_gt_masks_frames: list[torch.Tensor | None] | None = None
    if bool(outer_use_gt_contour):
        if prepared_gt_train_masks_frames is not None:
            outer_gt_masks_frames = prepared_gt_train_masks_frames
            logger.info("stage2 precompute: GT contours cache hit: prepared supervision cache")
        else:
            outer_gt_masks_frames = [None] * nF
            cache_sig = {
                "ver": 1,
                "frames_dir": str(frames_dir.resolve()),
                "frames_list": [str(x) for x in frames_list],
                "H": int(H),
                "W": int(W),
                "outer_gt_contour_fuse_mode": str(outer_contour_fuse),
                "outer_gt_contour_modnet_ref_size": int(outer_gt_contour_modnet_ref_size),
                "outer_gt_contour_modnet_threshold": float(outer_gt_contour_modnet_threshold),
                "use_person_mask": bool(use_person_mask),
                "person_masks_npz": None if person_masks_npz is None else str(Path(person_masks_npz)),
                "person_mask_dilate_px": int(person_mask_dilate_px),
                "person_mask_erode_px": int(person_mask_erode_px),
                "person_mask_eval_clean": bool(person_mask_eval_clean),
                "person_mask_eval_refine_modnet": bool(person_mask_eval_refine_modnet),
                "modnet_ckpt_path": None if modnet_ckpt_path is None else str(modnet_ckpt_path),
            }
            cache_key = hashlib.sha1(
                json.dumps(cache_sig, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            cache_dir = out_ckpt_path.parent
            cache_path = cache_dir / f"outer_gt_contours_cache_{cache_key}.npz"
            cache_loaded = False
            if cache_path.exists():
                try:
                    c = np.load(cache_path, allow_pickle=True)
                    c_names = [str(x) for x in c["frame_names"].tolist()]
                    c_masks = np.asarray(c["masks"], dtype=np.uint8)
                    if (
                        len(c_names) == nF
                        and c_masks.ndim == 3
                        and c_masks.shape[0] == nF
                        and c_masks.shape[1] == H
                        and c_masks.shape[2] == W
                        and all(a == b for a, b in zip(c_names, frames_list))
                    ):
                        for i in range(nF):
                            m = (c_masks[i] > 0).astype(np.uint8)
                            if int(m.sum()) > 0:
                                outer_gt_masks_frames[i] = (
                                    torch.from_numpy(m.astype(np.float32)).to(device=device).unsqueeze(0)
                                )
                        cache_loaded = True
                        logger.info(f"stage2 precompute: GT contours cache hit: {cache_path}")
                except Exception as e:
                    logger.warning(f"Warning: failed to load GT contour cache ({cache_path}): {e}")
            if not cache_loaded:
                cache_masks = np.zeros((nF, H, W), dtype=np.uint8)
                for i in range(nF):
                    gt_mask_np = estimate_gt_person_contour_mask_np(
                        gts[i],
                        modnet_ref_size=int(max(32, outer_gt_contour_modnet_ref_size)),
                        modnet_threshold=float(np.clip(float(outer_gt_contour_modnet_threshold), 0.0, 1.0)),
                        modnet_ckpt_path=modnet_ckpt_path,
                    )
                    pm_np = None
                    if person_masks_frames is not None and person_masks_frames[i] is not None:
                        pm_np = (person_masks_frames[i][0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    final_mask_np = fuse_outer_contour_mask_np(
                        gt_mask_np,
                        pm_np,
                        fuse_mode=outer_contour_fuse,
                    )
                    if final_mask_np is not None:
                        outer_gt_masks_frames[i] = (
                            torch.from_numpy(final_mask_np.astype(np.float32)).to(device=device).unsqueeze(0)
                        )
                        cache_masks[i] = final_mask_np
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_path,
                        frame_names=np.asarray(frames_list),
                        masks=cache_masks,
                    )
                    logger.info(f"stage2 precompute: GT contours cache saved: {cache_path}")
                except Exception as e:
                    logger.warning(f"Warning: failed to save GT contour cache ({cache_path}): {e}")

    person_contour_band_frames: list[torch.Tensor | None] | None = None
    if person_masks_frames is not None and float(person_mask_contour_leak_weight) > 0.0:
        person_contour_band_frames = [None] * nF
        for i in range(nF):
            pm = person_masks_frames[i]
            if pm is None:
                continue
            person_contour_band_frames[i] = compute_contour_band_torch(
                pm,
                width_px=max(1, int(person_mask_contour_width_px)),
            )

    outer_head_eval_masks_frames: list[torch.Tensor | None] | None = None
    if face_boxes is not None or person_masks_frames is not None:
        outer_head_eval_masks_frames = [None] * nF
        face_pad_h = float(max(outer_head_priority_face_pad, 0.0))
        up_scale_h = float(max(outer_head_priority_up_scale, 0.0))
        down_scale_h = float(max(outer_head_priority_down_scale, 0.0))
        top_ratio_h = float(np.clip(float(outer_head_priority_person_top_ratio), 0.02, 0.40))
        for i in range(nF):
            m_np = np.zeros((H, W), dtype=np.uint8)
            pm_np = None
            if person_masks_frames is not None and person_masks_frames[i] is not None:
                pm_np = (person_masks_frames[i][0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            if face_boxes is not None and face_boxes[i] is not None:
                fx0, fy0, fx1, fy1 = face_boxes[i]
                fw = max(float(fx1 - fx0), 1.0)
                fh = max(float(fy1 - fy0), 1.0)
                x0 = max(0, int(np.floor(float(fx0) - face_pad_h * fw)))
                x1 = min(W - 1, int(np.ceil(float(fx1) + face_pad_h * fw)))
                y0 = max(0, int(np.floor(float(fy0) - (up_scale_h + face_pad_h) * fh)))
                y1 = min(H - 1, int(np.ceil(float(fy1) + (down_scale_h + 0.25 * face_pad_h) * fh)))
                if x1 >= x0 and y1 >= y0:
                    m_np[y0:y1 + 1, x0:x1 + 1] = 1
            if pm_np is not None and int(pm_np.sum()) > 0:
                ys, xs = np.where(pm_np > 0)
                if ys.size > 0 and xs.size > 0:
                    py0, py1 = int(ys.min()), int(ys.max())
                    px0, px1 = int(xs.min()), int(xs.max())
                    ph = max(py1 - py0 + 1, 1)
                    pw = max(px1 - px0 + 1, 1)
                    top_h = max(2, int(round(top_ratio_h * float(ph))))
                    x_pad = max(2, int(round(0.10 * float(pw))))
                    hx0 = max(0, px0 - x_pad)
                    hx1 = min(W - 1, px1 + x_pad)
                    hy0 = max(0, py0 - top_h)
                    hy1 = min(H - 1, py0 + top_h)
                    m_np[hy0:hy1 + 1, hx0:hx1 + 1] = 1
                    support = cv2.dilate(pm_np, np.ones((11, 11), dtype=np.uint8), iterations=1)
                    head_band = np.zeros_like(pm_np, dtype=np.uint8)
                    head_band[hy0:hy1 + 1, hx0:hx1 + 1] = 1
                    support = np.maximum(support, head_band)
                    m_np = (m_np * support).astype(np.uint8)
            if int(m_np.sum()) > 0:
                outer_head_eval_masks_frames[i] = (
                    torch.from_numpy(m_np.astype(np.float32)).to(device=device).unsqueeze(0)
                )

    frame_weights_t = None
    frame_loss_weights_t = None
    frame_align_scores_t = None
    if frame_weights is not None:
        if len(frame_weights) != nF:
            raise ValueError("frame_weights length must match number of frames.")
        w = torch.tensor(frame_weights, dtype=torch.float32)
        if (w <= 0).any():
            raise ValueError("frame_weights must be > 0.")
        frame_weights_t = w / w.sum()
    if frame_loss_weights is not None:
        if len(frame_loss_weights) != nF:
            raise ValueError("frame_loss_weights length must match number of frames.")
        frame_loss_weights_t = torch.tensor(frame_loss_weights, dtype=torch.float32)
    if frame_align_scores is not None:
        if len(frame_align_scores) != nF:
            raise ValueError("frame_align_scores length must match number of frames.")
        frame_align_scores_t = torch.tensor(frame_align_scores, dtype=torch.float32)
        frame_align_scores_t = torch.where(
            torch.isfinite(frame_align_scores_t),
            frame_align_scores_t,
            torch.ones_like(frame_align_scores_t),
        ).clamp_min(0.0)

    return Stage2TrainingSupervisionState(
        person_masks_frames=person_masks_frames,
        prepared_inner_masks_frames=prepared_inner_masks_frames,
        prepared_outer_masks_frames=prepared_outer_masks_frames,
        prepared_smpl_masks_frames=prepared_smpl_masks_frames,
        prepared_gt_train_masks_frames=prepared_gt_train_masks_frames,
        prepared_inner_rgb_targets_frames=prepared_inner_rgb_targets_frames,
        prepared_outer_rgb_targets_frames=prepared_outer_rgb_targets_frames,
        prepared_combined_rgb_targets_frames=prepared_combined_rgb_targets_frames,
        outer_gt_masks_frames=outer_gt_masks_frames,
        person_contour_band_frames=person_contour_band_frames,
        outer_head_eval_masks_frames=outer_head_eval_masks_frames,
        frame_weights_t=frame_weights_t,
        frame_loss_weights_t=frame_loss_weights_t,
        frame_align_scores_t=frame_align_scores_t,
        person_fuse=person_fuse,
        outer_contour_fuse=outer_contour_fuse,
    )
def prepare_stage2_supervision_artifacts(
    *,
    project_root: Path,
    stage1_dir: Path,
    out_dir: Path,
    stage2_cfg: dict,
    frames_dir: Path,
    frames: list[str],
    K_list: list[float],
    bind_mesh_indices: list[int] | None,
    bind_vertices_npz_path: Path | None,
    mesh_faces_np: np.ndarray | None,
    logger,
) -> Path:
    if len(frames) <= 0:
        raise RuntimeError("No frames selected for stage2 supervision export.")
    if bind_mesh_indices is None or len(bind_mesh_indices) != len(frames):
        raise RuntimeError("Stage2 supervision export requires resolved bind_mesh_indices for all selected frames.")
    if bind_vertices_npz_path is None or (not bind_vertices_npz_path.exists()):
        raise RuntimeError(
            "Stage2 supervision export currently requires bind_vertices_npz with verts_cam. "
            f"Missing or invalid: {bind_vertices_npz_path}"
        )
    if mesh_faces_np is None or len(mesh_faces_np) <= 0:
        raise RuntimeError("Stage2 supervision export requires mesh faces, but none were found.")

    preview_root = ensure_dir(out_dir / "supervision_preview")
    inner_rgb_dir = ensure_dir(preview_root / "inner_rgb")
    combined_rgb_dir = ensure_dir(preview_root / "combined_rgb")
    outer_mask_dir = ensure_dir(preview_root / "outer_mask")
    outer_rgb_dir = ensure_dir(preview_root / "outer_rgb")
    contour_mask_dir = ensure_dir(preview_root / "gt_contour_mask")
    contour_train_mask_dir = ensure_dir(preview_root / "gt_train_contour_mask")
    smpl_mask_dir = ensure_dir(preview_root / "smpl_mask")
    cache_path = out_dir / "supervision_cache.npz"
    summary_path = out_dir / "supervision_summary.json"

    first_img = cv2.imread(str(frames_dir / frames[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read first selected frame: {frames_dir / frames[0]}")
    H, W = int(first_img.shape[0]), int(first_img.shape[1])
    K_fallback_np = np.asarray(K_list, dtype=np.float32).reshape(3, 3)
    R_np = np.eye(3, dtype=np.float32)
    t_np = np.zeros((3,), dtype=np.float32)
    vseq = np.load(bind_vertices_npz_path, allow_pickle=True)
    verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)

    use_person_mask = bool(stage2_cfg.get("use_person_mask", False))
    person_masks_npz = stage2_cfg.get("person_masks_npz", str(stage1_dir / "person_masks.npz"))
    person_mask_dilate_px = int(stage2_cfg.get("person_mask_dilate_px", 0))
    person_mask_erode_px = int(stage2_cfg.get("person_mask_erode_px", 0))
    outer_contour_fuse = str(stage2_cfg.get("outer_gt_contour_fuse_mode", "union")).strip().lower()
    outer_gt_contour_modnet_ref_size = int(stage2_cfg.get("outer_gt_contour_modnet_ref_size", 512))
    outer_gt_contour_modnet_threshold = float(stage2_cfg.get("outer_gt_contour_modnet_threshold", 0.5))
    modnet_ckpt_path = stage2_cfg.get("modnet_ckpt_path", None)
    smpl_mask_dilate_px = int(stage2_cfg.get("smpl_mask_dilate_px", 0))
    smpl_mask_erode_px = int(stage2_cfg.get("smpl_mask_erode_px", 0))
    strict_inner_mask = bool(stage2_cfg.get("inner_use_strict_smpl_mask", True))
    smpl_color_mask_erode_px = int(stage2_cfg.get("smpl_color_mask_erode_px", 0))
    inner_matches_smpl = bool(strict_inner_mask) and int(smpl_color_mask_erode_px) <= 0
    inner_mask_dir = smpl_mask_dir if inner_matches_smpl else ensure_dir(preview_root / "inner_mask")

    person_mask_map: dict[str, np.ndarray] = {}
    person_masks_npz_path = Path(str(person_masks_npz)) if person_masks_npz is not None else None
    if person_masks_npz_path is not None and not person_masks_npz_path.is_absolute():
        person_masks_npz_path = project_root / person_masks_npz_path
    if use_person_mask:
        if person_masks_npz_path is None or (not person_masks_npz_path.exists()):
            raise FileNotFoundError(f"person_masks_npz not found: {person_masks_npz_path}")
        pdata = np.load(person_masks_npz_path, allow_pickle=True)
        p_names = [str(x) for x in pdata["frame_names"].tolist()]
        p_masks = np.asarray(pdata["masks"])
        name_to_idx = {n: i for i, n in enumerate(p_names)}
        pd = max(person_mask_dilate_px, 0)
        pe = max(person_mask_erode_px, 0)
        k_d = np.ones((2 * pd + 1, 2 * pd + 1), dtype=np.uint8) if pd > 0 else None
        k_e = np.ones((2 * pe + 1, 2 * pe + 1), dtype=np.uint8) if pe > 0 else None
        for fn in frames:
            if fn not in name_to_idx:
                continue
            m = p_masks[int(name_to_idx[fn])].astype(np.float32)
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.uint8)
            if k_d is not None:
                m = cv2.dilate(m, k_d, iterations=1)
            if k_e is not None:
                m = cv2.erode(m, k_e, iterations=1)
            person_mask_map[str(fn)] = m

    inner_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    outer_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    smpl_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    gt_contour_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    gt_train_contour_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    inner_rgb_targets = np.full((len(frames), H, W, 3), 255, dtype=np.uint8)
    outer_rgb_targets = np.full((len(frames), H, W, 3), 255, dtype=np.uint8)
    combined_rgb_targets = np.full((len(frames), H, W, 3), 255, dtype=np.uint8)
    summary_rows: list[dict] = []

    for i, fn in enumerate(frames):
        img_bgr = cv2.imread(str(frames_dir / fn), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read selected frame: {frames_dir / fn}")
        rgb_u8 = img_bgr[:, :, ::-1].copy()
        gt_t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        mesh_idx = int(bind_mesh_indices[i])
        if mesh_idx < 0 or mesh_idx >= int(verts_seq_np.shape[0]):
            raise IndexError(f"bind mesh index out of range for supervision export: {mesh_idx}")
        verts_np = np.asarray(verts_seq_np[mesh_idx], dtype=np.float32)
        K_np = _read_stage1_frame_camera_K(stage1_dir=stage1_dir, frame_index=int(mesh_idx), frame_name=str(fn), fallback_K=K_fallback_np)
        u_raw, v_raw, valid_v = project_points_numpy(verts_np, K_np, R_np, t_np)
        uv_raw = np.stack([u_raw, v_raw], axis=1).astype(np.float32)
        smpl_mask_np = rasterize_mesh_mask(
            verts=verts_np,
            faces=mesh_faces_np,
            K=K_np,
            R=R_np,
            t=t_np,
            H=H,
            W=W,
            dilate_px=smpl_mask_dilate_px,
            erode_px=smpl_mask_erode_px,
        ).astype(np.uint8)
        smpl_mask_t = torch.from_numpy(smpl_mask_np.astype(np.float32)).unsqueeze(0)
        smpl_mask_for_color = smpl_mask_t
        if smpl_color_mask_erode_px > 0:
            smpl_mask_for_color = torch.from_numpy(
                rasterize_mesh_mask(
                    verts=verts_np,
                    faces=mesh_faces_np,
                    K=K_np,
                    R=R_np,
                    t=t_np,
                    H=H,
                    W=W,
                    dilate_px=smpl_mask_dilate_px,
                    erode_px=smpl_mask_erode_px + smpl_color_mask_erode_px,
                ).astype(np.float32)
            ).unsqueeze(0)
        person_mask_np = person_mask_map.get(str(fn), None)
        person_mask_t = None if person_mask_np is None else torch.from_numpy(person_mask_np.astype(np.float32)).unsqueeze(0)
        gt_train_mask_np = estimate_gt_person_contour_mask_np(
            gt_t,
            hint_mask_u8=(smpl_mask_np * 255).astype(np.uint8),
            modnet_ref_size=int(max(32, outer_gt_contour_modnet_ref_size)),
            modnet_threshold=float(np.clip(float(outer_gt_contour_modnet_threshold), 0.0, 1.0)),
            modnet_ckpt_path=modnet_ckpt_path,
        ).astype(np.uint8)
        if person_mask_np is not None:
            if outer_contour_fuse == "replace":
                gt_train_mask_np = person_mask_np
            elif outer_contour_fuse == "union":
                gt_train_mask_np = np.maximum(gt_train_mask_np, person_mask_np)
            elif outer_contour_fuse == "intersect":
                inter = (gt_train_mask_np * person_mask_np).astype(np.uint8)
                gt_area = int(np.sum(gt_train_mask_np > 0))
                inter_area = int(np.sum(inter > 0))
                keep_ratio = (float(inter_area) / float(max(gt_area, 1))) if gt_area > 0 else 0.0
                if keep_ratio >= 0.995:
                    gt_train_mask_np = inter
        gt_eval_mask_np = gt_train_mask_np.copy()
        smpl_mask_eval_np = smpl_mask_np
        if int(np.sum(gt_eval_mask_np > 0)) > 0 and int(np.sum(smpl_mask_np > 0)) > 0:
            uv_aligned, align_report = estimate_mask_similarity_transform(
                uv=uv_raw,
                valid_v=valid_v,
                faces=mesh_faces_np,
                image_hw=(H, W),
                smpl_mask=(smpl_mask_np * 255).astype(np.uint8),
                person_mask=(gt_eval_mask_np * 255).astype(np.uint8),
            )
            if bool(align_report.get("accepted", False)):
                smpl_mask_eval_np = _rasterize_uv_mask(uv=uv_aligned, valid_v=valid_v, faces=mesh_faces_np, image_hw=(H, W)).astype(np.uint8)
        smpl_mask_eval_t = torch.from_numpy(smpl_mask_eval_np.astype(np.float32)).unsqueeze(0)
        smpl_mask_eval_for_color = smpl_mask_eval_t
        if smpl_color_mask_erode_px > 0:
            k = np.ones((2 * smpl_color_mask_erode_px + 1, 2 * smpl_color_mask_erode_px + 1), dtype=np.uint8)
            smpl_mask_eval_for_color = torch.from_numpy(cv2.erode(smpl_mask_eval_np.astype(np.uint8), k, iterations=1).astype(np.float32)).unsqueeze(0)
        inner_mask_t = resolve_inner_supervision_mask(
            strict_inner_mask=strict_inner_mask,
            smpl_mask=smpl_mask_eval_t,
            smpl_mask_for_color=smpl_mask_eval_for_color,
            person_mask_curr=person_mask_t,
        )
        outer_mask_np = (gt_eval_mask_np * (1 - smpl_mask_eval_np)).astype(np.uint8)
        inner_mask_np = smpl_mask_eval_np.copy() if inner_matches_smpl else np.zeros((H, W), dtype=np.uint8)
        if (not inner_matches_smpl) and inner_mask_t is not None:
            inner_mask_np = (inner_mask_t[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        inner_masks[i] = inner_mask_np
        outer_masks[i] = outer_mask_np
        smpl_masks[i] = smpl_mask_eval_np
        gt_contour_masks[i] = gt_eval_mask_np
        gt_train_contour_masks[i] = gt_train_mask_np
        inner_rgb_targets[i] = _masked_rgb_u8(rgb_u8, inner_mask_np)
        outer_rgb_targets[i] = _masked_rgb_u8(rgb_u8, outer_mask_np)
        combined_rgb_targets[i] = _masked_rgb_u8(rgb_u8, gt_eval_mask_np)
        stem = Path(fn).stem
        if not inner_matches_smpl:
            cv2.imwrite(str(inner_mask_dir / f"{stem}.png"), inner_mask_np * 255)
        cv2.imwrite(str(outer_mask_dir / f"{stem}.png"), outer_mask_np * 255)
        cv2.imwrite(str(smpl_mask_dir / f"{stem}.png"), smpl_mask_eval_np * 255)
        cv2.imwrite(str(contour_mask_dir / f"{stem}.png"), gt_eval_mask_np * 255)
        cv2.imwrite(str(contour_train_mask_dir / f"{stem}.png"), gt_train_mask_np * 255)
        cv2.imwrite(str(inner_rgb_dir / f"{stem}.png"), inner_rgb_targets[i][:, :, ::-1])
        cv2.imwrite(str(outer_rgb_dir / f"{stem}.png"), outer_rgb_targets[i][:, :, ::-1])
        cv2.imwrite(str(combined_rgb_dir / f"{stem}.png"), combined_rgb_targets[i][:, :, ::-1])
        summary_rows.append(
            {
                "frame_idx": int(i),
                "frame_name": str(fn),
                "inner_pixels": int(inner_mask_np.sum()),
                "outer_pixels": int(outer_mask_np.sum()),
                "smpl_pixels": int(smpl_mask_eval_np.sum()),
                "gt_contour_pixels": int(gt_eval_mask_np.sum()),
                "gt_train_contour_pixels": int(gt_train_mask_np.sum()),
            }
        )

    np.savez_compressed(
        cache_path,
        frame_names=np.asarray(frames),
        inner_masks=inner_masks,
        outer_masks=outer_masks,
        smpl_masks=smpl_masks,
        gt_contour_masks=gt_contour_masks,
        gt_train_contour_masks=gt_train_contour_masks,
        inner_rgb_targets=inner_rgb_targets,
        outer_rgb_targets=outer_rgb_targets,
        combined_rgb_targets=combined_rgb_targets,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frame_count": int(len(frames)),
                "cache_path": str(cache_path),
                "preview_dir": str(preview_root),
                "inner_mask_alias_to_smpl_mask": bool(inner_matches_smpl),
                "frames": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved stage2 supervision cache: {cache_path}")
    logger.info(f"Saved stage2 supervision preview: {preview_root}")
    return cache_path
