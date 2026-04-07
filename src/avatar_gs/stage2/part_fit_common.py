
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'stage2_part_fit_{int(time.time() * 1000)}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(str(path), encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def clone_g(g):
    from avatar_gs.gaussian.structured_gaussian import Gaussian3D
    return Gaussian3D(
        xyz=g.xyz.clone(),
        log_scales=g.log_scales.clone(),
        quat=g.quat.clone(),
        opacity_logit=g.opacity_logit.clone(),
        rgb_logit=g.rgb_logit.clone(),
        bind_idx=g.bind_idx.clone() if g.bind_idx is not None else None,
        bind_offset=g.bind_offset.clone() if g.bind_offset is not None else None,
        bind_face_idx=g.bind_face_idx.clone() if g.bind_face_idx is not None else None,
        effective_face_idx=g.effective_face_idx.clone() if g.effective_face_idx is not None else None,
        bind_bary=g.bind_bary.clone() if g.bind_bary is not None else None,
        bind_hybrid_w=g.bind_hybrid_w.clone() if g.bind_hybrid_w is not None else None,
        bind_face_offset_local=g.bind_face_offset_local.clone() if g.bind_face_offset_local is not None else None,
        shell_layer=g.shell_layer.clone() if g.shell_layer is not None else None,
        bind_mix_weight=g.bind_mix_weight.clone() if g.bind_mix_weight is not None else None,
        prune_state=g.prune_state.clone() if g.prune_state is not None else None,
        inner_bind_mode=g.inner_bind_mode.clone() if g.inner_bind_mode is not None else None,
        quality_fail_counter=g.quality_fail_counter.clone() if g.quality_fail_counter is not None else None,
        color_conf_ema=g.color_conf_ema.clone() if g.color_conf_ema is not None else None,
        geom_conf_ema=g.geom_conf_ema.clone() if g.geom_conf_ema is not None else None,
    )


def axis_angle_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    if v.ndim == 2:
        return torch.stack([axis_angle_to_rotmat(x) for x in v], dim=0)
    th = torch.linalg.norm(v)
    if float(th.item()) < 1e-8:
        return torch.eye(3, dtype=v.dtype, device=v.device)
    a = v / th.clamp(min=1e-8)
    x, y, z = a[0], a[1], a[2]
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    K = torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])
    I = torch.eye(3, dtype=v.dtype, device=v.device)
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def load_stage1_kmap(stage1_dir: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    z = np.load(str(stage1_dir / 'smpl_params_sequence.npz'), allow_pickle=True)
    names = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
    focal = np.asarray(z['camera_focal_xy'], dtype=np.float32).reshape(-1, 2)
    princ = np.asarray(z['camera_princpt_xy'], dtype=np.float32).reshape(-1, 2)
    idx = {n: i for i, n in enumerate(names)}
    kmap: dict[str, np.ndarray] = {}
    for i, n in enumerate(names):
        k = np.eye(3, dtype=np.float32)
        k[0, 0], k[1, 1], k[0, 2], k[1, 2] = float(focal[i, 0]), float(focal[i, 1]), float(princ[i, 0]), float(princ[i, 1])
        kmap[n] = k
    return idx, kmap


def load_supervision_align(supervision_cache: Path, frame_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(str(supervision_cache), allow_pickle=True)
    all_names = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
    idx = {n: i for i, n in enumerate(all_names)}
    sel = np.asarray([idx[n] for n in frame_names], dtype=np.int64)
    return (
        np.asarray(z['smpl_align_scales'], dtype=np.float32)[sel],
        np.asarray(z['smpl_align_tx_px'], dtype=np.float32)[sel],
        np.asarray(z['smpl_align_ty_px'], dtype=np.float32)[sel],
        np.asarray(z['smpl_align_accepted'], dtype=np.uint8)[sel],
    )


def bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.10) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size <= 0:
        raise RuntimeError('mask is empty')
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = max(int(x1 - x0 + 1), int(y1 - y0 + 1))
    pad = int(round(float(side) * float(pad_frac)))
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    half = 0.5 * float(side + 2 * pad)
    h, w = mask.shape[:2]
    nx0 = max(int(math.floor(cx - half)), 0)
    ny0 = max(int(math.floor(cy - half)), 0)
    nx1 = min(int(math.ceil(cx + half)), w - 1)
    ny1 = min(int(math.ceil(cy + half)), h - 1)
    return nx0, ny0, nx1, ny1


def crop_intrinsics_to_box(K: torch.Tensor, x0: int, y0: int, x1: int, y1: int, out_h: int, out_w: int) -> torch.Tensor:
    Kc = K.clone()
    src_w = max(int(x1) - int(x0) + 1, 1)
    src_h = max(int(y1) - int(y0) + 1, 1)
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    Kc[0, 0] *= sx
    Kc[1, 1] *= sy
    Kc[0, 2] = (Kc[0, 2] - float(x0)) * sx
    Kc[1, 2] = (Kc[1, 2] - float(y0)) * sy
    return Kc


def render_box_resized(g, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, box_xyxy: tuple[int, int, int, int], out_size: int, bg_rgb=(1.0, 1.0, 1.0)):
    from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
    x0, y0, x1, y1 = box_xyxy
    Kc = crop_intrinsics_to_box(K, x0, y0, x1, y1, out_size, out_size)
    return render_gaussians_3d(g=g, K=Kc, R=R, t=t, H=out_size, W=out_size, bg=bg_rgb, return_alpha=True)


def resize_crop_rgb(rgb_u8: np.ndarray, box_xyxy: tuple[int, int, int, int], out_size: int) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy
    crop = np.asarray(rgb_u8[y0:y1 + 1, x0:x1 + 1], dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def resize_crop_mask(mask: np.ndarray, box_xyxy: tuple[int, int, int, int], out_size: int) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy
    crop = np.asarray(mask[y0:y1 + 1, x0:x1 + 1], dtype=np.uint8)
    out = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return np.asarray(out > 0, dtype=np.uint8)


def project_points_torch(xyz: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    xyz_cam = (xyz @ R.T) + t.view(1, 3)
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    x = xyz_cam[:, 0] / z.clamp(min=1e-6)
    y = xyz_cam[:, 1] / z.clamp(min=1e-6)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v, valid


def sample_rgb_at_uv(image_chw: torch.Tensor, u: torch.Tensor, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    H = int(image_chw.shape[-2])
    W = int(image_chw.shape[-1])
    out = torch.zeros((int(u.shape[0]), 3), dtype=image_chw.dtype, device=image_chw.device)
    if bool(valid.any()):
        uu = torch.round(u[valid]).long().clamp(0, W - 1)
        vv = torch.round(v[valid]).long().clamp(0, H - 1)
        out[valid] = image_chw[:, vv, uu].permute(1, 0)
    return out


def sample_mask_at_uv(mask_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    H = int(mask_hw.shape[-2])
    W = int(mask_hw.shape[-1])
    out = torch.zeros((int(u.shape[0]),), dtype=torch.float32, device=mask_hw.device)
    if bool(valid.any()):
        uu = torch.round(u[valid]).long().clamp(0, W - 1)
        vv = torch.round(v[valid]).long().clamp(0, H - 1)
        out[valid] = mask_hw[vv, uu].float()
    return out


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.expand_as(pred).float()
    return (torch.abs(pred - gt) * w).sum() / (w.sum() + 1e-6)


def weighted_masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    w = mask.expand_as(pred).float()
    if weight is not None:
        w = w * weight.expand_as(pred).float()
    return (torch.abs(pred - gt) * w).sum() / (w.sum() + 1e-6)


def build_detail_weight(rgb_u8: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(rgb_u8, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if bool(region_mask.any()):
        vals = mag[region_mask > 0]
        denom = float(np.percentile(vals, 95)) if vals.size > 0 else float(mag.max())
    else:
        denom = float(mag.max())
    denom = max(denom, 1e-6)
    norm = np.clip(mag / denom, 0.0, 1.0)
    weight = (1.0 + 4.0 * norm).astype(np.float32)
    weight[region_mask <= 0] = 0.0
    return weight


def high_freq(x: torch.Tensor) -> torch.Tensor:
    blur = F.avg_pool2d(x.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    return x - blur


def symmetry_mae(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_flip = torch.flip(pred, dims=[2])
    mask_flip = torch.flip(mask, dims=[2])
    valid = (mask > 0.5) & (mask_flip > 0.5)
    if not bool(valid.any()):
        return pred.sum() * 0.0
    return (torch.abs(pred - pred_flip) * valid.float()).sum() / (valid.float().sum() * float(pred.shape[0]) + 1e-6)


def logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def rgb_to_u8_chw(rgb: torch.Tensor) -> np.ndarray:
    return np.clip(rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)


def save_rgb(path: Path, rgb_chw: torch.Tensor) -> None:
    arr = rgb_to_u8_chw(rgb_chw)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def save_compare(path: Path, gt_chw: torch.Tensor, pred_chw: torch.Tensor, mask_hw: torch.Tensor) -> None:
    gt = rgb_to_u8_chw(gt_chw)
    pred = rgb_to_u8_chw(pred_chw)
    mask = (mask_hw[0].detach().cpu().numpy() > 0.5)
    gt_bg = gt.copy(); gt_bg[~mask] = 255
    pred_bg = pred.copy(); pred_bg[~mask] = 255
    canvas = np.concatenate([gt_bg, pred_bg], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 0:
        return float('nan')
    e = np.square(pred - gt)[mask].mean()
    if e <= 1e-12:
        return float('inf')
    return float(20.0 * math.log10(1.0 / math.sqrt(float(e))))


def ssim_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for i in range(3):
        x, y = pred[:, :, i].astype(np.float32), gt[:, :, i].astype(np.float32)
        mx, my = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
        mx2, my2, mxy = mx * mx, my * my, mx * my
        sx2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx2
        sy2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my2
        sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mxy
        num = (2.0 * mxy + c1) * (2.0 * sxy + c2)
        den = (mx2 + my2 + c1) * (sx2 + sy2 + c2)
        vals.append(float((num / np.clip(den, EPS, None)).mean()))
    return float(np.mean(vals))


def masked_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 0:
        return float('nan')
    pc = pred.copy(); gc = gt.copy()
    pc[~mask] = 1.0
    gc[~mask] = 1.0
    return ssim_rgb(pc, gc)
