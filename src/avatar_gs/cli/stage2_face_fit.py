
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / 'src', THIS_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from avatar_gs.gaussian.binding import parse_binding_mode_config
from avatar_gs.gaussian.binding_layers import apply_inner_binding_per_frame
from avatar_gs.gaussian.structured_gaussian import Gaussian3D, sanitize_inner_surface_binding_
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.segmentation.smpl_face_semantics import load_stage1_vertices_sequence, rasterize_visible_face_index_map, resolve_frame_image_path
from avatar_gs.stage2.fashn_semantics import load_stage2_face_semantics
from avatar_gs.stage2.mesh_runtime import Stage2MeshRuntime

EPS = 1e-8


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'stage2_face_fit_{int(time.time() * 1000)}')
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


def clone_g(g: Gaussian3D) -> Gaussian3D:
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
        raise RuntimeError('face mask is empty')
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


def render_box_resized(g: Gaussian3D, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, box_xyxy: tuple[int, int, int, int], out_size: int, bg_rgb=(1.0, 1.0, 1.0)) -> tuple[torch.Tensor, torch.Tensor]:
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


def project_points_torch(xyz: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_cam = (xyz @ R.T) + t.view(1, 3)
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    u = torch.zeros_like(z)
    v = torch.zeros_like(z)
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


def build_detail_weight(rgb_u8: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(rgb_u8, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if bool(face_mask.any()):
        vals = mag[face_mask > 0]
        denom = float(np.percentile(vals, 95)) if vals.size > 0 else float(mag.max())
    else:
        denom = float(mag.max())
    denom = max(denom, 1e-6)
    norm = np.clip(mag / denom, 0.0, 1.0)
    weight = (1.0 + 4.0 * norm).astype(np.float32)
    weight[face_mask <= 0] = 0.0
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


@dataclass
class TeacherTarget:
    frame_name: str
    frame_idx_local: int
    stage1_idx: int
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    rgb_full: torch.Tensor
    face_mask_full: torch.Tensor
    box_xyxy: tuple[int, int, int, int]
    gt_crop: torch.Tensor
    gt_mask_crop: torch.Tensor
    detail_weight_crop: torch.Tensor
    pose_xyz: torch.Tensor
    pose_quat: torch.Tensor
    pose_log_scales: torch.Tensor
    surface_rgb: torch.Tensor
    surface_valid: torch.Tensor
    surface_weight: torch.Tensor


def main() -> None:
    ap = argparse.ArgumentParser('Two-frame face-only fitter')
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--stage2_ckpt', default='outputs/stage2_r111_twoframe_surfacecolor_full_20260406_0505/gaussians3d.pt')
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--frames_dir', default='data/frames/input')
    ap.add_argument('--smpl_face_semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    ap.add_argument('--teacher_frames', nargs='+', default=['000050.jpg'])
    ap.add_argument('--target_source', default='inner_rgb')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--out_size', type=int, default=512)
    ap.add_argument('--lr_rgb', type=float, default=0.05)
    ap.add_argument('--lr_opacity', type=float, default=0.02)
    ap.add_argument('--lr_scale', type=float, default=0.01)
    ap.add_argument('--face_conf_min', type=float, default=0.55)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve() if not Path(args.stage1_dir).is_absolute() else Path(args.stage1_dir).resolve()
    ckpt_path = (project_root / args.stage2_ckpt).resolve() if not Path(args.stage2_ckpt).is_absolute() else Path(args.stage2_ckpt).resolve()
    supervision_cache = (project_root / args.supervision_cache).resolve() if not Path(args.supervision_cache).is_absolute() else Path(args.supervision_cache).resolve()
    frames_dir = (project_root / args.frames_dir).resolve() if not Path(args.frames_dir).is_absolute() else Path(args.frames_dir).resolve()
    semantics_path = (project_root / args.smpl_face_semantics_npz).resolve() if not Path(args.smpl_face_semantics_npz).is_absolute() else Path(args.smpl_face_semantics_npz).resolve()

    teacher_frames = [str(x) for x in args.teacher_frames]
    tag = str(args.tag).strip() or f"twoframe_facefit_{'_'.join(Path(x).stem for x in teacher_frames)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = (project_root / 'outputs' / 'stage2_face_fit' / tag).resolve()
    train_dir = (out_dir / 'train').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / 'face_fit.log')
    logger.info('start face-only two-frame fit: %s', tag)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device=%s', device)

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    g_base = Gaussian3D.load_state_dict(ckpt['gaussians']).to(str(device))
    sanitize_inner_surface_binding_(g_base)
    ckpt_frames = [str(x) for x in ckpt.get('frames', teacher_frames)]
    ckpt_idx = {n: i for i, n in enumerate(ckpt_frames)}
    cam_w = torch.as_tensor(ckpt.get('cam_w', None), dtype=torch.float32, device=device) if ckpt.get('cam_w', None) is not None else None
    cam_t = torch.as_tensor(ckpt.get('cam_t', None), dtype=torch.float32, device=device) if ckpt.get('cam_t', None) is not None else None

    stage1_idx, kmap = load_stage1_kmap(stage1_dir)
    verts_frame_names, verts_cam_all = load_stage1_vertices_sequence(stage1_dir)
    verts_idx = {n: i for i, n in enumerate(verts_frame_names)}
    semantics = load_stage2_face_semantics(semantics_path)
    semantics_npz = np.load(str(semantics_path), allow_pickle=True)
    faces_np = np.asarray(semantics_npz['faces'], dtype=np.int32)
    face_label_id = int(semantics.raw_label_names.index('face'))
    face_sem_mask = (np.asarray(semantics.raw_label_ids, dtype=np.int64) == face_label_id) & (np.asarray(semantics.raw_label_confidence, dtype=np.float32) >= float(args.face_conf_min))

    shell = g_base.shell_layer if g_base.shell_layer is not None else torch.zeros((int(g_base.xyz.shape[0]),), dtype=torch.int64, device=device)
    eff = g_base.effective_face_idx if g_base.effective_face_idx is not None else None
    if eff is None:
        raise RuntimeError('checkpoint has no effective_face_idx; cannot isolate real face subset')
    eff_np = eff.detach().cpu().numpy().astype(np.int64)
    face_gaussian_mask_np = (shell.detach().cpu().numpy().astype(np.int64) == 0)
    valid_eff = (eff_np >= 0) & (eff_np < int(face_sem_mask.shape[0]))
    face_gaussian_mask_np &= valid_eff
    face_gaussian_mask_np[valid_eff] &= face_sem_mask[eff_np[valid_eff]]
    if int(face_gaussian_mask_np.sum()) <= 0:
        raise RuntimeError('real face gaussian subset is empty; semantics or checkpoint binding is broken')
    face_gaussian_mask = torch.from_numpy(face_gaussian_mask_np).to(device=device, dtype=torch.bool)
    logger.info('face gaussian count=%d / total=%d', int(face_gaussian_mask.sum().item()), int(face_gaussian_mask.numel()))

    scales, txs, tys, accepted = load_supervision_align(supervision_cache, teacher_frames)
    sup_npz = np.load(str(supervision_cache), allow_pickle=True)
    sup_names = [str(x) for x in np.asarray(sup_npz['frame_names']).reshape(-1).tolist()]
    sup_idx = {n: i for i, n in enumerate(sup_names)}
    inner_rgb_targets = np.asarray(sup_npz['inner_rgb_targets'], dtype=np.uint8)
    inner_masks = np.asarray(sup_npz['inner_masks'], dtype=np.uint8)
    target_source = str(args.target_source).strip().lower()
    if target_source not in ('inner_rgb', 'raw_rgb'):
        raise RuntimeError(f'Unsupported target_source: {args.target_source}')
    mesh_runtime = Stage2MeshRuntime.from_stage2_config(
        project_root=project_root,
        stage1_dir=stage1_dir,
        cfg=dict(ckpt),
        frames=teacher_frames,
        logger=logger,
        device=str(device),
        bind_to_mesh=True,
        auto_from_stage2_report=False,
        stage2_dir_for_report=None,
    )
    mesh_runtime.set_frame_smpl_align_from_arrays(frame_names=teacher_frames, scales=scales, tx_px=txs, ty_px=tys, accepted=accepted)
    binding_cfg = parse_binding_mode_config(ckpt.get('binding', None), default_mode='games_face')
    bind_mask = (g_base.bind_idx.to(device=device, dtype=torch.long) >= 0) if g_base.bind_idx is not None else None
    if bind_mask is None or not bool(bind_mask.any()):
        raise RuntimeError('checkpoint has no valid bind_idx; cannot bind face gaussians to SMPL frames')
    bind_idx_t = g_base.bind_idx.to(device=device, dtype=torch.long)
    bind_offset_t = g_base.bind_offset.to(device=device, dtype=torch.float32) if g_base.bind_offset is not None else None
    bind_face_idx_t = g_base.bind_face_idx.to(device=device, dtype=torch.long) if g_base.bind_face_idx is not None else None
    bind_bary_t = g_base.bind_bary.to(device=device, dtype=torch.float32) if g_base.bind_bary is not None else None
    quat_blend = float(ckpt.get('bind_quat_normal_blend', 1.0))

    teachers: list[TeacherTarget] = []
    for local_idx, frame_name in enumerate(teacher_frames):
        if frame_name not in stage1_idx or frame_name not in verts_idx:
            raise RuntimeError(f'missing stage1 alignment data for {frame_name}')
        image_path = resolve_frame_image_path(frames_dir, frame_name)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'failed to read image: {image_path}')
        raw_rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if target_source == 'inner_rgb':
            if frame_name not in sup_idx:
                raise RuntimeError(f'missing supervision target for {frame_name}')
            rgb_u8 = np.asarray(inner_rgb_targets[int(sup_idx[frame_name])], dtype=np.uint8)
            inner_mask_full = np.asarray(inner_masks[int(sup_idx[frame_name])] > 0, dtype=np.uint8)
        else:
            rgb_u8 = raw_rgb_u8
            inner_mask_full = np.ones(raw_rgb_u8.shape[:2], dtype=np.uint8)
        K_np = np.asarray(kmap[frame_name], dtype=np.float32)
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=np.asarray(verts_cam_all[int(verts_idx[frame_name])], dtype=np.float32),
            faces=faces_np,
            K=K_np,
            image_hw=rgb_u8.shape[:2],
        )
        pix_valid = face_index_map >= 0
        face_mask_2d = np.zeros(face_index_map.shape[:2], dtype=np.uint8)
        if bool(pix_valid.any()):
            fids = face_index_map[pix_valid].astype(np.int64)
            is_face = face_sem_mask[fids]
            face_mask_2d[pix_valid] = is_face.astype(np.uint8)
        face_mask_2d = np.asarray((face_mask_2d > 0) & (inner_mask_full > 0), dtype=np.uint8)
        if int(face_mask_2d.sum()) <= 0:
            raise RuntimeError(f'face mask is empty for {frame_name}')
        box = bbox_from_mask(face_mask_2d)
        detail_weight_full = build_detail_weight(rgb_u8, face_mask_2d)
        gt_crop_u8 = resize_crop_rgb(rgb_u8, box, int(args.out_size))
        gt_mask_u8 = resize_crop_mask(face_mask_2d, box, int(args.out_size))
        detail_weight_u8 = resize_crop_rgb(np.repeat((detail_weight_full * 255.0).astype(np.uint8)[:, :, None], 3, axis=2), box, int(args.out_size))
        gt_crop = torch.from_numpy(gt_crop_u8.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        gt_mask = torch.from_numpy(gt_mask_u8.astype(np.float32)).unsqueeze(0).to(device)
        detail_weight_crop = torch.from_numpy(detail_weight_u8[:, :, 0].astype(np.float32) / 255.0).unsqueeze(0).to(device)

        g_bound = clone_g(g_base)
        frame = mesh_runtime.load_frame_binding_frame(local_idx, K_np=K_np, binding_cfg_dict=ckpt.get('binding', None), bind_cache_size=8)
        apply_inner_binding_per_frame(
            g=g_bound,
            frame=frame,
            bind_mask=bind_mask,
            bind_vertex_idx=bind_idx_t,
            bind_vertex_offset=bind_offset_t,
            bind_face_idx=bind_face_idx_t,
            bind_bary=bind_bary_t,
            cfg=binding_cfg,
            quat_blend=quat_blend,
            prev_vertex_dirs=None,
        )
        pose_xyz = g_bound.xyz[face_gaussian_mask].detach()
        pose_quat = F.normalize(g_bound.quat[face_gaussian_mask].detach(), dim=1, eps=1e-8)
        pose_log_scales = g_bound.log_scales[face_gaussian_mask].detach()

        if cam_w is None or cam_t is None or frame_name not in ckpt_idx:
            R = torch.eye(3, dtype=torch.float32, device=device)
            t = torch.zeros((3,), dtype=torch.float32, device=device)
        else:
            j = ckpt_idx[frame_name]
            R = axis_angle_to_rotmat(cam_w[j])
            t = cam_t[j]
        K_t = torch.from_numpy(K_np).to(device=device, dtype=torch.float32)
        rgb_full_t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        face_mask_full_t = torch.from_numpy(face_mask_2d.astype(np.float32)).to(device)
        u, v, valid = project_points_torch(pose_xyz, K_t, R, t)
        valid = valid & (u >= 0.0) & (u <= float(rgb_u8.shape[1] - 1)) & (v >= 0.0) & (v <= float(rgb_u8.shape[0] - 1))
        visible_face = sample_mask_at_uv(face_mask_full_t, u, v, valid) > 0.5
        detail_weight_full_t = torch.from_numpy(detail_weight_full.astype(np.float32)).to(device)
        surface_weight = sample_mask_at_uv(detail_weight_full_t, u, v, valid & visible_face)
        sampled_rgb = sample_rgb_at_uv(rgb_full_t, u, v, valid & visible_face)
        teachers.append(TeacherTarget(
            frame_name=frame_name,
            frame_idx_local=local_idx,
            stage1_idx=int(stage1_idx[frame_name]),
            K=K_t,
            R=R,
            t=t,
            rgb_full=rgb_full_t,
            face_mask_full=face_mask_full_t,
            box_xyxy=box,
            gt_crop=gt_crop,
            gt_mask_crop=gt_mask,
            detail_weight_crop=detail_weight_crop,
            pose_xyz=pose_xyz,
            pose_quat=pose_quat,
            pose_log_scales=pose_log_scales,
            surface_rgb=sampled_rgb.detach(),
            surface_valid=(valid & visible_face).detach(),
            surface_weight=surface_weight.detach(),
        ))
        logger.info('teacher %s | target_source=%s | face_pixels=%d | box=%s | visible_face_gauss=%d', frame_name, target_source, int(face_mask_2d.sum()), str(box), int((valid & visible_face).sum().item()))

    face_rgb_param = torch.nn.Parameter(g_base.rgb_logit[face_gaussian_mask].detach().clone())
    face_opacity_param = torch.nn.Parameter(g_base.opacity_logit[face_gaussian_mask].detach().clone())
    face_scale_delta = torch.nn.Parameter(torch.zeros((int(face_rgb_param.shape[0]), 1), dtype=torch.float32, device=device))

    with torch.no_grad():
        rgb_sum = torch.zeros((int(face_rgb_param.shape[0]), 3), dtype=torch.float32, device=device)
        rgb_cnt = torch.zeros((int(face_rgb_param.shape[0]), 1), dtype=torch.float32, device=device)
        for teacher in teachers:
            if bool(teacher.surface_valid.any()):
                rgb_sum[teacher.surface_valid] += teacher.surface_rgb[teacher.surface_valid]
                rgb_cnt[teacher.surface_valid] += 1.0
        base_rgb = torch.sigmoid(face_rgb_param.data)
        avg_rgb = torch.where(rgb_cnt > 0.0, rgb_sum / rgb_cnt.clamp(min=1.0), base_rgb)
        face_rgb_param.data.copy_(logit(avg_rgb))

    optimizer = torch.optim.Adam([
        {'params': [face_rgb_param], 'lr': float(args.lr_rgb)},
        {'params': [face_opacity_param], 'lr': float(args.lr_opacity)},
        {'params': [face_scale_delta], 'lr': float(args.lr_scale)},
    ])

    best_loss = float('inf')
    best_state = None
    history: list[dict[str, float]] = []
    for it in range(1, int(args.iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        scale_bias = 0.75 * torch.tanh(face_scale_delta)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss_render_acc = 0.0
        for teacher in teachers:
            log_scales = (teacher.pose_log_scales + scale_bias.repeat(1, 3)).clamp(min=-10.0, max=0.0)
            g_face = Gaussian3D(
                xyz=teacher.pose_xyz,
                log_scales=log_scales,
                quat=teacher.pose_quat,
                opacity_logit=face_opacity_param,
                rgb_logit=face_rgb_param,
            )
            pred_crop, alpha_crop = render_box_resized(g_face, teacher.K, teacher.R, teacher.t, teacher.box_xyxy, int(args.out_size), bg_rgb=(1.0, 1.0, 1.0))
            gt_crop = teacher.gt_crop
            gt_mask = teacher.gt_mask_crop
            loss_rgb = masked_l1(pred_crop, gt_crop, gt_mask)
            loss_edge = masked_l1(high_freq(pred_crop), high_freq(gt_crop), gt_mask)
            loss_alpha = F.l1_loss(alpha_crop, gt_mask)
            if bool(teacher.surface_valid.any()):
                curr_surface_rgb = torch.sigmoid(face_rgb_param[teacher.surface_valid]).clamp(0.0, 1.0)
                loss_surface = F.l1_loss(curr_surface_rgb, teacher.surface_rgb[teacher.surface_valid])
            else:
                loss_surface = pred_crop.sum() * 0.0
            loss_sym = symmetry_mae(pred_crop, gt_mask)
            loss_scale_reg = torch.relu(scale_bias.mean() + 0.15) + 0.05 * torch.abs(scale_bias).mean()
            loss = (1.0 * loss_rgb + 0.55 * loss_edge + 0.35 * loss_alpha + 0.85 * loss_surface + 0.08 * loss_sym + 0.02 * loss_scale_reg)
            total_loss = total_loss + loss
            loss_render_acc += float(loss_rgb.detach().item())
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            face_rgb_param.data.clamp_(-12.0, 12.0)
            face_opacity_param.data.clamp_(-12.0, 12.0)
        total_value = float(total_loss.detach().item())
        history.append({'iter': int(it), 'loss': total_value})
        if total_value < best_loss:
            best_loss = total_value
            best_state = {
                'rgb_logit': face_rgb_param.detach().clone(),
                'opacity_logit': face_opacity_param.detach().clone(),
                'scale_delta': face_scale_delta.detach().clone(),
                'iter': int(it),
                'loss': float(total_value),
            }
        if (it == 1) or (it % int(max(args.log_every, 1)) == 0) or (it == int(args.iters)):
            logger.info('iter %04d/%04d | total=%.6f | render_l1_avg=%.6f', it, int(args.iters), total_value, loss_render_acc / max(len(teachers), 1))

    if best_state is None:
        raise RuntimeError('training produced no best_state')

    face_rgb_final = best_state['rgb_logit'].to(device)
    face_opacity_final = best_state['opacity_logit'].to(device)
    face_scale_final = best_state['scale_delta'].to(device)

    per_frame: list[dict[str, float | str]] = []
    for teacher in teachers:
        log_scales = (teacher.pose_log_scales + 0.75 * torch.tanh(face_scale_final).repeat(1, 3)).clamp(min=-10.0, max=0.0)
        g_face = Gaussian3D(
            xyz=teacher.pose_xyz,
            log_scales=log_scales,
            quat=teacher.pose_quat,
            opacity_logit=face_opacity_final,
            rgb_logit=face_rgb_final,
        )
        pred_crop, alpha_crop = render_box_resized(g_face, teacher.K, teacher.R, teacher.t, teacher.box_xyxy, int(args.out_size), bg_rgb=(1.0, 1.0, 1.0))
        save_rgb(out_dir / 'face_only' / f"{Path(teacher.frame_name).stem}_face.png", pred_crop)
        save_compare(out_dir / 'face_only' / f"{Path(teacher.frame_name).stem}_compare.png", teacher.gt_crop, pred_crop, teacher.gt_mask_crop)
        save_rgb(out_dir / 'face_only' / f"{Path(teacher.frame_name).stem}_alpha_rgb.png", alpha_crop.repeat(3, 1, 1))

        pred_np = pred_crop.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        gt_np = teacher.gt_crop.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        mask_np = (teacher.gt_mask_crop[0].detach().cpu().numpy() > 0.5)
        mae_v = float(np.abs(pred_np - gt_np)[mask_np].mean()) if int(mask_np.sum()) > 0 else float('nan')
        psnr_v = psnr(pred_np, gt_np, mask_np)
        ssim_v = masked_ssim(pred_np, gt_np, mask_np)
        sym_v = float(symmetry_mae(pred_crop, teacher.gt_mask_crop).detach().item())
        per_frame.append({
            'frame_name': teacher.frame_name,
            'mae': mae_v,
            'psnr': psnr_v,
            'ssim': ssim_v,
            'symmetry_mae': sym_v,
        })

    fit_state = {
        'face_gaussian_mask': face_gaussian_mask.detach().cpu(),
        'teacher_frames': teacher_frames,
        'rgb_logit': face_rgb_final.detach().cpu(),
        'opacity_logit': face_opacity_final.detach().cpu(),
        'scale_delta': face_scale_final.detach().cpu(),
        'best_iter': int(best_state['iter']),
        'best_loss': float(best_state['loss']),
    }
    torch.save(fit_state, train_dir / 'face_fit_state.pt')

    preview_ckpt = dict(ckpt)
    g_preview = clone_g(g_base)
    with torch.no_grad():
        g_preview.rgb_logit[face_gaussian_mask] = face_rgb_final.detach()
        g_preview.opacity_logit[face_gaussian_mask] = face_opacity_final.detach()
        g_preview.log_scales[face_gaussian_mask] = g_preview.log_scales[face_gaussian_mask] + 0.75 * torch.tanh(face_scale_final).repeat(1, 3).detach()
    preview_ckpt['gaussians'] = g_preview.state_dict()
    torch.save(preview_ckpt, train_dir / 'gaussians3d_facefit_preview.pt')

    summary = {
        'tag': tag,
        'teacher_frames': teacher_frames,
        'init_ckpt': str(ckpt_path),
        'face_semantics_npz': str(semantics_path),
        'frames_dir': str(frames_dir),
        'train_dir': str(train_dir),
        'out_dir': str(out_dir),
        'device': str(device),
        'target_source': str(target_source),
        'iters': int(args.iters),
        'out_size': int(args.out_size),
        'best_iter': int(best_state['iter']),
        'best_loss': float(best_state['loss']),
        'face_gaussian_count': int(face_gaussian_mask.sum().item()),
        'per_frame': per_frame,
        'mean_mae': float(np.mean([x['mae'] for x in per_frame])),
        'mean_ssim': float(np.mean([x['ssim'] for x in per_frame])),
        'mean_symmetry_mae': float(np.mean([x['symmetry_mae'] for x in per_frame])),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('done | best_iter=%d | mean_mae=%.6f | mean_ssim=%.6f | mean_symmetry_mae=%.6f', summary['best_iter'], summary['mean_mae'], summary['mean_ssim'], summary['mean_symmetry_mae'])


if __name__ == '__main__':
    main()
