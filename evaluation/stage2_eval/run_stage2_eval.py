from __future__ import annotations

import argparse, json, logging, math, shutil, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / 'src', THIS_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from avatar_gs.gaussian.binding import build_vertex_adjacency, parse_binding_mode_config
from avatar_gs.gaussian.binding_layers import apply_inner_binding_per_frame, apply_outer_anchor_per_frame
from avatar_gs.gaussian.structured_gaussian import Gaussian3D, sanitize_inner_surface_binding_
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d
from avatar_gs.segmentation.fashn import FASHN_COARSE_GROUPS, build_fashn_raw_to_coarse_id, get_fashn_label_maps, load_fashn_parser, predict_fashn_segmentation
from avatar_gs.stage2 import Stage2MeshRuntime, load_stage2_supervision_cache
from avatar_gs.stage2.fashn_semantics import load_stage2_face_semantics
from avatar_gs.stage2.mesh_runtime import compute_vertex_normals_np
from avatar_gs.utils.cfg import load_yaml
from evaluation.fashn_face_semantics.export_artifacts import RAW_PALETTE, save_fashn_2d_preview

MIN_PART_PIXELS = 64
EPS = 1e-8


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'stage2_eval_{int(time.time()*1000)}')
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


def jfix(x):
    if isinstance(x, dict):
        return {str(k): jfix(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jfix(v) for v in x]
    if isinstance(x, np.ndarray):
        return jfix(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def save_rgb(path: Path, rgb):
    if torch.is_tensor(rgb):
        arr = rgb.detach().cpu().permute(1, 2, 0).numpy()
    else:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
    bgr = cv2.cvtColor(np.clip(arr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f'Failed to write {path}')


def save_u8(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f'Failed to write {path}')


def rgb_np(x: torch.Tensor) -> np.ndarray:
    return np.asarray(x.detach().cpu().permute(1, 2, 0).numpy(), dtype=np.float32).clip(0.0, 1.0)


def mask_np(x: torch.Tensor, th: float = 0.5) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr > th, dtype=bool)


def axis_angle_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    if v.ndim == 2:
        return torch.stack([axis_angle_to_rotmat(x) for x in v], dim=0)
    th = torch.linalg.norm(v)
    if float(th.item()) < 1e-8:
        return torch.eye(3, dtype=v.dtype, device=v.device)
    a = v / th.clamp(min=1e-8)
    x, y, z = a[0], a[1], a[2]
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    K = torch.stack([torch.stack([zero, -z, y]), torch.stack([z, zero, -x]), torch.stack([-y, x, zero])])
    I = torch.eye(3, dtype=v.dtype, device=v.device)
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def alpha_to_mask(alpha: torch.Tensor, th: float, gamma: float) -> torch.Tensor:
    a = alpha.clamp(0.0, 1.0)
    if float(gamma) != 1.0:
        a = torch.pow(a, float(gamma))
    if float(th) > 0.0:
        a = (a > float(th)).to(dtype=a.dtype)
    return a


def composite(fg_rgb: torch.Tensor, fg_alpha: torch.Tensor, bg_rgb: torch.Tensor) -> torch.Tensor:
    return fg_rgb + (1.0 - fg_alpha) * bg_rgb


def clone_g(g: Gaussian3D) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz.clone(), log_scales=g.log_scales.clone(), quat=g.quat.clone(), opacity_logit=g.opacity_logit.clone(), rgb_logit=g.rgb_logit.clone(),
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


def select_g(g: Gaussian3D, m: torch.Tensor) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz[m], log_scales=g.log_scales[m], quat=g.quat[m], opacity_logit=g.opacity_logit[m], rgb_logit=g.rgb_logit[m],
        bind_idx=g.bind_idx[m] if g.bind_idx is not None else None,
        bind_offset=g.bind_offset[m] if g.bind_offset is not None else None,
        bind_face_idx=g.bind_face_idx[m] if g.bind_face_idx is not None else None,
        effective_face_idx=g.effective_face_idx[m] if g.effective_face_idx is not None else None,
        bind_bary=g.bind_bary[m] if g.bind_bary is not None else None,
        bind_hybrid_w=g.bind_hybrid_w[m] if g.bind_hybrid_w is not None else None,
        bind_face_offset_local=g.bind_face_offset_local[m] if g.bind_face_offset_local is not None else None,
        shell_layer=g.shell_layer[m] if g.shell_layer is not None else None,
        bind_mix_weight=g.bind_mix_weight[m] if g.bind_mix_weight is not None else None,
        prune_state=g.prune_state[m] if g.prune_state is not None else None,
        inner_bind_mode=g.inner_bind_mode[m] if g.inner_bind_mode is not None else None,
        quality_fail_counter=g.quality_fail_counter[m] if g.quality_fail_counter is not None else None,
        color_conf_ema=g.color_conf_ema[m] if g.color_conf_ema is not None else None,
        geom_conf_ema=g.geom_conf_ema[m] if g.geom_conf_ema is not None else None,
    )


def diff_map(pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> np.ndarray:
    d = np.abs(pred - gt).mean(axis=2)
    d[~m] = 0.0
    return cv2.applyColorMap(np.clip(d * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def bbox(mask: np.ndarray, pad: int = 4):
    ys, xs = np.where(mask)
    if ys.size <= 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = int(ys.max()) + pad + 1
    x0 = max(int(xs.min()) - pad, 0)
    x1 = int(xs.max()) + pad + 1
    return y0, y1, x0, x1


def face_only_side_by_side(gt_u8: np.ndarray, pred_u8: np.ndarray, face_mask: np.ndarray, pad: int = 24):
    face_mask = np.asarray(face_mask, dtype=bool)
    bb = bbox(face_mask, pad=pad)
    if bb is None:
        return None
    y0, y1, x0, x1 = bb
    gt_face = np.asarray(gt_u8, dtype=np.uint8).copy()
    pred_face = np.asarray(pred_u8, dtype=np.uint8).copy()
    gt_face[~face_mask] = 255
    pred_face[~face_mask] = 255
    gt_crop = gt_face[y0:y1, x0:x1]
    pred_crop = pred_face[y0:y1, x0:x1]
    return np.concatenate([
        cv2.cvtColor(gt_crop, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(pred_crop, cv2.COLOR_RGB2BGR),
    ], axis=1)


def mae(pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> float:
    if int(m.sum()) <= 0:
        return float('nan')
    return float(np.abs(pred - gt)[m].mean())


def psnr(pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> float:
    if int(m.sum()) <= 0:
        return float('nan')
    e = np.square(pred - gt)[m].mean()
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


def ssim(pred: np.ndarray, gt: np.ndarray, m: np.ndarray) -> float:
    bb = bbox(m)
    if bb is None:
        return float('nan')
    y0, y1, x0, x1 = bb
    mc = m[y0:y1, x0:x1]
    pc, gc = pred[y0:y1, x0:x1].copy(), gt[y0:y1, x0:x1].copy()
    pc[~mc], gc[~mc] = 1.0, 1.0
    return ssim_rgb(pc, gc)


def clean_mask(mask_u8: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=np.uint8)
    out = np.zeros_like(m, dtype=np.uint8)
    out[labels == int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1] = 255
    return out


def boundary_mask(mask_u8: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    out = np.zeros_like(m, dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(out, contours, -1, color=255, thickness=1, lineType=cv2.LINE_8)
    return out


def boundary_stats(src: np.ndarray, tgt: np.ndarray):
    s, t = np.asarray(src, dtype=np.uint8) > 0, np.asarray(tgt, dtype=np.uint8) > 0
    if int(s.sum()) <= 0:
        return {'mean_px': float('nan'), 'p95_px': float('nan'), 'within_2px': float('nan')}
    dist = cv2.distanceTransform((~t).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[s].astype(np.float32).reshape(-1)
    return {'mean_px': float(vals.mean()), 'p95_px': float(np.percentile(vals, 95)), 'within_2px': float((vals <= 2.0).mean())}


def mask_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray):
    p, g = np.asarray(pred_u8, dtype=np.uint8) > 0, np.asarray(gt_u8, dtype=np.uint8) > 0
    inter, union = float(np.logical_and(p, g).sum()), float(max(1, np.logical_or(p, g).sum()))
    pa, ga = float(max(1, p.sum())), float(max(1, g.sum()))
    p2g, g2p = boundary_stats(boundary_mask(pred_u8), boundary_mask(gt_u8)), boundary_stats(boundary_mask(gt_u8), boundary_mask(pred_u8))
    return {'iou': inter / union, 'outside_ratio': float(np.logical_and(p, np.logical_not(g)).sum()) / pa, 'inside_coverage': inter / ga, 'area_ratio': pa / ga, 'pred_to_gt_boundary_mean_px': float(p2g['mean_px']), 'pred_to_gt_boundary_p95_px': float(p2g['p95_px']), 'gt_to_pred_boundary_mean_px': float(g2p['mean_px']), 'gt_to_pred_boundary_p95_px': float(g2p['p95_px'])}


def load_stage1_kmap(stage1_dir: Path):
    z = np.load(str(stage1_dir / 'smpl_params_sequence.npz'), allow_pickle=True)
    names = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
    focal = np.asarray(z['camera_focal_xy'], dtype=np.float32).reshape(-1, 2)
    princ = np.asarray(z['camera_princpt_xy'], dtype=np.float32).reshape(-1, 2)
    idx = {n: i for i, n in enumerate(names)}
    kmap = {}
    for i, n in enumerate(names):
        k = np.eye(3, dtype=np.float32)
        k[0, 0], k[1, 1], k[0, 2], k[1, 2] = float(focal[i, 0]), float(focal[i, 1]), float(princ[i, 0]), float(princ[i, 1])
        kmap[n] = k
    return idx, kmap


def target_arrays(sup, target_layer: str, i: int):
    mode = str(target_layer).strip().lower()
    if mode == 'inner':
        return np.asarray(sup['inner_rgb_targets'][i], dtype=np.uint8), np.asarray(sup['inner_masks'][i] > 0, dtype=bool)
    if mode == 'outer':
        return np.asarray(sup['outer_rgb_targets'][i], dtype=np.uint8), np.asarray(sup['outer_masks'][i] > 0, dtype=bool)
    if mode == 'combined':
        m = np.asarray((sup['inner_masks'][i] > 0) | (sup['outer_masks'][i] > 0), dtype=bool)
        return np.asarray(sup['combined_rgb_targets'][i], dtype=np.uint8), m
    raise ValueError(f'Unsupported target_layer: {target_layer}')


class EvalRenderer:
    def __init__(self, project_root: Path, stage1_dir: Path, stage2_dir: Path, ckpt_path: Path, frame_names, sup, target_layer: str, alpha_threshold: float, logger: logging.Logger):
        self.project_root, self.stage1_dir, self.stage2_dir = project_root, stage1_dir, stage2_dir
        self.frame_names, self.sup, self.target_layer, self.alpha_threshold, self.logger = [str(x) for x in frame_names], sup, str(target_layer).strip().lower(), float(alpha_threshold), logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg_root = load_yaml(project_root / 'configs' / 'render.yaml') if (project_root / 'configs' / 'render.yaml').exists() else {}
        self.cfg = dict(cfg_root.get('render', {})) if isinstance(cfg_root, dict) else {}
        self.deform_alpha_gamma = float(self.cfg.get('deform_alpha_gamma', 1.0))
        self.deform_alpha_gain_max = float(self.cfg.get('deform_alpha_gain_max', 1.0))
        self.bound_opacity_min = float(self.cfg.get('bound_opacity_min', 0.04))
        self.bind_align_quat_to_mesh_normals = bool(self.cfg.get('bind_align_quat_to_mesh_normals', True))
        self.bind_quat_normal_blend = float(np.clip(self.cfg.get('bind_quat_normal_blend', 1.0), 0.0, 1.0))
        self.render_bound_only = bool(self.cfg.get('deform_render_bound_only', True))
        self.bind_cache_size = int(self.cfg.get('bind_cache_size', 8))
        self.k_scale = float(self.cfg.get('k_scale', 1.0))
        self.prev_vertex_dirs = None
        self.mesh_normal_cache = {}
        self.stage1_idx, self.kmap = load_stage1_kmap(stage1_dir)
        self.ckpt = torch.load(str(ckpt_path), map_location='cpu')
        self.g = Gaussian3D.load_state_dict(self.ckpt['gaussians']).to(str(self.device))
        sanitize_inner_surface_binding_(self.g)
        self.ckpt_frames = [str(x) for x in self.ckpt.get('frames', [])]
        self.ckpt_idx = {n: i for i, n in enumerate(self.ckpt_frames)}
        self.cam_w = torch.as_tensor(self.ckpt.get('cam_w', None), dtype=torch.float32, device=self.device) if self.ckpt.get('cam_w', None) is not None else None
        self.cam_t = torch.as_tensor(self.ckpt.get('cam_t', None), dtype=torch.float32, device=self.device) if self.ckpt.get('cam_t', None) is not None else None
        binding_cfg_dict = self.cfg.get('binding', self.ckpt.get('binding', None))
        default_mode = 'games_face' if (self.g.bind_face_idx is not None and self.g.bind_bary is not None) else 'legacy'
        self.binding_cfg = parse_binding_mode_config(binding_cfg_dict, default_mode=default_mode)
        self.mesh_runtime = Stage2MeshRuntime.from_stage2_config(project_root=project_root, stage1_dir=stage1_dir, cfg=self.cfg, frames=self.frame_names, logger=logger, device=str(self.device), bind_to_mesh=bool(self.cfg.get('bind_to_mesh', True)), auto_from_stage2_report=bool(self.cfg.get('bind_rigid_align_auto_from_stage2', True)), stage2_dir_for_report=stage2_dir)
        self.mesh_runtime.set_frame_smpl_align_from_arrays(frame_names=self.frame_names, scales=np.asarray(sup['smpl_align_scales'], dtype=np.float32), tx_px=np.asarray(sup['smpl_align_tx_px'], dtype=np.float32), ty_px=np.asarray(sup['smpl_align_ty_px'], dtype=np.float32), accepted=np.asarray(sup['smpl_align_accepted'], dtype=np.uint8))
        self.bind_idx_t = self.g.bind_idx.to(device=self.device, dtype=torch.long) if self.g.bind_idx is not None else None
        self.bind_vertex_offset_t = self.g.bind_offset.to(device=self.device, dtype=torch.float32) if self.g.bind_offset is not None else None
        self.bind_face_idx_t = self.g.bind_face_idx.to(device=self.device, dtype=torch.long) if self.g.bind_face_idx is not None else None
        self.bind_bary_t = self.g.bind_bary.to(device=self.device, dtype=torch.float32) if self.g.bind_bary is not None else None
        self.binding_enabled = bool(self.g.bind_idx is not None and self.mesh_runtime.context.bind_mesh_dir is not None)
        self.bind_faces_np = None if self.mesh_runtime.context.bind_faces_np is None else np.asarray(self.mesh_runtime.context.bind_faces_np, dtype=np.int32)
        self.bind_adj = None
        if self.bind_faces_np is not None and len(self.bind_faces_np) > 0:
            verts_ref, _ = self.mesh_runtime.load_frame_mesh_verts(0, K_np=self.frame_K(self.frame_names[0]), bind_cache_size=self.bind_cache_size)
            self.bind_adj = build_vertex_adjacency(int(verts_ref.shape[0]), self.bind_faces_np)

    def frame_K(self, frame_name: str):
        k = np.asarray(self.kmap[frame_name], dtype=np.float32).copy()
        if float(self.k_scale) != 1.0:
            k[0, 0] *= self.k_scale; k[1, 1] *= self.k_scale; k[0, 2] *= self.k_scale; k[1, 2] *= self.k_scale
        return k

    def camera(self, frame_name: str):
        j = self.ckpt_idx[frame_name]
        if self.cam_w is None or self.cam_t is None:
            R = torch.eye(3, dtype=torch.float32, device=self.device); t = torch.zeros((3,), dtype=torch.float32, device=self.device)
        else:
            R = axis_angle_to_rotmat(self.cam_w[j]); t = self.cam_t[j]
        K = torch.from_numpy(self.frame_K(frame_name)).to(device=self.device, dtype=torch.float32)
        return R, t, K

    def load_normals(self, i: int, verts_t: torch.Tensor):
        if not self.bind_align_quat_to_mesh_normals or self.bind_faces_np is None or len(self.bind_faces_np) <= 0:
            return None
        if int(i) in self.mesh_normal_cache:
            return self.mesh_normal_cache[int(i)]
        verts_np = self.mesh_runtime.frame_mesh_np_cache.get(int(i), None)
        if verts_np is None:
            verts_np = verts_t.detach().cpu().numpy().astype(np.float32)
            self.mesh_runtime.frame_mesh_np_cache[int(i)] = verts_np
        n = torch.from_numpy(compute_vertex_normals_np(verts_np, self.bind_faces_np)).to(device=self.device, dtype=torch.float32)
        self.mesh_normal_cache[int(i)] = n
        return n

    def apply_binding(self, i: int, frame_name: str, g: Gaussian3D):
        if not self.binding_enabled or self.bind_idx_t is None:
            return g, None
        bind_mask = self.bind_idx_t >= 0
        if not bool(bind_mask.any()):
            return g, None
        k_np = self.frame_K(frame_name)
        if str(self.binding_cfg.mode).strip().lower() == 'legacy':
            verts_t, _ = self.mesh_runtime.load_frame_mesh_verts(i, K_np=k_np, bind_cache_size=self.bind_cache_size)
            normals_t = self.load_normals(i, verts_t)
            bind_idx_use = self.bind_idx_t[bind_mask]
            valid_local = bind_idx_use < int(verts_t.shape[0])
            valid_mask = bind_mask.clone(); valid_mask[bind_mask] = valid_local
            if bool(valid_mask.any()):
                v_idx = self.bind_idx_t[valid_mask]
                xyz = verts_t.index_select(0, v_idx)
                if self.bind_vertex_offset_t is not None:
                    xyz = xyz + self.bind_vertex_offset_t[valid_mask]
                g.xyz[valid_mask] = xyz.detach()
                if self.bind_align_quat_to_mesh_normals and normals_t is not None and self.bind_quat_normal_blend > 0.0:
                    nsel = F.normalize(normals_t.index_select(0, v_idx), dim=1, eps=1e-8)
                    qtar = F.normalize(torch.stack([torch.ones((nsel.shape[0],), device=self.device), nsel[:, 0], nsel[:, 1], nsel[:, 2]], dim=1), dim=1, eps=1e-8)
                    qold = g.quat[valid_mask]
                    qmix = (1.0 - self.bind_quat_normal_blend) * qold + self.bind_quat_normal_blend * qtar
                    g.quat[valid_mask] = F.normalize(qmix, dim=1, eps=1e-8)
            return g, valid_mask
        frame = self.mesh_runtime.load_frame_binding_frame(i, K_np=k_np, binding_cfg_dict=self.cfg.get('binding', self.ckpt.get('binding', None)), bind_cache_size=self.bind_cache_size)
        prev, valid_inner = apply_inner_binding_per_frame(g=g, frame=frame, bind_mask=bind_mask, bind_vertex_idx=self.bind_idx_t, bind_vertex_offset=self.bind_vertex_offset_t, bind_face_idx=self.bind_face_idx_t, bind_bary=self.bind_bary_t, cfg=self.binding_cfg, quat_blend=self.bind_quat_normal_blend, prev_vertex_dirs=self.prev_vertex_dirs)
        prev, valid_outer = apply_outer_anchor_per_frame(g=g, frame=frame, bind_mask=bind_mask, bind_vertex_idx=self.bind_idx_t, bind_vertex_offset=self.bind_vertex_offset_t, bind_face_idx=self.bind_face_idx_t, bind_bary=self.bind_bary_t, cfg=self.binding_cfg, quat_blend=self.bind_quat_normal_blend, prev_vertex_dirs=prev)
        self.prev_vertex_dirs = prev
        return g, (valid_inner | valid_outer)

    def layer_mask(self, g: Gaussian3D, bind_mask_curr):
        keep = torch.ones((int(g.xyz.shape[0]),), dtype=torch.bool, device=self.device)
        if self.target_layer == 'inner':
            if g.shell_layer is not None:
                keep &= g.shell_layer.to(device=self.device, dtype=torch.int64) == 0
            elif g.bind_idx is not None:
                keep &= g.bind_idx.to(device=self.device, dtype=torch.long) >= 0
        elif self.target_layer == 'outer':
            if g.shell_layer is not None:
                keep &= g.shell_layer.to(device=self.device, dtype=torch.int64) == 1
            else:
                keep &= torch.zeros_like(keep)
        elif self.target_layer != 'combined':
            raise ValueError(f'Unsupported target_layer: {self.target_layer}')
        if self.render_bound_only and g.bind_idx is not None:
            keep &= g.bind_idx.to(device=self.device, dtype=torch.long) >= 0
        if bind_mask_curr is not None:
            keep &= bind_mask_curr.to(device=self.device)
        if self.bound_opacity_min > 0.0:
            keep &= torch.sigmoid(g.opacity_logit).squeeze(-1) >= self.bound_opacity_min
        if not bool(keep.any()):
            raise RuntimeError(f'No gaussians left after filtering for {self.target_layer}')
        return keep

    @torch.no_grad()
    def render(self, i: int, frame_name: str):
        R, t, K = self.camera(frame_name)
        gd = clone_g(self.g)
        gd, valid = self.apply_binding(i, frame_name, gd)
        lm = self.layer_mask(gd, valid)
        gr = select_g(gd, lm)
        H, W = int(self.sup['inner_rgb_targets'].shape[1]), int(self.sup['inner_rgb_targets'].shape[2])
        rgb, alpha = render_gaussians_3d(g=gr, K=K, R=R, t=t, H=H, W=W, bg=None, return_alpha=True)
        m = alpha_to_mask(alpha, self.alpha_threshold, self.deform_alpha_gamma)
        gain = (m / alpha.clamp(min=1e-4)).clamp(min=0.0, max=self.deform_alpha_gain_max)
        pred = composite((rgb * gain).clamp(0.0, 1.0), m, torch.ones_like(rgb))
        return pred, m, float(lm.float().mean().item()) if int(lm.numel()) > 0 else 0.0


def copy_ref_preview(project_root: Path, frame_name: str, out_dir: Path):
    src = project_root / 'evaluation' / 'eval_outputs' / 'fashn_face_semantics' / 'fashn_2d_labels' / f'{Path(frame_name).stem}_fashn_2d_labels.png'
    if not src.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def write_report(path: Path, summary: dict, part_summary: dict, worst_frames: list):
    lines = ['# Stage2 Supervision Evaluation Report', '', f"- Frame count: {summary['frame_count']}", f"- Target layer: {summary['target_layer']}", f"- Stage2 checkpoint: `{summary['stage2_ckpt']}`", f"- Worst overall part: {summary['worst_part_name']}", '', '## Global Summary', '', f"- Mean MAE: {summary['global_metrics']['mae_mean']:.6f}", f"- Mean PSNR: {summary['global_metrics']['psnr_mean']:.4f}" if summary['global_metrics']['psnr_mean'] is not None else '- Mean PSNR: null', f"- Mean SSIM: {summary['global_metrics']['ssim_mean']:.6f}", f"- Mean Mask IoU: {summary['global_metrics']['mask_iou_mean']:.6f}", '', '## Worst Parts', '']
    for name, row in sorted(part_summary.items(), key=lambda kv: float(kv[1]['badness_mean']), reverse=True)[:10]:
        lines.append(f"- `{name}` | group={row['group_name']} | badness_mean={row['badness_mean']:.6f} | mae_mean={row['mae_mean']:.6f} | ssim_mean={row['ssim_mean']:.6f} | coverage_mean={row['render_coverage_ratio_mean']:.6f}")
    lines += ['', '## Worst Frames', '']
    for row in worst_frames[:10]:
        lines.append(f"- `{row['frame_name']}` | badness={row['frame_badness']:.6f} | mae={row['global_metrics']['mae']:.6f} | ssim={row['global_metrics']['ssim']:.6f} | worst_parts={', '.join(str(x['part_name']) for x in row['worst_parts'])}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser('Stage2 supervision-frame evaluation')
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--stage2_dir', default='outputs/stage2')
    ap.add_argument('--stage2_ckpt', default='')
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--frames_dir', default='data/frames/input')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--target_layer', default='inner', choices=['inner', 'outer', 'combined'])
    ap.add_argument('--alpha_threshold', type=float, default=0.03)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--frame_names', nargs='*', default=[])
    ap.add_argument('--face_only_render', action='store_true')
    ap.add_argument('--face_only_pad_px', type=int, default=24)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve() if not Path(args.stage1_dir).is_absolute() else Path(args.stage1_dir).resolve()
    stage2_dir = (project_root / args.stage2_dir).resolve() if not Path(args.stage2_dir).is_absolute() else Path(args.stage2_dir).resolve()
    if str(args.stage2_ckpt).strip():
        ckpt_path = (project_root / args.stage2_ckpt).resolve() if not Path(args.stage2_ckpt).is_absolute() else Path(args.stage2_ckpt).resolve()
    else:
        ckpt_path = (stage2_dir / 'gaussians3d.pt').resolve()
    sup_path = (project_root / args.supervision_cache).resolve() if not Path(args.supervision_cache).is_absolute() else Path(args.supervision_cache).resolve()
    if str(args.out_dir).strip():
        out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    else:
        out_dir = (project_root / 'evaluation' / 'eval_outputs' / 'stage2_eval' / f"stage2_eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    if bool(args.face_only_render):
        for p in (out_dir / 'face_only',):
            p.mkdir(parents=True, exist_ok=True)
    else:
        for p in (out_dir / 'overlays',):
            p.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / 'stage2_eval.log')
    logger.info('Stage2 eval start: %s', out_dir)

    z = np.load(str(sup_path), allow_pickle=True)
    all_frames = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
    sup = load_stage2_supervision_cache(cache_path=sup_path, frames=all_frames, logger=logger)
    if sup is None:
        raise RuntimeError(f'Cannot load supervision cache: {sup_path}')
    if len(args.frame_names) > 0:
        name_to_idx = {str(name): i for i, name in enumerate(all_frames)}
        sel_list = [int(name_to_idx[n]) for n in args.frame_names if n in name_to_idx]
        if not sel_list:
            raise RuntimeError(f'No requested frame_names found in supervision cache: {args.frame_names}')
        sel = np.asarray(sel_list, dtype=np.int64)
        frames = [all_frames[i] for i in sel.tolist()]
        sup = {k: (np.asarray(v)[sel] if isinstance(v, np.ndarray) and v.shape[0] == len(all_frames) else v) for k, v in sup.items()}
        sup['frame_names'] = frames
    elif int(args.max_frames) > 0:
        sel = np.arange(min(int(args.max_frames), len(all_frames)), dtype=np.int64)
        frames = [all_frames[i] for i in sel.tolist()]
        sup = {k: (np.asarray(v)[sel] if isinstance(v, np.ndarray) and v.shape[0] == len(all_frames) else v) for k, v in sup.items()}
        sup['frame_names'] = frames
    else:
        frames = list(all_frames)

    stage1_idx, _kmap = load_stage1_kmap(stage1_dir)
    miss_stage1 = [n for n in frames if n not in stage1_idx]
    if miss_stage1:
        raise RuntimeError(f'Frames missing in stage1 smpl_params_sequence.npz: {miss_stage1[:5]}')

    sem_npz = stage1_dir / 'fashn_face_semantics' / 'smpl_face_semantics.npz'
    sem_match = False
    if sem_npz.exists():
        load_stage2_face_semantics(sem_npz)
        sem_z = np.load(str(sem_npz), allow_pickle=True)
        sem_frames = [str(x) for x in np.asarray(sem_z['frame_names']).reshape(-1).tolist()] if 'frame_names' in sem_z.files else []
        sem_match = all(str(name) in set(sem_frames) for name in frames)
        logger.info('Stage1 semantics frame match: %s', str(bool(sem_match)).lower())

    ids_to_labels, _ = get_fashn_label_maps(runtime_dir=None)
    raw_to_group = build_fashn_raw_to_coarse_id(ids_to_labels)
    face_raw_id = next((int(rid) for rid, name in ids_to_labels.items() if str(name) == 'face'), None)
    parser = load_fashn_parser(runtime_dir=None, model_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, ckpt_path, frames, sup, args.target_layer, args.alpha_threshold, logger)

    jsonl = (out_dir / 'frame_metrics.jsonl').open('w', encoding='utf-8')
    badlog = (out_dir / 'bad_regions.log').open('w', encoding='utf-8')
    frame_reports, all_part_rows = [], []

    for i, frame_name in enumerate(frames):
        gt_u8, eval_mask = target_arrays(sup, args.target_layer, i)
        gt = np.asarray(gt_u8, dtype=np.float32) / 255.0
        pred_t, pred_m_t, bind_valid_ratio = renderer.render(i, frame_name)
        pred = rgb_np(pred_t)
        pred_mask = mask_np(pred_m_t)
        gt_mask_u8, pred_mask_u8 = clean_mask(eval_mask.astype(np.uint8) * 255), clean_mask(pred_mask.astype(np.uint8) * 255)
        seg = predict_fashn_segmentation(np.asarray(gt_u8, dtype=np.uint8), parser=parser)
        gm = {'mae': mae(pred, gt, eval_mask), 'psnr': psnr(pred, gt, eval_mask), 'ssim': ssim(pred, gt, eval_mask)}
        gm.update(mask_metrics(pred_mask_u8, gt_mask_u8))
        frame_badness = float(gm['mae'] + (1.0 - gm['ssim']) + (1.0 - gm['inside_coverage']))
        rows = []

        for raw_id in sorted(int(k) for k in ids_to_labels.keys()):
            part_name = str(ids_to_labels[raw_id])
            if part_name == 'background':
                continue
            pm = (np.asarray(seg, dtype=np.int32) == raw_id) & np.asarray(eval_mask, dtype=bool)
            pix = int(pm.sum())
            if pix < MIN_PART_PIXELS:
                rows.append({'frame_name': frame_name, 'part_id': raw_id, 'part_name': part_name, 'status': 'skipped_small_region', 'pixel_count': pix})
                continue
            cov = float(pred_mask[pm].mean()) if pix > 0 else float('nan')
            row = {'frame_name': frame_name, 'part_id': raw_id, 'part_name': part_name, 'group_id': int(raw_to_group[raw_id]) if raw_id < int(raw_to_group.shape[0]) else 3, 'group_name': str(FASHN_COARSE_GROUPS[int(raw_to_group[raw_id])]) if raw_id < int(raw_to_group.shape[0]) else 'unknown', 'status': 'ok', 'pixel_count': pix, 'mae': mae(pred, gt, pm), 'psnr': psnr(pred, gt, pm), 'ssim': ssim(pred, gt, pm), 'render_coverage_ratio': cov}
            row['badness_score'] = float(row['mae'] + (1.0 - row['ssim']) + (1.0 - row['render_coverage_ratio']))
            rows.append(row)
            all_part_rows.append(row)
        worst = sorted([r for r in rows if r.get('status') == 'ok'], key=lambda x: float(x['badness_score']), reverse=True)[:3]
        badlog.write(f"[{frame_name}] worst_regions: " + '; '.join(f"{r['part_name']}(badness={r['badness_score']:.6f}, mae={r['mae']:.6f}, ssim={r['ssim']:.6f}, coverage={r['render_coverage_ratio']:.6f})" for r in worst) + '\n')
        badlog.flush()
        pred_u8 = np.clip(pred * 255.0, 0.0, 255.0).astype(np.uint8)
        stem = Path(frame_name).stem
        if bool(args.face_only_render) and face_raw_id is not None:
            face_mask = np.asarray(seg == int(face_raw_id), dtype=bool)
            face_view = face_only_side_by_side(np.asarray(gt_u8, dtype=np.uint8), pred_u8, face_mask, pad=int(args.face_only_pad_px))
            if face_view is not None:
                face_path = out_dir / 'face_only' / f'{stem}_face.png'
                save_u8(face_path, face_view)
            else:
                face_path = None
        else:
            dmap = diff_map(pred, gt, np.asarray(eval_mask, dtype=bool))
            seg_color = np.zeros(seg.shape + (3,), dtype=np.uint8)
            valid = (seg >= 0) & (seg < int(RAW_PALETTE.shape[0]))
            if bool(valid.any()):
                seg_color[valid] = RAW_PALETTE[seg[valid].astype(np.int64)]
            gt_overlay = (0.55 * np.asarray(gt_u8, dtype=np.float32) + 0.45 * seg_color.astype(np.float32)).clip(0, 255).astype(np.uint8)
            top = np.concatenate([cv2.cvtColor(gt_u8, cv2.COLOR_RGB2BGR), cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR)], axis=1)
            bottom = np.concatenate([cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR), dmap], axis=1)
            overlay = np.concatenate([top, bottom], axis=0)
            overlay_path = out_dir / 'overlays' / f'{stem}_overlay.png'
            save_u8(overlay_path, overlay)
            face_path = None
        rep = {'frame_index': i, 'frame_name': frame_name, 'global_metrics': gm, 'frame_badness': frame_badness, 'bind_valid_ratio': bind_valid_ratio, 'worst_parts': worst, 'part_metrics': rows, 'outputs': {'face_only': str(face_path)} if bool(args.face_only_render) else {'overlay': str(overlay_path)}}
        frame_reports.append(rep)
        jsonl.write(json.dumps(jfix(rep), ensure_ascii=False) + '\n'); jsonl.flush()
        logger.info('Frame %s done | mae=%.6f | ssim=%.6f | mask_iou=%.6f | worst=%s', frame_name, float(gm['mae']), float(gm['ssim']), float(gm['iou']), ', '.join(str(x['part_name']) for x in worst) if worst else 'none')

    jsonl.close()
    part_summary = {}
    grouped = {}
    for row in all_part_rows:
        grouped.setdefault(str(row['part_name']), []).append(row)
    for name, rows in grouped.items():
        bad = np.asarray([float(r['badness_score']) for r in rows], dtype=np.float32)
        mae_arr = np.asarray([float(r['mae']) for r in rows], dtype=np.float32)
        ssim_arr = np.asarray([float(r['ssim']) for r in rows], dtype=np.float32)
        cov_arr = np.asarray([float(r['render_coverage_ratio']) for r in rows], dtype=np.float32)
        gid = int(rows[0]['group_id']) if rows else 3
        part_summary[name] = {'part_id': int(rows[0]['part_id']) if rows else -1, 'group_id': gid, 'group_name': str(FASHN_COARSE_GROUPS[gid]), 'frame_hits': len(rows), 'pixel_count_total': int(sum(int(r['pixel_count']) for r in rows)), 'badness_mean': float(bad.mean()), 'badness_p90': float(np.percentile(bad, 90)), 'mae_mean': float(mae_arr.mean()), 'ssim_mean': float(ssim_arr.mean()), 'render_coverage_ratio_mean': float(cov_arr.mean())}
    badlog.write('\n=== Overall Worst Parts ===\n')
    for name, row in sorted(part_summary.items(), key=lambda kv: float(kv[1]['badness_mean']), reverse=True)[:10]:
        badlog.write(f"{name}: badness_mean={row['badness_mean']:.6f}, mae_mean={row['mae_mean']:.6f}, ssim_mean={row['ssim_mean']:.6f}, coverage_mean={row['render_coverage_ratio_mean']:.6f}\n")
    badlog.close()

    worst_frames = sorted(frame_reports, key=lambda x: float(x['frame_badness']), reverse=True)
    worst_part_name = next(iter(sorted(part_summary.items(), key=lambda kv: float(kv[1]['badness_mean']), reverse=True)), ('none', {}))[0]
    gm_mae = np.asarray([float(r['global_metrics']['mae']) for r in frame_reports], dtype=np.float32)
    gm_psnr = np.asarray([float(r['global_metrics']['psnr']) for r in frame_reports], dtype=np.float32)
    gm_ssim = np.asarray([float(r['global_metrics']['ssim']) for r in frame_reports], dtype=np.float32)
    gm_iou = np.asarray([float(r['global_metrics']['iou']) for r in frame_reports], dtype=np.float32)
    summary = {'frame_count': len(frame_reports), 'target_layer': str(args.target_layer), 'stage1_dir': str(stage1_dir), 'stage2_dir': str(stage2_dir), 'stage2_ckpt': str(ckpt_path), 'supervision_cache': str(sup_path), 'semantics_frame_match': bool(sem_match), 'global_metrics': {'mae_mean': float(gm_mae.mean()), 'psnr_mean': float(gm_psnr[np.isfinite(gm_psnr)].mean()) if bool(np.isfinite(gm_psnr).any()) else None, 'ssim_mean': float(gm_ssim.mean()), 'mask_iou_mean': float(gm_iou.mean())}, 'worst_part_name': str(worst_part_name), 'worst_frames': [{'frame_name': r['frame_name'], 'frame_badness': float(r['frame_badness']), 'worst_parts': [str(x['part_name']) for x in r['worst_parts']]} for r in worst_frames[:10]], 'part_summary': part_summary}
    (out_dir / 'summary.json').write_text(json.dumps(jfix(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(out_dir / 'report.md', summary, part_summary, worst_frames)
    logger.info('Stage2 eval done: %s', out_dir / 'summary.json')


if __name__ == '__main__':
    main()
