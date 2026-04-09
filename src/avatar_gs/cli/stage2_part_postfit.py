
from __future__ import annotations

import argparse
import json
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
from avatar_gs.segmentation.fashn import get_fashn_label_maps, load_fashn_parser, predict_fashn_segmentation
from avatar_gs.stage2.fashn_semantics import load_stage2_face_semantics
from avatar_gs.stage2.mesh_runtime import Stage2MeshRuntime
from avatar_gs.stage2.part_fit_common import (
    setup_logger,
    clone_g,
    axis_angle_to_rotmat,
    load_stage1_kmap,
    load_supervision_align,
    bbox_from_mask,
    render_box_resized,
    resize_crop_rgb,
    resize_crop_mask,
    project_points_torch,
    sample_rgb_at_uv,
    sample_mask_at_uv,
    masked_l1,
    weighted_masked_l1,
    build_detail_weight,
    high_freq,
    symmetry_mae,
    logit,
    save_rgb,
    save_compare,
    psnr,
    masked_ssim,
)
from avatar_gs.segmentation.smpl_face_semantics import resolve_frame_image_path, load_stage1_vertices_sequence, rasterize_visible_face_index_map


@dataclass
class PartTarget:
    part_name: str
    frame_name: str
    local_frame_idx: int
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    box_xyxy: tuple[int, int, int, int]
    gt_crop: torch.Tensor
    gt_mask_crop: torch.Tensor
    detail_weight_crop: torch.Tensor
    pose_xyz: torch.Tensor
    pose_quat: torch.Tensor
    pose_log_scales: torch.Tensor
    surface_rgb: torch.Tensor
    surface_valid: torch.Tensor
    gaussian_local_indices: torch.Tensor
    teacher_role: str


def main() -> None:
    ap = argparse.ArgumentParser('Part-teacher postfit for inner gaussians')
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--stage2_ckpt', required=True)
    ap.add_argument('--teacher_bank_json', required=True)
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--frames_dir', default='data/frames/input')
    ap.add_argument('--smpl_face_semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    ap.add_argument('--target_source', default='inner_rgb')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--out_size', type=int, default=512)
    ap.add_argument('--lr_rgb', type=float, default=0.03)
    ap.add_argument('--lr_opacity', type=float, default=0.01)
    ap.add_argument('--lr_scale', type=float, default=0.005)
    ap.add_argument('--min_part_teachers', type=int, default=1)
    ap.add_argument('--cloth_min_support', type=int, default=2)
    ap.add_argument('--cloth_core_support', type=int, default=3)
    ap.add_argument('--cloth_core_erode_px', type=int, default=0)
    ap.add_argument('--face_teacher_override', default='')
    ap.add_argument('--face_force_single_teacher', action='store_true', default=False)
    ap.add_argument('--fixed_source_ckpt', default='')
    ap.add_argument('--fixed_parts', nargs='*', default=[])
    ap.add_argument('--train_parts', nargs='*', default=[])
    ap.add_argument('--fixed_box_frame', default='')
    ap.add_argument('--fixed_box_pad_px', type=int, default=40)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve() if not Path(args.stage1_dir).is_absolute() else Path(args.stage1_dir).resolve()
    ckpt_path = (project_root / args.stage2_ckpt).resolve() if not Path(args.stage2_ckpt).is_absolute() else Path(args.stage2_ckpt).resolve()
    teacher_bank_path = (project_root / args.teacher_bank_json).resolve() if not Path(args.teacher_bank_json).is_absolute() else Path(args.teacher_bank_json).resolve()
    supervision_cache = (project_root / args.supervision_cache).resolve() if not Path(args.supervision_cache).is_absolute() else Path(args.supervision_cache).resolve()
    frames_dir = (project_root / args.frames_dir).resolve() if not Path(args.frames_dir).is_absolute() else Path(args.frames_dir).resolve()
    semantics_path = (project_root / args.smpl_face_semantics_npz).resolve() if not Path(args.smpl_face_semantics_npz).is_absolute() else Path(args.smpl_face_semantics_npz).resolve()

    tag = str(args.tag).strip() or f"part_teacher_postfit_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = (project_root / 'outputs' / 'stage2' / 'part_fit' / tag).resolve()
    train_dir = (out_dir / 'train').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / 'part_fit.log')
    logger.info('start part-teacher postfit: %s', tag)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device=%s', device)

    teacher_bank = json.loads(teacher_bank_path.read_text(encoding='utf-8'))
    teacher_bank_map = teacher_bank['teacher_bank']

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    g_base = Gaussian3D.load_state_dict(ckpt['gaussians']).to(str(device))
    sanitize_inner_surface_binding_(g_base)
    fixed_ckpt_path = str(args.fixed_source_ckpt).strip()
    g_fixed = None
    if fixed_ckpt_path:
        fixed_path = (project_root / fixed_ckpt_path).resolve() if not Path(fixed_ckpt_path).is_absolute() else Path(fixed_ckpt_path).resolve()
        fixed_ckpt = torch.load(str(fixed_path), map_location='cpu')
        g_fixed = Gaussian3D.load_state_dict(fixed_ckpt['gaussians']).to(str(device))
        sanitize_inner_surface_binding_(g_fixed)
        if int(g_fixed.xyz.shape[0]) != int(g_base.xyz.shape[0]):
            raise RuntimeError('fixed_source_ckpt gaussian count mismatch')
    shell = g_base.shell_layer if g_base.shell_layer is not None else torch.zeros((int(g_base.xyz.shape[0]),), dtype=torch.int64, device=device)
    eff = g_base.effective_face_idx
    if eff is None:
        raise RuntimeError('checkpoint has no effective_face_idx')
    eff_np = eff.detach().cpu().numpy().astype(np.int64)
    inner_mask_np = shell.detach().cpu().numpy().astype(np.int64) == 0

    stage1_idx, kmap = load_stage1_kmap(stage1_dir)
    sup = np.load(str(supervision_cache), allow_pickle=True)
    sup_names = [str(x) for x in np.asarray(sup['frame_names']).reshape(-1).tolist()]
    sup_idx = {n: i for i, n in enumerate(sup_names)}
    inner_rgb_targets = np.asarray(sup['inner_rgb_targets'], dtype=np.uint8)
    inner_masks = np.asarray(sup['inner_masks'], dtype=np.uint8)

    semantics = load_stage2_face_semantics(semantics_path)
    raw_label_names = [str(x) for x in semantics.raw_label_names]
    raw_label_ids = np.asarray(semantics.raw_label_ids, dtype=np.int64)
    semantics_npz = np.load(str(semantics_path), allow_pickle=True)
    faces_np = np.asarray(semantics_npz['faces'], dtype=np.int32)

    # Map semantic part to gaussian subset.
    part_gaussian_masks: dict[str, np.ndarray] = {}
    for part_name in raw_label_names:
        if part_name == 'background':
            continue
        if part_name not in raw_label_names:
            continue
        part_id = raw_label_names.index(part_name)
        valid_eff = (eff_np >= 0) & (eff_np < int(raw_label_ids.shape[0]))
        m = inner_mask_np.copy() & valid_eff
        m[valid_eff] &= (raw_label_ids[eff_np[valid_eff]] == part_id)
        part_gaussian_masks[part_name] = m

    fixed_parts = [str(x) for x in (args.fixed_parts or [])]
    train_parts = {str(x) for x in (args.train_parts or []) if str(x).strip()}
    active_train_mask_np = np.zeros((int(g_base.xyz.shape[0]),), dtype=bool)
    fixed_extra_mask_np = np.zeros((int(g_base.xyz.shape[0]),), dtype=bool)
    # keep only parts with teachers and non-empty gaussian subset
    active_parts = {}
    for part_name, payload in teacher_bank_map.items():
        if train_parts and part_name not in train_parts:
            continue
        teachers = [dict(x) for x in list(payload.get('teachers', []))]
        if part_name == 'face' and str(args.face_teacher_override).strip():
            face_name = str(args.face_teacher_override).strip()
            forced = None
            for item in teachers:
                if str(item.get('frame_name', '')) == face_name:
                    forced = dict(item)
                    break
            if forced is None:
                forced = {'frame_name': face_name, 'score': None, 'override': True, 'teacher_role': 'base'}
            else:
                forced['teacher_role'] = 'base'
            teachers = [forced]
        elif part_name == 'face' and bool(args.face_force_single_teacher) and len(teachers) > 0:
            teachers = [dict(teachers[0])]
            teachers[0]['teacher_role'] = 'base'
        if len(teachers) < int(args.min_part_teachers):
            continue
        gm = part_gaussian_masks.get(part_name, None)
        if gm is None or int(np.asarray(gm, dtype=np.uint8).sum()) <= 0:
            continue
        base_teachers = [t for t in teachers if str(t.get('teacher_role', 'base')) == 'base']
        extra_teachers = [t for t in teachers if str(t.get('teacher_role', 'base')) == 'extra']
        if len(base_teachers) < int(args.min_part_teachers):
            base_teachers = teachers[:]
            extra_teachers = []
        active_parts[part_name] = {'base': base_teachers, 'extra': extra_teachers}
    for part_name in fixed_parts:
        if part_name in active_parts:
            del active_parts[part_name]
    for part_name in active_parts.keys():
        active_train_mask_np |= part_gaussian_masks[part_name]
    logger.info('active teacher parts=%s', ', '.join(sorted(active_parts.keys())))
    logger.info('fixed parts from fixed_source_ckpt=%s', ', '.join(fixed_parts) if fixed_parts else '(none)')
    logger.info('train parts=%s', ', '.join(sorted(train_parts)) if train_parts else '(all active teacher parts)')

    all_selected_mask_np = np.zeros((int(g_base.xyz.shape[0]),), dtype=bool)
    for part_name in active_parts.keys():
        all_selected_mask_np |= part_gaussian_masks[part_name]
    if bool(fixed_extra_mask_np.any()):
        all_selected_mask_np &= ~fixed_extra_mask_np
    selected_mask = torch.from_numpy(all_selected_mask_np).to(device=device, dtype=torch.bool)
    logger.info('selected gaussians=%d / total=%d', int(selected_mask.sum().item()), int(selected_mask.numel()))

    # local index map from global selected subset
    global_to_local = -np.ones((int(g_base.xyz.shape[0]),), dtype=np.int64)
    sel_idx_np = np.flatnonzero(all_selected_mask_np).astype(np.int64)
    global_to_local[sel_idx_np] = np.arange(sel_idx_np.shape[0], dtype=np.int64)

    teacher_frames_unique = []
    for payload in active_parts.values():
        for teacher_group in (payload['base'], payload['extra']):
            for item in teacher_group:
                fn = str(item['frame_name'])
                if fn not in teacher_frames_unique:
                    teacher_frames_unique.append(fn)
    scales, txs, tys, accepted = load_supervision_align(supervision_cache, teacher_frames_unique)
    local_frame_map = {fn: i for i, fn in enumerate(teacher_frames_unique)}

    mesh_runtime = Stage2MeshRuntime.from_stage2_config(
        project_root=project_root,
        stage1_dir=stage1_dir,
        cfg=dict(ckpt),
        frames=teacher_frames_unique,
        logger=logger,
        device=str(device),
        bind_to_mesh=True,
        auto_from_stage2_report=False,
        stage2_dir_for_report=None,
    )
    mesh_runtime.set_frame_smpl_align_from_arrays(frame_names=teacher_frames_unique, scales=scales, tx_px=txs, ty_px=tys, accepted=accepted)

    binding_cfg = parse_binding_mode_config(ckpt.get('binding', None), default_mode='games_face')
    bind_mask = (g_base.bind_idx.to(device=device, dtype=torch.long) >= 0) if g_base.bind_idx is not None else None
    if bind_mask is None or not bool(bind_mask.any()):
        raise RuntimeError('checkpoint has no valid bind_idx')
    bind_idx_t = g_base.bind_idx.to(device=device, dtype=torch.long)
    bind_offset_t = g_base.bind_offset.to(device=device, dtype=torch.float32) if g_base.bind_offset is not None else None
    bind_face_idx_t = g_base.bind_face_idx.to(device=device, dtype=torch.long) if g_base.bind_face_idx is not None else None
    bind_bary_t = g_base.bind_bary.to(device=device, dtype=torch.float32) if g_base.bind_bary is not None else None
    quat_blend = float(ckpt.get('bind_quat_normal_blend', 1.0))

    fixed_box_frame = str(args.fixed_box_frame).strip()
    if fixed_box_frame:
        if fixed_box_frame not in stage1_idx:
            raise RuntimeError(f'fixed_box_frame missing stage1 data: {fixed_box_frame}')
        scales_box, txs_box, tys_box, accepted_box = load_supervision_align(supervision_cache, [fixed_box_frame])
        box_runtime = Stage2MeshRuntime.from_stage2_config(
            project_root=project_root,
            stage1_dir=stage1_dir,
            cfg=dict(ckpt),
            frames=[fixed_box_frame],
            logger=logger,
            device=str(device),
            bind_to_mesh=True,
            auto_from_stage2_report=False,
            stage2_dir_for_report=None,
        )
        box_runtime.set_frame_smpl_align_from_arrays(frame_names=[fixed_box_frame], scales=scales_box, tx_px=txs_box, ty_px=tys_box, accepted=accepted_box)
        K_box = np.asarray(kmap[fixed_box_frame], dtype=np.float32)
        verts_names_box, verts_cam_all_box = load_stage1_vertices_sequence(stage1_dir)
        verts_idx_box = {n: i for i, n in enumerate(verts_names_box)}
        image_path_box = resolve_frame_image_path(frames_dir, fixed_box_frame)
        bgr_box = cv2.imread(str(image_path_box), cv2.IMREAD_COLOR)
        if bgr_box is None:
            raise RuntimeError(f'failed to read fixed_box_frame image: {image_path_box}')
        raw_rgb_box = cv2.cvtColor(bgr_box, cv2.COLOR_BGR2RGB)
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=np.asarray(verts_cam_all_box[int(verts_idx_box[fixed_box_frame])], dtype=np.float32),
            faces=faces_np,
            K=K_box,
            image_hw=raw_rgb_box.shape[:2],
        )
        pix_valid = face_index_map >= 0
        face_mask_2d = np.zeros(face_index_map.shape[:2], dtype=np.uint8)
        if bool(pix_valid.any()):
            fids = face_index_map[pix_valid].astype(np.int64)
            face_part_id = int(raw_label_names.index('face'))
            hair_part_id = int(raw_label_names.index('hair')) if 'hair' in raw_label_names else -1
            is_head = (raw_label_ids[fids] == face_part_id)
            if hair_part_id >= 0:
                is_head |= (raw_label_ids[fids] == hair_part_id)
            face_mask_2d[pix_valid] = is_head.astype(np.uint8)
        x0, y0, x1, y1 = bbox_from_mask(face_mask_2d)
        pad = int(args.fixed_box_pad_px)
        x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0); x1 = min(x1 + pad, raw_rgb_box.shape[1] - 1); y1 = min(y1 + pad, raw_rgb_box.shape[0] - 1)
        box_frame = box_runtime.load_frame_binding_frame(0, K_np=K_box, binding_cfg_dict=ckpt.get('binding', None), bind_cache_size=8)
        g_box = clone_g(g_base)
        apply_inner_binding_per_frame(
            g=g_box,
            frame=box_frame,
            bind_mask=bind_mask,
            bind_vertex_idx=bind_idx_t,
            bind_vertex_offset=bind_offset_t,
            bind_face_idx=bind_face_idx_t,
            bind_bary=bind_bary_t,
            cfg=binding_cfg,
            quat_blend=quat_blend,
            prev_vertex_dirs=None,
        )
        xyz = g_box.xyz.detach().cpu().numpy().astype(np.float32)
        xyz_cam = xyz
        z = xyz_cam[:, 2]
        validz = z > 1e-6
        u = K_box[0, 0] * (xyz_cam[:, 0] / np.clip(z, 1e-6, None)) + K_box[0, 2]
        v = K_box[1, 1] * (xyz_cam[:, 1] / np.clip(z, 1e-6, None)) + K_box[1, 2]
        inside = validz & (u >= float(x0)) & (u <= float(x1)) & (v >= float(y0)) & (v <= float(y1))
        fixed_extra_mask_np |= (inside & inner_mask_np & (~active_train_mask_np))
        logger.info('fixed box frame=%s box=(%d,%d,%d,%d) fixed_extra=%d active_train_overlap_blocked=%d', fixed_box_frame, x0, y0, x1, y1, int(fixed_extra_mask_np.sum()), int((inside & inner_mask_np & active_train_mask_np).sum()))


    ids_to_labels, labels_to_ids = get_fashn_label_maps()
    parser = load_fashn_parser(device='cuda')
    part_targets: list[PartTarget] = []
    seg_cache = {}
    verts_names_all, verts_cam_all = load_stage1_vertices_sequence(stage1_dir)
    verts_idx_all = {str(n): i for i, n in enumerate(verts_names_all)}
    projected_label_cache: dict[str, np.ndarray] = {}

    cloth_like_parts = {'top', 'pants', 'torso', 'dress', 'skirt'}
    consistency_parts = {'top', 'pants', 'torso', 'dress', 'skirt', 'arms', 'hands', 'feet'}

    for part_name, teacher_payload in active_parts.items():
        part_id = int(labels_to_ids[part_name])
        part_global_mask_np = part_gaussian_masks[part_name]
        part_global_idx = np.flatnonzero(part_global_mask_np).astype(np.int64)
        part_local_idx = torch.from_numpy(global_to_local[part_global_idx]).to(device=device, dtype=torch.long)
        for teacher_role, teachers in (('base', teacher_payload['base']), ('extra', teacher_payload['extra'])):
            for item in teachers:
                frame_name = str(item['frame_name'])
                local_idx = int(local_frame_map[frame_name])
                if frame_name not in stage1_idx:
                    continue
                if frame_name not in sup_idx:
                    continue
                image_path = resolve_frame_image_path(frames_dir, frame_name)
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                raw_rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                target_source = str(args.target_source).strip().lower()
                if target_source == 'inner_rgb':
                    rgb_u8 = np.asarray(inner_rgb_targets[int(sup_idx[frame_name])], dtype=np.uint8)
                    inner_mask_full = np.asarray(inner_masks[int(sup_idx[frame_name])] > 0, dtype=np.uint8)
                else:
                    rgb_u8 = raw_rgb_u8
                    inner_mask_full = np.ones(raw_rgb_u8.shape[:2], dtype=np.uint8)
                if frame_name not in seg_cache:
                    seg_cache[frame_name] = predict_fashn_segmentation(rgb_u8, parser=parser)
                seg = np.asarray(seg_cache[frame_name], dtype=np.int32)
                if frame_name not in projected_label_cache:
                    if frame_name not in verts_idx_all:
                        projected_label_cache[frame_name] = np.full(rgb_u8.shape[:2], fill_value=-1, dtype=np.int32)
                    else:
                        face_index_map, _ = rasterize_visible_face_index_map(
                            verts_cam=np.asarray(verts_cam_all[int(verts_idx_all[frame_name])], dtype=np.float32),
                            faces=faces_np,
                            K=np.asarray(kmap[frame_name], dtype=np.float32),
                            image_hw=rgb_u8.shape[:2],
                        )
                        valid = face_index_map >= 0
                        projected_labels = np.full(face_index_map.shape, fill_value=-1, dtype=np.int32)
                        if bool(valid.any()):
                            fids = face_index_map[valid].astype(np.int64)
                            projected_labels[valid] = raw_label_ids[fids]
                        projected_label_cache[frame_name] = projected_labels
                projected_labels = np.asarray(projected_label_cache[frame_name], dtype=np.int32)
                seg_mask = np.asarray(seg == part_id, dtype=bool)
                proj_mask = np.asarray(projected_labels == part_id, dtype=bool)
                if part_name in consistency_parts:
                    part_mask_bool = seg_mask & proj_mask & (inner_mask_full > 0)
                else:
                    part_mask_bool = seg_mask & (inner_mask_full > 0)
                part_mask_2d = np.asarray(part_mask_bool, dtype=np.uint8)
                if part_name in cloth_like_parts and int(args.cloth_core_erode_px) > 0 and int(part_mask_2d.sum()) > 64:
                    k = max(int(args.cloth_core_erode_px), 1)
                    kernel = np.ones((k, k), dtype=np.uint8)
                    eroded = cv2.erode(part_mask_2d, kernel, iterations=1)
                    if int(eroded.sum()) > 128:
                        part_mask_2d = eroded
                if int(part_mask_2d.sum()) <= 32:
                    continue
                box = bbox_from_mask(part_mask_2d)
                gt_crop_u8 = resize_crop_rgb(rgb_u8, box, int(args.out_size))
                gt_mask_u8 = resize_crop_mask(part_mask_2d, box, int(args.out_size))
                detail_weight_full = build_detail_weight(rgb_u8, part_mask_2d)
                detail_weight_u8 = resize_crop_rgb(np.repeat((detail_weight_full * 255.0).astype(np.uint8)[:, :, None], 3, axis=2), box, int(args.out_size))
                gt_crop = torch.from_numpy(gt_crop_u8.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
                gt_mask = torch.from_numpy(gt_mask_u8.astype(np.float32)).unsqueeze(0).to(device)
                detail_weight_crop = torch.from_numpy(detail_weight_u8[:, :, 0].astype(np.float32) / 255.0).unsqueeze(0).to(device)

                g_bound = clone_g(g_base)
                frame = mesh_runtime.load_frame_binding_frame(local_idx, K_np=np.asarray(kmap[frame_name], dtype=np.float32), binding_cfg_dict=ckpt.get('binding', None), bind_cache_size=8)
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
                pose_xyz = g_bound.xyz[selected_mask][part_local_idx].detach()
                pose_quat = F.normalize(g_bound.quat[selected_mask][part_local_idx].detach(), dim=1, eps=1e-8)
                pose_log_scales = g_bound.log_scales[selected_mask][part_local_idx].detach()

                ckpt_frames = [str(x) for x in ckpt.get('frames', teacher_frames_unique)]
                ckpt_idx = {n: i for i, n in enumerate(ckpt_frames)}
                cam_w = torch.as_tensor(ckpt.get('cam_w', None), dtype=torch.float32, device=device) if ckpt.get('cam_w', None) is not None else None
                cam_t = torch.as_tensor(ckpt.get('cam_t', None), dtype=torch.float32, device=device) if ckpt.get('cam_t', None) is not None else None
                if cam_w is None or cam_t is None or frame_name not in ckpt_idx:
                    R = torch.eye(3, dtype=torch.float32, device=device)
                    t = torch.zeros((3,), dtype=torch.float32, device=device)
                else:
                    j = ckpt_idx[frame_name]
                    R = axis_angle_to_rotmat(cam_w[j])
                    t = cam_t[j]
                K_t = torch.from_numpy(np.asarray(kmap[frame_name], dtype=np.float32)).to(device=device, dtype=torch.float32)
                rgb_full_t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
                part_mask_full_t = torch.from_numpy(part_mask_2d.astype(np.float32)).to(device)
                u, v, valid = project_points_torch(pose_xyz, K_t, R, t)
                valid = valid & (u >= 0.0) & (u <= float(rgb_u8.shape[1] - 1)) & (v >= 0.0) & (v <= float(rgb_u8.shape[0] - 1))
                visible_part = sample_mask_at_uv(part_mask_full_t, u, v, valid) > 0.5
                sampled_rgb = sample_rgb_at_uv(rgb_full_t, u, v, valid & visible_part)
                part_targets.append(PartTarget(
                    part_name=part_name,
                    frame_name=frame_name,
                    local_frame_idx=local_idx,
                    K=K_t,
                    R=R,
                    t=t,
                    box_xyxy=box,
                    gt_crop=gt_crop,
                    gt_mask_crop=gt_mask,
                    detail_weight_crop=detail_weight_crop,
                    pose_xyz=pose_xyz,
                    pose_quat=pose_quat,
                    pose_log_scales=pose_log_scales,
                    surface_rgb=sampled_rgb.detach(),
                    surface_valid=(valid & visible_part).detach(),
                    gaussian_local_indices=part_local_idx,
                    teacher_role=teacher_role,
                ))
                logger.info('target part=%s role=%s frame=%s part_pixels=%d visible_part_gauss=%d', part_name, teacher_role, frame_name, int(part_mask_2d.sum()), int((valid & visible_part).sum().item()))

    if len(part_targets) <= 0:
        raise RuntimeError('no usable part targets built from teacher bank')

    selected_global_idx = torch.from_numpy(sel_idx_np).to(device=device, dtype=torch.long)
    base_support_counts = torch.zeros((int(selected_global_idx.numel()),), dtype=torch.int32, device=device)
    any_support_counts = torch.zeros((int(selected_global_idx.numel()),), dtype=torch.int32, device=device)
    target_visible_cache: dict[int, torch.Tensor] = {}
    for tgt in part_targets:
        if int(tgt.gaussian_local_indices.numel()) <= 0:
            continue
        visible_local = tgt.gaussian_local_indices[tgt.surface_valid]
        if int(visible_local.numel()) <= 0:
            continue
        visible_local = torch.unique(visible_local)
        target_visible_cache[id(tgt)] = visible_local
        any_support_counts[visible_local] += 1
        if str(tgt.teacher_role) == 'base':
            base_support_counts[visible_local] += 1

    keep_local_mask = any_support_counts >= 1
    if not bool(keep_local_mask.any()):
        raise RuntimeError('no visible gaussians after teacher visibility pass')
    min_support = max(int(args.cloth_min_support), 1)
    core_support = max(int(args.cloth_core_support), min_support + 1)
    base_visible_mask = base_support_counts >= 1
    base_deficit_mask = (base_support_counts < min_support) & keep_local_mask

    base_targets_all = [t for t in part_targets if str(t.teacher_role) != 'extra']
    extra_targets_all = [t for t in part_targets if str(t.teacher_role) == 'extra']
    target_part_names = {str(t.part_name) for t in part_targets}
    active_cloth_mask_np = np.zeros((int(g_base.xyz.shape[0]),), dtype=bool)
    for _pn in target_part_names:
        if _pn in cloth_like_parts:
            _gm = part_gaussian_masks.get(_pn, None)
            if _gm is not None:
                active_cloth_mask_np |= np.asarray(_gm, dtype=bool)
    cloth_selected_mask = torch.from_numpy(active_cloth_mask_np[selected_global_idx.detach().cpu().numpy()]).to(device=device, dtype=torch.bool)
    use_cloth_core_mask = bool(cloth_selected_mask.any())
    base_core_mask = torch.where(cloth_selected_mask, (base_support_counts >= core_support) & keep_local_mask, keep_local_mask)
    selected_extra_targets: list[PartTarget] = []
    support_counts = base_support_counts.clone()
    remaining_deficit = (support_counts < min_support) & keep_local_mask
    extra_pool = list(extra_targets_all)
    greedy_log = []
    while bool(remaining_deficit.any()) and extra_pool:
        best_idx = -1
        best_gain = 0
        best_vis = None
        for idx, tgt in enumerate(extra_pool):
            visible_local = target_visible_cache.get(id(tgt), None)
            if visible_local is None or int(visible_local.numel()) <= 0:
                continue
            gain = int(remaining_deficit[visible_local].sum().item())
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_vis = visible_local
        if best_idx < 0 or best_gain <= 0 or best_vis is None:
            break
        picked = extra_pool.pop(best_idx)
        selected_extra_targets.append(picked)
        support_counts[best_vis] += 1
        remaining_deficit = (support_counts < min_support) & keep_local_mask
        greedy_log.append((picked.frame_name, int(best_gain)))

    final_targets = base_targets_all + selected_extra_targets
    final_any_support_counts = torch.zeros((int(selected_global_idx.numel()),), dtype=torch.int32, device=device)
    extra_keep_mask = torch.zeros((int(selected_global_idx.numel()),), dtype=torch.bool, device=device)
    for tgt in final_targets:
        visible_local = target_visible_cache.get(id(tgt), None)
        if visible_local is None or int(visible_local.numel()) <= 0:
            continue
        final_any_support_counts[visible_local] += 1
        if str(tgt.teacher_role) == 'extra':
            extra_keep_mask[visible_local] = True
    keep_local_mask = torch.where(cloth_selected_mask, base_core_mask | (extra_keep_mask & base_deficit_mask), final_any_support_counts >= 1)
    logger.info('support stats | any_visible=%d / %d | base_visible=%d | base_core=%d | base_supported=%d | base_deficit=%d | selected_extra=%d | remain_deficit=%d | min_support=%d | core_support=%d | use_cloth_core=%s', int(keep_local_mask.sum().item()), int(selected_global_idx.numel()), int(base_visible_mask.sum().item()), int((base_core_mask & keep_local_mask).sum().item()), int(((base_support_counts >= min_support) & keep_local_mask).sum().item()), int((base_deficit_mask & keep_local_mask).sum().item()), int(len(selected_extra_targets)), int((remaining_deficit & keep_local_mask).sum().item()), int(min_support), int(core_support), str(use_cloth_core_mask).lower())
    if greedy_log:
        logger.info('extra greedy picks=%s', greedy_log)

    keep_local_idx = torch.nonzero(keep_local_mask, as_tuple=False).squeeze(1)
    old_to_new = -torch.ones((int(selected_global_idx.numel()),), dtype=torch.long, device=device)
    old_to_new[keep_local_idx] = torch.arange(int(keep_local_idx.numel()), device=device, dtype=torch.long)
    selected_mask_reduced = torch.zeros_like(selected_mask)
    selected_mask_reduced[selected_global_idx[keep_local_idx]] = True

    filtered_targets: list[PartTarget] = []
    for tgt in final_targets:
        if str(tgt.teacher_role) == 'extra' and str(tgt.part_name) in cloth_like_parts:
            local_keep = extra_keep_mask[tgt.gaussian_local_indices] & base_deficit_mask[tgt.gaussian_local_indices]
        elif str(tgt.teacher_role) == 'extra':
            local_keep = keep_local_mask[tgt.gaussian_local_indices]
        elif str(tgt.part_name) in cloth_like_parts:
            local_keep = base_core_mask[tgt.gaussian_local_indices]
        else:
            local_keep = keep_local_mask[tgt.gaussian_local_indices]
        if not bool(local_keep.any()):
            continue
        new_local = old_to_new[tgt.gaussian_local_indices[local_keep]]
        vis = tgt.surface_valid[local_keep]
        filtered_targets.append(PartTarget(
            part_name=tgt.part_name,
            frame_name=tgt.frame_name,
            local_frame_idx=tgt.local_frame_idx,
            K=tgt.K,
            R=tgt.R,
            t=tgt.t,
            box_xyxy=tgt.box_xyxy,
            gt_crop=tgt.gt_crop,
            gt_mask_crop=tgt.gt_mask_crop,
            detail_weight_crop=tgt.detail_weight_crop,
            pose_xyz=tgt.pose_xyz[local_keep],
            pose_quat=tgt.pose_quat[local_keep],
            pose_log_scales=tgt.pose_log_scales[local_keep],
            surface_rgb=tgt.surface_rgb[local_keep],
            surface_valid=vis,
            gaussian_local_indices=new_local,
            teacher_role=tgt.teacher_role,
        ))
    part_targets = filtered_targets
    selected_mask = selected_mask_reduced
    base_targets = [t for t in part_targets if str(t.teacher_role) != 'extra']
    extra_targets = [t for t in part_targets if str(t.teacher_role) == 'extra']
    use_extra_phase = bool(extra_targets)
    extra_start_iter = max(int((int(args.iters) * 4 + 4) // 5), 1)
    logger.info('filtered targets=%d after base/extra gating | base=%d extra=%d extra_phase=%s extra_start_iter=%d', len(part_targets), len(base_targets), len(extra_targets), str(use_extra_phase).lower(), int(extra_start_iter))

    sel_rgb_param = torch.nn.Parameter(g_base.rgb_logit[selected_mask].detach().clone())
    sel_opacity_param = torch.nn.Parameter(g_base.opacity_logit[selected_mask].detach().clone())
    sel_scale_delta = torch.nn.Parameter(torch.zeros((int(sel_rgb_param.shape[0]), 1), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([
        {'params': [sel_rgb_param], 'lr': float(args.lr_rgb)},
        {'params': [sel_opacity_param], 'lr': float(args.lr_opacity)},
        {'params': [sel_scale_delta], 'lr': float(args.lr_scale)},
    ])

    best_loss = float('inf')
    best_state = None
    history = []
    for it in range(1, int(args.iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        scale_bias = 0.55 * torch.tanh(sel_scale_delta)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        iter_targets = base_targets
        if use_extra_phase and it > int(extra_start_iter):
            iter_targets = base_targets + extra_targets
        for tgt in iter_targets:
            local_idx = tgt.gaussian_local_indices
            log_scales = (tgt.pose_log_scales + scale_bias[local_idx].repeat(1, 3)).clamp(min=-10.0, max=0.0)
            g_part = Gaussian3D(
                xyz=tgt.pose_xyz,
                log_scales=log_scales,
                quat=tgt.pose_quat,
                opacity_logit=sel_opacity_param[local_idx],
                rgb_logit=sel_rgb_param[local_idx],
            )
            pred_crop, alpha_crop = render_box_resized(g_part, tgt.K, tgt.R, tgt.t, tgt.box_xyxy, int(args.out_size), bg_rgb=(1.0, 1.0, 1.0))
            gt_crop = tgt.gt_crop
            gt_mask = tgt.gt_mask_crop
            detail_w = tgt.detail_weight_crop
            is_cloth = tgt.part_name in cloth_like_parts
            loss_rgb = weighted_masked_l1(pred_crop, gt_crop, gt_mask, detail_w if is_cloth else None)
            loss_edge = weighted_masked_l1(high_freq(pred_crop), high_freq(gt_crop), gt_mask, detail_w if is_cloth else None)
            loss_alpha = F.l1_loss(alpha_crop, gt_mask)
            if bool(tgt.surface_valid.any()):
                curr_surface_rgb = torch.sigmoid(sel_rgb_param[local_idx][tgt.surface_valid]).clamp(0.0, 1.0)
                loss_surface = F.l1_loss(curr_surface_rgb, tgt.surface_rgb[tgt.surface_valid])
            else:
                loss_surface = pred_crop.sum() * 0.0
            loss_sym = symmetry_mae(pred_crop, gt_mask) if tgt.part_name in {'top', 'pants', 'torso'} else pred_crop.sum() * 0.0
            if is_cloth and str(tgt.teacher_role) != 'extra':
                # Base cloth teachers define texture; bias toward high-frequency retention instead of smooth averaging.
                loss = 0.95 * loss_rgb + 1.35 * loss_edge + 0.20 * loss_alpha + 0.45 * loss_surface + 0.02 * loss_sym
            elif is_cloth:
                loss = 1.00 * loss_rgb + 0.75 * loss_edge + 0.20 * loss_alpha + 0.50 * loss_surface + 0.02 * loss_sym
            else:
                loss = 1.0 * loss_rgb + 0.45 * loss_edge + 0.30 * loss_alpha + 0.75 * loss_surface + 0.04 * loss_sym
            if str(tgt.teacher_role) == 'extra':
                loss = 0.45 * loss
            total_loss = total_loss + loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            sel_rgb_param.data.clamp_(-12.0, 12.0)
            sel_opacity_param.data.clamp_(-12.0, 12.0)
        total_value = float(total_loss.detach().item())
        history.append({'iter': int(it), 'loss': total_value})
        if total_value < best_loss:
            best_loss = total_value
            best_state = {
                'rgb_logit': sel_rgb_param.detach().clone(),
                'opacity_logit': sel_opacity_param.detach().clone(),
                'scale_delta': sel_scale_delta.detach().clone(),
                'iter': int(it),
                'loss': float(total_value),
            }
        if (it == 1) or (it % 50 == 0) or (it == int(args.iters)):
            logger.info('iter %04d/%04d | total=%.6f', it, int(args.iters), total_value)

    if best_state is None:
        raise RuntimeError('training produced no best_state')

    g_preview = clone_g(g_base)
    with torch.no_grad():
        g_preview.rgb_logit[selected_mask] = best_state['rgb_logit'].detach()
        g_preview.opacity_logit[selected_mask] = best_state['opacity_logit'].detach()
        g_preview.log_scales[selected_mask] = g_preview.log_scales[selected_mask] + 0.55 * torch.tanh(best_state['scale_delta']).repeat(1, 3).detach()
    if g_fixed is not None and (fixed_parts or bool(fixed_extra_mask_np.any())):
        with torch.no_grad():
            for part_name in fixed_parts:
                part_mask_np = part_gaussian_masks.get(part_name, None)
                if part_mask_np is None:
                    continue
                part_mask = torch.from_numpy(np.asarray(part_mask_np, dtype=bool)).to(device=device, dtype=torch.bool)
                if not bool(part_mask.any()):
                    continue
                g_preview.rgb_logit[part_mask] = g_fixed.rgb_logit[part_mask]
                g_preview.opacity_logit[part_mask] = g_fixed.opacity_logit[part_mask]
                g_preview.log_scales[part_mask] = g_fixed.log_scales[part_mask]
                g_preview.xyz[part_mask] = g_fixed.xyz[part_mask]
                g_preview.quat[part_mask] = g_fixed.quat[part_mask]
            if bool(fixed_extra_mask_np.any()):
                extra_mask = torch.from_numpy(np.asarray(fixed_extra_mask_np, dtype=bool)).to(device=device, dtype=torch.bool)
                g_preview.rgb_logit[extra_mask] = g_fixed.rgb_logit[extra_mask]
                g_preview.opacity_logit[extra_mask] = g_fixed.opacity_logit[extra_mask]
                g_preview.log_scales[extra_mask] = g_fixed.log_scales[extra_mask]
                g_preview.xyz[extra_mask] = g_fixed.xyz[extra_mask]
                g_preview.quat[extra_mask] = g_fixed.quat[extra_mask]
    out_ckpt = dict(ckpt)
    out_ckpt['gaussians'] = g_preview.state_dict()
    torch.save(out_ckpt, train_dir / 'gaussians3d_partfit_preview.pt')

    # Save a few compare images for inspection.
    preview_targets = part_targets[: min(12, len(part_targets))]
    for idx, tgt in enumerate(preview_targets):
        local_idx = tgt.gaussian_local_indices
        log_scales = (tgt.pose_log_scales + 0.55 * torch.tanh(best_state['scale_delta'])[local_idx].repeat(1, 3)).clamp(min=-10.0, max=0.0)
        g_part = Gaussian3D(
            xyz=tgt.pose_xyz,
            log_scales=log_scales,
            quat=tgt.pose_quat,
            opacity_logit=best_state['opacity_logit'][local_idx],
            rgb_logit=best_state['rgb_logit'][local_idx],
        )
        pred_crop, alpha_crop = render_box_resized(g_part, tgt.K, tgt.R, tgt.t, tgt.box_xyxy, int(args.out_size), bg_rgb=(1.0, 1.0, 1.0))
        stem = f"{idx:02d}_{tgt.part_name}_{Path(tgt.frame_name).stem}"
        save_rgb(out_dir / 'previews' / f'{stem}_pred.png', pred_crop)
        save_compare(out_dir / 'previews' / f'{stem}_compare.png', tgt.gt_crop, pred_crop, tgt.gt_mask_crop)

    summary = {
        'tag': tag,
        'target_source': str(args.target_source),
        'teacher_bank_json': str(teacher_bank_path),
        'fixed_source_ckpt': fixed_ckpt_path,
        'fixed_parts': fixed_parts,
        'fixed_box_frame': fixed_box_frame,
        'fixed_box_extra_count': int(fixed_extra_mask_np.sum()),
        'active_parts': sorted(active_parts.keys()),
        'selected_gaussian_count': int(selected_mask.sum().item()),
        'target_count': len(part_targets),
        'best_iter': int(best_state['iter']),
        'best_loss': float(best_state['loss']),
        'train_dir': str(train_dir),
        'out_dir': str(out_dir),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('done | active_parts=%d | selected_gaussians=%d | targets=%d', len(active_parts), int(selected_mask.sum().item()), len(part_targets))


if __name__ == '__main__':
    main()
