#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in THIS_FILE.parents if (p / "configs").is_dir() and (p / "src").is_dir())
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from avatar_gs.segmentation.fashn import get_fashn_label_maps, load_fashn_parser, predict_fashn_segmentation
from avatar_gs.segmentation.smpl_face_semantics import rasterize_visible_face_index_map
from avatar_gs.stage2 import load_stage2_supervision_cache
from evaluation.stage2_combiined_eval.run_stage2_combined_eval import merge_eval_checkpoint
from evaluation.stage2_eval.run_stage2_eval import EvalRenderer, rgb_np, setup_logger, target_arrays


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def jfix(x):
    if isinstance(x, dict):
        return {str(k): jfix(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jfix(v) for v in x]
    if isinstance(x, tuple):
        return [jfix(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def abdomen_proxy_mask(part_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(part_mask > 0, dtype=bool)
    ys, xs = np.where(m)
    if ys.size <= 0:
        return np.zeros_like(m, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = max(y1 - y0 + 1, 1)
    w = max(x1 - x0 + 1, 1)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    yn = (yy - y0).astype(np.float32) / max(h - 1, 1)
    xn = (xx - x0).astype(np.float32) / max(w - 1, 1)
    proxy = (xn >= 0.20) & (xn <= 0.82) & (yn >= 0.42) & (yn <= 0.88)
    return m & proxy


def main() -> None:
    ap = argparse.ArgumentParser('Attenuate harmful outer top faces using abdomen-proxy combined-vs-inner attribution.')
    ap.add_argument('--project_root', default=str(PROJECT_ROOT))
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--stage2_dir', default='outputs/stage2')
    ap.add_argument('--inner_ckpt', default='outputs/stage2/gaussians3d.pt')
    ap.add_argument('--outer_ckpt', required=True)
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--smpl_face_semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--alpha_threshold', type=float, default=0.03)
    ap.add_argument('--harmful_margin', type=float, default=0.01)
    ap.add_argument('--effect_min', type=float, default=0.003)
    ap.add_argument('--min_face_votes', type=float, default=96.0)
    ap.add_argument('--min_harmful_ratio', type=float, default=0.18)
    ap.add_argument('--top_faces', type=int, default=48)
    ap.add_argument('--opacity_scale', type=float, default=0.60)
    ap.add_argument('--opacity_floor', type=float, default=0.01)
    args = ap.parse_args()

    project_root = resolve_path(Path.cwd(), args.project_root) if not Path(args.project_root).is_absolute() else Path(args.project_root).resolve()
    stage1_dir = resolve_path(project_root, args.stage1_dir)
    stage2_dir = resolve_path(project_root, args.stage2_dir)
    out_dir = resolve_path(project_root, args.out_dir)
    work_dir = resolve_path(project_root, args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / 'attenuate_harmful_outer_top_faces.log')

    inner_ckpt = resolve_path(project_root, args.inner_ckpt)
    outer_ckpt = resolve_path(project_root, args.outer_ckpt)
    sup_path = resolve_path(project_root, args.supervision_cache)
    sem_path = resolve_path(project_root, args.smpl_face_semantics_npz)

    merged_ckpt = work_dir / 'combined_inner_existing_outer_d2.pt'
    merge_eval_checkpoint(
        inner_ckpt,
        outer_ckpt,
        merged_ckpt,
        work_dir / 'combined_inner_existing_outer_d2_manifest.json',
        deactivate_outer_exposed=True,
        smpl_face_semantics_npz=sem_path,
    )

    z = np.load(str(sup_path), allow_pickle=True)
    frames = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
    sup = load_stage2_supervision_cache(cache_path=sup_path, frames=frames, logger=logger)
    if sup is None:
        raise RuntimeError(f'Cannot load supervision cache: {sup_path}')

    sem = np.load(str(sem_path), allow_pickle=True)
    raw_ids = np.asarray(sem['raw_label_ids'], dtype=np.int64).reshape(-1)
    raw_names = [str(x) for x in np.asarray(sem['raw_label_names']).reshape(-1).tolist()]
    if 'top' not in raw_names:
        raise RuntimeError('SMPL face semantics has no raw label named top')
    top_raw_id = int(raw_names.index('top'))

    ids_to_labels, _ = get_fashn_label_maps(runtime_dir=None)
    top_fashn_id = None
    for rid, name in ids_to_labels.items():
        if str(name) == 'top':
            top_fashn_id = int(rid)
            break
    if top_fashn_id is None:
        raise RuntimeError('FASHN label maps have no top label')

    parser = load_fashn_parser(runtime_dir=None, model_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    combined_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, 'combined', float(args.alpha_threshold), logger)
    inner_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, 'inner', float(args.alpha_threshold), logger)
    outer_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, 'outer', 0.0, logger)

    face_scores: dict[int, dict[str, float]] = {}
    harmful_margin = float(args.harmful_margin)
    effect_min = float(args.effect_min)
    alpha_threshold = float(args.alpha_threshold)

    for i, frame_name in enumerate(frames):
        gt_u8, eval_mask = target_arrays(sup, 'combined', i)
        gt = np.asarray(gt_u8, dtype=np.float32) / 255.0
        pred_c_t, _, _ = combined_renderer.render(i, frame_name)
        pred_i_t, _, _ = inner_renderer.render(i, frame_name)
        _, outer_alpha_t, _ = outer_renderer.render(i, frame_name)
        pred_c = rgb_np(pred_c_t)
        pred_i = rgb_np(pred_i_t)
        outer_alpha = np.asarray(outer_alpha_t.detach().cpu().squeeze(0).numpy(), dtype=np.float32).clip(0.0, 1.0)
        err_c = np.abs(pred_c - gt).mean(axis=2).astype(np.float32)
        err_i = np.abs(pred_i - gt).mean(axis=2).astype(np.float32)
        effect = np.abs(pred_c - pred_i).mean(axis=2).astype(np.float32)
        seg = predict_fashn_segmentation(np.asarray(gt_u8, dtype=np.uint8), parser=parser)

        R, t, K = combined_renderer.camera(frame_name)
        k_np = combined_renderer.frame_K(frame_name)
        verts_t, _ = combined_renderer.mesh_runtime.load_frame_mesh_verts(i, K_np=k_np, bind_cache_size=combined_renderer.bind_cache_size)
        verts_cam = (verts_t.detach().cpu().numpy().astype(np.float32) @ R.detach().cpu().numpy().astype(np.float32).T) + t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
        faces_np = np.asarray(combined_renderer.mesh_runtime.context.bind_faces_np, dtype=np.int32)
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=verts_cam,
            faces=faces_np,
            K=K.detach().cpu().numpy().astype(np.float32),
            image_hw=(int(gt.shape[0]), int(gt.shape[1])),
        )
        valid_face = (face_index_map >= 0) & (face_index_map < int(raw_ids.shape[0]))
        part_mask = (np.asarray(seg, dtype=np.int32) == int(top_fashn_id)) & np.asarray(eval_mask, dtype=bool) & valid_face
        proxy = abdomen_proxy_mask(part_mask)
        if int(proxy.sum()) < 64:
            continue
        harmful = proxy & (outer_alpha > alpha_threshold) & ((err_c - err_i) > harmful_margin) & (effect > effect_min)
        active = proxy & (outer_alpha > alpha_threshold) & (effect > effect_min)
        helpful = proxy & (outer_alpha > alpha_threshold) & ((err_i - err_c) > harmful_margin) & (effect > effect_min)
        logger.info(
            'frame=%s proxy=%d harmful=%d helpful=%d active=%d delta_mean=%.6f',
            frame_name,
            int(proxy.sum()),
            int(harmful.sum()),
            int(helpful.sum()),
            int(active.sum()),
            float((err_c[proxy] - err_i[proxy]).mean()),
        )
        for mode_name, mask in (('harmful', harmful), ('active', active), ('helpful', helpful)):
            if not bool(mask.any()):
                continue
            face_ids = face_index_map[mask].astype(np.int64)
            keep_sem = raw_ids[face_ids] == int(top_raw_id)
            face_ids = face_ids[keep_sem]
            if int(face_ids.size) == 0:
                continue
            uniq, inv = np.unique(face_ids, return_inverse=True)
            counts = np.bincount(inv).astype(np.float64)
            for fid, count in zip(uniq.tolist(), counts.tolist()):
                row = face_scores.setdefault(int(fid), {'harmful_votes': 0.0, 'active_votes': 0.0, 'helpful_votes': 0.0, 'frames_harmful': 0.0, 'frames_active': 0.0})
                row[f'{mode_name}_votes'] += float(count)
                if mode_name == 'harmful':
                    row['frames_harmful'] += 1.0
                if mode_name == 'active':
                    row['frames_active'] += 1.0

    ranked = []
    for fid, row in face_scores.items():
        harmful_votes = float(row['harmful_votes'])
        active_votes = float(row['active_votes'])
        helpful_votes = float(row['helpful_votes'])
        harmful_ratio = harmful_votes / max(active_votes, 1.0)
        net_votes = harmful_votes - 0.5 * helpful_votes
        if harmful_votes < float(args.min_face_votes):
            continue
        if harmful_ratio < float(args.min_harmful_ratio):
            continue
        ranked.append({
            'face_id': int(fid),
            'harmful_votes': harmful_votes,
            'active_votes': active_votes,
            'helpful_votes': helpful_votes,
            'harmful_ratio': harmful_ratio,
            'net_votes': net_votes,
            'frames_harmful': int(row['frames_harmful']),
            'frames_active': int(row['frames_active']),
        })
    ranked.sort(key=lambda r: (float(r['net_votes']), float(r['harmful_ratio']), float(r['harmful_votes'])), reverse=True)
    ranked = ranked[: max(int(args.top_faces), 0)]
    selected_faces = [int(r['face_id']) for r in ranked]

    ck = torch.load(str(outer_ckpt), map_location='cpu')
    g = ck['gaussians']
    shell = torch.as_tensor(g['shell_layer']).view(-1)
    prune = torch.as_tensor(g['prune_state']).view(-1)
    face = torch.as_tensor(g['bind_face_idx']).view(-1).to(dtype=torch.long)
    opacity_logit = torch.as_tensor(g['opacity_logit'], dtype=torch.float32).clone().view(-1)

    if selected_faces:
        selected_t = torch.as_tensor(selected_faces, dtype=torch.long)
        selected_rows = (shell == 1) & (prune == 0) & torch.isin(face, selected_t)
    else:
        selected_rows = torch.zeros_like(shell, dtype=torch.bool)

    old_alpha = torch.sigmoid(opacity_logit[selected_rows]).clone() if bool(selected_rows.any()) else torch.zeros((0,), dtype=torch.float32)
    if bool(selected_rows.any()):
        new_alpha = torch.clamp(old_alpha * float(args.opacity_scale), min=float(args.opacity_floor), max=0.999)
        new_logits = torch.log(new_alpha / (1.0 - new_alpha))
        opacity_logit[selected_rows] = new_logits.to(dtype=opacity_logit.dtype)
    else:
        new_alpha = old_alpha

    g['opacity_logit'] = opacity_logit.reshape_as(g['opacity_logit'])
    manifest = {
        'source': str(outer_ckpt),
        'inner_ckpt': str(inner_ckpt),
        'parts': ['top'],
        'selected_faces': selected_faces,
        'selected_face_rows': ranked,
        'selected_face_count': int(len(selected_faces)),
        'attenuated_rows': int(selected_rows.sum().item()),
        'opacity_scale': float(args.opacity_scale),
        'opacity_floor': float(args.opacity_floor),
        'alpha_threshold': float(args.alpha_threshold),
        'harmful_margin': float(args.harmful_margin),
        'effect_min': float(args.effect_min),
        'min_face_votes': float(args.min_face_votes),
        'min_harmful_ratio': float(args.min_harmful_ratio),
        'top_faces': int(args.top_faces),
        'opacity_change_mean_abs': float((new_alpha - old_alpha).abs().mean().item()) if int(old_alpha.numel()) > 0 else 0.0,
        'opacity_change_max_abs': float((new_alpha - old_alpha).abs().max().item()) if int(old_alpha.numel()) > 0 else 0.0,
    }
    ck['outer_harmful_top_face_attenuation'] = manifest
    out_ckpt = out_dir / 'gaussians3d.pt'
    torch.save(ck, str(out_ckpt))
    (out_dir / 'harmful_top_face_attenuation_manifest.json').write_text(json.dumps(jfix({k: v for k, v in manifest.items() if k != 'selected_face_rows'}), ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'harmful_top_face_rows.json').write_text(json.dumps(jfix(ranked), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(jfix({k: v for k, v in manifest.items() if k != 'selected_face_rows'}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()