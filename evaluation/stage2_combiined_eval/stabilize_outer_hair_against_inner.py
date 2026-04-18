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
PROJECT_ROOT = next((p for p in THIS_FILE.parents if (p / 'configs').is_dir() and (p / 'src').is_dir()), None)
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path.cwd().resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / 'src'):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def rgb_to_logit(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(rgb / (1.0 - rgb))


def opacity_to_logit(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.log(alpha / (1.0 - alpha))


def main() -> None:
    ap = argparse.ArgumentParser('Stabilize hair-bound outer gaussians against inner hair appearance without disabling them.')
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--inner_ckpt', default='outputs/stage2/gaussians3d.pt')
    ap.add_argument('--outer_ckpt', required=True)
    ap.add_argument('--smpl_face_semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--rgb_blend', type=float, default=0.72)
    ap.add_argument('--max_rgb_delta', type=float, default=0.10)
    ap.add_argument('--opacity_scale', type=float, default=0.55)
    ap.add_argument('--max_opacity', type=float, default=0.14)
    ap.add_argument('--min_opacity', type=float, default=0.012)
    ap.add_argument('--face_fallback_mix', type=float, default=0.65)
    args = ap.parse_args()

    project_root = resolve_path(Path.cwd(), args.project_root) if not Path(args.project_root).is_absolute() else Path(args.project_root).resolve()
    inner_ckpt = resolve_path(project_root, args.inner_ckpt)
    outer_ckpt = resolve_path(project_root, args.outer_ckpt)
    sem_path = resolve_path(project_root, args.smpl_face_semantics_npz)
    out_dir = resolve_path(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inner = torch.load(str(inner_ckpt), map_location='cpu')
    outer = torch.load(str(outer_ckpt), map_location='cpu')
    sem = np.load(str(sem_path), allow_pickle=True)
    raw_label_names = [str(x) for x in np.asarray(sem['raw_label_names']).reshape(-1).tolist()]
    raw_label_ids = np.asarray(sem['raw_label_ids'], dtype=np.int64).reshape(-1)
    if 'hair' not in raw_label_names:
        raise RuntimeError('hair label missing from smpl semantics')
    hair_raw_id = int(raw_label_names.index('hair'))

    g_in = inner['gaussians']
    g_out = outer['gaussians']

    bind_in = torch.as_tensor(g_in['bind_face_idx'], dtype=torch.long).view(-1)
    shell_in = torch.as_tensor(g_in['shell_layer'], dtype=torch.long).view(-1)
    bind_out = torch.as_tensor(g_out['bind_face_idx'], dtype=torch.long).view(-1)
    shell_out = torch.as_tensor(g_out['shell_layer'], dtype=torch.long).view(-1)
    prune_out = torch.as_tensor(g_out.get('prune_state', torch.zeros_like(shell_out)), dtype=torch.long).view(-1)

    valid_in = (bind_in >= 0) & (bind_in < int(raw_label_ids.shape[0]))
    valid_out = (bind_out >= 0) & (bind_out < int(raw_label_ids.shape[0]))
    in_hair = valid_in & (shell_in == 0) & torch.from_numpy(raw_label_ids[bind_in.numpy()] == hair_raw_id)
    out_hair = valid_out & (shell_out == 1) & (prune_out == 0) & torch.from_numpy(raw_label_ids[bind_out.numpy()] == hair_raw_id)

    if int(in_hair.sum().item()) <= 0:
        raise RuntimeError('no inner hair rows found')
    if int(out_hair.sum().item()) <= 0:
        raise RuntimeError('no outer hair rows found')

    rgb_in = torch.sigmoid(torch.as_tensor(g_in['rgb_logit'], dtype=torch.float32))
    rgb_out = torch.sigmoid(torch.as_tensor(g_out['rgb_logit'], dtype=torch.float32))
    op_in = torch.sigmoid(torch.as_tensor(g_in['opacity_logit'], dtype=torch.float32)).reshape(-1)
    op_out = torch.sigmoid(torch.as_tensor(g_out['opacity_logit'], dtype=torch.float32)).reshape(-1)

    global_inner_rgb = (rgb_in[in_hair] * op_in[in_hair, None]).sum(dim=0) / op_in[in_hair].sum().clamp_min(1e-8)

    face_to_rgb_sum = {}
    face_to_w_sum = {}
    in_rows = torch.where(in_hair)[0].tolist()
    for row in in_rows:
        face_id = int(bind_in[row].item())
        w = float(op_in[row].item())
        if w <= 0.0:
            continue
        rgbv = rgb_in[row]
        face_to_rgb_sum[face_id] = face_to_rgb_sum.get(face_id, torch.zeros(3, dtype=torch.float32)) + rgbv * w
        face_to_w_sum[face_id] = face_to_w_sum.get(face_id, 0.0) + w

    target_rgb = rgb_out.clone()
    target_op = op_out.clone()
    reports = []
    changed_rows = torch.where(out_hair)[0].tolist()
    for row in changed_rows:
        face_id = int(bind_out[row].item())
        if face_id in face_to_rgb_sum and face_to_w_sum[face_id] > 1e-8:
            inner_face_rgb = face_to_rgb_sum[face_id] / float(face_to_w_sum[face_id])
            ref_rgb = float(args.face_fallback_mix) * inner_face_rgb + (1.0 - float(args.face_fallback_mix)) * global_inner_rgb
        else:
            ref_rgb = global_inner_rgb
        before_rgb = rgb_out[row].clone()
        desired = before_rgb + float(args.rgb_blend) * (ref_rgb - before_rgb)
        delta = (desired - before_rgb).clamp(-float(args.max_rgb_delta), float(args.max_rgb_delta))
        after_rgb = (before_rgb + delta).clamp(1.0e-4, 1.0 - 1.0e-4)
        target_rgb[row] = after_rgb

        before_op = float(op_out[row].item())
        after_op = min(max(before_op * float(args.opacity_scale), float(args.min_opacity)), float(args.max_opacity))
        target_op[row] = float(after_op)

        reports.append({
            'row': int(row),
            'face_id': face_id,
            'rgb_before': [float(x) for x in before_rgb.tolist()],
            'rgb_after': [float(x) for x in after_rgb.tolist()],
            'opacity_before': before_op,
            'opacity_after': float(after_op),
        })

    g_out['rgb_logit'] = rgb_to_logit(target_rgb).to(dtype=torch.as_tensor(g_out['rgb_logit']).dtype)
    g_out['opacity_logit'] = opacity_to_logit(target_op).reshape_as(torch.as_tensor(g_out['opacity_logit'], dtype=torch.float32)).to(dtype=torch.as_tensor(g_out['opacity_logit']).dtype)

    out_ckpt = out_dir / 'gaussians3d.pt'
    torch.save(outer, str(out_ckpt))

    report = {
        'source_outer_ckpt': str(outer_ckpt),
        'inner_ckpt': str(inner_ckpt),
        'hair_label_id': hair_raw_id,
        'inner_hair_rows': int(in_hair.sum().item()),
        'outer_hair_rows_changed': int(out_hair.sum().item()),
        'global_inner_hair_rgb': [float(x) for x in global_inner_rgb.tolist()],
        'rgb_blend': float(args.rgb_blend),
        'max_rgb_delta': float(args.max_rgb_delta),
        'opacity_scale': float(args.opacity_scale),
        'max_opacity': float(args.max_opacity),
        'min_opacity': float(args.min_opacity),
        'face_fallback_mix': float(args.face_fallback_mix),
        'changed_rows_sample': reports[:128],
        'mean_opacity_before': float(op_out[out_hair].mean().item()),
        'mean_opacity_after': float(target_op[out_hair].mean().item()),
        'weighted_rgb_before': [float(x) for x in ((rgb_out[out_hair] * op_out[out_hair, None]).sum(dim=0) / op_out[out_hair].sum().clamp_min(1e-8)).tolist()],
        'weighted_rgb_after': [float(x) for x in ((target_rgb[out_hair] * target_op[out_hair, None]).sum(dim=0) / target_op[out_hair].sum().clamp_min(1e-8)).tolist()],
    }
    (out_dir / 'stabilize_outer_hair_against_inner_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
