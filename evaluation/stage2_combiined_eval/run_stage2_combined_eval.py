#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import errno
import os
import sys
import time
import types
import shutil as _shutil
from pathlib import Path

import cv2
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in THIS_FILE.parents
    if (p / 'configs').is_dir() and (p / 'src').is_dir()
)
for p in (PROJECT_ROOT, PROJECT_ROOT / 'src'):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def make_render_preview_renderer(out_dir: Path, target_layer: str):
    preview_dir = out_dir / '_render_preview' / 'fg_only' / str(target_layer).strip().lower()

    def render_from_preview(self, i: int, frame_name: str):
        path = preview_dir / f'{Path(frame_name).stem}_front.png'
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'combined eval render preview missing: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        # The render CLI writes white-background fg_only images; non-white pixels are the rendered support.
        mask_np = (np.any(rgb < (250.0 / 255.0), axis=2)).astype(np.float32)[None, ...]
        mask = torch.from_numpy(mask_np).contiguous()
        return pred, mask, 1.0

    return render_from_preview


def make_masked_render_preview_builder(original_builder, supervision_cache: Path, clip_mask_dilate_px: int = 0):
    z_cache = {'loaded': False}

    def load_cache():
        if not z_cache['loaded']:
            z = np.load(str(supervision_cache), allow_pickle=True)
            z_cache['frame_names'] = [str(x) for x in np.asarray(z['frame_names']).reshape(-1).tolist()]
            z_cache['inner_masks'] = np.asarray(z['inner_masks'], dtype=np.uint8)
            z_cache['outer_masks'] = np.asarray(z['outer_masks'], dtype=np.uint8)
            z_cache['loaded'] = True
        return z_cache

    def masked_builder(**kwargs):
        preview_map = original_builder(**kwargs)
        target_layer = str(kwargs.get('target_layer', '')).strip().lower()
        if target_layer != 'combined':
            return preview_map
        cache = load_cache()
        idx_map = {name: i for i, name in enumerate(cache['frame_names'])}
        for frame_name, path in preview_map.items():
            i = idx_map.get(str(frame_name))
            if i is None:
                raise RuntimeError(f'combined preview mask frame missing in supervision cache: {frame_name}')
            mask = ((cache['inner_masks'][i] > 0) | (cache['outer_masks'][i] > 0)).astype(np.uint8)
            if int(clip_mask_dilate_px) > 0:
                k = 2 * int(clip_mask_dilate_px) + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f'combined preview image missing before mask clip: {path}')
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img[mask == 0] = (255, 255, 255)
            if not cv2.imwrite(str(path), img):
                raise RuntimeError(f'failed to write masked combined preview: {path}')
        return preview_map

    return masked_builder

def make_recoverable_rmtree(project_root: Path):
    trash_root = project_root / 'trash' / 'stage2_combined_eval'
    trash_root.mkdir(parents=True, exist_ok=True)
    readme = project_root / 'trash' / 'README.md'
    if not readme.exists():
        readme.write_text(
            '# trash\n\n'
            'Recoverable files moved by evaluation/training helpers. '\
            'Do not treat this as a source directory; review and delete manually only when you no longer need recovery.\n',
            encoding='utf-8',
        )

    def recoverable_rmtree(path, ignore_errors=False, onerror=None):
        src = Path(path)
        if not src.exists():
            return
        try:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            dst = trash_root / f'{stamp}_{src.name}'
            suffix = 1
            while dst.exists():
                dst = trash_root / f'{stamp}_{src.name}_{suffix}'
                suffix += 1
            try:
                os.rename(src, dst)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # Cross-device tempdirs, usually from /tmp, cannot be renamed into the repo trash.
                # Copy them for recoverability and leave the original tempdir in place instead of deleting it.
                if src.is_dir():
                    _shutil.copytree(src, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(src, dst)
        except Exception:
            if ignore_errors:
                return
            raise

    return recoverable_rmtree


def tensor_maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    if not (a.dtype.is_floating_point or b.dtype.is_floating_point):
        return float((a != b).to(torch.int32).max().item()) if a.numel() else 0.0
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item()) if a.numel() else 0.0


def merge_eval_checkpoint(
    inner_ckpt: Path,
    outer_ckpt: Path,
    out_ckpt: Path,
    manifest_path: Path,
    *,
    deactivate_outer_exposed: bool = False,
    smpl_face_semantics_npz: Path | None = None,
) -> Path:
    inner = torch.load(str(inner_ckpt), map_location='cpu')
    outer = torch.load(str(outer_ckpt), map_location='cpu')
    if 'gaussians' not in inner or 'gaussians' not in outer:
        raise RuntimeError('Both checkpoints must contain a gaussians state dict.')
    inner_g = inner['gaussians']
    outer_g_src = outer['gaussians']
    if 'shell_layer' not in outer_g_src:
        raise RuntimeError(f'outer checkpoint has no shell_layer: {outer_ckpt}')
    if 'shell_layer' not in inner_g:
        raise RuntimeError(f'inner checkpoint has no shell_layer: {inner_ckpt}')
    shell = torch.as_tensor(outer_g_src['shell_layer'], dtype=torch.long)
    inner_shell = torch.as_tensor(inner_g['shell_layer'], dtype=torch.long)
    if tuple(shell.shape) != tuple(inner_shell.shape):
        raise RuntimeError(f'shell_layer shape mismatch: inner={tuple(inner_shell.shape)} outer={tuple(shell.shape)}')
    if not torch.equal(shell, inner_shell):
        raise RuntimeError('shell_layer layout differs between inner and outer checkpoints; refusing positional merge.')

    inner_mask = shell == 0
    outer_mask = shell == 1
    if not bool(inner_mask.any()):
        raise RuntimeError('No shell_layer==0 rows found for inner replacement.')
    if not bool(outer_mask.any()):
        raise RuntimeError('No shell_layer==1 rows found to keep as outer.')

    merged = copy.deepcopy(outer)
    merged_g = merged['gaussians']
    replaced_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, outer_value in list(merged_g.items()):
        inner_value = inner_g.get(key)
        if (
            torch.is_tensor(outer_value)
            and torch.is_tensor(inner_value)
            and outer_value.ndim >= 1
            and tuple(outer_value.shape) == tuple(inner_value.shape)
            and int(outer_value.shape[0]) == int(shell.shape[0])
        ):
            new_value = outer_value.clone()
            new_value[inner_mask] = inner_value[inner_mask].to(dtype=new_value.dtype)
            merged_g[key] = new_value
            replaced_keys.append(key)
        else:
            skipped_keys.append(key)

    deactivated_outer_exposed = 0
    deactivated_mask = torch.zeros_like(shell, dtype=torch.bool)
    if bool(deactivate_outer_exposed):
        if smpl_face_semantics_npz is None or not Path(smpl_face_semantics_npz).exists():
            raise FileNotFoundError(f'deactivate_outer_exposed requires smpl semantics npz: {smpl_face_semantics_npz}')
        face_idx_t = merged_g.get('bind_face_idx')
        if not torch.is_tensor(face_idx_t):
            raise RuntimeError('deactivate_outer_exposed requires gaussian bind_face_idx.')
        sem = np.load(str(smpl_face_semantics_npz), allow_pickle=True)
        if 'exclude_outer_mask' not in sem.files:
            raise RuntimeError(f'smpl semantics npz has no exclude_outer_mask: {smpl_face_semantics_npz}')
        exclude = torch.from_numpy(np.asarray(sem['exclude_outer_mask'], dtype=np.uint8).reshape(-1) > 0)
        face_idx = torch.as_tensor(face_idx_t, dtype=torch.long)
        valid = outer_mask & (face_idx >= 0) & (face_idx < int(exclude.shape[0]))
        blocked = torch.zeros_like(outer_mask, dtype=torch.bool)
        if bool(valid.any()):
            blocked[valid] = exclude[face_idx[valid]]
        deactivated_mask = blocked
        deactivated_outer_exposed = int(blocked.sum().item())
        if deactivated_outer_exposed > 0:
            opacity = merged_g.get('opacity_logit')
            if not torch.is_tensor(opacity):
                raise RuntimeError('deactivate_outer_exposed requires opacity_logit.')
            opacity_new = opacity.clone()
            opacity_new[blocked] = torch.minimum(
                opacity_new[blocked],
                torch.full_like(opacity_new[blocked], -12.0),
            )
            merged_g['opacity_logit'] = opacity_new

    for key in ('K', 'cam_w', 'cam_t', 'frame_Ks', 'frames', 'H', 'W', 'bg_rgb'):
        if key in inner:
            merged[key] = copy.deepcopy(inner[key])

    merged.setdefault('eval_merge', {})
    merged['eval_merge'] = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'inner_source': str(inner_ckpt),
        'outer_source': str(outer_ckpt),
        'rule': 'shell_layer==0 rows from inner_source; shell_layer==1 rows from outer_source',
        'inner_rows_replaced': int(inner_mask.sum().item()),
        'outer_rows_kept': int(outer_mask.sum().item()),
        'deactivate_outer_exposed': bool(deactivate_outer_exposed),
        'smpl_face_semantics_npz': str(smpl_face_semantics_npz) if smpl_face_semantics_npz is not None else '',
        'outer_exposed_rows_deactivated': int(deactivated_outer_exposed),
        'replaced_gaussian_keys': replaced_keys,
        'skipped_gaussian_keys': skipped_keys,
    }

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out_ckpt))

    check = torch.load(str(out_ckpt), map_location='cpu')
    check_g = check['gaussians']
    validation = {}
    outer_validation_mask = outer_mask & (~deactivated_mask)
    for key in replaced_keys:
        v = check_g[key]
        iv = inner_g[key]
        ov = outer_g_src[key]
        validation[key] = {
            'inner_rows_maxdiff_vs_inner': tensor_maxdiff(v[inner_mask], iv[inner_mask]),
            'outer_rows_maxdiff_vs_outer': tensor_maxdiff(v[outer_validation_mask], ov[outer_validation_mask]),
            'outer_rows_validated': int(outer_validation_mask.sum().item()),
        }
    manifest = dict(merged['eval_merge'])
    manifest['merged_checkpoint'] = str(out_ckpt)
    manifest['validation'] = validation
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return out_ckpt

def main() -> None:
    ap = argparse.ArgumentParser('Stage2 combined evaluation wrapper; does not modify inner artifacts.')
    ap.add_argument('--project_root', default=str(PROJECT_ROOT))
    ap.add_argument('--stage1_dir', default='outputs/stage1')
    ap.add_argument('--stage2_dir', default='outputs/stage2')
    ap.add_argument('--inner_ckpt', default='outputs/stage2/gaussians3d.pt')
    ap.add_argument('--outer_ckpt', default='outputs/stage2_outeronly_d2_full_20260409_run1/gaussians3d.pt')
    ap.add_argument('--supervision_cache', default='outputs/stage2/supervision_cache.npz')
    ap.add_argument('--frames_dir', default='data/frames/input')
    ap.add_argument('--out_dir', default='evaluation/eval_outputs/stage2_combined')
    ap.add_argument('--work_dir', default='outputs/stage2_combined_eval_work')
    ap.add_argument('--alpha_threshold', type=float, default=0.03)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--frame_names', nargs='*', default=[])
    ap.add_argument('--face_only_render', action='store_true')
    ap.add_argument('--face_only_pad_px', type=int, default=24)
    ap.add_argument('--use_render_preview_pred', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--clip_render_preview_to_target_mask', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--clip_mask_dilate_px', type=int, default=0)
    ap.add_argument('--deactivate_outer_exposed', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--smpl_face_semantics_npz', default='outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz')
    args = ap.parse_args()

    project_root = resolve_path(Path.cwd(), args.project_root) if not Path(args.project_root).is_absolute() else Path(args.project_root).resolve()
    inner_ckpt = resolve_path(project_root, args.inner_ckpt)
    outer_ckpt = resolve_path(project_root, args.outer_ckpt)
    work_dir = resolve_path(project_root, args.work_dir)
    merged_ckpt = work_dir / 'combined_inner_existing_outer_d2.pt'
    manifest_path = work_dir / 'combined_inner_existing_outer_d2_manifest.json'
    merge_eval_checkpoint(
        inner_ckpt,
        outer_ckpt,
        merged_ckpt,
        manifest_path,
        deactivate_outer_exposed=bool(args.deactivate_outer_exposed),
        smpl_face_semantics_npz=resolve_path(project_root, args.smpl_face_semantics_npz),
    )

    import evaluation.stage2_eval.run_stage2_eval as stage2_eval

    if bool(args.use_render_preview_pred):
        out_dir_resolved = resolve_path(project_root, args.out_dir)
        stage2_eval.EvalRenderer.render = make_render_preview_renderer(out_dir_resolved, 'combined')
    if bool(args.clip_render_preview_to_target_mask):
        sup_path_resolved = resolve_path(project_root, args.supervision_cache)
        stage2_eval.build_render_preview_map = make_masked_render_preview_builder(
            stage2_eval.build_render_preview_map,
            sup_path_resolved,
            clip_mask_dilate_px=int(args.clip_mask_dilate_px),
        )
    stage2_eval.shutil = types.SimpleNamespace(copy2=_shutil.copy2, rmtree=make_recoverable_rmtree(project_root))
    eval_argv = [
        'run_stage2_eval.py',
        '--project_root', str(project_root),
        '--stage1_dir', args.stage1_dir,
        '--stage2_dir', args.stage2_dir,
        '--stage2_ckpt', str(merged_ckpt),
        '--supervision_cache', args.supervision_cache,
        '--frames_dir', args.frames_dir,
        '--out_dir', args.out_dir,
        '--target_layer', 'combined',
        '--alpha_threshold', str(float(args.alpha_threshold)),
        '--max_frames', str(int(args.max_frames)),
    ]
    if args.frame_names:
        eval_argv.extend(['--frame_names', *args.frame_names])
    if args.face_only_render:
        eval_argv.append('--face_only_render')
        eval_argv.extend(['--face_only_pad_px', str(int(args.face_only_pad_px))])

    old_argv = sys.argv[:]
    try:
        sys.argv = eval_argv
        stage2_eval.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
