#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in THIS_FILE.parents if (p / 'configs').is_dir() and (p / 'src').is_dir())


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def run(cmd: list[str], project_root: Path) -> None:
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


def load_summary(eval_dir: Path) -> dict:
    return json.loads((eval_dir / 'summary.json').read_text(encoding='utf-8'))


def main() -> None:
    ap = argparse.ArgumentParser('Reproduce the current clean stage2 outer chain from an explicit archived base outer checkpoint.')
    ap.add_argument('--project_root', default=str(PROJECT_ROOT))
    ap.add_argument('--python_bin', default='/data/miniconda3/envs/ega/bin/python')
    ap.add_argument('--base_outer_ckpt', required=True)
    ap.add_argument('--reference_eval_dir', default='evaluation/eval_outputs/stage2_combined')
    ap.add_argument('--tag_root', required=True)
    args = ap.parse_args()

    project_root = resolve_path(Path.cwd(), args.project_root) if not Path(args.project_root).is_absolute() else Path(args.project_root).resolve()
    py = str(resolve_path(project_root, args.python_bin) if not Path(args.python_bin).is_absolute() else Path(args.python_bin).resolve())
    base_outer_ckpt = str(resolve_path(project_root, args.base_outer_ckpt))
    reference_eval_dir = resolve_path(project_root, args.reference_eval_dir)
    tag_root = str(args.tag_root).strip()
    if not tag_root:
        raise RuntimeError('tag_root is required')

    step1_out = f'outputs/{tag_root}_step1_attenuate_faces'
    step1_work = f'outputs/{tag_root}_step1_work'
    step2_out = f'outputs/{tag_root}_step2_attenuate_rows'
    step2_work = f'outputs/{tag_root}_step2_work'
    step3_out = f'outputs/{tag_root}_step3_rebalance_color'
    step3_work = f'outputs/{tag_root}_step3_work'
    final_eval = f'evaluation/eval_outputs/{tag_root}_combined'
    final_eval_work = f'outputs/{tag_root}_eval_work'

    run([
        py,
        'evaluation/stage2_combiined_eval/attenuate_harmful_outer_top_faces.py',
        '--outer_ckpt', base_outer_ckpt,
        '--out_dir', step1_out,
        '--work_dir', step1_work,
        '--alpha_threshold', '0.03',
        '--harmful_margin', '0.006',
        '--effect_min', '0.004',
        '--min_face_votes', '220',
        '--min_harmful_ratio', '0.42',
        '--top_faces', '32',
        '--opacity_scale', '0.35',
        '--opacity_floor', '0.01',
    ], project_root)

    run([
        py,
        'evaluation/stage2_combiined_eval/attenuate_harmful_outer_top_rows.py',
        '--outer_ckpt', f'{step1_out}/gaussians3d.pt',
        '--out_dir', step2_out,
        '--work_dir', step2_work,
        '--frames', 'all',
        '--alpha_threshold', '0.03',
        '--harmful_margin', '0.006',
        '--helpful_margin', '0.006',
        '--effect_min', '0.0035',
        '--sample_radius', '3',
        '--min_harm_score', '0.0012',
        '--min_harm_hits', '3',
        '--min_net_ratio', '0.22',
        '--helpful_weight', '0.85',
        '--top_rows', '24',
        '--min_opacity_scale', '0.55',
        '--max_opacity_scale', '0.82',
        '--opacity_floor', '0.01',
    ], project_root)

    run([
        py,
        'evaluation/stage2_combiined_eval/rebalance_outer_top_green_residual.py',
        '--outer_ckpt', f'{step2_out}/gaussians3d.pt',
        '--out_dir', step3_out,
        '--work_dir', step3_work,
        '--frames', 'all',
        '--alpha_threshold', '0.03',
        '--effect_min', '0.0035',
        '--sample_radius', '3',
        '--green_margin', '0.007',
        '--gray_margin', '0.012',
        '--dark_margin', '0.014',
        '--helpful_margin', '0.004',
        '--min_artifact_score', '0.35',
        '--min_artifact_hits', '8',
        '--min_artifact_ratio', '0.45',
        '--top_rows', '32',
        '--residual_gain', '0.35',
        '--max_rgb_delta', '0.025',
        '--green_cap_ratio', '0.25',
    ], project_root)

    run([
        py,
        'evaluation/stage2_combiined_eval/run_stage2_combined_eval.py',
        '--inner_ckpt', 'outputs/stage2/gaussians3d.pt',
        '--outer_ckpt', f'{step3_out}/gaussians3d.pt',
        '--out_dir', final_eval,
        '--work_dir', final_eval_work,
        '--alpha_threshold', '0.03',
        '--deactivate_outer_exposed',
        '--use_render_preview_pred',
        '--clip_render_preview_to_target_mask',
    ], project_root)

    ref = load_summary(reference_eval_dir)
    new = load_summary(resolve_path(project_root, final_eval))
    report = {
        'tag_root': tag_root,
        'base_outer_ckpt': base_outer_ckpt,
        'reference_eval_dir': str(reference_eval_dir),
        'reproduced_eval_dir': str(resolve_path(project_root, final_eval)),
        'reference_global': ref['global_metrics'],
        'reproduced_global': new['global_metrics'],
        'global_delta': {
            'mae_mean': float(new['global_metrics']['mae_mean'] - ref['global_metrics']['mae_mean']),
            'psnr_mean': float(new['global_metrics']['psnr_mean'] - ref['global_metrics']['psnr_mean']),
            'ssim_mean': float(new['global_metrics']['ssim_mean'] - ref['global_metrics']['ssim_mean']),
        },
        'reference_top': ref['part_summary']['top'],
        'reproduced_top': new['part_summary']['top'],
        'top_delta': {
            'mae_mean': float(new['part_summary']['top']['mae_mean'] - ref['part_summary']['top']['mae_mean']),
            'ssim_mean': float(new['part_summary']['top']['ssim_mean'] - ref['part_summary']['top']['ssim_mean']),
            'badness_mean': float(new['part_summary']['top']['badness_mean'] - ref['part_summary']['top']['badness_mean']),
        },
        'reference_pants': ref['part_summary']['pants'],
        'reproduced_pants': new['part_summary']['pants'],
        'pants_delta': {
            'mae_mean': float(new['part_summary']['pants']['mae_mean'] - ref['part_summary']['pants']['mae_mean']),
            'ssim_mean': float(new['part_summary']['pants']['ssim_mean'] - ref['part_summary']['pants']['ssim_mean']),
            'badness_mean': float(new['part_summary']['pants']['badness_mean'] - ref['part_summary']['pants']['badness_mean']),
        },
    }
    out_manifest = resolve_path(project_root, step3_out) / 'reproduce_t126_clean_report.json'
    out_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
