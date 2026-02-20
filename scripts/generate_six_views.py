"""
Generate required render outputs under outputs/renders:
  - fg_only: 6 views (no background, gaussian human + keypoints)
  - t-pose : 6 views (T-pose gaussian human + keypoints)
  - real   : 1 view (front only, GT background + gaussian human)

Usage:
  python scripts/generate_six_views.py --config configs/render.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


VIEWS = [
    ("front", 0.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 90.0, 0.0),
    ("right", -90.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
]


def _run_render(config: str, overrides: list[str]) -> None:
    cmd = [sys.executable, "-m", "avatar_gs.cli.render", "--config", config, "--override", *overrides]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _pick_suffix_png(folder: Path, suffix_name: str) -> Path:
    cands = sorted(folder.glob(f"*_{suffix_name}.png"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No rendered file matched '*_{suffix_name}.png' in {folder}")
    return cands[-1]


def _move_or_copy(src: Path, dst: Path) -> None:
    if src == dst:
        return
    _safe_unlink(dst)
    try:
        src.replace(dst)
    except PermissionError:
        # Windows may keep a handle briefly (thumbnail/indexer); fallback to copy.
        import shutil

        shutil.copy2(src, dst)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # Skip locked files; named outputs will still be overwritten later.
        pass


def _prepare_render_root(render_root: Path) -> tuple[Path, Path, Path]:
    render_root.mkdir(parents=True, exist_ok=True)
    keep = {"fg_only", "real", "t-pose"}
    for p in render_root.iterdir():
        if p.is_dir() and p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file() and p.suffix.lower() == ".png":
            _safe_unlink(p)

    fg_dir = render_root / "fg_only"
    real_dir = render_root / "real"
    tp_dir = render_root / "t-pose"
    for d in (fg_dir, real_dir, tp_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"):
            _safe_unlink(f)
    return fg_dir, real_dir, tp_dir


def _base_overrides(args, out_dir: Path, yaw: float, pitch: float, suffix: str) -> list[str]:
    return [
        f"render.stage2_dir={args.stage2_dir}",
        f"render.out_dir={out_dir.as_posix()}",
        "render.frame_select=last",
        "render.max_frames=1",
        "render.clear_out_dir_pngs=false",
        f"render.k_scale={args.k_scale}",
        f"render.view_yaw_deg={yaw}",
        f"render.view_pitch_deg={pitch}",
        f"render.file_suffix={suffix}",
        f"render.center_subject={'true' if args.center_subject else 'false'}",
        "render.center_subject_use_mesh=true",
        f"render.center_subject_depth_min={args.center_subject_depth_min}",
        "render.overlay_smpl_skeleton=true",
        "render.smpl_skeleton_source=joints",
        f"render.smpl_skeleton_joint_radius={args.joint_radius}",
        "render.smpl_skeleton_line_thickness=0",
        "render.view_axis_mode=z_up",
        "render.center_subject_joint_idx=-1",
        "render.fg_only_alpha_gamma=1.0",
        "render.tpose_alpha_gamma=1.0",
        "render.fg_only_alpha_gain_max=1.0",
        "render.tpose_alpha_gain_max=1.0",
        "render.subject_fit_enabled=true",
        "render.subject_fit_fill=0.82",
        "render.subject_fit_iters=2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/render.yaml")
    parser.add_argument("--stage2_dir", type=str, default="outputs/stage2")
    parser.add_argument("--render_root", type=str, default="outputs/renders")
    parser.add_argument("--k_scale", type=float, default=0.85)
    parser.add_argument("--center_subject", action="store_true", default=True)
    parser.add_argument("--center_subject_depth_min", type=float, default=0.1)
    parser.add_argument("--joint_radius", type=int, default=3)
    args = parser.parse_args()

    render_root = Path(args.render_root)
    fg_dir, real_dir, tp_dir = _prepare_render_root(render_root)

    # fg_only: six views
    for name, yaw, pitch in VIEWS:
        overrides = _base_overrides(args, fg_dir, yaw, pitch, name) + [
            "render.mode=fg_only",
            "render.use_gt_background=false",
            "render.save_main_image=false",
            "render.save_real_image=false",
            "render.save_fg_nobg_skeleton=true",
            "render.render_tpose_image=false",
        ]
        _run_render(args.config, overrides)
        src = _pick_suffix_png(fg_dir, name)
        dst = fg_dir / f"{name}.png"
        if src != dst:
            _move_or_copy(src, dst)

    # t-pose: six views
    for name, yaw, pitch in VIEWS:
        overrides = _base_overrides(args, tp_dir, yaw, pitch, name) + [
            "render.mode=tpose",
            "render.use_gt_background=false",
            "render.save_main_image=false",
            "render.save_real_image=false",
            "render.save_fg_nobg_skeleton=false",
            "render.render_tpose_image=true",
        ]
        _run_render(args.config, overrides)
        src = _pick_suffix_png(tp_dir, name)
        dst = tp_dir / f"{name}.png"
        if src != dst:
            _move_or_copy(src, dst)

    # real: front only, with GT background
    name, yaw, pitch = VIEWS[0]
    overrides = _base_overrides(args, real_dir, yaw, pitch, name) + [
        "render.use_gt_background=true",
        "render.save_main_image=false",
        "render.save_real_image=true",
        "render.save_fg_nobg_skeleton=false",
        "render.render_tpose_image=false",
        "render.real_out_subdir=.",
        "render.center_subject=false",
        "render.k_scale=1.0",
    ]
    _run_render(args.config, overrides)
    src = _pick_suffix_png(real_dir, name)
    dst = real_dir / "front.png"
    if src != dst:
        _move_or_copy(src, dst)

    # Validate output counts.
    expected_views = [n for n, _, _ in VIEWS]
    missing_fg = [n for n in expected_views if not (fg_dir / f"{n}.png").exists()]
    missing_tp = [n for n in expected_views if not (tp_dir / f"{n}.png").exists()]
    missing_real = [] if (real_dir / "front.png").exists() else ["front"]
    if missing_fg or missing_tp or missing_real:
        raise RuntimeError(
            f"Missing expected outputs: fg_only={missing_fg}, t-pose={missing_tp}, real={missing_real}"
        )

    fg_n = len(list(fg_dir.glob("*.png")))
    tp_n = len(list(tp_dir.glob("*.png")))
    real_n = len(list(real_dir.glob("*.png")))
    print(f"Done. fg_only={fg_n}, t-pose={tp_n}, real={real_n}")


if __name__ == "__main__":
    main()
