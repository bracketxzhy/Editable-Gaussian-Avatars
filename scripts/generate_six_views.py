"""
Generate six canonical views (front/back/left/right/top/bottom) for
`fg_only` and `t-pose` outputs by calling the project's render CLI.

Usage:
  python scripts/generate_six_views.py --config configs/stage2_gaussian.yaml

This script invokes `python -m avatar_gs.cli.render` multiple times with
overrides for `view_yaw_deg`/`view_pitch_deg` and output subdirs. It sets
`center_subject=true` and reduces `k_scale` slightly to avoid feet touching
edges.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def call_render(config: str, overrides: list[str]):
    cmd = [sys.executable, "-m", "avatar_gs.cli.render", "--config", config, "--override"]
    cmd.append(" ".join(overrides))
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_gaussian.yaml")
    parser.add_argument("--k_scale", type=float, default=0.85, help="intrinsic scale (smaller=wider FOV)")
    parser.add_argument("--center_subject", type=str, default="true")
    args = parser.parse_args()

    cfg = args.config
    # Canonical six views: (yaw, pitch)
    views = {
        "front": (0.0, 0.0),
        "back": (180.0, 0.0),
        "left": (90.0, 0.0),
        "right": (-90.0, 0.0),
        "top": (0.0, -90.0),
        "bottom": (0.0, 90.0),
    }

    # 1) FG-only views: write directly into outputs/renders/fg_only as <view>.png
    fg_dir = Path("outputs/renders/fg_only")
    fg_dir.mkdir(parents=True, exist_ok=True)
    for view_name, (yaw, pitch) in views.items():
        overrides = [
            # do not set render.mode=fg_only (that special-cases subdir to '.')
            "render.save_main_image=false",
            "render.save_real_image=false",
            "render.save_fg_nobg_skeleton=true",
            "render.clear_out_dir_pngs=false",
            "render.fg_nobg_out_subdir=fg_only",
            f"render.view_yaw_deg={yaw}",
            f"render.view_pitch_deg={pitch}",
            f"render.k_scale={args.k_scale}",
            f"render.center_subject={args.center_subject}",
            f"render.center_subject_use_mesh=true",
        ]
        try:
            call_render(cfg, overrides)
        except subprocess.CalledProcessError as e:
            print(f"FG render failed for {view_name}: {e}")
            continue
        # move produced file (frame stem) to canonical view name
        stem = "000756"  # expected frame stem; adjust if needed
        src = Path("outputs/renders") / "fg_only" / f"{stem}.png"
        if src.exists():
            dst = fg_dir / f"{view_name}.png"
            src.replace(dst)
        else:
            print(f"Warning: expected FG output not found: {src}")

    # 2) T-pose views: run render per view and collect t-pose outputs into outputs/renders/t-pose
    tp_dir = Path("outputs/renders/t-pose")
    tp_dir.mkdir(parents=True, exist_ok=True)
    for view_name, (yaw, pitch) in views.items():
        overrides = [
            "render.save_main_image=false",
            "render.save_real_image=false",
            "render.save_fg_nobg_skeleton=false",
            "render.clear_out_dir_pngs=false",
            "render.tpose_out_subdir=t-pose",
            f"render.view_yaw_deg={yaw}",
            f"render.view_pitch_deg={pitch}",
            f"render.k_scale={args.k_scale}",
            f"render.center_subject={args.center_subject}",
            f"render.center_subject_use_mesh=true",
        ]
        try:
            call_render(cfg, overrides)
        except subprocess.CalledProcessError as e:
            print(f"T-pose render failed for {view_name}: {e}")
            continue
        stem = "000756"
        src = Path("outputs/renders") / "t-pose" / f"{stem}.png"
        if src.exists():
            dst = tp_dir / f"{view_name}.png"
            src.replace(dst)
        else:
            print(f"Warning: expected T-pose output not found: {src}")

    # 3) Real composite: render front view with GT background enabled and save a single image
    real_dir = Path("outputs/renders/real")
    real_dir.mkdir(parents=True, exist_ok=True)
    yaw, pitch = views["front"]
    overrides = [
        "render.save_main_image=false",
        "render.save_fg_nobg_skeleton=true",
        "render.save_real_image=true",
        "render.clear_out_dir_pngs=false",
        "render.fg_nobg_out_subdir=fg_only",
        "render.real_out_subdir=real",
        "render.use_gt_background=true",
        f"render.view_yaw_deg={yaw}",
        f"render.view_pitch_deg={pitch}",
        f"render.k_scale={args.k_scale}",
        f"render.center_subject={args.center_subject}",
        f"render.center_subject_use_mesh=true",
    ]
    try:
        call_render(cfg, overrides)
    except subprocess.CalledProcessError as e:
        print(f"Real render failed: {e}")
    # rename produced real image to a clear name
    stem = "000756"
    src_real = Path("outputs/renders") / "real" / f"{stem}.png"
    if src_real.exists():
        dst_real = real_dir / f"{stem}.png"
        src_real.replace(dst_real)
    else:
        print(f"Warning: expected real output not found: {src_real}")


if __name__ == "__main__":
    main()
