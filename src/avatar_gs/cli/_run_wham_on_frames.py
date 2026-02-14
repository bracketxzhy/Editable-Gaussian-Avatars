from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _make_temp_video(frames: list[Path], fps: float) -> tuple[Path, Path]:
    if len(frames) <= 0:
        raise RuntimeError("No frames provided for WHAM input video.")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")

    h, w = first.shape[:2]
    tmp_dir = Path(tempfile.mkdtemp(prefix="wham_frames_"))
    mp4_path = tmp_dir / "input.mp4"

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create temp video writer: {mp4_path}")

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if ih != h or iw != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()

    if not mp4_path.exists():
        raise RuntimeError(f"Failed to build temp video: {mp4_path}")

    print(f"Built temp video from {len(frames)} frames: {mp4_path} ({w}x{h})")
    return mp4_path, tmp_dir


def _run_wham_api(video_path: Path, output_dir: Path, wham_repo_dir: Path, run_global: bool) -> None:
    # WHAM imports use repository-relative modules and config paths.
    sys.path.insert(0, str(wham_repo_dir))
    prev_cwd = Path.cwd()
    try:
        os.chdir(str(wham_repo_dir))
        from wham_api import WHAM_API  # pylint: disable=import-outside-toplevel

        api = WHAM_API()
        results, tracking_results, slam_results = api(
            str(video_path),
            output_dir=str(output_dir),
            run_global=bool(run_global),
            visualize=False,
        )
    finally:
        os.chdir(str(prev_cwd))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, output_dir / "wham_results.pth")
    joblib.dump(tracking_results, output_dir / "tracking_results.pth")
    joblib.dump(slam_results, output_dir / "slam_results.pth")

    # Keep a compact metadata file for downstream checks.
    meta = {
        "n_tracks": int(len(results) if isinstance(results, dict) else 0),
        "run_global": bool(run_global),
    }
    np.savez(output_dir / "wham_meta.npz", **meta)


def main() -> None:
    parser = argparse.ArgumentParser("Run WHAM from an image-frame directory")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wham_repo_dir", type=str, default="third_party/WHAM")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--estimate_local_only", action="store_true")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    wham_repo_dir = Path(args.wham_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not wham_repo_dir.exists():
        raise FileNotFoundError(f"WHAM repo not found: {wham_repo_dir}")

    frames = _list_frames(frames_dir)
    if len(frames) <= 0:
        raise RuntimeError(f"No frames found under: {frames_dir}")

    temp_video = None
    temp_dir = None
    try:
        temp_video, temp_dir = _make_temp_video(frames, fps=float(args.fps))
        _run_wham_api(
            video_path=temp_video,
            output_dir=output_dir,
            wham_repo_dir=wham_repo_dir,
            run_global=(not bool(args.estimate_local_only)),
        )
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"WHAM done. Output: {output_dir}")


if __name__ == "__main__":
    main()
