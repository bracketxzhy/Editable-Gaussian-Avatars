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
    writer = None
    out_path = None
    writer_specs = [
        ("input.mp4", "mp4v"),
        ("input.avi", "MJPG"),
        ("input.avi", "XVID"),
    ]
    for fname, fourcc_tag in writer_specs:
        cand = tmp_dir / fname
        wobj = cv2.VideoWriter(
            str(cand),
            cv2.VideoWriter_fourcc(*fourcc_tag),
            float(fps),
            (int(w), int(h)),
        )
        if wobj.isOpened():
            writer = wobj
            out_path = cand
            break
        wobj.release()
    if writer is None or out_path is None:
        raise RuntimeError(f"Failed to create temp video writer under: {tmp_dir}")

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if ih != h or iw != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()

    if not out_path.exists():
        raise RuntimeError(f"Failed to build temp video: {out_path}")

    print(f"Built temp video from {len(frames)} frames: {out_path} ({w}x{h})")
    return out_path, tmp_dir


def _patch_numpy_legacy_aliases() -> None:
    """Patch deprecated NumPy aliases required by legacy chumpy/smpl loaders."""
    alias_map = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in alias_map.items():
        if np.__dict__.get(name, None) is None:
            setattr(np, name, value)


def _resolve_wham_asset_root(output_dir: Path, wham_repo_dir: Path) -> Path | None:
    # 1) Explicit env override
    env_root = os.environ.get("WHAM_ASSET_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    # 2) Stage1 convention: outputs/stage1/{wham,runtime/...}
    cand = output_dir.resolve().parent / "runtime" / "wham_runtime_assets"
    if cand.exists():
        return cand

    # 3) WHAM demo convention
    cand = wham_repo_dir.resolve() / "dataset"
    if cand.exists():
        return cand

    return None


def _run_wham_api(video_path: Path, output_dir: Path, wham_repo_dir: Path, run_global: bool) -> None:
    # WHAM imports use top-level modules like `lib.*`.
    # Our project also has `src/lib`, so isolate import path to avoid module collisions.
    prev_cwd = Path.cwd()
    prev_sys_path = list(sys.path)
    src_root = Path(__file__).resolve().parents[2]

    try:
        _patch_numpy_legacy_aliases()
        sys.path = [
            p for p in sys.path
            if Path(p or ".").resolve() != src_root.resolve()
        ]

        # Ensure WHAM asset root is set even when running this script standalone.
        wham_asset_root = _resolve_wham_asset_root(output_dir=output_dir, wham_repo_dir=wham_repo_dir)
        if wham_asset_root is not None:
            os.environ["WHAM_ASSET_ROOT"] = str(wham_asset_root)
            ckpt_dir = wham_asset_root / "checkpoints"
            if ckpt_dir.exists() and os.environ.get("WHAM_CHECKPOINT_DIR", "").strip() == "":
                os.environ["WHAM_CHECKPOINT_DIR"] = str(ckpt_dir)

        # Prefer WHAM-bundled ViTPose fork (contains ViT backbone registry entries).
        vitpose_dir = wham_repo_dir / "third-party" / "ViTPose"
        if vitpose_dir.exists():
            sys.path.insert(0, str(vitpose_dir))
            if os.environ.get("WHAM_VITPOSE_DIR", "").strip() == "":
                os.environ["WHAM_VITPOSE_DIR"] = str(vitpose_dir)

        sys.path.insert(0, str(wham_repo_dir))

        for name in list(sys.modules.keys()):
            if name == "lib" or name.startswith("lib.") or name == "mmpose" or name.startswith("mmpose."):
                sys.modules.pop(name, None)

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
        sys.path = prev_sys_path

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


