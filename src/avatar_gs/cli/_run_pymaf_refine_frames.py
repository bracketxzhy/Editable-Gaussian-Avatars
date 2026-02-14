from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _parse_indices(raw: str, n: int) -> list[int]:
    s = str(raw).strip()
    if s == "":
        return list(range(n))

    chunks = s.replace(",", " ").split()
    out: list[int] = []
    for c in chunks:
        i = int(c)
        if i < 0:
            i = n + i
        if i < 0 or i >= n:
            continue
        if i not in out:
            out.append(i)

    if len(out) <= 0:
        raise ValueError(f"No valid frame indices parsed from: {raw}")
    return out


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser("Run PyMAF-X refine on selected frame indices")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--pymaf_repo_dir", type=str, default="third_party/PyMAF-X")
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--frame_indices", type=str, default="-1")
    parser.add_argument("--single_person", action="store_true")
    parser.add_argument("--cfg_file", type=str, default="configs/pymafx_config.yaml")
    parser.add_argument("--detection_threshold", type=float, default=0.3)
    parser.add_argument("--detector_batch_size", type=int, default=1)
    parser.add_argument("--model_batch_size", type=int, default=8)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_folder = Path(args.output_folder).resolve()
    pymaf_repo_dir = Path(args.pymaf_repo_dir).resolve()
    pretrained_model = Path(args.pretrained_model).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not pymaf_repo_dir.exists():
        raise FileNotFoundError(f"PyMAF-X repo not found: {pymaf_repo_dir}")
    if not pretrained_model.exists():
        raise FileNotFoundError(f"PyMAF checkpoint not found: {pretrained_model}")

    frames = _list_frames(frames_dir)
    if len(frames) <= 0:
        raise RuntimeError(f"No frames found under: {frames_dir}")

    selected_idx = _parse_indices(args.frame_indices, len(frames))
    selected_files = [frames[i] for i in selected_idx]

    output_folder.mkdir(parents=True, exist_ok=True)
    selected_input_dir = output_folder / "selected_frames_input"
    if selected_input_dir.exists():
        shutil.rmtree(selected_input_dir, ignore_errors=True)
    selected_input_dir.mkdir(parents=True, exist_ok=True)

    for p in selected_files:
        _safe_link_or_copy(p, selected_input_dir / p.name)

    cmd = [
        sys.executable,
        "-m",
        "apps.demo_smplx",
        "--image_folder",
        str(selected_input_dir),
        "--output_folder",
        str(output_folder),
        "--pretrained_model",
        str(pretrained_model),
        "--cfg_file",
        str(args.cfg_file),
        "--detection_threshold",
        str(float(args.detection_threshold)),
        "--detector_batch_size",
        str(int(args.detector_batch_size)),
        "--model_batch_size",
        str(int(args.model_batch_size)),
        "--tracking_method",
        "pose",
        "--no_render",
        "--misc",
        "TRAIN.BHF_MODE",
        "full_body",
        "MODEL.PyMAF.HAND_VIS_TH",
        "0.1",
    ]
    if bool(args.single_person):
        cmd.append("--single_person")

    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(pymaf_repo_dir), env=env, check=True)

    raw_output_dir = output_folder / selected_input_dir.name
    raw_output_pkl = raw_output_dir / "output.pkl"
    if not raw_output_pkl.exists():
        raise FileNotFoundError(f"PyMAF result not found: {raw_output_pkl}")

    selected_result_dir = output_folder / "selected_frames"
    selected_result_dir.mkdir(parents=True, exist_ok=True)
    result_pkl = selected_result_dir / "output.pkl"
    shutil.copy2(raw_output_pkl, result_pkl)

    meta = {
        "selected_files": [p.name for p in selected_files],
        "selected_indices_global": [int(i) for i in selected_idx],
        "frames_dir": str(frames_dir),
    }
    with (selected_result_dir / "selected_frames_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"PyMAF refine done: {result_pkl}")


if __name__ == "__main__":
    main()
