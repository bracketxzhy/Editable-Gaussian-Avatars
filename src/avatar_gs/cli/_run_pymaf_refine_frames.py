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
    s = str(raw).strip().lower()
    if s in ("", "-1", "all", "*"):
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


def _python_module_available(module_name: str, python_exe: str) -> bool:
    cmd = [str(python_exe), "-c", f"import {module_name}"]
    try:
        r = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return int(r.returncode) == 0
    except Exception:
        return False


def _ensure_openpifpaf_available(python_exe: str) -> bool:
    if _python_module_available("openpifpaf", python_exe):
        print("openpifpaf is already available.")
        return True

    print("openpifpaf is missing. Trying to install it for the current environment ...")
    install_cmds = [
        # Prefer a no-deps/no-build-isolation path to avoid strict old torch pins.
        [
            str(python_exe), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--no-build-isolation", "--no-deps",
            "openpifpaf==0.13.11",
        ],
        [
            str(python_exe), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--no-build-isolation", "--no-deps",
            "openpifpaf==0.13.1",
        ],
        [
            str(python_exe), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "openpifpaf",
        ],
    ]

    for cmd in install_cmds:
        spec = cmd[-1]
        r = subprocess.run(cmd, check=False)
        if int(r.returncode) == 0 and _python_module_available("openpifpaf", python_exe):
            print(f"Installed {spec} successfully.")
            return True
        print(f"Install attempt failed for: {spec}")

    # openpifpaf plugins may require pysparkling even when installed with --no-deps.
    _ = subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "pysparkling",
        ],
        check=False,
    )
    if _python_module_available("openpifpaf", python_exe):
        print("openpifpaf import is now available after installing pysparkling.")
        return True

    print("Warning: openpifpaf is still unavailable; PyMAF-X will use its fallback detector mode.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser("Run PyMAF-X refine on selected frame indices")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--pymaf_repo_dir", type=str, default="third_party/PyMAF-X")
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--frame_indices", type=str, default="-1")
    parser.add_argument("--single_person", action="store_true")
    parser.add_argument("--cfg_file", type=str, default="configs/pymafx_config.yaml")
    parser.add_argument("--detection_threshold", type=float, default=0.60)
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

    openpifpaf_available = _ensure_openpifpaf_available(sys.executable)

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
        "0.15",
        "DATA.RESCALE_B",
        "1.03",
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
        "openpifpaf_available": bool(openpifpaf_available),
    }
    with (selected_result_dir / "selected_frames_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"PyMAF refine done: {result_pkl}")


if __name__ == "__main__":
    main()
