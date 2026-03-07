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


def _resolve_model_name(pretrained_model: str) -> str:
    raw = str(pretrained_model).strip()
    if raw == "":
        raise ValueError("pretrained_model is empty.")
    p = Path(raw)
    if p.suffixes[-2:] == [".pth", ".tar"]:
        return p.name[: -len(".pth.tar")]
    if p.suffix in (".pt", ".pth", ".ckpt"):
        return p.stem
    return p.name


def _ensure_pretrained_model(repo_dir: Path, pretrained_model: str) -> str:
    pretrain_dir = repo_dir / "pretrained_models"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    raw = str(pretrained_model).strip()
    p = Path(raw)
    if p.exists():
        model_name = _resolve_model_name(raw)
        if p.suffixes[-2:] == [".pth", ".tar"]:
            dst = pretrain_dir / f"{model_name}.pth.tar"
        elif p.suffix in (".pt", ".pth", ".ckpt"):
            dst = pretrain_dir / f"{model_name}{p.suffix}"
        else:
            dst = pretrain_dir / p.name
        _safe_link_or_copy(p.resolve(), dst)
        return model_name

    model_name = _resolve_model_name(raw)
    ckpt = pretrain_dir / f"{model_name}.pth.tar"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SMPLer-X checkpoint not found: {raw}. Expected existing model stem or file at {ckpt}."
        )
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser("Run SMPLer-X inference on selected frame indices")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--smplerx_repo_dir", type=str, default="third_party/SMPLer-X")
    parser.add_argument("--pretrained_model", type=str, required=True, help="SMPLer-X model stem or checkpoint path")
    parser.add_argument("--frame_indices", type=str, default="-1")
    parser.add_argument("--single_person", action="store_true")
    parser.add_argument("--bbox_thr", type=int, default=50)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_folder = Path(args.output_folder).resolve()
    smplerx_repo_dir = Path(args.smplerx_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not smplerx_repo_dir.exists():
        raise FileNotFoundError(f"SMPLer-X repo not found: {smplerx_repo_dir}")

    frames = _list_frames(frames_dir)
    if len(frames) <= 0:
        raise RuntimeError(f"No frames found under: {frames_dir}")

    selected_idx = _parse_indices(args.frame_indices, len(frames))
    selected_files = [frames[i] for i in selected_idx]

    output_folder.mkdir(parents=True, exist_ok=True)
    selected_input_dir = output_folder / "selected_frames_input"
    selected_result_dir = output_folder / "selected_frames"
    if selected_input_dir.exists():
        shutil.rmtree(selected_input_dir, ignore_errors=True)
    if selected_result_dir.exists():
        shutil.rmtree(selected_result_dir, ignore_errors=True)
    selected_input_dir.mkdir(parents=True, exist_ok=True)
    selected_result_dir.mkdir(parents=True, exist_ok=True)

    selected_input_files: list[str] = []
    for local_idx, p in enumerate(selected_files, start=1):
        dst_name = f"{local_idx:06d}{p.suffix.lower()}"
        selected_input_files.append(dst_name)
        _safe_link_or_copy(p, selected_input_dir / dst_name)

    model_name = _ensure_pretrained_model(smplerx_repo_dir, args.pretrained_model)
    inference_py = smplerx_repo_dir / "main" / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(f"SMPLer-X inference entry not found: {inference_py}")

    cmd = [
        sys.executable,
        str(inference_py),
        "--num_gpus",
        "1",
        "--pretrained_model",
        model_name,
        "--img_path",
        str(selected_input_dir),
        "--start",
        "1",
        "--end",
        str(len(selected_files)),
        "--output_folder",
        str(selected_result_dir),
        "--bbox_thr",
        str(int(args.bbox_thr)),
        "--iou_thr",
        str(float(args.iou_thr)),
    ]
    if not bool(args.single_person):
        cmd.append("--multi_person")

    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[3]
    py_paths = [
        str((project_root / "src").resolve()),
        str((smplerx_repo_dir / "main" / "transformer_utils").resolve()),
    ]
    if str(env.get("PYTHONPATH", "")).strip():
        py_paths.append(str(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    subprocess.run(cmd, cwd=str(smplerx_repo_dir / "main"), env=env, check=True)

    meta = {
        "backend": "smplerx",
        "selected_files": [p.name for p in selected_files],
        "selected_indices_global": [int(i) for i in selected_idx],
        "selected_input_files": selected_input_files,
        "frames_dir": str(frames_dir),
        "model_name": model_name,
        "single_person": bool(args.single_person),
    }
    with (selected_result_dir / "selected_frames_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"SMPLer-X inference done: {selected_result_dir}")


if __name__ == "__main__":
    main()
