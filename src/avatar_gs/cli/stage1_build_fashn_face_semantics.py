from __future__ import annotations

import argparse
import json
from pathlib import Path

from avatar_gs.segmentation.smpl_face_semantics import build_stage1_fashn_face_semantics


def _resolve_project_path(project_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser("Build SMPL face-level semantics from local FASHN parsing and stage1 SMPL output")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--stage1_dir", type=str, default="outputs/stage1")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--runtime_dir", type=str, default="")
    parser.add_argument("--max_edge", type=int, default=384)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--min_non_bg_votes", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage1_dir = _resolve_project_path(project_root, args.stage1_dir)
    frames_dir = _resolve_project_path(project_root, args.frames_dir)
    out_dir = (
        (stage1_dir / "fashn_face_semantics").resolve()
        if str(args.out_dir).strip() == ""
        else _resolve_project_path(project_root, args.out_dir)
    )
    model_dir = None if str(args.model_dir).strip() == "" else _resolve_project_path(project_root, args.model_dir)
    runtime_dir = None if str(args.runtime_dir).strip() == "" else _resolve_project_path(project_root, args.runtime_dir)

    summary = build_stage1_fashn_face_semantics(
        stage1_dir=stage1_dir,
        frames_dir=frames_dir,
        out_dir=out_dir,
        device=str(args.device),
        model_dir=model_dir,
        runtime_dir=runtime_dir,
        max_edge=int(args.max_edge),
        frame_stride=int(args.frame_stride),
        max_frames=int(args.max_frames),
        min_non_bg_votes=int(args.min_non_bg_votes),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
