from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch


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
        shutil.copy2(src, dst)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _prepare_render_root(render_root: Path, clear_fg: bool, clear_deform: bool, clear_real: bool) -> tuple[Path, Path, Path]:
    render_root.mkdir(parents=True, exist_ok=True)
    keep = {"fg_only", "real", "deform"}
    for p in render_root.iterdir():
        if p.is_dir() and p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)

    fg_dir = render_root / "fg_only"
    real_dir = render_root / "real"
    deform_dir = render_root / "deform"
    for d in (fg_dir, real_dir, deform_dir):
        d.mkdir(parents=True, exist_ok=True)

    if clear_fg:
        for f in fg_dir.glob("*.png"):
            _safe_unlink(f)
    if clear_deform:
        for f in deform_dir.glob("*.png"):
            _safe_unlink(f)
    if clear_real:
        for f in real_dir.glob("*.png"):
            _safe_unlink(f)

    return fg_dir, real_dir, deform_dir


def _parse_frame_idx_from_name(name: str) -> int | None:
    stem = Path(str(name)).stem.strip().lower()
    if stem.isdigit():
        return int(stem)
    if stem.startswith("frame_") and stem[6:].isdigit():
        return int(stem[6:])
    return None


def _load_stage2_frames(stage2_dir: Path) -> list[str]:
    ckpt = stage2_dir / "gaussians3d.pt"
    if not ckpt.exists():
        pts = sorted(stage2_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint (*.pt) under: {stage2_dir}")
        ckpt = pts[-1]
    obj = torch.load(str(ckpt), map_location="cpu")
    frames = obj.get("frames", None)
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise RuntimeError("Stage2 checkpoint missing non-empty 'frames'.")
    return [str(x) for x in frames]


def _map_frame_indices_to_internal(requested: list[int], stage2_frames: list[str]) -> list[int]:
    if not requested:
        raise ValueError("fg_frame_indices is empty while fg_frame_select=indices")
    n = len(stage2_frames)
    frame_to_pos = {}
    for pos, fn in enumerate(stage2_frames):
        parsed = _parse_frame_idx_from_name(fn)
        if parsed is not None and parsed not in frame_to_pos:
            frame_to_pos[parsed] = int(pos)

    out = []
    for r in requested:
        rr = int(r)
        if 0 <= rr < n:
            out.append(rr)
            continue
        if rr in frame_to_pos:
            out.append(int(frame_to_pos[rr]))
            continue
        raise IndexError(
            f"Requested frame index {rr} is out of stage2 range (0..{n-1}) and not found in stage2 frame names."
        )
    return out


def _base_overrides(args, out_dir: Path, yaw: float, pitch: float, suffix: str) -> list[str]:
    return [
        f"render.stage2_dir={args.stage2_dir}",
        f"render.out_dir={out_dir.as_posix()}",
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
        "render.view_axis_mode=legacy",
        "render.center_subject_joint_idx=-1",
        "render.fg_only_alpha_threshold=0.04",
        "render.deform_alpha_threshold=0.04",
        "render.fg_only_alpha_gamma=1.0",
        "render.deform_alpha_gamma=1.0",
        "render.fg_only_alpha_gain_max=1.0",
        "render.deform_alpha_gain_max=1.0",
        "render.subject_fit_enabled=true",
        "render.subject_fit_fill=0.90",
        "render.subject_fit_iters=2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", choices=["all", "fg_only", "deform", "real"])
    parser.add_argument("--config", type=str, default="configs/render.yaml")
    parser.add_argument("--stage2_dir", type=str, default="outputs/stage2")
    parser.add_argument("--render_root", type=str, default="outputs/renders")
    parser.add_argument("--k_scale", type=float, default=1.0)
    parser.add_argument("--center_subject", action="store_true", default=True)
    parser.add_argument("--center_subject_depth_min", type=float, default=0.1)
    parser.add_argument("--joint_radius", type=int, default=3)

    parser.add_argument("--fg_frame_select", type=str, default="last", choices=["first", "last", "indices"])
    parser.add_argument("--fg_frame_indices", type=int, nargs="*", default=None)

    parser.add_argument("--deform_source", type=str, default="smplx_params", choices=["smplx_params", "tpose"])
    parser.add_argument("--deform_smplx_params_npz", type=str, default="outputs/stage1/smpl_params_sequence.npz")
    parser.add_argument("--deform_smplx_frame_idx", type=int, default=-1)
    parser.add_argument("--deform_smplx_frame_name", type=str, default="")
    parser.add_argument("--deform_smplx_model_type", type=str, default="smplx")
    parser.add_argument("--deform_smplx_gender", type=str, default="neutral")
    parser.add_argument("--deform_smplx_model_root", type=str, default="")
    args = parser.parse_args()

    do_fg = args.task in ("all", "fg_only")
    do_deform = args.task in ("all", "deform")
    do_real = args.task in ("all", "real")
    if not do_real:
        print(f"[INFO] task={args.task}: real branch will be skipped.")

    stage2_dir = Path(args.stage2_dir)
    stage2_frames = _load_stage2_frames(stage2_dir)
    fg_indices_internal = None
    if str(args.fg_frame_select).strip().lower() == "indices":
        fg_req = list(args.fg_frame_indices or [])
        fg_indices_internal = _map_frame_indices_to_internal(fg_req, stage2_frames)

    render_root = Path(args.render_root)
    fg_dir, real_dir, deform_dir = _prepare_render_root(
        render_root=render_root,
        clear_fg=do_fg,
        clear_deform=do_deform,
        clear_real=do_real,
    )

    if do_fg:
        for name, yaw, pitch in VIEWS:
            overrides = _base_overrides(args, fg_dir, yaw, pitch, name) + [
                f"render.frame_select={args.fg_frame_select}",
                "render.max_frames=1",
                "render.mode=fg_only",
                "render.use_gt_background=false",
                "render.save_main_image=false",
                "render.save_real_image=false",
                "render.save_fg_nobg_skeleton=true",
                "render.deform_render_image=false",
            ]
            if fg_indices_internal:
                idx_str = ",".join(str(int(x)) for x in fg_indices_internal)
                overrides += [f"render.frame_indices={idx_str}"]
            _run_render(args.config, overrides)
            src = _pick_suffix_png(fg_dir, name)
            if not src.name.endswith(f"_{name}.png"):
                raise RuntimeError(f"fg_only output naming mismatch for view '{name}': {src.name}")

    if do_deform:
        use_standard_pose_preset = str(args.deform_source).strip().lower() == "tpose"
        if use_standard_pose_preset:
            print("[INFO] deform uses SMPL-X params + standard pose preset (t-pose as one deform example).")
        for name, yaw, pitch in VIEWS:
            overrides = _base_overrides(args, deform_dir, yaw, pitch, name) + [
                f"render.frame_select={args.fg_frame_select}",
                "render.max_frames=1",
                "render.mode=deform",
                "render.use_gt_background=false",
                "render.save_main_image=false",
                "render.save_real_image=false",
                "render.save_fg_nobg_skeleton=false",
                "render.deform_render_image=true",
                "render.deform_out_subdir=.",
                "render.deform_source=smplx_params",
                f"render.deform_smplx_params_npz={args.deform_smplx_params_npz}",
                f"render.deform_smplx_model_type={args.deform_smplx_model_type}",
                f"render.deform_smplx_gender={args.deform_smplx_gender}",
                f"render.deform_smplx_use_standard_pose={'true' if use_standard_pose_preset else 'false'}",
            ]
            if fg_indices_internal:
                idx_str = ",".join(str(int(x)) for x in fg_indices_internal)
                overrides += [f"render.frame_indices={idx_str}"]
            if int(args.deform_smplx_frame_idx) >= 0:
                overrides += [f"render.deform_smplx_frame_index={int(args.deform_smplx_frame_idx)}"]
            if str(args.deform_smplx_frame_name).strip():
                overrides += [f"render.deform_smplx_frame_name={str(args.deform_smplx_frame_name).strip()}"]
            if str(args.deform_smplx_model_root).strip():
                overrides += [f"render.deform_smplx_model_root={str(args.deform_smplx_model_root).strip()}"]
            _run_render(args.config, overrides)
            src = _pick_suffix_png(deform_dir, name)
            if not src.name.endswith(f"_{name}.png"):
                raise RuntimeError(f"deform output naming mismatch for view '{name}': {src.name}")
            dst = deform_dir / f"{name}.png"
            if src != dst:
                _move_or_copy(src, dst)

    if do_real:
        name, yaw, pitch = VIEWS[0]
        overrides = _base_overrides(args, real_dir, yaw, pitch, name) + [
            f"render.frame_select={args.fg_frame_select}",
            "render.max_frames=1",
            "render.use_gt_background=true",
            "render.save_main_image=false",
            "render.save_real_image=true",
            "render.save_fg_nobg_skeleton=false",
            "render.deform_render_image=false",
            "render.real_out_subdir=.",
            "render.center_subject=false",
            "render.k_scale=1.0",
        ]
        if fg_indices_internal:
            idx_str = ",".join(str(int(x)) for x in fg_indices_internal)
            overrides += [f"render.frame_indices={idx_str}"]
        _run_render(args.config, overrides)
        src = _pick_suffix_png(real_dir, name)
        if not src.name.endswith(f"_{name}.png"):
            raise RuntimeError(f"real output naming mismatch for view '{name}': {src.name}")
        dst = real_dir / "front.png"
        if src != dst:
            _move_or_copy(src, dst)
        if not dst.exists():
            raise RuntimeError(
                "real render expected outputs/renders/real/front.png but file is missing. "
                f"task={args.task}, overrides={' '.join(overrides)}"
            )

    print(
        "Done.",
        f"fg_only={len(list(fg_dir.glob('*.png')))}",
        f"deform={len(list(deform_dir.glob('*.png')))}",
        f"real={len(list(real_dir.glob('*.png')))}",
    )


if __name__ == "__main__":
    main()


