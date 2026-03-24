from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _find_eval_report(candidate_dir: Path, project_root: Path) -> Path:
    eval_root = project_root / "evaluation" / "output_images"
    name = candidate_dir.name
    if name == "stage1":
        p = eval_root / "stage1" / "stage1_eval_report_all.json"
        if p.exists():
            return p
    stripped = re.sub(r"_\d{8}$", "", name)
    options = [
        eval_root / f"{name}_stage1" / "stage1_eval_report_all.json",
        eval_root / f"{stripped}_stage1" / "stage1_eval_report_all.json",
        eval_root / name / "stage1_eval_report_all.json",
        eval_root / stripped / "stage1_eval_report_all.json",
    ]
    for p in options:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing eval report for candidate_dir={candidate_dir}")


def _score_metrics(m: dict[str, Any]) -> float:
    iou = float(m.get("iou", 0.0))
    outside = float(m.get("outside_ratio", 1.0))
    bmean = float(m.get("pred_to_gt_boundary_mean_px", 1.0e6))
    bp95 = float(m.get("pred_to_gt_boundary_p95_px", 1.0e6))
    inside = float(m.get("inside_coverage", 0.0))
    area = float(m.get("area_ratio", 0.0))
    undersize = max(0.0, 0.92 - area)
    oversize = max(0.0, area - 1.0)
    return (
        11.0 * iou
        - 24.0 * outside
        - 0.28 * bmean
        - 0.05 * bp95
        + 1.4 * inside
        - 2.2 * undersize
        - 1.6 * oversize
    )


def _load_report_map(report_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(report_json.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rep in data.get("frame_reports", []):
        frame_name = str(rep.get("frame_name", ""))
        if not frame_name:
            continue
        metrics = dict(rep.get("metrics_after", {}))
        metrics["_score"] = _score_metrics(metrics)
        out[frame_name] = metrics
    return out


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {str(k): z[k] for k in z.files}


def _frame_index_map(frame_names: np.ndarray) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(frame_names.tolist())}


def _is_framewise_array(arr: np.ndarray, n_frames: int) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim >= 1 and int(arr.shape[0]) == int(n_frames)


def _copy_frame_obj(src_dir: Path, dst_dir: Path, frame_name: str) -> None:
    frame_stem = Path(frame_name).stem
    src = src_dir / "sequence" / f"frame_{frame_stem}.obj"
    if not src.exists():
        raise FileNotFoundError(f"Missing source OBJ for frame {frame_name}: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def main() -> None:
    ap = argparse.ArgumentParser("Merge best per-frame stage1 outputs from tested candidate runs")
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="outputs/stage1_bestof")
    ap.add_argument("--candidate_stage1_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--base_stage1_dir", type=str, default="outputs/stage1")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    base_stage1_dir = (project_root / args.base_stage1_dir).resolve()
    candidate_dirs = [(project_root / p).resolve() for p in args.candidate_stage1_dirs]

    if base_stage1_dir not in candidate_dirs:
        candidate_dirs = [base_stage1_dir] + candidate_dirs

    base_params = _load_npz_dict(base_stage1_dir / "smpl_params_sequence.npz")
    base_verts = _load_npz_dict(base_stage1_dir / "smpl_vertices_sequence.npz")
    base_masks = _load_npz_dict(base_stage1_dir / "person_masks.npz")
    base_frame_names = np.asarray(base_params["frame_names"], dtype=object)
    n_frames = int(base_frame_names.shape[0])

    candidate_data: list[dict[str, Any]] = []
    for cand_dir in candidate_dirs:
        report_map = _load_report_map(_find_eval_report(cand_dir, project_root))
        params_npz = _load_npz_dict(cand_dir / "smpl_params_sequence.npz")
        verts_npz = _load_npz_dict(cand_dir / "smpl_vertices_sequence.npz")
        masks_npz = _load_npz_dict(cand_dir / "person_masks.npz")
        keypoints = json.loads((cand_dir / "keypoints.json").read_text(encoding="utf-8"))
        frame_map = _frame_index_map(np.asarray(params_npz["frame_names"], dtype=object))
        candidate_data.append(
            {
                "dir": cand_dir,
                "report": report_map,
                "params": params_npz,
                "verts": verts_npz,
                "masks": masks_npz,
                "keypoints": keypoints,
                "frame_map": frame_map,
            }
        )

    selected_sources: list[str] = []
    selected_scores = np.full((n_frames,), np.nan, dtype=np.float32)

    merged_params: dict[str, Any] = {}
    for key, arr in base_params.items():
        if _is_framewise_array(arr, n_frames):
            merged_params[key] = np.empty_like(arr)
        else:
            merged_params[key] = arr

    merged_verts: dict[str, Any] = {}
    for key, arr in base_verts.items():
        if _is_framewise_array(arr, n_frames):
            merged_verts[key] = np.empty_like(arr)
        else:
            merged_verts[key] = arr

    merged_masks: dict[str, Any] = {}
    for key, arr in base_masks.items():
        if _is_framewise_array(arr, n_frames):
            merged_masks[key] = np.empty_like(arr)
        else:
            merged_masks[key] = arr

    merged_keypoints: dict[str, Any] = {}

    for frame_idx, frame_name_obj in enumerate(base_frame_names.tolist()):
        frame_name = str(frame_name_obj)
        best = None
        best_rank = None
        for cand in candidate_data:
            cand_idx = cand["frame_map"].get(frame_name, None)
            if cand_idx is None:
                continue
            metrics = cand["report"].get(frame_name, None)
            if metrics is None:
                continue
            rank = (
                float(metrics["_score"]),
                -float(metrics.get("outside_ratio", 1.0)),
                -float(metrics.get("pred_to_gt_boundary_p95_px", 1.0e6)),
                float(metrics.get("iou", 0.0)),
            )
            if best is None or rank > best_rank:
                best = (cand, cand_idx, metrics)
                best_rank = rank
        if best is None:
            raise RuntimeError(f"No candidate result found for frame {frame_name}")
        cand, cand_idx, metrics = best
        selected_sources.append(str(cand["dir"]))
        selected_scores[frame_idx] = float(metrics["_score"])

        for key, arr in cand["params"].items():
            if key not in merged_params:
                continue
            if _is_framewise_array(arr, n_frames=len(cand["params"]["frame_names"])):
                merged_params[key][frame_idx] = arr[cand_idx]
        for key, arr in cand["verts"].items():
            if key not in merged_verts:
                continue
            if _is_framewise_array(arr, n_frames=len(cand["verts"]["frame_names"])):
                merged_verts[key][frame_idx] = arr[cand_idx]
        for key, arr in cand["masks"].items():
            if key not in merged_masks:
                continue
            if _is_framewise_array(arr, n_frames=len(cand["masks"]["frame_names"])):
                merged_masks[key][frame_idx] = arr[cand_idx]
        merged_keypoints[frame_name] = cand["keypoints"][frame_name]
        _copy_frame_obj(cand["dir"], out_dir, frame_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence").mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_stage1_dir / "cameras.json", out_dir / "cameras.json")
    with (out_dir / "keypoints.json").open("w", encoding="utf-8") as f:
        json.dump(merged_keypoints, f, indent=2)

    merged_params["frame_names"] = base_frame_names
    merged_params["bestof_source_dir"] = np.asarray(selected_sources, dtype=object)
    merged_params["bestof_score"] = selected_scores
    np.savez(out_dir / "smpl_params_sequence.npz", **merged_params)

    merged_verts["frame_names"] = base_frame_names
    np.savez(out_dir / "smpl_vertices_sequence.npz", **merged_verts)

    merged_masks["frame_names"] = base_frame_names
    np.savez(out_dir / "person_masks.npz", **merged_masks)

    if (base_stage1_dir / "smpl_joints_sequence.npz").exists():
        shutil.copy2(base_stage1_dir / "smpl_joints_sequence.npz", out_dir / "smpl_joints_sequence.npz")
    if (base_stage1_dir / "smpl_skeleton_sequence.npz").exists():
        shutil.copy2(base_stage1_dir / "smpl_skeleton_sequence.npz", out_dir / "smpl_skeleton_sequence.npz")
    if (base_stage1_dir / "smplestx").exists():
        # Keep the full raw backend outputs from the primary run; best-of only replaces final stage1 assets.
        if not (out_dir / "smplestx").exists():
            shutil.copytree(base_stage1_dir / "smplestx", out_dir / "smplestx")

    summary = {
        "base_stage1_dir": str(base_stage1_dir),
        "candidate_stage1_dirs": [str(p) for p in candidate_dirs],
        "sequence_frame_count": n_frames,
        "bestof_source_counts": {
            str(k): int(v) for k, v in zip(*np.unique(np.asarray(selected_sources, dtype=object), return_counts=True))
        },
    }
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    (out_dir / "final" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
