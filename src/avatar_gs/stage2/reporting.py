from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


def default_stage2_quality_thresholds() -> dict[str, float]:
    return {
        "outer_active_outside_contour_ratio_max": 0.015,
        "outer_active_outside_person_ratio_max": 0.010,
        "outer_active_outside_alpha_gt_0_05_ratio_max": 0.002,
        "hand_leakage_ratio_max": 0.003,
        "outer_arm_nontrigger_activation_ratio_max": 0.005,
        "outer_projection_outside_anyframe_ratio_max": 0.010,
    }


def compute_outer_dynamic_caps(
    *,
    edge_vals: torch.Tensor,
    support_vals: torch.Tensor,
    outer_offset_auto_range_enable: bool,
    outer_offset_auto_range_min_scale: float,
    outer_offset_auto_range_edge_mix: float,
    shell_offset_max_t: float,
    shell_offset_max_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    e = edge_vals.detach().float().clamp(0.0, 1.0)
    s = support_vals.detach().float().clamp(0.0, 1.0)
    if not bool(outer_offset_auto_range_enable):
        t_cap = torch.full_like(e, float(shell_offset_max_t))
        n_cap = torch.full_like(e, float(shell_offset_max_n))
        return t_cap, n_cap
    min_scale = float(np.clip(float(outer_offset_auto_range_min_scale), 0.05, 1.0))
    edge_mix = float(np.clip(float(outer_offset_auto_range_edge_mix), 0.0, 1.0))
    strength = (edge_mix * e + (1.0 - edge_mix) * s).clamp(0.0, 1.0)
    scale = min_scale + (1.0 - min_scale) * strength
    t_cap = float(shell_offset_max_t) * scale
    n_cap = float(shell_offset_max_n) * scale
    return t_cap, n_cap


@dataclass(slots=True)
class Stage2QualityEvaluation:
    quality_pass: bool
    report: dict[str, object]


def evaluate_stage2_quality(
    *,
    shell_enable: bool,
    quality_thresholds: dict[str, float],
    head_detached_coverage_baseline: float | None,
    last_quality_metrics: dict[str, float] | None,
) -> Stage2QualityEvaluation:
    quality_pass = False
    quality_report: dict[str, object] = {
        "enabled": bool(shell_enable),
        "thresholds": dict(quality_thresholds),
        "head_detached_coverage_baseline": None
        if head_detached_coverage_baseline is None
        else float(head_detached_coverage_baseline),
        "final": last_quality_metrics,
        "pass": False,
    }
    if bool(shell_enable) and last_quality_metrics is not None:
        head_baseline = (
            float(head_detached_coverage_baseline)
            if head_detached_coverage_baseline is not None
            else float(last_quality_metrics["head_detached_coverage"])
        )
        quality_pass = (
            float(last_quality_metrics["outer_active_outside_contour_ratio"])
            <= float(quality_thresholds["outer_active_outside_contour_ratio_max"])
            and float(last_quality_metrics["outer_active_outside_person_ratio"])
            <= float(quality_thresholds["outer_active_outside_person_ratio_max"])
            and float(last_quality_metrics["outer_active_outside_alpha_gt_0.05_ratio"])
            <= float(quality_thresholds["outer_active_outside_alpha_gt_0_05_ratio_max"])
            and float(last_quality_metrics["hand_leakage_ratio"])
            <= float(quality_thresholds["hand_leakage_ratio_max"])
            and float(last_quality_metrics["outer_arm_nontrigger_activation_ratio"])
            <= float(quality_thresholds["outer_arm_nontrigger_activation_ratio_max"])
            and float(last_quality_metrics["outer_projection_outside_anyframe_ratio"])
            <= float(quality_thresholds["outer_projection_outside_anyframe_ratio_max"])
            and float(last_quality_metrics["head_detached_coverage"]) >= float(head_baseline)
        )
        quality_report["head_detached_coverage_baseline"] = float(head_baseline)
        quality_report["pass"] = bool(quality_pass)
    return Stage2QualityEvaluation(quality_pass=bool(quality_pass), report=quality_report)


def write_stage2_quality_log(path: Path, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def write_stage2_json_report(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_outer_face_records(
    *,
    outer_face_weight_sum: torch.Tensor | None,
    outer_face_gap_half_weighted_sum: torch.Tensor | None,
    outer_face_gap_scale_weighted_sum: torch.Tensor | None,
    outer_face_first_hit_frame_idx: list[int | None] | None,
    outer_face_contrib_frames: list[set[int]] | None,
    outer_face_spawned_mask: torch.Tensor | None,
    outer_face_spawn_iter: list[int | None] | None,
    outer_face_spawn_frame_idx: list[int | None] | None,
    frames_list: list[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if (
        outer_face_weight_sum is None
        or outer_face_gap_half_weighted_sum is None
        or outer_face_gap_scale_weighted_sum is None
        or outer_face_first_hit_frame_idx is None
        or outer_face_contrib_frames is None
        or outer_face_spawned_mask is None
        or outer_face_spawn_iter is None
        or outer_face_spawn_frame_idx is None
    ):
        return records
    face_weight_np = outer_face_weight_sum.detach().cpu().numpy().astype(np.float32)
    face_gap_half_np = outer_face_gap_half_weighted_sum.detach().cpu().numpy().astype(np.float32)
    face_gap_scale_np = outer_face_gap_scale_weighted_sum.detach().cpu().numpy().astype(np.float32)
    face_spawned_np = outer_face_spawned_mask.detach().cpu().numpy().astype(bool)
    for face_idx_i, weight_sum_i in enumerate(face_weight_np.tolist()):
        contrib_idx_i = sorted(int(v) for v in outer_face_contrib_frames[face_idx_i])
        spawned_i = bool(face_spawned_np[face_idx_i])
        if (weight_sum_i <= 0.0) and (not spawned_i) and len(contrib_idx_i) <= 0:
            continue
        denom_i = max(float(weight_sum_i), 1e-8)
        first_hit_idx_i = outer_face_first_hit_frame_idx[face_idx_i]
        spawn_frame_idx_i = outer_face_spawn_frame_idx[face_idx_i]
        records.append(
            {
                "face_idx": int(face_idx_i),
                "first_hit_frame_idx": None if first_hit_idx_i is None else int(first_hit_idx_i),
                "first_hit_frame_name": None if first_hit_idx_i is None else str(frames_list[int(first_hit_idx_i)]),
                "contrib_frame_indices": contrib_idx_i,
                "contrib_frame_names": [str(frames_list[int(v)]) for v in contrib_idx_i],
                "weight_sum": float(weight_sum_i),
                "gap_half": float(face_gap_half_np[face_idx_i] / denom_i),
                "gap_scale": float(face_gap_scale_np[face_idx_i] / denom_i),
                "spawned": spawned_i,
                "spawn_iter": None if outer_face_spawn_iter[face_idx_i] is None else int(outer_face_spawn_iter[face_idx_i]),
                "spawn_frame_idx": None if spawn_frame_idx_i is None else int(spawn_frame_idx_i),
                "spawn_frame_name": None if spawn_frame_idx_i is None else str(frames_list[int(spawn_frame_idx_i)]),
            }
        )
    return records


def build_outer_spawn_report(
    *,
    frames_list: list[str],
    outer_spawn_stats_by_frame: dict[str, dict[str, int]],
    outer_spawn_iter_records: list[dict[str, object]],
    outer_face_records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "frames": [
            {
                "frame_idx": int(i),
                "frame_name": str(fn),
                "total": int(outer_spawn_stats_by_frame.get(str(fn), {}).get("total", 0)),
                "trigger": int(outer_spawn_stats_by_frame.get(str(fn), {}).get("trigger", 0)),
                "fallback": int(outer_spawn_stats_by_frame.get(str(fn), {}).get("fallback", 0)),
                "iters_hit": int(outer_spawn_stats_by_frame.get(str(fn), {}).get("iters_hit", 0)),
            }
            for i, fn in enumerate(frames_list)
        ],
        "total_spawned": int(sum(int(v.get("total", 0)) for v in outer_spawn_stats_by_frame.values())),
        "trigger_spawned": int(sum(int(v.get("trigger", 0)) for v in outer_spawn_stats_by_frame.values())),
        "fallback_spawned": int(sum(int(v.get("fallback", 0)) for v in outer_spawn_stats_by_frame.values())),
        "iter_records": outer_spawn_iter_records,
        "face_records": outer_face_records,
    }
