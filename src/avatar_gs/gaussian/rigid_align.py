from __future__ import annotations

from typing import Callable
import numpy as np
from tqdm import tqdm


def _as_float3(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return default
    return default


def normalize_rigid_align_config(cfg, default_enabled: bool = False) -> dict | None:
    """
    Normalize flexible rigid-align config into a stable dictionary.
    Returns None when cfg is None / unsupported.
    """
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        return {
            "enabled": bool(cfg),
            "auto": False,
            "pre_stage2": True,
            "require_pre_stage2": True,
            "require_quality": True,
            "min_iou_after": 0.70,
            "max_outside_after": 0.20,
            "init_from_bbox": True,
            "use_best_if_quality_fail": True,
            "rx_deg": 0.0,
            "ry_deg": 0.0,
            "rz_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "sample_frames": 8,
            "eval_downsample": 0.5,
            "outside_weight": 2.0,
            "levels": 3,
            "iters_per_level": 2,
            "rot_step_deg": 3.0,
            "trans_step": (0.04, 0.04, 0.06),
            "max_rot_deg": 20.0,
            "max_trans": (0.35, 0.35, 0.45),
            "person_masks_npz": None,
        }
    if not isinstance(cfg, dict):
        return None

    # Support ckpt payload style: {"enabled":..., "params": {...}, ...}
    base = dict(cfg)
    nested_params = base.get("params", None)
    if isinstance(nested_params, dict):
        merged = dict(nested_params)
        for k, v in base.items():
            if k != "params":
                merged[k] = v
        base = merged

    rot_default = (
        float(base.get("rx_deg", 0.0)),
        float(base.get("ry_deg", 0.0)),
        float(base.get("rz_deg", 0.0)),
    )
    trans_default = (
        float(base.get("tx", 0.0)),
        float(base.get("ty", 0.0)),
        float(base.get("tz", 0.0)),
    )
    rx, ry, rz = _as_float3(base.get("rot_deg", None), rot_default)
    tx, ty, tz = _as_float3(base.get("trans", None), trans_default)

    out = {
        "enabled": bool(base.get("enabled", base.get("enable", default_enabled))),
        "auto": bool(base.get("auto", base.get("auto_enable", False))),
        "pre_stage2": bool(base.get("pre_stage2", True)),
        "require_pre_stage2": bool(base.get("require_pre_stage2", True)),
        "require_quality": bool(base.get("require_quality", True)),
        "min_iou_after": float(base.get("min_iou_after", 0.70)),
        "max_outside_after": float(base.get("max_outside_after", 0.20)),
        "init_from_bbox": bool(base.get("init_from_bbox", True)),
        "use_best_if_quality_fail": bool(base.get("use_best_if_quality_fail", True)),
        "rx_deg": float(rx),
        "ry_deg": float(ry),
        "rz_deg": float(rz),
        "tx": float(tx),
        "ty": float(ty),
        "tz": float(tz),
        "sample_frames": max(1, int(base.get("sample_frames", 8))),
        "eval_downsample": float(np.clip(float(base.get("eval_downsample", 0.5)), 0.2, 1.0)),
        "outside_weight": float(max(0.0, float(base.get("outside_weight", 2.0)))),
        "levels": max(1, int(base.get("levels", 3))),
        "iters_per_level": max(1, int(base.get("iters_per_level", 2))),
        "rot_step_deg": float(max(0.05, float(base.get("rot_step_deg", 3.0)))),
        "trans_step": _as_float3(base.get("trans_step", None), (0.04, 0.04, 0.06)),
        "max_rot_deg": float(max(1.0, float(base.get("max_rot_deg", 20.0)))),
        "max_trans": _as_float3(base.get("max_trans", None), (0.35, 0.35, 0.45)),
        "person_masks_npz": None if base.get("person_masks_npz", None) is None else str(base.get("person_masks_npz")),
    }
    return out


def euler_deg_to_rotmat_np(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build rotation matrix from XYZ Euler angles in degrees."""
    rx = float(np.deg2rad(rx_deg))
    ry = float(np.deg2rad(ry_deg))
    rz = float(np.deg2rad(rz_deg))

    sx, cx = float(np.sin(rx)), float(np.cos(rx))
    sy, cy = float(np.sin(ry)), float(np.cos(ry))
    sz, cz = float(np.sin(rz)), float(np.cos(rz))

    rx_m = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    ry_m = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rz_m = np.asarray(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    # Column-vector convention: v' = R * v. We apply row vectors as v @ R.T.
    return (rz_m @ ry_m @ rx_m).astype(np.float32)


def apply_rigid_transform_np(verts: np.ndarray, cfg: dict | None) -> np.ndarray:
    if cfg is None or not bool(cfg.get("enabled", False)):
        return np.asarray(verts, dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32)
    R = euler_deg_to_rotmat_np(
        float(cfg.get("rx_deg", 0.0)),
        float(cfg.get("ry_deg", 0.0)),
        float(cfg.get("rz_deg", 0.0)),
    )
    t = np.asarray(
        [
            float(cfg.get("tx", 0.0)),
            float(cfg.get("ty", 0.0)),
            float(cfg.get("tz", 0.0)),
        ],
        dtype=np.float32,
    ).reshape(1, 3)
    return (v @ R.T + t).astype(np.float32)


def coordinate_descent_rigid_search(
    score_fn: Callable[[dict], float],
    init_params: dict,
    rot_step_deg: float = 3.0,
    trans_step: tuple[float, float, float] = (0.04, 0.04, 0.06),
    levels: int = 3,
    iters_per_level: int = 2,
    max_rot_deg: float = 20.0,
    max_trans: tuple[float, float, float] = (0.35, 0.35, 0.45),
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_leave: bool = False,
) -> tuple[dict, float, float]:
    """
    Coordinate-descent search on rigid params:
    [rx_deg, ry_deg, rz_deg, tx, ty, tz].
    """
    cur = {
        "rx_deg": float(init_params.get("rx_deg", 0.0)),
        "ry_deg": float(init_params.get("ry_deg", 0.0)),
        "rz_deg": float(init_params.get("rz_deg", 0.0)),
        "tx": float(init_params.get("tx", 0.0)),
        "ty": float(init_params.get("ty", 0.0)),
        "tz": float(init_params.get("tz", 0.0)),
    }
    cur_score = float(score_fn(cur))
    init_score = float(cur_score)

    step_rot = float(max(0.05, rot_step_deg))
    step_t = [
        float(max(1e-4, trans_step[0])),
        float(max(1e-4, trans_step[1])),
        float(max(1e-4, trans_step[2])),
    ]
    max_rot = float(max(1.0, max_rot_deg))
    max_t = [
        float(max(1e-4, max_trans[0])),
        float(max(1e-4, max_trans[1])),
        float(max(1e-4, max_trans[2])),
    ]

    dims = [
        ("rx_deg", lambda s: float(np.clip(s, -max_rot, max_rot))),
        ("ry_deg", lambda s: float(np.clip(s, -max_rot, max_rot))),
        ("rz_deg", lambda s: float(np.clip(s, -max_rot, max_rot))),
        ("tx", lambda s: float(np.clip(s, -max_t[0], max_t[0]))),
        ("ty", lambda s: float(np.clip(s, -max_t[1], max_t[1]))),
        ("tz", lambda s: float(np.clip(s, -max_t[2], max_t[2]))),
    ]

    n_levels = int(max(1, levels))
    n_inner = int(max(1, iters_per_level))
    total_steps = n_levels * n_inner * len(dims) * 2
    pbar = tqdm(
        total=total_steps,
        desc=(progress_desc or "Rigid align search"),
        leave=bool(progress_leave),
        disable=(not bool(show_progress)),
    )
    try:
        for _lvl in range(n_levels):
            for _inner in range(n_inner):
                improved = False
                for di, (key, clamp_fn) in enumerate(dims):
                    step = step_rot if di < 3 else step_t[di - 3]
                    best_local = cur_score
                    best_value = cur[key]
                    for sgn in (-1.0, 1.0):
                        cand = dict(cur)
                        cand[key] = clamp_fn(cand[key] + sgn * step)
                        pbar.update(1)
                        if float(cand[key]) == float(cur[key]):
                            continue
                        s = float(score_fn(cand))
                        if s > best_local + 1e-9:
                            best_local = s
                            best_value = float(cand[key])
                            if bool(show_progress):
                                pbar.set_postfix(score=f"{best_local:.4f}", key=key)
                    if best_value != cur[key]:
                        cur[key] = best_value
                        cur_score = best_local
                        improved = True
                if not improved:
                    break
            step_rot *= 0.5
            step_t = [x * 0.5 for x in step_t]
    finally:
        pbar.close()

    return cur, init_score, cur_score
