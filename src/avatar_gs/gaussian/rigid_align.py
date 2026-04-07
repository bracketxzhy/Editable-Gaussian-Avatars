from __future__ import annotations

from typing import Callable
import numpy as np

from avatar_gs.utils.progress import tqdm


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
    progress_leave: bool = True,
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
