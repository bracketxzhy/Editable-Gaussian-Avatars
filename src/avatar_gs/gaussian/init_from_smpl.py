from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from avatar_gs.gaussian.structured_gaussian import Gaussian3D


def load_obj_vertices(obj_path: str | Path) -> np.ndarray:
    vs = []
    with Path(obj_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vs.append([float(x), float(y), float(z)])
    if len(vs) == 0:
        raise RuntimeError(f"No vertices found in OBJ: {obj_path}")
    return np.asarray(vs, dtype=np.float32)


def init_gaussians_from_mesh_3d(
    mesh_obj: str | Path,
    n_keep: int = 3000,
    seed: int = 42,
    n_bg: int = 0,
    bg_extent: tuple[float, float, float] = (3.0, 3.0, 0.6),
    bg_center_z: float = 2.5,
    bg_log_scale: float = -3.5,
    bg_opacity_logit: float = -1.5,
    bg_exclude_bbox_pad: float = 0.0,
) -> Gaussian3D:
    rng = np.random.default_rng(seed)
    V = load_obj_vertices(mesh_obj)

    if V.shape[0] > n_keep:
        idx = rng.choice(V.shape[0], size=n_keep, replace=False)
        V = V[idx]
    else:
        idx = np.arange(V.shape[0], dtype=np.int64)

    # Background gaussians (unbound)
    if int(n_bg) > 0:
        ex, ey, ez = float(bg_extent[0]), float(bg_extent[1]), float(bg_extent[2])
        pad = float(bg_exclude_bbox_pad)
        vmin = V.min(axis=0) - pad
        vmax = V.max(axis=0) + pad

        def _sample_bg(n: int) -> np.ndarray:
            return np.stack(
                [
                    rng.uniform(-ex, ex, size=n),
                    rng.uniform(-ey, ey, size=n),
                    rng.uniform(bg_center_z - ez, bg_center_z + ez, size=n),
                ],
                axis=-1,
            ).astype(np.float32)

        # Exclude a padded body bbox to reduce ghosting around the subject.
        if pad > 0.0:
            collected = []
            remaining = int(n_bg)
            tries = 0
            while remaining > 0 and tries < 20:
                tries += 1
                cand = _sample_bg(max(remaining * 4, 256))
                inside = (
                    (cand[:, 0] >= vmin[0]) & (cand[:, 0] <= vmax[0]) &
                    (cand[:, 1] >= vmin[1]) & (cand[:, 1] <= vmax[1]) &
                    (cand[:, 2] >= vmin[2]) & (cand[:, 2] <= vmax[2])
                )
                cand = cand[~inside]
                if cand.shape[0] == 0:
                    continue
                take = min(remaining, cand.shape[0])
                collected.append(cand[:take])
                remaining -= take
            if remaining > 0:
                # Fallback: fill the rest without exclusion.
                collected.append(_sample_bg(remaining))
            V_bg = np.concatenate(collected, axis=0).astype(np.float32)
        else:
            V_bg = _sample_bg(int(n_bg))
        V = np.concatenate([V, V_bg], axis=0)
        idx_bg = -np.ones((int(n_bg),), dtype=np.int64)
        idx = np.concatenate([idx, idx_bg], axis=0)

    N = V.shape[0]
    xyz = torch.from_numpy(V)  # (N,3)
    bind_idx = torch.from_numpy(np.asarray(idx, dtype=np.int64))

    log_scales = torch.full((N, 3), float(np.log(0.01)), dtype=torch.float32)
    if int(n_bg) > 0:
        log_scales[-int(n_bg):, :] = float(bg_log_scale)
    quat = torch.zeros((N, 4), dtype=torch.float32)
    quat[:, 0] = 1.0
    opacity_logit = torch.full((N, 1), 0.0, dtype=torch.float32)
    if int(n_bg) > 0:
        opacity_logit[-int(n_bg):, :] = float(bg_opacity_logit)

    rgb = torch.rand((N, 3), dtype=torch.float32) * 0.6 + 0.2
    rgb_logit = torch.log(rgb / (1.0 - rgb))

    return Gaussian3D(
        xyz=xyz,
        log_scales=log_scales,
        quat=quat,
        opacity_logit=opacity_logit,
        rgb_logit=rgb_logit,
        bind_idx=bind_idx,
    )
