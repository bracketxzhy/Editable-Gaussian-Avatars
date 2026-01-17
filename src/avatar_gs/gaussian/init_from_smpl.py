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
) -> Gaussian3D:
    rng = np.random.default_rng(seed)
    V = load_obj_vertices(mesh_obj)

    if V.shape[0] > n_keep:
        idx = rng.choice(V.shape[0], size=n_keep, replace=False)
        V = V[idx]

    N = V.shape[0]
    xyz = torch.from_numpy(V)  # (N,3)

    log_scales = torch.full((N, 3), float(np.log(0.01)), dtype=torch.float32)
    quat = torch.zeros((N, 4), dtype=torch.float32)
    quat[:, 0] = 1.0
    opacity_logit = torch.full((N, 1), 0.0, dtype=torch.float32)

    rgb = torch.rand((N, 3), dtype=torch.float32) * 0.6 + 0.2
    rgb_logit = torch.log(rgb / (1.0 - rgb))

    return Gaussian3D(
        xyz=xyz,
        log_scales=log_scales,
        quat=quat,
        opacity_logit=opacity_logit,
        rgb_logit=rgb_logit,
    )
