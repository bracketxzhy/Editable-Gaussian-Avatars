from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from avatar_gs.gaussian.structured_gaussian import Gaussian2D

def load_obj_vertices(obj_path: str | Path) -> np.ndarray:
    vs = []
    with Path(obj_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vs.append([float(x), float(y), float(z)])
    return np.asarray(vs, dtype=np.float32)

def project_pinhole(V: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    # world -> camera (R=I)
    X = V + t[None, :]  # (N,3)
    x = X[:, 0] / (X[:, 2] + 1e-6)
    y = X[:, 1] / (X[:, 2] + 1e-6)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)

def init_gaussians_from_mesh_2d(
    mesh_obj: str | Path,
    K_list: list[float],
    t_list: list[float],
    n_keep: int = 800,
    seed: int = 42,
) -> Gaussian2D:
    rng = np.random.default_rng(seed)
    V = load_obj_vertices(mesh_obj)
    if V.shape[0] > n_keep:
        idx = rng.choice(V.shape[0], size=n_keep, replace=False)
        V = V[idx]

    K = np.asarray(K_list, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t_list, dtype=np.float32)

    uv = project_pinhole(V, K, t)  # (N,2)

    N = uv.shape[0]
    mu = torch.from_numpy(uv)
    sigma = torch.full((N, 1), 2.0, dtype=torch.float32)
    alpha = torch.full((N, 1), 0.5, dtype=torch.float32)
    rgb = torch.rand((N, 3), dtype=torch.float32) * 0.6 + 0.2  # 初始化别太极端
    return Gaussian2D(mu=mu, sigma=sigma, alpha=alpha, rgb=rgb)
