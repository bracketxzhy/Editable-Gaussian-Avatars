from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

# —— SMPL 简化骨架映射（MediaPipe → SMPL joints 子集）
# 这里只选 torso + limbs，足够稳定
MP_TO_SMPL = {
    11: 2,  # left shoulder
    12: 5,  # right shoulder
    23: 9,  # left hip
    24: 12, # right hip
}

def project_points(V, K, t):
    X = V + t[None, :]
    x = X[:, 0] / (X[:, 2] + 1e-6)
    y = X[:, 1] / (X[:, 2] + 1e-6)
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return np.stack([u, v], axis=1)

def fit_smpl_from_keypoints(
    keypoints_json: str | Path,
    out_dir: str | Path,
    K_list: list[float],
    n_verts: int = 1200,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(keypoints_json, "r", encoding="utf-8") as f:
        kpts_all = json.load(f)

    # 用第一帧
    first = next(iter(kpts_all.values()))
    kpts = np.array(first, dtype=np.float32)

    # 伪 SMPL mesh（你后面会换成真实 smplx）
    rng = np.random.default_rng(0)
    V = rng.normal(size=(n_verts, 3)).astype(np.float32)
    V[:,1] *= 0.6
    V[:,2] += 2.5

    K = np.asarray(K_list, dtype=np.float32).reshape(3,3)

    def residual(x):
        tx, ty, tz = x
        proj = project_points(V, K, np.array([tx, ty, tz], dtype=np.float32))
        res = []
        for mp_idx, _ in MP_TO_SMPL.items():
            if kpts[mp_idx,2] < 0.5:
                continue
            res.append(proj[mp_idx] - kpts[mp_idx,:2])
        return np.concatenate(res, axis=0)

    x0 = np.array([0,0,2.5], dtype=np.float32)
    sol = least_squares(residual, x0, verbose=0)

    pose = np.zeros((72,), dtype=np.float32)
    betas = np.zeros((10,), dtype=np.float32)
    trans = sol.x.astype(np.float32)
    scale = np.array([1.0], dtype=np.float32)

    np.savez(out_dir / "smpl_params.npz",
             pose=pose, betas=betas, trans=trans, scale=scale)

    # 输出 mesh
    with (out_dir / "smpl_mesh.obj").open("w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    return out_dir