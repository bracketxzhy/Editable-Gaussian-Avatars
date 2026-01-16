from __future__ import annotations
from pathlib import Path
import numpy as np

def dummy_fit_smpl_from_frames(
    frames_dir: str | Path,
    out_dir: str | Path,
    n_verts: int = 1000,
    seed: int = 42,
) -> dict:
    """
    最小占位：不做真实拟合，只生成可用的 smpl_params + 一个“人体样”的粗网格（球/椭球）OBJ。
    目的：先打通 stage1 -> stage2 -> render 的全流程文件接口。
    """
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 伪 SMPL 参数：pose(72), betas(10), trans(3), scale(1)
    pose = np.zeros((72,), dtype=np.float32)
    betas = np.zeros((10,), dtype=np.float32)
    trans = np.zeros((3,), dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)

    np.savez(out_dir / "smpl_params.npz", pose=pose, betas=betas, trans=trans, scale=scale)

    # 生成一个简单的“椭球点云”，再做一个非常粗的三角剖分（这里直接输出点当作顶点 + 随机三角形）
    # 注意：这只是占位 mesh，用于初始化 Gaussians（按顶点）
    verts = rng.normal(size=(n_verts, 3)).astype(np.float32)
    verts[:, 0] *= 0.25  # x
    verts[:, 1] *= 0.60  # y（身高）
    verts[:, 2] *= 0.20  # z
    verts[:, 1] += 0.2   # 上移一点

    # 随机三角面（占位）
    faces = rng.integers(low=0, high=n_verts, size=(n_verts // 3, 3), dtype=np.int32)

    mesh_path = out_dir / "smpl_mesh.obj"
    with mesh_path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            # obj face index 从 1 开始
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    return {
        "smpl_params": str(out_dir / "smpl_params.npz"),
        "smpl_mesh": str(mesh_path),
    }
