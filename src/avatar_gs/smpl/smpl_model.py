import numpy as np

class DummySMPL:
    def __init__(self, n_verts: int = 1200):
        self.n_verts = n_verts

    def forward(self, pose: np.ndarray, betas: np.ndarray):
        # 返回一个“人体样”的顶点
        v = np.random.randn(self.n_verts, 3).astype(np.float32)
        v[:,1] *= 0.6
        return v
