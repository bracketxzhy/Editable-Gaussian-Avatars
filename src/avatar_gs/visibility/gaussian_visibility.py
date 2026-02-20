from __future__ import annotations

import numpy as np
import torch


def _project_points(xyz: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # xyz: (N,3), K/R: (3,3), t: (3,)
    Xc = (xyz @ R.transpose(0, 1)) + t.view(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = torch.zeros_like(z)
    v = torch.zeros_like(z)
    if bool(valid.any()):
        xv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (xv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (xv[:, 1] / zv) + K[1, 2]
    return u, v, valid


def compute_gaussian_visibility(
    *,
    xyz: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_h: int,
    image_w: int,
    alpha_map: torch.Tensor | None = None,
    person_mask: torch.Tensor | None = None,
    renderer_meta: dict | None = None,
    min_alpha: float = 0.02,
) -> torch.Tensor:
    """Compute per-gaussian visibility in [0,1].

    Priority:
    1) renderer meta contribution if present
    2) projection + alpha/person-mask approximation
    """
    n = int(xyz.shape[0])
    dev = xyz.device
    vis = torch.ones((n,), dtype=torch.float32, device=dev)

    if renderer_meta is not None:
        # Accept common keys without forcing renderer internals.
        for key in ("gaussian_weights", "contrib", "contribution", "weights"):
            if key in renderer_meta:
                arr = renderer_meta[key]
                if torch.is_tensor(arr):
                    w = arr.to(device=dev, dtype=torch.float32).view(-1)
                    if int(w.numel()) == n:
                        return w.clamp(0.0, 1.0)
                else:
                    w_np = np.asarray(arr, dtype=np.float32).reshape(-1)
                    if int(w_np.size) == n:
                        return torch.from_numpy(w_np).to(device=dev).clamp(0.0, 1.0)

    u, v, valid = _project_points(xyz, K, R, t)
    inside = valid & (u >= 0) & (u <= float(image_w - 1)) & (v >= 0) & (v <= float(image_h - 1))
    vis = torch.where(inside, vis, torch.zeros_like(vis))

    if alpha_map is not None:
        # alpha_map: (1,H,W)
        a = alpha_map
        if a.ndim == 3 and a.shape[0] == 1:
            ai = torch.zeros((n,), dtype=torch.float32, device=dev)
            idx = torch.nonzero(inside, as_tuple=False).view(-1)
            if int(idx.numel()) > 0:
                uu = torch.round(u[idx]).long().clamp(0, image_w - 1)
                vv = torch.round(v[idx]).long().clamp(0, image_h - 1)
                ai[idx] = a[0, vv, uu]
            vis = vis * torch.clamp(ai / max(float(min_alpha), 1e-6), 0.0, 1.0)

    if person_mask is not None:
        # person_mask: (1,H,W)
        pm = person_mask
        if pm.ndim == 3 and pm.shape[0] == 1:
            mi = torch.zeros((n,), dtype=torch.float32, device=dev)
            idx = torch.nonzero(inside, as_tuple=False).view(-1)
            if int(idx.numel()) > 0:
                uu = torch.round(u[idx]).long().clamp(0, image_w - 1)
                vv = torch.round(v[idx]).long().clamp(0, image_h - 1)
                mi[idx] = pm[0, vv, uu]
            vis = vis * mi.clamp(0.0, 1.0)

    return vis.clamp(0.0, 1.0)
