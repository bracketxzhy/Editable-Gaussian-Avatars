from __future__ import annotations

import cv2
import numpy as np
import torch


def build_border_attenuation(mask: np.ndarray, width: int) -> np.ndarray:
    """Build per-pixel attenuation map in [0,1], lower near mask border."""
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got {m.shape}")
    w = max(int(width), 0)
    if w == 0:
        return np.ones_like(m, dtype=np.float32)

    # distance inside mask to nearest boundary.
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    att = np.clip(dist / float(w), 0.0, 1.0)
    att = np.where(m > 0, att, 0.0).astype(np.float32)
    return att


def sample_border_weight(
    attenuation_map: np.ndarray | torch.Tensor,
    uv: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sample attenuation map at projected uv for each gaussian."""
    if not torch.is_tensor(uv):
        raise TypeError("uv must be torch.Tensor")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must be (N,2), got {uv.shape}")
    if valid.ndim != 1 or valid.shape[0] != uv.shape[0]:
        raise ValueError("valid shape mismatch")

    dev = uv.device
    n = int(uv.shape[0])
    out = torch.zeros((n,), dtype=torch.float32, device=dev)

    if torch.is_tensor(attenuation_map):
        att = attenuation_map.to(device=dev, dtype=torch.float32)
        if att.ndim != 2:
            raise ValueError(f"attenuation_map tensor must be (H,W), got {att.shape}")
        H, W = int(att.shape[0]), int(att.shape[1])
        idx = torch.nonzero(valid, as_tuple=False).view(-1)
        if int(idx.numel()) > 0:
            uu = torch.round(uv[idx, 0]).long().clamp(0, W - 1)
            vv = torch.round(uv[idx, 1]).long().clamp(0, H - 1)
            out[idx] = att[vv, uu]
        return out.clamp(0.0, 1.0)

    att_np = np.asarray(attenuation_map, dtype=np.float32)
    if att_np.ndim != 2:
        raise ValueError(f"attenuation_map must be (H,W), got {att_np.shape}")
    H, W = int(att_np.shape[0]), int(att_np.shape[1])
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if int(idx.numel()) > 0:
        uu = torch.round(uv[idx, 0]).long().clamp(0, W - 1).cpu().numpy()
        vv = torch.round(uv[idx, 1]).long().clamp(0, H - 1).cpu().numpy()
        vals = att_np[vv, uu]
        out[idx] = torch.from_numpy(vals).to(device=dev, dtype=torch.float32)
    return out.clamp(0.0, 1.0)
