from __future__ import annotations

import torch
import torch.nn.functional as F


def build_outer_trigger_mask(
    *,
    outer_base_mask: torch.Tensor | None,
    smpl_ref: torch.Tensor | None,
    edge_map_curr: torch.Tensor | None,
    outer_trigger_smpl_gap_px: int,
    outer_trigger_edge_th: float,
    outer_trigger_min_pixels: int,
    outer_trigger_use_edge_gate: bool,
    outer_trigger_ring_width_px: int,
    outer_contour_inside_only: bool,
    outer_edge_trigger_enable: bool,
    dilate_fn,
    contour_band_fn,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    if outer_base_mask is None:
        return None, None, 0

    outer_trigger_mask = outer_base_mask.clamp(0.0, 1.0)
    if smpl_ref is not None:
        smpl_ref_gap = dilate_fn(
            smpl_ref,
            max(int(outer_trigger_smpl_gap_px), 0),
        )
        outer_trigger_mask = (outer_trigger_mask * (1.0 - smpl_ref_gap)).clamp(0.0, 1.0)

    if int(outer_trigger_ring_width_px) > 0:
        outer_trigger_mask = contour_band_fn(outer_trigger_mask, width_px=int(outer_trigger_ring_width_px)).clamp(0.0, 1.0)

    if bool(outer_trigger_use_edge_gate):
        if edge_map_curr is None:
            edge_map_curr = torch.zeros_like(outer_trigger_mask)
        edge_gate = (edge_map_curr > float(max(outer_trigger_edge_th, 0.0))).float()
        outer_trigger_mask = (outer_trigger_mask * edge_gate).clamp(0.0, 1.0)

    trigger_pixels = int((outer_trigger_mask > 0.5).sum().item())
    min_px = max(int(outer_trigger_min_pixels), 0)
    if trigger_pixels < min_px and smpl_ref is not None and int(outer_trigger_smpl_gap_px) > 0:
        relaxed = (outer_base_mask.clamp(0.0, 1.0) * (1.0 - smpl_ref.clamp(0.0, 1.0))).clamp(0.0, 1.0)
        if int(outer_trigger_ring_width_px) > 0:
            relaxed = contour_band_fn(relaxed, width_px=1).clamp(0.0, 1.0)
        relaxed_pixels = int((relaxed > 0.5).sum().item())
        if relaxed_pixels > trigger_pixels:
            outer_trigger_mask = relaxed
            trigger_pixels = relaxed_pixels
    if trigger_pixels <= 0:
        return None, None, 0

    outer_supervision_mask = outer_trigger_mask
    if bool(outer_contour_inside_only):
        outer_supervision_mask = (outer_supervision_mask * outer_base_mask).clamp(0.0, 1.0)

    edge_trigger_map_curr = None
    if bool(outer_edge_trigger_enable):
        if edge_map_curr is not None:
            edge_trigger_map_curr = (outer_supervision_mask * edge_map_curr).clamp(0.0, 1.0)
        else:
            edge_trigger_map_curr = outer_supervision_mask

    return outer_supervision_mask, edge_trigger_map_curr, trigger_pixels


def map_trigger_ring_to_faces(
    *,
    trigger_ring_mask: torch.Tensor | None,
    verts_t: torch.Tensor | None,
    faces_t: torch.Tensor | None,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_h: int,
    image_w: int,
    sample_dilate_px: int = 0,
) -> tuple[torch.Tensor | None, int]:
    if trigger_ring_mask is None or verts_t is None or faces_t is None:
        return None, 0
    if trigger_ring_mask.ndim != 3:
        return None, 0
    if int(faces_t.shape[0]) <= 0:
        return None, 0

    mask = trigger_ring_mask.clamp(0.0, 1.0)
    if int(sample_dilate_px) > 0:
        k = int(sample_dilate_px) * 2 + 1
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=int(sample_dilate_px)).squeeze(0)
    trigger_pixels = int((mask > 0.5).sum().item())
    if trigger_pixels <= 0:
        return None, 0

    tri = verts_t.index_select(0, faces_t.reshape(-1)).reshape(-1, 3, 3)
    centroids = tri.mean(dim=1)
    Xc = centroids @ R.T + t.view(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    if not bool(valid.any()):
        return None, trigger_pixels
    u = K[0, 0] * (Xc[:, 0] / z.clamp_min(1e-6)) + K[0, 2]
    v = K[1, 1] * (Xc[:, 1] / z.clamp_min(1e-6)) + K[1, 2]
    ui = u.round().to(dtype=torch.long)
    vi = v.round().to(dtype=torch.long)
    inside = valid & (ui >= 0) & (ui < int(image_w)) & (vi >= 0) & (vi < int(image_h))
    if not bool(inside.any()):
        return None, trigger_pixels

    sample = torch.zeros((int(faces_t.shape[0]),), dtype=torch.float32, device=verts_t.device)
    inside_idx = torch.nonzero(inside, as_tuple=False).view(-1)
    sample[inside_idx] = mask[0, vi[inside_idx], ui[inside_idx]]
    return sample, trigger_pixels
