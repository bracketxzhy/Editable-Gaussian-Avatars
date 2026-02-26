from __future__ import annotations

import torch


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
    if trigger_pixels < max(int(outer_trigger_min_pixels), 0):
        return None, None, trigger_pixels

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

