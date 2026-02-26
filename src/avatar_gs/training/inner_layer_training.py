from __future__ import annotations

import torch


def resolve_inner_supervision_mask(
    *,
    strict_inner_mask: bool,
    smpl_mask: torch.Tensor | None,
    smpl_mask_for_color: torch.Tensor | None,
    person_mask_curr: torch.Tensor | None,
) -> torch.Tensor | None:
    if bool(strict_inner_mask) and smpl_mask is not None:
        return smpl_mask
    return smpl_mask_for_color if smpl_mask_for_color is not None else person_mask_curr

