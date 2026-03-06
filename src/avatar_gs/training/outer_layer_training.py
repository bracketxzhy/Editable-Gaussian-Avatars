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
    outer_detached_solid_enable: bool,
    outer_detached_solid_dilate_px: int,
    outer_detached_solid_close_px: int,
    outer_detached_solid_mix_weight: float,
    dilate_fn,
    erode_fn,
    contour_band_fn,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, torch.Tensor | None]:
    if outer_base_mask is None:
        return None, None, 0, None

    outer_base_mask = outer_base_mask.clamp(0.0, 1.0)
    outer_trigger_mask = outer_base_mask
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
        relaxed = (outer_base_mask * (1.0 - smpl_ref.clamp(0.0, 1.0))).clamp(0.0, 1.0)
        if int(outer_trigger_ring_width_px) > 0:
            relaxed = contour_band_fn(relaxed, width_px=1).clamp(0.0, 1.0)
        relaxed_pixels = int((relaxed > 0.5).sum().item())
        if relaxed_pixels > trigger_pixels:
            outer_trigger_mask = relaxed
            trigger_pixels = relaxed_pixels

    outer_supervision_mask = None
    if trigger_pixels > 0:
        outer_supervision_mask = outer_trigger_mask
        if bool(outer_contour_inside_only):
            outer_supervision_mask = (outer_supervision_mask * outer_base_mask).clamp(0.0, 1.0)

    detached_solid = None
    if bool(outer_detached_solid_enable) and smpl_ref is not None:
        smpl_ref_gap = dilate_fn(
            smpl_ref,
            max(int(outer_trigger_smpl_gap_px), 0),
        )
        detached_solid = (outer_base_mask * (1.0 - smpl_ref_gap)).clamp(0.0, 1.0)
        if int(outer_detached_solid_dilate_px) > 0:
            detached_solid = dilate_fn(detached_solid, int(outer_detached_solid_dilate_px)).clamp(0.0, 1.0)
        if int(outer_detached_solid_close_px) > 0:
            detached_solid = dilate_fn(detached_solid, int(outer_detached_solid_close_px))
            detached_solid = erode_fn(detached_solid, int(outer_detached_solid_close_px)).clamp(0.0, 1.0)
        if bool(outer_contour_inside_only):
            detached_solid = (detached_solid * outer_base_mask).clamp(0.0, 1.0)
        if int((detached_solid > 0.5).sum().item()) <= 0:
            detached_solid = None

    if detached_solid is not None:
        if outer_supervision_mask is None:
            outer_supervision_mask = detached_solid
        else:
            mix_w = float(max(0.0, min(1.0, float(outer_detached_solid_mix_weight))))
            if mix_w >= 1.0:
                outer_supervision_mask = detached_solid
            elif mix_w > 0.0:
                outer_supervision_mask = (
                    (1.0 - mix_w) * outer_supervision_mask + mix_w * detached_solid
                ).clamp(0.0, 1.0)

    if outer_supervision_mask is None:
        return None, None, 0, detached_solid
    trigger_pixels = int((outer_supervision_mask > 0.5).sum().item())
    if trigger_pixels <= 0:
        return None, None, 0, detached_solid

    edge_trigger_map_curr = None
    if bool(outer_edge_trigger_enable):
        if edge_map_curr is not None:
            edge_trigger_map_curr = (outer_supervision_mask * edge_map_curr).clamp(0.0, 1.0)
        else:
            edge_trigger_map_curr = outer_supervision_mask

    return outer_supervision_mask, edge_trigger_map_curr, trigger_pixels, detached_solid


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


def estimate_face_gap_targets(
    *,
    outer_mask: torch.Tensor | None,
    smpl_mask: torch.Tensor | None,
    verts_t: torch.Tensor | None,
    faces_t: torch.Tensor | None,
    face_n_t: torch.Tensor | None,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_h: int,
    image_w: int,
    max_search_px: int = 24,
    step_px: float = 1.0,
    normal_probe_world: float = 0.01,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if (
        outer_mask is None
        or smpl_mask is None
        or verts_t is None
        or faces_t is None
        or face_n_t is None
    ):
        return None, None, None
    if int(faces_t.shape[0]) <= 0:
        return None, None, None
    if outer_mask.ndim != 3 or smpl_mask.ndim != 3:
        return None, None, None
    steps_n = max(int(max_search_px), 1)
    step_px = float(max(step_px, 0.5))
    probe_world = float(max(normal_probe_world, 1e-4))

    tri = verts_t.index_select(0, faces_t.reshape(-1)).reshape(-1, 3, 3)
    centroids = tri.mean(dim=1)
    probe_pts = centroids + probe_world * face_n_t

    def _project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Xc = pts @ R.T + t.view(1, 3)
        z = Xc[:, 2]
        valid = z > 1e-6
        u = K[0, 0] * (Xc[:, 0] / z.clamp_min(1e-6)) + K[0, 2]
        v = K[1, 1] * (Xc[:, 1] / z.clamp_min(1e-6)) + K[1, 2]
        return u, v, valid

    u0, v0, valid0 = _project(centroids)
    u1, v1, valid1 = _project(probe_pts)
    du = u1 - u0
    dv = v1 - v0
    pix_per_world = torch.sqrt(du * du + dv * dv).clamp_min(1e-8) / probe_world
    dir_norm = torch.sqrt(du * du + dv * dv).clamp_min(1e-8)
    dir_u = du / dir_norm
    dir_v = dv / dir_norm

    ui0 = u0.round().to(dtype=torch.long)
    vi0 = v0.round().to(dtype=torch.long)
    inside0 = (
        valid0
        & valid1
        & (ui0 >= 0)
        & (ui0 < int(image_w))
        & (vi0 >= 0)
        & (vi0 < int(image_h))
    )
    if not bool(inside0.any()):
        z = torch.zeros((int(faces_t.shape[0]),), dtype=torch.float32, device=verts_t.device)
        return z, z, z

    smpl_center = torch.zeros((int(faces_t.shape[0]),), dtype=torch.float32, device=verts_t.device)
    outer_center = torch.zeros((int(faces_t.shape[0]),), dtype=torch.float32, device=verts_t.device)
    inside_idx = torch.nonzero(inside0, as_tuple=False).view(-1)
    smpl_center[inside_idx] = smpl_mask[0, vi0[inside_idx], ui0[inside_idx]]
    outer_center[inside_idx] = outer_mask[0, vi0[inside_idx], ui0[inside_idx]]
    ray_valid = inside0 & (smpl_center > 0.5) & (outer_center > 0.5)

    steps = torch.arange(
        1,
        steps_n + 1,
        device=verts_t.device,
        dtype=torch.float32,
    ).view(1, -1) * step_px

    def _sample_ray(mask: torch.Tensor, sign: float) -> torch.Tensor:
        uu = u0.unsqueeze(1) + sign * dir_u.unsqueeze(1) * steps
        vv = v0.unsqueeze(1) + sign * dir_v.unsqueeze(1) * steps
        ui = uu.round().to(dtype=torch.long)
        vi = vv.round().to(dtype=torch.long)
        inside = (
            ray_valid.unsqueeze(1)
            & (ui >= 0)
            & (ui < int(image_w))
            & (vi >= 0)
            & (vi < int(image_h))
        )
        out = torch.zeros_like(uu, dtype=torch.float32)
        if bool(inside.any()):
            out[inside] = mask[0, vi[inside], ui[inside]]
        return out

    def _first_exit(samples: torch.Tensor) -> torch.Tensor:
        inside = samples > 0.5
        exit_mask = ~inside
        any_exit = exit_mask.any(dim=1)
        idx = torch.full(
            (samples.shape[0],),
            float(steps_n + 1) * step_px,
            dtype=torch.float32,
            device=samples.device,
        )
        if bool(any_exit.any()):
            idx[any_exit] = (exit_mask[any_exit].float().argmax(dim=1).float() + 1.0) * step_px
        return idx

    smpl_pos = _sample_ray(smpl_mask, sign=1.0)
    smpl_neg = _sample_ray(smpl_mask, sign=-1.0)
    outer_pos = _sample_ray(outer_mask, sign=1.0)
    outer_neg = _sample_ray(outer_mask, sign=-1.0)

    smpl_exit_pos = _first_exit(smpl_pos)
    smpl_exit_neg = _first_exit(smpl_neg)
    outer_exit_pos = _first_exit(outer_pos)
    outer_exit_neg = _first_exit(outer_neg)

    gap_pos_px = (outer_exit_pos - smpl_exit_pos).clamp_min(0.0)
    gap_neg_px = (outer_exit_neg - smpl_exit_neg).clamp_min(0.0)
    choose_pos = gap_pos_px >= gap_neg_px
    gap_px = torch.where(choose_pos, gap_pos_px, gap_neg_px)
    gap_support = (gap_px > 0.0).float() * ray_valid.float()
    gap_world = gap_px / pix_per_world.clamp_min(1e-6)
    gap_half_world = 0.5 * gap_world
    gap_half_world = gap_half_world * gap_support
    gap_world = gap_world * gap_support
    gap_support = gap_support.clamp(0.0, 1.0)
    return gap_half_world, gap_world.clamp_min(0.0), gap_support
