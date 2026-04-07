from __future__ import annotations

import math

import torch
import torch.nn.functional as F

def _normalize_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def quat_from_rotmat_torch(R: torch.Tensor) -> torch.Tensor:
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must be (N,3,3), got {tuple(R.shape)}")
    r00 = R[:, 0, 0]
    r11 = R[:, 1, 1]
    r22 = R[:, 2, 2]
    trace = r00 + r11 + r22

    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)
    pos = trace > 0.0
    if bool(pos.any()):
        t = torch.sqrt(trace[pos] + 1.0) * 2.0
        q[pos, 0] = 0.25 * t
        q[pos, 1] = (R[pos, 2, 1] - R[pos, 1, 2]) / t
        q[pos, 2] = (R[pos, 0, 2] - R[pos, 2, 0]) / t
        q[pos, 3] = (R[pos, 1, 0] - R[pos, 0, 1]) / t

    m0 = (~pos) & (r00 > r11) & (r00 > r22)
    if bool(m0.any()):
        t = torch.sqrt(1.0 + R[m0, 0, 0] - R[m0, 1, 1] - R[m0, 2, 2]) * 2.0
        q[m0, 0] = (R[m0, 2, 1] - R[m0, 1, 2]) / t
        q[m0, 1] = 0.25 * t
        q[m0, 2] = (R[m0, 0, 1] + R[m0, 1, 0]) / t
        q[m0, 3] = (R[m0, 0, 2] + R[m0, 2, 0]) / t

    m1 = (~pos) & (~m0) & (r11 > r22)
    if bool(m1.any()):
        t = torch.sqrt(1.0 + R[m1, 1, 1] - R[m1, 0, 0] - R[m1, 2, 2]) * 2.0
        q[m1, 0] = (R[m1, 0, 2] - R[m1, 2, 0]) / t
        q[m1, 1] = (R[m1, 0, 1] + R[m1, 1, 0]) / t
        q[m1, 2] = 0.25 * t
        q[m1, 3] = (R[m1, 1, 2] + R[m1, 2, 1]) / t

    m2 = (~pos) & (~m0) & (~m1)
    if bool(m2.any()):
        t = torch.sqrt(1.0 + R[m2, 2, 2] - R[m2, 0, 0] - R[m2, 1, 1]) * 2.0
        q[m2, 0] = (R[m2, 1, 0] - R[m2, 0, 1]) / t
        q[m2, 1] = (R[m2, 0, 2] + R[m2, 2, 0]) / t
        q[m2, 2] = (R[m2, 1, 2] + R[m2, 2, 1]) / t
        q[m2, 3] = 0.25 * t

    return _normalize_torch(q, eps=1e-8)


def slerp_quat_torch(q0: torch.Tensor, q1: torch.Tensor, w: float | torch.Tensor) -> torch.Tensor:
    if q0.shape != q1.shape:
        raise ValueError(f"q0/q1 shape mismatch: {q0.shape} vs {q1.shape}")
    q0n = _normalize_torch(q0, eps=1e-8)
    q1n = _normalize_torch(q1, eps=1e-8)
    dot = (q0n * q1n).sum(dim=1, keepdim=True)
    q1n = torch.where(dot < 0.0, -q1n, q1n)
    dot = (q0n * q1n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    if not torch.is_tensor(w):
        wt = torch.full_like(dot, float(w))
    else:
        wt = w.to(device=q0.device, dtype=q0.dtype).view(-1, 1)
        if wt.shape[0] == 1 and dot.shape[0] > 1:
            wt = wt.expand_as(dot)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    close = sin_theta.abs() < 1e-6
    s0 = torch.sin((1.0 - wt) * theta) / sin_theta.clamp(min=1e-6)
    s1 = torch.sin(wt * theta) / sin_theta.clamp(min=1e-6)
    q = s0 * q0n + s1 * q1n
    q_lin = (1.0 - wt) * q0n + wt * q1n
    q = torch.where(close.expand_as(q), q_lin, q)
    return _normalize_torch(q, eps=1e-8)


def temporal_stabilize_vertex_dirs(
    e_min_curr: torch.Tensor,
    prev_dirs: torch.Tensor | None,
    enabled: bool,
    max_angle_deg: float = 75.0,
    alpha: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    e = _normalize_torch(e_min_curr, eps=1e-8)
    if (not enabled) or (prev_dirs is None) or prev_dirs.shape != e.shape:
        return e, e.detach()

    p = _normalize_torch(prev_dirs.to(device=e.device, dtype=e.dtype), eps=1e-8)
    dot = (e * p).sum(dim=1, keepdim=True)
    e = torch.where(dot < 0.0, -e, e)
    dot2 = (e * p).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    cos_thr = math.cos(math.radians(float(max(max_angle_deg, 0.0))))
    large = dot2 < cos_thr
    blend = _normalize_torch((1.0 - float(alpha)) * p + float(alpha) * e, eps=1e-8)
    e = torch.where(large.expand_as(e), blend, e)
    return e, e.detach()


def _blend_quat_linear(q_old: torch.Tensor, q_target: torch.Tensor, blend: float) -> torch.Tensor:
    if float(blend) >= 0.999:
        return _normalize_torch(q_target, eps=1e-8)
    q_mix = (1.0 - float(blend)) * q_old + float(blend) * q_target
    return _normalize_torch(q_mix, eps=1e-8)


def _safe_barycentric(bary: torch.Tensor) -> torch.Tensor:
    b = bary.clamp(min=0.0)
    s = b.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return b / s


def build_vertex_binding_targets(
    *,
    frame,
    bind_vertex_idx: torch.Tensor,
    bind_vertex_offset: torch.Tensor | None,
    bound_ids: torch.Tensor,
    scales_old: torch.Tensor,
    quat_old: torch.Tensor,
    mu_old: torch.Tensor,
    cfg,
    prev_vertex_dirs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Shared vertex-driven fallback targets for both inner and outer layers."""
    n_verts = int(frame.verts_t.shape[0])
    v_idx = bind_vertex_idx.index_select(0, bound_ids)
    v_valid = (v_idx >= 0) & (v_idx < n_verts)

    mu_vert = mu_old.clone()
    quat_vert = _normalize_torch(quat_old.clone(), eps=1e-8)
    s_long = torch.maximum(scales_old[:, 0], scales_old[:, 1])
    s_short = torch.minimum(scales_old[:, 0], scales_old[:, 1])
    s_n = torch.full_like(s_long, float(cfg.eps_normal))
    scale_vert = torch.stack([s_long, s_short, s_n], dim=1)

    e_min_all, prev_out = temporal_stabilize_vertex_dirs(
        e_min_curr=frame.vertex_e_min_t,
        prev_dirs=prev_vertex_dirs,
        enabled=bool(cfg.temporal_frame),
        max_angle_deg=float(cfg.max_temporal_angle_deg),
        alpha=float(cfg.temporal_alpha),
    )
    n_all = _normalize_torch(frame.vertex_n_t, eps=1e-8)
    e_max_all = _normalize_torch(torch.cross(n_all, e_min_all, dim=1), eps=1e-8)
    e_min_all = _normalize_torch(torch.cross(e_max_all, n_all, dim=1), eps=1e-8)

    if bool(v_valid.any()):
        local = torch.nonzero(v_valid, as_tuple=False).view(-1)
        vsel = v_idx.index_select(0, local)
        mu_v = frame.verts_t.index_select(0, vsel)
        if bind_vertex_offset is not None:
            mu_v = mu_v + bind_vertex_offset.index_select(0, bound_ids.index_select(0, local))
        mu_vert[local] = mu_v

        e_min = e_min_all.index_select(0, vsel)
        e_max = e_max_all.index_select(0, vsel)
        n_v = n_all.index_select(0, vsel)
        R_v = torch.stack([e_min, e_max, n_v], dim=-1)
        quat_vert[local] = quat_from_rotmat_torch(R_v)

    return mu_vert, quat_vert, scale_vert, v_valid, prev_out
