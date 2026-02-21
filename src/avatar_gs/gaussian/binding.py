from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from avatar_gs.gaussian.structured_gaussian import Gaussian3D


def _normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return x / n


def _normalize_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def build_vertex_adjacency(n_verts: int, faces: np.ndarray | None) -> list[np.ndarray]:
    if faces is None or len(faces) == 0 or int(n_verts) <= 0:
        return [np.zeros((0,), dtype=np.int64) for _ in range(max(int(n_verts), 0))]
    adj: list[set[int]] = [set() for _ in range(int(n_verts))]
    f = np.asarray(faces, dtype=np.int64)
    for tri in f:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if i0 < 0 or i1 < 0 or i2 < 0:
            continue
        if i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
            continue
        adj[i0].add(i1)
        adj[i0].add(i2)
        adj[i1].add(i0)
        adj[i1].add(i2)
        adj[i2].add(i0)
        adj[i2].add(i1)
    return [np.asarray(sorted(list(s)), dtype=np.int64) for s in adj]


def compute_area_weighted_vertex_normals_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
    eps: float = 1e-8,
) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float32)
    n = np.zeros_like(v, dtype=np.float32)
    if faces is None or len(faces) == 0 or v.shape[0] == 0:
        return n
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]  # (F,3,3)
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    fn = np.cross(e1, e2).astype(np.float32)  # area-weighted unnormalized face normal
    fn_norm = np.linalg.norm(fn, axis=1)
    valid = fn_norm > float(eps)
    fv = f[valid]
    fnv = fn[valid]
    for k in range(3):
        np.add.at(n, fv[:, k], fnv)
    n = _normalize_np(n, eps=eps).astype(np.float32)
    return n


def compute_face_frames_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if faces is None or len(faces) == 0:
        z = np.zeros((0, 3), dtype=np.float32)
        s = np.zeros((0,), dtype=np.float32)
        return z, z, z, s, s

    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]  # (F,3,3)
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    u = np.cross(e1, e2).astype(np.float32)
    n_f = _normalize_np(u, eps=eps).astype(np.float32)

    proj1 = e1 - (np.sum(e1 * n_f, axis=1, keepdims=True) * n_f)
    proj2 = e2 - (np.sum(e2 * n_f, axis=1, keepdims=True) * n_f)
    norm_proj1 = np.linalg.norm(proj1, axis=1)
    use_proj2 = norm_proj1 < float(eps)
    t1 = proj1.copy()
    if np.any(use_proj2):
        t1[use_proj2] = proj2[use_proj2]
    t1 = _normalize_np(t1, eps=eps).astype(np.float32)

    t2 = np.cross(n_f, t1).astype(np.float32)
    t2 = _normalize_np(t2, eps=eps).astype(np.float32)
    # Re-orthogonalize t1 to avoid drift.
    t1 = np.cross(t2, n_f).astype(np.float32)
    t1 = _normalize_np(t1, eps=eps).astype(np.float32)

    # Geometry-derived tangent scales.
    s_t1 = 0.5 * np.linalg.norm(e1, axis=1).astype(np.float32)
    s_t2 = np.abs(np.sum(e2 * t2, axis=1)).astype(np.float32)
    s_t2 = np.maximum(0.5 * s_t2, float(eps)).astype(np.float32)
    s_t1 = np.maximum(s_t1, float(eps)).astype(np.float32)
    return t1, t2, n_f, s_t1, s_t2


def compute_vertex_curvature_frames_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
    adjacency: list[np.ndarray],
    curv_conf_tau: float = 0.2,
    curv_degen_tau: float = 1e-4,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(verts, dtype=np.float32)
    n_verts = int(v.shape[0])
    n_v = compute_area_weighted_vertex_normals_np(v, faces, eps=eps)
    e_min = np.zeros((n_verts, 3), dtype=np.float32)
    e_max = np.zeros((n_verts, 3), dtype=np.float32)
    conf_arr = np.zeros((n_verts,), dtype=np.float32)
    reliable = np.zeros((n_verts,), dtype=np.uint8)

    g0 = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    g1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    for vi in range(n_verts):
        nv = n_v[vi]
        if not np.isfinite(nv).all():
            nv = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if float(np.linalg.norm(nv)) < float(eps):
            nv = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        nv = _normalize_np(nv.reshape(1, 3), eps=eps)[0]

        # Stable baseline tangent.
        g = g1 if abs(float(np.dot(g0, nv))) > 0.9 else g0
        t_base = g - float(np.dot(g, nv)) * nv
        if float(np.linalg.norm(t_base)) < float(eps):
            g = g0 if np.allclose(g, g1) else g1
            t_base = g - float(np.dot(g, nv)) * nv
        t_base = _normalize_np(t_base.reshape(1, 3), eps=eps)[0]
        b2 = np.cross(nv, t_base).astype(np.float32)
        b2 = _normalize_np(b2.reshape(1, 3), eps=eps)[0]

        neigh = adjacency[vi] if vi < len(adjacency) else np.zeros((0,), dtype=np.int64)
        use_curv = False
        e_curv = t_base
        if neigh.size >= 6:
            p = v[neigh] - v[vi]
            x = p @ t_base
            y = p @ b2
            z = p @ nv
            A = np.stack([0.5 * x * x, x * y, 0.5 * y * y], axis=1).astype(np.float32)
            # Require enough rank to fit Hessian.
            if np.linalg.matrix_rank(A) >= 3:
                try:
                    coeff, *_ = np.linalg.lstsq(A, z.astype(np.float32), rcond=None)
                    a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
                    H = np.asarray([[a, b], [b, c]], dtype=np.float32)
                    evals, evecs = np.linalg.eigh(H)
                    # Direction with smaller absolute curvature.
                    idx = int(np.argmin(np.abs(evals)))
                    u_min = evecs[:, idx]
                    e_curv = (float(u_min[0]) * t_base + float(u_min[1]) * b2).astype(np.float32)
                    e_curv = _normalize_np(e_curv.reshape(1, 3), eps=eps)[0]
                    k0, k1 = float(evals[0]), float(evals[1])
                    conf = abs(k0 - k1) / (abs(k0) + abs(k1) + float(eps))
                    conf_arr[vi] = float(conf)
                    if abs(k0 - k1) >= float(curv_degen_tau) and conf >= float(curv_conf_tau):
                        use_curv = True
                except np.linalg.LinAlgError:
                    use_curv = False

        e1 = e_curv if use_curv else t_base
        e2 = np.cross(nv, e1).astype(np.float32)
        e2 = _normalize_np(e2.reshape(1, 3), eps=eps)[0]
        e1 = np.cross(e2, nv).astype(np.float32)
        e1 = _normalize_np(e1.reshape(1, 3), eps=eps)[0]

        e_min[vi] = e1
        e_max[vi] = e2
        reliable[vi] = 1 if use_curv else 0

    return e_min, e_max, n_v, conf_arr, reliable


def compute_vertex_baseline_frames_np(
    vertex_normals: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    n_v = _normalize_np(np.asarray(vertex_normals, dtype=np.float32), eps=eps)
    n = int(n_v.shape[0])
    g0 = np.asarray([0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 3)
    g1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3)
    use_g1 = np.abs(np.sum(n_v * g0, axis=1, keepdims=True)) > 0.9
    g = np.where(use_g1, g1, g0)
    t = g - np.sum(g * n_v, axis=1, keepdims=True) * n_v
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    bad = t_norm < float(eps)
    if np.any(bad):
        g_alt = np.where(use_g1, g0, g1)
        t_alt = g_alt - np.sum(g_alt * n_v, axis=1, keepdims=True) * n_v
        t = np.where(bad, t_alt, t)
    e_min = _normalize_np(t, eps=eps).astype(np.float32)
    e_max = np.cross(n_v, e_min).astype(np.float32)
    e_max = _normalize_np(e_max, eps=eps).astype(np.float32)
    e_min = np.cross(e_max, n_v).astype(np.float32)
    e_min = _normalize_np(e_min, eps=eps).astype(np.float32)
    if int(e_min.shape[0]) != n or int(e_max.shape[0]) != n:
        raise RuntimeError("Baseline frame size mismatch.")
    return e_min, e_max


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
    # Linear fallback for nearly equal quaternions.
    q_lin = (1.0 - wt) * q0n + wt * q1n
    q = torch.where(close.expand_as(q), q_lin, q)
    return _normalize_torch(q, eps=1e-8)


@dataclass
class BindingModeConfig:
    mode: str = "games_face"
    hybrid_weight: float = 0.0
    hybrid_dynamic: bool = False
    mix_w_min: float = 0.05
    mix_w_max: float = 0.95
    eps_normal: float = 1e-3
    curv_conf_tau: float = 0.2
    curv_degen_tau: float = 1e-4
    temporal_frame: bool = True
    scale_clamp_min: float = 1e-4
    scale_clamp_max: float = 1.0
    max_temporal_angle_deg: float = 75.0
    temporal_alpha: float = 0.35
    debug_assertions: bool = False


@dataclass
class BindingFrameData:
    verts_t: torch.Tensor
    faces_t: torch.Tensor | None
    face_t1_t: torch.Tensor | None
    face_t2_t: torch.Tensor | None
    face_n_t: torch.Tensor | None
    face_s1_t: torch.Tensor | None
    face_s2_t: torch.Tensor | None
    vertex_n_t: torch.Tensor
    vertex_e_min_t: torch.Tensor
    vertex_e_max_t: torch.Tensor
    curv_conf_t: torch.Tensor
    curv_reliable_t: torch.Tensor


def build_binding_frame_data(
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
    adjacency: list[np.ndarray] | None,
    cfg: BindingModeConfig,
    device: str | torch.device,
) -> BindingFrameData:
    v_np = np.asarray(verts_np, dtype=np.float32)
    faces = None if faces_np is None else np.asarray(faces_np, dtype=np.int64)
    mode = str(cfg.mode).strip().lower()
    need_curvature = mode in ("vertex_curv", "hybrid")
    if need_curvature:
        if adjacency is None:
            adjacency = build_vertex_adjacency(int(v_np.shape[0]), faces)
        e_min_np, e_max_np, n_v_np, conf_np, reliable_np = compute_vertex_curvature_frames_np(
            verts=v_np,
            faces=faces,
            adjacency=adjacency,
            curv_conf_tau=float(cfg.curv_conf_tau),
            curv_degen_tau=float(cfg.curv_degen_tau),
        )
    else:
        n_v_np = compute_area_weighted_vertex_normals_np(v_np, faces, eps=1e-8)
        e_min_np, e_max_np = compute_vertex_baseline_frames_np(n_v_np, eps=1e-8)
        conf_np = np.zeros((int(v_np.shape[0]),), dtype=np.float32)
        reliable_np = np.zeros((int(v_np.shape[0]),), dtype=np.uint8)
    if faces is None or len(faces) == 0:
        f_t1 = f_t2 = f_n = None
        f_s1 = f_s2 = None
        faces_t = None
    else:
        t1_np, t2_np, n_f_np, s1_np, s2_np = compute_face_frames_np(v_np, faces)
        f_t1 = torch.from_numpy(t1_np).to(device=device, dtype=torch.float32)
        f_t2 = torch.from_numpy(t2_np).to(device=device, dtype=torch.float32)
        f_n = torch.from_numpy(n_f_np).to(device=device, dtype=torch.float32)
        f_s1 = torch.from_numpy(s1_np).to(device=device, dtype=torch.float32)
        f_s2 = torch.from_numpy(s2_np).to(device=device, dtype=torch.float32)
        faces_t = torch.from_numpy(faces).to(device=device, dtype=torch.long)

    return BindingFrameData(
        verts_t=torch.from_numpy(v_np).to(device=device, dtype=torch.float32),
        faces_t=faces_t,
        face_t1_t=f_t1,
        face_t2_t=f_t2,
        face_n_t=f_n,
        face_s1_t=f_s1,
        face_s2_t=f_s2,
        vertex_n_t=torch.from_numpy(n_v_np).to(device=device, dtype=torch.float32),
        vertex_e_min_t=torch.from_numpy(e_min_np).to(device=device, dtype=torch.float32),
        vertex_e_max_t=torch.from_numpy(e_max_np).to(device=device, dtype=torch.float32),
        curv_conf_t=torch.from_numpy(conf_np).to(device=device, dtype=torch.float32),
        curv_reliable_t=torch.from_numpy(reliable_np).to(device=device, dtype=torch.float32),
    )


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


def update_gaussians_with_binding(
    g: Gaussian3D,
    frame: BindingFrameData,
    bind_mask: torch.Tensor,
    bind_vertex_idx: torch.Tensor,
    bind_vertex_offset: torch.Tensor | None,
    cfg: BindingModeConfig,
    *,
    bind_face_idx: torch.Tensor | None = None,
    bind_bary: torch.Tensor | None = None,
    quat_blend: float = 1.0,
    prev_vertex_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if bind_mask.ndim != 1:
        raise ValueError(f"bind_mask must be (N,), got {bind_mask.shape}")
    if not bool(bind_mask.any()):
        return prev_vertex_dirs, bind_mask

    mode = str(cfg.mode).strip().lower()
    if mode not in ("games_face", "vertex_curv", "hybrid", "legacy"):
        raise ValueError(f"Unsupported binding mode: {cfg.mode}")
    if mode == "legacy":
        return prev_vertex_dirs, bind_mask

    bound_ids = torch.nonzero(bind_mask, as_tuple=False).view(-1)
    n_bound = int(bound_ids.shape[0])
    if n_bound <= 0:
        return prev_vertex_dirs, bind_mask

    n_verts = int(frame.verts_t.shape[0])
    scales_old = g.scales().index_select(0, bound_ids)
    quat_old = g.quat.index_select(0, bound_ids)
    mu_old = g.xyz.index_select(0, bound_ids)

    # Vertex-anchored components.
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
        quat_v = quat_from_rotmat_torch(R_v)
        quat_vert[local] = quat_v

    # Face-pinned components.
    have_face_meta = (
        bind_face_idx is not None
        and bind_bary is not None
        and frame.faces_t is not None
        and frame.face_t1_t is not None
        and frame.face_t2_t is not None
        and frame.face_n_t is not None
    )
    f_valid = torch.zeros((n_bound,), dtype=torch.bool, device=bind_mask.device)
    mu_face = mu_old.clone()
    quat_face = _normalize_torch(quat_old.clone(), eps=1e-8)
    scale_face = scale_vert.clone()
    if have_face_meta:
        f_idx = bind_face_idx.index_select(0, bound_ids)
        n_faces = int(frame.faces_t.shape[0])
        f_valid = (f_idx >= 0) & (f_idx < n_faces)
        if bool(f_valid.any()):
            local = torch.nonzero(f_valid, as_tuple=False).view(-1)
            fs = f_idx.index_select(0, local)
            bary = _safe_barycentric(bind_bary.index_select(0, bound_ids.index_select(0, local)))
            tri_idx = frame.faces_t.index_select(0, fs)  # (M,3)
            tri = frame.verts_t.index_select(0, tri_idx.reshape(-1)).reshape(-1, 3, 3)
            mu_f = (bary.unsqueeze(-1) * tri).sum(dim=1)
            mu_face[local] = mu_f

            t1 = frame.face_t1_t.index_select(0, fs)
            t2 = frame.face_t2_t.index_select(0, fs)
            nf = frame.face_n_t.index_select(0, fs)
            R_f = torch.stack([t1, t2, nf], dim=-1)
            quat_f = quat_from_rotmat_torch(R_f)
            quat_face[local] = quat_f

            # Geometry + learnable tangent blend.
            s_geom_1 = frame.face_s1_t.index_select(0, fs) if frame.face_s1_t is not None else None
            s_geom_2 = frame.face_s2_t.index_select(0, fs) if frame.face_s2_t is not None else None
            s_old = scales_old.index_select(0, local)
            if s_geom_1 is None or s_geom_2 is None:
                s_t1 = s_old[:, 0]
                s_t2 = s_old[:, 1]
            else:
                s_t1 = 0.5 * s_old[:, 0] + 0.5 * s_geom_1
                s_t2 = 0.5 * s_old[:, 1] + 0.5 * s_geom_2
            s_n_local = torch.full_like(s_t1, float(cfg.eps_normal))
            scale_face[local] = torch.stack([s_t1, s_t2, s_n_local], dim=1)

    # Compose final representation.
    target_mu = mu_old.clone()
    target_quat = _normalize_torch(quat_old.clone(), eps=1e-8)
    target_scale = scales_old.clone()
    w = float(np.clip(float(cfg.hybrid_weight), 0.0, 1.0))
    if bool(cfg.hybrid_dynamic):
        if g.bind_mix_weight is not None:
            w_dyn = g.bind_mix_weight.index_select(0, bound_ids).view(-1)
            w_dyn = w_dyn.clamp(min=float(cfg.mix_w_min), max=float(cfg.mix_w_max))
        elif g.bind_hybrid_w is not None:
            w_dyn = g.bind_hybrid_w.index_select(0, bound_ids).view(-1)
            w_dyn = w_dyn.clamp(min=float(cfg.mix_w_min), max=float(cfg.mix_w_max))
        else:
            w_dyn = torch.full((n_bound,), float(w), dtype=target_mu.dtype, device=target_mu.device)
    else:
        w_dyn = torch.full((n_bound,), float(w), dtype=target_mu.dtype, device=target_mu.device)

    if mode == "games_face":
        use_face = f_valid
        use_vert = (~use_face) & v_valid
        if bool(use_face.any()):
            idx = torch.nonzero(use_face, as_tuple=False).view(-1)
            target_mu[idx] = mu_face.index_select(0, idx)
            target_quat[idx] = quat_face.index_select(0, idx)
            target_scale[idx] = scale_face.index_select(0, idx)
        if bool(use_vert.any()):
            idx = torch.nonzero(use_vert, as_tuple=False).view(-1)
            target_mu[idx] = mu_vert.index_select(0, idx)
            target_quat[idx] = quat_vert.index_select(0, idx)
            target_scale[idx] = scale_vert.index_select(0, idx)
    elif mode == "vertex_curv":
        use_vert = v_valid
        use_face = (~use_vert) & f_valid
        if bool(use_vert.any()):
            idx = torch.nonzero(use_vert, as_tuple=False).view(-1)
            target_mu[idx] = mu_vert.index_select(0, idx)
            target_quat[idx] = quat_vert.index_select(0, idx)
            target_scale[idx] = scale_vert.index_select(0, idx)
        if bool(use_face.any()):
            idx = torch.nonzero(use_face, as_tuple=False).view(-1)
            target_mu[idx] = mu_face.index_select(0, idx)
            target_quat[idx] = quat_face.index_select(0, idx)
            target_scale[idx] = scale_face.index_select(0, idx)
    else:  # hybrid
        both = v_valid & f_valid
        only_face = f_valid & (~v_valid)
        only_vert = v_valid & (~f_valid)
        if bool(both.any()):
            idx = torch.nonzero(both, as_tuple=False).view(-1)
            mu_f = mu_face.index_select(0, idx)
            mu_v = mu_vert.index_select(0, idx)
            q_f = quat_face.index_select(0, idx)
            q_v = quat_vert.index_select(0, idx)
            s_f = scale_face.index_select(0, idx)
            s_v = scale_vert.index_select(0, idx)
            w_loc = w_dyn.index_select(0, idx).view(-1, 1)
            target_mu[idx] = (1.0 - w_loc) * mu_f + w_loc * mu_v
            target_quat[idx] = slerp_quat_torch(q_f, q_v, w_loc.view(-1))
            target_scale[idx] = (1.0 - w_loc) * s_f + w_loc * s_v
        if bool(only_face.any()):
            idx = torch.nonzero(only_face, as_tuple=False).view(-1)
            target_mu[idx] = mu_face.index_select(0, idx)
            target_quat[idx] = quat_face.index_select(0, idx)
            target_scale[idx] = scale_face.index_select(0, idx)
        if bool(only_vert.any()):
            idx = torch.nonzero(only_vert, as_tuple=False).view(-1)
            target_mu[idx] = mu_vert.index_select(0, idx)
            target_quat[idx] = quat_vert.index_select(0, idx)
            target_scale[idx] = scale_vert.index_select(0, idx)

    # Optional extra local shell offset in face tangent frame.
    if g.bind_face_offset_local is not None and have_face_meta and bool(f_valid.any()):
        local = torch.nonzero(f_valid, as_tuple=False).view(-1)
        fs = f_idx.index_select(0, local)
        off_local = g.bind_face_offset_local.index_select(0, bound_ids.index_select(0, local))
        t1 = frame.face_t1_t.index_select(0, fs)
        t2 = frame.face_t2_t.index_select(0, fs)
        nf = frame.face_n_t.index_select(0, fs)
        dxyz = (
            off_local[:, 0:1] * t1
            + off_local[:, 1:2] * t2
            + off_local[:, 2:3] * nf
        )
        target_mu[local] = target_mu.index_select(0, local) + dxyz

    target_scale = target_scale.clamp(
        min=float(cfg.scale_clamp_min),
        max=float(cfg.scale_clamp_max),
    )
    target_log_scale = torch.log(target_scale.clamp(min=1e-8))
    target_quat = _normalize_torch(target_quat, eps=1e-8)

    if bool(cfg.debug_assertions):
        with torch.no_grad():
            qn = torch.linalg.norm(target_quat, dim=1)
            if not bool(torch.isfinite(qn).all()):
                raise RuntimeError("Non-finite quaternion in binding update.")
            if not bool(torch.isfinite(target_scale).all()):
                raise RuntimeError("Non-finite scale in binding update.")

    with torch.no_grad():
        g.xyz.index_copy_(0, bound_ids, target_mu.detach())
        q_old_local = g.quat.index_select(0, bound_ids)
        q_apply = _blend_quat_linear(q_old=q_old_local, q_target=target_quat, blend=float(quat_blend))
        g.quat.index_copy_(0, bound_ids, q_apply.detach())
        g.log_scales.index_copy_(0, bound_ids, target_log_scale.detach())

    valid_local = v_valid | f_valid
    valid_mask = torch.zeros_like(bind_mask)
    valid_mask[bound_ids] = valid_local
    return prev_out, valid_mask


def parse_binding_mode_config(
    cfg_dict: dict | None,
    *,
    default_mode: str = "games_face",
) -> BindingModeConfig:
    d = {} if cfg_dict is None else dict(cfg_dict)
    mode = str(d.get("mode", default_mode)).strip().lower()
    if mode not in ("games_face", "vertex_curv", "hybrid", "legacy"):
        mode = default_mode
    return BindingModeConfig(
        mode=mode,
        hybrid_weight=float(d.get("hybrid_weight", 0.0)),
        hybrid_dynamic=bool(d.get("hybrid_dynamic", False)),
        mix_w_min=float(d.get("mix_w_min", 0.05)),
        mix_w_max=float(d.get("mix_w_max", 0.95)),
        eps_normal=float(d.get("eps_normal", 1e-3)),
        curv_conf_tau=float(d.get("curv_conf_tau", 0.2)),
        curv_degen_tau=float(d.get("curv_degen_tau", 1e-4)),
        temporal_frame=bool(d.get("temporal_frame", True)),
        scale_clamp_min=float(d.get("scale_clamp_min", 1e-4)),
        scale_clamp_max=float(d.get("scale_clamp_max", 1.0)),
        max_temporal_angle_deg=float(d.get("max_temporal_angle_deg", 75.0)),
        temporal_alpha=float(d.get("temporal_alpha", 0.35)),
        debug_assertions=bool(d.get("debug_assertions", False)),
    )


def export_binding_mode_config(cfg: BindingModeConfig) -> dict[str, object]:
    return {
        "mode": str(cfg.mode),
        "hybrid_weight": float(cfg.hybrid_weight),
        "hybrid_dynamic": bool(cfg.hybrid_dynamic),
        "mix_w_min": float(cfg.mix_w_min),
        "mix_w_max": float(cfg.mix_w_max),
        "eps_normal": float(cfg.eps_normal),
        "curv_conf_tau": float(cfg.curv_conf_tau),
        "curv_degen_tau": float(cfg.curv_degen_tau),
        "temporal_frame": bool(cfg.temporal_frame),
        "scale_clamp_min": float(cfg.scale_clamp_min),
        "scale_clamp_max": float(cfg.scale_clamp_max),
        "max_temporal_angle_deg": float(cfg.max_temporal_angle_deg),
        "temporal_alpha": float(cfg.temporal_alpha),
        "debug_assertions": bool(cfg.debug_assertions),
    }


def choose_default_binding_mode(
    available_fields: Iterable[str],
    has_face_binding: bool,
) -> str:
    if has_face_binding:
        return "games_face"
    if "bind_idx" in set(available_fields):
        return "legacy"
    return "legacy"
