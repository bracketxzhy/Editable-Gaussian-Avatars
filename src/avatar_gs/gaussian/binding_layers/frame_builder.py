from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .types import BindingFrameData, BindingModeConfig


def _normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return x / n


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
    tri = v[f]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    fn = np.cross(e1, e2).astype(np.float32)
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
    tri = v[f]
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
    t1 = np.cross(t2, n_f).astype(np.float32)
    t1 = _normalize_np(t1, eps=eps).astype(np.float32)

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
            if np.linalg.matrix_rank(A) >= 3:
                try:
                    coeff, *_ = np.linalg.lstsq(A, z.astype(np.float32), rcond=None)
                    a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
                    H = np.asarray([[a, b], [b, c]], dtype=np.float32)
                    evals, evecs = np.linalg.eigh(H)
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


def parse_binding_mode_config(
    cfg_dict: dict | None,
    *,
    default_mode: str = "games_face",
) -> BindingModeConfig:
    d = {} if cfg_dict is None else dict(cfg_dict)
    mode = str(d.get("mode", default_mode)).strip().lower()
    if mode not in ("games_face", "vertex_curv", "hybrid"):
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
        outer_geom_scale_blend=float(d.get("outer_geom_scale_blend", 0.0)),
        outer_scale_clamp_max=float(d.get("outer_scale_clamp_max", d.get("scale_clamp_max", 1.0))),
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
        "outer_geom_scale_blend": float(cfg.outer_geom_scale_blend),
        "outer_scale_clamp_max": float(cfg.outer_scale_clamp_max),
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
        return "vertex_curv"
    return "vertex_curv"

