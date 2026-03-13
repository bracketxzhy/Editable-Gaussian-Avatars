from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.shell_policy import estimate_inner_count


def load_obj_mesh(obj_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    vs = []
    fs = []
    with Path(obj_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                toks = line.strip().split()
                if len(toks) < 4:
                    continue
                _, x, y, z = toks[:4]
                vs.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                toks = line.strip().split()[1:]
                if len(toks) < 3:
                    continue
                idxs = []
                n_curr = len(vs)
                for tok in toks:
                    base = tok.split("/")[0]
                    if base == "":
                        continue
                    vi = int(base)
                    if vi < 0:
                        vi = n_curr + vi + 1
                    idxs.append(vi - 1)
                if len(idxs) < 3:
                    continue
                for k in range(1, len(idxs) - 1):
                    fs.append([idxs[0], idxs[k], idxs[k + 1]])
    if len(vs) == 0:
        raise RuntimeError(f"No vertices found in OBJ: {obj_path}")
    verts = np.asarray(vs, dtype=np.float32)
    faces = np.asarray(fs, dtype=np.int64) if len(fs) > 0 else None
    return verts, faces


def load_obj_vertices(obj_path: str | Path) -> np.ndarray:
    verts, _ = load_obj_mesh(obj_path)
    return verts


def _vertex_normals_np(verts: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float32)
    if faces is None or len(faces) == 0:
        n = np.zeros_like(v, dtype=np.float32)
        if int(n.shape[0]) > 0:
            n[:, 2] = 1.0
        return n
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(v, dtype=np.float32)
    tri = v[f]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    fn = np.cross(e1, e2).astype(np.float32)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    n = n / norm
    return n.astype(np.float32)


def _sample_points_on_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = verts[faces]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    valid = area > 1e-12
    if not np.any(valid):
        idx = rng.choice(verts.shape[0], size=n_samples, replace=True).astype(np.int64)
        pts = verts[idx].astype(np.float32)
        off = np.zeros_like(pts, dtype=np.float32)
        face_idx = -np.ones((n_samples,), dtype=np.int64)
        bary = np.zeros((n_samples, 3), dtype=np.float32)
        return pts, idx, off, face_idx, bary

    faces_v = faces[valid]
    face_ids_v = np.nonzero(valid)[0].astype(np.int64)
    area_v = area[valid]
    probs = area_v / area_v.sum()
    pick = rng.choice(faces_v.shape[0], size=n_samples, replace=True, p=probs)
    tri_idx = faces_v[pick]
    tri_v = verts[tri_idx]

    u1 = rng.random(n_samples).astype(np.float32)
    u2 = rng.random(n_samples).astype(np.float32)
    su1 = np.sqrt(u1)
    w0 = 1.0 - su1
    w1 = su1 * (1.0 - u2)
    w2 = su1 * u2

    pts = (
        w0[:, None] * tri_v[:, 0, :]
        + w1[:, None] * tri_v[:, 1, :]
        + w2[:, None] * tri_v[:, 2, :]
    ).astype(np.float32)

    w = np.stack([w0, w1, w2], axis=1)
    anchor_local = np.argmax(w, axis=1)
    anchor_idx = tri_idx[np.arange(n_samples), anchor_local].astype(np.int64)
    off = (pts - verts[anchor_idx]).astype(np.float32)
    face_idx = face_ids_v[pick].astype(np.int64)
    bary = np.stack([w0, w1, w2], axis=1).astype(np.float32)
    return pts, anchor_idx, off, face_idx, bary


def _sample_face_centroids(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    allowed_face_ids: np.ndarray | None = None,
    unique_faces: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(n_samples) <= 0 or faces is None or len(faces) == 0:
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.int64)
        return z3, z1, z3, z1, z3

    faces_all = np.asarray(faces, dtype=np.int64)
    face_ids = np.arange(int(faces_all.shape[0]), dtype=np.int64)
    if allowed_face_ids is not None and int(np.asarray(allowed_face_ids).size) > 0:
        sel = np.asarray(allowed_face_ids, dtype=np.int64)
        sel = sel[(sel >= 0) & (sel < int(faces_all.shape[0]))]
        if int(sel.size) > 0:
            face_ids = np.unique(sel)
    tri = verts[faces_all[face_ids]]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    valid = area > 1e-12
    if not np.any(valid):
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.int64)
        return z3, z1, z3, z1, z3

    face_ids_v = face_ids[valid]
    tri_v = verts[faces_all[face_ids_v]]
    area_v = area[valid]
    probs = area_v / np.clip(area_v.sum(), 1e-8, None)
    n_pick = int(n_samples)
    replace = True
    if bool(unique_faces):
        n_pick = min(int(n_samples), int(face_ids_v.shape[0]))
        replace = False
    pick = rng.choice(face_ids_v.shape[0], size=int(n_pick), replace=replace, p=probs)
    fs = face_ids_v[pick]
    tri_pick = verts[faces_all[fs]]
    bary = np.full((int(n_pick), 3), 1.0 / 3.0, dtype=np.float32)
    pts = tri_pick.mean(axis=1).astype(np.float32)
    anchor_idx = faces_all[fs, 0].astype(np.int64)
    off = (pts - verts[anchor_idx]).astype(np.float32)
    return pts, anchor_idx, off, fs.astype(np.int64), bary


def _estimate_vertex_curvature_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
) -> np.ndarray:
    n_verts = int(verts.shape[0])
    if n_verts <= 0 or faces is None or len(faces) == 0:
        return np.zeros((max(n_verts, 0),), dtype=np.float32)
    nrm = _vertex_normals_np(verts, faces)
    f = np.asarray(faces, dtype=np.int64)
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in f:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= n_verts:
            continue
        nbrs[i0].add(i1); nbrs[i0].add(i2)
        nbrs[i1].add(i0); nbrs[i1].add(i2)
        nbrs[i2].add(i0); nbrs[i2].add(i1)
    curv = np.zeros((n_verts,), dtype=np.float32)
    for vi in range(n_verts):
        if len(nbrs[vi]) == 0:
            continue
        ni = nrm[vi]
        dots = np.clip(np.sum(nrm[list(nbrs[vi])] * ni[None, :], axis=1), -1.0, 1.0)
        curv[vi] = float(np.mean(1.0 - dots))
    return np.clip(curv, 0.0, 1.0).astype(np.float32)


def _estimate_vertex_local_scale_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float32)
    n_verts = int(v.shape[0])
    if n_verts <= 0:
        return np.zeros((0,), dtype=np.float32)
    if faces is None or len(faces) == 0:
        return np.full((n_verts,), 0.01, dtype=np.float32)

    f = np.asarray(faces, dtype=np.int64)
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in f:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= n_verts:
            continue
        nbrs[i0].add(i1); nbrs[i0].add(i2)
        nbrs[i1].add(i0); nbrs[i1].add(i2)
        nbrs[i2].add(i0); nbrs[i2].add(i1)
    scale_ref = np.zeros((n_verts,), dtype=np.float32)
    for vi in range(n_verts):
        if len(nbrs[vi]) <= 0:
            continue
        nb = np.asarray(sorted(list(nbrs[vi])), dtype=np.int64)
        d = np.linalg.norm(v[nb] - v[vi], axis=1).astype(np.float32)
        if d.size > 0:
            scale_ref[vi] = float(np.mean(d))
    valid = scale_ref > 1e-8
    fill = float(np.median(scale_ref[valid])) if np.any(valid) else 0.01
    scale_ref[~valid] = fill
    return np.clip(scale_ref, 1e-4, None).astype(np.float32)


def _compute_face_half_scales_np(
    verts: np.ndarray,
    faces: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if faces is None or len(faces) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    n = np.cross(e1, e2).astype(np.float32)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(n_norm, 1e-8, None)
    t1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-8, None)
    t2 = np.cross(n, t1).astype(np.float32)
    t2 = t2 / np.clip(np.linalg.norm(t2, axis=1, keepdims=True), 1e-8, None)
    s1 = np.maximum(0.5 * np.linalg.norm(e1, axis=1), 1e-4).astype(np.float32)
    s2 = np.maximum(0.5 * np.abs(np.sum(e2 * t2, axis=1)), 1e-4).astype(np.float32)
    return s1, s2


def init_gaussians_from_mesh_3d(
    mesh_obj: str | Path,
    n_keep: int = 3000,
    seed: int = 42,
    n_bg: int = 0,
    bg_extent: tuple[float, float, float] = (3.0, 3.0, 0.6),
    bg_center_z: float = 2.5,
    bg_log_scale: float = -3.5,
    bg_opacity_logit: float = -1.5,
    bg_exclude_bbox_pad: float = 0.0,
    sample_on_surface: bool = True,
    inner_count_auto: bool = False,
    inner_count_kV: float = 1.2,
    inner_count_kF: float = 0.35,
    inner_count_min: int = 80000,
    inner_count_max: int = 220000,
    shell_enable: bool = False,
    shell_init_ratio: float = 0.15,
    shell_offset_n: float = 0.015,
    inner_dual_candidate_enable: bool = True,
    inner_high_curv_dual_activate_th: float = 0.2,
    inner_face_candidate_ratio: float = 0.35,
    inner_vertex_candidate_ratio: float = 0.65,
    inner_bind_mode_default: str = "vertex",
    inner_local_scale_init_enable: bool = True,
    inner_local_scale_init_vertex_ratio: float = 0.40,
    inner_local_scale_init_face_ratio: float = 0.75,
    inner_local_scale_init_normal_ratio: float = 0.45,
    inner_local_scale_init_min: float = 8e-4,
    inner_local_scale_init_max: float = 1.2e-2,
    outer_init_from_inner: bool = False,
    outer_dormant_slot_ratio: float = 0.30,
    outer_dormant_face_ratio: float = 0.0,
    outer_face_anchor_rule: str = "centroid",
    outer_pruned_opacity_logit: float = -12.0,
) -> Gaussian3D:
    rng = np.random.default_rng(seed)
    V_all, F_all = load_obj_mesh(mesh_obj)

    if bool(inner_count_auto):
        n_keep_eff = estimate_inner_count(
            n_verts=int(V_all.shape[0]),
            n_faces=0 if F_all is None else int(F_all.shape[0]),
            k_v=float(inner_count_kV),
            k_f=float(inner_count_kF),
            n_min=int(inner_count_min),
            n_max=int(inner_count_max),
        )
    else:
        n_keep_eff = int(n_keep)

    if int(n_keep_eff) <= 0:
        raise ValueError(f"n_keep must be > 0. Got {n_keep_eff}")

    default_mode = str(inner_bind_mode_default).strip().lower()
    default_mode = "face" if default_mode == "face" else "vertex"
    if bool(sample_on_surface) and F_all is not None and len(F_all) > 0 and bool(inner_dual_candidate_enable):
        n_inner = int(n_keep_eff)
        face_ratio = float(np.clip(float(inner_face_candidate_ratio), 0.0, 1.0))
        vert_ratio = float(np.clip(float(inner_vertex_candidate_ratio), 0.0, 1.0))
        if face_ratio + vert_ratio <= 1e-6:
            vert_ratio = 1.0
            face_ratio = 0.0
        s = face_ratio + vert_ratio
        face_ratio /= s
        n_face = int(round(float(n_inner) * face_ratio))
        n_face = max(0, min(n_inner, n_face))
        n_vertex = max(0, n_inner - n_face)

        if n_vertex > 0:
            idx_v = rng.choice(V_all.shape[0], size=n_vertex, replace=(V_all.shape[0] < n_vertex)).astype(np.int64)
            V_v = V_all[idx_v].astype(np.float32)
            off_v = np.zeros_like(V_v, dtype=np.float32)
            face_v = -np.ones((n_vertex,), dtype=np.int64)
            bary_v = np.zeros((n_vertex, 3), dtype=np.float32)
            mode_v = np.zeros((n_vertex,), dtype=np.int64)
        else:
            V_v = np.zeros((0, 3), dtype=np.float32)
            idx_v = np.zeros((0,), dtype=np.int64)
            off_v = np.zeros((0, 3), dtype=np.float32)
            face_v = np.zeros((0,), dtype=np.int64)
            bary_v = np.zeros((0, 3), dtype=np.float32)
            mode_v = np.zeros((0,), dtype=np.int64)

        face_allow = None
        if n_face > 0:
            curv = _estimate_vertex_curvature_np(V_all, F_all)
            curv_th = float(np.clip(float(inner_high_curv_dual_activate_th), 0.0, 1.0))
            high_v = np.nonzero(curv >= curv_th)[0].astype(np.int64)
            if int(high_v.size) > 0:
                tri = np.asarray(F_all, dtype=np.int64)
                face_allow = np.nonzero(
                    np.isin(tri[:, 0], high_v) | np.isin(tri[:, 1], high_v) | np.isin(tri[:, 2], high_v)
                )[0].astype(np.int64)
        V_f, idx_f, off_f, face_f, bary_f = _sample_face_centroids(
            verts=V_all,
            faces=F_all,
            n_samples=n_face,
            rng=rng,
            allowed_face_ids=face_allow,
        )
        mode_f = np.ones((int(V_f.shape[0]),), dtype=np.int64)

        V_fg = np.concatenate([V_v, V_f], axis=0)
        idx_fg = np.concatenate([idx_v, idx_f], axis=0)
        off_fg = np.concatenate([off_v, off_f], axis=0)
        face_fg = np.concatenate([face_v, face_f], axis=0)
        bary_fg = np.concatenate([bary_v, bary_f], axis=0)
        inner_bind_mode_fg = np.concatenate([mode_v, mode_f], axis=0)
        if int(V_fg.shape[0]) > 1:
            perm = rng.permutation(int(V_fg.shape[0]))
            V_fg = V_fg[perm]
            idx_fg = idx_fg[perm]
            off_fg = off_fg[perm]
            face_fg = face_fg[perm]
            bary_fg = bary_fg[perm]
            inner_bind_mode_fg = inner_bind_mode_fg[perm]
        if int(V_fg.shape[0]) > int(n_inner):
            V_fg = V_fg[:n_inner]
            idx_fg = idx_fg[:n_inner]
            off_fg = off_fg[:n_inner]
            face_fg = face_fg[:n_inner]
            bary_fg = bary_fg[:n_inner]
            inner_bind_mode_fg = inner_bind_mode_fg[:n_inner]
        elif int(V_fg.shape[0]) < int(n_inner):
            pad = int(n_inner - V_fg.shape[0])
            idx_pad = rng.choice(V_all.shape[0], size=pad, replace=True).astype(np.int64)
            V_pad = V_all[idx_pad].astype(np.float32)
            V_fg = np.concatenate([V_fg, V_pad], axis=0)
            idx_fg = np.concatenate([idx_fg, idx_pad], axis=0)
            off_fg = np.concatenate([off_fg, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            face_fg = np.concatenate([face_fg, -np.ones((pad,), dtype=np.int64)], axis=0)
            bary_fg = np.concatenate([bary_fg, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            inner_bind_mode_fg = np.concatenate([inner_bind_mode_fg, np.zeros((pad,), dtype=np.int64)], axis=0)
    elif bool(sample_on_surface) and F_all is not None and len(F_all) > 0:
        if default_mode == "face":
            V_fg, idx_fg, off_fg, face_fg, bary_fg = _sample_face_centroids(
                verts=V_all,
                faces=F_all,
                n_samples=int(n_keep_eff),
                rng=rng,
            )
            inner_bind_mode_fg = np.ones((int(V_fg.shape[0]),), dtype=np.int64)
        else:
            V_fg, idx_fg, off_fg, face_fg, bary_fg = _sample_points_on_surface(
                verts=V_all,
                faces=F_all,
                n_samples=int(n_keep_eff),
                rng=rng,
            )
            inner_bind_mode_fg = np.zeros((int(V_fg.shape[0]),), dtype=np.int64)
    else:
        replace = V_all.shape[0] < int(n_keep_eff)
        idx_fg = rng.choice(V_all.shape[0], size=int(n_keep_eff), replace=replace).astype(np.int64)
        V_fg = V_all[idx_fg].astype(np.float32)
        off_fg = np.zeros_like(V_fg, dtype=np.float32)
        face_fg = -np.ones((int(n_keep_eff),), dtype=np.int64)
        bary_fg = np.zeros((int(n_keep_eff), 3), dtype=np.float32)
        inner_bind_mode_fg = np.zeros((int(n_keep_eff),), dtype=np.int64)

    V = V_fg
    idx = idx_fg
    bind_offset = np.zeros_like(off_fg, dtype=np.float32)
    bind_face_idx = face_fg
    bind_bary = bary_fg

    shell_layer = np.zeros((V.shape[0],), dtype=np.int64)
    face_local_offset = np.zeros((V.shape[0], 3), dtype=np.float32)
    inner_bind_mode = inner_bind_mode_fg.astype(np.int64, copy=True)

    if bool(shell_enable) and float(shell_init_ratio) > 0.0 and int(V_fg.shape[0]) > 0:
        outer_ratio = float(outer_dormant_slot_ratio) if float(outer_dormant_slot_ratio) > 0.0 else float(shell_init_ratio)
        n_outer_ratio = int(round(float(outer_ratio) * float(V_fg.shape[0])))
        n_outer_faces = 0
        if F_all is not None and int(len(F_all)) > 0 and float(outer_dormant_face_ratio) > 0.0:
            n_outer_faces = int(round(float(outer_dormant_face_ratio) * float(len(F_all))))
        # Capacity is controlled by the stronger budget between inner-ratio and face-based floor.
        n_outer = max(int(n_outer_ratio), int(n_outer_faces))
        if n_outer > 0:
            if bool(outer_init_from_inner):
                face_src = np.asarray(bind_face_idx, dtype=np.int64)
                valid_pick = np.nonzero(face_src >= 0)[0].astype(np.int64)
                if int(valid_pick.size) > 0:
                    _, first_idx = np.unique(face_src[valid_pick], return_index=True)
                    unique_pick = valid_pick[np.sort(first_idx)]
                    if int(unique_pick.size) > 0:
                        n_outer = min(int(n_outer), int(unique_pick.size))
                        choose = rng.choice(unique_pick.shape[0], size=n_outer, replace=False).astype(np.int64)
                        pick = unique_pick[choose]
                    else:
                        n_outer = 0
                        pick = np.zeros((0,), dtype=np.int64)
                else:
                    n_outer = 0
                    pick = np.zeros((0,), dtype=np.int64)
                normals = _vertex_normals_np(V_all, F_all)
                if n_outer > 0:
                    base_anchor = idx_fg[pick]
                    base_pts = V_fg[pick]
                    nrm = normals[base_anchor]
                    shell_pts = (base_pts + float(shell_offset_n) * nrm).astype(np.float32)
                    shell_off = (shell_pts - V_all[base_anchor]).astype(np.float32)
                    shell_face_idx = bind_face_idx[pick].astype(np.int64)
                    shell_bary = bind_bary[pick].astype(np.float32)
                    shell_face_local = np.zeros((n_outer, 3), dtype=np.float32)
                    shell_face_local[:, 2] = float(shell_offset_n)
                else:
                    shell_pts = np.zeros((0, 3), dtype=np.float32)
                    base_anchor = np.zeros((0,), dtype=np.int64)
                    shell_off = np.zeros((0, 3), dtype=np.float32)
                    shell_face_idx = np.zeros((0,), dtype=np.int64)
                    shell_bary = np.zeros((0, 3), dtype=np.float32)
                    shell_face_local = np.zeros((0, 3), dtype=np.float32)
            else:
                if str(outer_face_anchor_rule).strip().lower() != "centroid":
                    raise ValueError(f"outer_face_anchor_rule only supports 'centroid'. Got: {outer_face_anchor_rule}")
                shell_pts, base_anchor, shell_off, shell_face_idx, shell_bary = _sample_face_centroids(
                    verts=V_all,
                    faces=F_all,
                    n_samples=n_outer,
                    rng=rng,
                    unique_faces=True,
                )
                n_outer = int(shell_pts.shape[0])
                shell_face_local = np.zeros((n_outer, 3), dtype=np.float32)

            if n_outer > 0:
                V = np.concatenate([V, shell_pts], axis=0)
                idx = np.concatenate([idx, base_anchor.astype(np.int64)], axis=0)
                bind_offset = np.concatenate([bind_offset, shell_off], axis=0)
                bind_face_idx = np.concatenate([bind_face_idx, shell_face_idx], axis=0)
                bind_bary = np.concatenate([bind_bary, shell_bary], axis=0)
                shell_layer = np.concatenate([shell_layer, np.ones((n_outer,), dtype=np.int64)], axis=0)
                face_local_offset = np.concatenate([face_local_offset, shell_face_local], axis=0)
                inner_bind_mode = np.concatenate([inner_bind_mode, -np.ones((n_outer,), dtype=np.int64)], axis=0)

    if int(n_bg) > 0:
        ex, ey, ez = float(bg_extent[0]), float(bg_extent[1]), float(bg_extent[2])
        pad = float(bg_exclude_bbox_pad)
        vmin = V.min(axis=0) - pad
        vmax = V.max(axis=0) + pad

        def _sample_bg(n: int) -> np.ndarray:
            return np.stack(
                [
                    rng.uniform(-ex, ex, size=n),
                    rng.uniform(-ey, ey, size=n),
                    rng.uniform(bg_center_z - ez, bg_center_z + ez, size=n),
                ],
                axis=-1,
            ).astype(np.float32)

        if pad > 0.0:
            collected = []
            remaining = int(n_bg)
            tries = 0
            while remaining > 0 and tries < 20:
                tries += 1
                cand = _sample_bg(max(remaining * 4, 256))
                inside = (
                    (cand[:, 0] >= vmin[0]) & (cand[:, 0] <= vmax[0])
                    & (cand[:, 1] >= vmin[1]) & (cand[:, 1] <= vmax[1])
                    & (cand[:, 2] >= vmin[2]) & (cand[:, 2] <= vmax[2])
                )
                cand = cand[~inside]
                if cand.shape[0] == 0:
                    continue
                take = min(remaining, cand.shape[0])
                collected.append(cand[:take])
                remaining -= take
            if remaining > 0:
                collected.append(_sample_bg(remaining))
            V_bg = np.concatenate(collected, axis=0).astype(np.float32)
        else:
            V_bg = _sample_bg(int(n_bg))

        V = np.concatenate([V, V_bg], axis=0)
        idx_bg = -np.ones((int(n_bg),), dtype=np.int64)
        idx = np.concatenate([idx, idx_bg], axis=0)
        off_bg = np.zeros((int(n_bg), 3), dtype=np.float32)
        bind_offset = np.concatenate([bind_offset, off_bg], axis=0)
        face_bg = -np.ones((int(n_bg),), dtype=np.int64)
        bary_bg = np.zeros((int(n_bg), 3), dtype=np.float32)
        bind_face_idx = np.concatenate([bind_face_idx, face_bg], axis=0)
        bind_bary = np.concatenate([bind_bary, bary_bg], axis=0)
        shell_layer = np.concatenate([shell_layer, -np.ones((int(n_bg),), dtype=np.int64)], axis=0)
        face_local_offset = np.concatenate([face_local_offset, np.zeros((int(n_bg), 3), dtype=np.float32)], axis=0)
        inner_bind_mode = np.concatenate([inner_bind_mode, -np.ones((int(n_bg),), dtype=np.int64)], axis=0)

    N = V.shape[0]
    xyz = torch.from_numpy(V)
    bind_idx = torch.from_numpy(np.asarray(idx, dtype=np.int64))
    bind_offset_t = torch.from_numpy(np.asarray(bind_offset, dtype=np.float32))
    bind_face_idx_t = torch.from_numpy(np.asarray(bind_face_idx, dtype=np.int64))
    bind_bary_t = torch.from_numpy(np.asarray(bind_bary, dtype=np.float32))

    log_scales = torch.full((N, 3), float(np.log(0.01)), dtype=torch.float32)
    if bool(inner_local_scale_init_enable) and int(N) > 0:
        inner_mask_np = np.asarray(shell_layer == 0, dtype=bool)
        if bool(np.any(inner_mask_np)):
            v_scale_ref = _estimate_vertex_local_scale_np(V_all, F_all)
            f_s1_ref, f_s2_ref = _compute_face_half_scales_np(V_all, F_all)
            local_scales = np.full((N, 3), 0.01, dtype=np.float32)

            inner_ids = np.nonzero(inner_mask_np)[0].astype(np.int64)
            anchor_idx_inner = np.asarray(idx, dtype=np.int64)[inner_ids]
            anchor_idx_safe = np.clip(anchor_idx_inner, 0, max(int(v_scale_ref.shape[0]) - 1, 0))
            base_vertex = v_scale_ref[anchor_idx_safe]
            s_t_vertex = np.clip(
                float(inner_local_scale_init_vertex_ratio) * base_vertex,
                float(inner_local_scale_init_min),
                float(inner_local_scale_init_max),
            ).astype(np.float32)
            local_scales[inner_ids, 0] = s_t_vertex
            local_scales[inner_ids, 1] = s_t_vertex
            local_scales[inner_ids, 2] = np.clip(
                float(inner_local_scale_init_normal_ratio) * s_t_vertex,
                float(inner_local_scale_init_min),
                float(inner_local_scale_init_max),
            ).astype(np.float32)

            face_idx_inner = np.asarray(bind_face_idx, dtype=np.int64)[inner_ids]
            valid_face = (face_idx_inner >= 0) & (face_idx_inner < int(f_s1_ref.shape[0]))
            if bool(np.any(valid_face)):
                inner_face_ids = inner_ids[valid_face]
                face_sel = face_idx_inner[valid_face]
                s1 = np.clip(
                    float(inner_local_scale_init_face_ratio) * f_s1_ref[face_sel],
                    float(inner_local_scale_init_min),
                    float(inner_local_scale_init_max),
                ).astype(np.float32)
                s2 = np.clip(
                    float(inner_local_scale_init_face_ratio) * f_s2_ref[face_sel],
                    float(inner_local_scale_init_min),
                    float(inner_local_scale_init_max),
                ).astype(np.float32)
                s_n = np.clip(
                    float(inner_local_scale_init_normal_ratio) * np.minimum(s1, s2),
                    float(inner_local_scale_init_min),
                    float(inner_local_scale_init_max),
                ).astype(np.float32)
                local_scales[inner_face_ids, 0] = s1
                local_scales[inner_face_ids, 1] = s2
                local_scales[inner_face_ids, 2] = s_n
            log_scales = torch.log(torch.from_numpy(np.clip(local_scales, 1e-6, None)).to(dtype=torch.float32))
    if int(n_bg) > 0:
        log_scales[-int(n_bg):, :] = float(bg_log_scale)
    quat = torch.zeros((N, 4), dtype=torch.float32)
    quat[:, 0] = 1.0
    opacity_logit = torch.full((N, 1), 0.0, dtype=torch.float32)
    if int(n_bg) > 0:
        opacity_logit[-int(n_bg):, :] = float(bg_opacity_logit)

    rgb = torch.rand((N, 3), dtype=torch.float32) * 0.6 + 0.2
    rgb_logit = torch.log(rgb / (1.0 - rgb))

    bind_mix_weight = torch.full((N, 1), 0.5, dtype=torch.float32)
    prune_state = torch.zeros((N,), dtype=torch.int64)
    if bool(shell_enable) and not bool(outer_init_from_inner):
        outer_mask = torch.from_numpy(np.asarray(shell_layer == 1))
        prune_state[outer_mask] = 2
        opacity_logit[outer_mask, :] = float(outer_pruned_opacity_logit)

    return Gaussian3D(
        xyz=xyz,
        log_scales=log_scales,
        quat=quat,
        opacity_logit=opacity_logit,
        rgb_logit=rgb_logit,
        bind_idx=bind_idx,
        bind_offset=bind_offset_t,
        bind_face_idx=bind_face_idx_t,
        bind_bary=bind_bary_t,
        bind_face_offset_local=torch.from_numpy(np.asarray(face_local_offset, dtype=np.float32)),
        shell_layer=torch.from_numpy(np.asarray(shell_layer, dtype=np.int64)),
        bind_mix_weight=bind_mix_weight,
        prune_state=prune_state,
        inner_bind_mode=torch.from_numpy(np.asarray(inner_bind_mode, dtype=np.int64)),
    )
