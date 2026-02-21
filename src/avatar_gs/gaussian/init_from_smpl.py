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

    if bool(sample_on_surface) and F_all is not None and len(F_all) > 0:
        V_fg, idx_fg, off_fg, face_fg, bary_fg = _sample_points_on_surface(
            verts=V_all,
            faces=F_all,
            n_samples=int(n_keep_eff),
            rng=rng,
        )
    else:
        replace = V_all.shape[0] < int(n_keep_eff)
        idx_fg = rng.choice(V_all.shape[0], size=int(n_keep_eff), replace=replace).astype(np.int64)
        V_fg = V_all[idx_fg].astype(np.float32)
        off_fg = np.zeros_like(V_fg, dtype=np.float32)
        face_fg = -np.ones((int(n_keep_eff),), dtype=np.int64)
        bary_fg = np.zeros((int(n_keep_eff), 3), dtype=np.float32)

    V = V_fg
    idx = idx_fg
    bind_offset = off_fg
    bind_face_idx = face_fg
    bind_bary = bary_fg

    shell_layer = np.zeros((V.shape[0],), dtype=np.int64)
    face_local_offset = np.zeros((V.shape[0], 3), dtype=np.float32)

    if bool(shell_enable) and float(shell_init_ratio) > 0.0 and int(V_fg.shape[0]) > 0:
        n_outer = int(round(float(shell_init_ratio) * float(V_fg.shape[0])))
        if n_outer > 0:
            pick = rng.choice(V_fg.shape[0], size=n_outer, replace=(V_fg.shape[0] < n_outer)).astype(np.int64)
            normals = _vertex_normals_np(V_all, F_all)
            base_anchor = idx_fg[pick]
            base_pts = V_fg[pick]
            nrm = normals[base_anchor]
            shell_pts = (base_pts + float(shell_offset_n) * nrm).astype(np.float32)
            shell_off = (shell_pts - V_all[base_anchor]).astype(np.float32)

            V = np.concatenate([V, shell_pts], axis=0)
            idx = np.concatenate([idx, base_anchor.astype(np.int64)], axis=0)
            bind_offset = np.concatenate([bind_offset, shell_off], axis=0)
            bind_face_idx = np.concatenate([bind_face_idx, bind_face_idx[pick].astype(np.int64)], axis=0)
            bind_bary = np.concatenate([bind_bary, bind_bary[pick].astype(np.float32)], axis=0)
            shell_layer = np.concatenate([shell_layer, np.ones((n_outer,), dtype=np.int64)], axis=0)

            shell_face_local = np.zeros((n_outer, 3), dtype=np.float32)
            shell_face_local[:, 2] = float(shell_offset_n)
            face_local_offset = np.concatenate([face_local_offset, shell_face_local], axis=0)

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

    N = V.shape[0]
    xyz = torch.from_numpy(V)
    bind_idx = torch.from_numpy(np.asarray(idx, dtype=np.int64))
    bind_offset_t = torch.from_numpy(np.asarray(bind_offset, dtype=np.float32))
    bind_face_idx_t = torch.from_numpy(np.asarray(bind_face_idx, dtype=np.int64))
    bind_bary_t = torch.from_numpy(np.asarray(bind_bary, dtype=np.float32))

    log_scales = torch.full((N, 3), float(np.log(0.01)), dtype=torch.float32)
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
    )
