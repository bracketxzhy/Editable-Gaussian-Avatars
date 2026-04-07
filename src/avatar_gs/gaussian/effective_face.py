from __future__ import annotations

import numpy as np
import torch

from avatar_gs.gaussian.binding_layers.frame_builder import (
    compute_area_weighted_vertex_normals_np,
    compute_face_frames_np,
)


def build_vertex_proxy_face_indices_np(
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
) -> np.ndarray:
    verts = np.asarray(verts_np, dtype=np.float32)
    n_verts = int(verts.shape[0])
    proxy = np.full((n_verts,), -1, dtype=np.int64)
    if faces_np is None or int(len(faces_np)) <= 0 or n_verts <= 0:
        return proxy

    faces = np.asarray(faces_np, dtype=np.int64)
    _, _, face_n, face_s1, face_s2 = compute_face_frames_np(verts, faces)
    vertex_n = compute_area_weighted_vertex_normals_np(verts, faces)
    face_scale = np.maximum(face_s1, face_s2).astype(np.float32)

    best_scale = np.full((n_verts,), np.inf, dtype=np.float32)
    best_align = np.full((n_verts,), -np.inf, dtype=np.float32)
    for fi in range(int(faces.shape[0])):
        tri = faces[fi]
        if tri.shape[0] != 3:
            continue
        scale_f = float(face_scale[fi])
        n_f = face_n[fi]
        for vk in tri.tolist():
            vi = int(vk)
            if vi < 0 or vi >= n_verts:
                continue
            align = float(np.dot(vertex_n[vi], n_f))
            better = False
            if scale_f < float(best_scale[vi]) - 1e-8:
                better = True
            elif abs(scale_f - float(best_scale[vi])) <= 1e-8:
                if align > float(best_align[vi]) + 1e-6:
                    better = True
                elif abs(align - float(best_align[vi])) <= 1e-6 and (
                    proxy[vi] < 0 or fi < int(proxy[vi])
                ):
                    better = True
            if better:
                proxy[vi] = int(fi)
                best_scale[vi] = scale_f
                best_align[vi] = align
    return proxy


def compute_effective_face_indices_np(
    *,
    bind_idx: np.ndarray | None,
    bind_face_idx: np.ndarray | None,
    shell_layer: np.ndarray | None,
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
) -> np.ndarray:
    if bind_idx is None and bind_face_idx is None:
        return np.zeros((0,), dtype=np.int64)

    n = int(bind_idx.shape[0]) if bind_idx is not None else int(bind_face_idx.shape[0])
    eff = np.full((n,), -1, dtype=np.int64)

    if bind_face_idx is not None:
        face_idx = np.asarray(bind_face_idx, dtype=np.int64).reshape(-1)
        n_faces = 0 if faces_np is None else int(len(faces_np))
        face_valid = (face_idx >= 0) & (face_idx < n_faces)
        eff[face_valid] = face_idx[face_valid]
    else:
        face_idx = np.full((n,), -1, dtype=np.int64)

    if bind_idx is None or faces_np is None or int(len(faces_np)) <= 0 or shell_layer is None:
        return eff

    bind_idx_np = np.asarray(bind_idx, dtype=np.int64).reshape(-1)
    shell_np = np.asarray(shell_layer, dtype=np.int64).reshape(-1)
    proxy_face = build_vertex_proxy_face_indices_np(verts_np=verts_np, faces_np=faces_np)
    valid_vertex = (bind_idx_np >= 0) & (bind_idx_np < int(proxy_face.shape[0]))
    need_proxy = (shell_np == 0) & (eff < 0) & valid_vertex
    if bool(np.any(need_proxy)):
        eff[need_proxy] = proxy_face[bind_idx_np[need_proxy]]
    return eff


def compute_effective_face_indices_torch(
    *,
    bind_idx_t: torch.Tensor | None,
    bind_face_idx_t: torch.Tensor | None,
    shell_layer_t: torch.Tensor | None,
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
    device: torch.device | str,
) -> torch.Tensor:
    bind_idx_np = None if bind_idx_t is None else bind_idx_t.detach().cpu().numpy()
    bind_face_idx_np = None if bind_face_idx_t is None else bind_face_idx_t.detach().cpu().numpy()
    shell_layer_np = None if shell_layer_t is None else shell_layer_t.detach().cpu().numpy()
    eff_np = compute_effective_face_indices_np(
        bind_idx=bind_idx_np,
        bind_face_idx=bind_face_idx_np,
        shell_layer=shell_layer_np,
        verts_np=verts_np,
        faces_np=faces_np,
    )
    return torch.from_numpy(eff_np).to(device=device, dtype=torch.int64)
