from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json

import numpy as np
import torch

from avatar_gs.data.sequence_mesh import SequenceMeshProvider
from avatar_gs.gaussian.binding import BindingFrameData, build_binding_frame_data, build_vertex_adjacency, parse_binding_mode_config
from avatar_gs.gaussian.init_from_smpl import load_obj_mesh, load_obj_vertices
from avatar_gs.alignment import apply_similarity_to_camera_points
from avatar_gs.gaussian.rigid_align import apply_rigid_transform_np
from .rigid_align import normalize_rigid_align_config, run_stage2_rigid_align_search


def resolve_effective_rigid_align_config(
    *,
    explicit_cfg,
    stage2_dir: Path | None = None,
    auto_from_stage2: bool = False,
    logger=None,
) -> dict | None:
    cfg = normalize_rigid_align_config(explicit_cfg, default_enabled=False)
    if auto_from_stage2 and (cfg is None or not bool(cfg.get("enabled", False))) and stage2_dir is not None:
        report_path = stage2_dir / "bind_rigid_align_pre_stage2.json"
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    report_cfg = json.load(f)
                report_norm = normalize_rigid_align_config(report_cfg, default_enabled=False)
                if report_norm is not None and bool(report_norm.get("enabled", False)):
                    cfg = report_norm
                    if logger is not None:
                        logger.info(f"Loaded bind_rigid_align from pre-stage2 report: {report_path}")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Failed to load pre-stage2 rigid align report ({report_path}): {e}")
    return cfg


def resolve_mesh_sequence_dir(*, project_root: Path, stage1_dir: Path | None = None, bind_mesh_dir_cfg: str | Path | None, logger=None) -> Path | None:
    if bind_mesh_dir_cfg is None:
        return None
    bind_mesh_dir = Path(str(bind_mesh_dir_cfg))
    if not bind_mesh_dir.is_absolute():
        bind_mesh_dir = project_root / bind_mesh_dir
    if stage1_dir is not None:
        seq_dir_candidate = stage1_dir / "sequence"
        if seq_dir_candidate.exists():
            bind_mesh_dir = seq_dir_candidate
            if logger is not None:
                logger.info(f"Using bind mesh sequence dir: {bind_mesh_dir}")
    if bind_mesh_dir.exists():
        return bind_mesh_dir
    return bind_mesh_dir


def resolve_bind_mesh_indices(*, frames: list[str], bind_seq_npz_path: Path | None, bind_mesh_dir: Path | None, logger=None) -> list[int]:
    if bind_seq_npz_path is not None and bind_seq_npz_path.exists():
        seq = np.load(bind_seq_npz_path, allow_pickle=True)
        if "frame_names" in seq.files:
            seq_names = [str(x) for x in seq["frame_names"].tolist()]
            name_to_idx = {n: i for i, n in enumerate(seq_names)}
            missing = []
            bind_mesh_indices = []
            for fn in frames:
                if fn in name_to_idx:
                    bind_mesh_indices.append(int(name_to_idx[fn]))
                else:
                    missing.append(fn)
            if not missing:
                return bind_mesh_indices
            if bind_mesh_dir is not None:
                try:
                    seq_provider = SequenceMeshProvider(bind_mesh_dir)
                    seq_provider.scan()
                    seq_frame_ids = seq_provider.get_frame_ids()
                    if len(seq_frame_ids) > 0:
                        if logger is not None:
                            logger.warning(
                                "Some frame names not found in smpl_params_sequence.npz; falling back to sequential mesh indices from bind_mesh_sequence_dir."
                            )
                        return list(range(len(frames)))
                except Exception:
                    pass
            if len(seq_names) == len(frames):
                if logger is not None:
                    logger.warning(
                        "Some frame names not found in smpl_params_sequence.npz; falling back to sequential mesh indices."
                    )
                return list(range(len(frames)))
            raise RuntimeError(
                f"Binding failed: {len(missing)} frames not found in smpl_params_sequence.npz. Example missing: {missing[0]}"
            )
    return [int(i) for i in range(len(frames))]




def compute_vertex_normals_np(verts: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    if faces is None or len(faces) == 0:
        return np.zeros_like(verts, dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32)
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
    return (n / norm).astype(np.float32)


def _build_face_adjacency_lists(faces_np: np.ndarray) -> list[list[int]]:
    f = np.asarray(faces_np, dtype=np.int64)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(f):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i0, i1 in ((a, b), (b, c), (c, a)):
            key = (i0, i1) if i0 < i1 else (i1, i0)
            edge_to_faces.setdefault(key, []).append(int(fi))
    neigh = [set() for _ in range(int(f.shape[0]))]
    for flist in edge_to_faces.values():
        if len(flist) < 2:
            continue
        for fi in flist:
            for fj in flist:
                if fi != fj:
                    neigh[fi].add(fj)
    return [sorted(list(x)) for x in neigh]


def pack_face_neighbor_arrays(faces_np: np.ndarray, *, rings: int = 1) -> np.ndarray:
    neigh_1 = _build_face_adjacency_lists(faces_np)
    rings = max(int(rings), 1)
    all_neigh: list[list[int]] = []
    max_deg = 0
    for fi in range(len(neigh_1)):
        frontier = {int(fi)}
        visited = {int(fi)}
        acc: set[int] = set()
        for _ in range(rings):
            nxt: set[int] = set()
            for fk in frontier:
                for nb in neigh_1[fk]:
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
                        acc.add(nb)
            frontier = nxt
            if len(frontier) <= 0:
                break
        lst = sorted(list(acc))
        all_neigh.append(lst)
        max_deg = max(max_deg, len(lst))
    if max_deg <= 0:
        return np.full((len(neigh_1), 0), -1, dtype=np.int64)
    neigh_idx = np.full((len(neigh_1), max_deg), -1, dtype=np.int64)
    for fi, lst in enumerate(all_neigh):
        if len(lst) > 0:
            neigh_idx[fi, : len(lst)] = np.asarray(lst, dtype=np.int64)
    return neigh_idx


def compute_face_neighbor_weights(
    *,
    neigh_idx: np.ndarray,
    face_normals_np: np.ndarray,
    sharp_allow_dot: float,
) -> np.ndarray:
    neigh_w = np.zeros_like(neigh_idx, dtype=np.float32)
    if neigh_idx.size == 0:
        return neigh_w
    sharp_allow_dot = float(np.clip(float(sharp_allow_dot), -1.0, 0.999))
    for fi in range(int(neigh_idx.shape[0])):
        for k in range(int(neigh_idx.shape[1])):
            nb = int(neigh_idx[fi, k])
            if nb < 0:
                continue
            dot = float(np.clip(np.dot(face_normals_np[fi], face_normals_np[nb]), -1.0, 1.0))
            if dot <= sharp_allow_dot:
                w = 0.0
            else:
                w = (dot - sharp_allow_dot) / max(1e-6, 1.0 - sharp_allow_dot)
            neigh_w[fi, k] = float(np.clip(w, 0.0, 1.0))
    return neigh_w
@dataclass(slots=True)
class Stage2MeshContext:
    frame_names: list[str]
    bind_mesh_dir: Path | None
    bind_seq_npz_path: Path | None
    bind_vertices_npz_path: Path | None
    bind_deform_obj_path: Path | None
    bind_mesh_indices: list[int] | None
    bind_faces_np: np.ndarray | None
    bind_rigid_align_cfg: dict | None
    bind_rigid_apply_cfg: dict | None
    bind_mesh_offset_np: np.ndarray
    bind_mesh_to_camera: bool
    bind_inverse_export_upright: bool
    bind_mesh_obj_space_mode: str
    mesh_obj_in_camera_space: bool
    exp_upright_rot: np.ndarray
    exp_center_xy: np.ndarray
    exp_floor_z: float
    exp_has_center_xy: bool
    exp_has_floor: bool
    mesh_ids: list[int]
    mesh_ids_set: set[int]
    fallback_mesh_idx: int | None
    verts_seq_np: np.ndarray | None
    verts_seq_ids_set: set[int]
    verts_seq_fallback_idx: int
    mesh_path_map: dict[int, Path]
    frame_smpl_align_scales: np.ndarray
    frame_smpl_align_tx_px: np.ndarray
    frame_smpl_align_ty_px: np.ndarray
    frame_smpl_align_accepted: np.ndarray


class Stage2MeshRuntime:
    def __init__(self, *, context: Stage2MeshContext, device: str | torch.device = "cpu") -> None:
        self.context = context
        self.device = device
        self.mesh_cache: dict[int, torch.Tensor] = {}
        self.mesh_np_cache: dict[int, np.ndarray] = {}
        self.mesh_binding_cache: dict[int, BindingFrameData] = {}
        self.mesh_raw_cache: dict[int, np.ndarray] = {}
        self.mesh_cache_order: list[int] = []
        self.frame_mesh_cache: dict[int, torch.Tensor] = {}
        self.frame_mesh_np_cache: dict[int, np.ndarray] = {}
        self.frame_binding_cache: dict[int, BindingFrameData] = {}
        self.frame_cache_order: list[int] = []
        self.bind_adjacency: list[np.ndarray] | None = None

    @classmethod
    def from_stage2_config(
        cls,
        *,
        project_root: Path,
        stage1_dir: Path,
        cfg: dict[str, Any],
        frames: list[str],
        logger,
        device: str | torch.device,
        bind_to_mesh: bool,
        auto_from_stage2_report: bool = False,
        stage2_dir_for_report: Path | None = None,
    ) -> "Stage2MeshRuntime":
        bind_mesh_dir = resolve_mesh_sequence_dir(
            project_root=project_root,
            stage1_dir=stage1_dir,
            bind_mesh_dir_cfg=cfg.get("bind_mesh_sequence_dir", str(stage1_dir)),
            logger=logger,
        )
        bind_seq_npz_cfg = cfg.get("bind_sequence_npz", str(stage1_dir / "smpl_params_sequence.npz"))
        bind_seq_npz_path = Path(str(bind_seq_npz_cfg)) if bind_seq_npz_cfg is not None else None
        if bind_seq_npz_path is not None and not bind_seq_npz_path.is_absolute():
            bind_seq_npz_path = project_root / bind_seq_npz_path
        bind_vertices_npz_cfg = cfg.get("bind_vertices_npz", str(stage1_dir / "smpl_vertices_sequence.npz"))
        bind_vertices_npz_path = Path(str(bind_vertices_npz_cfg)) if bind_vertices_npz_cfg is not None else None
        if bind_vertices_npz_path is not None and not bind_vertices_npz_path.is_absolute():
            bind_vertices_npz_path = project_root / bind_vertices_npz_path
        if bind_vertices_npz_path is not None and not bind_vertices_npz_path.exists():
            bind_vertices_npz_path = None
        bind_deform_obj_cfg = cfg.get("bind_deform_obj", None)
        bind_deform_obj_path = Path(str(bind_deform_obj_cfg)) if bind_deform_obj_cfg is not None else None
        if bind_deform_obj_path is not None and not bind_deform_obj_path.is_absolute():
            bind_deform_obj_path = project_root / bind_deform_obj_path
        bind_mesh_indices = None
        if bind_to_mesh:
            bind_mesh_indices = resolve_bind_mesh_indices(
                frames=frames,
                bind_seq_npz_path=bind_seq_npz_path,
                bind_mesh_dir=bind_mesh_dir,
                logger=logger,
            )
        rigid_align_cfg = resolve_effective_rigid_align_config(
            explicit_cfg=cfg.get("bind_rigid_align", None),
            stage2_dir=stage2_dir_for_report,
            auto_from_stage2=auto_from_stage2_report,
            logger=logger,
        )
        bind_rigid_apply_cfg = (
            {
                "enabled": True,
                "rx_deg": float(rigid_align_cfg.get("rx_deg", 0.0)),
                "ry_deg": float(rigid_align_cfg.get("ry_deg", 0.0)),
                "rz_deg": float(rigid_align_cfg.get("rz_deg", 0.0)),
                "tx": float(rigid_align_cfg.get("tx", 0.0)),
                "ty": float(rigid_align_cfg.get("ty", 0.0)),
                "tz": float(rigid_align_cfg.get("tz", 0.0)),
            }
            if rigid_align_cfg is not None and bool(rigid_align_cfg.get("enabled", False))
            else None
        )
        bind_mesh_offset_cfg = cfg.get("bind_mesh_offset", (0.0, 0.0, 0.0))
        if not (isinstance(bind_mesh_offset_cfg, (list, tuple)) and len(bind_mesh_offset_cfg) == 3):
            bind_mesh_offset_cfg = (0.0, 0.0, 0.0)
        bind_mesh_offset_np = np.asarray(
            [[float(bind_mesh_offset_cfg[0]), float(bind_mesh_offset_cfg[1]), float(bind_mesh_offset_cfg[2])]],
            dtype=np.float32,
        )
        bind_mesh_to_camera = bool(cfg.get("bind_mesh_to_camera", False))
        bind_inverse_export_upright = bool(cfg.get("bind_inverse_export_upright", True))
        bind_mesh_obj_space_mode = str(cfg.get("bind_mesh_obj_space", "auto")).strip().lower()
        if bind_mesh_obj_space_mode not in ("auto", "camera", "blender"):
            bind_mesh_obj_space_mode = "auto"
        mesh_obj_in_camera_space = bool(bind_mesh_obj_space_mode == "camera")
        exp_upright_rot = np.eye(3, dtype=np.float32)
        exp_center_xy = np.zeros((2,), dtype=np.float32)
        exp_floor_z = 0.0
        exp_has_center_xy = False
        exp_has_floor = False
        if bind_seq_npz_path is not None and bind_seq_npz_path.exists():
            seq = np.load(bind_seq_npz_path, allow_pickle=True)
            if bind_mesh_obj_space_mode == "auto" and "fit_strategy" in seq.files:
                fs_arr = np.asarray(seq["fit_strategy"]).reshape(-1)
                if fs_arr.size > 0 and str(fs_arr[0]).strip().lower() in ("stage1_motion_prior", "vibe"):
                    mesh_obj_in_camera_space = True
            if "export_upright_rot" in seq.files:
                exp_upright_rot = np.asarray(seq["export_upright_rot"], dtype=np.float32).reshape(3, 3)
            if "export_center_xy" in seq.files:
                exp_has_center_xy = bool(int(np.asarray(seq["export_center_xy"]).reshape(-1)[0]))
            if "export_center_xy_value" in seq.files:
                exp_center_xy = np.asarray(seq["export_center_xy_value"], dtype=np.float32).reshape(2,)
            if "export_on_floor" in seq.files:
                exp_has_floor = bool(int(np.asarray(seq["export_on_floor"]).reshape(-1)[0]))
            if "export_floor_z" in seq.files:
                exp_floor_z = float(np.asarray(seq["export_floor_z"], dtype=np.float32).reshape(-1)[0])
        mesh_path_map: dict[int, Path] = {}
        mesh_ids: list[int] = []
        mesh_ids_set: set[int] = set()
        fallback_mesh_idx = None
        bind_faces_np = None
        if bind_mesh_dir is not None and bind_mesh_dir.exists():
            seq_provider = SequenceMeshProvider(bind_mesh_dir)
            seq_provider.scan()
            mesh_ids = seq_provider.get_frame_ids()
            mesh_ids_set = set(mesh_ids)
            if len(mesh_ids) > 0:
                mesh_path_map = {int(i): seq_provider.get_path(int(i)) for i in mesh_ids}
                first_mesh_idx = int(bind_mesh_indices[0]) if bind_mesh_indices else int(mesh_ids[0])
                if first_mesh_idx not in mesh_ids_set:
                    raise FileNotFoundError(f"Bind mesh frame index not found: {first_mesh_idx}")
                try:
                    _, bind_faces_np = load_obj_mesh(mesh_path_map[first_mesh_idx])
                except Exception:
                    bind_faces_np = None
        verts_seq_np = None
        verts_seq_ids_set: set[int] = set()
        verts_seq_fallback_idx = -1
        if bind_vertices_npz_path is not None and bind_vertices_npz_path.exists():
            vseq = np.load(bind_vertices_npz_path, allow_pickle=True)
            if "verts_cam" in vseq.files:
                verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)
                if verts_seq_np.ndim == 3 and verts_seq_np.shape[-1] == 3:
                    verts_seq_ids_set = set(range(int(verts_seq_np.shape[0])))
                else:
                    verts_seq_np = None
        context = Stage2MeshContext(
            frame_names=[str(x) for x in frames],
            bind_mesh_dir=bind_mesh_dir,
            bind_seq_npz_path=bind_seq_npz_path,
            bind_vertices_npz_path=bind_vertices_npz_path,
            bind_deform_obj_path=bind_deform_obj_path,
            bind_mesh_indices=bind_mesh_indices,
            bind_faces_np=bind_faces_np,
            bind_rigid_align_cfg=rigid_align_cfg,
            bind_rigid_apply_cfg=bind_rigid_apply_cfg,
            bind_mesh_offset_np=bind_mesh_offset_np,
            bind_mesh_to_camera=bind_mesh_to_camera,
            bind_inverse_export_upright=bind_inverse_export_upright,
            bind_mesh_obj_space_mode=bind_mesh_obj_space_mode,
            mesh_obj_in_camera_space=mesh_obj_in_camera_space,
            exp_upright_rot=exp_upright_rot,
            exp_center_xy=exp_center_xy,
            exp_floor_z=exp_floor_z,
            exp_has_center_xy=exp_has_center_xy,
            exp_has_floor=exp_has_floor,
            mesh_ids=mesh_ids,
            mesh_ids_set=mesh_ids_set,
            fallback_mesh_idx=fallback_mesh_idx,
            verts_seq_np=verts_seq_np,
            verts_seq_ids_set=verts_seq_ids_set,
            verts_seq_fallback_idx=verts_seq_fallback_idx,
            mesh_path_map=mesh_path_map,
            frame_smpl_align_scales=np.ones((len(frames),), dtype=np.float32),
            frame_smpl_align_tx_px=np.zeros((len(frames),), dtype=np.float32),
            frame_smpl_align_ty_px=np.zeros((len(frames),), dtype=np.float32),
            frame_smpl_align_accepted=np.zeros((len(frames),), dtype=np.uint8),
        )
        return cls(context=context, device=device)

    def clear_caches(self) -> None:
        self.mesh_cache.clear()
        self.mesh_np_cache.clear()
        self.mesh_binding_cache.clear()
        self.mesh_raw_cache.clear()
        self.mesh_cache_order.clear()
        self.frame_mesh_cache.clear()
        self.frame_mesh_np_cache.clear()
        self.frame_binding_cache.clear()
        self.frame_cache_order.clear()
        self.bind_adjacency = None

    def set_effective_rigid_align_config(self, rigid_align_cfg: dict | None) -> None:
        self.context.bind_rigid_align_cfg = None if rigid_align_cfg is None else dict(rigid_align_cfg)
        self.context.bind_rigid_apply_cfg = (
            None
            if rigid_align_cfg is None or not bool(rigid_align_cfg.get("enabled", False))
            else {
                "enabled": True,
                "rx_deg": float(rigid_align_cfg.get("rx_deg", 0.0)),
                "ry_deg": float(rigid_align_cfg.get("ry_deg", 0.0)),
                "rz_deg": float(rigid_align_cfg.get("rz_deg", 0.0)),
                "tx": float(rigid_align_cfg.get("tx", 0.0)),
                "ty": float(rigid_align_cfg.get("ty", 0.0)),
                "tz": float(rigid_align_cfg.get("tz", 0.0)),
            }
        )
        self.clear_caches()

    def set_rigid_apply_config(self, rigid_apply_cfg: dict | None) -> None:
        self.context.bind_rigid_apply_cfg = None if rigid_apply_cfg is None else dict(rigid_apply_cfg)
        self.clear_caches()

    def set_frame_smpl_align_from_arrays(
        self,
        *,
        frame_names: list[str],
        scales: np.ndarray,
        tx_px: np.ndarray,
        ty_px: np.ndarray,
        accepted: np.ndarray,
    ) -> None:
        if [str(x) for x in frame_names] != list(self.context.frame_names):
            raise ValueError("frame_names must exactly match mesh runtime frame order.")
        n = len(self.context.frame_names)
        self.context.frame_smpl_align_scales = np.asarray(scales, dtype=np.float32).reshape(n)
        self.context.frame_smpl_align_tx_px = np.asarray(tx_px, dtype=np.float32).reshape(n)
        self.context.frame_smpl_align_ty_px = np.asarray(ty_px, dtype=np.float32).reshape(n)
        self.context.frame_smpl_align_accepted = np.asarray(accepted, dtype=np.uint8).reshape(n)
        self.clear_caches()

    def resolve_mesh_idx(self, mesh_idx: int) -> int:
        mesh_idx_use = int(mesh_idx)
        if self.context.verts_seq_np is not None:
            if mesh_idx_use not in self.context.verts_seq_ids_set:
                raise FileNotFoundError(f"Vertex sequence frame index not found: {mesh_idx_use}")
        elif self.context.mesh_ids_set:
            if mesh_idx_use not in self.context.mesh_ids_set:
                raise FileNotFoundError(f"Mesh frame index not found: {mesh_idx_use}")
        return mesh_idx_use

    def load_mesh_np_raw(self, mesh_idx: int) -> tuple[np.ndarray, int]:
        mesh_idx_use = self.resolve_mesh_idx(mesh_idx)
        if mesh_idx_use in self.mesh_raw_cache:
            return self.mesh_raw_cache[mesh_idx_use], mesh_idx_use
        if self.context.verts_seq_np is not None:
            verts_np = np.asarray(self.context.verts_seq_np[mesh_idx_use], dtype=np.float32)
        else:
            obj_path = self.context.mesh_path_map.get(mesh_idx_use, None)
            if obj_path is None:
                raise FileNotFoundError(f"Mesh frame index not found: {mesh_idx_use}")
            verts_np = load_obj_vertices(obj_path)
            if self.context.bind_mesh_to_camera and (not bool(self.context.mesh_obj_in_camera_space)):
                if self.context.bind_inverse_export_upright:
                    if self.context.exp_has_center_xy:
                        verts_np[:, 0] += float(self.context.exp_center_xy[0])
                        verts_np[:, 1] += float(self.context.exp_center_xy[1])
                    if self.context.exp_has_floor:
                        verts_np[:, 2] += float(self.context.exp_floor_z)
                    verts_np = verts_np @ self.context.exp_upright_rot.astype(np.float32)
                verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
        self.mesh_raw_cache[mesh_idx_use] = np.asarray(verts_np, dtype=np.float32)
        return self.mesh_raw_cache[mesh_idx_use], mesh_idx_use

    def apply_mesh_transforms_np(self, verts_np: np.ndarray) -> np.ndarray:
        v = np.asarray(verts_np, dtype=np.float32)
        if self.context.bind_rigid_apply_cfg is not None:
            v = apply_rigid_transform_np(v, self.context.bind_rigid_apply_cfg)
        return np.asarray(v + self.context.bind_mesh_offset_np, dtype=np.float32)

    def apply_frame_similarity_np(self, verts_np: np.ndarray, *, frame_idx: int, K_np: np.ndarray | None) -> np.ndarray:
        if K_np is None:
            return np.asarray(verts_np, dtype=np.float32)
        if frame_idx < 0 or frame_idx >= len(self.context.frame_names):
            return np.asarray(verts_np, dtype=np.float32)
        if int(self.context.frame_smpl_align_accepted[frame_idx]) <= 0:
            return np.asarray(verts_np, dtype=np.float32)
        return np.asarray(
            apply_similarity_to_camera_points(
                np.asarray(verts_np, dtype=np.float32),
                fx=float(K_np[0, 0]),
                fy=float(K_np[1, 1]),
                cx=float(K_np[0, 2]),
                cy=float(K_np[1, 2]),
                scale=float(self.context.frame_smpl_align_scales[frame_idx]),
                tx_px=float(self.context.frame_smpl_align_tx_px[frame_idx]),
                ty_px=float(self.context.frame_smpl_align_ty_px[frame_idx]),
            ),
            dtype=np.float32,
        )

    def load_mesh_verts(self, mesh_idx: int, *, bind_cache_size: int = 8) -> tuple[torch.Tensor, int]:
        mesh_idx_use = self.resolve_mesh_idx(mesh_idx)
        if mesh_idx_use in self.mesh_cache:
            return self.mesh_cache[mesh_idx_use], mesh_idx_use
        verts_raw_np, mesh_idx_use = self.load_mesh_np_raw(mesh_idx_use)
        verts_np = self.apply_mesh_transforms_np(verts_raw_np)
        verts_t = torch.from_numpy(verts_np).to(device=self.device, dtype=torch.float32)
        self.mesh_cache[mesh_idx_use] = verts_t
        self.mesh_np_cache[mesh_idx_use] = np.asarray(verts_np, dtype=np.float32)
        self.mesh_cache_order.append(mesh_idx_use)
        if len(self.mesh_cache_order) > int(bind_cache_size):
            old = self.mesh_cache_order.pop(0)
            self.mesh_cache.pop(old, None)
            self.mesh_np_cache.pop(old, None)
            self.mesh_binding_cache.pop(old, None)
            self.mesh_raw_cache.pop(old, None)
        return verts_t, mesh_idx_use

    def load_frame_mesh_verts(
        self,
        frame_idx: int,
        *,
        K_np: np.ndarray | None,
        bind_cache_size: int = 8,
    ) -> tuple[torch.Tensor, int]:
        frame_idx = int(frame_idx)
        if frame_idx in self.frame_mesh_cache:
            mesh_idx_use = int(self.resolve_mesh_idx(self.context.bind_mesh_indices[frame_idx])) if self.context.bind_mesh_indices is not None else frame_idx
            return self.frame_mesh_cache[frame_idx], mesh_idx_use
        if self.context.bind_mesh_indices is None or frame_idx < 0 or frame_idx >= len(self.context.bind_mesh_indices):
            raise IndexError(f"frame_idx out of range for frame-bound mesh load: {frame_idx}")
        mesh_idx_use = int(self.resolve_mesh_idx(self.context.bind_mesh_indices[frame_idx]))
        verts_raw_np, mesh_idx_use = self.load_mesh_np_raw(mesh_idx_use)
        verts_np = self.apply_mesh_transforms_np(verts_raw_np)
        verts_np = self.apply_frame_similarity_np(verts_np, frame_idx=frame_idx, K_np=K_np)
        verts_t = torch.from_numpy(verts_np).to(device=self.device, dtype=torch.float32)
        self.frame_mesh_cache[frame_idx] = verts_t
        self.frame_mesh_np_cache[frame_idx] = np.asarray(verts_np, dtype=np.float32)
        self.frame_cache_order.append(frame_idx)
        if len(self.frame_cache_order) > int(bind_cache_size):
            old = int(self.frame_cache_order.pop(0))
            self.frame_mesh_cache.pop(old, None)
            self.frame_mesh_np_cache.pop(old, None)
            self.frame_binding_cache.pop(old, None)
        return verts_t, mesh_idx_use

    def load_binding_frame(self, mesh_idx: int, *, binding_cfg_dict: dict | None, bind_cache_size: int = 8) -> BindingFrameData:
        mesh_idx_use = self.resolve_mesh_idx(mesh_idx)
        if mesh_idx_use in self.mesh_binding_cache:
            return self.mesh_binding_cache[mesh_idx_use]
        if mesh_idx_use not in self.mesh_np_cache:
            self.load_mesh_verts(mesh_idx_use, bind_cache_size=bind_cache_size)
        verts_np = self.mesh_np_cache[mesh_idx_use]
        if self.bind_adjacency is None or len(self.bind_adjacency) != int(verts_np.shape[0]):
            self.bind_adjacency = build_vertex_adjacency(int(verts_np.shape[0]), self.context.bind_faces_np)
        binding_cfg = parse_binding_mode_config(binding_cfg_dict, default_mode="games_face")
        frame = build_binding_frame_data(
            verts_np=verts_np,
            faces_np=self.context.bind_faces_np,
            adjacency=self.bind_adjacency,
            cfg=binding_cfg,
            device=self.device,
        )
        self.mesh_binding_cache[mesh_idx_use] = frame
        return frame

    def load_frame_binding_frame(
        self,
        frame_idx: int,
        *,
        K_np: np.ndarray | None,
        binding_cfg_dict: dict | None,
        bind_cache_size: int = 8,
    ) -> BindingFrameData:
        frame_idx = int(frame_idx)
        if frame_idx in self.frame_binding_cache:
            return self.frame_binding_cache[frame_idx]
        if frame_idx not in self.frame_mesh_np_cache:
            self.load_frame_mesh_verts(frame_idx, K_np=K_np, bind_cache_size=bind_cache_size)
        verts_np = np.asarray(self.frame_mesh_np_cache[frame_idx], dtype=np.float32)
        if self.bind_adjacency is None or len(self.bind_adjacency) != int(verts_np.shape[0]):
            self.bind_adjacency = build_vertex_adjacency(int(verts_np.shape[0]), self.context.bind_faces_np)
        binding_cfg = parse_binding_mode_config(binding_cfg_dict, default_mode="games_face")
        frame = build_binding_frame_data(
            verts_np=verts_np,
            faces_np=self.context.bind_faces_np,
            adjacency=self.bind_adjacency,
            cfg=binding_cfg,
            device=self.device,
        )
        self.frame_binding_cache[frame_idx] = frame
        return frame


@dataclass(slots=True)
class Stage2TrainingBindingState:
    mesh_normal_cache: dict[int, torch.Tensor]
    smpl_mask_cache: dict[int, torch.Tensor]
    bind_idx_t: torch.Tensor | None
    bind_vertex_offset_t: torch.Tensor | None
    bind_face_idx_t: torch.Tensor | None
    bind_bary_t: torch.Tensor | None
    bind_faces_np: np.ndarray | None
    bind_face_neighbor_idx_t: torch.Tensor | None
    bind_face_neighbor_w_t: torch.Tensor | None
    outer_face_weight_sum: torch.Tensor | None
    outer_face_align_peak: torch.Tensor | None
    outer_face_gap_half_weighted_sum: torch.Tensor | None
    outer_face_gap_scale_weighted_sum: torch.Tensor | None
    outer_face_spawned_mask: torch.Tensor | None
    outer_face_first_hit_frame_idx: list[int | None] | None
    outer_face_contrib_frames: list[set[int]] | None
    outer_face_spawn_iter: list[int | None] | None
    outer_face_spawn_frame_idx: list[int | None] | None
    bind_rigid_align_meta: dict | None
    bind_rigid_apply_cfg: dict | None
    load_mesh_np_raw: Callable[[int], tuple[np.ndarray, int]]
    load_mesh_verts: Callable[[int], tuple[torch.Tensor, torch.Tensor | None, int]]
    load_binding_frame: Callable[[int], BindingFrameData]


def initialize_stage2_training_binding_state(
    *,
    binding_enabled: bool,
    mesh_runtime: Stage2MeshRuntime | None,
    bind_vertex_indices: torch.Tensor | None,
    g_bind_face_idx: torch.Tensor | None,
    g_bind_bary: torch.Tensor | None,
    g_bind_offset: torch.Tensor | None,
    frames_list: list[str],
    bind_mesh_indices: list[int],
    bind_cache_size: int,
    device: str | torch.device,
    use_smpl_silhouette_mask: bool,
    bind_align_quat_to_mesh_normals: bool,
    binding_cfg,
    binding_cfg_dict: dict | None,
    outer_airspace_neighbor_rings: int,
    outer_airspace_sharp_allow_dot: float,
    bind_rigid_align_enabled: bool,
    bind_rigid_align_cfg: dict | None,
    person_masks_npz: str | Path | None,
    K_np: np.ndarray,
    frame_K_nps: list[np.ndarray] | None,
    image_hw: tuple[int, int],
    axis_angle_to_rotmat_fn: Callable[[torch.Tensor], torch.Tensor],
    cam_w: torch.Tensor,
    cam_t: torch.Tensor,
    logger,
) -> Stage2TrainingBindingState:
    mesh_normal_cache: dict[int, torch.Tensor] = {}
    smpl_mask_cache: dict[int, torch.Tensor] = {}
    bind_idx_t = None
    bind_vertex_offset_t = None
    bind_face_idx_t = None
    bind_bary_t = None
    bind_faces_np = None
    bind_face_neighbor_idx_t: torch.Tensor | None = None
    bind_face_neighbor_w_t: torch.Tensor | None = None
    outer_face_weight_sum: torch.Tensor | None = None
    outer_face_align_peak: torch.Tensor | None = None
    outer_face_gap_half_weighted_sum: torch.Tensor | None = None
    outer_face_gap_scale_weighted_sum: torch.Tensor | None = None
    outer_face_spawned_mask: torch.Tensor | None = None
    outer_face_first_hit_frame_idx: list[int | None] | None = None
    outer_face_contrib_frames: list[set[int]] | None = None
    outer_face_spawn_iter: list[int | None] | None = None
    outer_face_spawn_frame_idx: list[int | None] | None = None
    bind_rigid_align_meta: dict | None = None
    bind_rigid_apply_cfg = None if mesh_runtime is None or mesh_runtime.context.bind_rigid_apply_cfg is None else dict(mesh_runtime.context.bind_rigid_apply_cfg)

    def _load_mesh_np_raw(mesh_idx: int) -> tuple[np.ndarray, int]:
        if mesh_runtime is None:
            raise RuntimeError('binding mesh runtime is not available')
        return mesh_runtime.load_mesh_np_raw(mesh_idx)

    def _load_mesh_verts(frame_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        if mesh_runtime is None:
            raise RuntimeError('binding mesh runtime is not available')
        K_curr_np = frame_K_nps[int(frame_idx)] if frame_K_nps is not None else K_np
        verts_t, mesh_idx_use = mesh_runtime.load_frame_mesh_verts(
            int(frame_idx),
            K_np=K_curr_np,
            bind_cache_size=bind_cache_size,
        )
        normals_t = mesh_normal_cache.get(mesh_idx_use, None)
        if normals_t is None and bool(bind_align_quat_to_mesh_normals) and bind_faces_np is not None:
            verts_np = mesh_runtime.frame_mesh_np_cache.get(int(frame_idx), None)
            if verts_np is None:
                verts_np = verts_t.detach().cpu().numpy().astype(np.float32)
                mesh_runtime.frame_mesh_np_cache[int(frame_idx)] = verts_np
            normals_np = compute_vertex_normals_np(verts_np, bind_faces_np)
            normals_t = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)
            mesh_normal_cache[mesh_idx_use] = normals_t
        return verts_t, normals_t, mesh_idx_use

    def _load_binding_frame(frame_idx: int) -> BindingFrameData:
        if mesh_runtime is None:
            raise RuntimeError('binding mesh runtime is not available')
        K_curr_np = frame_K_nps[int(frame_idx)] if frame_K_nps is not None else K_np
        return mesh_runtime.load_frame_binding_frame(
            int(frame_idx),
            K_np=K_curr_np,
            binding_cfg_dict=binding_cfg_dict,
            bind_cache_size=bind_cache_size,
        )

    if binding_enabled:
        if mesh_runtime is None:
            raise ValueError(
                'binding_enabled requires stage2_mesh_runtime; use stage2 entrypoint orchestration instead of the legacy direct binding path.'
            )
        if bind_vertex_indices is None:
            raise ValueError('binding_enabled requires bind_vertex_indices.')
        bind_idx_t = bind_vertex_indices.to(device=device, dtype=torch.long)
        if g_bind_face_idx is not None:
            bind_face_idx_t = g_bind_face_idx.to(device=device, dtype=torch.long)
        if g_bind_bary is not None:
            bind_bary_t = g_bind_bary.to(device=device, dtype=torch.float32)
        if g_bind_offset is not None:
            bind_vertex_offset_t = g_bind_offset.to(device=device, dtype=torch.float32)
        if len(bind_mesh_indices) != len(frames_list):
            raise ValueError('bind_mesh_indices must align with frames_list length.')
        bind_faces_np = mesh_runtime.context.bind_faces_np
        bind_mesh_offset_np = np.asarray(mesh_runtime.context.bind_mesh_offset_np, dtype=np.float32).reshape(1, 3)

        if (
            bool(use_smpl_silhouette_mask)
            or bool(bind_align_quat_to_mesh_normals)
            or str(binding_cfg.mode).lower() in ('games_face', 'hybrid')
        ):
            if len(bind_mesh_indices) == 0:
                raise RuntimeError('use_smpl_silhouette_mask=true but bind_mesh_indices is empty.')
            if bind_faces_np is None or len(bind_faces_np) == 0:
                if bool(use_smpl_silhouette_mask):
                    raise RuntimeError('use_smpl_silhouette_mask=true but mesh has no faces; cannot build silhouette.')
                if str(binding_cfg.mode).lower() == 'games_face':
                    raise RuntimeError('binding.mode=games_face requires mesh faces, but loaded mesh has no faces.')
                bind_faces_np = None

        if bind_faces_np is not None and len(bind_faces_np) > 0:
            verts_ref_np, _ = mesh_runtime.load_mesh_np_raw(int(bind_mesh_indices[0]))
            tri_ref = np.asarray(verts_ref_np, dtype=np.float32)[np.asarray(bind_faces_np, dtype=np.int64)]
            face_normals_ref = np.cross(
                tri_ref[:, 1, :] - tri_ref[:, 0, :],
                tri_ref[:, 2, :] - tri_ref[:, 0, :],
            ).astype(np.float32)
            face_normals_ref = face_normals_ref / np.clip(
                np.linalg.norm(face_normals_ref, axis=1, keepdims=True),
                1e-8,
                None,
            )
            face_neigh_idx_np = pack_face_neighbor_arrays(
                bind_faces_np,
                rings=int(max(outer_airspace_neighbor_rings, 1)),
            )
            face_neigh_w_np = compute_face_neighbor_weights(
                neigh_idx=face_neigh_idx_np,
                face_normals_np=face_normals_ref,
                sharp_allow_dot=float(outer_airspace_sharp_allow_dot),
            )
            bind_face_neighbor_idx_t = torch.as_tensor(face_neigh_idx_np, device=device, dtype=torch.long)
            bind_face_neighbor_w_t = torch.as_tensor(face_neigh_w_np, device=device, dtype=torch.float32)
            n_bind_faces = int(bind_faces_np.shape[0])
            outer_face_weight_sum = torch.zeros((n_bind_faces,), dtype=torch.float32, device=device)
            outer_face_align_peak = torch.zeros((n_bind_faces,), dtype=torch.float32, device=device)
            outer_face_gap_half_weighted_sum = torch.zeros((n_bind_faces,), dtype=torch.float32, device=device)
            outer_face_gap_scale_weighted_sum = torch.zeros((n_bind_faces,), dtype=torch.float32, device=device)
            outer_face_spawned_mask = torch.zeros((n_bind_faces,), dtype=torch.bool, device=device)
            outer_face_first_hit_frame_idx = [None] * n_bind_faces
            outer_face_contrib_frames = [set() for _ in range(n_bind_faces)]
            outer_face_spawn_iter = [None] * n_bind_faces
            outer_face_spawn_frame_idx = [None] * n_bind_faces

        if (
            bind_rigid_align_enabled
            and bind_rigid_align_cfg is not None
            and bool(bind_rigid_align_cfg.get('auto', False))
            and bind_faces_np is not None
        ):
            pm_cfg = bind_rigid_align_cfg.get('person_masks_npz', None)
            pm_path = Path(pm_cfg) if pm_cfg is not None else (Path(person_masks_npz) if person_masks_npz is not None else None)
            if pm_path is not None:
                align_result = run_stage2_rigid_align_search(
                    cfg=bind_rigid_align_cfg,
                    frames=[str(x) for x in frames_list],
                    bind_mesh_indices=[int(x) for x in bind_mesh_indices],
                    K_np=K_np,
                    image_hw=image_hw,
                    bind_faces_np=bind_faces_np,
                    mesh_loader=lambda mesh_idx: _load_mesh_np_raw(mesh_idx)[0],
                    person_masks_npz_path=pm_path,
                    logger=logger,
                    bind_mesh_offset_np=bind_mesh_offset_np,
                    camera_resolver=lambda fj: (
                        K_np,
                        axis_angle_to_rotmat_fn(cam_w[fj]).detach().cpu().numpy().astype(np.float32),
                        cam_t[fj].detach().cpu().numpy().astype(np.float32),
                    ),
                )
                if align_result.metrics is not None and align_result.effective_cfg is not None:
                    mesh_runtime.set_effective_rigid_align_config(align_result.effective_cfg)
                    bind_rigid_apply_cfg = None if mesh_runtime.context.bind_rigid_apply_cfg is None else dict(mesh_runtime.context.bind_rigid_apply_cfg)
                    mesh_normal_cache.clear()
                    smpl_mask_cache.clear()
                    bind_rigid_align_meta = {
                        'enabled': True,
                        'auto': True,
                        'params': dict(bind_rigid_apply_cfg),
                        'sample_frames': int(align_result.metrics.get('sample_frames', 0.0)),
                        'score_before': float(align_result.metrics.get('score_before', 0.0)),
                        'score_after': float(align_result.metrics.get('score_after', 0.0)),
                        'iou_before': float(align_result.metrics.get('iou_before', 0.0)),
                        'iou_after': float(align_result.metrics.get('iou_after', 0.0)),
                        'outside_before': float(align_result.metrics.get('outside_before', 0.0)),
                        'outside_after': float(align_result.metrics.get('outside_after', 0.0)),
                        'person_masks_npz': str(pm_path),
                    }

        if bind_rigid_align_enabled and bind_rigid_apply_cfg is not None and bind_rigid_align_meta is None:
            bind_rigid_align_meta = {
                'enabled': True,
                'auto': False,
                'params': dict(bind_rigid_apply_cfg),
            }
            logger.info(
                'bind_rigid_align manual: ' +
                f"rot=({bind_rigid_apply_cfg['rx_deg']:.3f},{bind_rigid_apply_cfg['ry_deg']:.3f},{bind_rigid_apply_cfg['rz_deg']:.3f})deg, " +
                f"trans=({bind_rigid_apply_cfg['tx']:.4f},{bind_rigid_apply_cfg['ty']:.4f},{bind_rigid_apply_cfg['tz']:.4f})"
            )

    return Stage2TrainingBindingState(
        mesh_normal_cache=mesh_normal_cache,
        smpl_mask_cache=smpl_mask_cache,
        bind_idx_t=bind_idx_t,
        bind_vertex_offset_t=bind_vertex_offset_t,
        bind_face_idx_t=bind_face_idx_t,
        bind_bary_t=bind_bary_t,
        bind_faces_np=bind_faces_np,
        bind_face_neighbor_idx_t=bind_face_neighbor_idx_t,
        bind_face_neighbor_w_t=bind_face_neighbor_w_t,
        outer_face_weight_sum=outer_face_weight_sum,
        outer_face_align_peak=outer_face_align_peak,
        outer_face_gap_half_weighted_sum=outer_face_gap_half_weighted_sum,
        outer_face_gap_scale_weighted_sum=outer_face_gap_scale_weighted_sum,
        outer_face_spawned_mask=outer_face_spawned_mask,
        outer_face_first_hit_frame_idx=outer_face_first_hit_frame_idx,
        outer_face_contrib_frames=outer_face_contrib_frames,
        outer_face_spawn_iter=outer_face_spawn_iter,
        outer_face_spawn_frame_idx=outer_face_spawn_frame_idx,
        bind_rigid_align_meta=bind_rigid_align_meta,
        bind_rigid_apply_cfg=bind_rigid_apply_cfg,
        load_mesh_np_raw=_load_mesh_np_raw,
        load_mesh_verts=_load_mesh_verts,
        load_binding_frame=_load_binding_frame,
    )
