from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NPZ = REPO_ROOT / "assets" / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_TARGETS = [
    REPO_ROOT / "assets" / "smplx" / "SMPL-X__FLAME_vertex_ids.npy",
    REPO_ROOT / "third_party" / "SMPLest-X" / "human_models" / "human_model_files" / "smplx" / "SMPL-X__FLAME_vertex_ids.npy",
]


def build_vertex_ids(model_npz: Path) -> np.ndarray:
    data = np.load(model_npz)
    faces = np.asarray(data["f"], dtype=np.int64)
    base_idx = np.asarray(data["lmk_faces_idx"], dtype=np.int64).reshape(-1)
    dyn_idx = np.asarray(data["dynamic_lmk_faces_idx"], dtype=np.int64).reshape(-1)
    face_idx = np.unique(np.concatenate([base_idx, dyn_idx], axis=0))
    face_idx = face_idx[(face_idx >= 0) & (face_idx < faces.shape[0])]
    if face_idx.size == 0:
        raise RuntimeError(f"No valid landmark faces found in {model_npz}")
    vertex_ids = np.unique(faces[face_idx].reshape(-1)).astype(np.int64)
    if vertex_ids.size == 0:
        raise RuntimeError(f"No vertex ids derived from {model_npz}")
    return vertex_ids


def main() -> None:
    model_npz = DEFAULT_MODEL_NPZ
    if not model_npz.exists():
        raise FileNotFoundError(f"SMPL-X model asset not found: {model_npz}")
    vertex_ids = build_vertex_ids(model_npz)
    for target in DEFAULT_TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, vertex_ids)
        print(f"wrote {target} ({vertex_ids.size} ids)")


if __name__ == "__main__":
    main()
