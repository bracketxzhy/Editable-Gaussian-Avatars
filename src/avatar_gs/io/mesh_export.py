from __future__ import annotations

from pathlib import Path

import numpy as np


def build_obj_face_text(faces: np.ndarray) -> str:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")
    f1 = f + 1
    return "".join(f"f {int(tri[0])} {int(tri[1])} {int(tri[2])}\n" for tri in f1)


def export_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray, face_text: str | None = None) -> None:
    """Write a triangle mesh OBJ with explicit vertex/face records."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {verts.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")

    verts_text = "".join(f"v {float(v[0]):.7f} {float(v[1]):.7f} {float(v[2]):.7f}\n" for v in verts)
    faces_text = face_text if face_text is not None else build_obj_face_text(f)
    with out_path.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write(verts_text)
        fp.write(faces_text)
