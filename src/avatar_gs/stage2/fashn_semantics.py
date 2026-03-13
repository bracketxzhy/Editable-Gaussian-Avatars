from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Stage2FaceSemantics:
    npz_path: Path
    face_count: int
    raw_label_ids: np.ndarray
    raw_label_confidence: np.ndarray
    group_ids: np.ndarray
    group_confidence: np.ndarray
    exclude_outer_mask: np.ndarray | None
    raw_label_names: tuple[str, ...]
    group_names: tuple[str, ...]


def load_stage2_face_semantics(
    npz_path: str | Path,
    *,
    expected_face_count: int | None = None,
) -> Stage2FaceSemantics:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"SMPL face semantics asset not found: {path}")

    z = np.load(str(path), allow_pickle=True)
    required = (
        "raw_label_ids",
        "raw_label_confidence",
        "group_ids",
        "group_confidence",
    )
    missing = [key for key in required if key not in z.files]
    if missing:
        raise RuntimeError(f"Bad SMPL face semantics asset {path}: missing keys {missing}")

    raw_label_ids = np.asarray(z["raw_label_ids"], dtype=np.int16).reshape(-1)
    raw_label_confidence = np.asarray(z["raw_label_confidence"], dtype=np.float32).reshape(-1)
    group_ids = np.asarray(z["group_ids"], dtype=np.int16).reshape(-1)
    group_confidence = np.asarray(z["group_confidence"], dtype=np.float32).reshape(-1)
    face_count = int(raw_label_ids.shape[0])

    if int(raw_label_confidence.shape[0]) != face_count:
        raise RuntimeError(
            f"Bad SMPL face semantics asset {path}: raw_label_confidence length mismatch "
            f"({raw_label_confidence.shape[0]} vs {face_count})"
        )
    if int(group_ids.shape[0]) != face_count or int(group_confidence.shape[0]) != face_count:
        raise RuntimeError(
            f"Bad SMPL face semantics asset {path}: group length mismatch "
            f"({group_ids.shape[0]}, {group_confidence.shape[0]} vs {face_count})"
        )
    if expected_face_count is not None and int(expected_face_count) != face_count:
        raise RuntimeError(
            f"SMPL face semantics face count mismatch: asset={face_count}, expected={expected_face_count} "
            f"from current stage2 mesh. Asset: {path}"
        )

    exclude_outer_mask = (
        np.asarray(z["exclude_outer_mask"], dtype=np.uint8).reshape(-1)
        if "exclude_outer_mask" in z.files
        else None
    )
    if exclude_outer_mask is not None and int(exclude_outer_mask.shape[0]) != face_count:
        raise RuntimeError(
            f"Bad SMPL face semantics asset {path}: exclude_outer_mask length mismatch "
            f"({exclude_outer_mask.shape[0]} vs {face_count})"
        )

    raw_label_names = tuple(str(x) for x in np.asarray(z["raw_label_names"]).reshape(-1).tolist()) if "raw_label_names" in z.files else ()
    group_names = tuple(str(x) for x in np.asarray(z["group_names"]).reshape(-1).tolist()) if "group_names" in z.files else ()

    return Stage2FaceSemantics(
        npz_path=path,
        face_count=face_count,
        raw_label_ids=raw_label_ids,
        raw_label_confidence=raw_label_confidence,
        group_ids=group_ids,
        group_confidence=group_confidence,
        exclude_outer_mask=exclude_outer_mask,
        raw_label_names=raw_label_names,
        group_names=group_names,
    )
