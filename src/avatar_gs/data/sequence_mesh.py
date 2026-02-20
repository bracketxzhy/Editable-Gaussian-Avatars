from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re

import numpy as np

from avatar_gs.gaussian.init_from_smpl import load_obj_mesh


_FRAME_RE = re.compile(r"^frame_(\d+)$", re.IGNORECASE)
_DIGIT_RE = re.compile(r"^(\d+)$")


def _parse_frame_idx_from_stem(stem: str) -> int | None:
    m = _FRAME_RE.match(stem)
    if m is not None:
        return int(m.group(1))
    m = _DIGIT_RE.match(stem)
    if m is not None:
        return int(m.group(1))
    return None


class SequenceMeshProvider:
    """Read stage1 sequence meshes with stable frame-index mapping and LRU cache."""

    def __init__(self, sequence_dir: str | Path, cache_size: int = 16):
        self.sequence_dir = Path(sequence_dir)
        self.cache_size = max(int(cache_size), 1)
        self._frame_to_path: dict[int, Path] = {}
        self._sorted_frame_ids: list[int] = []
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def scan(self) -> None:
        if not self.sequence_dir.exists():
            raise FileNotFoundError(
                f"Sequence directory not found: {self.sequence_dir}. Run stage1 first."
            )

        frame_to_path: dict[int, Path] = {}
        for p in self.sequence_dir.glob("*.obj"):
            frame_idx = _parse_frame_idx_from_stem(p.stem)
            if frame_idx is None:
                continue
            if frame_idx in frame_to_path:
                # Prefer canonical frame_XXXXXX.obj naming over legacy numeric naming.
                old = frame_to_path[frame_idx]
                old_is_frame = old.stem.lower().startswith("frame_")
                new_is_frame = p.stem.lower().startswith("frame_")
                if old_is_frame and not new_is_frame:
                    continue
            frame_to_path[frame_idx] = p

        if len(frame_to_path) == 0:
            raise FileNotFoundError(
                f"No sequence OBJ meshes found in {self.sequence_dir}. "
                "Expected files like frame_000000.obj (legacy 000000.obj is also accepted)."
            )

        self._frame_to_path = frame_to_path
        self._sorted_frame_ids = sorted(frame_to_path.keys())
        self._cache.clear()

    def get_frame_ids(self) -> list[int]:
        if not self._sorted_frame_ids:
            self.scan()
        return list(self._sorted_frame_ids)

    def get_frame_count(self) -> int:
        if not self._sorted_frame_ids:
            self.scan()
        return len(self._sorted_frame_ids)

    def get_path(self, frame_idx: int) -> Path:
        if not self._frame_to_path:
            self.scan()
        key = int(frame_idx)
        if key not in self._frame_to_path:
            raise KeyError(f"Frame index {key} not found in sequence: {self.sequence_dir}")
        return self._frame_to_path[key]

    def has_frame(self, frame_idx: int) -> bool:
        if not self._frame_to_path:
            self.scan()
        return int(frame_idx) in self._frame_to_path

    def get_mesh(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._frame_to_path:
            self.scan()
        key = int(frame_idx)
        if key in self._cache:
            mesh = self._cache.pop(key)
            self._cache[key] = mesh
            return mesh

        path = self.get_path(key)
        verts, faces = load_obj_mesh(path)
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        self._cache[key] = (verts, faces)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return verts, faces

