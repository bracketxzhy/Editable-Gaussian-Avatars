from __future__ import annotations

import importlib.util
import sys
import threading
import warnings
from pathlib import Path

import numpy as np


_FASHN_LOCK = threading.Lock()
_FASHN_IMPORT_PATCHED = False
_FASHN_PARSER_CACHE: dict[tuple[str, str, str], object] = {}
_FASHN_LABEL_CACHE: dict[str, tuple[dict[int, str], dict[str, int]]] = {}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FASHN_MODEL_DIR = PROJECT_ROOT / "third_party" / "FASHN"
DEFAULT_FASHN_RUNTIME_DIR = PROJECT_ROOT / "assets" / "fashn_local" / "runtime"

FASHN_EXPOSED_LABELS = frozenset({"face", "arms", "hands", "legs", "feet", "torso"})
FASHN_COVERED_LABELS = frozenset({"hair", "top", "dress", "skirt", "pants", "belt", "bag", "hat", "scarf", "glasses", "jewelry"})

FASHN_RAW_TO_COARSE = {
    "background": "background",
    "face": "exposed",
    "hair": "covered",
    "top": "covered",
    "dress": "covered",
    "skirt": "covered",
    "pants": "covered",
    "belt": "covered",
    "bag": "covered",
    "hat": "covered",
    "scarf": "covered",
    "glasses": "covered",
    "arms": "exposed",
    "hands": "exposed",
    "legs": "exposed",
    "feet": "exposed",
    "torso": "exposed",
    "jewelry": "covered",
}

FASHN_COARSE_GROUPS = ("background", "exposed", "covered", "unknown")
FASHN_COARSE_TO_ID = {name: idx for idx, name in enumerate(FASHN_COARSE_GROUPS)}


def resolve_fashn_model_dir(model_dir: str | Path | None = None) -> Path:
    path = DEFAULT_FASHN_MODEL_DIR if model_dir is None else Path(model_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing FASHN model directory: {path}")
    return path


def resolve_fashn_runtime_dir(runtime_dir: str | Path | None = None) -> Path:
    path = DEFAULT_FASHN_RUNTIME_DIR if runtime_dir is None else Path(runtime_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing FASHN runtime directory: {path}")
    return path


def _disable_broken_sklearn_probe() -> None:
    global _FASHN_IMPORT_PATCHED
    if _FASHN_IMPORT_PATCHED:
        return
    orig_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            return None
        return orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = patched_find_spec
    _FASHN_IMPORT_PATCHED = True


def _ensure_runtime_importable(runtime_dir: Path) -> None:
    runtime_str = str(runtime_dir)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)
    _disable_broken_sklearn_probe()


def normalize_fashn_device(device: str | None = None) -> str:
    dev = str(device or "cuda").strip().lower()
    if dev in ("", "auto"):
        dev = "cuda"
    if dev.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return dev
            warnings.warn("FASHN requested CUDA but torch.cuda.is_available() is false; falling back to CPU.")
            return "cpu"
        except Exception:
            warnings.warn("FASHN requested CUDA but torch import failed; falling back to CPU.")
            return "cpu"
    return dev


def get_fashn_label_maps(
    *,
    runtime_dir: str | Path | None = None,
) -> tuple[dict[int, str], dict[str, int]]:
    runtime_path = resolve_fashn_runtime_dir(runtime_dir)
    cache_key = str(runtime_path)
    cached = _FASHN_LABEL_CACHE.get(cache_key, None)
    if cached is not None:
        return cached
    with _FASHN_LOCK:
        cached = _FASHN_LABEL_CACHE.get(cache_key, None)
        if cached is not None:
            return cached
        _ensure_runtime_importable(runtime_path)
        from fashn_human_parser import IDS_TO_LABELS, LABELS_TO_IDS

        ids_to_labels = {int(k): str(v) for k, v in dict(IDS_TO_LABELS).items()}
        labels_to_ids = {str(k): int(v) for k, v in dict(LABELS_TO_IDS).items()}
        cached = (ids_to_labels, labels_to_ids)
        _FASHN_LABEL_CACHE[cache_key] = cached
        return cached


def load_fashn_parser(
    *,
    model_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    device: str = "cuda",
):
    runtime_path = resolve_fashn_runtime_dir(runtime_dir)
    model_path = resolve_fashn_model_dir(model_dir)
    device_norm = normalize_fashn_device(device)
    cache_key = (str(runtime_path), str(model_path), device_norm)
    cached = _FASHN_PARSER_CACHE.get(cache_key, None)
    if cached is not None:
        return cached
    with _FASHN_LOCK:
        cached = _FASHN_PARSER_CACHE.get(cache_key, None)
        if cached is not None:
            return cached
        _ensure_runtime_importable(runtime_path)
        from fashn_human_parser import FashnHumanParser

        parser = FashnHumanParser(model_id=str(model_path), device=device_norm)
        _FASHN_PARSER_CACHE[cache_key] = parser
        return parser


def predict_fashn_segmentation(
    image_rgb: np.ndarray,
    *,
    parser=None,
    model_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    device: str = "cuda",
) -> np.ndarray:
    if parser is None:
        parser = load_fashn_parser(
            model_dir=model_dir,
            runtime_dir=runtime_dir,
            device=device,
        )
    seg = parser.predict(np.asarray(image_rgb, dtype=np.uint8))
    return np.asarray(seg, dtype=np.int32)


def build_fashn_raw_to_coarse_id(ids_to_labels: dict[int, str]) -> np.ndarray:
    max_id = max(ids_to_labels.keys()) if ids_to_labels else -1
    lut = np.full((max_id + 1,), fill_value=FASHN_COARSE_TO_ID["unknown"], dtype=np.int16)
    for raw_id, raw_name in ids_to_labels.items():
        coarse_name = FASHN_RAW_TO_COARSE.get(str(raw_name), "unknown")
        lut[int(raw_id)] = int(FASHN_COARSE_TO_ID[coarse_name])
    return lut
