from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "third_party" / "fashn_runtime"
MODEL_DIR = ROOT / "third_party" / "fashn-human-parser-snapshot"

EXPOSED_LABELS = {"face", "arms", "hands", "legs", "feet", "torso"}
COVERED_LABELS = {"hair", "top", "dress", "skirt", "pants", "belt", "bag", "hat", "scarf", "glasses", "jewelry"}

PALETTE = [
    0, 0, 0,
    220, 20, 60,
    255, 140, 0,
    65, 105, 225,
    186, 85, 211,
    255, 215, 0,
    46, 139, 87,
    255, 99, 71,
    112, 128, 144,
    160, 82, 45,
    72, 61, 139,
    70, 130, 180,
    154, 205, 50,
    233, 150, 122,
    255, 182, 193,
    127, 255, 212,
    210, 180, 140,
    255, 250, 205,
]


def _disable_broken_sklearn_probe() -> None:
    orig_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            return None
        return orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = patched_find_spec


def _import_parser():
    if not RUNTIME_DIR.exists():
        raise FileNotFoundError(f"Missing runtime dir: {RUNTIME_DIR}")
    sys.path.insert(0, str(RUNTIME_DIR))
    _disable_broken_sklearn_probe()
    from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS, LABELS_TO_IDS

    return FashnHumanParser, IDS_TO_LABELS, LABELS_TO_IDS


def _iter_images(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    if not inp.is_dir():
        raise FileNotFoundError(f"Input path not found: {inp}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _save_palette_mask(path: Path, seg: np.ndarray) -> None:
    img = Image.fromarray(seg.astype(np.uint8), mode="P")
    palette = PALETTE + [0] * max(0, 768 - len(PALETTE))
    img.putpalette(palette)
    img.save(path)


def _overlay(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    palette = np.asarray(PALETTE, dtype=np.uint8).reshape(-1, 3)
    color = palette[seg.clip(0, len(palette) - 1)]
    out = (0.45 * rgb.astype(np.float32) + 0.55 * color.astype(np.float32)).clip(0, 255)
    return out.astype(np.uint8)


def _mask_from_labels(seg: np.ndarray, labels_to_ids: dict[str, int], names: set[str]) -> np.ndarray:
    ids = [int(labels_to_ids[name]) for name in sorted(names) if name in labels_to_ids]
    if not ids:
        return np.zeros_like(seg, dtype=np.uint8)
    return (np.isin(seg, np.asarray(ids, dtype=np.uint8))).astype(np.uint8) * 255


def _stats(seg: np.ndarray, ids_to_labels: dict[int, str]) -> dict[str, object]:
    vals, counts = np.unique(seg, return_counts=True)
    by_label: dict[str, int] = {}
    for v, c in zip(vals.tolist(), counts.tolist()):
        by_label[str(ids_to_labels.get(int(v), f"id_{int(v)}"))] = int(c)
    return {
        "height": int(seg.shape[0]),
        "width": int(seg.shape[1]),
        "labels": by_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local FASHN human parser and export exposed/covered masks.")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory of images.")
    parser.add_argument("--out_dir", type=str, default="outputs/fashn_parser", help="Output directory.")
    parser.add_argument("--model_dir", type=str, default=str(MODEL_DIR), help="Local FASHN model snapshot directory.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu or cuda.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing local model snapshot: {model_dir}")

    FashnHumanParser, ids_to_labels, labels_to_ids = _import_parser()
    runner = FashnHumanParser(model_id=str(model_dir), device=str(args.device))

    images = _iter_images(input_path)
    if not images:
        raise RuntimeError(f"No images found under: {input_path}")

    for image_path in images:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        seg = runner.predict(rgb)
        exposed = _mask_from_labels(seg, labels_to_ids, EXPOSED_LABELS)
        covered = _mask_from_labels(seg, labels_to_ids, COVERED_LABELS)
        overlay = _overlay(rgb, seg)

        stem_dir = out_dir / image_path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)

        _save_palette_mask(stem_dir / "parsing_palette.png", seg)
        Image.fromarray(seg.astype(np.uint8), mode="L").save(stem_dir / "parsing_ids.png")
        Image.fromarray(exposed, mode="L").save(stem_dir / "exposed_mask.png")
        Image.fromarray(covered, mode="L").save(stem_dir / "covered_mask.png")
        Image.fromarray(overlay, mode="RGB").save(stem_dir / "overlay.png")
        with (stem_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(_stats(seg, ids_to_labels), f, indent=2)

        print(f"Saved: {stem_dir}")


if __name__ == "__main__":
    main()
