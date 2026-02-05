from __future__ import annotations

from pathlib import Path


def _has_images(folder: Path) -> bool:
    return any(folder.glob("*.jpg")) or any(folder.glob("*.png"))


def resolve_h36m_frames_dir(project_root: Path, data_cfg: dict) -> Path:
    """
    Resolve Human3.6M image folder for a single fixed-camera sequence.

    Expected default layout:
      data/human36m/<subject>/<action>/<camera>/*.jpg
    """
    hcfg = data_cfg.get("h36m", {})
    root_dir = Path(hcfg.get("root_dir", "data/human36m"))
    if not root_dir.is_absolute():
        root_dir = (project_root / root_dir).resolve()

    subject = str(hcfg.get("subject", "S1"))
    action = str(hcfg.get("action", "Directions"))
    camera = str(hcfg.get("camera", "54138969"))

    candidates = [
        root_dir / subject / action / camera,
        root_dir / subject / action / "Images" / camera,
        root_dir / subject / action,
    ]

    for p in candidates:
        if p.exists() and p.is_dir() and _has_images(p):
            return p

    expected = root_dir / subject / action / camera
    raise FileNotFoundError(
        "Cannot find Human3.6M frames directory. "
        f"Tried under {expected}. Please place frames as *.jpg/*.png in one of: {candidates}"
    )
