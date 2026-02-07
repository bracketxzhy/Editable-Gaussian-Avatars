from __future__ import annotations

from pathlib import Path
import re


def _has_images(folder: Path) -> bool:
    return any(folder.glob("*.jpg")) or any(folder.glob("*.png"))


_PAT_TRIAL = re.compile(r"^(S\d+)_(.+?)_(\d+)\.(\d+)_(\d+)\.(jpg|png)$", re.IGNORECASE)
_PAT_NO_TRIAL = re.compile(r"^(S\d+)_(.+?)\.(\d+)_(\d+)\.(jpg|png)$", re.IGNORECASE)


def parse_h36m_image_name(name: str) -> dict | None:
    """
    Support common flat Human3.6M export naming, e.g.
      - S1_Directions_1.54138969_000001.jpg  (with trial index)
      - S1_Directions.54138969_000001.jpg    (no trial index)

    Returns dict(subject, action, trial, camera, frame).
    """
    n = str(name)
    m = _PAT_TRIAL.match(n)
    if m:
        subj, action, trial, cam, fr, ext = m.groups()
        return {
            "subject": subj,
            "action": action,
            "trial": int(trial),
            "camera": str(cam),
            "frame": int(fr),
            "ext": ext.lower(),
        }
    m = _PAT_NO_TRIAL.match(n)
    if m:
        subj, action, cam, fr, ext = m.groups()
        return {
            "subject": subj,
            "action": action,
            "trial": None,
            "camera": str(cam),
            "frame": int(fr),
            "ext": ext.lower(),
        }
    return None


def resolve_h36m_image_paths(project_root: Path, data_cfg: dict) -> list[Path]:
    """
    Resolve a list of Human3.6M images from a flat folder (multi-view images mixed in one dir),
    optionally using an image list txt file.

    Expected config:
      data:
        h36m:
          root_dir: data/frames/input/human3.6/S1
          images_list: data/frames/input/human3.6/h36m_annot/train_images.txt  # optional
          subject: S1               # optional
          action: Directions        # optional
          trial: 1                  # optional
          cameras: [54138969, ...]  # optional
          stride: 1                 # optional
          max_images: -1            # optional
    """
    hcfg = data_cfg.get("h36m", {})
    root_dir = Path(hcfg.get("root_dir", "data/human36m"))
    if not root_dir.is_absolute():
        root_dir = (project_root / root_dir).resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"h36m.root_dir not found: {root_dir}")

    images_list = hcfg.get("images_list", None)
    list_path = None
    if images_list is not None and str(images_list).strip():
        list_path = Path(str(images_list))
        if not list_path.is_absolute():
            list_path = (project_root / list_path).resolve()
        if not list_path.exists():
            raise FileNotFoundError(f"h36m.images_list not found: {list_path}")

    subj_f = str(hcfg.get("subject", "")).strip() or None
    act_f = str(hcfg.get("action", "")).strip() or None
    trial_f = hcfg.get("trial", None)
    trial_f = int(trial_f) if trial_f is not None and str(trial_f).strip() != "" else None

    cams = hcfg.get("cameras", None)
    cam_set = None
    if cams is not None:
        cam_set = {str(c) for c in cams}

    stride = int(hcfg.get("stride", 1))
    max_images = int(hcfg.get("max_images", -1))

    def _ok(meta: dict | None) -> bool:
        if meta is None:
            return False
        if subj_f is not None and meta["subject"] != subj_f:
            return False
        if act_f is not None and meta["action"] != act_f:
            return False
        if trial_f is not None and meta["trial"] != trial_f:
            return False
        if cam_set is not None and meta["camera"] not in cam_set:
            return False
        return True

    paths: list[Path] = []
    if list_path is not None:
        for line in list_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            # support list entries with/without path components
            candidate = root_dir / s
            if not candidate.exists():
                # fallback: sometimes the list uses slightly different naming; try basename
                candidate = root_dir / Path(s).name
            if not candidate.exists():
                continue
            meta = parse_h36m_image_name(candidate.name)
            if _ok(meta):
                paths.append(candidate)
    else:
        for p in root_dir.glob("*.jpg"):
            meta = parse_h36m_image_name(p.name)
            if _ok(meta):
                paths.append(p)
        for p in root_dir.glob("*.png"):
            meta = parse_h36m_image_name(p.name)
            if _ok(meta):
                paths.append(p)

    # Sort to keep temporal grouping stable: (frame, camera, trial)
    def _sort_key(p: Path):
        meta = parse_h36m_image_name(p.name) or {}
        return (
            int(meta.get("frame", 0)),
            str(meta.get("camera", "")),
            -1 if meta.get("trial", None) is None else int(meta.get("trial")),
            p.name,
        )

    paths = sorted(paths, key=_sort_key)
    if stride > 1:
        paths = paths[::stride]
    if max_images > 0:
        paths = paths[:max_images]
    if len(paths) == 0:
        raise FileNotFoundError(
            "No Human3.6M images selected. "
            f"root_dir={root_dir}, subject={subj_f}, action={act_f}, trial={trial_f}, cameras={cams}, list={list_path}"
        )
    return paths


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
