from __future__ import annotations

import os
from pathlib import Path


def _default_motion_prior_data_dir() -> str:
    env_dir_new = os.environ.get("AVATAR_GS_MOTION_PRIOR_DATA_DIR", "").strip()
    if env_dir_new:
        return env_dir_new
    env_dir_old = os.environ.get("AVATAR_GS_VIBE_DATA_DIR", "").strip()
    if env_dir_old:
        return env_dir_old
    new_dir = Path("data") / "motion_prior_data"
    return str(new_dir)


MOTION_PRIOR_DATA_DIR = _default_motion_prior_data_dir()
