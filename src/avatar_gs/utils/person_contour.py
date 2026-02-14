from __future__ import annotations

from pathlib import Path
import sys


# Compatibility shim: real evaluation implementation moved to
# evaluation/stage1_eval/person_contour.py.
_repo_root = Path(__file__).resolve().parents[3]
_eval_dir = _repo_root / "evaluation" / "stage1_eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from person_contour import *  # type: ignore # noqa: F401,F403
