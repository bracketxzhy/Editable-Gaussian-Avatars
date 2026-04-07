from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "evaluation" / "stage1_eval" / "run_stage1_eval.py"
    if not target.exists():
        raise FileNotFoundError(f"Moved script not found: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
