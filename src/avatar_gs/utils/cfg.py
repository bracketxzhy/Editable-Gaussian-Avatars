from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def resolve_path(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())


def parse_args_base(desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--override", type=str, nargs="*", default=None,
                        help="Optional overrides as key=value, e.g. video.path=data/videos/a.mp4")
    return parser


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list[str]]) -> Dict[str, Any]:
    if not overrides:
        return cfg

    def set_by_dotted_key(d: Dict[str, Any], dotted: str, value: Any):
        keys = dotted.split(".")
        cur = d
        for kk in keys[:-1]:
            if kk not in cur or not isinstance(cur[kk], dict):
                cur[kk] = {}
            cur = cur[kk]
        cur[keys[-1]] = value

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Use key=value.")
        k, v = item.split("=", 1)
        # 简单类型推断
        vv: Any = v
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except ValueError:
                vv = v
        set_by_dotted_key(cfg, k, vv)
    return cfg


def save_yaml(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
