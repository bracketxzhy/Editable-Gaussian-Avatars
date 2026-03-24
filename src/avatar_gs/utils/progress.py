from __future__ import annotations

from typing import Any

from tqdm import tqdm as _tqdm
from tqdm import trange as _trange


def tqdm(*args: Any, **kwargs: Any):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return _tqdm(*args, **kwargs)


def trange(*args: Any, **kwargs: Any):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return _trange(*args, **kwargs)
