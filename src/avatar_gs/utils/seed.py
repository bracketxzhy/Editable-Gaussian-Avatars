from __future__ import annotations
import os
import random
import numpy as np

try:
    import torch
except Exception:
    torch = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 让结果更可复现（会影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
