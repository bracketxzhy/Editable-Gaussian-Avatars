from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameResidualEMA:
    alpha: float = 0.9
    value: float = 0.0
    initialized: bool = False

    def update(self, residual: float) -> float:
        r = float(residual)
        if not self.initialized:
            self.value = r
            self.initialized = True
            return self.value
        a = min(max(float(self.alpha), 0.0), 1.0)
        self.value = a * self.value + (1.0 - a) * r
        return self.value


def frame_weight_from_residual(residual: float, k: float) -> float:
    import math

    return float(math.exp(-float(k) * float(residual)))
