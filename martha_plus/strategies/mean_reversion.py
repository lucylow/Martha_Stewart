"""Mean-reversion strategy. Long-only by default to avoid implicit short risk."""
from __future__ import annotations
from statistics import mean
from typing import Sequence
from ..indicators import rolling_std
from ..types import Bar, Signal
from .base import require_bars

class MeanReversionStrategy:
    def __init__(self, window: int = 30, entry_z: float = 2.0, exit_z: float = 0.5,
                 max_weight: float = 0.15, allow_short: bool = False,
                 strategy_id: str = "mean_reversion"):
        if window < 3 or entry_z <= exit_z or exit_z < 0:
            raise ValueError("require window >= 3 and entry_z > exit_z >= 0")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z
        self.max_weight, self.allow_short = max_weight, allow_short
        self.strategy_id = strategy_id

    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal:
        require_bars(bars, self.window, self.strategy_id)
        closes = [b.close for b in bars[-self.window:]]
        center = mean(closes)
        deviation_series = rolling_std(closes, self.window)
        deviation = deviation_series[-1] or 0.0
        z = (closes[-1] - center) / deviation if deviation > 0 else 0.0
        if z <= -self.entry_z:
            target, reason = self.max_weight, "price below lower reversion band"
        elif self.allow_short and z >= self.entry_z:
            target, reason = -self.max_weight, "price above upper reversion band"
        elif abs(z) <= self.exit_z:
            target, reason = 0.0, "inside exit band"
        else:
            target, reason = 0.0, "no entry: z-score below threshold"
        strength = min(1.0, abs(z) / max(self.entry_z, 1e-9))
        return Signal(symbol, target, strength, f"{reason}; z={z:.3f}", bars[-1].timestamp,
                      {"zscore": z, "mean": center, "std": deviation})
