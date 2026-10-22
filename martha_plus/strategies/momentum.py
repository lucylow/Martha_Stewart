"""Long-only momentum strategy using fast/slow moving-average spread."""
from __future__ import annotations
from typing import Sequence
from ..indicators import ema
from ..types import Bar, Signal
from .base import require_bars

class MomentumStrategy:
    def __init__(self, fast_window: int = 8, slow_window: int = 32,
                 max_weight: float = 0.20, min_edge_bps: float = 2.0,
                 strategy_id: str = "momentum"):
        if fast_window < 2 or slow_window <= fast_window:
            raise ValueError("windows must satisfy 2 <= fast_window < slow_window")
        if not 0 < max_weight <= 1 or min_edge_bps < 0:
            raise ValueError("invalid weight or edge")
        self.fast_window, self.slow_window = fast_window, slow_window
        self.max_weight, self.min_edge_bps = max_weight, min_edge_bps
        self.strategy_id = strategy_id

    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal:
        require_bars(bars, self.slow_window + 1, self.strategy_id)
        closes = [b.close for b in bars]
        fast = ema(closes, self.fast_window)[-1]
        slow = ema(closes, self.slow_window)[-1]
        assert fast is not None and slow is not None
        edge_bps = (fast / slow - 1.0) * 10_000
        direction = 1.0 if edge_bps > self.min_edge_bps else 0.0
        strength = min(1.0, abs(edge_bps) / 100.0)
        target = self.max_weight * direction * strength
        return Signal(symbol, target, strength,
                      f"EMA({self.fast_window}) vs EMA({self.slow_window}); edge={edge_bps:.2f}bps",
                      timestamp=bars[-1].timestamp,
                      metadata={"fast_ema": fast, "slow_ema": slow, "edge_bps": edge_bps})
