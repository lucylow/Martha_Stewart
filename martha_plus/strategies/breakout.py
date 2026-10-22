"""Donchian-style breakout strategy with an optional ATR volatility filter."""
from __future__ import annotations
from typing import Sequence
from ..indicators import atr
from ..types import Bar, Signal
from .base import require_bars

class BreakoutStrategy:
    strategy_id = "breakout"

    def __init__(self, lookback: int = 20, max_weight: float = 0.15,
                 atr_window: int = 14, max_atr_fraction: float = 0.08,
                 allow_short: bool = False):
        if lookback < 2 or atr_window < 2:
            raise ValueError("lookback and ATR window must be >= 2")
        if not 0 < max_weight <= 1 or not 0 < max_atr_fraction <= 1:
            raise ValueError("weights and volatility cap must be in (0, 1]")
        self.lookback = lookback
        self.max_weight = max_weight
        self.atr_window = atr_window
        self.max_atr_fraction = max_atr_fraction
        self.allow_short = allow_short

    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal:
        require_bars(bars, max(self.lookback, self.atr_window) + 1, self.strategy_id)
        history = bars[-self.lookback-1:-1]
        previous_high = max(bar.high for bar in history)
        previous_low = min(bar.low for bar in history)
        closes = [bar.close for bar in bars]
        atr_value = atr([bar.high for bar in bars], [bar.low for bar in bars], closes,
                        self.atr_window)[-1]
        atr_fraction = (atr_value / closes[-1]) if atr_value is not None and closes[-1] else 1.0
        if atr_fraction > self.max_atr_fraction:
            target, reason = 0.0, "ATR filter blocked entry"
        elif closes[-1] > previous_high:
            target, reason = self.max_weight, "upside breakout"
        elif self.allow_short and closes[-1] < previous_low:
            target, reason = -self.max_weight, "downside breakout"
        else:
            target, reason = 0.0, "no breakout"
        range_size = max(previous_high - previous_low, 1e-9)
        strength = min(1.0, abs(closes[-1] - (previous_high if target >= 0 else previous_low)) / range_size)
        return Signal(symbol, target, strength, f"{reason}; ATR fraction={atr_fraction:.4f}",
                      bars[-1].timestamp,
                      {"prior_high": previous_high, "prior_low": previous_low,
                       "atr_fraction": atr_fraction})
