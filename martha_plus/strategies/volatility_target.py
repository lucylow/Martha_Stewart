"""Volatility-managed trend strategy. Uses only lagged closes for signal computation."""
from __future__ import annotations
from statistics import mean
from typing import Sequence
from ..indicators import returns, ewma_volatility
from ..types import Bar, Signal
from .base import require_bars

class VolatilityTargetStrategy:
    strategy_id = "volatility_target"

    def __init__(self, momentum_window: int = 20, vol_window: int = 20,
                 target_daily_vol: float = 0.01, max_weight: float = 0.25,
                 allow_short: bool = False):
        if momentum_window < 2 or vol_window < 2 or target_daily_vol <= 0:
            raise ValueError("windows and target volatility must be positive")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.target_daily_vol = target_daily_vol
        self.max_weight = max_weight
        self.allow_short = allow_short

    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal:
        require_bars(bars, max(self.momentum_window, self.vol_window) + 2, self.strategy_id)
        closes = [bar.close for bar in bars]
        momentum = closes[-1] / closes[-self.momentum_window-1] - 1.0
        volatility_series = ewma_volatility(returns(closes[-self.vol_window-1:]), decay=0.94)
        volatility = mean(volatility_series[-self.vol_window:]) if volatility_series else 0.0
        direction = 1.0 if momentum > 0 else (-1.0 if self.allow_short and momentum < 0 else 0.0)
        scale = min(1.0, self.target_daily_vol / volatility) if volatility > 0 else 1.0
        target = direction * self.max_weight * scale
        strength = min(1.0, abs(momentum) / 0.05) * (scale if direction else 0.0)
        return Signal(symbol, target, strength,
                      f"momentum={momentum:.4f}; EWMA vol={volatility:.4f}; scale={scale:.3f}",
                      bars[-1].timestamp,
                      {"momentum": momentum, "ewma_volatility": volatility, "volatility_scale": scale})
