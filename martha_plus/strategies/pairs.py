"""Research-only two-asset spread z-score strategy.

This produces a signal for one leg at a time and is not a complete hedged-pair
execution system. The caller must coordinate both legs and reject unbalanced risk.
"""
from __future__ import annotations
from statistics import mean
from typing import Sequence
from ..math_utils import stddev
from ..types import Bar, Signal

class PairsZScoreStrategy:
    strategy_id = "pairs_zscore"

    def __init__(self, lookback: int = 60, entry_z: float = 2.0,
                 max_weight: float = 0.10, allow_short: bool = False):
        if lookback < 10 or entry_z <= 0 or not 0 < max_weight <= 1:
            raise ValueError("invalid pairs strategy settings")
        self.lookback, self.entry_z = lookback, entry_z
        self.max_weight, self.allow_short = max_weight, allow_short

    def spread_zscore(self, left: Sequence[Bar], right: Sequence[Bar]) -> float:
        if len(left) < self.lookback or len(right) < self.lookback:
            raise ValueError("insufficient bars for pairs strategy")
        left = left[-self.lookback:]
        right = right[-self.lookback:]
        if [b.timestamp for b in left] != [b.timestamp for b in right]:
            raise ValueError("pair bars must be timestamp-aligned")
        spread = [a.close - b.close for a, b in zip(left, right)]
        deviation = stddev(spread)
        return (spread[-1] - mean(spread)) / deviation if deviation else 0.0

    def generate(self, symbol: str, left: Sequence[Bar], right: Sequence[Bar]) -> Signal:
        z = self.spread_zscore(left, right)
        if z <= -self.entry_z:
            target = self.max_weight
            reason = "left leg cheap versus paired leg; coordinate hedge before trading"
        elif self.allow_short and z >= self.entry_z:
            target = -self.max_weight
            reason = "left leg rich versus paired leg; coordinate hedge before trading"
        else:
            target, reason = 0.0, "no paired-spread entry"
        return Signal(symbol, target, min(1.0, abs(z) / self.entry_z),
                      f"{reason}; spread z={z:.3f}", left[-1].timestamp,
                      {"spread_zscore": z, "paired_symbol": right[-1].symbol})
