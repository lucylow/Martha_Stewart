"""Weighted signal ensemble with strict symbol and timestamp consistency."""
from __future__ import annotations
from typing import Sequence
from ..types import Bar, Signal
from .base import Strategy

class WeightedEnsemble:
    strategy_id = "weighted_ensemble"

    def __init__(self, strategies: Sequence[tuple[Strategy, float]],
                 max_abs_weight: float = 0.25):
        if not strategies:
            raise ValueError("at least one strategy is required")
        if any(weight < 0 for _, weight in strategies):
            raise ValueError("strategy weights cannot be negative")
        if sum(weight for _, weight in strategies) <= 0:
            raise ValueError("strategy weights must have positive sum")
        if not 0 < max_abs_weight <= 1:
            raise ValueError("max_abs_weight must be in (0, 1]")
        self.strategies = tuple(strategies)
        self.max_abs_weight = max_abs_weight

    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal:
        outputs = [(strategy.generate(symbol, bars), weight)
                   for strategy, weight in self.strategies]
        total_weight = sum(weight for _, weight in outputs)
        target = sum(signal.target_weight * weight for signal, weight in outputs) / total_weight
        strength = sum(signal.strength * weight for signal, weight in outputs) / total_weight
        target = max(-self.max_abs_weight, min(self.max_abs_weight, target))
        details = "; ".join(f"{signal.reason}" for signal, _ in outputs)
        metadata: dict[str, float | str | bool] = {"component_count": len(outputs)}
        for i, (signal, _) in enumerate(outputs):
            metadata[f"component_{i}_weight"] = signal.target_weight
            metadata[f"component_{i}_reason"] = signal.reason
        return Signal(symbol, target, strength, f"ensemble: {details}", bars[-1].timestamp, metadata)
