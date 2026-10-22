"""Deterministic portfolio target allocation helpers."""
from __future__ import annotations
from math import isfinite
from typing import Mapping


def normalize_weights(weights: Mapping[str, float], gross_limit: float = 1.0) -> dict[str, float]:
    if not 0 < gross_limit <= 1:
        raise ValueError("gross_limit must be in (0, 1]")
    if any(not isfinite(v) for v in weights.values()):
        raise ValueError("weights must be finite")
    gross = sum(abs(v) for v in weights.values())
    scale = min(1.0, gross_limit / gross) if gross else 1.0
    return {symbol.strip().upper(): weight * scale for symbol, weight in weights.items()}


def equal_weight_targets(symbols: list[str] | tuple[str, ...], gross_limit: float = 1.0,
                         long_only: bool = True) -> dict[str, float]:
    unique = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not unique:
        return {}
    base = gross_limit / len(unique)
    return {symbol: base for symbol in unique} if long_only else {
        symbol: (base if i % 2 == 0 else -base) for i, symbol in enumerate(unique)
    }


def inverse_volatility_targets(volatilities: Mapping[str, float], gross_limit: float = 1.0,
                               min_volatility: float = 1e-4) -> dict[str, float]:
    if min_volatility <= 0:
        raise ValueError("min_volatility must be positive")
    scores = {symbol.strip().upper(): 1.0 / max(abs(float(vol)), min_volatility)
              for symbol, vol in volatilities.items()}
    total = sum(scores.values())
    if total == 0:
        return {}
    return {symbol: gross_limit * score / total for symbol, score in scores.items()}
