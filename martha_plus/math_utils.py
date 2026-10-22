"""Numerically defensive helpers. All inputs are interpreted as ordinary floats."""
from __future__ import annotations
from math import isfinite, sqrt
from statistics import mean, median
from typing import Iterable, Sequence


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isfinite(float(v))]


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def variance(values: Sequence[float], sample: bool = True) -> float:
    if len(values) < (2 if sample else 1):
        return 0.0
    center = mean(values)
    denom = len(values) - 1 if sample else len(values)
    return sum((x - center) ** 2 for x in values) / denom


def stddev(values: Sequence[float], sample: bool = True) -> float:
    return sqrt(max(0.0, variance(values, sample)))


def covariance(x: Sequence[float], y: Sequence[float], sample: bool = True) -> float:
    if len(x) != len(y):
        raise ValueError("covariance inputs must have equal length")
    minimum = 2 if sample else 1
    if len(x) < minimum:
        return 0.0
    mx, my = mean(x), mean(y)
    denom = len(x) - 1 if sample else len(x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    sx, sy = stddev(x), stddev(y)
    if sx == 0 or sy == 0:
        return 0.0
    return max(-1.0, min(1.0, covariance(x, y) / (sx * sy)))


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lo, hi = int(position), min(len(ordered) - 1, int(position) + 1)
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def compound_returns(returns: Iterable[float]) -> float:
    wealth = 1.0
    for value in returns:
        wealth *= 1.0 + value
    return wealth - 1.0


def max_drawdown(equity_curve: Sequence[float]) -> tuple[float, int, int]:
    """Return (positive drawdown fraction, peak index, trough index)."""
    if not equity_curve:
        return 0.0, -1, -1
    peak = float(equity_curve[0])
    peak_index = 0
    worst = 0.0
    worst_peak, worst_trough = 0, 0
    for index, raw in enumerate(equity_curve):
        value = float(raw)
        if value > peak:
            peak, peak_index = value, index
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > worst:
            worst, worst_peak, worst_trough = drawdown, peak_index, index
    return worst, worst_peak, worst_trough


def annualized_volatility(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return stddev(returns) * sqrt(periods_per_year)


def sharpe_ratio(returns: Sequence[float], risk_free_rate_per_period: float = 0.0,
                 periods_per_year: float = 252.0) -> float:
    if not returns:
        return 0.0
    excess = [r - risk_free_rate_per_period for r in returns]
    deviation = stddev(excess)
    if deviation == 0:
        return 0.0
    return mean(excess) / deviation * sqrt(periods_per_year)


def robust_zscore(value: float, sample: Sequence[float]) -> float:
    """Median/MAD z-score, resilient to a few extreme observations."""
    if len(sample) < 3:
        return 0.0
    center = median(sample)
    mad = median([abs(x - center) for x in sample])
    if mad == 0:
        deviation = stddev(sample)
        return (value - center) / deviation if deviation else 0.0
    return 0.67448975 * (value - center) / mad
