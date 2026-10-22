"""Benchmark-relative analytics for portfolio and strategy return streams."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean
from typing import Sequence


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    observations: int
    total_return: float
    benchmark_return: float
    active_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha_per_period: float
    r_squared: float
    up_capture: float
    down_capture: float
    correlation: float
    maximum_relative_drawdown: float


def _validate_pair(portfolio: Sequence[float], benchmark: Sequence[float]) -> None:
    if len(portfolio) != len(benchmark) or len(portfolio) < 2:
        raise ValueError("portfolio and benchmark returns must have equal lengths >= 2")
    if any(not isfinite(v) or v < -1 for v in list(portfolio) + list(benchmark)):
        raise ValueError("returns must be finite and no less than -100%")


def _compound(values: Sequence[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result - 1.0


def benchmark_metrics(portfolio: Sequence[float], benchmark: Sequence[float], *,
                      periods_per_year: float = 252.0) -> BenchmarkMetrics:
    _validate_pair(portfolio, benchmark)
    if periods_per_year <= 0 or not isfinite(periods_per_year):
        raise ValueError("periods_per_year must be positive")
    p, b = list(portfolio), list(benchmark)
    active = [x - y for x, y in zip(p, b)]
    avg_active = mean(active)
    active_var = sum((x - avg_active) ** 2 for x in active) / (len(active) - 1)
    tracking = sqrt(max(0.0, active_var) * periods_per_year)
    info = (avg_active * periods_per_year) / tracking if tracking else 0.0
    p_mean, b_mean = mean(p), mean(b)
    var_b = sum((x - b_mean) ** 2 for x in b) / (len(b) - 1)
    cov = sum((x - p_mean) * (y - b_mean) for x, y in zip(p, b)) / (len(p) - 1)
    beta = cov / var_b if var_b > 0 else 0.0
    alpha = p_mean - beta * b_mean
    var_p = sum((x - p_mean) ** 2 for x in p) / (len(p) - 1)
    correlation = cov / sqrt(var_b * var_p) if var_b > 0 and var_p > 0 else 0.0
    r_squared = min(1.0, max(0.0, correlation * correlation))
    up_idxs = [i for i, val in enumerate(b) if val > 0]
    down_idxs = [i for i, val in enumerate(b) if val < 0]
    up_capture = (mean([p[i] for i in up_idxs]) / mean([b[i] for i in up_idxs])
                  if up_idxs and mean([b[i] for i in up_idxs]) else 0.0)
    down_capture = (mean([p[i] for i in down_idxs]) / mean([b[i] for i in down_idxs])
                    if down_idxs and mean([b[i] for i in down_idxs]) else 0.0)
    rel_equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pr, br in zip(p, b):
        rel_equity *= (1 + pr) / (1 + br) if br > -1 else 1.0
        peak = max(peak, rel_equity)
        if peak:
            max_dd = min(max_dd, rel_equity / peak - 1.0)
    return BenchmarkMetrics(len(p), _compound(p), _compound(b), _compound(p) - _compound(b), tracking,
        info, beta, alpha, r_squared, up_capture, down_capture, correlation, abs(max_dd))


def rolling_active_return(portfolio: Sequence[float], benchmark: Sequence[float], *,
                           window: int, annualize: bool = False, periods_per_year: float = 252.0) -> list[float]:
    _validate_pair(portfolio, benchmark)
    if window < 1 or window > len(portfolio):
        raise ValueError("window must fit return series")
    output = []
    for start in range(len(portfolio) - window + 1):
        end = start + window
        value = _compound(portfolio[start:end]) - _compound(benchmark[start:end])
        if annualize:
            if periods_per_year <= 0:
                raise ValueError("periods_per_year must be positive")
            if value <= -1:
                output.append(-1.0)
            else:
                output.append((1 + value) ** (periods_per_year / window) - 1)
        else:
            output.append(value)
    return output


def rolling_beta(portfolio: Sequence[float], benchmark: Sequence[float], *, window: int) -> list[float]:
    _validate_pair(portfolio, benchmark)
    if window < 2 or window > len(portfolio):
        raise ValueError("window must be in [2, sample length]")
    values = []
    for start in range(len(portfolio) - window + 1):
        p = portfolio[start:start + window]
        b = benchmark[start:start + window]
        pb, bb = mean(p), mean(b)
        var_b = sum((x - bb) ** 2 for x in b)
        cov = sum((x - pb) * (y - bb) for x, y in zip(p, b))
        values.append(cov / var_b if var_b else 0.0)
    return values


def active_return_contributions(portfolio_returns: Sequence[float],
                                benchmark_returns: Sequence[float], *,
                                weights: Sequence[float] | None = None) -> dict[str, float | int]:
    _validate_pair(portfolio_returns, benchmark_returns)
    if weights is None:
        weights = [1.0] * len(portfolio_returns)
    if len(weights) != len(portfolio_returns) or any(not isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must be finite, non-negative, and match observations")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("at least one weight must be positive")
    active = [(p - b) * w / total_weight for p, b, w in zip(portfolio_returns, benchmark_returns, weights)]
    positive = sum(x for x in active if x > 0)
    negative = sum(x for x in active if x < 0)
    return {"observations": len(active), "weighted_active_return": sum(active),
            "positive_contribution": positive, "negative_contribution": negative,
            "best_period_contribution": max(active), "worst_period_contribution": min(active)}
