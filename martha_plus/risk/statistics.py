"""Historical loss-tail metrics for research reports.

These are sample statistics only: they are not forward-looking forecasts and can
understate risk when the sample is short, nonstationary, or missing tail events.
"""
from __future__ import annotations
from statistics import mean
from typing import Sequence
from ..math_utils import percentile, stddev


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Return a positive loss fraction at the selected historical confidence level."""
    _validate(returns, confidence)
    quantile_return = percentile(returns, (1.0 - confidence) * 100.0)
    return max(0.0, -quantile_return)


def expected_shortfall(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Return average loss among observations at/below the lower-tail quantile."""
    _validate(returns, confidence)
    cutoff = percentile(returns, (1.0 - confidence) * 100.0)
    tail = [value for value in returns if value <= cutoff]
    return max(0.0, -mean(tail)) if tail else 0.0


def downside_deviation(returns: Sequence[float], target: float = 0.0,
                       periods_per_year: float | None = None) -> float:
    if not returns:
        return 0.0
    deviations = [min(0.0, value - target) ** 2 for value in returns]
    result = (sum(deviations) / len(deviations)) ** 0.5
    return result * (periods_per_year ** 0.5) if periods_per_year else result


def omega_ratio(returns: Sequence[float], threshold: float = 0.0) -> float:
    """Sum gains above threshold divided by absolute shortfall below threshold."""
    gains = sum(max(0.0, value - threshold) for value in returns)
    losses = sum(max(0.0, threshold - value) for value in returns)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def gain_to_pain_ratio(returns: Sequence[float]) -> float:
    pain = sum(abs(value) for value in returns if value < 0)
    gains = sum(value for value in returns if value > 0)
    if pain == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / pain


def tail_report(returns: Sequence[float], confidence: float = 0.95) -> dict[str, float | int]:
    _validate(returns, confidence)
    return {
        "observations": len(returns),
        "historical_var": historical_var(returns, confidence),
        "expected_shortfall": expected_shortfall(returns, confidence),
        "downside_deviation": downside_deviation(returns),
        "omega_ratio": omega_ratio(returns),
        "gain_to_pain_ratio": gain_to_pain_ratio(returns),
        "sample_volatility": stddev(returns),
    }


def _validate(returns: Sequence[float], confidence: float) -> None:
    if not returns:
        raise ValueError("at least one return is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
