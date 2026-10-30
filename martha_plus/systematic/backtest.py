"""Minimal systematic signal simulation with explicit timing and costs.

Signals observed at close t are applied to return t+1. Transaction costs are
charged on absolute position changes. This is an educational gross-to-net
accounting primitive, not a fill model or investment performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from ..quant_research.performance.metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Period-aligned output for a lagged single-asset strategy."""

    positions: tuple[float, ...]
    gross_returns: tuple[float, ...]
    turnover: tuple[float, ...]
    costs: tuple[float, ...]
    net_returns: tuple[float, ...]
    total_turnover: float
    total_cost: float
    sharpe: float
    drawdown: dict


def lagged_signal_backtest(
    signals: Sequence[float],
    asset_returns: Sequence[float],
    *,
    cost_bps: float = 0.0,
    exposure_limit: float = 1.0,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """Evaluate one asset with a one-observation signal delay.

    ``signals[t]`` is the desired signed exposure known at the end of period t;
    ``positions[t]`` therefore equals the bounded signal from t-1, and is zero
    for the first return. Costs are ``abs(position change) * cost_bps / 10_000``.
    Inputs are assumed to be aligned, survivorship-correct total returns; this
    function does not correct corporate actions or simulate order-book fills.
    """
    if len(signals) != len(asset_returns):
        raise ValueError("signals and asset_returns must have identical lengths")
    if isinstance(cost_bps, bool) or not isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if isinstance(exposure_limit, bool) or not isfinite(exposure_limit) or exposure_limit <= 0:
        raise ValueError("exposure_limit must be finite and positive")
    if isinstance(periods_per_year, bool) or not isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    values = tuple(float(value) for value in signals)
    returns = tuple(float(value) for value in asset_returns)
    if any(not isfinite(value) for value in values):
        raise ValueError("signals must be finite")
    if any(not isfinite(value) or value <= -1.0 for value in returns):
        raise ValueError("asset returns must be finite and greater than -100%")
    positions: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    gross: list[float] = []
    net: list[float] = []
    prior = 0.0
    rate = cost_bps / 10_000.0
    for i, asset_return in enumerate(returns):
        position = 0.0 if i == 0 else max(-exposure_limit, min(exposure_limit, values[i - 1]))
        traded = abs(position - prior)
        cost = traded * rate
        gross_return = position * asset_return
        net_return = gross_return - cost
        if net_return <= -1.0:
            raise ValueError("net strategy return cannot be at or below -100%; model insolvency explicitly")
        positions.append(position)
        turnover.append(traded)
        costs.append(cost)
        gross.append(gross_return)
        net.append(net_return)
        prior = position
    sharpe = sharpe_ratio(net, periods_per_year=periods_per_year) if len(net) >= 2 else 0.0
    drawdown = max_drawdown(net) if net else {"max_drawdown": 0.0, "peak_index": None, "trough_index": None}
    return BacktestResult(
        tuple(positions), tuple(gross), tuple(turnover), tuple(costs), tuple(net),
        sum(turnover), sum(costs), sharpe, drawdown,
    )
