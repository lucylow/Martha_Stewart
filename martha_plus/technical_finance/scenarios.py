"""Portfolio scenario engines and reverse-stress diagnostics."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .core import FinanceInputError, Matrix, Number, finite, positive, vector
from .portfolio import portfolio_variance


def factor_scenario(exposures: Mapping[str, Number],
                    shocks: Mapping[str, Number],
                    *, nav: Number = 1.0) -> dict[str, float]:
    """Linear factor P&L scenario with normalized NAV impact."""
    if set(exposures) != set(shocks):
        raise FinanceInputError("exposure and shock factors must match")
    n = finite(nav)
    if n <= 0:
        raise FinanceInputError("nav must be positive")
    contributions = {key: finite(exposures[key])*finite(shocks[key]) for key in exposures}
    total = sum(contributions.values())
    contributions["total_pnl"] = total
    contributions["return_on_nav"] = total/n
    return contributions


def covariance_stress(returns: Sequence[Vector], *, volatility_multiplier: Number = 1.5,
                      correlation_multiplier: Number = 1.0) -> tuple[tuple[float, ...], ...]:
    """Construct a stress covariance by scaling vols and off-diagonal correlations."""
    from .portfolio import covariance_matrix
    cov = covariance_matrix(returns)
    n = len(cov)
    vol_scale, corr_scale = finite(volatility_multiplier), finite(correlation_multiplier)
    if vol_scale <= 0 or corr_scale < 0:
        raise FinanceInputError("volatility multiplier must be positive; correlation multiplier non-negative")
    std = [math.sqrt(cov[i][i]) for i in range(n)]
    result = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                result[i][j] = cov[i][i]*vol_scale**2
            else:
                corr = cov[i][j]/(std[i]*std[j]) if std[i]*std[j] else 0.0
                stressed_corr = max(-1.0, min(1.0, corr*corr_scale))
                result[i][j] = stressed_corr*std[i]*vol_scale*std[j]*vol_scale
    return tuple(tuple(row) for row in result)


def scenario_matrix_pnl(weights: Vector, scenarios: Mapping[str, Vector],
                         *, nav: Number = 1.0) -> dict[str, float]:
    """Linear factor/asset scenario P&L given named shock vectors."""
    w = vector(weights)
    value = finite(nav)
    if value <= 0:
        raise FinanceInputError("nav must be positive")
    result = {}
    for name, raw in scenarios.items():
        shock = vector(raw)
        if len(shock) != len(w):
            raise FinanceInputError(f"scenario {name} dimension mismatch")
        result[name] = value*sum(a*b for a, b in zip(w, shock))
    return result


def reverse_stress_required_move(exposure: Number, loss_limit: Number) -> float:
    """Absolute factor shock required to breach a loss budget under linear exposure."""
    delta, loss = finite(exposure), positive(loss_limit)
    if delta == 0:
        raise FinanceInputError("zero exposure cannot breach the loss limit")
    return loss/abs(delta)


def reverse_stress_one_factor(exposures: Mapping[str, Number],
                              loss_limit: Number) -> dict[str, float]:
    """Compute individual one-factor shocks required to reach a loss threshold."""
    return {factor: reverse_stress_required_move(exposure, loss_limit)
            for factor, exposure in exposures.items()}


def stress_loss_distribution(scenario_pnls: Mapping[str, Number]) -> dict[str, float]:
    """Summarize a finite scenario set without assigning scenario probabilities."""
    values = [finite(v) for v in scenario_pnls.values()]
    if not values:
        raise FinanceInputError("scenario_pnls cannot be empty")
    return {"worst_pnl": min(values), "best_pnl": max(values),
            "mean_pnl": sum(values)/len(values),
            "worst_loss": max(0.0, -min(values)),
            "scenario_count": float(len(values))}


def correlation_breakdown(correlation: Matrix, stress_level: Number = 1.0) -> tuple[tuple[float, ...], ...]:
    """Move off-diagonal correlations toward +1 by a convex stress factor."""
    values = tuple(tuple(finite(x) for x in row) for row in correlation)
    n = len(values)
    if n == 0 or any(len(row) != n for row in values):
        raise FinanceInputError("correlation matrix must be square")
    level = finite(stress_level)
    if not 0 <= level <= 1:
        raise FinanceInputError("stress_level must be in [0,1]")
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            value = values[i][j]
            if not -1 <= value <= 1:
                raise FinanceInputError("correlations must be in [-1,1]")
            row.append(1.0 if i == j else value+(1-value)*level)
        result.append(tuple(row))
    return tuple(result)


def factor_volatility_shock(volatilities: Mapping[str, Number],
                            shocks: Mapping[str, Number]) -> dict[str, float]:
    """Apply additive volatility shocks in absolute volatility units."""
    if set(volatilities) != set(shocks):
        raise FinanceInputError("volatility and shock keys must match")
    result = {k: finite(volatilities[k])+finite(shocks[k]) for k in volatilities}
    if any(v < 0 for v in result.values()):
        raise FinanceInputError("stressed volatility cannot be negative")
    return result


def liquidity_haircut_pnl(position_value: Number, liquidation_discount: Number) -> float:
    """Mark-to-stressed-liquidation value loss for a positive asset position."""
    position, discount = finite(position_value), finite(liquidation_discount)
    if position < 0 or not 0 <= discount <= 1:
        raise FinanceInputError("position must be non-negative and discount in [0,1]")
    return -position*discount
