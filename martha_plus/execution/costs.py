"""Transaction-cost estimates separate from the simplistic paper-fill price model."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True, slots=True)
class CostEstimate:
    notional: float
    commission: float
    spread_cost: float
    slippage_cost: float
    total_cost: float
    effective_cost_bps: float


def estimate_costs(notional: float, commission_bps: float = 1.0,
                   spread_bps: float = 2.0, slippage_bps: float = 2.0,
                   average_daily_dollar_volume: float | None = None,
                   impact_coefficient_bps: float = 10.0) -> CostEstimate:
    """Estimate round-trip-independent one-way costs on a single order.

    The square-root impact term is a rough sensitivity model, not a calibrated
    market-impact guarantee. If volume is given, impact is added to base slippage.
    """
    if notional < 0 or min(commission_bps, spread_bps, slippage_bps, impact_coefficient_bps) < 0:
        raise ValueError("notional and cost rates cannot be negative")
    impact = 0.0
    if average_daily_dollar_volume is not None:
        if average_daily_dollar_volume <= 0:
            raise ValueError("average daily dollar volume must be positive")
        participation = min(1.0, notional / average_daily_dollar_volume)
        impact = impact_coefficient_bps * sqrt(participation)
    commission = notional * commission_bps / 10_000.0
    spread_cost = notional * spread_bps / 10_000.0
    slippage_cost = notional * (slippage_bps + impact) / 10_000.0
    total = commission + spread_cost + slippage_cost
    bps = total / notional * 10_000.0 if notional > 0 else 0.0
    return CostEstimate(notional, commission, spread_cost, slippage_cost, total, bps)
