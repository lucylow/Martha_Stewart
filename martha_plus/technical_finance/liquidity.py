"""Liquidity-aware portfolio diagnostics and transaction-cost estimators."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .core import FinanceInputError, Number, finite, positive, vector


def amihud_illiquidity(returns: Sequence[Number], dollar_volume: Sequence[Number]) -> float:
    """Mean |return| / dollar volume; larger values indicate less liquidity."""
    r, volume = vector(returns), vector(dollar_volume)
    if len(r) != len(volume):
        raise FinanceInputError("returns and dollar_volume must align")
    if any(v <= 0 for v in volume):
        raise FinanceInputError("dollar volume must be positive")
    return sum(abs(a)/v for a, v in zip(r, volume))/len(r)


def turnover(previous_weights: Mapping[str, Number],
             current_weights: Mapping[str, Number]) -> dict[str, float]:
    """Gross traded weight, one-way turnover and buys/sells from two portfolios."""
    keys = set(previous_weights) | set(current_weights)
    changes = {k: finite(current_weights.get(k, 0))-finite(previous_weights.get(k, 0)) for k in keys}
    buys = sum(max(0.0, x) for x in changes.values())
    sells = sum(max(0.0, -x) for x in changes.values())
    return {"buys": buys, "sells": sells, "gross_turnover": buys+sells,
            "one_way_turnover": 0.5*(buys+sells), "net_flow": buys-sells}


def linear_transaction_cost(notional: Number, half_spread_bps: Number,
                            commission_bps: Number = 0.0,
                            exchange_fee_bps: Number = 0.0) -> float:
    """One-way cost estimate as absolute notional times bps components."""
    value = abs(finite(notional))
    spread, commission, fee = map(finite, (half_spread_bps, commission_bps, exchange_fee_bps))
    if min(spread, commission, fee) < 0:
        raise FinanceInputError("cost components must be non-negative")
    return value*(spread+commission+fee)/10000


def square_root_impact(notional: Number, daily_volume: Number,
                       volatility: Number, coefficient: Number = 0.5) -> float:
    """Square-root market impact estimate in return units.

    This empirical form is a rough scenario input, not a calibrated execution
    model. The coefficient is market- and venue-dependent.
    """
    q, adv = abs(finite(notional)), positive(daily_volume)
    vol, eta = positive(volatility, allow_zero=True), positive(coefficient, allow_zero=True)
    return eta*vol*math.sqrt(q/adv)


def liquidity_adjusted_var(base_var: Number, position_value: Number,
                           liquidation_horizon_days: Number,
                           daily_volume: Number, half_spread_bps: Number,
                           impact_coefficient: Number = 0.5) -> dict[str, float]:
    """Add a simple spread and square-root liquidation-cost haircut to VaR."""
    var, position = positive(base_var, allow_zero=True), positive(position_value)
    horizon = positive(liquidation_horizon_days)
    adv = positive(daily_volume)
    if position/adv > 1:
        participation = position/adv
    else:
        participation = position/adv
    spread_cost = linear_transaction_cost(position, half_spread_bps)
    impact_rate = positive(impact_coefficient, allow_zero=True)*math.sqrt(participation)
    impact_cost = position*impact_rate
    total = var+spread_cost+impact_cost
    return {"base_var": var, "spread_cost": spread_cost,
            "impact_cost": impact_cost, "liquidity_adjusted_var": total,
            "liquidation_horizon_days": horizon}


def days_to_liquidate(position_value: Number, daily_volume: Number,
                      participation_rate: Number = 0.1) -> float:
    """Trading days to unwind a position at a fixed ADV participation rate."""
    position, adv, participation = positive(position_value), positive(daily_volume), positive(participation_rate)
    if participation > 1:
        raise FinanceInputError("participation_rate must not exceed one")
    return position/(adv*participation)


def capacity_estimate(expected_alpha: Number, annual_volatility: Number,
                      daily_volume: Number, max_participation: Number,
                      impact_coefficient: Number = 0.5) -> dict[str, float]:
    """Crude capacity screen where alpha is compared to square-root impact.

    Solves eta * vol * sqrt(q/ADV) = alpha for a daily traded notional q.
    Returns zero where expected alpha is non-positive.
    """
    alpha, vol, adv = finite(expected_alpha), positive(annual_volatility), positive(daily_volume)
    participation, eta = finite(max_participation), positive(impact_coefficient)
    if not 0 < participation <= 1:
        raise FinanceInputError("max_participation must be in (0,1]")
    daily_vol = vol/math.sqrt(252)
    q_break_even = adv*(max(0.0, alpha)/(eta*daily_vol))**2 if daily_vol else 0.0
    capacity = min(adv*participation, q_break_even)
    return {"daily_capacity": capacity, "participation_cap": adv*participation,
            "impact_break_even_capacity": q_break_even}


def implementation_shortfall(decision_price: Number, execution_prices: Sequence[Number],
                              quantities: Sequence[Number], side: str,
                              *, opportunity_cost: Number = 0.0) -> dict[str, float]:
    """Implementation shortfall in price units and basis points of decision notional."""
    decision = positive(decision_price)
    prices, sizes = vector(execution_prices), vector(quantities)
    if len(prices) != len(sizes) or any(q < 0 for q in sizes):
        raise FinanceInputError("execution prices and non-negative quantities must align")
    kind = side.lower()
    if kind not in {"buy", "sell"}:
        raise FinanceInputError("side must be buy or sell")
    sign = 1.0 if kind == "buy" else -1.0
    total_qty = sum(sizes)
    if total_qty <= 0:
        raise FinanceInputError("total executed quantity must be positive")
    executed_cost = sum(q*(p-decision)*sign for p, q in zip(prices, sizes))
    opportunity = finite(opportunity_cost)
    total = executed_cost+opportunity
    notional = total_qty*decision
    return {"executed_shortfall": executed_cost,
            "opportunity_cost": opportunity,
            "total_shortfall": total,
            "shortfall_bps": total/notional*10000}
