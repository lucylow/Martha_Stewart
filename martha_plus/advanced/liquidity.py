"""Liquidity, market-impact, and capacity diagnostics for research simulations."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean
from typing import Mapping, Sequence
from ..types import Quote, Side


@dataclass(frozen=True, slots=True)
class LiquidityEstimate:
    symbol: str
    observations: int
    median_spread_bps: float
    p95_spread_bps: float
    median_dollar_volume: float
    average_dollar_volume: float
    median_quoted_depth_dollars: float
    zero_volume_fraction: float


@dataclass(frozen=True, slots=True)
class CostEstimate:
    notional: float
    spread_cost: float
    commission: float
    temporary_impact: float
    permanent_impact: float
    total_cost: float
    cost_bps: float
    assumptions: Mapping[str, float | str]


def _quantile(values: Sequence[float], q: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    pos = (len(data) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def liquidity_estimate(symbol: str, closes: Sequence[float], volumes: Sequence[float],
                       quotes: Sequence[Quote] = ()) -> LiquidityEstimate:
    if len(closes) != len(volumes):
        raise ValueError("closes and volumes must align")
    px = [float(x) for x in closes]
    vol = [float(x) for x in volumes]
    if any(not isfinite(x) or x <= 0 for x in px) or any(not isfinite(x) or x < 0 for x in vol):
        raise ValueError("prices must be positive and volumes non-negative finite")
    dollar = [p * v for p, v in zip(px, vol)]
    spreads = [q.spread_bps for q in quotes if q.symbol == symbol.strip().upper()]
    depths = [(q.bid * q.bid_size + q.ask * q.ask_size) for q in quotes
              if q.symbol == symbol.strip().upper()]
    if any(not isfinite(x) for x in spreads + depths):
        raise ValueError("quote measurements must be finite")
    return LiquidityEstimate(symbol.strip().upper(), len(px), _quantile(spreads, .5),
                             _quantile(spreads, .95), _quantile(dollar, .5),
                             mean(dollar) if dollar else 0.0, _quantile(depths, .5),
                             sum(v == 0 for v in vol) / len(vol) if vol else 0.0)


def square_root_impact(notional: float, average_daily_dollar_volume: float,
                       daily_volatility: float, *, impact_coefficient: float = 0.1) -> float:
    """Estimate one-way fractional impact as k * sigma * sqrt(participation)."""
    values = (notional, average_daily_dollar_volume, daily_volatility, impact_coefficient)
    if any(not isfinite(float(x)) for x in values):
        raise ValueError("impact parameters must be finite")
    if notional < 0 or average_daily_dollar_volume <= 0 or daily_volatility < 0 or impact_coefficient < 0:
        raise ValueError("invalid impact parameters")
    participation = notional / average_daily_dollar_volume
    return impact_coefficient * daily_volatility * sqrt(participation)


def estimate_transaction_cost(*, side: Side, quantity: float, price: float,
                              spread_bps: float, commission_bps: float,
                              average_daily_dollar_volume: float,
                              daily_volatility: float, impact_coefficient: float = 0.1,
                              permanent_impact_fraction: float = 0.25) -> CostEstimate:
    """Transparent toy cost model for scenario analysis; not a fill-price guarantee."""
    if not isinstance(side, Side):
        raise ValueError("side must be Side enum")
    values = (quantity, price, spread_bps, commission_bps, average_daily_dollar_volume,
              daily_volatility, impact_coefficient, permanent_impact_fraction)
    if any(not isfinite(float(x)) for x in values):
        raise ValueError("cost inputs must be finite")
    if quantity < 0 or price <= 0 or min(spread_bps, commission_bps, daily_volatility, impact_coefficient) < 0:
        raise ValueError("invalid cost inputs")
    if average_daily_dollar_volume <= 0 or not 0 <= permanent_impact_fraction <= 1:
        raise ValueError("invalid ADV or permanent-impact fraction")
    notional = quantity * price
    spread = notional * spread_bps / 20_000  # half-spread cost paid on one-way execution
    commission = notional * commission_bps / 10_000
    temporary_fraction = square_root_impact(notional, average_daily_dollar_volume,
                                            daily_volatility, impact_coefficient=impact_coefficient)
    temporary = notional * temporary_fraction
    permanent = temporary * permanent_impact_fraction
    total = spread + commission + temporary + permanent
    return CostEstimate(notional, spread, commission, temporary, permanent, total,
                        total / notional * 10_000 if notional else 0.0,
                        {"spread_bps": spread_bps, "commission_bps": commission_bps,
                         "daily_volatility": daily_volatility, "impact_coefficient": impact_coefficient,
                         "permanent_impact_fraction": permanent_impact_fraction,
                         "assumption": "square-root impact is a stylized model, not a calibrated prediction"})


def capacity_curve(average_daily_dollar_volume: float, daily_volatility: float,
                   participation_rates: Sequence[float], impact_coefficient: float = 0.1) -> list[dict[str, float]]:
    if average_daily_dollar_volume <= 0 or daily_volatility < 0:
        raise ValueError("ADV must be positive and volatility non-negative")
    result = []
    for rate in participation_rates:
        r = float(rate)
        if not isfinite(r) or not 0 < r <= 1:
            raise ValueError("participation rates must be in (0, 1]")
        notional = average_daily_dollar_volume * r
        impact = square_root_impact(notional, average_daily_dollar_volume,
                                    daily_volatility, impact_coefficient=impact_coefficient)
        result.append({"participation_rate": r, "daily_notional_capacity": notional,
                       "estimated_impact_bps": impact * 10_000})
    return result


def quote_imbalance(quote: Quote) -> dict[str, float]:
    total = quote.bid_size + quote.ask_size
    depth_imbalance = (quote.bid_size - quote.ask_size) / total if total > 0 else 0.0
    weighted = quote.bid * quote.ask_size + quote.ask * quote.bid_size
    microprice = weighted / total if total > 0 else quote.midpoint
    return {"midpoint": quote.midpoint, "microprice": microprice,
            "microprice_deviation_bps": (microprice / quote.midpoint - 1) * 10_000,
            "depth_imbalance": depth_imbalance, "spread_bps": quote.spread_bps,
            "bid_depth_dollars": quote.bid * quote.bid_size,
            "ask_depth_dollars": quote.ask * quote.ask_size}


def participation_summary(filled_notional: float, market_dollar_volume: float) -> dict[str, float | bool]:
    if not isfinite(filled_notional) or not isfinite(market_dollar_volume) or filled_notional < 0 or market_dollar_volume < 0:
        raise ValueError("notional and market volume must be finite and non-negative")
    if market_dollar_volume == 0:
        return {"participation": 0.0, "zero_market_volume": True, "over_10_percent": False}
    rate = filled_notional / market_dollar_volume
    return {"participation": rate, "zero_market_volume": False, "over_10_percent": rate > 0.10}
