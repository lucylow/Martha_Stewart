"""Liquidity horizon, liquidation cost and liquidity-adjusted risk estimates."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean


def liquidation_horizon(position_values: Sequence[float], average_daily_dollar_volume: Sequence[float],
                        participation_rate: float = 0.1) -> tuple[float,...]:
    p=finite_series(position_values); adv=finite_series(average_daily_dollar_volume)
    if len(p)!=len(adv) or not 0<participation_rate<=1 or any(v<=0 for v in adv): raise ValueError("invalid positions or participation")
    return tuple(abs(value)/(volume*participation_rate) for value,volume in zip(p,adv))


def liquidation_cost(position_values: Sequence[float], adv: Sequence[float],
                     daily_volatility: Sequence[float], *, participation_rate: float = 0.1,
                     impact_coefficient: float = 0.1, spread_bps: Sequence[float] | None = None) -> dict:
    p=finite_series(position_values); a=finite_series(adv); v=finite_series(daily_volatility)
    spreads=[0.0]*len(p) if spread_bps is None else finite_series(spread_bps)
    if not(len(p)==len(a)==len(v)==len(spreads)) or any(x<=0 for x in a) or any(x<0 for x in v+spreads): raise ValueError("invalid liquidity inputs")
    if not 0<participation_rate<=1 or impact_coefficient<0: raise ValueError("invalid impact assumptions")
    rows=[]; total=0.0
    for value,volume,sigma,spread in zip(p,a,v,spreads):
        participation=abs(value)/(volume*max(participation_rate,1e-12))
        impact=impact_coefficient*sigma*sqrt(max(0.0,abs(value)/volume))
        spread_cost=spread/10000.0/2.0
        rate=impact+spread_cost; cost=abs(value)*rate; total+=cost
        rows.append({"liquidation_days":participation,"impact_rate":impact,"spread_rate":spread_cost,"cost":cost})
    gross=sum(abs(x) for x in p)
    return {"total_cost":total,"cost_bps":total/gross*10000 if gross else 0.0,
            "by_asset":tuple(rows),"assumptions":{"participation_rate":participation_rate,"impact_coefficient":impact_coefficient}}


def liquidity_adjusted_var(var_loss: float, position_values: Sequence[float], adv: Sequence[float],
                            daily_volatility: Sequence[float], *, confidence: float = 0.99,
                            participation_rate: float = 0.1, impact_coefficient: float = 0.1,
                            spread_bps: Sequence[float] | None = None) -> dict:
    if var_loss<0 or not 0.5<confidence<1: raise ValueError("invalid VaR or confidence")
    cost=liquidation_cost(position_values,adv,daily_volatility,participation_rate=participation_rate,
                          impact_coefficient=impact_coefficient,spread_bps=spread_bps)
    return {"market_var":var_loss,"liquidation_cost":cost["total_cost"],
            "liquidity_adjusted_var":var_loss+cost["total_cost"],"confidence":confidence,
            "liquidity_cost_bps":cost["cost_bps"]}


def amihud_illiquidity(returns: Sequence[float], dollar_volume: Sequence[float]) -> float:
    r=finite_series(returns); v=finite_series(dollar_volume)
    if len(r)!=len(v) or any(x<=0 for x in v): raise ValueError("returns and positive volumes must align")
    return mean([abs(a)/b for a,b in zip(r,v)])


def capacity_estimate(average_daily_volume: float, participation_rate: float,
                      rebalance_days: int = 1) -> float:
    if average_daily_volume<=0 or not 0<participation_rate<=1 or rebalance_days<1: raise ValueError("invalid capacity inputs")
    return average_daily_volume*participation_rate*rebalance_days
