"""Futures fair value, curve carry and continuous-contract construction."""
from __future__ import annotations
from math import exp, log
from typing import Sequence
from ..core import finite_series


def fair_futures_price(spot: float, rate: float, maturity: float, *,
                       income_yield: float = 0.0, storage_cost: float = 0.0) -> float:
    """Continuous-compounding cost-of-carry fair value."""
    if spot<=0 or maturity<0 or income_yield<0 or storage_cost<0:
        raise ValueError("invalid spot, maturity, yield or storage cost")
    return spot*exp((rate+storage_cost-income_yield)*maturity)


def implied_carry(futures: float, spot: float, maturity: float,
                  *, income_yield: float = 0.0) -> float:
    if futures<=0 or spot<=0 or maturity<=0: raise ValueError("prices and maturity must be positive")
    return log(futures/spot)/maturity+income_yield


def calendar_spread_nearby_far(nearby: float, deferred: float) -> dict:
    if nearby<=0 or deferred<=0: raise ValueError("futures prices must be positive")
    spread=deferred-nearby
    return {"spread":spread,"curve_state":"contango" if spread>0 else "backwardation" if spread<0 else "flat",
            "percentage_spread":spread/nearby}


def annualized_roll_yield(front: float, next_contract: float, days_to_roll: float) -> float:
    if front<=0 or next_contract<=0 or days_to_roll<=0: raise ValueError("prices and roll days must be positive")
    return (front/next_contract-1.0)*365.0/days_to_roll


def continuous_contract(prices_by_contract: Sequence[Sequence[float]], roll_indices: Sequence[int],
                        method: str = "difference") -> dict:
    """Back-adjust a stitched futures series at known roll indices.

    ``prices_by_contract`` is ordered by maturity.  Each contract row must be
    aligned to the same time grid; a roll index identifies the first observation
    belonging to the next contract.  This assumes a research-quality roll map.
    """
    contracts=[finite_series(c,name="contract prices") for c in prices_by_contract]
    if len(contracts)<2 or any(len(c)!=len(contracts[0]) for c in contracts): raise ValueError("contracts must be aligned")
    if method not in {"difference","ratio"}: raise ValueError("method must be difference or ratio")
    rolls=list(map(int,roll_indices))
    if len(rolls)!=len(contracts)-1 or sorted(rolls)!=rolls or any(r<=0 or r>=len(contracts[0]) for r in rolls):
        raise ValueError("one increasing valid roll index is required per contract transition")
    active=[0]*len(contracts[0])
    for j,r in enumerate(rolls,1):
        for t in range(r,len(active)): active[t]=j
    raw=[contracts[active[t]][t] for t in range(len(active))]
    adjusted=list(raw); adjustment=0.0; ratio=1.0
    # Work backwards so historical prices are placed on the latest contract scale.
    for j in range(len(rolls)-1,-1,-1):
        r=rolls[j]; old=contracts[j][r-1]; new=contracts[j+1][r]
        if method=="difference": adjustment += new-old
        else:
            if old<=0: raise ValueError("ratio adjustment requires positive roll prices")
            ratio *= new/old
        for t in range(r): adjusted[t]=raw[t]+adjustment if method=="difference" else raw[t]*ratio
    return {"raw":tuple(raw),"adjusted":tuple(adjusted),"active_contract":tuple(active),"method":method}


def term_structure_slope(maturities: Sequence[float], prices: Sequence[float]) -> dict:
    t=finite_series(maturities); p=finite_series(prices)
    if len(t)!=len(p) or len(t)<2 or any(v<=0 for v in t) or any(v<=0 for v in p): raise ValueError("invalid futures curve")
    first=p[0]; last=p[-1]
    return {"slope":(last-first)/(t[-1]-t[0]),"log_slope":log(last/first)/(t[-1]-t[0]),
            "contango":last>first,"maturity_span":t[-1]-t[0]}
