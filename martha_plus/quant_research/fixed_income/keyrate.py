"""Key-rate, curve-shape and parallel/nonparallel rate-risk diagnostics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series


def curve_shape_factors(tenors: Sequence[float], zero_rates: Sequence[float]) -> dict:
    t=finite_series(tenors); r=finite_series(zero_rates)
    if len(t)!=len(r) or len(t)<3 or any(v<=0 for v in t) or any(b<=a for a,b in zip(t,t[1:])): raise ValueError("need increasing tenor grid")
    short=r[0]; belly=r[len(r)//2]; long=r[-1]
    return {"level":sum(r)/len(r),"slope":long-short,"curvature":2*belly-short-long,
            "short_rate":short,"belly_rate":belly,"long_rate":long}


def key_rate_duration_from_prices(base_price: float, up_prices: Sequence[float],
                                  down_prices: Sequence[float], bump: float = 1e-4) -> tuple[float,...]:
    up=finite_series(up_prices); down=finite_series(down_prices)
    if base_price<=0 or len(up)!=len(down) or bump<=0: raise ValueError("invalid key-rate prices")
    return tuple((down[i]-up[i])/(2*bump*base_price) for i in range(len(up)))


def curve_factor_exposure(key_rate_durations: Sequence[float], tenors: Sequence[float]) -> dict:
    k=finite_series(key_rate_durations); t=finite_series(tenors)
    if len(k)!=len(t) or not t: raise ValueError("tenors and key-rate durations must align")
    # Simple orthogonal contrasts: parallel, steepening and butterfly.
    n=len(k); level=sum(k)/n
    slope=sum((i-(n-1)/2)*k[i] for i in range(n))/sum((i-(n-1)/2)**2 for i in range(n)) if n>1 else 0.0
    curvature=k[n//2]-(k[0]+k[-1])/2 if n>=3 else 0.0
    return {"level_exposure":level,"slope_exposure":slope,"curvature_exposure":curvature,
            "key_rate_durations":tuple(k),"tenors":tuple(t)}


def twist_scenario(tenors: Sequence[float], short_shock: float,
                   long_shock: float) -> tuple[float,...]:
    t=finite_series(tenors)
    if any(v<0 for v in t) or max(t)<=0: raise ValueError("tenors must be nonnegative")
    maximum=max(t)
    return tuple(short_shock+(long_shock-short_shock)*v/maximum for v in t)


def key_rate_pnl(key_rate_durations: Sequence[float], market_value: float,
                 key_rate_shocks: Sequence[float]) -> float:
    d=finite_series(key_rate_durations); s=finite_series(key_rate_shocks)
    if len(d)!=len(s) or market_value<0: raise ValueError("invalid exposures or shocks")
    return -market_value*sum(a*b for a,b in zip(d,s))
