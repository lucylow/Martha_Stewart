"""Credit survival curves, hazard rates and CDS spread approximations."""
from __future__ import annotations
from math import exp
from typing import Sequence
from ..core import finite_series


def survival_from_hazards(times: Sequence[float], hazard_rates: Sequence[float]) -> tuple[float,...]:
    t=finite_series(times,name="times"); h=finite_series(hazard_rates,name="hazard_rates")
    if len(t)!=len(h) or any(v<=0 for v in t) or any(v<0 for v in h) or any(b<=a for a,b in zip(t,t[1:])):
        raise ValueError("times must increase and hazards be nonnegative")
    previous=0.0; cumulative=0.0; out=[]
    for time,hazard in zip(t,h):
        cumulative+=hazard*(time-previous); previous=time
        out.append(exp(-cumulative))
    return tuple(out)


def hazard_from_survival(times: Sequence[float], survival: Sequence[float]) -> tuple[float,...]:
    t=finite_series(times); s=finite_series(survival)
    if len(t)!=len(s) or any(v<=0 for v in t) or any(not 0<v<=1 for v in s): raise ValueError("invalid curve")
    if any(b>=a for a,b in zip(s,s[1:])): raise ValueError("survival must be non-increasing")
    previous_t=0.0; previous_s=1.0; out=[]
    for ti,si in zip(t,s):
        dt=ti-previous_t
        if dt<=0: raise ValueError("times must increase")
        out.append(-__import__('math').log(si/previous_s)/dt)
        previous_t,previous_s=ti,si
    return tuple(out)


def default_probabilities(survival: Sequence[float]) -> tuple[float,...]:
    s=finite_series(survival)
    if any(v<0 or v>1 for v in s) or any(b>a for a,b in zip(s,s[1:])): raise ValueError("invalid survival probabilities")
    previous=1.0; out=[]
    for v in s: out.append(previous-v); previous=v
    return tuple(out)


def risky_annuity(payment_times: Sequence[float], discount_factors: Sequence[float],
                  survival: Sequence[float], *, accrual: Sequence[float] | None = None) -> float:
    t=finite_series(payment_times); d=finite_series(discount_factors); s=finite_series(survival)
    if not (len(t)==len(d)==len(s)) or any(v<0 for v in d) or any(v<0 or v>1 for v in s): raise ValueError("curve dimensions mismatch")
    alpha=[t[0]]+[t[i]-t[i-1] for i in range(1,len(t))] if accrual is None else finite_series(accrual)
    if len(alpha)!=len(t) or any(v<=0 for v in alpha): raise ValueError("accrual factors must be positive")
    return sum(a*b*c for a,b,c in zip(alpha,d,s))


def cds_par_spread(payment_times: Sequence[float], discount_factors: Sequence[float],
                   survival: Sequence[float], recovery: float = 0.4,
                   *, accrual: Sequence[float] | None = None) -> dict:
    """Discrete premium/protection leg par spread; ignores accrued premium on default."""
    if not 0<=recovery<1: raise ValueError("recovery must be in [0,1)")
    t=finite_series(payment_times); d=finite_series(discount_factors); s=finite_series(survival)
    if not (len(t)==len(d)==len(s)): raise ValueError("curve arrays must match")
    premium=risky_annuity(t,d,s,accrual=accrual)
    defaults=default_probabilities(s)
    protection=sum(d[i]*defaults[i]*(1-recovery) for i in range(len(t)))
    spread=protection/premium if premium>0 else float("inf")
    return {"par_spread":spread,"protection_leg_pv":protection,"premium_leg_annuity":premium,
            "recovery":recovery,"approximation":"discrete premium leg; no accrued premium-on-default correction"}


def credit_cs01(payment_times: Sequence[float], discount_factors: Sequence[float],
                survival: Sequence[float], recovery: float = 0.4, bump: float = 1e-4) -> float:
    h=hazard_from_survival(payment_times,survival)
    up=survival_from_hazards(payment_times,[v+bump for v in h])
    down=survival_from_hazards(payment_times,[max(0.0,v-bump) for v in h])
    base=cds_par_spread(payment_times,discount_factors,survival,recovery)["par_spread"]
    pu=cds_par_spread(payment_times,discount_factors,up,recovery)["par_spread"]
    pd=cds_par_spread(payment_times,discount_factors,down,recovery)["par_spread"]
    return (pu-pd)/(2*bump) if bump>0 else base
