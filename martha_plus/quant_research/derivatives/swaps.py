"""Interest-rate swap valuation using supplied discount and forward curves."""
from __future__ import annotations
from typing import Sequence
from ..core import finite_series


def par_swap_rate(payment_times: Sequence[float], discount_factors: Sequence[float],
                  accrual_fractions: Sequence[float] | None = None) -> float:
    t=finite_series(payment_times); d=finite_series(discount_factors)
    if len(t)!=len(d) or any(v<=0 for v in t) or any(v<=0 for v in d): raise ValueError("invalid curve")
    accrual=[t[0]]+[t[i]-t[i-1] for i in range(1,len(t))] if accrual_fractions is None else finite_series(accrual_fractions)
    if len(accrual)!=len(t) or any(v<=0 for v in accrual): raise ValueError("invalid accrual fractions")
    annuity=sum(a*b for a,b in zip(accrual,d))
    return (1.0-d[-1])/annuity


def swap_pv(fixed_rate: float, payment_times: Sequence[float],
            discount_factors: Sequence[float], forward_rates: Sequence[float],
            accrual_fractions: Sequence[float] | None = None,
            notional: float = 1.0, payer_fixed: bool = True) -> dict:
    t=finite_series(payment_times); d=finite_series(discount_factors); f=finite_series(forward_rates)
    if not(len(t)==len(d)==len(f)) or notional<=0: raise ValueError("curve dimensions or notional invalid")
    alpha=[t[0]]+[t[i]-t[i-1] for i in range(1,len(t))] if accrual_fractions is None else finite_series(accrual_fractions)
    if len(alpha)!=len(t) or any(v<=0 for v in alpha): raise ValueError("invalid accrual fractions")
    fixed_leg=notional*fixed_rate*sum(a*b for a,b in zip(alpha,d))
    floating_leg=notional*sum(a*b*c for a,b,c in zip(alpha,f,d))
    value=floating_leg-fixed_leg
    if not payer_fixed: value=-value
    return {"pv":value,"fixed_leg_pv":fixed_leg,"floating_leg_pv":floating_leg,
            "annuity":notional*sum(a*b for a,b in zip(alpha,d)),"payer_fixed":payer_fixed}


def swap_dv01(fixed_rate: float, payment_times: Sequence[float],
              discount_up: Sequence[float], discount_down: Sequence[float],
              forwards: Sequence[float], bump: float = 1e-4,
              accrual_fractions: Sequence[float] | None = None,
              notional: float = 1.0) -> float:
    if bump<=0: raise ValueError("bump must be positive")
    up=swap_pv(fixed_rate,payment_times,discount_up,forwards,accrual_fractions,notional)["pv"]
    down=swap_pv(fixed_rate,payment_times,discount_down,forwards,accrual_fractions,notional)["pv"]
    return (up-down)/(2*bump)*1e-4


def discount_factors_from_zero_rates(times: Sequence[float], zero_rates: Sequence[float],
                                     compounding: str = "continuous") -> tuple[float,...]:
    from math import exp
    t=finite_series(times); r=finite_series(zero_rates)
    if len(t)!=len(r) or any(v<0 for v in t): raise ValueError("curve arrays mismatch")
    if compounding=="continuous": return tuple(exp(-a*b) for a,b in zip(t,r))
    if compounding=="annual":
        if any(1+b<=0 for b in r): raise ValueError("annual discount base must be positive")
        return tuple((1+b)**(-a) for a,b in zip(t,r))
    raise ValueError("compounding must be continuous or annual")
