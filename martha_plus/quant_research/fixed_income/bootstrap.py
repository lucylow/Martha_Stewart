"""Discount-curve bootstrapping from deposits and par swap quotes."""
from __future__ import annotations
from math import exp,log
from typing import Sequence
from ..core import finite_series


def discount_from_zero_rate(rate: float, maturity: float, compounding: str = "continuous") -> float:
    if maturity<0: raise ValueError("maturity must be non-negative")
    if compounding=="continuous": return exp(-rate*maturity)
    if compounding=="annual":
        if 1+rate<=0: raise ValueError("annual compounding base must be positive")
        return (1+rate)**(-maturity)
    raise ValueError("compounding must be continuous or annual")


def zero_rate_from_discount(discount_factor: float, maturity: float) -> float:
    if discount_factor<=0 or maturity<=0: raise ValueError("discount factor and maturity must be positive")
    return -log(discount_factor)/maturity


def interpolate_discount(maturity: float, knot_times: Sequence[float],
                          discount_factors: Sequence[float]) -> float:
    """Log-linear interpolation/extrapolation of positive discount factors."""
    t=finite_series(knot_times); d=finite_series(discount_factors)
    if len(t)!=len(d) or any(v<=0 for v in t) or any(v<=0 for v in d): raise ValueError("invalid curve knots")
    if any(b<=a for a,b in zip(t,t[1:])): raise ValueError("knot times must increase")
    if maturity<=0: return 1.0
    if maturity<=t[0]: return exp(log(d[0])*maturity/t[0])
    if maturity>=t[-1]:
        if len(t)==1: return exp(log(d[-1])*maturity/t[-1])
        slope=(log(d[-1])-log(d[-2]))/(t[-1]-t[-2])
        return exp(log(d[-1])+slope*(maturity-t[-1]))
    for i in range(1,len(t)):
        if t[i-1]<=maturity<=t[i]:
            w=(maturity-t[i-1])/(t[i]-t[i-1])
            return exp((1-w)*log(d[i-1])+w*log(d[i]))
    raise RuntimeError("interpolation failed")


def bootstrap_discount_curve(deposits: Sequence[tuple[float,float]],
                             swaps: Sequence[tuple[float,float]], *,
                             payment_frequency: int = 1) -> dict:
    """Bootstrap from (maturity, simple deposit rate) and (maturity, par swap rate).

    Swap coupons are assumed to occur at regular intervals equal to
    ``1/payment_frequency`` years. Earlier payment-date discount factors must
    already be bootstrapped or interpolatable from prior knots.
    """
    if payment_frequency<1: raise ValueError("payment_frequency must be positive")
    knots={0.0:1.0}; instruments=[]
    for maturity,rate in sorted(deposits):
        maturity=float(maturity); rate=float(rate)
        if maturity<=0 or 1+rate*maturity<=0: raise ValueError("invalid deposit instrument")
        df=1.0/(1.0+rate*maturity)
        knots[maturity]=df; instruments.append({"type":"deposit","maturity":maturity,"rate":rate,"discount_factor":df})
    for maturity,rate in sorted(swaps):
        maturity=float(maturity); rate=float(rate)
        if maturity<=0 or rate<0: raise ValueError("invalid swap quote")
        step=1.0/payment_frequency
        count=round(maturity*payment_frequency)
        if abs(count*step-maturity)>1e-8: raise ValueError("swap maturity must align to payment frequency")
        times=[step*i for i in range(1,count+1)]
        known_annuity=0.0
        for t in times[:-1]:
            if t not in knots:
                earlier=sorted(k for k in knots if k>0)
                # If this is the first non-deposit coupon date, use the current
                # par quote as a flat-zero bootstrap anchor. Later dates use
                # log-linear interpolation through the calibrated nodes.
                if not earlier:
                    knots[t]=exp(-rate*t)
                else:
                    knots[t]=interpolate_discount(t,earlier,[knots[k] for k in earlier])
            known_annuity+=step*knots[t]
        final_df=(1.0-rate*known_annuity)/(1.0+rate*step)
        if final_df<=0: raise ValueError("quotes imply a non-positive discount factor")
        knots[maturity]=final_df
        instruments.append({"type":"swap","maturity":maturity,"rate":rate,"discount_factor":final_df})
    times=sorted(k for k in knots if k>0); dfs=[knots[k] for k in times]
    zeros=[zero_rate_from_discount(df,t) for t,df in zip(times,dfs)]
    forwards=[]; prev_t=0.0; prev_df=1.0
    for t,df in zip(times,dfs):
        forwards.append(-log(df/prev_df)/(t-prev_t)); prev_t,prev_df=t,df
    return {"times":tuple(times),"discount_factors":tuple(dfs),"zero_rates":tuple(zeros),
            "instantaneous_forward_averages":tuple(forwards),"instruments":tuple(instruments),
            "method":"deposit simple-interest plus par-swap bootstrap"}


def parallel_curve_shift(discount_factors: Sequence[float], times: Sequence[float], shift: float) -> tuple[float,...]:
    d=finite_series(discount_factors); t=finite_series(times)
    if len(d)!=len(t) or any(v<=0 for v in d) or any(v<0 for v in t): raise ValueError("curve arrays invalid")
    return tuple(df*exp(-shift*maturity) for df,maturity in zip(d,t))
