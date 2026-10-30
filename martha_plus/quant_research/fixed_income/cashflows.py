"""Fixed-income cash-flow valuation, duration, convexity and immunization."""
from __future__ import annotations
from math import exp
from typing import Sequence
from ..core import finite_series


def bond_cashflows(face_value: float, coupon_rate: float, maturity: float,
                   frequency: int = 2) -> tuple[tuple[float,float],...]:
    if face_value<=0 or coupon_rate<0 or maturity<=0 or frequency<1: raise ValueError("invalid bond terms")
    periods=round(maturity*frequency)
    if abs(periods/frequency-maturity)>1e-8: raise ValueError("maturity must align to coupon frequency")
    coupon=face_value*coupon_rate/frequency
    return tuple((i/frequency,coupon+(face_value if i==periods else 0.0)) for i in range(1,periods+1))


def present_value(cashflows: Sequence[tuple[float,float]], zero_rates: Sequence[float],
                  times: Sequence[float] | None = None, compounding: str = "continuous") -> float:
    r=finite_series(zero_rates); t=finite_series(times if times is not None else [cf[0] for cf in cashflows])
    if len(r)!=len(cashflows) or len(t)!=len(cashflows): raise ValueError("cashflow curve dimensions mismatch")
    total=0.0
    for (_,amount),rate,maturity in zip(cashflows,r,t):
        if maturity<0: raise ValueError("cashflow time must be nonnegative")
        if compounding=="continuous": df=exp(-rate*maturity)
        elif compounding=="annual":
            if 1+rate<=0: raise ValueError("annual discount base must be positive")
            df=(1+rate)**(-maturity)
        else: raise ValueError("unsupported compounding")
        total+=amount*df
    return total


def macaulay_duration(cashflows: Sequence[tuple[float,float]], yield_rate: float,
                      frequency: int = 2) -> float:
    if not cashflows or yield_rate<=-frequency or frequency<1: raise ValueError("invalid cashflows or yield")
    pv=[]
    for time,amount in cashflows:
        if time<0 or amount<0: raise ValueError("cashflow times and amounts must be nonnegative")
        pv.append(amount/(1+yield_rate/frequency)**(time*frequency))
    price=sum(pv)
    return sum(cf[0]*v for cf,v in zip(cashflows,pv))/price if price>0 else 0.0


def modified_duration(cashflows: Sequence[tuple[float,float]], yield_rate: float,
                      frequency: int = 2) -> float:
    return macaulay_duration(cashflows,yield_rate,frequency)/(1+yield_rate/frequency)


def convexity(cashflows: Sequence[tuple[float,float]], yield_rate: float,
              frequency: int = 2) -> float:
    if not cashflows or yield_rate<=-frequency or frequency<1: raise ValueError("invalid cashflows or yield")
    price=sum(amount/(1+yield_rate/frequency)**(time*frequency) for time,amount in cashflows)
    if price<=0: return 0.0
    numerator=sum(amount*time*(time+1/frequency)/(1+yield_rate/frequency)**(time*frequency+2) for time,amount in cashflows)
    return numerator/price


def dollar_duration(price: float, modified_duration_value: float, basis_points: float = 1.0) -> float:
    if price<0 or modified_duration_value<0 or basis_points<=0: raise ValueError("invalid duration inputs")
    return price*modified_duration_value*basis_points/10000.0


def key_rate_duration(base_price: float, bumped_up_prices: Sequence[float],
                      bumped_down_prices: Sequence[float], bump: float = 1e-4) -> tuple[float,...]:
    up=finite_series(bumped_up_prices); down=finite_series(bumped_down_prices)
    if len(up)!=len(down) or base_price<=0 or bump<=0: raise ValueError("invalid key rate inputs")
    return tuple((d-u)/(2*bump*base_price) for u,d in zip(up,down))


def immunizing_weights(durations: Sequence[float], target_duration: float,
                       *, cash_duration: float = 0.0) -> tuple[float,float]:
    """Two-instrument duration match with weights summing to one."""
    d=finite_series(durations)
    if len(d)!=2 or target_duration<min(d) or target_duration>max(d) or d[0]==d[1]:
        raise ValueError("target duration must lie between two distinct instrument durations")
    w1=(d[1]-target_duration)/(d[1]-d[0]); return (w1,1-w1)


def duration_gap(asset_duration: float, liability_duration: float,
                 asset_value: float, liability_value: float) -> float:
    if min(asset_duration,liability_duration,asset_value,liability_value)<0: raise ValueError("duration and values must be nonnegative")
    return asset_duration*asset_value-liability_duration*liability_value
