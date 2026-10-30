"""High-frequency realized-volatility and jump-variation estimators."""
from __future__ import annotations
from math import pi, sqrt
from typing import Sequence
from ..core import finite_series, mean


def realized_variance(intraday_returns: Sequence[float]) -> float:
    r=finite_series(intraday_returns,name="intraday_returns")
    return sum(v*v for v in r)


def bipower_variation(intraday_returns: Sequence[float]) -> float:
    r=finite_series(intraday_returns,name="intraday_returns",min_length=2)
    return (pi/2.0)*sum(abs(r[i])*abs(r[i-1]) for i in range(1,len(r)))


def realized_quarticity(intraday_returns: Sequence[float]) -> float:
    r=finite_series(intraday_returns,name="intraday_returns")
    return (len(r)/3.0)*sum(v**4 for v in r)


def jump_variation(intraday_returns: Sequence[float], *, nonnegative: bool = True) -> dict:
    rv=realized_variance(intraday_returns); bv=bipower_variation(intraday_returns)
    jump=rv-bv
    if nonnegative: jump=max(0.0,jump)
    return {"realized_variance":rv,"bipower_variation":bv,"jump_variation":jump,
            "continuous_variation":min(rv,bv),"jump_fraction":jump/rv if rv>0 else 0.0}


def realized_kernel(intraday_returns: Sequence[float], bandwidth: int | None = None) -> float:
    """Bartlett realized kernel to reduce additive market-microstructure noise."""
    r=finite_series(intraday_returns,name="intraday_returns")
    n=len(r); h=max(1,int(sqrt(n))) if bandwidth is None else bandwidth
    if h<1 or h>=n: raise ValueError("bandwidth must be in [1,n-1]")
    gamma0=sum(v*v for v in r)
    value=gamma0
    for lag in range(1,h+1):
        gamma=sum(r[t]*r[t-lag] for t in range(lag,n))
        weight=1.0-lag/(h+1.0)
        value+=2.0*weight*gamma
    return max(0.0,value)


def hayashi_yoshida_covariance(times_x: Sequence[float], returns_x: Sequence[float],
                               times_y: Sequence[float], returns_y: Sequence[float]) -> float:
    """Asynchronous covariance using overlapping return intervals.

    Timestamps must be strictly increasing and intervals are (t[i-1], t[i]].
    """
    tx=list(map(float,times_x)); ty=list(map(float,times_y))
    rx=finite_series(returns_x); ry=finite_series(returns_y)
    if len(tx)!=len(rx)+1 or len(ty)!=len(ry)+1: raise ValueError("timestamps must have one more item than returns")
    if any(b<=a for a,b in zip(tx,tx[1:])) or any(b<=a for a,b in zip(ty,ty[1:])): raise ValueError("timestamps must be strictly increasing")
    total=0.0
    for i,a in enumerate(rx):
        for j,b in enumerate(ry):
            overlap=min(tx[i+1],ty[j+1])-max(tx[i],ty[j])
            if overlap>0: total+=a*b
    return total


def volatility_signature(intraday_returns: Sequence[float], sampling_steps: Sequence[int]) -> list[dict]:
    """Compute RV at subsampling intervals to inspect noise sensitivity."""
    r=finite_series(intraday_returns)
    results=[]
    for step in sampling_steps:
        if step<1 or step>len(r): raise ValueError("sampling steps must be in [1,n]")
        subsampled=[sum(r[i:i+step]) for i in range(0,len(r)-step+1,step)]
        results.append({"step":step,"observations":len(subsampled),"realized_variance":realized_variance(subsampled)})
    return results


def two_scale_realized_variance(intraday_returns: Sequence[float], scales: int = 5) -> float:
    """Simple two-scale noise correction; returns a nonnegative estimate."""
    r=finite_series(intraday_returns)
    if scales<2 or scales>len(r): raise ValueError("scales must be between 2 and n")
    estimates=[]
    for offset in range(scales):
        sample=[sum(r[i:i+scales]) for i in range(offset,len(r)-scales+1,scales)]
        if len(sample)>=2: estimates.append(realized_variance(sample)*scales)
    if not estimates: return realized_variance(r)
    corrected=sum(estimates)/len(estimates)
    noise=(scales/(len(r)-scales+1))*realized_variance(r)
    return max(0.0,(corrected-noise)/(1.0-scales/max(len(r),scales+1)))
