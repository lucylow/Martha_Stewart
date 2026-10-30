"""Pairs and cointegration diagnostics for relative-value research."""
from __future__ import annotations
from dataclasses import dataclass
from math import log
from typing import Sequence
from ..core import finite_series, mean, variance
from ..regression import ols
from .stationarity import adf_test

@dataclass(frozen=True)
class CointegrationResult:
    intercept: float
    hedge_ratio: float
    residuals: tuple[float,...]
    adf_statistic: float
    adf_p_value_approx: float
    half_life: float
    nobs: int
    method: str = "Engle-Granger two-step; residual ADF p-value is approximate"


def hedge_ratio(y: Sequence[float], x: Sequence[float], *, include_intercept: bool = True) -> tuple[float,float,list[float]]:
    a,b=finite_series(y,name="y",min_length=20),finite_series(x,name="x",min_length=20)
    if len(a)!=len(b): raise ValueError("series lengths must match")
    fit=ols([b],a,add_intercept=include_intercept)
    intercept=fit.coefficients[0] if include_intercept else 0.0
    beta=fit.coefficients[-1]
    return intercept,beta,list(fit.residuals)


def mean_reversion_half_life(spread: Sequence[float]) -> float:
    """Estimate AR(1) error-correction half-life in observations."""
    s=finite_series(spread,name="spread",min_length=10)
    lag=s[:-1]; delta=[s[i]-s[i-1] for i in range(1,len(s))]
    fit=ols([lag],delta,add_intercept=True)
    speed=fit.coefficients[1]
    if speed >= 0: return float("inf")
    if speed <= -1: return 0.0
    return -log(2.0)/log(1.0+speed)


def engle_granger(y: Sequence[float], x: Sequence[float], *, max_lags: int | None = None) -> CointegrationResult:
    intercept,beta,residuals=hedge_ratio(y,x)
    test=adf_test(residuals,max_lags=max_lags,regression="n")
    return CointegrationResult(intercept,beta,tuple(residuals),test.statistic,
                               test.p_value_approx,mean_reversion_half_life(residuals),len(residuals))


def rolling_hedge_ratio(y: Sequence[float], x: Sequence[float], window: int) -> list[float | None]:
    a,b=finite_series(y),finite_series(x)
    if len(a)!=len(b) or window<10 or window>len(a): raise ValueError("invalid series or window")
    out: list[float | None]=[None]*(window-1)
    for i in range(window-1,len(a)):
        _,beta,_=hedge_ratio(a[i-window+1:i+1],b[i-window+1:i+1])
        out.append(beta)
    return out


def spread_zscore(spread: Sequence[float], window: int) -> list[float | None]:
    s=finite_series(spread)
    if window<3 or window>len(s): raise ValueError("invalid window")
    out: list[float | None]=[None]*(window-1)
    for i in range(window-1,len(s)):
        w=s[i-window+1:i+1]; m=mean(w); v=variance(w)
        out.append(0.0 if v==0 else (s[i]-m)/v**0.5)
    return out


def pairs_positions(zscores: Sequence[float | None], entry: float = 2.0, exit: float = 0.5) -> list[int]:
    """Stateful spread position: +1 means long spread; -1 means short spread."""
    if entry<=0 or exit<0 or exit>=entry: raise ValueError("require entry > exit >= 0")
    state=0; result=[]
    for z in zscores:
        if z is None: result.append(0); continue
        if state==0:
            if z<=-entry: state=1
            elif z>=entry: state=-1
        elif abs(z)<=exit: state=0
        result.append(state)
    return result
