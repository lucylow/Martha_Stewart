"""Path-aware portfolio performance statistics and drawdown analytics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series, mean, stddev, variance, normal_ppf


def equity_curve(returns: Sequence[float], initial: float = 1.0) -> list[float]:
    r=finite_series(returns)
    if initial<=0 or any(v < -1 for v in r): raise ValueError("initial equity must be positive and returns >= -100%")
    value=initial; out=[]
    for v in r: value*=1+v; out.append(value)
    return out


def drawdown_series(returns: Sequence[float], initial: float = 1.0) -> list[float]:
    equity=equity_curve(returns,initial); peak=initial; out=[]
    for value in equity:
        peak=max(peak,value); out.append(value/peak-1.0)
    return out


def max_drawdown(returns: Sequence[float], initial: float = 1.0) -> dict:
    dd=drawdown_series(returns,initial)
    trough=min(range(len(dd)),key=dd.__getitem__); peak=max(range(trough+1),key=lambda i: equity_curve(returns,initial)[i]) if trough>=0 else 0
    recovery=next((i for i in range(trough+1,len(dd)) if dd[i]>=0),None)
    return {"max_drawdown":dd[trough],"peak_index":peak,"trough_index":trough,"recovery_index":recovery,
            "duration":None if recovery is None else recovery-peak}


def downside_deviation(returns: Sequence[float], target: float = 0.0, periods_per_year: float = 252.0) -> float:
    x=finite_series(returns)
    if periods_per_year<=0: raise ValueError("periods_per_year must be positive")
    downside=[min(0.0,v-target)**2 for v in x]
    return sqrt(sum(downside)/len(x))*sqrt(periods_per_year)


def sortino_ratio(returns: Sequence[float], target: float = 0.0, periods_per_year: float = 252.0) -> float:
    x=finite_series(returns,min_length=2)
    dd=downside_deviation(x,target,periods_per_year)
    excess=(mean(x)-target)*periods_per_year
    return excess/dd if dd>0 else float("inf")


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0, periods_per_year: float = 252.0) -> float:
    x=finite_series(returns,min_length=2)
    if periods_per_year<=0: raise ValueError("periods_per_year must be positive")
    excess=[v-risk_free for v in x]; sd=stddev(excess)
    return mean(excess)/sd*sqrt(periods_per_year) if sd>0 else float("inf")


def calmar_ratio(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    x=finite_series(returns,min_length=2); mdd=abs(max_drawdown(x)["max_drawdown"])
    annual=(__import__('math').prod(1+v for v in x)**(periods_per_year/len(x))-1) if all(v>-1 for v in x) else -1.0
    return annual/mdd if mdd>0 else float("inf")


def ulcer_index(returns: Sequence[float]) -> float:
    dd=drawdown_series(returns)
    return sqrt(sum(v*v for v in dd)/len(dd))


def omega_ratio(returns: Sequence[float], threshold: float = 0.0) -> float:
    x=finite_series(returns)
    gains=sum(max(0.0,v-threshold) for v in x)
    losses=sum(max(0.0,threshold-v) for v in x)
    return gains/losses if losses>0 else float("inf")


def gain_to_pain(returns: Sequence[float]) -> float:
    x=finite_series(returns)
    return sum(x)/sum(abs(v) for v in x if v<0) if any(v<0 for v in x) else float("inf")


def information_ratio(portfolio: Sequence[float], benchmark: Sequence[float], periods_per_year: float = 252.0) -> float:
    a=finite_series(portfolio); b=finite_series(benchmark)
    if len(a)!=len(b): raise ValueError("series lengths must match")
    active=[x-y for x,y in zip(a,b)]; te=stddev(active)
    return mean(active)/te*sqrt(periods_per_year) if te>0 else float("inf")


def annualized_return(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    x=finite_series(returns)
    if periods_per_year<=0 or any(v<=-1 for v in x): raise ValueError("invalid period or return")
    return __import__('math').exp(sum(__import__('math').log1p(v) for v in x)*periods_per_year/len(x))-1.0


def rolling_metric(returns: Sequence[float], window: int, metric: str = "sharpe",
                   periods_per_year: float = 252.0) -> list[float | None]:
    x=finite_series(returns)
    if window<2 or window>len(x): raise ValueError("invalid rolling window")
    functions={"sharpe":lambda z:sharpe_ratio(z,periods_per_year=periods_per_year),
               "sortino":lambda z:sortino_ratio(z,periods_per_year=periods_per_year),
               "volatility":lambda z:stddev(z)*sqrt(periods_per_year),
               "return":lambda z:annualized_return(z,periods_per_year)}
    if metric not in functions: raise ValueError("metric must be sharpe, sortino, volatility or return")
    return [None]*(window-1)+[functions[metric](x[i-window+1:i+1]) for i in range(window-1,len(x))]


def rolling_beta(portfolio: Sequence[float], benchmark: Sequence[float], window: int) -> list[float | None]:
    from ..core import covariance
    a=finite_series(portfolio); b=finite_series(benchmark)
    if len(a)!=len(b) or window<3 or window>len(a): raise ValueError("invalid window or series")
    result=[None]*(window-1)
    for i in range(window-1,len(a)):
        x=a[i-window+1:i+1]; y=b[i-window+1:i+1]
        v=variance(y)
        result.append(covariance(x,y)/v if v>0 else 0.0)
    return result


def hit_rate(returns: Sequence[float]) -> float:
    x=finite_series(returns)
    return sum(v>0 for v in x)/len(x)


def tail_ratio(returns: Sequence[float], probability: float = 0.95) -> float:
    from ..core import quantile
    x=finite_series(returns)
    if not 0.5<probability<1: raise ValueError("probability must be in (0.5,1)")
    upper=quantile(x,probability); lower=abs(quantile(x,1-probability))
    return upper/lower if lower>0 else float("inf")


def summary(returns: Sequence[float], periods_per_year: float = 252.0) -> dict:
    x=finite_series(returns,min_length=2); dd=max_drawdown(x)
    return {"observations":len(x),"total_return":equity_curve(x)[-1]-1,
            "annualized_return":annualized_return(x,periods_per_year),
            "annualized_volatility":stddev(x)*sqrt(periods_per_year),
            "sharpe":sharpe_ratio(x,periods_per_year=periods_per_year),
            "sortino":sortino_ratio(x,periods_per_year=periods_per_year),
            "max_drawdown":dd["max_drawdown"],"calmar":calmar_ratio(x,periods_per_year),
            "ulcer_index":ulcer_index(x),"hit_rate":hit_rate(x)}
