"""Cross-sectional alpha diagnostics: IC, decay, turnover and breadth."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series, mean, correlation, rank, stddev


def information_coefficient(signal: Sequence[float], forward_returns: Sequence[float], *, rank_based: bool = False) -> float:
    s=finite_series(signal,name="signal"); r=finite_series(forward_returns,name="forward_returns")
    if len(s)!=len(r) or len(s)<3: raise ValueError("inputs must have equal length >= 3")
    return correlation(rank(s),rank(r)) if rank_based else correlation(s,r)


def cross_sectional_ic(signal_by_date: Sequence[Sequence[float]], returns_by_date: Sequence[Sequence[float]],
                       *, rank_based: bool = True, min_assets: int = 5) -> list[float | None]:
    if len(signal_by_date)!=len(returns_by_date): raise ValueError("date dimensions must match")
    result=[]
    for sig,ret in zip(signal_by_date,returns_by_date):
        if len(sig)!=len(ret): raise ValueError("cross-section lengths must match")
        if len(sig)<min_assets: result.append(None)
        else: result.append(information_coefficient(sig,ret,rank_based=rank_based))
    return result


def ic_summary(ic_series: Sequence[float | None]) -> dict:
    x=[float(v) for v in ic_series if v is not None]
    if len(x)<2: raise ValueError("at least two valid IC observations are required")
    m=mean(x); sd=stddev(x); t=m/(sd/sqrt(len(x))) if sd else float("inf")
    return {"mean_ic":m,"ic_std":sd,"ic_ir":m/sd if sd else float("inf"),
            "t_statistic":t,"positive_fraction":sum(v>0 for v in x)/len(x),"observations":len(x)}


def alpha_decay(signal: Sequence[float], forward_return_horizons: Sequence[Sequence[float]],
                *, rank_based: bool = True) -> list[dict]:
    """Measure IC for each supplied forward horizon; caller defines alignment."""
    out=[]
    for h,returns in enumerate(forward_return_horizons,1):
        out.append({"horizon":h,"ic":information_coefficient(signal,returns,rank_based=rank_based)})
    return out


def factor_turnover(weights_by_date: Sequence[Sequence[float]]) -> list[float]:
    """One-way turnover = 0.5 * sum(abs(weight_t - weight_t-1))."""
    if len(weights_by_date)<2: return []
    result=[]
    for prev,curr in zip(weights_by_date,weights_by_date[1:]):
        if len(prev)!=len(curr): raise ValueError("weight cross-sections must have fixed asset count")
        result.append(0.5*sum(abs(float(a)-float(b)) for a,b in zip(prev,curr)))
    return result


def effective_breadth(weights: Sequence[float]) -> float:
    """Inverse Herfindahl concentration of absolute weights."""
    w=finite_series(weights,name="weights")
    gross=sum(abs(v) for v in w)
    if gross==0: return 0.0
    shares=[abs(v)/gross for v in w]
    return 1.0/sum(v*v for v in shares)


def cross_sectional_zscore(values: Sequence[float], *, clip: float | None = 5.0) -> list[float]:
    x=finite_series(values)
    m=mean(x); s=stddev(x,ddof=0)
    out=[0.0 if s==0 else (v-m)/s for v in x]
    if clip is not None:
        if clip<=0: raise ValueError("clip must be positive or None")
        out=[max(-clip,min(clip,v)) for v in out]
    return out


def neutralize_by_group(values: Sequence[float], groups: Sequence[str]) -> list[float]:
    x=finite_series(values)
    if len(x)!=len(groups): raise ValueError("group count must match observations")
    sums={}; counts={}
    for v,g in zip(x,groups): sums[g]=sums.get(g,0.0)+v; counts[g]=counts.get(g,0)+1
    return [v-sums[g]/counts[g] for v,g in zip(x,groups)]


def quantile_portfolio_returns(signals: Sequence[Sequence[float]],
                               returns: Sequence[Sequence[float]], quantiles: int = 5) -> list[dict]:
    """Equal-weight quantile returns by date with stable tie handling."""
    if quantiles<2 or len(signals)!=len(returns): raise ValueError("invalid quantile count or dates")
    output=[]
    for date,(s,r) in enumerate(zip(signals,returns)):
        sig=finite_series(s); ret=finite_series(r)
        if len(sig)!=len(ret) or len(sig)<quantiles: raise ValueError("invalid cross-section")
        order=sorted(range(len(sig)),key=lambda i:(sig[i],i))
        buckets=[[] for _ in range(quantiles)]
        for rank_idx,asset in enumerate(order): buckets[min(quantiles-1,rank_idx*quantiles//len(order))].append(ret[asset])
        means=[mean(bucket) for bucket in buckets]
        output.append({"date_index":date,"quantile_returns":tuple(means),"long_short":means[-1]-means[0]})
    return output


def factor_return_series(exposures: Sequence[Sequence[float]], asset_returns: Sequence[Sequence[float]]) -> list[float]:
    """Cross-sectional factor-mimicking returns using z-scored unit exposures."""
    if len(exposures)!=len(asset_returns): raise ValueError("date dimensions must match")
    out=[]
    for ex,ret in zip(exposures,asset_returns):
        e=finite_series(ex); r=finite_series(ret)
        if len(e)!=len(r): raise ValueError("cross-section lengths must match")
        z=cross_sectional_zscore(e)
        gross=sum(abs(v) for v in z)
        out.append(0.0 if gross==0 else sum(a*b for a,b in zip(z,r))/gross*2.0)
    return out
