"""Market-model event studies and cumulative abnormal-return analysis."""
from __future__ import annotations
from typing import Sequence
from ..core import finite_series,mean,stddev
from ..regression import ols
from ..statistics import chi_square_survival


def market_model_abnormal_returns(asset_returns: Sequence[float], market_returns: Sequence[float],
                                  estimation_start: int, estimation_end: int,
                                  event_start: int, event_end: int) -> dict:
    """Fit market model on estimation window and calculate event-window AR/CAR."""
    asset=finite_series(asset_returns,name="asset_returns")
    market=finite_series(market_returns,name="market_returns")
    if len(asset)!=len(market): raise ValueError("asset and market lengths must match")
    n=len(asset)
    if not(0<=estimation_start<estimation_end<=n and 0<=event_start<event_end<=n): raise ValueError("invalid windows")
    if not(estimation_end<=event_start or event_end<=estimation_start): raise ValueError("estimation and event windows must not overlap")
    fit=ols([market[estimation_start:estimation_end]],asset[estimation_start:estimation_end])
    alpha,beta=fit.coefficients
    expected=[alpha+beta*market[i] for i in range(event_start,event_end)]
    abnormal=[asset[i]-expected[j] for j,i in enumerate(range(event_start,event_end))]
    return {"alpha":alpha,"beta":beta,"abnormal_returns":tuple(abnormal),
            "cumulative_abnormal_return":sum(abnormal),"event_start":event_start,"event_end":event_end,
            "estimation_r_squared":fit.r_squared,"estimation_residual_std":stddev(fit.residuals)}


def average_abnormal_returns(event_panels: Sequence[Sequence[float]]) -> dict:
    """Cross-sectional average abnormal return (AAR) and cumulative AAR."""
    rows=[finite_series(row,name="event panel") for row in event_panels]
    if not rows or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("event panel must be rectangular")
    width=len(rows[0]); means=[mean([r[t] for r in rows]) for t in range(width)]
    ses=[stddev([r[t] for r in rows])/len(rows)**0.5 if len(rows)>1 else 0.0 for t in range(width)]
    tstats=[means[i]/ses[i] if ses[i]>0 else float("inf") for i in range(width)]
    cumulative=[]; running=0.0
    for v in means: running+=v; cumulative.append(running)
    return {"aar":tuple(means),"standard_errors":tuple(ses),"t_statistics":tuple(tstats),
            "caar":tuple(cumulative),"events":len(rows)}


def standardized_car(car: float, estimation_residual_std: float,
                     event_length: int, estimation_length: int) -> float:
    if estimation_residual_std<0 or event_length<1 or estimation_length<2: raise ValueError("invalid standardization inputs")
    scale=estimation_residual_std*(event_length+event_length**2/estimation_length)**0.5
    return car/scale if scale>0 else 0.0


def event_window_returns(prices: Sequence[float], event_index: int,
                         before: int, after: int) -> tuple[float,...]:
    from ..core import log
    x=finite_series(prices,name="prices",min_length=2)
    if any(v<=0 for v in x) or before<0 or after<0 or event_index<before or event_index+after>=len(x):
        raise ValueError("invalid event index or price window")
    returns=[log(x[i]/x[i-1]) for i in range(1,len(x))]
    start=event_index-before; end=event_index+after
    return tuple(returns[start:end+1])


def event_return_significance(abnormal_returns: Sequence[float]) -> dict:
    from ..core import normal_cdf
    x=finite_series(abnormal_returns,name="abnormal_returns",min_length=2)
    sd=stddev(x)
    se=sd/(len(x)**0.5)
    car=sum(x)
    z=car/(sd*(len(x)**0.5)) if sd>0 else 0.0
    p=2*(1-normal_cdf(abs(z)))
    return {"car":car,"standard_error":se,"z_statistic":z,"p_value_approx":p,
            "observations":len(x),"assumption":"independent, homoskedastic abnormal returns"}
