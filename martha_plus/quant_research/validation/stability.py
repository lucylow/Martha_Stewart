"""Out-of-sample stability diagnostics and block-bootstrap inference."""
from __future__ import annotations
from math import sqrt
import random
from typing import Sequence
from ..core import finite_series,mean,quantile
from ..performance.metrics import sharpe_ratio


def moving_block_bootstrap(values: Sequence[float], samples: int = 1000, block_size: int = 5,
                           *, seed: int = 7) -> list[tuple[float,...]]:
    x=finite_series(values,name="values")
    if samples<1 or block_size<1 or block_size>len(x): raise ValueError("invalid bootstrap controls")
    rng=random.Random(seed); n=len(x); blocks=(n+block_size-1)//block_size; out=[]
    for _ in range(samples):
        draw=[]
        for _b in range(blocks):
            start=rng.randrange(n)
            draw.extend(x[(start+j)%n] for j in range(block_size))
        out.append(tuple(draw[:n]))
    return out


def bootstrap_sharpe(returns: Sequence[float], samples: int = 1000, block_size: int = 5,
                     confidence: float = 0.95, periods_per_year: float = 252.0,
                     seed: int = 7) -> dict:
    if not 0<confidence<1: raise ValueError("confidence must be in (0,1)")
    paths=moving_block_bootstrap(returns,samples,block_size,seed=seed)
    values=[sharpe_ratio(path,periods_per_year=periods_per_year) for path in paths]
    tail=(1-confidence)/2
    return {"point_estimate":sharpe_ratio(returns,periods_per_year=periods_per_year),
            "lower":quantile(values,tail),"upper":quantile(values,1-tail),
            "median":quantile(values,0.5),"confidence":confidence,"samples":samples,"block_size":block_size}


def rolling_parameter_stability(parameter_path: Sequence[Sequence[float]],
                                 names: Sequence[str] | None = None) -> dict:
    rows=[finite_series(row,name="parameter row") for row in parameter_path]
    if len(rows)<2 or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("parameter path must be rectangular with >=2 rows")
    labels=list(names) if names is not None else [f"parameter_{i}" for i in range(len(rows[0]))]
    if len(labels)!=len(rows[0]): raise ValueError("names length mismatch")
    summaries=[]
    for j,name in enumerate(labels):
        values=[r[j] for r in rows]; m=mean(values); sd=sqrt(sum((v-m)**2 for v in values)/(len(values)-1))
        summaries.append({"name":name,"mean":m,"std":sd,"coefficient_of_variation":sd/abs(m) if m!=0 else float("inf"),
                          "min":min(values),"max":max(values),"range":max(values)-min(values)})
    return {"parameters":tuple(summaries),"windows":len(rows)}


def regime_performance(returns: Sequence[float], regimes: Sequence[int]) -> dict:
    x=finite_series(returns)
    if len(x)!=len(regimes): raise ValueError("regime labels must match returns")
    grouped={}
    for r,g in zip(x,regimes): grouped.setdefault(int(g),[]).append(r)
    return {g:{"observations":len(v),"mean_return":mean(v),"volatility":sqrt(sum((a-mean(v))**2 for a in v)/max(1,len(v)-1)),
               "hit_rate":sum(a>0 for a in v)/len(v)} for g,v in sorted(grouped.items())}


def probability_of_loss(returns: Sequence[float], block_size: int = 5,
                        samples: int = 1000, seed: int = 7) -> dict:
    paths=moving_block_bootstrap(returns,samples,block_size,seed=seed)
    totals=[__import__('math').prod(1+v for v in path)-1 for path in paths]
    return {"probability_negative_total_return":sum(v<0 for v in totals)/len(totals),
            "median_total_return":quantile(totals,0.5),"fifth_percentile":quantile(totals,0.05),
            "samples":samples,"block_size":block_size}
