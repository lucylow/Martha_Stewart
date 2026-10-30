"""Panel-data estimators and Fama-MacBeth cross-sectional regressions."""
from __future__ import annotations
from collections import defaultdict
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean
from ..regression import ols


def within_transform(values: Sequence[float], entities: Sequence[str],
                     times: Sequence[str] | None = None, *, two_way: bool = False) -> list[float]:
    """Entity demeaning, optionally followed by two-way fixed-effect transform."""
    x=finite_series(values)
    if len(x)!=len(entities) or (times is not None and len(times)!=len(x)): raise ValueError("panel identifiers must align")
    groups=defaultdict(list)
    for i,g in enumerate(entities): groups[g].append(i)
    emean={g:mean([x[i] for i in idx]) for g,idx in groups.items()}
    if not two_way:
        return [v-emean[g] for v,g in zip(x,entities)]
    if times is None: raise ValueError("time identifiers required for two-way effects")
    tgroups=defaultdict(list)
    for i,g in enumerate(times): tgroups[g].append(i)
    tmean={g:mean([x[i] for i in idx]) for g,idx in tgroups.items()}
    grand=mean(x)
    return [v-emean[e]-tmean[t]+grand for v,e,t in zip(x,entities,times)]


def fixed_effects_regression(features: Sequence[Sequence[float]], target: Sequence[float],
                             entities: Sequence[str], times: Sequence[str] | None = None,
                             *, two_way: bool = False) -> dict:
    y=finite_series(target); cols=[finite_series(c) for c in features]
    if any(len(c)!=len(y) for c in cols): raise ValueError("feature lengths must match target")
    yw=within_transform(y,entities,times,two_way=two_way)
    xw=[within_transform(c,entities,times,two_way=two_way) for c in cols]
    fit=ols(xw,yw,add_intercept=False)
    return {"coefficients":fit.coefficients,"standard_errors":fit.standard_errors,
            "t_statistics":fit.t_statistics,"r_squared_within":fit.r_squared,
            "residuals":fit.residuals,"fitted_within":fit.fitted,"nobs":fit.nobs,
            "entities":len(set(entities)),"time_effects":two_way}


def fama_macbeth(features_by_time: Sequence[Sequence[Sequence[float]]],
                 returns_by_time: Sequence[Sequence[float]], *, add_intercept: bool = True) -> dict:
    """Two-pass Fama-MacBeth: cross-sectional slopes, then time-series mean."""
    if len(features_by_time)!=len(returns_by_time) or len(returns_by_time)<3: raise ValueError("need aligned dates >= 3")
    estimates=[]; n_assets=[]
    for features,ret in zip(features_by_time,returns_by_time):
        y=finite_series(ret,min_length=3)
        if any(len(c)!=len(y) for c in features): raise ValueError("cross-section lengths mismatch")
        fit=ols(features,y,add_intercept=add_intercept)
        estimates.append(fit.coefficients); n_assets.append(len(y))
    p=len(estimates[0])
    if any(len(row)!=p for row in estimates): raise ValueError("number of factors must remain fixed")
    means=[mean([row[j] for row in estimates]) for j in range(p)]
    ses=[sqrt(sum((row[j]-means[j])**2 for row in estimates)/(len(estimates)-1)/len(estimates)) for j in range(p)]
    t=[means[j]/ses[j] if ses[j]>0 else float("inf") for j in range(p)]
    return {"mean_coefficients":tuple(means),"standard_errors":tuple(ses),"t_statistics":tuple(t),
            "coefficient_path":tuple(tuple(r) for r in estimates),"periods":len(estimates),
            "average_assets":mean(n_assets)}


def cluster_means(values: Sequence[float], cluster_ids: Sequence[str]) -> dict:
    x=finite_series(values)
    if len(x)!=len(cluster_ids): raise ValueError("cluster IDs must match values")
    groups=defaultdict(list)
    for value,key in zip(x,cluster_ids): groups[key].append(value)
    return {key:{"mean":mean(vals),"count":len(vals)} for key,vals in groups.items()}


def panel_balance(entities: Sequence[str], times: Sequence[str]) -> dict:
    if len(entities)!=len(times) or not entities: raise ValueError("identifiers must be aligned and non-empty")
    table=defaultdict(set)
    for entity,time in zip(entities,times): table[entity].add(time)
    counts=[len(v) for v in table.values()]
    return {"entities":len(table),"periods":len(set(times)),"balanced":len(set(counts))==1,
            "min_periods_per_entity":min(counts),"max_periods_per_entity":max(counts),
            "missing_entity_periods":len(table)*len(set(times))-len(set(zip(entities,times)))}
