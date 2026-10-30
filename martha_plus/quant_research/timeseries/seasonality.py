"""Calendar and intraday seasonal-pattern diagnostics."""
from __future__ import annotations
from collections import defaultdict
from typing import Sequence
from ..core import finite_series,mean,stddev


def seasonal_profile(values: Sequence[float], labels: Sequence[str], *, min_count: int = 2) -> dict:
    x=finite_series(values,name="values")
    if len(x)!=len(labels) or min_count<1: raise ValueError("labels must align and min_count positive")
    groups=defaultdict(list)
    for value,label in zip(x,labels): groups[str(label)].append(value)
    profile={}
    overall=mean(x)
    for label,group in sorted(groups.items()):
        if len(group)>=min_count:
            profile[label]={"mean":mean(group),"std":stddev(group) if len(group)>1 else 0.0,
                            "count":len(group),"effect_vs_overall":mean(group)-overall}
    return {"overall_mean":overall,"profile":profile,"categories":len(profile)}


def weekday_profile(returns: Sequence[float], weekdays: Sequence[int]) -> dict:
    x=finite_series(returns)
    if len(x)!=len(weekdays) or any(v<0 or v>6 for v in weekdays): raise ValueError("weekdays must align and be 0..6")
    return seasonal_profile(x,[str(v) for v in weekdays])


def intraday_profile(returns: Sequence[float], time_buckets: Sequence[int],
                     *, measure: str = "absolute") -> dict:
    x=finite_series(returns)
    if len(x)!=len(time_buckets) or measure not in {"absolute","squared","signed"}: raise ValueError("invalid buckets or measure")
    if any(v<0 for v in time_buckets): raise ValueError("time buckets must be nonnegative")
    if measure=="absolute": values=[abs(v) for v in x]
    elif measure=="squared": values=[v*v for v in x]
    else: values=x
    return seasonal_profile(values,[str(v) for v in time_buckets])


def seasonal_adjust(values: Sequence[float], labels: Sequence[str], *, shrinkage: float = 0.0) -> dict:
    """Subtract shrunk group means to create a deseasonalized series."""
    x=finite_series(values)
    if len(x)!=len(labels) or not 0<=shrinkage<=1: raise ValueError("invalid labels or shrinkage")
    profile=seasonal_profile(x,labels)
    overall=profile["overall_mean"]
    effects={k:v["effect_vs_overall"]*(1-shrinkage) for k,v in profile["profile"].items()}
    adjusted=[v-effects.get(str(g),0.0) for v,g in zip(x,labels)]
    return {"adjusted":tuple(adjusted),"seasonal_effects":effects,"overall_mean":overall,"shrinkage":shrinkage}


def event_day_profile(returns: Sequence[float], event_offsets: Sequence[int]) -> dict:
    """Aggregate returns by relative event day (e.g., -2, -1, 0, +1)."""
    x=finite_series(returns)
    if len(x)!=len(event_offsets): raise ValueError("event offsets must match returns")
    groups=defaultdict(list)
    for value,offset in zip(x,event_offsets): groups[int(offset)].append(value)
    return {k:{"mean":mean(v),"count":len(v),"positive_fraction":sum(x>0 for x in v)/len(v)} for k,v in sorted(groups.items())}
