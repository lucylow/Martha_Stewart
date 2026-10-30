"""Sequential change detection and mean-shift segmentation."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,stddev


def cusum(values: Sequence[float], reference: float = 0.0, allowance: float = 0.5,
          threshold: float = 5.0, scale: float | None = None) -> dict:
    """Two-sided standardized CUSUM detector for mean shifts."""
    x=finite_series(values,name="values")
    if allowance<0 or threshold<=0: raise ValueError("invalid detector thresholds")
    sigma=stddev(x) if scale is None else scale
    if sigma<=0: raise ValueError("scale must be positive")
    pos=neg=0.0; up=[]; down=[]; signals=[]
    for i,v in enumerate(x):
        z=(v-reference)/sigma
        pos=max(0.0,pos+z-allowance); neg=min(0.0,neg+z+allowance)
        up.append(pos); down.append(neg)
        if pos>=threshold: signals.append({"index":i,"direction":"up","statistic":pos}); pos=0.0
        elif neg<=-threshold: signals.append({"index":i,"direction":"down","statistic":neg}); neg=0.0
    return {"positive_path":tuple(up),"negative_path":tuple(down),"signals":tuple(signals),"scale":sigma}


def page_hinkley(values: Sequence[float], delta: float = 0.005, threshold: float = 50.0,
                 forgetting: float = 1.0) -> dict:
    """Page-Hinkley mean-shift detector with optional forgetting of the minimum."""
    x=finite_series(values,name="values")
    if delta<0 or threshold<=0 or not 0<forgetting<=1: raise ValueError("invalid Page-Hinkley parameters")
    avg=0.0; cumulative=0.0; minimum=0.0; alarms=[]; path=[]
    for i,v in enumerate(x,1):
        avg+=(v-avg)/i
        cumulative=forgetting*cumulative+(v-avg-delta)
        minimum=min(minimum,cumulative)
        statistic=cumulative-minimum; path.append(statistic)
        if statistic>threshold:
            alarms.append(i-1); cumulative=0.0; minimum=0.0
    return {"alarms":tuple(alarms),"statistic_path":tuple(path)}


def binary_segmentation(values: Sequence[float], *, min_segment: int = 20,
                        penalty: float = 3.0, max_changes: int = 10) -> dict:
    """Greedy mean-shift segmentation using an SSE improvement criterion."""
    x=finite_series(values,name="values",min_length=10)
    if min_segment<2 or penalty<0 or max_changes<0 or 2*min_segment>len(x): raise ValueError("invalid segmentation controls")
    segments=[(0,len(x))]; changes=[]
    def sse(a,b):
        m=mean(x[a:b]); return sum((v-m)**2 for v in x[a:b])
    while len(changes)<max_changes:
        best=None
        for idx,(a,b) in enumerate(segments):
            if b-a<2*min_segment: continue
            base=sse(a,b)
            for split in range(a+min_segment,b-min_segment+1):
                gain=base-sse(a,split)-sse(split,b)
                if best is None or gain>best[0]: best=(gain,idx,split)
        if best is None or best[0]<=penalty*stddev(x)**2: break
        gain,idx,split=best; a,b=segments.pop(idx); segments[idx:idx]=[(a,split),(split,b)]; changes.append(split)
    segments.sort()
    return {"change_points":tuple(sorted(changes)),"segments":tuple(segments),
            "segment_means":tuple(mean(x[a:b]) for a,b in segments),"segment_count":len(segments)}


def rolling_mean_shift(values: Sequence[float], window: int = 20) -> list[float | None]:
    x=finite_series(values)
    if window<2 or 2*window>len(x): raise ValueError("window must fit twice within sample")
    out=[None]*(2*window-1)
    for i in range(2*window-1,len(x)):
        out.append(mean(x[i-window+1:i+1])-mean(x[i-2*window+1:i-window+1]))
    return out
