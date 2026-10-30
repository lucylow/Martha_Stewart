"""Long-memory, scaling and roughness diagnostics for financial series."""
from __future__ import annotations
from math import log,sqrt
from typing import Sequence
from ..core import finite_series,mean,stddev


def hurst_rescaled_range(values: Sequence[float], block_sizes: Sequence[int] | None = None) -> dict:
    """Estimate H from log mean(R/S) against log block length.

    This is a descriptive scaling estimator. Short samples, trends and structural
    breaks can bias it substantially; it is not a standalone trading signal.
    """
    x=finite_series(values,name="values",min_length=30); n=len(x)
    sizes=list(block_sizes) if block_sizes is not None else sorted(set([max(8,n//16),max(10,n//8),max(12,n//4),max(16,n//2)]))
    points=[]
    for size in sizes:
        if size<8 or size>n: continue
        stats=[]
        for start in range(0,n-size+1,size):
            block=x[start:start+size]; m=mean(block); sd=stddev(block)
            if sd==0: continue
            cumulative=0.0; path=[]
            for value in block: cumulative+=value-m; path.append(cumulative)
            stats.append((max(path)-min(path))/sd)
        if stats and mean(stats)>0: points.append((log(size),log(mean(stats))))
    if len(points)<2: raise ValueError("need at least two usable block sizes")
    mx=mean([p[0] for p in points]); my=mean([p[1] for p in points])
    den=sum((a-mx)**2 for a,b in points)
    slope=sum((a-mx)*(b-my) for a,b in points)/den
    return {"hurst":slope,"points":tuple(points),"interpretation":"persistent" if slope>0.55 else "anti-persistent" if slope<0.45 else "near random-walk scaling"}


def detrended_fluctuation_analysis(values: Sequence[float], window_sizes: Sequence[int] | None = None) -> dict:
    """First-order DFA exponent from integrated demeaned series."""
    x=finite_series(values,name="values",min_length=30); mu=mean(x)
    integrated=[]; running=0.0
    for value in x: running+=value-mu; integrated.append(running)
    sizes=list(window_sizes) if window_sizes is not None else [8,12,16,24,32,48]
    points=[]
    for size in sizes:
        if size<4 or size>len(x)//2: continue
        fluctuations=[]
        for start in range(0,len(x)-size+1,size):
            block=integrated[start:start+size]; mt=(size-1)/2; mb=mean(block)
            den=sum((i-mt)**2 for i in range(size))
            slope=sum((i-mt)*(v-mb) for i,v in enumerate(block))/den
            intercept=mb-slope*mt
            fluctuations.extend((v-intercept-slope*i)**2 for i,v in enumerate(block))
        f=sqrt(mean(fluctuations))
        if f>0: points.append((log(size),log(f)))
    if len(points)<2: raise ValueError("need at least two valid DFA windows")
    mx=mean([a for a,b in points]); my=mean([b for a,b in points])
    slope=sum((a-mx)*(b-my) for a,b in points)/sum((a-mx)**2 for a,b in points)
    return {"dfa_exponent":slope,"points":tuple(points),"windows":len(points)}


def rescaled_range_path(values: Sequence[float], window: int) -> tuple[float,...]:
    x=finite_series(values)
    if window<2 or window>len(x): raise ValueError("invalid window")
    out=[]
    for i in range(window-1,len(x)):
        block=x[i-window+1:i+1]; m=mean(block); sd=stddev(block)
        if sd==0: out.append(0.0); continue
        c=0.0; path=[]
        for v in block: c+=v-m; path.append(c)
        out.append((max(path)-min(path))/sd)
    return tuple(out)


def autocorrelation_decay(values: Sequence[float], max_lag: int = 20) -> dict:
    from .forecasting import autocorrelation
    rho=autocorrelation(values,max_lag)
    half_life=next((i for i,v in enumerate(rho[1:],1) if abs(v)<=0.5),None)
    return {"autocorrelation":rho,"half_correlation_lag":half_life,
            "sum_positive_autocorrelation":sum(max(0.0,v) for v in rho[1:])}
