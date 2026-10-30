"""Autocorrelation and periodogram tools for cyclic market behaviour."""
from __future__ import annotations
from math import cos,pi,sin,sqrt
from typing import Sequence
from ..core import finite_series,mean
from ..linalg import solve
from .forecasting import autocorrelation


def periodogram(values: Sequence[float], *, detrend: bool = True, taper: bool = True) -> list[dict]:
    """Naive DFT periodogram for real series; excludes zero-frequency DC bin."""
    x=finite_series(values,name="values",min_length=4); n=len(x)
    if detrend:
        mu=mean(x); x=[v-mu for v in x]
    if taper:
        x=[v*(0.5-0.5*cos(2*pi*i/(n-1))) for i,v in enumerate(x)]
    out=[]
    for k in range(1,n//2+1):
        real=sum(x[t]*cos(2*pi*k*t/n) for t in range(n))
        imag=-sum(x[t]*sin(2*pi*k*t/n) for t in range(n))
        power=(real*real+imag*imag)/n
        out.append({"frequency":k/n,"period":n/k,"power":power})
    return out


def dominant_frequencies(values: Sequence[float], count: int = 5) -> list[dict]:
    if count<1: raise ValueError("count must be positive")
    spec=periodogram(values)
    return sorted(spec,key=lambda row:row["power"],reverse=True)[:count]


def partial_autocorrelation(values: Sequence[float], max_lag: int) -> tuple[float,...]:
    """PACF via Durbin-Levinson recursion from sample autocorrelations."""
    rho=autocorrelation(values,max_lag)
    if max_lag<0: raise ValueError("max_lag must be non-negative")
    pacf=[1.0]; phi=[]; variance=1.0
    for k in range(1,max_lag+1):
        if k==1: coeff=rho[1]
        else:
            numerator=rho[k]-sum(phi[j-1]*rho[k-j] for j in range(1,k))
            coeff=numerator/max(variance,1e-15)
        old=phi[:]
        phi=[old[j-1]-coeff*old[k-j-1] for j in range(1,k)] + [coeff] if k>1 else [coeff]
        variance*=max(1e-12,1-coeff*coeff)
        pacf.append(coeff)
    return tuple(pacf)


def acf_confidence_band(observations: int, confidence_z: float = 1.96) -> float:
    if observations<1 or confidence_z<=0: raise ValueError("invalid observations or z")
    return confidence_z/sqrt(observations)


def spectral_entropy(values: Sequence[float]) -> float:
    """Normalized Shannon entropy of the non-DC periodogram."""
    spec=periodogram(values); power=[max(0.0,r["power"]) for r in spec]; total=sum(power)
    if total<=0: return 0.0
    probs=[v/total for v in power if v>0]
    entropy=-sum(p*__import__('math').log(p) for p in probs)
    import math
    return entropy/math.log(len(probs)) if len(probs)>1 else 0.0


def rolling_autocorrelation(values: Sequence[float], window: int, max_lag: int = 5) -> list[tuple[float,...] | None]:
    x=finite_series(values)
    if window<max_lag+3 or window>len(x): raise ValueError("window must exceed max_lag+2")
    out=[None]*(window-1)
    for i in range(window-1,len(x)): out.append(autocorrelation(x[i-window+1:i+1],max_lag))
    return out
