"""Signal denoising and frequency-domain-inspired research filters."""
from __future__ import annotations
from math import cos, pi
from typing import Sequence
from ..core import finite_series


def low_pass(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    """First-order causal low-pass filter (no future observations used)."""
    x=finite_series(values)
    if not 0<alpha<=1: raise ValueError("alpha must be in (0,1]")
    out=[x[0]]
    for v in x[1:]: out.append(alpha*v+(1-alpha)*out[-1])
    return out


def high_pass(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    x=finite_series(values)
    smooth=low_pass(x,alpha)
    return [a-b for a,b in zip(x,smooth)]


def triangular_kernel(window: int) -> list[float]:
    if window<2: raise ValueError("window must be at least two")
    center=(window-1)/2
    weights=[1.0-abs(i-center)/(center+1.0) for i in range(window)]
    total=sum(weights)
    return [w/total for w in weights]


def kernel_smooth(values: Sequence[float], kernel: Sequence[float], *, center: bool = False) -> list[float | None]:
    x=finite_series(values); w=finite_series(kernel,name="kernel")
    if any(v<0 for v in w) or sum(w)<=0: raise ValueError("kernel must be non-negative with positive sum")
    total=sum(w); w=[v/total for v in w]
    if center:
        left=(len(w)-1)//2; right=len(w)-left-1
        out=[None]*left
        for i in range(left,len(x)-right): out.append(sum(w[j]*x[i-left+j] for j in range(len(w))))
        out += [None]*right
        return out
    out=[None]*(len(w)-1)
    for i in range(len(w)-1,len(x)):
        out.append(sum(w[j]*x[i-len(w)+1+j] for j in range(len(w))))
    return out


def detrend(values: Sequence[float]) -> list[float]:
    x=finite_series(values)
    n=len(x); mt=(n-1)/2; mx=sum(x)/n
    denom=sum((i-mt)**2 for i in range(n))
    slope=sum((i-mt)*(v-mx) for i,v in enumerate(x))/denom if denom else 0.0
    intercept=mx-slope*mt
    return [v-(intercept+slope*i) for i,v in enumerate(x)]


def one_sided_hann(window: int) -> list[float]:
    """Causal Hann-like taper normalized to sum to one."""
    if window<2: raise ValueError("window must be at least two")
    w=[0.5-0.5*cos(2*pi*i/(2*window-1)) for i in range(window)]
    total=sum(w)
    return [v/total for v in w]


def causal_local_level(values: Sequence[float], process_noise: float = 1e-4,
                       observation_noise: float = 1e-2) -> dict:
    """Scalar local-level Kalman smoother for online trend extraction."""
    x=finite_series(values)
    if process_noise<0 or observation_noise<=0: raise ValueError("invalid noise variances")
    level=x[0]; p=1.0; levels=[]; variances=[]
    for v in x:
        p+=process_noise
        gain=p/(p+observation_noise)
        level+=gain*(v-level)
        p=(1-gain)*p
        levels.append(level); variances.append(p)
    return {"level":tuple(levels),"variance":tuple(variances),"residual":tuple(a-b for a,b in zip(x,levels))}
