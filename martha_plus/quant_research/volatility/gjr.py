"""Asymmetric GJR-GARCH(1,1) model for leverage-effect research."""
from __future__ import annotations
from dataclasses import dataclass
from math import log,pi
from typing import Sequence
from ..core import finite_series,mean,variance

@dataclass(frozen=True)
class GJRResult:
    omega: float
    alpha: float
    gamma: float
    beta: float
    conditional_variance: tuple[float,...]
    log_likelihood: float
    persistence: float
    leverage_effect: bool
    converged: bool
    iterations: int


def _path(x,omega,alpha,gamma,beta):
    h=max(variance(x),1e-10); out=[]
    for r in x:
        out.append(h)
        h=omega+alpha*r*r+gamma*(r<0)*r*r+beta*h
    return out


def _ll(x,omega,alpha,gamma,beta):
    persistence=alpha+0.5*gamma+beta
    if omega<=0 or min(alpha,gamma,beta)<0 or persistence>=0.999: return float("-inf")
    h=_path(x,omega,alpha,gamma,beta)
    return -0.5*sum(log(2*pi)+log(max(v,1e-15))+r*r/max(v,1e-15) for r,v in zip(x,h))


def fit_gjr_garch(returns: Sequence[float], *, mean_adjust: bool = True,
                  max_iter: int = 100, tolerance: float = 1e-6) -> GJRResult:
    """Constrained grid/local search Gaussian QMLE for GJR-GARCH(1,1)."""
    x=finite_series(returns,name="returns",min_length=30)
    if mean_adjust: x=[v-mean(x) for v in x]
    var=max(variance(x),1e-10); best=(float("-inf"),0.05*var,0.05,0.05,0.85)
    for alpha in (0.02,0.05,0.08,0.12):
        for gamma in (0.0,0.05,0.10,0.15):
            for beta in (0.65,0.75,0.82,0.88,0.93):
                if alpha+0.5*gamma+beta>=0.995: continue
                omega=var*(1-alpha-0.5*gamma-beta)
                ll=_ll(x,omega,alpha,gamma,beta)
                if ll>best[0]: best=(ll,omega,alpha,gamma,beta)
    ll,omega,alpha,gamma,beta=best; step=0.025; converged=False
    for iteration in range(1,max_iter+1):
        improved=False
        for idx in range(3):
            for direction in (-1,1):
                pars=[alpha,gamma,beta]; pars[idx]=max(0.0,pars[idx]+direction*step)
                a,g,b=pars
                if a+0.5*g+b>=0.998: continue
                w=max(1e-12,var*(1-a-0.5*g-b)); candidate=_ll(x,w,a,g,b)
                if candidate>ll: ll,omega,alpha,gamma,beta=candidate,w,a,g,b; improved=True
        if not improved: step*=0.5
        if step<tolerance: converged=True; break
    return GJRResult(omega,alpha,gamma,beta,tuple(_path(x,omega,alpha,gamma,beta)),ll,
                     alpha+0.5*gamma+beta,gamma>1e-8,converged,iteration)


def forecast_gjr(result: GJRResult, last_return: float, horizon: int = 1) -> tuple[float,...]:
    if horizon<1: raise ValueError("horizon must be positive")
    h=result.conditional_variance[-1]
    h=result.omega+result.alpha*last_return**2+result.gamma*(last_return<0)*last_return**2+result.beta*h
    forecasts=[h]
    persistence=result.persistence
    unconditional=result.omega/max(1e-12,1-persistence)
    for _ in range(1,horizon): forecasts.append(unconditional+(persistence)*(forecasts[-1]-unconditional))
    return tuple(forecasts)
