"""GARCH(1,1) Gaussian quasi-maximum-likelihood estimation."""
from __future__ import annotations
from dataclasses import dataclass
from math import log, pi
from typing import Sequence
from ..core import finite_series, mean, variance

@dataclass(frozen=True)
class GARCHResult:
    omega: float
    alpha: float
    beta: float
    unconditional_variance: float
    conditional_variance: tuple[float,...]
    standardized_residuals: tuple[float,...]
    log_likelihood: float
    persistence: float
    converged: bool
    iterations: int


def _variance_path(x, omega, alpha, beta):
    v0=variance(x)
    h=max(v0,1e-12); out=[]
    for r in x:
        out.append(h)
        h=omega+alpha*r*r+beta*h
    return out


def _loglik(x, omega, alpha, beta):
    if omega<=0 or alpha<0 or beta<0 or alpha+beta>=0.9999: return float("-inf")
    h=_variance_path(x,omega,alpha,beta)
    return -0.5*sum(log(2*pi)+log(max(v,1e-15))+r*r/max(v,1e-15) for r,v in zip(x,h))


def fit_garch11(returns: Sequence[float], *, mean_adjust: bool = True,
                max_iter: int = 80, tolerance: float = 1e-6) -> GARCHResult:
    """Fit Gaussian QMLE through deterministic coarse-to-fine parameter search.

    The estimator is intentionally transparent, not a replacement for
    constrained numerical optimization in institutional risk production.
    """
    x=finite_series(returns,name="returns",min_length=30)
    if max_iter<1 or tolerance<=0: raise ValueError("invalid optimizer controls")
    if mean_adjust:
        mu=mean(x); x=[v-mu for v in x]
    var=max(variance(x),1e-10)
    best=(float("-inf"),0.05*var,0.08,0.88)
    # Search a grid respecting covariance-stationarity constraints.
    for alpha in (0.02,0.05,0.08,0.12,0.16,0.20):
        for beta in (0.60,0.70,0.78,0.84,0.88,0.92,0.95,0.97):
            if alpha+beta>=0.995: continue
            omega=max(1e-12,var*(1-alpha-beta))
            ll=_loglik(x,omega,alpha,beta)
            if ll>best[0]: best=(ll,omega,alpha,beta)
    ll,omega,alpha,beta=best; step_a=0.04; step_b=0.04; converged=False
    for iteration in range(1,max_iter+1):
        improved=False
        for da,db in ((step_a,0),(-step_a,0),(0,step_b),(0,-step_b),(step_a,step_b),(-step_a,step_b),(step_a,-step_b),(-step_a,-step_b)):
            a=max(0.001,min(0.35,alpha+da)); b=max(0.0,min(0.995-a,beta+db))
            w=max(1e-12,var*(1-a-b)); candidate=_loglik(x,w,a,b)
            if candidate>ll:
                ll,omega,alpha,beta=candidate,w,a,b; improved=True
        if not improved:
            step_a*=0.5; step_b*=0.5
        if max(step_a,step_b)<tolerance:
            converged=True; break
    h=_variance_path(x,omega,alpha,beta)
    standardized=[r/(v**0.5) for r,v in zip(x,h)]
    return GARCHResult(omega,alpha,beta,omega/(1-alpha-beta),tuple(h),tuple(standardized),ll,alpha+beta,converged,iteration)


def forecast_variance(result: GARCHResult, horizon: int = 1) -> tuple[float,...]:
    if horizon<1: raise ValueError("horizon must be positive")
    current=result.conditional_variance[-1]
    one=result.omega+result.alpha*(result.standardized_residuals[-1]**2*current)+result.beta*current
    forecasts=[one]
    for _ in range(1,horizon): forecasts.append(result.omega+(result.alpha+result.beta)*forecasts[-1])
    return tuple(forecasts)


def garch_value_at_risk(result: GARCHResult, confidence: float = 0.99, horizon: int = 1) -> float:
    from ..core import normal_ppf
    if not 0.5<confidence<1: raise ValueError("confidence must be in (0.5,1)")
    variance_h=sum(forecast_variance(result,horizon))
    return normal_ppf(confidence)*variance_h**0.5
