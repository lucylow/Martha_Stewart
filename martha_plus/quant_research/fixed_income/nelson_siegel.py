"""Nelson-Siegel curve fitting and key-rate risk utilities."""
from __future__ import annotations
from math import exp
from typing import Sequence
from ..core import finite_series, mean
from ..regression import ols


def nelson_siegel_loadings(maturities: Sequence[float], tau: float) -> list[tuple[float,float,float]]:
    t=finite_series(maturities,name="maturities")
    if tau<=0 or any(v<0 for v in t): raise ValueError("tau must be positive and maturities non-negative")
    rows=[]
    for m in t:
        z=m/tau
        if z<1e-8: l1,l2=1.0,0.0
        else:
            l1=(1-exp(-z))/z
            l2=l1-exp(-z)
        rows.append((1.0,l1,l2))
    return rows


def nelson_siegel_rate(maturity: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    x=nelson_siegel_loadings([maturity],tau)[0]
    return beta0*x[0]+beta1*x[1]+beta2*x[2]


def fit_nelson_siegel(maturities: Sequence[float], yields: Sequence[float],
                      tau_grid: Sequence[float] | None = None) -> dict:
    """Profile-grid fit: solve betas by OLS for each candidate tau."""
    t=finite_series(maturities,name="maturities",min_length=5); y=finite_series(yields,name="yields",min_length=5)
    if len(t)!=len(y) or any(v<0 for v in t): raise ValueError("maturities and yields must align")
    taus=[float(v) for v in (tau_grid or [0.25,0.5,0.75,1,1.5,2,3,5,7,10,15])]
    if not taus or any(v<=0 for v in taus): raise ValueError("tau candidates must be positive")
    best=None
    for tau in taus:
        loading=nelson_siegel_loadings(t,tau)
        fit=ols([[r[j] for r in loading] for j in range(3)],y,add_intercept=False)
        sse=sum(v*v for v in fit.residuals)
        if best is None or sse<best[0]: best=(sse,tau,fit)
    sse,tau,fit=best
    return {"beta0":fit.coefficients[0],"beta1":fit.coefficients[1],"beta2":fit.coefficients[2],
            "tau":tau,"sse":sse,"fitted":fit.fitted,"residuals":fit.residuals,
            "method":"profile grid; beta coefficients by OLS"}


def fitted_curve(maturities: Sequence[float], model: dict) -> list[float]:
    return [nelson_siegel_rate(t,model["beta0"],model["beta1"],model["beta2"],model["tau"]) for t in maturities]


def key_rate_dv01(price_up: Sequence[float], price_down: Sequence[float], bump_bp: float = 1.0) -> tuple[float,...]:
    up=finite_series(price_up); down=finite_series(price_down)
    if len(up)!=len(down) or bump_bp<=0: raise ValueError("price arrays must align and bump must be positive")
    return tuple((d-u)/(2*bump_bp) for u,d in zip(up,down))


def principal_components(yield_changes: Sequence[Sequence[float]], components: int = 3) -> dict:
    from ..risk.covariance import sample_covariance,pca_risk
    cov=sample_covariance(yield_changes)
    return pca_risk(cov,components)
