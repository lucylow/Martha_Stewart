"""Regime-conditioned covariance and forward risk forecasting."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series
from .covariance import sample_covariance
from ..linalg import matvec,dot


def regime_covariances(returns: Sequence[Sequence[float]], regimes: Sequence[int],
                       min_observations: int = 5) -> dict:
    """Estimate a covariance matrix per discrete regime label."""
    rows=[finite_series(r,name="return row") for r in returns]
    if len(rows)!=len(regimes) or not rows or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("panel and regime labels must align")
    labels=sorted(set(regimes)); matrices={}; counts={}
    for label in labels:
        subset=[rows[i] for i,g in enumerate(regimes) if g==label]
        if len(subset)<min_observations: continue
        cov=sample_covariance(subset); matrices[label]=tuple(tuple(r) for r in cov); counts[label]=len(subset)
    return {"covariances":matrices,"counts":counts,"excluded_regimes":tuple(g for g in labels if g not in matrices)}


def probability_weighted_covariance(covariances: Sequence[Sequence[Sequence[float]]],
                                    probabilities: Sequence[float]) -> tuple[tuple[float,...],...]:
    mats=[[[float(v) for v in row] for row in m] for m in covariances]; p=list(map(float,probabilities))
    if not mats or len(mats)!=len(p) or any(v<0 for v in p) or sum(p)<=0: raise ValueError("invalid covariance mixture")
    n=len(mats[0])
    if any(len(m)!=n or any(len(row)!=n for row in m) for m in mats): raise ValueError("matrix dimensions mismatch")
    p=[v/sum(p) for v in p]
    result=[[sum(p[k]*mats[k][i][j] for k in range(len(mats))) for j in range(n)] for i in range(n)]
    return tuple(tuple(r) for r in result)


def markov_regime_forecast(current_probabilities: Sequence[float],
                           transition: Sequence[Sequence[float]], horizon: int = 1) -> dict:
    """Project hidden-state probabilities using the fitted transition matrix."""
    p=list(map(float,current_probabilities)); t=[[float(v) for v in row] for row in transition]
    n=len(p)
    if n<1 or len(t)!=n or any(len(row)!=n for row in t) or horizon<1: raise ValueError("invalid state model")
    if any(v<0 for v in p) or sum(p)<=0 or any(v<0 for row in t for v in row): raise ValueError("probabilities must be nonnegative")
    p=[v/sum(p) for v in p]
    if any(abs(sum(row)-1)>1e-6 for row in t): raise ValueError("transition rows must sum to one")
    path=[]
    for _ in range(horizon):
        p=[sum(p[i]*t[i][j] for i in range(n)) for j in range(n)]; path.append(tuple(p))
    return {"probability_path":tuple(path),"terminal_probabilities":tuple(p)}


def regime_mixture_portfolio_risk(weights: Sequence[float], covariances: Sequence[Sequence[Sequence[float]]],
                                  probabilities: Sequence[float]) -> dict:
    from .covariance import sample_covariance
    cov=probability_weighted_covariance(covariances,probabilities)
    w=list(map(float,weights)); variance_p=dot(w,matvec(cov,w))
    per_regime=[sqrt(max(0.0,dot(w,matvec(m,w)))) for m in covariances]
    p=list(map(float,probabilities)); total=sum(p); p=[v/total for v in p]
    return {"mixture_covariance":cov,"mixture_volatility":sqrt(max(0.0,variance_p)),
            "regime_volatilities":tuple(per_regime),"probabilities":tuple(p),
            "expected_regime_volatility":sum(a*b for a,b in zip(p,per_regime))}
