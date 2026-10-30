"""Alpha ensemble diversification and breadth diagnostics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,correlation,stddev
from ..linalg import solve


def alpha_correlation_matrix(alpha_returns: Sequence[Sequence[float]]) -> list[list[float]]:
    cols=[finite_series(c,name="alpha return") for c in alpha_returns]
    if not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("alpha return series must align")
    return [[correlation(a,b) for b in cols] for a in cols]


def diversification_ratio(weights: Sequence[float], alpha_covariance: Sequence[Sequence[float]]) -> float:
    w=list(map(float,weights)); n=len(w)
    if len(alpha_covariance)!=n or any(len(row)!=n for row in alpha_covariance): raise ValueError("dimension mismatch")
    numerator=sum(abs(w[i])*sqrt(max(0.0,alpha_covariance[i][i])) for i in range(n))
    portfolio_var=sum(w[i]*alpha_covariance[i][j]*w[j] for i in range(n) for j in range(n))
    return numerator/sqrt(max(0.0,portfolio_var)) if portfolio_var>0 else 0.0


def minimum_variance_alpha_weights(covariance: Sequence[Sequence[float]],
                                   *, ridge: float = 1e-8) -> tuple[float,...]:
    n=len(covariance)
    if n<1 or any(len(row)!=n for row in covariance) or ridge<0: raise ValueError("invalid covariance")
    ones=[1.0]*n
    inv_one=solve(covariance,ones,ridge=max(ridge,1e-12))
    total=sum(inv_one)
    if abs(total)<1e-14: raise ValueError("minimum variance weights are not normalizable")
    return tuple(v/total for v in inv_one)


def alpha_breadth(alpha_returns: Sequence[Sequence[float]]) -> dict:
    cols=[finite_series(c) for c in alpha_returns]
    if not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("alpha returns must align")
    corr=alpha_correlation_matrix(cols); n=len(cols)
    off=[abs(corr[i][j]) for i in range(n) for j in range(i+1,n)]
    avg=sum(off)/len(off) if off else 0.0
    # Effective breadth approximation: number of independent streams under equicorrelation.
    breadth=n/(1+(n-1)*avg) if n else 0.0
    return {"alpha_count":n,"average_absolute_pairwise_correlation":avg,
            "effective_breadth_approx":breadth,"correlation_matrix":tuple(tuple(r) for r in corr)}


def weighted_alpha_ensemble(alpha_returns: Sequence[Sequence[float]], weights: Sequence[float]) -> dict:
    cols=[finite_series(c) for c in alpha_returns]; w=list(map(float,weights))
    if len(cols)!=len(w) or not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("input dimensions mismatch")
    if any(v<0 for v in w) or sum(w)<=0: raise ValueError("weights must be nonnegative with positive sum")
    total=sum(w); w=[v/total for v in w]
    combined=[sum(w[j]*cols[j][t] for j in range(len(cols))) for t in range(len(cols[0]))]
    return {"combined_returns":tuple(combined),"weights":tuple(w),"mean":mean(combined),
            "volatility":stddev(combined),"breadth":alpha_breadth(cols)}
