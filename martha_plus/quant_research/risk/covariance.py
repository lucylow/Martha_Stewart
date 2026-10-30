"""Covariance estimators, eigenvalue cleaning and PCA risk summaries."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series, mean, variance
from ..linalg import transpose, eigen_symmetric, symmetrize


def _panel(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    rows=[finite_series(row,name="return row",min_length=1) for row in returns]
    if len(rows)<3 or any(len(r)!=len(rows[0]) for r in rows):
        raise ValueError("returns must be a rectangular time x asset matrix with >=3 periods")
    return rows


def sample_covariance(returns: Sequence[Sequence[float]], *, ddof: int = 1) -> list[list[float]]:
    x=_panel(returns); assets=transpose(x); p=len(assets)
    if ddof<0 or len(x)<=ddof: raise ValueError("invalid ddof")
    means=[mean(a) for a in assets]
    return [[sum((a-means[i])*(b-means[j]) for a,b in zip(assets[i],assets[j]))/(len(x)-ddof)
             for j in range(p)] for i in range(p)]


def ewma_covariance(returns: Sequence[Sequence[float]], decay: float = 0.94,
                    *, demean: bool = True) -> list[list[float]]:
    """Exponentially weighted covariance with normalized finite-sample weights."""
    x=_panel(returns)
    if not 0<decay<1: raise ValueError("decay must be in (0,1)")
    p=len(x[0]); mu=[mean([row[j] for row in x]) for j in range(p)] if demean else [0.0]*p
    weights=[(1-decay)*decay**(len(x)-1-i) for i in range(len(x))]
    total=sum(weights); weights=[w/total for w in weights]
    return [[sum(weights[t]*(x[t][i]-mu[i])*(x[t][j]-mu[j]) for t in range(len(x))) for j in range(p)] for i in range(p)]


def shrink_covariance(returns: Sequence[Sequence[float]], intensity: float | None = None,
                      *, target: str = "diagonal") -> dict:
    """Shrink sample covariance toward diagonal or constant-correlation target.

    The automatic intensity is a conservative heuristic, not the analytical
    Ledoit-Wolf estimator. Supply an independently estimated intensity when
    exact shrinkage calibration is required.
    """
    x=_panel(returns); s=sample_covariance(x); p=len(s); n=len(x)
    if target not in {"diagonal","constant_correlation"}: raise ValueError("unsupported target")
    if target=="diagonal": t=[[s[i][i] if i==j else 0.0 for j in range(p)] for i in range(p)]
    else:
        vols=[sqrt(max(s[i][i],0.0)) for i in range(p)]
        corr=[s[i][j]/(vols[i]*vols[j]) if vols[i]*vols[j]>0 else 0.0 for i in range(p) for j in range(p) if i!=j]
        avg=sum(corr)/len(corr) if corr else 0.0
        t=[[s[i][i] if i==j else avg*vols[i]*vols[j] for j in range(p)] for i in range(p)]
    lam=min(1.0,max(0.0, p/(n+p))) if intensity is None else intensity
    if not 0<=lam<=1: raise ValueError("intensity must be in [0,1]")
    out=[[(1-lam)*s[i][j]+lam*t[i][j] for j in range(p)] for i in range(p)]
    return {"covariance":tuple(tuple(r) for r in out),"sample_covariance":tuple(tuple(r) for r in s),
            "target":target,"intensity":lam,"estimator":"heuristic linear shrinkage" if intensity is None else "user-specified shrinkage"}


def robust_covariance(returns: Sequence[Sequence[float]], winsor_quantile: float = 0.025) -> list[list[float]]:
    """Winsorized covariance; robust to isolated extreme observations."""
    from ..core import quantile
    x=_panel(returns); p=len(x[0]); cols=transpose(x)
    clean=[]
    for col in cols:
        lo,hi=quantile(col,winsor_quantile),quantile(col,1-winsor_quantile)
        clean.append([min(hi,max(lo,v)) for v in col])
    return sample_covariance(transpose(clean))


def pca_risk(covariance: Sequence[Sequence[float]], components: int | None = None) -> dict:
    """Principal-component decomposition with explained-variance ratios."""
    vals,vecs=eigen_symmetric(covariance)
    if any(v < -1e-9 for v in vals): raise ValueError("covariance has materially negative eigenvalues")
    k=len(vals) if components is None else components
    if k<1 or k>len(vals): raise ValueError("invalid component count")
    total=sum(max(0.0,v) for v in vals)
    ratios=[0.0 if total==0 else max(0.0,v)/total for v in vals]
    return {"eigenvalues":tuple(vals[:k]),"eigenvectors":tuple(tuple(row[:k]) for row in vecs),
            "explained_variance_ratio":tuple(ratios[:k]),"cumulative_explained":sum(ratios[:k])}


def eigenvalue_clean(covariance: Sequence[Sequence[float]], floor: float = 1e-8) -> dict:
    """Clip small eigenvalues and reconstruct a symmetric PSD covariance."""
    if floor<=0: raise ValueError("floor must be positive")
    vals,vecs=eigen_symmetric(covariance)
    cleaned=[max(floor,v) for v in vals]
    n=len(vals); out=[[sum(vecs[i][k]*cleaned[k]*vecs[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    return {"covariance":tuple(tuple(r) for r in symmetrize(out)),"original_eigenvalues":tuple(vals),"cleaned_eigenvalues":tuple(cleaned)}


def correlation_matrix(covariance: Sequence[Sequence[float]]) -> list[list[float]]:
    s=[list(map(float,r)) for r in covariance]
    if not s or len(s)!=len(s[0]) or any(len(r)!=len(s) for r in s): raise ValueError("covariance must be square")
    sd=[sqrt(max(0.0,s[i][i])) for i in range(len(s))]
    return [[1.0 if i==j and sd[i]>0 else (s[i][j]/(sd[i]*sd[j]) if sd[i]*sd[j]>0 else 0.0) for j in range(len(s))] for i in range(len(s))]
