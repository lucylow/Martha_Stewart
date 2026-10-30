"""Robust estimation and uncertainty-penalized portfolio construction."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,quantile
from ..linalg import matvec,dot


def median_of_means(values: Sequence[float], groups: int = 5) -> dict:
    """Median-of-means location estimator with uneven final block allowed."""
    x=finite_series(values,name="values")
    if groups<1 or groups>len(x): raise ValueError("groups must be in [1,n]")
    buckets=[[] for _ in range(groups)]
    for i,v in enumerate(x): buckets[min(groups-1,i*groups//len(x))].append(v)
    means=[mean(b) for b in buckets if b]
    return {"estimate":quantile(means,0.5),"group_means":tuple(means),"groups":len(means),
            "sample_mean":mean(x),"robustness_note":"median-of-means is less sensitive to heavy-tailed outliers"}


def huber_location(values: Sequence[float], tuning: float = 1.345,
                   max_iter: int = 200, tolerance: float = 1e-9) -> dict:
    """Huber M-estimator of location via iteratively reweighted least squares."""
    from ..core import stddev
    x=finite_series(values,name="values")
    if tuning<=0 or max_iter<1 or tolerance<=0: raise ValueError("invalid Huber controls")
    scale=max(stddev(x),1e-12); mu=quantile(x,0.5); converged=False
    for iteration in range(1,max_iter+1):
        residual=[(v-mu)/scale for v in x]
        weights=[1.0 if abs(r)<=tuning else tuning/abs(r) for r in residual]
        updated=sum(w*v for w,v in zip(weights,x))/sum(weights)
        if abs(updated-mu)<tolerance: mu=updated; converged=True; break
        mu=updated
    return {"location":mu,"scale":scale,"iterations":iteration,"converged":converged,"tuning":tuning}


def robust_expected_returns(asset_returns: Sequence[Sequence[float]], method: str = "huber",
                            tuning: float = 1.345) -> tuple[float,...]:
    """Robust per-asset mean estimator from time x asset return data."""
    rows=[finite_series(r,name="return row") for r in asset_returns]
    if not rows or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("returns panel must be rectangular")
    cols=list(zip(*rows)); result=[]
    for col in cols:
        if method=="huber": result.append(huber_location(col,tuning)["location"])
        elif method=="median": result.append(quantile(col,0.5))
        elif method=="median_of_means": result.append(median_of_means(col)["estimate"])
        else: raise ValueError("method must be huber, median or median_of_means")
    return tuple(result)


def worst_case_expected_returns(mean_returns: Sequence[float],
                                estimation_error: Sequence[float],
                                uncertainty_radius: float = 1.0) -> tuple[float,...]:
    """Diagonal ellipsoidal lower confidence bound for long-only assets."""
    mu=list(map(float,mean_returns)); se=list(map(float,estimation_error))
    if len(mu)!=len(se) or any(v<0 for v in se) or uncertainty_radius<0: raise ValueError("invalid uncertainty inputs")
    return tuple(mu[i]-uncertainty_radius*se[i] for i in range(len(mu)))


def robust_mean_variance(mean_returns: Sequence[float], covariance: Sequence[Sequence[float]],
                         estimation_error: Sequence[float], *, uncertainty_radius: float = 1.0,
                         risk_aversion: float = 5.0, lower: float = 0.0, upper: float = 1.0) -> dict:
    """Optimize against conservative expected returns rather than point estimates."""
    from .constraints import optimize_mean_variance
    conservative=worst_case_expected_returns(mean_returns,estimation_error,uncertainty_radius)
    result=optimize_mean_variance(conservative,covariance,risk_aversion,lower=lower,upper=upper)
    return {**result,"conservative_expected_returns":conservative,"uncertainty_radius":uncertainty_radius,
            "point_estimate_expected_returns":tuple(mean_returns)}


def covariance_condition_number(covariance: Sequence[Sequence[float]]) -> float:
    from ..linalg import eigen_symmetric
    vals,_=eigen_symmetric(covariance)
    positive=[v for v in vals if v>1e-14]
    if not positive: return float("inf")
    return max(positive)/min(positive)


def covariance_stability(covariance_windows: Sequence[Sequence[Sequence[float]]]) -> dict:
    """Summarize element-wise covariance drift across rolling estimation windows."""
    mats=[[[float(v) for v in row] for row in m] for m in covariance_windows]
    if len(mats)<2: raise ValueError("at least two covariance windows required")
    n=len(mats[0])
    if any(len(m)!=n or any(len(row)!=n for row in m) for m in mats): raise ValueError("covariance dimensions must match")
    mean_mat=[[sum(m[i][j] for m in mats)/len(mats) for j in range(n)] for i in range(n)]
    drift=[sum(abs(mats[t][i][j]-mean_mat[i][j]) for t in range(len(mats)))/len(mats) for i in range(n) for j in range(n)]
    return {"mean_covariance":tuple(tuple(r) for r in mean_mat),"mean_absolute_element_drift":sum(drift)/len(drift),
            "window_count":len(mats),"condition_numbers":tuple(covariance_condition_number(m) for m in mats)}
