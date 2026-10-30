"""Uncertainty-aware portfolio diagnostics and covariance perturbations."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,variance
from ..linalg import dot,matvec


def parametric_estimation_error(sample_size: int, dimension: int, confidence: float = 0.95) -> dict:
    """Rule-of-thumb covariance estimation burden indicators."""
    from ..core import normal_ppf
    if sample_size<2 or dimension<1 or not 0.5<confidence<1: raise ValueError("invalid controls")
    ratio=dimension/sample_size
    return {"dimension_to_sample_ratio":ratio,"observations_per_asset":sample_size/dimension,
            "fisher_z_half_width":normal_ppf((1+confidence)/2)/sqrt(max(1,sample_size-3)),
            "high_dimensional_warning":ratio>0.1,
            "note":"diagnostic only; not a confidence interval for the full covariance matrix"}


def monte_carlo_weight_uncertainty(weight_paths: Sequence[Sequence[float]]) -> dict:
    paths=[finite_series(path,name="weight path") for path in weight_paths]
    if len(paths)<2 or any(len(p)!=len(paths[0]) for p in paths): raise ValueError("need >=2 equal-length weight vectors")
    assets=len(paths[0]); means=[]; stds=[]
    for j in range(assets):
        col=[p[j] for p in paths]; means.append(mean(col)); stds.append(sqrt(variance(col)))
    return {"mean_weights":tuple(means),"weight_standard_deviation":tuple(stds),
            "mean_absolute_deviation":tuple(mean([abs(p[j]-means[j]) for p in paths]) for j in range(assets)),
            "simulations":len(paths)}


def covariance_stress(covariance: Sequence[Sequence[float]], correlation_multiplier: float = 1.25,
                      volatility_multiplier: float = 1.2) -> list[list[float]]:
    """Scale volatilities and off-diagonal correlations for a simple stress."""
    s=[list(map(float,r)) for r in covariance]; n=len(s)
    if n==0 or any(len(r)!=n for r in s) or correlation_multiplier<0 or volatility_multiplier<=0:
        raise ValueError("invalid covariance or multipliers")
    sd=[sqrt(max(0.0,s[i][i]))*volatility_multiplier for i in range(n)]
    out=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            corr=1.0 if i==j else (s[i][j]/sqrt(max(1e-30,s[i][i]*s[j][j])))
            if i!=j: corr=max(-1.0,min(1.0,corr*correlation_multiplier))
            out[i][j]=corr*sd[i]*sd[j]
    return out
