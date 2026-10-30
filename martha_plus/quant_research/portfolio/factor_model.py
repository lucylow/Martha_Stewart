"""Linear factor risk models with specific risk and covariance reconstruction."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,variance
from ..regression import ols
from ..risk.covariance import sample_covariance
from ..linalg import transpose,matmul


def estimate_factor_exposures(asset_returns: Sequence[Sequence[float]],
                              factor_returns: Sequence[Sequence[float]],
                              *, add_intercept: bool = True) -> dict:
    """Estimate asset-by-factor exposures using time-series OLS.

    Inputs are assets x time and factors x time.  Residual variances are specific
    risk estimates; factors are assumed observed without error.
    """
    assets=[finite_series(x,name="asset return") for x in asset_returns]
    factors=[finite_series(x,name="factor return") for x in factor_returns]
    if not assets or not factors or any(len(x)!=len(assets[0]) for x in assets+factors): raise ValueError("panels must be aligned")
    exposures=[]; alpha=[]; specific=[]; residuals=[]
    for asset in assets:
        fit=ols(factors,asset,add_intercept=add_intercept)
        exposures.append(fit.coefficients[1:] if add_intercept else fit.coefficients)
        alpha.append(fit.coefficients[0] if add_intercept else 0.0)
        specific.append(variance(fit.residuals))
        residuals.append(fit.residuals)
    return {"exposures":tuple(tuple(r) for r in exposures),"alpha":tuple(alpha),
            "specific_variance":tuple(specific),"residuals":tuple(tuple(r) for r in residuals),
            "assets":len(assets),"factors":len(factors),"observations":len(assets[0])}


def factor_covariance(factor_returns: Sequence[Sequence[float]]) -> list[list[float]]:
    cols=[finite_series(c) for c in factor_returns]
    if not cols or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("factor series must align")
    return sample_covariance(list(zip(*cols)))


def reconstruct_asset_covariance(exposures: Sequence[Sequence[float]],
                                 factor_cov: Sequence[Sequence[float]],
                                 specific_variance: Sequence[float]) -> list[list[float]]:
    """Sigma_asset = B F B' + diagonal(specific variance)."""
    b=[list(map(float,row)) for row in exposures]; d=list(map(float,specific_variance))
    if not b or len(d)!=len(b): raise ValueError("specific variance must match assets")
    k=len(b[0])
    if any(len(row)!=k for row in b) or len(factor_cov)!=k or any(len(row)!=k for row in factor_cov):
        raise ValueError("factor covariance dimensions mismatch")
    if any(v<0 for v in d): raise ValueError("specific variances must be non-negative")
    bt=transpose(b); fb=matmul(matmul(b,factor_cov),bt)
    for i,v in enumerate(d): fb[i][i]+=v
    return fb


def factor_risk_model(asset_returns: Sequence[Sequence[float]], factor_returns: Sequence[Sequence[float]]) -> dict:
    fit=estimate_factor_exposures(asset_returns,factor_returns)
    fc=factor_covariance(factor_returns)
    covariance=reconstruct_asset_covariance(fit["exposures"],fc,fit["specific_variance"])
    return {**fit,"factor_covariance":tuple(tuple(r) for r in fc),
            "asset_covariance":tuple(tuple(r) for r in covariance)}


def factor_marginal_contributions(weights: Sequence[float], exposures: Sequence[Sequence[float]],
                                   factor_cov: Sequence[Sequence[float]]) -> dict:
    w=list(map(float,weights)); b=[list(map(float,r)) for r in exposures]
    if len(w)!=len(b): raise ValueError("weights must match exposure rows")
    factor_exposure=[sum(w[i]*b[i][j] for i in range(len(w))) for j in range(len(b[0]))]
    marginal=[sum(factor_cov[i][j]*factor_exposure[j] for j in range(len(factor_exposure))) for i in range(len(factor_exposure))]
    variance_port=sum(factor_exposure[i]*marginal[i] for i in range(len(marginal)))
    vol=sqrt(max(0.0,variance_port))
    return {"portfolio_factor_exposure":tuple(factor_exposure),"factor_marginal_variance":tuple(marginal),
            "factor_component_variance":tuple(a*b for a,b in zip(factor_exposure,marginal)),"factor_volatility":vol}
