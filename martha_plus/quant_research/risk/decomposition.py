"""Euler volatility, factor-risk and active-risk decomposition."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..linalg import matvec, dot


def portfolio_volatility(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    w=list(map(float,weights))
    if len(covariance)!=len(w) or any(len(r)!=len(w) for r in covariance): raise ValueError("dimension mismatch")
    return sqrt(max(0.0,dot(w,matvec(covariance,w))))


def euler_risk_contribution(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> dict:
    """Component volatility contributions sum to total portfolio volatility."""
    w=list(map(float,weights)); vol=portfolio_volatility(w,covariance)
    marginal=matvec(covariance,w)
    mrc=[v/vol if vol>0 else 0.0 for v in marginal]
    crc=[a*b for a,b in zip(w,mrc)]
    return {"volatility":vol,"marginal_contribution":tuple(mrc),"component_contribution":tuple(crc),
            "percentage_contribution":tuple(v/vol if vol>0 else 0.0 for v in crc),
            "contribution_sum":sum(crc)}


def factor_risk_contribution(exposures: Sequence[Sequence[float]], factor_covariance: Sequence[Sequence[float]],
                             weights: Sequence[float]) -> dict:
    """Map asset weights through factor exposures, then decompose factor risk."""
    if not exposures or len(exposures)!=len(weights): raise ValueError("exposure rows must match weights")
    k=len(exposures[0])
    if any(len(row)!=k for row in exposures) or len(factor_covariance)!=k: raise ValueError("factor dimensions mismatch")
    factor_weights=[sum(weights[i]*exposures[i][j] for i in range(len(weights))) for j in range(k)]
    result=euler_risk_contribution(factor_weights,factor_covariance)
    return {"factor_exposures":tuple(factor_weights),"factor_risk":result}


def tracking_error(portfolio_weights: Sequence[float], benchmark_weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    if len(portfolio_weights)!=len(benchmark_weights): raise ValueError("weight lengths must match")
    active=[a-b for a,b in zip(portfolio_weights,benchmark_weights)]
    return portfolio_volatility(active,covariance)


def diversification_ratio(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    w=list(map(float,weights)); vol=portfolio_volatility(w,covariance)
    weighted=sum(abs(w[i])*sqrt(max(0.0,covariance[i][i])) for i in range(len(w)))
    return weighted/vol if vol>0 else 0.0
