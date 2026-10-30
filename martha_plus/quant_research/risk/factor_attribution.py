"""Scenario and Euler attribution of portfolio VaR and expected shortfall."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,quantile
from ..linalg import dot


def historical_var_contributions(asset_scenarios: Sequence[Sequence[float]],
                                 weights: Sequence[float], confidence: float = 0.95) -> dict:
    """Allocate historical VaR by average component loss at the portfolio VaR tail.

    This is a discrete tail-allocation convention. Contributions need not sum
    exactly to quantile VaR because quantiles are discontinuous in finite samples.
    """
    rows=[finite_series(r,name="scenario") for r in asset_scenarios]; w=list(map(float,weights))
    if not rows or any(len(r)!=len(w) for r in rows) or not 0.5<confidence<1: raise ValueError("invalid scenarios or confidence")
    component_losses=[tuple(-w[j]*row[j] for j in range(len(w))) for row in rows]
    portfolio_losses=[sum(row) for row in component_losses]
    threshold=quantile(portfolio_losses,confidence)
    tail=[i for i,v in enumerate(portfolio_losses) if v>=threshold]
    contributions=[mean([component_losses[i][j] for i in tail]) for j in range(len(w))]
    return {"portfolio_var":threshold,"component_tail_loss":tuple(contributions),
            "component_sum":sum(contributions),"tail_observations":len(tail),
            "residual_to_var":threshold-sum(contributions),"confidence":confidence}


def historical_es_contributions(asset_scenarios: Sequence[Sequence[float]],
                                 weights: Sequence[float], confidence: float = 0.95) -> dict:
    """Expected-shortfall contributions using the same tail scenarios as portfolio ES."""
    rows=[finite_series(r,name="scenario") for r in asset_scenarios]; w=list(map(float,weights))
    if not rows or any(len(r)!=len(w) for r in rows) or not 0.5<confidence<1: raise ValueError("invalid inputs")
    component_losses=[tuple(-w[j]*row[j] for j in range(len(w))) for row in rows]
    portfolio_losses=[sum(row) for row in component_losses]
    threshold=quantile(portfolio_losses,confidence); tail=[i for i,v in enumerate(portfolio_losses) if v>=threshold]
    contrib=[mean([component_losses[i][j] for i in tail]) for j in range(len(w))]
    return {"expected_shortfall":mean([portfolio_losses[i] for i in tail]),
            "component_es":tuple(contrib),"component_sum":sum(contrib),
            "tail_observations":len(tail),"confidence":confidence}


def normal_var_euler(weights: Sequence[float], covariance: Sequence[Sequence[float]],
                     confidence: float = 0.99) -> dict:
    """Euler contribution to Gaussian VaR (loss sign convention)."""
    from ..core import normal_ppf
    w=list(map(float,weights))
    if len(covariance)!=len(w) or any(len(r)!=len(w) for r in covariance) or not 0.5<confidence<1:
        raise ValueError("invalid covariance or confidence")
    marginal=[sum(covariance[i][j]*w[j] for j in range(len(w))) for i in range(len(w))]
    sigma2=sum(w[i]*marginal[i] for i in range(len(w))); sigma=sqrt(max(0.0,sigma2))
    z=normal_ppf(confidence); contributions=[z*v/sigma*w[i] if sigma>0 else 0.0 for i,v in enumerate(marginal)]
    return {"gaussian_var":z*sigma,"component_var":tuple(contributions),"sum_component_var":sum(contributions),
            "marginal_var":tuple(z*v/sigma if sigma>0 else 0.0 for v in marginal),"volatility":sigma}


def group_risk_attribution(weights: Sequence[float], component_risk: Sequence[float],
                           groups: Sequence[str]) -> dict:
    w=list(map(float,weights)); r=list(map(float,component_risk))
    if not(len(w)==len(r)==len(groups)): raise ValueError("weights, risk and groups must align")
    results={}
    for weight,risk,group in zip(w,r,groups):
        entry=results.setdefault(str(group),{"weight":0.0,"risk":0.0,"gross_weight":0.0})
        entry["weight"]+=weight; entry["risk"]+=risk; entry["gross_weight"]+=abs(weight)
    return results
