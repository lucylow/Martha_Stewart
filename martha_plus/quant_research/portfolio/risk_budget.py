"""Equal and custom risk-budget portfolios using Euler contributions."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..linalg import matvec,dot


def risk_contributions(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> dict:
    w=list(map(float,weights)); marginal=matvec(covariance,w)
    variance_p=dot(w,marginal); vol=sqrt(max(0.0,variance_p))
    component=[w[i]*marginal[i]/vol if vol>0 else 0.0 for i in range(len(w))]
    shares=[v/vol if vol>0 else 0.0 for v in component]
    return {"volatility":vol,"marginal_risk":tuple(marginal),"component_risk":tuple(component),"risk_shares":tuple(shares)}


def equal_risk_contribution(covariance: Sequence[Sequence[float]], *,
                            target_budget: Sequence[float] | None = None,
                            max_iter: int = 5000, tolerance: float = 1e-8) -> dict:
    """Iterative proportional risk-contribution solver for long-only portfolios.

    It minimizes deviations in percentage risk budgets through a multiplicative
    update. It may converge slowly or fail for singular covariance structures.
    """
    n=len(covariance)
    if n<2 or any(len(row)!=n for row in covariance): raise ValueError("covariance must be square with >=2 assets")
    budget=[1/n]*n if target_budget is None else list(map(float,target_budget))
    if len(budget)!=n or any(v<=0 for v in budget) or max_iter<1 or tolerance<=0: raise ValueError("invalid risk budgets or solver controls")
    total=sum(budget); budget=[v/total for v in budget]
    # Volatility-scaled starting weights reduce the initial risk imbalance.
    vols=[sqrt(max(1e-12,covariance[i][i])) for i in range(n)]
    w=[1/v for v in vols]; w=[v/sum(w) for v in w]
    converged=False; errors=[]
    for iteration in range(1,max_iter+1):
        rc=risk_contributions(w,covariance)["risk_shares"]
        errors=[rc[i]-budget[i] for i in range(n)]
        if max(abs(v) for v in errors)<tolerance: converged=True; break
        # Multiplicative correction; damping avoids oscillation.
        updated=[w[i]*(budget[i]/max(rc[i],1e-12))**0.35 for i in range(n)]
        s=sum(updated); w=[v/s for v in updated]
    risk=risk_contributions(w,covariance)
    return {"weights":tuple(w),"target_budget":tuple(budget),"realized_risk_shares":risk["risk_shares"],
            "volatility":risk["volatility"],"max_budget_error":max(abs(a-b) for a,b in zip(risk["risk_shares"],budget)),
            "converged":converged,"iterations":iteration}


def risk_budget_portfolio(covariance: Sequence[Sequence[float]], budgets: Sequence[float],
                          max_iter: int = 5000, tolerance: float = 1e-8) -> dict:
    return equal_risk_contribution(covariance,budgets,max_iter=max_iter,tolerance=tolerance)


def risk_parity_concentration(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> dict:
    rc=risk_contributions(weights,covariance)["risk_shares"]
    hhi=sum(v*v for v in rc)
    return {"risk_share_hhi":hhi,"effective_risk_breadth":1/hhi if hhi>0 else 0.0,
            "largest_risk_share":max(rc,default=0.0),"risk_shares":tuple(rc)}
