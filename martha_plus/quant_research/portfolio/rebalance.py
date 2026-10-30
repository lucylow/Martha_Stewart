"""Transaction-cost-aware rebalancing and portfolio drift analytics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series
from ..linalg import matvec,dot


def drifted_weights(target_weights: Sequence[float], returns_since_rebalance: Sequence[float]) -> tuple[float,...]:
    w=list(map(float,target_weights)); r=list(map(float,returns_since_rebalance))
    if len(w)!=len(r) or any(v< -1 for v in r): raise ValueError("weights and returns must align; returns >= -100%")
    values=[a*(1+b) for a,b in zip(w,r)]; total=sum(values)
    if total<=0: raise ValueError("portfolio value is non-positive")
    return tuple(v/total for v in values)


def proportional_transaction_cost(trades: Sequence[float], prices: Sequence[float],
                                  rates: Sequence[float], fixed_cost: float = 0.0) -> dict:
    q=list(map(float,trades)); p=list(map(float,prices)); c=list(map(float,rates))
    if not q or len(q)!=len(p) or len(q)!=len(c) or fixed_cost<0: raise ValueError("inputs must align")
    if any(v<=0 for v in p) or any(v<0 for v in c): raise ValueError("prices positive and cost rates non-negative")
    variable=sum(abs(qi)*pi*ci for qi,pi,ci in zip(q,p,c))
    active=sum(qi!=0 for qi in q)
    return {"variable_cost":variable,"fixed_cost":active*fixed_cost,"total_cost":variable+active*fixed_cost,
            "active_trades":active}


def impact_cost(trades: Sequence[float], prices: Sequence[float], adv: Sequence[float],
                volatility: Sequence[float], coefficient: float = 0.1,
                commission_bps: float = 0.0) -> dict:
    """Square-root impact estimate; parameters require market-specific calibration."""
    q=list(map(float,trades)); p=list(map(float,prices)); a=list(map(float,adv)); v=list(map(float,volatility))
    if not q or not (len(q)==len(p)==len(a)==len(v)): raise ValueError("input lengths must match")
    if coefficient<0 or commission_bps<0 or any(x<=0 for x in p+a) or any(x<0 for x in v): raise ValueError("invalid market inputs")
    rows=[]; total=0.0
    for qty,price,volume,sigma in zip(q,p,a,v):
        participation=abs(qty)/volume
        impact_rate=coefficient*sigma*sqrt(participation)+commission_bps/10000
        cost=abs(qty)*price*impact_rate; total+=cost
        rows.append({"participation":participation,"impact_rate":impact_rate,"cost":cost})
    return {"total_cost":total,"by_asset":tuple(rows),"model":"calibrated square-root impact approximation"}


def optimize_rebalance(current_weights: Sequence[float], expected_returns: Sequence[float],
                       covariance: Sequence[Sequence[float]], *, risk_aversion: float = 5.0,
                       transaction_cost_bps: float = 10.0, turnover_penalty: float = 0.0,
                       lower: float = 0.0, upper: float = 1.0,
                       iterations: int = 3000, learning_rate: float = 0.02) -> dict:
    """Optimize mean-variance utility net of linear trade costs and L1 turnover.

    A smoothed sign subgradient is used around current weights.  The solution is
    approximate and should be compared with a convex solver for critical use.
    """
    from .constraints import project_box_simplex
    current=list(map(float,current_weights)); mu=list(map(float,expected_returns)); n=len(mu)
    if len(current)!=n or len(covariance)!=n or any(len(r)!=n for r in covariance): raise ValueError("dimension mismatch")
    if risk_aversion<=0 or transaction_cost_bps<0 or turnover_penalty<0 or learning_rate<=0 or iterations<1:
        raise ValueError("invalid optimizer settings")
    w=project_box_simplex(current,lower,upper); cost=transaction_cost_bps/10000
    for it in range(1,iterations+1):
        risk=[risk_aversion*v for v in matvec(covariance,w)]
        gradient=[]
        for i in range(n):
            diff=w[i]-current[i]
            gradient.append(mu[i]-risk[i]-(cost+turnover_penalty)*(1 if diff>1e-10 else -1 if diff< -1e-10 else 0))
        candidate=project_box_simplex([w[i]+learning_rate*gradient[i] for i in range(n)],lower,upper)
        delta=max(abs(a-b) for a,b in zip(candidate,w)); w=candidate
        if delta<1e-9: break
        if it%250==0: learning_rate*=0.8
    turnover=0.5*sum(abs(a-b) for a,b in zip(w,current))
    gross=sum(mu[i]*w[i] for i in range(n)); variance_p=dot(w,matvec(covariance,w))
    return {"weights":tuple(w),"expected_return":gross,"volatility":sqrt(max(0,variance_p)),
            "one_way_turnover":turnover,"estimated_cost_return":2*cost*turnover,
            "iterations":it,"converged":delta<1e-9}


def rebalance_threshold(current_weights: Sequence[float], target_weights: Sequence[float],
                        threshold: float) -> dict:
    a=list(map(float,current_weights)); b=list(map(float,target_weights))
    if len(a)!=len(b) or threshold<0: raise ValueError("invalid weights or threshold")
    differences=[y-x for x,y in zip(a,b)]
    should=sum(abs(v) for v in differences)>threshold
    return {"should_rebalance":should,"active_differences":tuple(v for v in differences if abs(v)>threshold/len(a)),
            "one_way_turnover":0.5*sum(abs(v) for v in differences)}
