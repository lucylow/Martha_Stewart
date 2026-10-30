"""Projected-gradient portfolio optimization with explicit constraints."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..linalg import matvec, dot


def project_simplex(values: Sequence[float], total: float = 1.0) -> list[float]:
    """Euclidean projection onto {w >= 0, sum(w)=total}."""
    v=list(map(float,values))
    if not v or total<=0: raise ValueError("values nonempty and total positive required")
    u=sorted(v,reverse=True); cumulative=0.0; rho=0
    for j,value in enumerate(u,1):
        cumulative+=value
        if value-(cumulative-total)/j>0: rho=j
    if rho==0: return [total/len(v)]*len(v)
    theta=(sum(u[:rho])-total)/rho
    return [max(0.0,x-theta) for x in v]


def project_box_simplex(values: Sequence[float], lower: float = 0.0,
                        upper: float = 1.0, total: float = 1.0,
                        iterations: int = 100) -> list[float]:
    """Projection onto box and budget constraints by scalar threshold search."""
    v=list(map(float,values)); n=len(v)
    if not v or lower>upper or n*lower>total or n*upper<total: raise ValueError("infeasible box-simplex constraints")
    lo=min(v)-upper; hi=max(v)-lower
    for _ in range(iterations):
        mid=(lo+hi)/2
        w=[min(upper,max(lower,x-mid)) for x in v]
        if sum(w)>total: lo=mid
        else: hi=mid
    out=[min(upper,max(lower,x-(lo+hi)/2)) for x in v]
    # Correct tiny residual while respecting bounds.
    residual=total-sum(out)
    for i in range(n):
        room=(upper-out[i]) if residual>0 else (out[i]-lower)
        move=(1 if residual>0 else -1)*min(abs(residual),room)
        out[i]+=move; residual-=move
        if abs(residual)<1e-10: break
    return out


def optimize_mean_variance(expected_returns: Sequence[float], covariance: Sequence[Sequence[float]],
                           risk_aversion: float = 5.0, *, lower: float = 0.0,
                           upper: float = 1.0, max_iter: int = 5000,
                           tolerance: float = 1e-9, initial: Sequence[float] | None = None) -> dict:
    """Maximize mu'w - lambda/2 w'Sigma w under box and full-investment constraints."""
    mu=list(map(float,expected_returns)); n=len(mu)
    if len(covariance)!=n or any(len(r)!=n for r in covariance): raise ValueError("covariance dimensions mismatch")
    if risk_aversion<=0 or max_iter<1 or tolerance<=0: raise ValueError("invalid optimization controls")
    w=project_box_simplex(initial if initial is not None else [1/n]*n,lower,upper)
    # Lipschitz constant is a conservative row-sum bound for symmetric covariance.
    lipschitz=max(sum(abs(float(v)) for v in row) for row in covariance)*risk_aversion
    step=1.0/max(lipschitz,1e-8); converged=False
    for iteration in range(1,max_iter+1):
        risk_gradient=[risk_aversion*v for v in matvec(covariance,w)]
        gradient=[mu[i]-risk_gradient[i] for i in range(n)]
        candidate=project_box_simplex([w[i]+step*gradient[i] for i in range(n)],lower,upper)
        delta=max(abs(a-b) for a,b in zip(candidate,w))
        w=candidate
        if delta<tolerance: converged=True; break
    ret=dot(mu,w); vol=sqrt(max(0.0,dot(w,matvec(covariance,w))))
    objective=ret-0.5*risk_aversion*vol*vol
    return {"weights":tuple(w),"expected_return":ret,"volatility":vol,"objective":objective,
            "converged":converged,"iterations":iteration,"constraints":{"lower":lower,"upper":upper,"sum":sum(w)}}


def turnover_constrained_rebalance(target: Sequence[float], current: Sequence[float],
                                   max_one_way_turnover: float) -> tuple[float,...]:
    """Scale movement toward target to obey half-L1 turnover budget."""
    a=list(map(float,target)); b=list(map(float,current))
    if len(a)!=len(b) or max_one_way_turnover<0: raise ValueError("invalid weights or turnover budget")
    turnover=0.5*sum(abs(x-y) for x,y in zip(a,b))
    if turnover<=max_one_way_turnover or turnover==0: return tuple(a)
    fraction=max_one_way_turnover/turnover
    return tuple(y+fraction*(x-y) for x,y in zip(a,b))
