"""Historical, parametric and reverse-stress scenario analytics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,quantile
from ..linalg import dot


def historical_scenario_pnl(weights: Sequence[float], scenarios: Sequence[Sequence[float]]) -> tuple[float,...]:
    w=list(map(float,weights))
    if not scenarios or any(len(row)!=len(w) for row in scenarios): raise ValueError("scenario dimensions mismatch")
    return tuple(dot(w,row) for row in scenarios)


def scenario_summary(weights: Sequence[float], scenarios: Sequence[Sequence[float]], confidence: float = 0.95) -> dict:
    if not 0.5<confidence<1: raise ValueError("confidence must be in (0.5,1)")
    pnl=historical_scenario_pnl(weights,scenarios); losses=[-v for v in pnl]
    var=quantile(losses,confidence)
    tail=[v for v in losses if v>=var]
    return {"scenario_count":len(pnl),"mean_pnl":mean(pnl),"worst_pnl":min(pnl),
            "best_pnl":max(pnl),"var_loss":var,"expected_shortfall_loss":mean(tail),
            "confidence":confidence,"pnl":pnl}


def factor_shock_pnl(exposures: Sequence[float], shocks: Sequence[float],
                     *, convexity: Sequence[float] | None = None) -> float:
    e=list(map(float,exposures)); s=list(map(float,shocks))
    if len(e)!=len(s): raise ValueError("exposures and shocks must match")
    linear=sum(a*b for a,b in zip(e,s))
    if convexity is None: return linear
    c=list(map(float,convexity))
    if len(c)!=len(s): raise ValueError("convexity dimensions mismatch")
    return linear+0.5*sum(c[i]*s[i]*s[i] for i in range(len(s)))


def reverse_stress_loss(weights: Sequence[float], factor_exposures: Sequence[Sequence[float]],
                        target_loss: float, shock_bounds: Sequence[tuple[float,float]],
                        *, step: float = 0.01, max_combinations: int = 100000) -> dict:
    """Enumerate coarse factor shocks to identify a low-norm target-loss breach.

    Grid enumeration is deliberately explicit and capped; this is not a global
    continuous optimizer.  Factor exposure matrix is assets x factors.
    """
    import itertools
    w=list(map(float,weights)); b=[list(map(float,r)) for r in factor_exposures]
    if len(w)!=len(b) or not b or any(len(r)!=len(shock_bounds) for r in b): raise ValueError("factor exposure dimensions mismatch")
    if target_loss<=0 or step<=0: raise ValueError("target loss and step must be positive")
    grids=[]
    for lo,hi in shock_bounds:
        if lo>hi: raise ValueError("shock bounds invalid")
        count=int((hi-lo)/step)+1
        grids.append([lo+i*step for i in range(count)])
    total=1
    for g in grids: total*=len(g)
    if total>max_combinations: raise ValueError("shock grid exceeds max_combinations")
    portfolio_exp=[sum(w[i]*b[i][j] for i in range(len(w))) for j in range(len(shock_bounds))]
    best=None
    for shock in itertools.product(*grids):
        loss=-sum(a*bv for a,bv in zip(portfolio_exp,shock))
        norm=sum(v*v for v in shock)**0.5
        if loss>=target_loss and (best is None or norm<best[0]): best=(norm,shock,loss)
    if best is None: return {"found":False,"target_loss":target_loss,"grid_points":total}
    return {"found":True,"shock":tuple(best[1]),"shock_norm":best[0],"loss":best[2],"target_loss":target_loss,"grid_points":total}


def stress_test_portfolio(returns: Sequence[Sequence[float]], weights: Sequence[float],
                          shocks: Sequence[Sequence[float]]) -> dict:
    """Apply scenario return shocks to time x asset panel and report portfolio effects."""
    w=list(map(float,weights))
    if not returns or any(len(row)!=len(w) for row in returns): raise ValueError("returns panel dimensions mismatch")
    baseline=[dot(w,row) for row in returns]
    scenario=[dot(w,row) for row in shocks]
    return {"baseline_mean":mean(baseline),"baseline_volatility":sqrt(sum((v-mean(baseline))**2 for v in baseline)/max(1,len(baseline)-1)),
            "scenario_pnl":tuple(scenario),"worst_scenario":min(scenario),"best_scenario":max(scenario)}
