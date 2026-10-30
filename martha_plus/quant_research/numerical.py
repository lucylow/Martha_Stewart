"""Transparent one-dimensional numerical optimization and calibration routines."""
from __future__ import annotations
from math import isfinite
from typing import Callable, Sequence


def bisect_root(function: Callable[[float],float], lower: float, upper: float,
                *, tolerance: float = 1e-10, max_iter: int = 200) -> dict:
    """Find a sign-changing root using bisection."""
    if not lower<upper or tolerance<=0 or max_iter<1: raise ValueError("invalid root-finding controls")
    fl,fu=float(function(lower)),float(function(upper))
    if not isfinite(fl+fu) or fl*fu>0: raise ValueError("endpoints must have finite opposite signs")
    for iteration in range(1,max_iter+1):
        mid=(lower+upper)/2; fm=float(function(mid))
        if not isfinite(fm): raise ValueError("function returned non-finite value")
        if abs(fm)<=tolerance or (upper-lower)/2<=tolerance:
            return {"root":mid,"value":fm,"iterations":iteration,"converged":True}
        if fl*fm<=0: upper,fu=mid,fm
        else: lower,fl=mid,fm
    return {"root":(lower+upper)/2,"value":float(function((lower+upper)/2)),"iterations":max_iter,"converged":False}


def golden_section_minimize(function: Callable[[float],float], lower: float, upper: float,
                             *, tolerance: float = 1e-8, max_iter: int = 500) -> dict:
    """Minimize a unimodal scalar objective on a closed interval."""
    if not lower<upper or tolerance<=0: raise ValueError("invalid interval or tolerance")
    ratio=(5**0.5-1)/2; c=upper-ratio*(upper-lower); d=lower+ratio*(upper-lower)
    fc,fd=float(function(c)),float(function(d))
    for iteration in range(1,max_iter+1):
        if upper-lower<tolerance: break
        if fc<fd:
            upper,d,fd=d,c,fc; c=upper-ratio*(upper-lower); fc=float(function(c))
        else:
            lower,c,fc=c,d,fd; d=lower+ratio*(upper-lower); fd=float(function(d))
    x=(lower+upper)/2
    return {"x":x,"value":float(function(x)),"iterations":iteration,"converged":upper-lower<tolerance}


def finite_difference_gradient(function: Callable[[Sequence[float]],float], point: Sequence[float],
                                step: float = 1e-5) -> tuple[float,...]:
    x=list(map(float,point))
    if not x or step<=0: raise ValueError("point must be non-empty and step positive")
    out=[]
    for i in range(len(x)):
        up=list(x); down=list(x); up[i]+=step; down[i]-=step
        out.append((float(function(up))-float(function(down)))/(2*step))
    return tuple(out)


def coordinate_descent(function: Callable[[Sequence[float]],float], initial: Sequence[float],
                       *, step: float = 0.1, tolerance: float = 1e-6,
                       max_iter: int = 1000, bounds: Sequence[tuple[float,float]] | None = None) -> dict:
    """Derivative-free coordinate search with deterministic step reduction."""
    x=list(map(float,initial))
    if not x or step<=0 or tolerance<=0 or max_iter<1: raise ValueError("invalid optimizer inputs")
    if bounds is not None and (len(bounds)!=len(x) or any(a>b for a,b in bounds)): raise ValueError("invalid bounds")
    if bounds:
        x=[min(bounds[i][1],max(bounds[i][0],v)) for i,v in enumerate(x)]
    value=float(function(x)); current=step; converged=False
    for iteration in range(1,max_iter+1):
        improved=False
        for i in range(len(x)):
            for direction in (-1,1):
                candidate=x[:]; candidate[i]+=direction*current
                if bounds: candidate[i]=min(bounds[i][1],max(bounds[i][0],candidate[i]))
                trial=float(function(candidate))
                if trial<value:
                    x,value=candidate,trial; improved=True
        if not improved: current*=0.5
        if current<tolerance: converged=True; break
    return {"point":tuple(x),"value":value,"step":current,"iterations":iteration,"converged":converged}


def newton_raphson(function: Callable[[float],float], derivative: Callable[[float],float],
                   initial: float, *, tolerance: float = 1e-10, max_iter: int = 100,
                   bounds: tuple[float,float] | None = None) -> dict:
    """Newton root solver with optional hard bounds and finite-value checks."""
    x=float(initial)
    if tolerance<=0 or max_iter<1: raise ValueError("invalid solver controls")
    for iteration in range(1,max_iter+1):
        fx=float(function(x)); d=float(derivative(x))
        if not isfinite(fx+d) or abs(d)<1e-15: break
        candidate=x-fx/d
        if bounds: candidate=min(bounds[1],max(bounds[0],candidate))
        if abs(candidate-x)<tolerance:
            return {"root":candidate,"value":float(function(candidate)),"iterations":iteration,"converged":True}
        x=candidate
    return {"root":x,"value":float(function(x)),"iterations":iteration,"converged":abs(float(function(x)))<tolerance}


def line_search_backtracking(objective: Callable[[Sequence[float]],float], point: Sequence[float],
                             direction: Sequence[float], *, initial_step: float = 1.0,
                             armijo: float = 1e-4, contraction: float = 0.5,
                             min_step: float = 1e-10) -> dict:
    """Armijo backtracking using a finite-difference directional derivative."""
    x=list(map(float,point)); d=list(map(float,direction))
    if len(x)!=len(d) or not x or initial_step<=0 or not 0<contraction<1 or not 0<armijo<1:
        raise ValueError("invalid line-search inputs")
    f0=float(objective(x)); grad=finite_difference_gradient(objective,x)
    slope=sum(a*b for a,b in zip(grad,d))
    if slope>=0: raise ValueError("direction must be a descent direction")
    alpha=initial_step
    while alpha>=min_step:
        candidate=[a+alpha*b for a,b in zip(x,d)]
        value=float(objective(candidate))
        if value<=f0+armijo*alpha*slope:
            return {"point":tuple(candidate),"value":value,"step":alpha,"accepted":True}
        alpha*=contraction
    return {"point":tuple(x),"value":f0,"step":0.0,"accepted":False}


def grid_search(function: Callable[[float],float], candidates: Sequence[float]) -> dict:
    if not candidates: raise ValueError("candidates must not be empty")
    scored=[(float(function(float(x))),float(x)) for x in candidates]
    value,x=min(scored)
    return {"x":x,"value":value,"scores":tuple((x,v) for v,x in scored)}
