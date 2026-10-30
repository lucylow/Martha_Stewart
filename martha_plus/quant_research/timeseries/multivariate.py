"""Multivariate dependence, lead-lag and dynamic-correlation diagnostics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,correlation,mean,variance
from ..linalg import transpose


def rolling_correlation(x: Sequence[float], y: Sequence[float], window: int) -> list[float | None]:
    a=finite_series(x); b=finite_series(y)
    if len(a)!=len(b) or window<3 or window>len(a): raise ValueError("invalid series or window")
    return [None]*(window-1)+[correlation(a[i-window+1:i+1],b[i-window+1:i+1]) for i in range(window-1,len(a))]


def lead_lag_correlation(leader: Sequence[float], follower: Sequence[float], max_lag: int = 10) -> dict:
    """Correlation of leader[t] with follower[t+lag]; positive lag means lead."""
    x=finite_series(leader); y=finite_series(follower)
    if len(x)!=len(y) or max_lag<0 or max_lag>=len(x)-2: raise ValueError("invalid series or max_lag")
    values=[]
    for lag in range(-max_lag,max_lag+1):
        if lag>=0: a=x[:len(x)-lag or None]; b=y[lag:]
        else: a=x[-lag:]; b=y[:len(y)+lag]
        values.append({"lag":lag,"correlation":correlation(a,b)})
    best=max(values,key=lambda row:abs(row["correlation"]))
    return {"correlations":tuple(values),"strongest_lag":best["lag"],"strongest_correlation":best["correlation"]}


def dcc_filter(standardized_residuals: Sequence[Sequence[float]],
               a: float = 0.02, b: float = 0.95) -> dict:
    """Dynamic conditional correlation recursion with supplied coefficients.

    This filters DCC matrices; it does not estimate the parameters or infer a
    likelihood. Inputs are time x asset standardized residuals.
    """
    from ..risk.covariance import sample_covariance
    from ..linalg import transpose
    rows=[finite_series(row,name="standardized residual row") for row in standardized_residuals]
    if len(rows)<5 or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("residual panel must be rectangular")
    if a<0 or b<0 or a+b>=1: raise ValueError("DCC coefficients require a,b >= 0 and a+b < 1")
    s=sample_covariance(rows); q=[list(r) for r in s]; path=[]
    for z in rows:
        outer=[[z[i]*z[j] for j in range(len(z))] for i in range(len(z))]
        q=[[(1-a-b)*s[i][j]+a*outer[i][j]+b*q[i][j] for j in range(len(z))] for i in range(len(z))]
        sd=[sqrt(max(q[i][i],1e-15)) for i in range(len(z))]
        corr=[[q[i][j]/(sd[i]*sd[j]) for j in range(len(z))] for i in range(len(z))]
        path.append(tuple(tuple(row) for row in corr))
    return {"correlation_path":tuple(path),"unconditional_covariance":tuple(tuple(r) for r in s),"a":a,"b":b}


def rolling_cross_sectional_dispersion(returns: Sequence[Sequence[float]]) -> tuple[float,...]:
    rows=[finite_series(r) for r in returns]
    if not rows or any(len(r)!=len(rows[0]) for r in rows): raise ValueError("panel must be rectangular")
    return tuple(sqrt(variance(row)) for row in rows)


def correlation_break_statistic(x: Sequence[float], y: Sequence[float], split: int) -> dict:
    a=finite_series(x); b=finite_series(y)
    if len(a)!=len(b) or split<3 or split>len(a)-3: raise ValueError("invalid split")
    before=correlation(a[:split],b[:split]); after=correlation(a[split:],b[split:])
    # Fisher-z approximate standard error for independent samples.
    from ..core import normal_cdf
    import math
    z1=math.atanh(max(-0.999999,min(0.999999,before)))
    z2=math.atanh(max(-0.999999,min(0.999999,after)))
    se=sqrt(1/(split-3)+1/(len(a)-split-3)); z=(z1-z2)/se
    return {"correlation_before":before,"correlation_after":after,"fisher_z":z,
            "p_value_approx":2*(1-normal_cdf(abs(z))),"split":split}
