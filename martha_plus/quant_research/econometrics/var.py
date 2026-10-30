"""Vector autoregression estimation, forecasting and Granger diagnostics."""
from __future__ import annotations
from math import log
from typing import Sequence
from ..core import finite_series,mean
from ..regression import ols
from ..linalg import transpose,matmul,eigen_symmetric
from ..statistics import chi_square_survival


def _var_design(series: Sequence[Sequence[float]], order: int):
    cols=[finite_series(c,name="series",min_length=10) for c in series]
    n=len(cols[0]); k=len(cols)
    if any(len(c)!=n for c in cols) or order<1 or order>=n//2: raise ValueError("invalid panel or VAR order")
    target=[[cols[j][t] for j in range(k)] for t in range(order,n)]
    features=[]
    for lag in range(1,order+1):
        for j in range(k): features.append([cols[j][t-lag] for t in range(order,n)])
    return cols,features,target


def fit_var(series: Sequence[Sequence[float]], order: int = 1, *, include_intercept: bool = True) -> dict:
    """Fit equation-by-equation OLS VAR(p); input is variable x time."""
    cols,features,target=_var_design(series,order)
    k=len(cols); fits=[]; residuals=[]; fitted=[]; coefs=[]
    for j in range(k):
        fit=ols(features,[r[j] for r in target],add_intercept=include_intercept)
        fits.append(fit); residuals.append(list(fit.residuals)); fitted.append(list(fit.fitted)); coefs.append(fit.coefficients)
    residual_panel=transpose(residuals)
    from ..risk.covariance import sample_covariance
    cov=sample_covariance(residual_panel)
    n=len(target); total_params=k*len(coefs[0]); logdet_approx=sum(log(max(cov[i][i],1e-30)) for i in range(k))
    return {"order":order,"variables":k,"coefficients":tuple(tuple(c) for c in coefs),
            "residuals":tuple(tuple(r) for r in residuals),"fitted":tuple(tuple(r) for r in fitted),
            "residual_covariance":tuple(tuple(r) for r in cov),"aic":logdet_approx+2*total_params/n,
            "bic":logdet_approx+log(n)*total_params/n,"nobs":n,"intercept":include_intercept,
            "last_values":tuple(tuple(cols[j][-order:]) for j in range(k))}


def forecast_var(model: dict, history: Sequence[Sequence[float]], horizon: int = 1) -> tuple[tuple[float,...],...]:
    cols=[list(map(float,c)) for c in history]; k=int(model["variables"]); p=int(model["order"])
    if len(cols)!=k or any(len(c)<p for c in cols) or horizon<1: raise ValueError("history or horizon invalid")
    coeffs=model["coefficients"]; intercept=model["intercept"]; out=[]
    for _ in range(horizon):
        row=[]
        for j in range(k):
            beta=coeffs[j]; pos=1 if intercept else 0
            value=beta[0] if intercept else 0.0
            for lag in range(1,p+1):
                for variable in range(k):
                    value+=beta[pos]*cols[variable][-lag]; pos+=1
            row.append(value)
        for j in range(k): cols[j].append(row[j])
        out.append(tuple(row))
    return tuple(out)


def granger_causality(series: Sequence[Sequence[float]], cause: int, effect: int,
                      order: int = 1) -> dict:
    """Likelihood-ratio Granger diagnostic; asymptotic chi-square p-value."""
    cols,features,target=_var_design(series,order); k=len(cols)
    if cause<0 or cause>=k or effect<0 or effect>=k or cause==effect: raise ValueError("invalid cause/effect index")
    unrestricted=ols(features,[r[effect] for r in target],add_intercept=True)
    restricted_features=[]; restricted_names=[]
    for lag in range(1,order+1):
        for variable in range(k):
            if variable!=cause: restricted_features.append(features[(lag-1)*k+variable])
    restricted=ols(restricted_features,[r[effect] for r in target],add_intercept=True)
    rss_u=sum(v*v for v in unrestricted.residuals); rss_r=sum(v*v for v in restricted.residuals)
    lr=len(target)*log(max(rss_r,1e-30)/max(rss_u,1e-30))
    df=order; p=chi_square_survival(max(0.0,lr),df)
    return {"cause_index":cause,"effect_index":effect,"order":order,"lr_statistic":lr,
            "degrees_freedom":df,"p_value_approx":p,"restricted_rss":rss_r,"unrestricted_rss":rss_u,
            "interpretation":"evidence of predictive content" if p<0.05 else "insufficient evidence at 5%"}


def choose_var_order(series: Sequence[Sequence[float]], max_order: int = 5, criterion: str = "bic") -> dict:
    if criterion not in {"aic","bic"}: raise ValueError("criterion must be aic or bic")
    results=[fit_var(series,p) for p in range(1,max_order+1)]
    best=min(results,key=lambda m:m[criterion])
    return {"selected_order":best["order"],"criterion":criterion,"score":best[criterion],
            "candidates":tuple({"order":m["order"],"aic":m["aic"],"bic":m["bic"]} for m in results)}
