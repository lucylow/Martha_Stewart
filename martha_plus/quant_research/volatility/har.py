"""Heterogeneous autoregressive realized-volatility model."""
from __future__ import annotations
from math import log,exp
from typing import Sequence
from ..core import finite_series,mean
from ..regression import ols


def har_features(realized_variance: Sequence[float]) -> dict:
    """Construct daily, five-session and twenty-two-session log-RV features."""
    rv=finite_series(realized_variance,name="realized_variance",min_length=25)
    if any(v<=0 for v in rv): raise ValueError("realized variance values must be positive")
    logrv=[log(v) for v in rv]
    daily=[]; weekly=[]; monthly=[]
    for i in range(22,len(rv)):
        daily.append(logrv[i-1])
        weekly.append(mean(logrv[i-5:i]))
        monthly.append(mean(logrv[i-22:i]))
    return {"target":tuple(logrv[22:]),"daily":tuple(daily),"weekly":tuple(weekly),"monthly":tuple(monthly),"start_index":22}


def fit_har(realized_variance: Sequence[float]) -> dict:
    f=har_features(realized_variance)
    fit=ols([f["daily"],f["weekly"],f["monthly"]],f["target"],add_intercept=True,
            names=["intercept","daily","weekly","monthly"])
    return {"coefficients":fit.coefficients,"names":fit.names,"fitted_log_variance":fit.fitted,
            "residuals":fit.residuals,"r_squared":fit.r_squared,"nobs":fit.nobs,
            "start_index":f["start_index"],"model":"HAR-RV log-linear OLS"}


def forecast_har(model: dict, realized_variance_history: Sequence[float], horizon: int = 1) -> tuple[float,...]:
    rv=finite_series(realized_variance_history,name="realized_variance",min_length=22)
    if any(v<=0 for v in rv) or horizon<1: raise ValueError("positive realized variance and horizon required")
    beta=list(model["coefficients"]); logrv=[log(v) for v in rv]; result=[]
    for _ in range(horizon):
        d=logrv[-1]; w=mean(logrv[-5:]); m=mean(logrv[-22:])
        prediction=beta[0]+beta[1]*d+beta[2]*w+beta[3]*m
        logrv.append(prediction); result.append(exp(prediction))
    return tuple(result)


def har_volatility_path(model: dict, history: Sequence[float], horizon: int = 1) -> tuple[float,...]:
    return tuple(v**0.5 for v in forecast_har(model,history,horizon))
