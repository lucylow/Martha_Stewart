"""Autoregressive forecasts and residual whiteness diagnostics."""
from __future__ import annotations
from math import log
from typing import Sequence
from ..core import finite_series,mean,variance
from ..regression import ols


def fit_ar(values: Sequence[float], order: int = 1, *, include_intercept: bool = True) -> dict:
    """Fit AR(p) by conditional least squares; returns in-sample fitted values."""
    x=finite_series(values,name="values",min_length=10)
    if order<1 or order>=len(x)//2: raise ValueError("order must be positive and less than half the sample")
    target=x[order:]
    features=[[x[t-lag] for t in range(order,len(x))] for lag in range(1,order+1)]
    fit=ols(features,target,add_intercept=include_intercept)
    residual=list(fit.residuals); n=len(target); k=order+int(include_intercept)
    sse=sum(v*v for v in residual); sigma2=sse/n
    return {"intercept":fit.coefficients[0] if include_intercept else 0.0,
            "coefficients":tuple(fit.coefficients[1:] if include_intercept else fit.coefficients),
            "fitted":fit.fitted,"residuals":fit.residuals,"sigma2":sigma2,
            "aic":n*log(max(sigma2,1e-30))+2*k,
            "bic":n*log(max(sigma2,1e-30))+k*log(n),"nobs":n,"order":order}


def forecast_ar(model: dict, history: Sequence[float], horizon: int = 1) -> tuple[float,...]:
    x=finite_series(history,name="history")
    coefs=list(model["coefficients"]); p=len(coefs)
    if len(x)<p or horizon<1: raise ValueError("history too short or horizon invalid")
    intercept=float(model.get("intercept",0.0)); path=[]
    for _ in range(horizon):
        value=intercept+sum(coefs[j]*x[-1-j] for j in range(p))
        x.append(value); path.append(value)
    return tuple(path)


def autocorrelation(values: Sequence[float], max_lag: int) -> tuple[float,...]:
    x=finite_series(values,name="values",min_length=3)
    if max_lag<0 or max_lag>=len(x): raise ValueError("invalid max_lag")
    mu=mean(x); den=sum((v-mu)**2 for v in x)
    if den==0: return tuple(1.0 if lag==0 else 0.0 for lag in range(max_lag+1))
    return tuple(sum((x[t]-mu)*(x[t-lag]-mu) for t in range(lag,len(x)))/den for lag in range(max_lag+1))


def ljung_box(values: Sequence[float], lags: int = 10, model_df: int = 0) -> dict:
    """Ljung-Box Q with chi-square survival approximation for integral df."""
    x=finite_series(values,name="residuals",min_length=10)
    if lags<1 or lags>=len(x) or model_df<0: raise ValueError("invalid lags or model degrees of freedom")
    rho=autocorrelation(x,lags)[1:]
    q=len(x)*(len(x)+2)*sum(r*r/(len(x)-k) for k,r in enumerate(rho,1))
    df=lags-model_df
    if df<=0: raise ValueError("lags must exceed model_df")
    # Integer-df chi-square survival via regularized gamma series/continued fraction.
    from ..statistics import chi_square_survival
    p=chi_square_survival(q,df)
    return {"q_statistic":q,"degrees_freedom":df,"p_value_approx":p,"lags":lags}


def forecast_errors(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    a=finite_series(actual); p=finite_series(predicted)
    if len(a)!=len(p): raise ValueError("series lengths must match")
    errors=[x-y for x,y in zip(a,p)]
    mae=sum(abs(v) for v in errors)/len(errors)
    rmse=(sum(v*v for v in errors)/len(errors))**0.5
    mape=sum(abs(e)/abs(y) for e,y in zip(errors,a) if y!=0)/max(1,sum(y!=0 for y in a))
    return {"mae":mae,"rmse":rmse,"mape":mape,"bias":mean(errors),"errors":tuple(errors)}


def expanding_forecast(values: Sequence[float], order: int, initial_train: int,
                       horizon: int = 1, refit_step: int = 1) -> list[dict]:
    """Expanding-window one-origin-at-a-time out-of-sample forecast path."""
    x=finite_series(values)
    if initial_train<=order or initial_train+horizon>len(x) or refit_step<1: raise ValueError("invalid forecast controls")
    results=[]
    for origin in range(initial_train,len(x)-horizon+1,refit_step):
        model=fit_ar(x[:origin],order)
        pred=forecast_ar(model,x[:origin],horizon)
        results.append({"origin":origin,"forecast":pred,"actual":tuple(x[origin:origin+horizon])})
    return results
