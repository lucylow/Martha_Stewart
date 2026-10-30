"""Forecast-loss and Value-at-Risk backtesting statistics."""
from __future__ import annotations
from math import log
from typing import Sequence
from ..core import finite_series,mean
from ..statistics import chi_square_survival


def qlike_loss(actual_variance: Sequence[float], forecast_variance: Sequence[float]) -> float:
    actual=finite_series(actual_variance); forecast=finite_series(forecast_variance)
    if len(actual)!=len(forecast) or any(v<=0 for v in actual+forecast): raise ValueError("variance paths must align and be positive")
    return mean([log(f)+a/f for a,f in zip(actual,forecast)])


def volatility_forecast_losses(actual_variance: Sequence[float], forecast_variance: Sequence[float]) -> dict:
    actual=finite_series(actual_variance); forecast=finite_series(forecast_variance)
    if len(actual)!=len(forecast) or any(v<0 for v in actual+forecast): raise ValueError("variance paths must align and be nonnegative")
    errors=[a-f for a,f in zip(actual,forecast)]
    mse=mean([v*v for v in errors]); mae=mean([abs(v) for v in errors])
    return {"mse":mse,"rmse":mse**0.5,"mae":mae,"qlike":qlike_loss(actual,forecast),"errors":tuple(errors)}


def kupiec_pof_test(losses: Sequence[float], var_forecasts: Sequence[float],
                    confidence: float = 0.99) -> dict:
    """Kupiec unconditional-coverage likelihood-ratio test."""
    x=finite_series(losses,name="losses"); q=finite_series(var_forecasts,name="var_forecasts")
    if len(x)!=len(q) or not 0.5<confidence<1: raise ValueError("inputs must align and confidence in (0.5,1)")
    breaches=sum(a>b for a,b in zip(x,q)); n=len(x); p=breaches/n; expected=1-confidence
    def term(prob,count):
        if count==0: return 0.0
        if prob<=0: return float("-inf")
        return count*log(prob)
    ll_null=term(expected,breaches)+term(1-expected,n-breaches)
    ll_alt=term(p,breaches)+term(1-p,n-breaches)
    lr=max(0.0,-2*(ll_null-ll_alt))
    return {"breaches":breaches,"observations":n,"observed_rate":p,"expected_rate":expected,
            "lr_statistic":lr,"p_value_approx":chi_square_survival(lr,1),"confidence":confidence}


def christoffersen_independence_test(losses: Sequence[float], var_forecasts: Sequence[float]) -> dict:
    """First-order Markov independence test for VaR breach indicators."""
    x=finite_series(losses); q=finite_series(var_forecasts)
    if len(x)!=len(q) or len(x)<5: raise ValueError("aligned inputs >= 5 required")
    hits=[int(a>b) for a,b in zip(x,q)]
    n00=n01=n10=n11=0
    for a,b in zip(hits,hits[1:]):
        if a==0 and b==0:n00+=1
        elif a==0 and b==1:n01+=1
        elif a==1 and b==0:n10+=1
        else:n11+=1
    p01=n01/max(1,n00+n01); p11=n11/max(1,n10+n11); p=(n01+n11)/max(1,n00+n01+n10+n11)
    def loglik(prob,success,failure):
        return (success*log(prob) if success else 0.0)+(failure*log(1-prob) if failure else 0.0)
    ll0=loglik(p,n01+n11,n00+n10)
    ll1=loglik(p01,n01,n00)+loglik(p11,n11,n10)
    lr=max(0.0,2*(ll1-ll0))
    return {"lr_statistic":lr,"p_value_approx":chi_square_survival(lr,1),"n00":n00,"n01":n01,"n10":n10,"n11":n11,
            "transition_0_to_1":p01,"transition_1_to_1":p11}


def conditional_coverage_test(losses: Sequence[float], var_forecasts: Sequence[float],
                              confidence: float = 0.99) -> dict:
    pof=kupiec_pof_test(losses,var_forecasts,confidence)
    ind=christoffersen_independence_test(losses,var_forecasts)
    lr=pof["lr_statistic"]+ind["lr_statistic"]
    return {"lr_statistic":lr,"degrees_freedom":2,"p_value_approx":chi_square_survival(lr,2),
            "unconditional_coverage":pof,"independence":ind}


def quantile_loss(actual: Sequence[float], forecast: Sequence[float], quantile_level: float) -> float:
    a=finite_series(actual); f=finite_series(forecast)
    if len(a)!=len(f) or not 0<quantile_level<1: raise ValueError("invalid inputs or quantile")
    errors=[x-y for x,y in zip(a,f)]
    return mean([quantile_level*e if e>=0 else (quantile_level-1)*e for e in errors])
