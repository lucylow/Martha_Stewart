"""Benchmark-relative performance, capture ratios and rolling active diagnostics."""
from __future__ import annotations
from math import sqrt
from typing import Sequence
from ..core import finite_series,mean,variance
from .metrics import sharpe_ratio,information_ratio,annualized_return


def up_down_capture(portfolio: Sequence[float], benchmark: Sequence[float]) -> dict:
    p=finite_series(portfolio); b=finite_series(benchmark)
    if len(p)!=len(b): raise ValueError("portfolio and benchmark lengths must match")
    up=[i for i,v in enumerate(b) if v>0]; down=[i for i,v in enumerate(b) if v<0]
    def compound(indices,values):
        return __import__('math').prod(1+values[i] for i in indices)**(1/len(indices))-1 if indices else 0.0
    up_p,up_b=compound(up,p),compound(up,b); down_p,down_b=compound(down,p),compound(down,b)
    return {"up_capture":up_p/up_b if up_b else float("nan"),
            "down_capture":down_p/down_b if down_b else float("nan"),
            "up_observations":len(up),"down_observations":len(down),
            "up_relative_return":up_p-up_b,"down_relative_return":down_p-down_b}


def beta_adjusted_alpha(portfolio: Sequence[float], benchmark: Sequence[float],
                        risk_free: float = 0.0, periods_per_year: float = 252.0) -> dict:
    from ..regression import ols
    p=finite_series(portfolio); b=finite_series(benchmark)
    if len(p)!=len(b) or periods_per_year<=0: raise ValueError("inputs must align")
    fit=ols([[v-risk_free for v in b]],[v-risk_free for v in p])
    alpha=fit.coefficients[0]; beta=fit.coefficients[1]
    return {"periodic_alpha":alpha,"annualized_alpha":alpha*periods_per_year,
            "beta":beta,"r_squared":fit.r_squared,"residual_volatility":sqrt(variance(fit.residuals))*sqrt(periods_per_year)}


def rolling_information_ratio(portfolio: Sequence[float], benchmark: Sequence[float],
                              window: int, periods_per_year: float = 252.0) -> list[float | None]:
    p=finite_series(portfolio); b=finite_series(benchmark)
    if len(p)!=len(b) or window<3 or window>len(p): raise ValueError("invalid window or inputs")
    return [None]*(window-1)+[information_ratio(p[i-window+1:i+1],b[i-window+1:i+1],periods_per_year) for i in range(window-1,len(p))]


def active_return_decomposition(portfolio: Sequence[float], benchmark: Sequence[float]) -> dict:
    p=finite_series(portfolio); b=finite_series(benchmark)
    if len(p)!=len(b): raise ValueError("series lengths must match")
    active=[a-bv for a,bv in zip(p,b)]
    return {"mean_active_return":mean(active),"tracking_error":sqrt(variance(active)),
            "information_ratio_periodic":mean(active)/sqrt(variance(active)) if variance(active)>0 else float("inf"),
            "positive_active_fraction":sum(v>0 for v in active)/len(active),
            "cumulative_active_arithmetic":sum(active),"active_returns":tuple(active)}


def benchmark_summary(portfolio: Sequence[float], benchmark: Sequence[float],
                      periods_per_year: float = 252.0) -> dict:
    p=finite_series(portfolio); b=finite_series(benchmark)
    return {"portfolio_annualized_return":annualized_return(p,periods_per_year),
            "benchmark_annualized_return":annualized_return(b,periods_per_year),
            "portfolio_sharpe":sharpe_ratio(p,periods_per_year=periods_per_year),
            "benchmark_sharpe":sharpe_ratio(b,periods_per_year=periods_per_year),
            "active":active_return_decomposition(p,b),"capture":up_down_capture(p,b),
            "information_ratio":information_ratio(p,b,periods_per_year)}
