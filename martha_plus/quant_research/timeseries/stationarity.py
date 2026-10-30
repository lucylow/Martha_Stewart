"""Stationarity diagnostics with explicitly approximate reference distributions.

ADF is estimated through an auxiliary OLS regression.  The returned ADF
p-value is a rough interpolation against published critical-value anchors;
serious inference should use a vetted econometrics package and specification.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Sequence
from ..core import finite_series, mean
from ..regression import ols

@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value_approx: float
    critical_values: dict[str,float]
    observations: int
    lags: int
    reject_5pct: bool
    interpretation: str


def _interp_p(stat: float, anchors: Sequence[tuple[float,float]]) -> float:
    points = sorted(anchors)
    if stat <= points[0][0]: return points[0][1]
    if stat >= points[-1][0]: return points[-1][1]
    for (x0,p0),(x1,p1) in zip(points,points[1:]):
        if x0 <= stat <= x1:
            return max(0.0,min(1.0,p0+(p1-p0)*(stat-x0)/(x1-x0)))
    return 1.0


def adf_test(values: Sequence[float], *, max_lags: int | None = None,
             regression: str = "c") -> TestResult:
    """Augmented Dickey-Fuller test, with c/ct/n deterministic terms.

    The statistic is the t-ratio for the lagged-level coefficient.  Critical
    values are conventional large-sample approximations, not finite-sample
    MacKinnon response-surface values.
    """
    x = finite_series(values, name="values", min_length=12)
    if regression not in {"c","ct","n"}:
        raise ValueError("regression must be 'c', 'ct', or 'n'")
    lag_count = max(0, min(len(x)//2-2, int(12*(len(x)/100)**0.25))) if max_lags is None else max_lags
    if lag_count < 0 or lag_count >= len(x)-5:
        raise ValueError("max_lags is invalid for sample size")
    dx = [x[i]-x[i-1] for i in range(1,len(x))]
    features = [x[lag_count:len(x)-1]]
    names = ["lagged_level"]
    for lag in range(1,lag_count+1):
        features.append(dx[lag_count-lag:len(dx)-lag])
        names.append(f"diff_lag_{lag}")
    if regression in {"c","ct"}:
        features.append([1.0]*len(features[0])); names.append("constant")
    if regression == "ct":
        features.append([float(i) for i in range(len(features[0]))]); names.append("trend")
    target = dx[lag_count:]
    fit = ols(features,target,add_intercept=False,names=names)
    idx = names.index("lagged_level")
    stat = fit.t_statistics[idx]
    critical = ({"1%":-3.43,"5%":-2.86,"10%":-2.57} if regression == "c" else
                {"1%":-3.96,"5%":-3.41,"10%":-3.12} if regression == "ct" else
                {"1%":-2.58,"5%":-1.95,"10%":-1.62})
    anchors = [(-4.5,0.005),(-3.43,0.01),(-2.86,0.05),(-2.57,0.10),(-2.0,0.25),(-1.0,0.75),(0.0,0.99)]
    p = _interp_p(stat,anchors)
    reject = stat < critical["5%"]
    return TestResult(stat,p,critical,len(target),lag_count,reject,
                      "evidence against a unit root" if reject else "unit-root null not rejected")


def kpss_test(values: Sequence[float], *, regression: str = "c", bandwidth: int | None = None) -> TestResult:
    """KPSS test with Bartlett HAC long-run variance estimate."""
    x = finite_series(values,name="values",min_length=10)
    if regression not in {"c","ct"}:
        raise ValueError("regression must be 'c' or 'ct'")
    n=len(x)
    if regression == "c":
        residual=[v-mean(x) for v in x]
        cv={"1%":0.739,"5%":0.463,"10%":0.347}
    else:
        t=list(range(n)); mt=(n-1)/2; mx=mean(x)
        den=sum((v-mt)**2 for v in t)
        slope=sum((a-mx)*(b-mt) for a,b in zip(x,t))/den
        intercept=mx-slope*mt
        residual=[v-intercept-slope*i for i,v in enumerate(x)]
        cv={"1%":0.216,"5%":0.146,"10%":0.119}
    lags=max(0,int(4*(n/100)**0.25)) if bandwidth is None else bandwidth
    if lags<0 or lags>=n: raise ValueError("bandwidth must be in [0,n-1]")
    cumulative=[]; s=0.0
    for e in residual: s+=e; cumulative.append(s)
    eta=sum(v*v for v in cumulative)/(n*n)
    gamma0=sum(v*v for v in residual)/n
    long_run=gamma0
    for lag in range(1,lags+1):
        gamma=sum(residual[t]*residual[t-lag] for t in range(lag,n))/n
        long_run += 2.0*(1.0-lag/(lags+1.0))*gamma
    stat=eta/max(long_run,1e-15)
    p=0.01 if stat>cv["1%"] else 0.05 if stat>cv["5%"] else 0.10 if stat>cv["10%"] else 0.5
    reject=stat>cv["5%"]
    return TestResult(stat,p,cv,n,lags,reject,
                      "evidence against stationarity" if reject else "stationarity null not rejected")


def variance_ratio(values: Sequence[float], horizon: int = 2, *, robust: bool = True) -> dict:
    """Lo-MacKinlay style variance ratio with heteroskedasticity correction."""
    x=finite_series(values,name="returns",min_length=10)
    if horizon<2 or horizon>=len(x)//2: raise ValueError("horizon must be >=2 and below half the sample")
    mu=mean(x); n=len(x)
    demean=[v-mu for v in x]
    var1=sum(v*v for v in demean)/(n-1)
    agg=[sum(x[i:i+horizon]) for i in range(n-horizon+1)]
    mu_h=mean(agg)
    varh=sum((v-mu_h)**2 for v in agg)/(len(agg)-1)
    ratio=varh/(horizon*var1) if var1>0 else float("nan")
    if robust:
        # Heteroskedasticity-consistent variance of the variance ratio.
        z=[v*v for v in demean]
        denom=sum(z)**2
        theta=sum((2.0*(horizon-j)/horizon)**2 * sum((z[t]*z[t-j]) for t in range(j,n))
                  for j in range(1,horizon))/max(denom,1e-30)
        se=sqrt(max(theta,1e-15))
    else:
        se=sqrt(2.0*(2*horizon-1)*(horizon-1)/(3.0*horizon*n))
    statistic=(ratio-1.0)/se
    return {"variance_ratio":ratio,"statistic":statistic,"standard_error":se,
            "horizon":horizon,"observations":n,"robust":robust}
