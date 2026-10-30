"""Regression diagnostics for collinearity, heteroskedasticity and breaks."""
from __future__ import annotations
from math import log
from typing import Sequence
from ..core import finite_series,mean,variance
from ..regression import ols,design_matrix
from ..linalg import transpose,inverse,matmul
from ..statistics import chi_square_survival


def variance_inflation_factors(features: Sequence[Sequence[float]]) -> tuple[float,...]:
    """Compute VIF_j = diagonal of inverse feature correlation matrix."""
    from ..risk.covariance import sample_covariance,correlation_matrix
    cols=[finite_series(c,name="feature",min_length=4) for c in features]
    if len(cols)<2 or any(len(c)!=len(cols[0]) for c in cols): raise ValueError("need >=2 aligned features")
    standardized=[]
    from ..core import stddev
    for c in cols:
        sd=stddev(c)
        if sd==0: raise ValueError("constant feature has undefined VIF")
        standardized.append([(v-mean(c))/sd for v in c])
    corr=sample_covariance(list(zip(*standardized)))
    inv=inverse(corr,ridge=1e-12)
    return tuple(max(0.0,inv[i][i]) for i in range(len(inv)))


def durbin_watson(residuals: Sequence[float]) -> float:
    e=finite_series(residuals,name="residuals",min_length=2)
    den=sum(v*v for v in e)
    return sum((e[i]-e[i-1])**2 for i in range(1,len(e)))/den if den>0 else 0.0


def breusch_pagan(residuals: Sequence[float], features: Sequence[Sequence[float]]) -> dict:
    """Breusch-Pagan LM test using an auxiliary regression of squared residuals."""
    e=finite_series(residuals,name="residuals",min_length=5)
    cols=[finite_series(c) for c in features]
    if any(len(c)!=len(e) for c in cols) or not cols: raise ValueError("feature dimensions mismatch")
    fit=ols(cols,[v*v for v in e],add_intercept=True)
    lm=len(e)*fit.r_squared; df=len(cols)
    return {"lm_statistic":lm,"degrees_freedom":df,"p_value_approx":chi_square_survival(max(0,lm),df),
            "auxiliary_r_squared":fit.r_squared}


def arch_lm_test(residuals: Sequence[float], lags: int = 5) -> dict:
    """Engle ARCH-LM test based on lagged squared residuals."""
    e=finite_series(residuals,name="residuals",min_length=10)
    if lags<1 or lags>=len(e)//2: raise ValueError("invalid ARCH lag count")
    sq=[v*v for v in e]; target=sq[lags:]
    features=[[sq[t-j] for t in range(lags,len(sq))] for j in range(1,lags+1)]
    fit=ols(features,target,add_intercept=True)
    lm=len(target)*fit.r_squared
    return {"lm_statistic":lm,"degrees_freedom":lags,"p_value_approx":chi_square_survival(max(0,lm),lags),
            "auxiliary_r_squared":fit.r_squared,"lags":lags}


def chow_break_test(features: Sequence[Sequence[float]], target: Sequence[float], split: int,
                    *, add_intercept: bool = True) -> dict:
    """Chow-style parameter stability test using pooled and split regressions."""
    y=finite_series(target,min_length=10); cols=[finite_series(c) for c in features]
    if any(len(c)!=len(y) for c in cols) or split<3 or split>len(y)-3: raise ValueError("invalid split or features")
    pooled=ols(cols,y,add_intercept=add_intercept)
    first=ols([c[:split] for c in cols],y[:split],add_intercept=add_intercept)
    second=ols([c[split:] for c in cols],y[split:],add_intercept=add_intercept)
    rss_p=sum(v*v for v in pooled.residuals); rss_1=sum(v*v for v in first.residuals); rss_2=sum(v*v for v in second.residuals)
    k=len(pooled.coefficients); n=len(y)
    numerator=max(0.0,(rss_p-rss_1-rss_2)/k); denominator=(rss_1+rss_2)/(n-2*k)
    statistic=numerator/denominator if denominator>0 else float("inf")
    return {"f_statistic":statistic,"numerator_df":k,"denominator_df":n-2*k,
            "pooled_rss":rss_p,"split_rss":rss_1+rss_2,"split":split,
            "note":"F reference distribution is not evaluated in the standard-library implementation"}


def white_test(residuals: Sequence[float], features: Sequence[Sequence[float]]) -> dict:
    """White heteroskedasticity test with squares and pairwise feature products."""
    e=finite_series(residuals,min_length=8); cols=[finite_series(c) for c in features]
    if not cols or any(len(c)!=len(e) for c in cols): raise ValueError("invalid feature panel")
    expanded=list(cols)
    expanded += [[v*v for v in c] for c in cols]
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            expanded.append([a*b for a,b in zip(cols[i],cols[j])])
    fit=ols(expanded,[v*v for v in e],add_intercept=True)
    df=len(fit.coefficients)-1; lm=len(e)*fit.r_squared
    return {"lm_statistic":lm,"degrees_freedom":df,"p_value_approx":chi_square_survival(max(0,lm),df),
            "auxiliary_r_squared":fit.r_squared,"auxiliary_regressors":df}


def residual_summary(residuals: Sequence[float]) -> dict:
    e=finite_series(residuals,name="residuals")
    return {"mean":mean(e),"variance":variance(e) if len(e)>1 else 0.0,
            "durbin_watson":durbin_watson(e),"arch_lm_5":arch_lm_test(e,min(5,max(1,len(e)//4))) if len(e)>=20 else None}
