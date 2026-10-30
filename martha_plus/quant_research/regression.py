"""OLS, ridge regression, robust covariance and time-series HAC inference."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Sequence
from .core import finite_series, mean, variance
from .linalg import transpose, matmul, matvec, solve, inverse, dot

@dataclass(frozen=True)
class OLSResult:
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    t_statistics: tuple[float, ...]
    fitted: tuple[float, ...]
    residuals: tuple[float, ...]
    r_squared: float
    adjusted_r_squared: float
    nobs: int
    rank: int
    names: tuple[str, ...]


def design_matrix(features: Sequence[Sequence[float]], *, add_intercept: bool = True) -> list[list[float]]:
    if not features:
        raise ValueError("features must not be empty")
    cols = len(features)
    if cols == 0:
        raise ValueError("features must contain at least one column")
    n = len(features[0])
    if n == 0 or any(len(c) != n for c in features):
        raise ValueError("feature columns must be non-empty and equal length")
    x = [[float(features[j][i]) for j in range(cols)] for i in range(n)]
    if add_intercept:
        x = [[1.0] + row for row in x]
    if not all(__import__('math').isfinite(v) for row in x for v in row):
        raise ValueError("features must be finite")
    return x


def ols(features: Sequence[Sequence[float]], target: Sequence[float], *,
        add_intercept: bool = True, names: Sequence[str] | None = None,
        ridge: float = 0.0) -> OLSResult:
    y = finite_series(target, name="target", min_length=3)
    x = design_matrix(features, add_intercept=add_intercept)
    if len(x) != len(y):
        raise ValueError("feature rows must match target length")
    n, p = len(x), len(x[0])
    if n <= p:
        raise ValueError("OLS requires more observations than parameters")
    xt = transpose(x)
    xtx = matmul(xt, x)
    beta = solve(xtx, matvec(xt, y), ridge=ridge)
    fitted = matvec(x, beta)
    residuals = [a-b for a,b in zip(y, fitted)]
    sse = sum(e*e for e in residuals)
    tss = sum((v-mean(y))**2 for v in y)
    r2 = 1.0 if tss == 0 and sse == 0 else (0.0 if tss == 0 else 1.0-sse/tss)
    adj = 1.0-(1.0-r2)*(n-1)/(n-p) if n > p else float("nan")
    sigma2 = sse/(n-p)
    inv_xtx = inverse(xtx, ridge=ridge)
    ses = [sqrt(max(0.0, sigma2*inv_xtx[i][i])) for i in range(p)]
    ts = [0.0 if se == 0 else b/se for b,se in zip(beta,ses)]
    default_names = (["intercept"] if add_intercept else []) + [f"x{i+1}" for i in range(len(features))]
    if names is not None:
        provided = list(names)
        if add_intercept and len(provided) == p-1:
            provided = ["intercept"] + provided
        if len(provided) != p:
            raise ValueError("names length must match number of coefficients")
        default_names = provided
    return OLSResult(tuple(beta), tuple(ses), tuple(ts), tuple(fitted), tuple(residuals),
                      r2, adj, n, p, tuple(default_names))


def ridge_regression(features: Sequence[Sequence[float]], target: Sequence[float],
                     penalty: float, *, add_intercept: bool = True) -> tuple[float, ...]:
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    y = finite_series(target)
    x = design_matrix(features, add_intercept=add_intercept)
    if len(x) != len(y):
        raise ValueError("feature rows must match target")
    xt = transpose(x)
    gram = matmul(xt, x)
    if add_intercept:
        for i in range(1, len(gram)):
            gram[i][i] += penalty
    else:
        for i in range(len(gram)):
            gram[i][i] += penalty
    return tuple(solve(gram, matvec(xt, y)))


def hac_covariance(features: Sequence[Sequence[float]], residuals: Sequence[float],
                   bandwidth: int | None = None) -> list[list[float]]:
    """Newey-West sandwich covariance for a supplied design and residual vector."""
    x = [list(map(float,row)) for row in features]
    e = finite_series(residuals, name="residuals", min_length=3)
    if len(x) != len(e) or not x or any(len(row) != len(x[0]) for row in x):
        raise ValueError("design and residual dimensions do not match")
    n, p = len(x), len(x[0])
    lags = max(0, int(4*(n/100)**(2/9))) if bandwidth is None else bandwidth
    if lags < 0 or lags >= n:
        raise ValueError("bandwidth must be in [0,n-1]")
    xtx_inv = inverse(matmul(transpose(x), x), ridge=1e-12)
    meat = [[0.0]*p for _ in range(p)]
    scores = [[x[t][j]*e[t] for j in range(p)] for t in range(n)]
    for t in range(n):
        for i in range(p):
            for j in range(p):
                meat[i][j] += scores[t][i]*scores[t][j]
    for lag in range(1, lags+1):
        weight = 1.0-lag/(lags+1.0)
        for t in range(lag,n):
            for i in range(p):
                for j in range(p):
                    v = weight*scores[t][i]*scores[t-lag][j]
                    meat[i][j] += v
                    meat[j][i] += v
    return matmul(matmul(xtx_inv, meat), xtx_inv)


def ols_hac(features: Sequence[Sequence[float]], target: Sequence[float], *,
            add_intercept: bool = True, bandwidth: int | None = None) -> dict:
    fit = ols(features, target, add_intercept=add_intercept)
    x = design_matrix(features, add_intercept=add_intercept)
    cov = hac_covariance(x, fit.residuals, bandwidth)
    se = tuple(sqrt(max(0.0,cov[i][i])) for i in range(len(cov)))
    t = tuple(0.0 if s == 0 else b/s for b,s in zip(fit.coefficients,se))
    return {"fit": fit, "hac_covariance": tuple(tuple(r) for r in cov),
            "hac_standard_errors": se, "hac_t_statistics": t}


def rolling_ols(features: Sequence[Sequence[float]], target: Sequence[float], window: int,
                *, add_intercept: bool = True) -> list[OLSResult | None]:
    y = finite_series(target)
    if window < 3 or window > len(y):
        raise ValueError("window must be between 3 and length of target")
    if any(len(col) != len(y) for col in features):
        raise ValueError("feature length mismatch")
    out: list[OLSResult | None] = [None]*(window-1)
    for end in range(window,len(y)+1):
        out.append(ols([c[end-window:end] for c in features], y[end-window:end], add_intercept=add_intercept))
    return out


def recursive_least_squares(features: Sequence[Sequence[float]], target: Sequence[float],
                            *, forgetting: float = 1.0, delta: float = 1000.0,
                            add_intercept: bool = True) -> list[tuple[float,...]]:
    """RLS coefficient path; forgetting < 1 discounts older observations."""
    y = finite_series(target)
    if not 0 < forgetting <= 1 or delta <= 0:
        raise ValueError("forgetting must be in (0,1], delta must be positive")
    x = design_matrix(features, add_intercept=add_intercept)
    if len(x) != len(y):
        raise ValueError("feature rows must match target")
    p = len(x[0]); beta = [0.0]*p
    cov = [[delta if i==j else 0.0 for j in range(p)] for i in range(p)]
    path = []
    for row, observed in zip(x,y):
        px = matvec(cov,row)
        denom = forgetting + dot(row,px)
        gain = [v/denom for v in px]
        error = observed-dot(beta,row)
        beta = [b+g*error for b,g in zip(beta,gain)]
        cov = [[(cov[i][j]-gain[i]*px[j])/forgetting for j in range(p)] for i in range(p)]
        path.append(tuple(beta))
    return path
