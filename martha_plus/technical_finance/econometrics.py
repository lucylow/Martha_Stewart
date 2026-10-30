"""Financial econometrics: OU calibration, pairs diagnostics and Kalman hedge."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import FinanceInputError, Number, finite, mean, variance, vector
from .regression import ols
from .statistics import autocorrelation


@dataclass(frozen=True)
class OUParameters:
    mean_level: float
    mean_reversion_speed: float
    volatility: float
    half_life: float
    residuals: tuple[float, ...]


def ar1_parameters(values: Sequence[Number]) -> dict[str, float]:
    """Fit x[t] = intercept + phi*x[t-1] + error and return AR(1) diagnostics."""
    data = vector(values, min_length=4)
    result = ols(data[1:], (data[:-1],), intercept=True)
    intercept, phi = result.coefficients
    if abs(phi) >= 1:
        half_life = math.inf if phi > 0 else 0.0
    elif phi <= 0:
        half_life = 0.0
    else:
        half_life = math.log(0.5)/math.log(phi)
    return {"intercept": intercept, "phi": phi, "r_squared": result.r_squared,
            "residual_std": math.sqrt(variance(result.residuals)), "half_life": half_life}


def estimate_ou(values: Sequence[Number], dt: Number = 1.0) -> OUParameters:
    """Estimate an Ornstein-Uhlenbeck process from equally spaced observations.

    The discrete AR(1) coefficient maps to kappa=-log(phi)/dt. The method
    assumes a positive, mean-reverting AR coefficient; it does not correct for
    irregular sampling or measurement noise.
    """
    data = vector(values, min_length=4)
    step = finite(dt)
    if step <= 0:
        raise FinanceInputError("dt must be positive")
    fit = ols(data[1:], (data[:-1],), intercept=True)
    intercept, phi = fit.coefficients
    if not 0 < phi < 1:
        raise FinanceInputError("OU mapping requires 0 < AR(1) phi < 1")
    kappa = -math.log(phi)/step
    theta = intercept/(1-phi)
    residual_var = variance(fit.residuals)
    # Exact-discretization relation: residual variance = sigma^2*(1-phi^2)/(2*kappa).
    sigma = math.sqrt(residual_var*2*kappa/(1-phi*phi))
    return OUParameters(theta, kappa, sigma, math.log(2)/kappa, fit.residuals)


def ou_transition(current: Number, theta: Number, kappa: Number,
                  sigma: Number, dt: Number = 1.0) -> tuple[float, float]:
    """Conditional mean and standard deviation after dt for an OU process."""
    x, level, speed, vol, step = map(finite, (current, theta, kappa, sigma, dt))
    if speed <= 0 or vol < 0 or step < 0:
        raise FinanceInputError("kappa must be positive, sigma/dt non-negative")
    decay = math.exp(-speed*step)
    conditional_mean = level+(x-level)*decay
    conditional_std = vol*math.sqrt((1-decay*decay)/(2*speed))
    return conditional_mean, conditional_std


def ou_expected_value(current: Number, theta: Number, kappa: Number,
                      dt: Number = 1.0) -> float:
    """Conditional expected state at horizon dt."""
    x, level, speed, step = map(finite, (current, theta, kappa, dt))
    if speed <= 0 or step < 0:
        raise FinanceInputError("kappa must be positive and dt non-negative")
    return level+(x-level)*math.exp(-speed*step)


def half_life_from_ar1(phi: Number, dt: Number = 1.0) -> float:
    """Half-life of a positive stationary AR(1) coefficient."""
    coefficient, step = finite(phi), finite(dt)
    if not 0 < coefficient < 1 or step <= 0:
        raise FinanceInputError("require 0 < phi < 1 and dt > 0")
    return math.log(0.5)/math.log(coefficient)*step


def engle_granger(y: Sequence[Number], x: Sequence[Number]) -> dict[str, float]:
    """Two-step cointegration diagnostic with residual AR(1) persistence.

    This reports the residual unit-root t-statistic from a simple auxiliary
    regression but deliberately does not label it with a generic ADF p-value:
    Engle-Granger critical values depend on deterministic terms and sample size.
    """
    left, right = vector(y, min_length=8), vector(x, min_length=8)
    if len(left) != len(right):
        raise FinanceInputError("cointegration series must align")
    first = ols(left, (right,), intercept=True)
    residuals = first.residuals
    diff = [residuals[i]-residuals[i-1] for i in range(1, len(residuals))]
    lagged = residuals[:-1]
    auxiliary = ols(diff, (lagged,), intercept=True)
    t_stat = auxiliary.t_statistics[1]
    return {"hedge_ratio": first.coefficients[1], "intercept": first.coefficients[0],
            "level_r_squared": first.r_squared,
            "residual_ar1": ar1_parameters(residuals)["phi"],
            "residual_unit_root_t_stat": t_stat,
            "warning": 1.0}


def adf_regression(values: Sequence[Number], lags: int = 1,
                   *, include_trend: bool = False) -> dict[str, float]:
    """Augmented Dickey-Fuller regression with intercept and optional trend.

    Returns the lagged-level t statistic and regression diagnostics. The caller
    must apply critical values appropriate to deterministic terms and sample size.
    """
    data = vector(values, min_length=8)
    if not isinstance(lags, int) or lags < 0 or len(data) <= lags+4:
        raise FinanceInputError("lags must be non-negative and leave enough observations")
    differences = [data[i]-data[i-1] for i in range(1, len(data))]
    start = lags
    y = differences[start:]
    lag_level = data[start:-1]
    predictors: list[tuple[float, ...]] = [tuple(lag_level)]
    names = ["lagged_level"]
    for lag in range(1, lags+1):
        predictors.append(tuple(differences[start-lag:len(differences)-lag]))
        names.append(f"delta_lag_{lag}")
    if include_trend:
        predictors.append(tuple(float(i) for i in range(len(y))))
        names.append("trend")
    fit = ols(y, predictors, intercept=True)
    return {"test_statistic": fit.t_statistics[1], "lagged_level_coefficient": fit.coefficients[1],
            "r_squared": fit.r_squared, "observations": float(fit.observations),
            "lags": float(lags), "critical_values_required": 1.0}


def kalman_hedge_ratio(y: Sequence[Number], x: Sequence[Number],
                       *, observation_variance: Number = 1e-4,
                       state_variance: Number = 1e-5,
                       initial_hedge: Number = 1.0) -> tuple[dict[str, float], ...]:
    """Recursive two-state Kalman filter for intercept and hedge ratio.

    State vector is [intercept, beta], random-walk transition is identity,
    observation equation is y_t = intercept_t + beta_t * x_t + noise.
    """
    ys, xs = vector(y, min_length=2), vector(x, min_length=2)
    if len(ys) != len(xs):
        raise FinanceInputError("y and x must align")
    r, q = finite(observation_variance), finite(state_variance)
    if r <= 0 or q < 0:
        raise FinanceInputError("observation variance must be positive; state variance non-negative")
    state = [0.0, finite(initial_hedge)]
    covariance = [[1.0, 0.0], [0.0, 1.0]]
    history = []
    for observed, regressor in zip(ys, xs):
        # Predict state covariance: P <- P + QI.
        p = [[covariance[i][j]+(q if i == j else 0.0) for j in range(2)] for i in range(2)]
        h = [1.0, regressor]
        ph = [p[i][0]*h[0]+p[i][1]*h[1] for i in range(2)]
        innovation_var = r+h[0]*ph[0]+h[1]*ph[1]
        innovation = observed-(h[0]*state[0]+h[1]*state[1])
        gain = [v/innovation_var for v in ph]
        state = [state[i]+gain[i]*innovation for i in range(2)]
        covariance = [[p[i][j]-gain[i]*ph[j] for j in range(2)] for i in range(2)]
        history.append({"intercept": state[0], "hedge_ratio": state[1],
                        "innovation": innovation, "innovation_variance": innovation_var,
                        "normalized_innovation": innovation/math.sqrt(innovation_var)})
    return tuple(history)


def hurst_exponent(values: Sequence[Number], *, min_lag: int = 2,
                   max_lag: int | None = None) -> float:
    """Estimate Hurst exponent from log variance of lagged differences.

    This simple variogram estimator is a descriptive statistic, not a robust
    test for long memory. A value near 0.5 is consistent with a random walk.
    """
    data = vector(values, min_length=8)
    end = max_lag or max(min_lag+1, len(data)//4)
    if not 2 <= min_lag < end < len(data):
        raise FinanceInputError("lag range must satisfy 2 <= min_lag < max_lag < n")
    x, y = [], []
    for lag in range(min_lag, end+1):
        differences = [data[i]-data[i-lag] for i in range(lag, len(data))]
        v = variance(differences, sample=False)
        if v > 0:
            x.append(math.log(lag))
            y.append(math.log(v))
    slope = ols(y, (x,), intercept=True).coefficients[1]
    return slope/2


def rolling_hedge_ratio(y: Sequence[Number], x: Sequence[Number],
                        window: int) -> tuple[float, ...]:
    """Trailing OLS hedge ratios for a pair of price series."""
    ys, xs = vector(y), vector(x)
    if len(ys) != len(xs) or not 3 <= window <= len(ys):
        raise FinanceInputError("series must align and window must be [3, n]")
    values = []
    for end in range(window, len(ys)+1):
        fit = ols(ys[end-window:end], (xs[end-window:end],), intercept=True)
        values.append(fit.coefficients[1])
    return tuple(values)


def spread_zscore(y: Sequence[Number], x: Sequence[Number],
                  hedge_ratio: Number, window: int) -> tuple[float, ...]:
    """Trailing z-score of a static-hedge pair spread."""
    ys, xs = vector(y), vector(x)
    beta = finite(hedge_ratio)
    if len(ys) != len(xs) or not 3 <= window <= len(ys):
        raise FinanceInputError("series must align and window must be [3, n]")
    spread = [a-beta*b for a, b in zip(ys, xs)]
    scores = []
    for end in range(window, len(spread)+1):
        sample = spread[end-window:end]
        sd = math.sqrt(variance(sample))
        scores.append((sample[-1]-mean(sample))/sd if sd else math.nan)
    return tuple(scores)


def rolling_correlation(x: Sequence[Number], y: Sequence[Number],
                         window: int) -> tuple[float, ...]:
    """Trailing Pearson correlation for aligned series."""
    from .core import correlation
    left, right = vector(x), vector(y)
    if len(left) != len(right) or not 2 <= window <= len(left):
        raise FinanceInputError("series must align and window must be [2, n]")
    return tuple(correlation(left[i-window:i], right[i-window:i])
                 for i in range(window, len(left)+1))
