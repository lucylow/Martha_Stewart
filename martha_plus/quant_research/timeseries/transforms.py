"""Time-series transforms used in feature engineering and alpha research."""
from __future__ import annotations
from math import exp, log
from typing import Callable, Sequence
from ..core import finite_series, mean, stddev, quantile


def simple_returns(prices: Sequence[float]) -> list[float]:
    p = finite_series(prices, name="prices", min_length=2)
    if any(v <= 0 for v in p):
        raise ValueError("prices must be strictly positive")
    return [p[i] / p[i-1] - 1.0 for i in range(1, len(p))]


def log_returns(prices: Sequence[float]) -> list[float]:
    p = finite_series(prices, name="prices", min_length=2)
    if any(v <= 0 for v in p):
        raise ValueError("prices must be strictly positive")
    return [log(p[i] / p[i-1]) for i in range(1, len(p))]


def difference(values: Sequence[float], order: int = 1) -> list[float]:
    x = finite_series(values)
    if order < 0 or order >= len(x):
        raise ValueError("order must be in [0, len(values)-1]")
    for _ in range(order):
        x = [x[i] - x[i-1] for i in range(1, len(x))]
    return x


def fractional_weights(order: float, size: int, threshold: float = 1e-5) -> list[float]:
    """Return truncated fractional-difference weights, newest observation first."""
    if not 0 <= order <= 2 or size < 1 or threshold <= 0:
        raise ValueError("order must be [0,2], size positive, threshold positive")
    weights = [1.0]
    for k in range(1, size):
        nxt = -weights[-1] * (order - k + 1.0) / k
        if abs(nxt) < threshold:
            break
        weights.append(nxt)
    return weights


def fractional_difference(values: Sequence[float], order: float, threshold: float = 1e-5) -> list[float]:
    """Fixed-width fractional differencing; output excludes incomplete windows."""
    x = finite_series(values)
    w = fractional_weights(order, len(x), threshold)
    width = len(w)
    return [sum(w[j] * x[i-j] for j in range(width)) for i in range(width-1, len(x))]


def rolling_apply(values: Sequence[float], window: int, fn: Callable[[Sequence[float]], float]) -> list[float | None]:
    x = finite_series(values)
    if window < 1 or window > len(x):
        raise ValueError("window must be between one and series length")
    return [None] * (window-1) + [float(fn(x[i-window+1:i+1])) for i in range(window-1, len(x))]


def rolling_mean(values: Sequence[float], window: int) -> list[float | None]:
    return rolling_apply(values, window, mean)


def rolling_std(values: Sequence[float], window: int, ddof: int = 1) -> list[float | None]:
    if window <= ddof:
        raise ValueError("window must exceed ddof")
    return rolling_apply(values, window, lambda z: stddev(z, ddof=ddof))


def ewm(values: Sequence[float], span: float, *, adjust: bool = False) -> list[float]:
    """Exponentially weighted mean. adjust=True uses normalized finite weights."""
    x = finite_series(values)
    if span <= 0:
        raise ValueError("span must be positive")
    alpha = 2.0 / (span + 1.0)
    out = []
    if adjust:
        numerator = denominator = 0.0
        for value in x:
            numerator = value + (1.0-alpha) * numerator
            denominator = 1.0 + (1.0-alpha) * denominator
            out.append(numerator / denominator)
    else:
        current = x[0]
        out.append(current)
        for value in x[1:]:
            current = alpha * value + (1.0-alpha) * current
            out.append(current)
    return out


def winsorize(values: Sequence[float], lower: float = 0.01, upper: float = 0.99) -> list[float]:
    x = finite_series(values)
    if not 0 <= lower < upper <= 1:
        raise ValueError("require 0 <= lower < upper <= 1")
    lo, hi = quantile(x, lower), quantile(x, upper)
    return [min(hi, max(lo, v)) for v in x]


def robust_zscore(values: Sequence[float], scale_factor: float = 1.4826) -> list[float]:
    """Median/MAD scale; returns zeros when the series is constant."""
    x = finite_series(values)
    med = quantile(x, 0.5)
    deviations = [abs(v-med) for v in x]
    mad = quantile(deviations, 0.5) * scale_factor
    return [0.0 if mad == 0 else (v-med)/mad for v in x]


def rolling_rank(values: Sequence[float], window: int) -> list[float | None]:
    from ..core import rank
    x = finite_series(values)
    if window < 2 or window > len(x):
        raise ValueError("window must be at least two and no larger than input")
    out: list[float | None] = [None] * (window-1)
    for i in range(window-1, len(x)):
        r = rank(x[i-window+1:i+1])
        out.append((r[-1]-1.0)/(window-1.0))
    return out


def zscore(values: Sequence[float], *, ddof: int = 1) -> list[float]:
    x = finite_series(values, min_length=ddof+1)
    m, s = mean(x), stddev(x, ddof=ddof)
    return [0.0 if s == 0 else (v-m)/s for v in x]


def lag(values: Sequence[float], periods: int = 1) -> list[float | None]:
    x = finite_series(values)
    if periods < 0 or periods >= len(x):
        raise ValueError("periods must be in [0, len(values)-1]")
    return [None] * periods + x[:len(x)-periods] if periods else list(x)


def cumulative_product(returns: Sequence[float], start: float = 1.0) -> list[float]:
    r = finite_series(returns)
    if any(v < -1 for v in r):
        raise ValueError("simple returns cannot be less than -100%")
    value, out = float(start), []
    for v in r:
        value *= 1.0 + v
        out.append(value)
    return out


def rolling_beta(y: Sequence[float], x: Sequence[float], window: int) -> list[float | None]:
    from ..regression import ols
    a, b = finite_series(y), finite_series(x)
    if len(a) != len(b) or window < 3 or window > len(a):
        raise ValueError("equal inputs and a valid window >= 3 are required")
    out: list[float | None] = [None] * (window-1)
    for i in range(window-1, len(a)):
        xx, yy = b[i-window+1:i+1], a[i-window+1:i+1]
        mx, my = mean(xx), mean(yy)
        den = sum((v-mx)**2 for v in xx)
        out.append(0.0 if den == 0 else sum((u-mx)*(v-my) for u,v in zip(xx,yy))/den)
    return out
