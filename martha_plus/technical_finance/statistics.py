"""Distributional and robust statistics for financial return series."""
from __future__ import annotations

import math
from typing import Sequence

from .core import FinanceInputError, Number, Vector, finite, mean, quantile, variance, vector


def median(values: Vector) -> float:
    data = sorted(vector(values))
    n = len(data)
    mid = n // 2
    return data[mid] if n % 2 else (data[mid - 1] + data[mid]) / 2.0


def mad(values: Vector, *, scale_normal: bool = False) -> float:
    """Median absolute deviation; optional 1.4826 Gaussian consistency factor."""
    data = vector(values)
    center = median(data)
    result = median(tuple(abs(x-center) for x in data))
    return result * 1.482602218505602 if scale_normal else result


def moments(values: Vector) -> dict[str, float]:
    """Return mean, sample volatility, adjusted skewness and excess kurtosis."""
    data = vector(values, min_length=3)
    n = len(data)
    center = mean(data)
    m2 = math.fsum((x-center)**2 for x in data) / n
    if m2 <= 0:
        raise FinanceInputError("moments require non-constant data")
    m3 = math.fsum((x-center)**3 for x in data) / n
    m4 = math.fsum((x-center)**4 for x in data) / n
    raw_skew = m3 / m2**1.5
    raw_excess = m4 / m2**2 - 3.0
    skew = math.sqrt(n*(n-1))/(n-2)*raw_skew
    if n > 3:
        kurtosis = ((n-1)/((n-2)*(n-3))) * ((n+1)*raw_excess + 6.0)
    else:
        kurtosis = raw_excess
    return {"mean": center, "volatility": math.sqrt(variance(data)), "skewness": skew, "excess_kurtosis": kurtosis}


def winsorize(values: Vector, lower_q: Number = 0.01, upper_q: Number = 0.99) -> tuple[float, ...]:
    """Cap observations at empirical quantiles without changing sample size."""
    data = vector(values)
    lower, upper = quantile(data, lower_q), quantile(data, upper_q)
    if lower > upper:
        raise FinanceInputError("lower quantile must not exceed upper quantile")
    return tuple(min(upper, max(lower, x)) for x in data)


def trimmed_mean(values: Vector, proportion: Number = 0.1) -> float:
    """Mean after trimming an equal fraction from both tails."""
    data = sorted(vector(values))
    p = finite(proportion, "proportion")
    if not 0 <= p < 0.5:
        raise FinanceInputError("proportion must be in [0, 0.5)")
    cut = int(len(data) * p)
    retained = data[cut:len(data)-cut] if cut else data
    return math.fsum(retained) / len(retained)


def downside_deviation(values: Vector, target: Number = 0.0, *, annualization: Number = 1.0) -> float:
    """Root-mean-square shortfall below a target, scaled by sqrt(periods)."""
    data = vector(values)
    threshold = finite(target, "target")
    scale = finite(annualization, "annualization")
    if scale <= 0:
        raise FinanceInputError("annualization must be positive")
    shortfalls = [min(0.0, x-threshold)**2 for x in data]
    return math.sqrt(math.fsum(shortfalls)/len(data) * scale)


def jarque_bera(values: Vector) -> dict[str, float]:
    """Jarque-Bera normality statistic and asymptotic chi-square(2) p-value."""
    stats = moments(values)
    n = len(vector(values))
    statistic = n/6.0 * (stats["skewness"]**2 + stats["excess_kurtosis"]**2/4.0)
    # For chi-square with two degrees of freedom, survival is exp(-x/2).
    return {"statistic": statistic, "p_value": math.exp(-0.5*statistic)}


def empirical_cdf(values: Vector, points: Vector) -> tuple[float, ...]:
    """Evaluate the right-continuous empirical distribution function."""
    data = sorted(vector(values))
    queries = vector(points, "points", min_length=0)
    return tuple(sum(1 for x in data if x <= q)/len(data) for q in queries)


def rank(values: Vector, *, ascending: bool = True, pct: bool = False) -> tuple[float, ...]:
    """Average-tie ranks in original observation order."""
    data = vector(values)
    order = sorted(range(len(data)), key=lambda i: data[i], reverse=not ascending)
    ranks = [0.0] * len(data)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and data[order[end]] == data[order[cursor]]:
            end += 1
        avg = (cursor + 1 + end) / 2.0
        for j in range(cursor, end):
            ranks[order[j]] = avg
        cursor = end
    if pct:
        denom = max(1, len(data)-1)
        ranks = [(r-1)/denom for r in ranks]
    return tuple(ranks)


def zscore(values: Vector, *, robust: bool = False) -> tuple[float, ...]:
    """Standardize observations using mean/std or median/MAD."""
    data = vector(values)
    if robust:
        center, scale = median(data), mad(data, scale_normal=True)
    else:
        center, scale = mean(data), math.sqrt(variance(data, sample=False))
    if scale <= 0:
        raise FinanceInputError("cannot standardize a constant vector")
    return tuple((x-center)/scale for x in data)


def autocorrelation(values: Vector, lag: int = 1) -> float:
    """Biased autocorrelation estimator with a demeaned full-series denominator."""
    data = vector(values, min_length=3)
    if not isinstance(lag, int) or lag < 0 or lag >= len(data):
        raise FinanceInputError("lag must be an integer in [0, n)")
    if lag == 0:
        return 1.0
    center = mean(data)
    denom = math.fsum((x-center)**2 for x in data)
    if denom == 0:
        raise FinanceInputError("autocorrelation is undefined for constant values")
    numer = math.fsum((data[i]-center)*(data[i-lag]-center) for i in range(lag, len(data)))
    return numer / denom


def autocorrelation_lags(values: Vector, max_lag: int = 10) -> tuple[float, ...]:
    """Return ACF values from lag zero through max_lag."""
    data = vector(values, min_length=3)
    if max_lag < 0 or max_lag >= len(data):
        raise FinanceInputError("max_lag must be in [0, n)")
    return tuple(autocorrelation(data, lag) for lag in range(max_lag+1))


def ljung_box(values: Vector, lags: int = 10) -> dict[str, float]:
    """Ljung-Box Q statistic with approximate chi-square survival probability."""
    data = vector(values, min_length=4)
    if not 1 <= lags < len(data):
        raise FinanceInputError("lags must be in [1, n)")
    acf = autocorrelation_lags(data, lags)[1:]
    n = len(data)
    q = n*(n+2)*math.fsum(r*r/(n-k) for k, r in enumerate(acf, start=1))
    # Wilson-Hilferty approximation for a chi-square tail.
    z = ((q/lags)**(1/3) - (1 - 2/(9*lags))) / math.sqrt(2/(9*lags))
    p = 0.5*math.erfc(z/math.sqrt(2))
    return {"statistic": q, "degrees_freedom": float(lags), "p_value_approx": p}


def tail_mean(values: Vector, q: Number = 0.05, *, lower: bool = True) -> float:
    """Average of the empirical worst q-tail observations."""
    data = sorted(vector(values))
    alpha = finite(q, "q")
    if not 0 < alpha <= 1:
        raise FinanceInputError("q must be in (0, 1]")
    count = max(1, math.ceil(len(data)*alpha))
    selected = data[:count] if lower else data[-count:]
    return math.fsum(selected)/len(selected)


def rolling_mean(values: Vector, window: int) -> tuple[float, ...]:
    """Trailing means; the first value is returned after a full window."""
    data = vector(values)
    if not isinstance(window, int) or window < 1 or window > len(data):
        raise FinanceInputError("window must be between 1 and series length")
    running = math.fsum(data[:window])
    result = [running/window]
    for i in range(window, len(data)):
        running += data[i] - data[i-window]
        result.append(running/window)
    return tuple(result)


def rolling_volatility(values: Vector, window: int, *, annualization: Number = 1.0) -> tuple[float, ...]:
    """Trailing sample standard deviation, annualized by sqrt(periods)."""
    data = vector(values)
    if not 2 <= window <= len(data):
        raise FinanceInputError("window must be in [2, n]")
    scale = finite(annualization, "annualization")
    if scale <= 0:
        raise FinanceInputError("annualization must be positive")
    return tuple(math.sqrt(variance(data[i-window:i]))*math.sqrt(scale) for i in range(window, len(data)+1))


def rolling_quantile(values: Vector, window: int, q: Number) -> tuple[float, ...]:
    """Trailing empirical quantile computed independently at each endpoint."""
    data = vector(values)
    if not 1 <= window <= len(data):
        raise FinanceInputError("window must be in [1, n]")
    return tuple(quantile(data[i-window:i], q) for i in range(window, len(data)+1))


def normality_report(values: Vector) -> dict[str, float]:
    """Compact distribution report useful before selecting a VaR model."""
    data = vector(values, min_length=4)
    result = moments(data)
    result.update(jarque_bera(data))
    result["median"] = median(data)
    result["mad"] = mad(data)
    result["minimum"] = min(data)
    result["maximum"] = max(data)
    result["q01"] = quantile(data, 0.01)
    result["q05"] = quantile(data, 0.05)
    result["q95"] = quantile(data, 0.95)
    result["q99"] = quantile(data, 0.99)
    return result
