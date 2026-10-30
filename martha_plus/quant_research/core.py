"""Shared types, input contracts and descriptive statistics."""
from __future__ import annotations
from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from statistics import median
from typing import Iterable, Sequence


def finite_series(values: Iterable[float], *, name: str = "values", min_length: int = 1) -> list[float]:
    out = [float(v) for v in values]
    if len(out) < min_length:
        raise ValueError(f"{name} requires at least {min_length} observations")
    if not all(isfinite(v) for v in out):
        raise ValueError(f"{name} contains a non-finite value")
    return out


def mean(values: Sequence[float]) -> float:
    x = finite_series(values)
    return sum(x) / len(x)


def variance(values: Sequence[float], *, ddof: int = 1) -> float:
    x = finite_series(values, min_length=ddof + 1)
    m = sum(x) / len(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - ddof)


def stddev(values: Sequence[float], *, ddof: int = 1) -> float:
    return sqrt(max(0.0, variance(values, ddof=ddof)))


def covariance(x: Sequence[float], y: Sequence[float], *, ddof: int = 1) -> float:
    a, b = finite_series(x), finite_series(y)
    if len(a) != len(b) or len(a) <= ddof:
        raise ValueError("series must have equal lengths greater than ddof")
    ma, mb = mean(a), mean(b)
    return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (len(a) - ddof)


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    den = stddev(x) * stddev(y)
    return 0.0 if den == 0 else covariance(x, y) / den


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def normal_ppf(p: float) -> float:
    """Acklam rational approximation to the standard-normal quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1.0 - 0.02425
    if p < lo:
        q = sqrt(-2.0 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > hi:
        q = sqrt(-2.0 * log(1.0-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def quantile(values: Sequence[float], p: float) -> float:
    x = sorted(finite_series(values))
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    pos = (len(x) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def rank(values: Sequence[float]) -> list[float]:
    x = finite_series(values)
    order = sorted(range(len(x)), key=x.__getitem__)
    out = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


@dataclass(frozen=True)
class TimeSeries:
    """Immutable equally spaced series with optional labels.

    Values are not assumed to be daily, although return annualization functions
    require callers to state the number of observations per year explicitly.
    """
    values: tuple[float, ...]
    index: tuple[str, ...] = ()
    name: str = "series"

    def __init__(self, values: Iterable[float], index: Iterable[str] = (), name: str = "series"):
        vals = tuple(finite_series(values, name=name))
        idx = tuple(str(v) for v in index)
        if idx and len(idx) != len(vals):
            raise ValueError("index length must match values length")
        if idx and len(set(idx)) != len(idx):
            raise ValueError("index labels must be unique")
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "index", idx)
        object.__setattr__(self, "name", str(name))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def lag(self, periods: int = 1) -> "TimeSeries":
        if periods < 0 or periods >= len(self):
            raise ValueError("lag periods must be between zero and length - 1")
        vals = (self.values[0],) * periods + self.values[:len(self)-periods] if periods else self.values
        return TimeSeries(vals, self.index, f"{self.name}_lag{periods}")

    def slice(self, start: int, stop: int) -> "TimeSeries":
        return TimeSeries(self.values[start:stop], self.index[start:stop] if self.index else (), self.name)
