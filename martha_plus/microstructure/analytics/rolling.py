"""Bounded online statistics and sampling utilities.

All stateful accumulators have explicit reset and snapshot methods.  Inputs are
validated at the boundary so downstream strategy code can avoid silent NaNs.
The module uses only the Python standard library and has no global state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from statistics import median
from typing import Deque, Iterable, Sequence


Number = int | float | Decimal


def _finite(value: Number) -> float:
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError("value must be finite")
    return result


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    count: int
    mean: float | None
    variance: float | None
    minimum: float | None
    maximum: float | None


class RollingMoments:
    """Fixed-width rolling mean/variance with O(1) update cost.

    The implementation keeps the current sum and sum of squares.  It is
    numerically adequate for short market-data windows; for long-lived
    high-dynamic-range series prefer :class:`WelfordAccumulator`.
    """

    def __init__(self, window: int, *, ddof: int = 1) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if ddof not in (0, 1):
            raise ValueError("ddof must be zero or one")
        self.window = window
        self.ddof = ddof
        self._values: Deque[float] = deque()
        self._sum = 0.0
        self._sum_squares = 0.0

    def update(self, value: Number) -> WindowSnapshot:
        x = _finite(value)
        self._values.append(x)
        self._sum += x
        self._sum_squares += x * x
        if len(self._values) > self.window:
            old = self._values.popleft()
            self._sum -= old
            self._sum_squares -= old * old
        return self.snapshot()

    def extend(self, values: Iterable[Number]) -> WindowSnapshot:
        for value in values:
            self.update(value)
        return self.snapshot()

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float | None:
        return self._sum / self.count if self.count else None

    @property
    def variance(self) -> float | None:
        n = self.count
        if n <= self.ddof:
            return None
        numerator = self._sum_squares - (self._sum * self._sum / n)
        # Round-off can make a mathematically non-negative value slightly < 0.
        return max(0.0, numerator / (n - self.ddof))

    @property
    def standard_deviation(self) -> float | None:
        value = self.variance
        return sqrt(value) if value is not None else None

    @property
    def minimum(self) -> float | None:
        return min(self._values) if self._values else None

    @property
    def maximum(self) -> float | None:
        return max(self._values) if self._values else None

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def snapshot(self) -> WindowSnapshot:
        return WindowSnapshot(self.count, self.mean, self.variance, self.minimum, self.maximum)

    def reset(self) -> None:
        self._values.clear()
        self._sum = 0.0
        self._sum_squares = 0.0


class WelfordAccumulator:
    """Numerically stable online mean and sample variance accumulator."""

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._minimum: float | None = None
        self._maximum: float | None = None

    def update(self, value: Number) -> None:
        x = _finite(value)
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        self._minimum = x if self._minimum is None else min(self._minimum, x)
        self._maximum = x if self._maximum is None else max(self._maximum, x)

    def merge(self, other: "WelfordAccumulator") -> None:
        """Merge independent accumulators (e.g. partitions of one dataset)."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self._mean, self._m2 = other.n, other._mean, other._m2
            self._minimum, self._maximum = other._minimum, other._maximum
            return
        total = self.n + other.n
        delta = other._mean - self._mean
        self._m2 += other._m2 + delta * delta * self.n * other.n / total
        self._mean += delta * other.n / total
        self.n = total
        self._minimum = min(self._minimum, other._minimum)
        self._maximum = max(self._maximum, other._maximum)

    @property
    def mean(self) -> float | None:
        return self._mean if self.n else None

    @property
    def variance(self) -> float | None:
        return self._m2 / (self.n - 1) if self.n > 1 else None

    @property
    def population_variance(self) -> float | None:
        return self._m2 / self.n if self.n else None

    def snapshot(self) -> WindowSnapshot:
        return WindowSnapshot(self.n, self.mean, self.variance, self._minimum, self._maximum)


class ExponentialMovingAverage:
    """Recursive EMA with explicit warm-up semantics."""

    def __init__(self, span: int, *, min_periods: int = 1) -> None:
        if span <= 0 or min_periods <= 0 or min_periods > span:
            raise ValueError("span and min_periods must be positive with min_periods <= span")
        self.span = span
        self.min_periods = min_periods
        self.alpha = 2.0 / (span + 1.0)
        self.count = 0
        self._value: float | None = None

    def update(self, value: Number) -> float | None:
        x = _finite(value)
        self.count += 1
        self._value = x if self._value is None else self.alpha * x + (1 - self.alpha) * self._value
        return self.value

    @property
    def value(self) -> float | None:
        return self._value if self.count >= self.min_periods else None

    def reset(self) -> None:
        self.count = 0
        self._value = None


class ExponentiallyWeightedVariance:
    """EWMA variance of a scalar series, with a stable recursive update."""

    def __init__(self, decay: float = 0.94, *, mean_decay: float | None = None) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must lie in (0, 1)")
        if mean_decay is not None and not 0 < mean_decay < 1:
            raise ValueError("mean_decay must lie in (0, 1)")
        self.decay = decay
        self.mean_decay = mean_decay if mean_decay is not None else decay
        self.count = 0
        self.mean: float | None = None
        self.variance: float | None = None

    def update(self, value: Number) -> float | None:
        x = _finite(value)
        if self.mean is None:
            self.mean = x
            self.variance = 0.0
        else:
            residual = x - self.mean
            self.variance = self.decay * (self.variance or 0.0) + (1 - self.decay) * residual * residual
            self.mean = self.mean_decay * self.mean + (1 - self.mean_decay) * x
        self.count += 1
        return self.variance

    @property
    def volatility(self) -> float | None:
        return sqrt(self.variance) if self.variance is not None else None


class RollingQuantile:
    """Exact rolling quantile for modest windows; deterministic and dependency-free."""

    def __init__(self, window: int, quantile: float) -> None:
        if window <= 0 or not 0 <= quantile <= 1:
            raise ValueError("invalid window or quantile")
        self.window = window
        self.quantile = quantile
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, value: Number) -> float:
        self._values.append(_finite(value))
        ordered = sorted(self._values)
        if len(ordered) == 1:
            return ordered[0]
        position = self.quantile * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    @property
    def value(self) -> float | None:
        if not self._values:
            return None
        ordered = sorted(self._values)
        position = self.quantile * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight


class RollingMedianAbsoluteDeviation:
    """Robust rolling scale estimator, useful for spread and latency outliers."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, value: Number) -> float:
        self._values.append(_finite(value))
        center = median(self._values)
        return median(abs(x - center) for x in self._values)

    @property
    def value(self) -> float | None:
        if not self._values:
            return None
        center = median(self._values)
        return median(abs(x - center) for x in self._values)


class ReturnSeries:
    """Convert positive price observations into simple/log returns."""

    def __init__(self, *, maxlen: int = 10_000, log_returns: bool = False) -> None:
        if maxlen <= 1:
            raise ValueError("maxlen must exceed one")
        self.prices: Deque[float] = deque(maxlen=maxlen)
        self.returns: Deque[float] = deque(maxlen=maxlen - 1)
        self.log_returns = log_returns

    def update(self, price: Number) -> float | None:
        current = _finite(price)
        if current <= 0:
            raise ValueError("price must be positive")
        result = None
        if self.prices:
            previous = self.prices[-1]
            if self.log_returns:
                from math import log
                result = log(current / previous)
            else:
                result = current / previous - 1.0
            self.returns.append(result)
        self.prices.append(current)
        return result

    def annualized_volatility(self, periods_per_year: float) -> float | None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if len(self.returns) < 2:
            return None
        values = list(self.returns)
        center = sum(values) / len(values)
        variance = sum((x - center) ** 2 for x in values) / (len(values) - 1)
        return sqrt(variance * periods_per_year)

    def reset(self) -> None:
        self.prices.clear()
        self.returns.clear()


@dataclass(frozen=True, slots=True)
class Bar:
    index: int
    start_ns: int
    end_ns: int
    open: int
    high: int
    low: int
    close: int
    volume: int
    trades: int


class VolumeBarBuilder:
    """Aggregate trade prints into bars with a fixed integer volume target."""

    def __init__(self, target_volume: int) -> None:
        if target_volume <= 0:
            raise ValueError("target_volume must be positive")
        self.target_volume = target_volume
        self._index = 0
        self._reset_partial()

    def _reset_partial(self) -> None:
        self._start: int | None = None
        self._end: int | None = None
        self._open: int | None = None
        self._high: int | None = None
        self._low: int | None = None
        self._close: int | None = None
        self._volume = 0
        self._trades = 0

    def update(self, ts_ns: int, price_ticks: int, quantity: int) -> tuple[Bar, ...]:
        if ts_ns < 0 or price_ticks <= 0 or quantity <= 0:
            raise ValueError("invalid trade event")
        if self._end is not None and ts_ns < self._end:
            raise ValueError("trade timestamps must be non-decreasing")
        emitted: list[Bar] = []
        remaining = quantity
        while remaining:
            if self._start is None:
                self._start, self._open = ts_ns, price_ticks
                self._high = self._low = self._close = price_ticks
            take = min(remaining, self.target_volume - self._volume)
            self._volume += take
            self._trades += 1
            self._high = max(self._high or price_ticks, price_ticks)
            self._low = min(self._low or price_ticks, price_ticks)
            self._close = price_ticks
            self._end = ts_ns
            remaining -= take
            if self._volume == self.target_volume:
                emitted.append(Bar(self._index, self._start, ts_ns, self._open or price_ticks,
                                   self._high or price_ticks, self._low or price_ticks,
                                   self._close or price_ticks, self._volume, self._trades))
                self._index += 1
                self._reset_partial()
        return tuple(emitted)

    def flush(self) -> Bar | None:
        if self._volume == 0:
            return None
        bar = Bar(self._index, self._start or 0, self._end or 0, self._open or 0,
                  self._high or 0, self._low or 0, self._close or 0,
                  self._volume, self._trades)
        self._index += 1
        self._reset_partial()
        return bar
