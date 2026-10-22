"""Pure-python technical indicators with predictable behavior on short windows."""
from __future__ import annotations
from math import sqrt
from statistics import mean, median
from typing import Sequence

from .math_utils import stddev, robust_zscore


def returns(prices: Sequence[float], log: bool = False) -> list[float]:
    result: list[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous <= 0 or current <= 0:
            raise ValueError("prices must be positive")
        if log:
            from math import log
            result.append(log(current / previous))
        else:
            result.append(current / previous - 1.0)
    return result


def sma(values: Sequence[float], window: int) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(values)
    rolling_sum = 0.0
    for i, value in enumerate(values):
        rolling_sum += value
        if i >= window:
            rolling_sum -= values[i - window]
        if i >= window - 1:
            output[i] = rolling_sum / window
    return output


def ema(values: Sequence[float], window: int) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(values)
    if len(values) < window:
        return output
    current = mean(values[:window])
    output[window - 1] = current
    alpha = 2.0 / (window + 1.0)
    for i in range(window, len(values)):
        current = alpha * values[i] + (1 - alpha) * current
        output[i] = current
    return output


def rolling_std(values: Sequence[float], window: int, sample: bool = True) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(values)
    for i in range(window - 1, len(values)):
        sample_values = values[i - window + 1:i + 1]
        output[i] = stddev(sample_values, sample=sample)
    return output


def zscore_series(values: Sequence[float], window: int) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(values)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1:i + 1]
        deviation = stddev(sample)
        output[i] = (values[i] - mean(sample)) / deviation if deviation else 0.0
    return output


def robust_zscore_series(values: Sequence[float], window: int) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(values)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1:i + 1]
        output[i] = robust_zscore(values[i], sample)
    return output


def rsi(prices: Sequence[float], window: int = 14) -> list[float | None]:
    _validate_window(window)
    output: list[float | None] = [None] * len(prices)
    if len(prices) <= window:
        return output
    changes = [b - a for a, b in zip(prices, prices[1:])]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    output[window] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(window, len(changes)):
        avg_gain = ((window - 1) * avg_gain + gains[i]) / window
        avg_loss = ((window - 1) * avg_loss + losses[i]) / window
        output[i + 1] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return output


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
        window: int = 14) -> list[float | None]:
    _validate_window(window)
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("high, low, close arrays must match")
    output: list[float | None] = [None] * len(closes)
    if len(closes) < window:
        return output
    true_ranges: list[float] = []
    for i, (high, low, close) in enumerate(zip(highs, lows, closes)):
        previous_close = closes[i - 1] if i else close
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    current = sum(true_ranges[:window]) / window
    output[window - 1] = current
    for i in range(window, len(closes)):
        current = ((window - 1) * current + true_ranges[i]) / window
        output[i] = current
    return output


def bollinger_bands(values: Sequence[float], window: int = 20,
                    num_std: float = 2.0) -> tuple[list[float | None], list[float | None], list[float | None]]:
    if num_std <= 0:
        raise ValueError("num_std must be positive")
    middle = sma(values, window)
    deviations = rolling_std(values, window)
    lower: list[float | None] = []
    upper: list[float | None] = []
    for m, s in zip(middle, deviations):
        lower.append(None if m is None or s is None else m - num_std * s)
        upper.append(None if m is None or s is None else m + num_std * s)
    return lower, middle, upper


def ewma_volatility(returns_: Sequence[float], decay: float = 0.94) -> list[float | None]:
    if not 0 < decay < 1:
        raise ValueError("decay must be in (0, 1)")
    output: list[float | None] = []
    variance = 0.0
    initialized = False
    for value in returns_:
        if not initialized:
            variance = value * value
            initialized = True
        else:
            variance = decay * variance + (1 - decay) * value * value
        output.append(sqrt(max(0.0, variance)))
    return output


def rolling_beta(asset_returns: Sequence[float], benchmark_returns: Sequence[float],
                 window: int) -> list[float | None]:
    _validate_window(window)
    if len(asset_returns) != len(benchmark_returns):
        raise ValueError("asset and benchmark returns must have equal length")
    out: list[float | None] = [None] * len(asset_returns)
    for i in range(window - 1, len(asset_returns)):
        x = benchmark_returns[i - window + 1:i + 1]
        y = asset_returns[i - window + 1:i + 1]
        var = sum((v - mean(x)) ** 2 for v in x)
        out[i] = sum((a - mean(x)) * (b - mean(y)) for a, b in zip(x, y)) / var if var else 0.0
    return out


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
