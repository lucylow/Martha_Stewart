"""Causal feature engineering for OHLCV histories.

Every generated feature at index i uses bars with index <= i. Labels and future
returns live in a separate helper to make look-ahead boundaries explicit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite, log, sqrt
from statistics import mean, median
from typing import Iterable, Mapping, Sequence
from ..types import Bar


@dataclass(frozen=True, slots=True)
class FeatureRow:
    symbol: str
    timestamp: object
    index: int
    values: Mapping[str, float | int | bool | str | None] = field(default_factory=dict)

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def as_dict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "timestamp": self.timestamp, "index": self.index, **self.values}


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    return_windows: tuple[int, ...] = (1, 5, 10, 20)
    moving_average_windows: tuple[int, ...] = (5, 10, 20, 50)
    volatility_windows: tuple[int, ...] = (10, 20, 60)
    volume_windows: tuple[int, ...] = (10, 20)
    include_range_features: bool = True
    include_calendar_features: bool = True

    def __post_init__(self) -> None:
        all_windows = self.return_windows + self.moving_average_windows + self.volatility_windows + self.volume_windows
        if any(w < 1 for w in all_windows):
            raise ValueError("feature windows must all be >= 1")
        for name in ("return_windows", "moving_average_windows", "volatility_windows", "volume_windows"):
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"{name} must not contain duplicates")


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _window(values: Sequence[float], end: int, length: int) -> Sequence[float]:
    return values[max(0, end - length + 1):end + 1]


class FeaturePipeline:
    """Create aligned, causal tabular features without pandas or NumPy."""
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(self, bars: Sequence[Bar]) -> list[FeatureRow]:
        if not bars:
            return []
        ordered = list(bars)
        symbol = ordered[0].symbol
        previous = None
        for bar in ordered:
            if bar.symbol != symbol:
                raise ValueError("one feature transform accepts a single symbol")
            if previous is not None and bar.timestamp <= previous:
                raise ValueError("bars must have strictly increasing timestamps")
            previous = bar.timestamp
        closes = [b.close for b in ordered]
        highs = [b.high for b in ordered]
        lows = [b.low for b in ordered]
        volumes = [b.volume for b in ordered]
        log_returns = [0.0] + [log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        rows: list[FeatureRow] = []
        for i, bar in enumerate(ordered):
            features: dict[str, float | int | bool | str | None] = {}
            for win in self.config.return_windows:
                if i >= win:
                    features[f"return_{win}"] = closes[i] / closes[i - win] - 1.0
                    features[f"log_return_{win}"] = log(closes[i] / closes[i - win])
                else:
                    features[f"return_{win}"] = None
                    features[f"log_return_{win}"] = None
            for win in self.config.moving_average_windows:
                values = _window(closes, i, win)
                average = mean(values)
                features[f"sma_{win}"] = average if len(values) == win else None
                features[f"price_to_sma_{win}"] = closes[i] / average - 1 if len(values) == win and average else None
            for win in self.config.volatility_windows:
                # Uses returns ending at i. A downstream next-bar target must be shifted separately.
                values = _window(log_returns[1:], i - 1, win) if i else []
                enough = i >= win
                features[f"volatility_{win}"] = _sample_std(values) * sqrt(252.0) if enough else None
                features[f"realized_variance_{win}"] = sum(v * v for v in values) if enough else None
            for win in self.config.volume_windows:
                values = _window(volumes, i, win)
                avg = mean(values)
                features[f"volume_mean_{win}"] = avg if len(values) == win else None
                features[f"volume_ratio_{win}"] = volumes[i] / avg if len(values) == win and avg > 0 else None
            if self.config.include_range_features:
                features["intrabar_range"] = (bar.high - bar.low) / bar.close
                features["close_location"] = ((bar.close - bar.low) / (bar.high - bar.low)
                                                if bar.high > bar.low else 0.5)
                features["gap_return"] = bar.open / ordered[i - 1].close - 1 if i else None
                features["body_fraction"] = (bar.close - bar.open) / max(bar.high - bar.low, 1e-12)
            if self.config.include_calendar_features:
                features["day_of_week"] = bar.timestamp.weekday()
                features["month"] = bar.timestamp.month
                features["is_month_start"] = bar.timestamp.day <= 3
                features["is_month_end"] = bar.timestamp.day >= 28
            rows.append(FeatureRow(symbol, bar.timestamp, i, features))
        return rows

    @staticmethod
    def forward_returns(bars: Sequence[Bar], horizon: int = 1) -> list[float | None]:
        """Return future close-to-close labels; last horizon rows are None."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        result: list[float | None] = []
        for i, bar in enumerate(bars):
            target = i + horizon
            result.append(bars[target].close / bar.close - 1.0 if target < len(bars) else None)
        return result

    @staticmethod
    def cross_sectional_rank(values_by_symbol: Mapping[str, float | None]) -> dict[str, float | None]:
        """Map available values to [0, 1] percentile ranks using average ties."""
        items = [(s, float(v)) for s, v in values_by_symbol.items()
                 if v is not None and isfinite(float(v))]
        items.sort(key=lambda item: (item[1], item[0]))
        ranks: dict[str, float | None] = {s: None for s in values_by_symbol}
        if not items:
            return ranks
        start = 0
        while start < len(items):
            end = start + 1
            while end < len(items) and items[end][1] == items[start][1]:
                end += 1
            average_rank = ((start + end - 1) / 2) / max(1, len(items) - 1)
            for j in range(start, end):
                ranks[items[j][0]] = average_rank
            start = end
        if len(items) == 1:
            ranks[items[0][0]] = 0.5
        return ranks

    @staticmethod
    def robust_zscore(values: Sequence[float], minimum_history: int = 10) -> list[float | None]:
        """Expanding median/MAD normalization, computed only from current/past data."""
        if minimum_history < 2:
            raise ValueError("minimum_history must be >= 2")
        output: list[float | None] = []
        for i, value in enumerate(values):
            history = [float(v) for v in values[:i + 1] if isfinite(float(v))]
            if len(history) < minimum_history:
                output.append(None)
                continue
            center = median(history)
            mad = median([abs(v - center) for v in history])
            output.append(0.0 if mad == 0 else 0.67448975 * (float(value) - center) / mad)
        return output


def align_feature_rows(rows_by_symbol: Mapping[str, Sequence[FeatureRow]]) -> list[dict[str, FeatureRow]]:
    """Align features by exact timestamp; never forward-fill missing observations."""
    timestamps = sorted({row.timestamp for rows in rows_by_symbol.values() for row in rows})
    by_symbol = {symbol: {row.timestamp: row for row in rows} for symbol, rows in rows_by_symbol.items()}
    return [{symbol: by_symbol[symbol][stamp] for symbol in sorted(by_symbol) if stamp in by_symbol[symbol]}
            for stamp in timestamps]
