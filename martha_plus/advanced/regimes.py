"""Interpretable market-regime detection without hidden state or look-ahead."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from statistics import mean
from typing import Sequence


class MarketRegime(str, Enum):
    UNKNOWN = "unknown"
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL_TRENDING = "high_vol_trending"
    LOW_VOL_RANGE = "low_vol_range"
    HIGH_VOL_RANGE = "high_vol_range"


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    index: int
    regime: MarketRegime
    volatility: float | None
    trend_strength: float | None
    direction: int
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    volatility_window: int = 20
    trend_window: int = 20
    low_vol_quantile: float = 0.5
    high_vol_multiplier: float = 1.5
    trend_threshold: float = 0.03
    hysteresis: float = 0.1

    def __post_init__(self) -> None:
        if self.volatility_window < 2 or self.trend_window < 2:
            raise ValueError("regime windows must be >= 2")
        if not 0 < self.low_vol_quantile < 1:
            raise ValueError("low_vol_quantile must be in (0, 1)")
        if self.high_vol_multiplier <= 1 or self.trend_threshold < 0:
            raise ValueError("invalid regime thresholds")
        if not 0 <= self.hysteresis < 1:
            raise ValueError("hysteresis must be in [0, 1)")


class RegimeDetector:
    """A rolling volatility/trend classifier with historical calibration only."""
    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def transform(self, closes: Sequence[float]) -> list[RegimeObservation]:
        values = [float(c) for c in closes]
        if any(not isfinite(c) or c <= 0 for c in values):
            raise ValueError("closes must be positive finite prices")
        result: list[RegimeObservation] = []
        realized: list[float] = []
        previous = MarketRegime.UNKNOWN
        for i, close in enumerate(values):
            vol = trend = None
            direction = 0
            if i >= self.config.volatility_window:
                returns = [values[j] / values[j - 1] - 1 for j in range(i - self.config.volatility_window + 1, i + 1)]
                avg = mean(returns)
                vol = sqrt(sum((r - avg) ** 2 for r in returns) / max(1, len(returns) - 1)) * sqrt(252)
                realized.append(vol)
            if i >= self.config.trend_window:
                start = values[i - self.config.trend_window]
                trend = close / start - 1
                direction = 1 if trend > 0 else (-1 if trend < 0 else 0)
            enough = vol is not None and trend is not None
            if not enough:
                regime, confidence, reason = MarketRegime.UNKNOWN, 0.0, "insufficient trailing history"
            else:
                calibration = realized[:-1] if len(realized) > 1 else realized
                baseline = sorted(calibration)[int((len(calibration) - 1) * self.config.low_vol_quantile)]
                high_vol = bool(vol > baseline * self.config.high_vol_multiplier)
                trending = abs(trend) >= self.config.trend_threshold
                if previous is not MarketRegime.UNKNOWN and not trending and abs(trend) > self.config.trend_threshold * (1 - self.config.hysteresis):
                    trending = previous in {MarketRegime.LOW_VOL_TRENDING, MarketRegime.HIGH_VOL_TRENDING}
                if high_vol:
                    regime = MarketRegime.HIGH_VOL_TRENDING if trending else MarketRegime.HIGH_VOL_RANGE
                else:
                    regime = MarketRegime.LOW_VOL_TRENDING if trending else MarketRegime.LOW_VOL_RANGE
                confidence = min(1.0, abs(trend) / max(self.config.trend_threshold, 1e-9)) if trending else min(1.0, (baseline * self.config.high_vol_multiplier - vol) / max(baseline, 1e-9))
                reason = f"trend={trend:.4f}; annualized_vol={vol:.4f}; baseline={baseline:.4f}"
            result.append(RegimeObservation(i, regime, vol, trend, direction, confidence, reason))
            if regime is not MarketRegime.UNKNOWN:
                previous = regime
        return result

    def current(self, closes: Sequence[float]) -> RegimeObservation:
        observations = self.transform(closes)
        if not observations:
            raise ValueError("closes must not be empty")
        return observations[-1]


def regime_exposure_multiplier(regime: MarketRegime, *, low_vol: float = 1.0,
                               high_vol: float = 0.5, unknown: float = 0.0) -> float:
    """Return an exposure scale; unknown history gets no added exposure by default."""
    if not all(isfinite(x) for x in (low_vol, high_vol, unknown)):
        raise ValueError("multipliers must be finite")
    if any(x < 0 or x > 1 for x in (low_vol, high_vol, unknown)):
        raise ValueError("multipliers must be in [0, 1]")
    if regime is MarketRegime.UNKNOWN:
        return unknown
    if regime in {MarketRegime.HIGH_VOL_RANGE, MarketRegime.HIGH_VOL_TRENDING}:
        return high_vol
    return low_vol
