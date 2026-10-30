"""Small adaptive-model controls not supplied by the existing model modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Iterable


def conformal_interval(
    prediction: float,
    calibration_residuals: Iterable[float],
    *,
    coverage: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided split-conformal interval around one point forecast.

    Uses absolute calibration residuals and the finite-sample corrected order
    statistic ``ceil((n + 1) * coverage)`` (one-indexed). If the requested
    order exceeds the calibration sample size, the valid interval is unbounded
    and ``(-inf, inf)`` is returned. Exchangeability of calibration and future
    errors is an assumption, not guaranteed by this function.
    """
    if isinstance(prediction, bool) or not isfinite(prediction):
        raise ValueError("prediction must be finite")
    if isinstance(coverage, bool) or not isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be finite and strictly between zero and one")
    residuals = []
    for residual in calibration_residuals:
        if isinstance(residual, bool) or not isfinite(residual):
            raise ValueError("calibration residuals must be finite numbers")
        residuals.append(abs(float(residual)))
    if not residuals:
        raise ValueError("at least one calibration residual is required")
    rank = ceil((len(residuals) + 1) * coverage)
    if rank > len(residuals):
        return float("-inf"), float("inf")
    radius = sorted(residuals)[rank - 1]
    center = float(prediction)
    return center - radius, center + radius


@dataclass(slots=True)
class PageHinkley:
    """Online two-sided CUSUM-style mean-shift alarm.

    ``delta`` is the tolerated per-observation drift and ``threshold`` controls
    sensitivity. A detected alarm clears accumulated evidence but retains the
    running mean. Calibrate both parameters to the sampling frequency and
    expected noise; an alarm is a diagnostic, not proof of structural change.
    """

    threshold: float = 5.0
    delta: float = 0.005
    forgetting: float = 1.0
    count: int = field(init=False, default=0)
    mean: float = field(init=False, default=0.0)
    _up: float = field(init=False, default=0.0, repr=False)
    _down: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if not isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if not isfinite(self.forgetting) or not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics and calibration mean."""
        self.count = 0
        self.mean = 0.0
        self._up = 0.0
        self._down = 0.0

    def update(self, value: float) -> bool:
        """Consume one finite observation and return whether drift was flagged."""
        if isinstance(value, bool) or not isfinite(value):
            raise ValueError("drift observations must be finite numbers")
        x = float(value)
        prior_mean = self.mean
        self.count += 1
        self.mean += (x - self.mean) / self.count
        deviation = x - prior_mean - self.delta
        self._up = max(0.0, self.forgetting * self._up + deviation)
        self._down = max(0.0, self.forgetting * self._down - deviation)
        alarm = max(self._up, self._down) > self.threshold
        if alarm:
            self._up = 0.0
            self._down = 0.0
        return alarm
