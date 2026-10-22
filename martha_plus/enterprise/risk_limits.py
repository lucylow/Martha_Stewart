"""Declarative portfolio limit evaluation with severity and breach history."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Callable, Iterable, Mapping, Sequence


class LimitMetric(str, Enum):
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    POSITION_WEIGHT = "position_weight"
    SECTOR_WEIGHT = "sector_weight"
    SYMBOL_WEIGHT = "symbol_weight"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    OPEN_ORDERS = "open_orders"
    DATA_AGE_SECONDS = "data_age_seconds"


class LimitComparison(str, Enum):
    MAX = "max"
    MIN = "min"
    ABS_MAX = "abs_max"


class BreachSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"
    HALT = "halt"


@dataclass(frozen=True, slots=True)
class RiskLimit:
    limit_id: str
    metric: LimitMetric
    comparison: LimitComparison
    threshold: float
    severity: BreachSeverity = BreachSeverity.BLOCK
    scope: str = "portfolio"
    enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.limit_id.strip() or not isfinite(self.threshold):
            raise ValueError("limit id and finite threshold are required")
        if self.comparison is LimitComparison.ABS_MAX and self.threshold < 0:
            raise ValueError("absolute threshold cannot be negative")


@dataclass(frozen=True, slots=True)
class LimitBreach:
    breach_id: str
    limit_id: str
    metric: LimitMetric
    observed: float
    threshold: float
    severity: BreachSeverity
    scope: str
    detected_at: str
    description: str
    acknowledged_by: str | None = None
    acknowledgement_note: str | None = None


@dataclass(frozen=True, slots=True)
class LimitEvaluation:
    checked_at: str
    observed: Mapping[str, float]
    breaches: tuple[LimitBreach, ...]
    permitted: bool
    halted: bool


class RiskLimitEngine:
    def __init__(self, limits: Iterable[RiskLimit] = (), *,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.clock = clock
        self._limits: dict[str, RiskLimit] = {}
        self._breaches: dict[str, LimitBreach] = {}
        self._counter = 0
        self._lock = RLock()
        for limit in limits:
            self.upsert(limit)

    def upsert(self, limit: RiskLimit) -> None:
        with self._lock:
            self._limits[limit.limit_id] = limit

    def disable(self, limit_id: str) -> bool:
        with self._lock:
            prior = self._limits.get(limit_id)
            if prior is None:
                return False
            self._limits[limit_id] = RiskLimit(prior.limit_id, prior.metric,
                prior.comparison, prior.threshold, prior.severity, prior.scope, False, prior.description)
            return True

    def evaluate(self, observed: Mapping[str, float]) -> LimitEvaluation:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        clean = {}
        for key, value in observed.items():
            if not isfinite(value):
                raise ValueError(f"observed value must be finite: {key}")
            clean[key] = float(value)
        with self._lock:
            limits = tuple(self._limits.values())
        new_breaches = []
        for limit in limits:
            if not limit.enabled:
                continue
            key = limit.scope if limit.metric not in (LimitMetric.POSITION_WEIGHT, LimitMetric.SECTOR_WEIGHT,
                                                       LimitMetric.SYMBOL_WEIGHT) else limit.scope
            value = clean.get(limit.metric.value + ":" + key, clean.get(limit.metric.value))
            if value is None:
                continue
            if limit.comparison is LimitComparison.MAX:
                breached = value > limit.threshold
            elif limit.comparison is LimitComparison.MIN:
                breached = value < limit.threshold
            else:
                breached = abs(value) > limit.threshold
            if breached:
                self._counter += 1
                ident = f"risk-{self._counter:08d}"
                record = LimitBreach(ident, limit.limit_id, limit.metric, value,
                    limit.threshold, limit.severity, limit.scope, now.isoformat(), limit.description)
                new_breaches.append(record)
        with self._lock:
            for breach in new_breaches:
                self._breaches[breach.breach_id] = breach
        halted = any(row.severity is BreachSeverity.HALT for row in new_breaches)
        blocked = any(row.severity in (BreachSeverity.BLOCK, BreachSeverity.HALT) for row in new_breaches)
        return LimitEvaluation(now.isoformat(), clean, tuple(new_breaches), not blocked, halted)

    def acknowledge(self, breach_id: str, *, actor: str, note: str) -> LimitBreach:
        if not actor.strip() or not note.strip():
            raise ValueError("actor and note are required")
        with self._lock:
            prior = self._breaches[breach_id]
            updated = LimitBreach(prior.breach_id, prior.limit_id, prior.metric,
                prior.observed, prior.threshold, prior.severity, prior.scope, prior.detected_at,
                prior.description, actor, note)
            self._breaches[breach_id] = updated
            return updated

    def breaches(self, *, severity: BreachSeverity | None = None,
                 limit_id: str | None = None) -> tuple[LimitBreach, ...]:
        with self._lock:
            rows = tuple(self._breaches.values())
        return tuple(row for row in rows if (severity is None or row.severity is severity)
                     and (limit_id is None or row.limit_id == limit_id))

    def summary(self) -> dict[str, int]:
        with self._lock:
            rows = tuple(self._breaches.values())
            limits = tuple(self._limits.values())
        return {"limits": len(limits), "enabled_limits": sum(l.enabled for l in limits),
                "breaches": len(rows), "unacknowledged": sum(r.acknowledged_by is None for r in rows),
                **{severity.value: sum(r.severity is severity for r in rows) for severity in BreachSeverity}}


def standard_risk_limits(*, max_gross_exposure: float = 2.0,
                         max_drawdown: float = 0.20,
                         max_daily_loss: float = 0.05,
                         max_data_age_seconds: float = 30.0) -> tuple[RiskLimit, ...]:
    """Return conservative example limits; tune them to strategy and account profile."""
    values = (max_gross_exposure, max_drawdown, max_daily_loss, max_data_age_seconds)
    if any(not isfinite(v) or v < 0 for v in values):
        raise ValueError("risk thresholds must be finite and non-negative")
    return (
        RiskLimit("gross-exposure", LimitMetric.GROSS_EXPOSURE, LimitComparison.MAX,
                  max_gross_exposure, BreachSeverity.BLOCK, description="gross exposure ceiling"),
        RiskLimit("drawdown-halt", LimitMetric.DRAWDOWN, LimitComparison.MAX,
                  max_drawdown, BreachSeverity.HALT, description="maximum drawdown trigger"),
        RiskLimit("daily-loss", LimitMetric.DAILY_LOSS, LimitComparison.ABS_MAX,
                  max_daily_loss, BreachSeverity.BLOCK, description="daily loss threshold"),
        RiskLimit("data-age", LimitMetric.DATA_AGE_SECONDS, LimitComparison.MAX,
                  max_data_age_seconds, BreachSeverity.BLOCK, description="stale-data ceiling"),
    )
