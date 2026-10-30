"""Health checks and alert rules for offline and paper-simulation runs."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Iterable, Mapping

from ..data.contracts import FeedHealth


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    state: HealthState
    observed: str
    threshold: str
    message: str


@dataclass(frozen=True, slots=True)
class HealthReport:
    generated_at_ns: int
    overall: HealthState
    checks: tuple[HealthCheck, ...]

    @property
    def healthy(self) -> bool:
        return self.overall is HealthState.HEALTHY


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    max_feed_age_ns: int = 5_000_000_000
    max_gap_count: int = 0
    max_out_of_order: int = 0
    max_reject_fraction: Decimal = Decimal("0.01")
    max_mean_latency_ns: int = 5_000_000

    def __post_init__(self) -> None:
        if min(self.max_feed_age_ns, self.max_gap_count, self.max_out_of_order,
               self.max_mean_latency_ns) < 0:
            raise ValueError("health thresholds must be non-negative")
        if not 0 <= self.max_reject_fraction <= 1:
            raise ValueError("max_reject_fraction must lie in [0, 1]")


class FeedHealthEvaluator:
    """Convert feed counters into explicit states without treating unknown as healthy."""

    def __init__(self, thresholds: HealthThresholds | None = None) -> None:
        self.thresholds = thresholds or HealthThresholds()

    def evaluate(self, health: FeedHealth, now_ns: int) -> tuple[HealthCheck, ...]:
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        t = self.thresholds
        checks: list[HealthCheck] = []
        if health.last_event_ns is None:
            checks.append(HealthCheck("feed_freshness", HealthState.UNKNOWN, "unknown",
                                      str(t.max_feed_age_ns), "no feed event has been observed"))
        else:
            age = max(0, now_ns - health.last_event_ns)
            state = HealthState.HEALTHY if age <= t.max_feed_age_ns else HealthState.UNHEALTHY
            checks.append(HealthCheck("feed_freshness", state, str(age),
                                      str(t.max_feed_age_ns), "feed timestamp age"))
        gap_state = HealthState.HEALTHY if health.gaps <= t.max_gap_count else HealthState.UNHEALTHY
        checks.append(HealthCheck("sequence_gaps", gap_state, str(health.gaps),
                                  str(t.max_gap_count), "provider sequence gaps"))
        order_state = HealthState.HEALTHY if health.out_of_order <= t.max_out_of_order else HealthState.DEGRADED
        checks.append(HealthCheck("out_of_order", order_state, str(health.out_of_order),
                                  str(t.max_out_of_order), "out-of-order feed observations"))
        fraction = Decimal(health.rejected) / Decimal(health.received) if health.received else Decimal(0)
        reject_state = HealthState.HEALTHY if fraction <= t.max_reject_fraction else HealthState.DEGRADED
        checks.append(HealthCheck("reject_fraction", reject_state, str(fraction),
                                  str(t.max_reject_fraction), "rejected feed message fraction"))
        if health.mean_latency_ns is None:
            latency_state = HealthState.UNKNOWN
            latency_observed = "unknown"
        else:
            latency_state = (HealthState.HEALTHY if health.mean_latency_ns <= t.max_mean_latency_ns
                             else HealthState.DEGRADED)
            latency_observed = str(health.mean_latency_ns)
        checks.append(HealthCheck("feed_latency", latency_state, latency_observed,
                                  str(t.max_mean_latency_ns), "mean receive latency"))
        return tuple(checks)

    @staticmethod
    def overall(checks: Iterable[HealthCheck]) -> HealthState:
        states = [x.state for x in checks]
        if not states or all(x is HealthState.UNKNOWN for x in states):
            return HealthState.UNKNOWN
        if HealthState.UNHEALTHY in states:
            return HealthState.UNHEALTHY
        if HealthState.DEGRADED in states or HealthState.UNKNOWN in states:
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    def report(self, health: FeedHealth, now_ns: int) -> HealthReport:
        checks = self.evaluate(health, now_ns)
        return HealthReport(now_ns, self.overall(checks), checks)


@dataclass(frozen=True, slots=True)
class AlertRule:
    name: str
    metric: str
    operator: str
    threshold: Decimal
    severity: str = "warning"

    def __post_init__(self) -> None:
        if not self.name or not self.metric or self.operator not in {"gt", "gte", "lt", "lte", "eq"}:
            raise ValueError("invalid alert rule")


@dataclass(frozen=True, slots=True)
class Alert:
    rule: str
    metric: str
    observed: Decimal
    threshold: Decimal
    severity: str


class AlertEvaluator:
    """Evaluate stateless alert rules over a point-in-time metric mapping."""

    def __init__(self, rules: Iterable[AlertRule]) -> None:
        self.rules = tuple(rules)
        if len({x.name for x in self.rules}) != len(self.rules):
            raise ValueError("alert rule names must be unique")

    def evaluate(self, metrics: Mapping[str, Decimal | int | float]) -> tuple[Alert, ...]:
        alerts = []
        for rule in self.rules:
            if rule.metric not in metrics:
                continue
            value = Decimal(str(metrics[rule.metric]))
            failed = {
                "gt": value > rule.threshold,
                "gte": value >= rule.threshold,
                "lt": value < rule.threshold,
                "lte": value <= rule.threshold,
                "eq": value == rule.threshold,
            }[rule.operator]
            if failed:
                alerts.append(Alert(rule.name, rule.metric, value, rule.threshold, rule.severity))
        return tuple(alerts)


class SimulationCircuitBreaker:
    """Manual/automatic stop state for a local simulator (never a live kill switch)."""

    def __init__(self, *, max_consecutive_rejections: int = 10,
                 max_anomalies: int = 0) -> None:
        if max_consecutive_rejections < 0 or max_anomalies < 0:
            raise ValueError("circuit breaker thresholds must be non-negative")
        self.max_consecutive_rejections = max_consecutive_rejections
        self.max_anomalies = max_anomalies
        self.consecutive_rejections = 0
        self.anomalies = 0
        self.tripped = False
        self.reason = ""

    def record_order_result(self, accepted: bool) -> bool:
        if self.tripped:
            return False
        self.consecutive_rejections = 0 if accepted else self.consecutive_rejections + 1
        if self.consecutive_rejections > self.max_consecutive_rejections:
            self.trip("consecutive_rejections")
        return not self.tripped

    def record_anomaly(self, count: int = 1) -> bool:
        if count < 0:
            raise ValueError("anomaly count must be non-negative")
        self.anomalies += count
        if self.anomalies > self.max_anomalies:
            self.trip("anomaly_threshold")
        return not self.tripped

    def trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason or "manual_trip"

    def reset(self) -> None:
        self.consecutive_rejections = 0
        self.anomalies = 0
        self.tripped = False
        self.reason = ""
