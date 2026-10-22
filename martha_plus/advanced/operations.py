"""Operational readiness checks, dependency aggregation, and SLO burn analysis."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping, Sequence


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: HealthStatus
    duration_seconds: float
    checked_at: str
    detail: str = ""
    critical: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    status: HealthStatus
    checked_at: str
    checks: tuple[HealthCheck, ...]
    failed: int
    degraded: int
    critical_failures: int

    @property
    def ready(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "ready": self.ready, "checked_at": self.checked_at,
                "failed": self.failed, "degraded": self.degraded,
                "critical_failures": self.critical_failures,
                "checks": [{"name": c.name, "status": c.status.value,
                            "duration_seconds": c.duration_seconds, "checked_at": c.checked_at,
                            "detail": c.detail, "critical": c.critical, "metadata": dict(c.metadata)}
                           for c in self.checks]}


class HealthRegistry:
    """Run named health checks with exception isolation and strict readiness semantics."""
    def __init__(self, *, clock: Callable[[], float] = monotonic,
                 wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._checks: dict[str, tuple[Callable[[], Any], bool]] = {}
        self._lock = RLock()
        self._clock = clock
        self._wall_clock = wall_clock
        self._history: list[HealthSnapshot] = []

    def register(self, name: str, probe: Callable[[], Any], *, critical: bool = True,
                 replace: bool = False) -> None:
        key = name.strip()
        if not key or not callable(probe):
            raise ValueError("check name and callable probe are required")
        with self._lock:
            if key in self._checks and not replace:
                raise ValueError(f"health check already registered: {key}")
            self._checks[key] = (probe, critical)

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._checks.pop(name, None) is not None

    def run(self) -> HealthSnapshot:
        with self._lock:
            checks = list(self._checks.items())
        results: list[HealthCheck] = []
        for name, (probe, critical) in checks:
            started = self._clock()
            when = self._wall_clock().astimezone(timezone.utc).isoformat()
            try:
                raw = probe()
                if isinstance(raw, HealthCheck):
                    status, detail, metadata = raw.status, raw.detail, dict(raw.metadata)
                elif isinstance(raw, Mapping):
                    status = HealthStatus(str(raw.get("status", "healthy")).lower())
                    detail = str(raw.get("detail", ""))
                    metadata = {str(k): v for k, v in raw.items() if k not in {"status", "detail"}}
                elif raw is False:
                    status, detail, metadata = HealthStatus.UNHEALTHY, "probe returned false", {}
                else:
                    status, detail, metadata = HealthStatus.HEALTHY, "", {}
            except Exception as exc:
                status, detail, metadata = HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}", {}
            duration = max(0.0, self._clock() - started)
            results.append(HealthCheck(name, status, duration, when, detail, critical, metadata))
        failed = sum(c.status is HealthStatus.UNHEALTHY for c in results)
        degraded = sum(c.status is HealthStatus.DEGRADED for c in results)
        critical_failures = sum(c.status is HealthStatus.UNHEALTHY and c.critical for c in results)
        if critical_failures:
            status = HealthStatus.UNHEALTHY
        elif failed or degraded or any(c.status is HealthStatus.UNKNOWN for c in results):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        snapshot = HealthSnapshot(status, self._wall_clock().astimezone(timezone.utc).isoformat(),
                                  tuple(results), failed, degraded, critical_failures)
        with self._lock:
            self._history.append(snapshot)
            self._history = self._history[-1000:]
        return snapshot

    @property
    def history(self) -> tuple[HealthSnapshot, ...]:
        with self._lock:
            return tuple(self._history)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {name: {"critical": critical, "probe": getattr(probe, "__name__", probe.__class__.__name__)}
                    for name, (probe, critical) in self._checks.items()}


def error_budget_burn(*, errors: int, requests: int, target_availability: float = 0.999,
                      window_seconds: float = 3600.0) -> dict[str, float | int | bool]:
    """Calculate observed error-budget burn rate for a rolling measurement window."""
    if errors < 0 or requests < 0 or errors > requests:
        raise ValueError("errors/requests must satisfy 0 <= errors <= requests")
    if not 0 < target_availability < 1 or window_seconds <= 0 or not isfinite(window_seconds):
        raise ValueError("invalid SLO target or window")
    error_rate = errors / requests if requests else 0.0
    allowed_error_rate = 1 - target_availability
    burn = error_rate / allowed_error_rate if allowed_error_rate > 0 else 0.0
    budget_consumed_fraction = min(1.0, error_rate / allowed_error_rate) if allowed_error_rate > 0 else 0.0
    return {"errors": errors, "requests": requests, "error_rate": error_rate,
            "target_availability": target_availability, "allowed_error_rate": allowed_error_rate,
            "burn_rate": burn, "window_seconds": window_seconds,
            "budget_exhausted": burn >= 1.0 - 1e-12, "budget_consumed_fraction": budget_consumed_fraction}


def latency_percentiles(latencies_seconds: Sequence[float]) -> dict[str, float | int]:
    values = sorted(float(x) for x in latencies_seconds)
    if any(not isfinite(x) or x < 0 for x in values):
        raise ValueError("latencies must be finite and non-negative")
    def q(frac: float) -> float:
        if not values:
            return 0.0
        pos = (len(values) - 1) * frac
        lo = int(pos); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"count": len(values), "p50": q(0.50), "p90": q(0.90), "p95": q(0.95),
            "p99": q(0.99), "max": values[-1] if values else 0.0}


def dependency_readiness(dependencies: Mapping[str, bool], critical: Sequence[str] = ()) -> dict[str, Any]:
    """Aggregate named dependency states without leaking endpoint credentials."""
    critical_set = set(critical)
    unknown_critical = critical_set - set(dependencies)
    failed = sorted(name for name, ok in dependencies.items() if not ok)
    critical_failed = sorted(name for name in failed if name in critical_set) + sorted(unknown_critical)
    return {"ready": not critical_failed, "status": "ready" if not critical_failed else "not_ready",
            "dependency_count": len(dependencies), "failed": failed,
            "critical_failed": critical_failed, "dependencies": dict(sorted(dependencies.items()))}
