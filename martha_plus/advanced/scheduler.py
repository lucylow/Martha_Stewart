"""Manual-tick job scheduler with idempotency keys and bounded retry policy."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from time import monotonic, sleep
from typing import Any, Callable, Mapping
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class JobSpec:
    name: str
    callback: Callable[[], Any] = field(compare=False, repr=False)
    interval: timedelta = timedelta(minutes=1)
    enabled: bool = True
    run_immediately: bool = False
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    max_runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not callable(self.callback):
            raise ValueError("job name and callable callback are required")
        if self.interval.total_seconds() <= 0 or not 0 <= self.max_retries <= 10:
            raise ValueError("invalid job interval or max_retries")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")


@dataclass(frozen=True, slots=True)
class JobRun:
    run_id: str
    job_name: str
    status: JobStatus
    scheduled_at: str
    started_at: str | None
    finished_at: str | None
    attempt: int
    duration_seconds: float
    result_summary: str | None = None
    error: str | None = None
    idempotency_key: str | None = None


class ManualScheduler:
    """Scheduler executes only when `tick(now)` is called; it starts no worker thread.

    This design is reproducible in tests and lets the hosting app control lifecycle.
    Callback timeouts cannot safely stop arbitrary synchronous Python functions, so
    max_runtime_seconds is recorded as a threshold, not a hard interruption mechanism.
    """
    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._jobs: dict[str, JobSpec] = {}
        self._next_due: dict[str, datetime] = {}
        self._runs: list[JobRun] = []
        self._completed_keys: set[tuple[str, str]] = set()
        self._lock = RLock()

    def register(self, spec: JobSpec, *, now: datetime | None = None, replace: bool = False) -> None:
        stamp = _aware_utc(now or datetime.now(timezone.utc))
        with self._lock:
            if spec.name in self._jobs and not replace:
                raise ValueError(f"job already registered: {spec.name}")
            self._jobs[spec.name] = spec
            self._next_due[spec.name] = stamp if spec.run_immediately else stamp + spec.interval

    def unregister(self, name: str) -> bool:
        with self._lock:
            self._next_due.pop(name, None)
            return self._jobs.pop(name, None) is not None

    def enable(self, name: str, enabled: bool, *, now: datetime | None = None) -> None:
        with self._lock:
            if name not in self._jobs:
                raise KeyError(name)
            old = self._jobs[name]
            self._jobs[name] = JobSpec(old.name, old.callback, old.interval, enabled,
                                       old.run_immediately, old.max_retries,
                                       old.retry_backoff_seconds, old.max_runtime_seconds)
            if enabled and self._next_due[name] < _aware_utc(now or datetime.now(timezone.utc)):
                self._next_due[name] = _aware_utc(now or datetime.now(timezone.utc))

    def tick(self, now: datetime | None = None, *, max_jobs: int = 100) -> list[JobRun]:
        stamp = _aware_utc(now or datetime.now(timezone.utc))
        if max_jobs < 1:
            raise ValueError("max_jobs must be >= 1")
        due: list[tuple[JobSpec, datetime]] = []
        with self._lock:
            for name in sorted(self._jobs):
                spec = self._jobs[name]
                target = self._next_due[name]
                if spec.enabled and target <= stamp and len(due) < max_jobs:
                    due.append((spec, target))
                    # Schedule from prior due time to avoid drift; skip missed intervals to first future slot.
                    while self._next_due[name] <= stamp:
                        self._next_due[name] += spec.interval
        return [self._run_job(spec, scheduled_at, stamp) for spec, scheduled_at in due]

    def _run_job(self, spec: JobSpec, scheduled_at: datetime, now: datetime) -> JobRun:
        started_stamp = _aware_utc(now)
        started = self._clock()
        last_error = None
        for attempt in range(spec.max_retries + 1):
            try:
                value = spec.callback()
                finished_clock = self._clock()
                duration = max(0.0, finished_clock - started)
                overrun = spec.max_runtime_seconds is not None and duration > spec.max_runtime_seconds
                run = JobRun(uuid4().hex, spec.name, JobStatus.FAILED if overrun else JobStatus.SUCCEEDED,
                             scheduled_at.isoformat(), started_stamp.isoformat(),
                             _aware_utc(now).isoformat(), attempt + 1, duration,
                             str(value)[:500], "runtime threshold exceeded" if overrun else None)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < spec.max_retries:
                    if spec.retry_backoff_seconds:
                        sleep(spec.retry_backoff_seconds * (2 ** attempt))
                    continue
                run = JobRun(uuid4().hex, spec.name, JobStatus.FAILED, scheduled_at.isoformat(),
                             started_stamp.isoformat(), _aware_utc(now).isoformat(), attempt + 1,
                             max(0.0, self._clock() - started), error=last_error)
        with self._lock:
            self._runs.append(run)
            self._runs = self._runs[-10_000:]
        return run

    def run_once(self, name: str, *, idempotency_key: str | None = None,
                 now: datetime | None = None) -> JobRun:
        stamp = _aware_utc(now or datetime.now(timezone.utc))
        with self._lock:
            if name not in self._jobs:
                raise KeyError(name)
            spec = self._jobs[name]
            if not spec.enabled:
                raise RuntimeError(f"job is disabled: {name}")
            if idempotency_key:
                key = (name, idempotency_key)
                if key in self._completed_keys:
                    return JobRun("duplicate", name, JobStatus.SKIPPED, stamp.isoformat(), None,
                                  None, 0, 0.0, "duplicate idempotency key", idempotency_key=idempotency_key)
            else:
                key = None
        run = self._run_job(spec, stamp, stamp)
        if key and run.status is JobStatus.SUCCEEDED:
            with self._lock:
                self._completed_keys.add(key)
        if idempotency_key:
            run = JobRun(run.run_id, run.job_name, run.status, run.scheduled_at, run.started_at,
                         run.finished_at, run.attempt, run.duration_seconds, run.result_summary,
                         run.error, idempotency_key)
        return run

    @property
    def history(self) -> tuple[JobRun, ...]:
        with self._lock:
            return tuple(self._runs)

    def describe(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: {"enabled": spec.enabled, "interval_seconds": spec.interval.total_seconds(),
                           "next_due": self._next_due[name].isoformat(), "max_retries": spec.max_retries}
                    for name, spec in sorted(self._jobs.items())}


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduler timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)
