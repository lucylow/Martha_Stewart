"""Dependency-aware in-process job runner with retries and cooperative cancellation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Event, RLock
from time import monotonic, sleep
from typing import Any, Callable, Iterable, Mapping
import traceback
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: str
    task: Callable[["JobContext"], Any]
    dependencies: tuple[str, ...] = ()
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    timeout_seconds: float | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not callable(self.task):
            raise ValueError("job_id and callable task are required")
        if self.max_attempts < 1 or self.retry_delay_seconds < 0:
            raise ValueError("attempt count must be positive and retry delay non-negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.job_id in self.dependencies:
            raise ValueError("job cannot depend on itself")


@dataclass(slots=True)
class JobContext:
    job_id: str
    attempt: int
    cancellation: Event
    shared: Mapping[str, Any] = field(default_factory=dict)

    def check_cancelled(self) -> None:
        if self.cancellation.is_set():
            raise JobCancelled(f"job {self.job_id} was cancelled")


class JobCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    status: JobStatus
    attempt: int
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    result: Any = None
    error: str | None = None
    trace: str | None = None


class JobRunner:
    """Sequential DAG runner; parallel execution is deliberately left to an adapter."""
    def __init__(self, *, clock=lambda: datetime.now(timezone.utc),
                 monotonic_clock=monotonic, shared: Mapping[str, Any] | None = None) -> None:
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.shared = dict(shared or {})
        self._specs: dict[str, JobSpec] = {}
        self._records: dict[str, JobRecord] = {}
        self._cancel: dict[str, Event] = {}
        self._lock = RLock()

    def add(self, spec: JobSpec) -> None:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        with self._lock:
            if spec.job_id in self._specs:
                raise ValueError(f"job already exists: {spec.job_id}")
            if any(dep not in self._specs for dep in spec.dependencies):
                raise ValueError("dependencies must be registered before dependent job")
            self._specs[spec.job_id] = spec
            self._records[spec.job_id] = JobRecord(spec.job_id, JobStatus.PENDING, 0, now.isoformat())
            self._cancel[spec.job_id] = Event()
            self._assert_acyclic_locked()

    def _assert_acyclic_locked(self) -> None:
        temporary, permanent = set(), set()
        def visit(node: str) -> None:
            if node in permanent:
                return
            if node in temporary:
                raise ValueError("job dependency graph contains a cycle")
            temporary.add(node)
            for dep in self._specs[node].dependencies:
                visit(dep)
            temporary.remove(node)
            permanent.add(node)
        for node in self._specs:
            visit(node)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._specs:
                return False
            self._cancel[job_id].set()
            rec = self._records[job_id]
            if rec.status is JobStatus.PENDING:
                self._records[job_id] = JobRecord(rec.job_id, JobStatus.CANCELLED, rec.attempt,
                    rec.submitted_at, finished_at=self.clock().isoformat(), error="cancelled before start")
            return True

    def record(self, job_id: str) -> JobRecord:
        with self._lock:
            return self._records[job_id]

    def records(self) -> tuple[JobRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def run(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._specs:
                raise KeyError(job_id)
            record = self._records[job_id]
            if record.status is not JobStatus.PENDING:
                return record
            spec = self._specs[job_id]
            for dep in spec.dependencies:
                dep_status = self._records[dep].status
                if dep_status is not JobStatus.SUCCEEDED:
                    skipped = JobRecord(job_id, JobStatus.SKIPPED, record.attempt,
                        record.submitted_at, finished_at=self.clock().isoformat(),
                        error=f"dependency {dep} is {dep_status.value}")
                    self._records[job_id] = skipped
                    return skipped
            if self._cancel[job_id].is_set():
                cancelled = JobRecord(job_id, JobStatus.CANCELLED, record.attempt,
                    record.submitted_at, finished_at=self.clock().isoformat(), error="cancelled before start")
                self._records[job_id] = cancelled
                return cancelled
            started = self.clock()
            self._records[job_id] = JobRecord(job_id, JobStatus.RUNNING, 0,
                                              record.submitted_at, started_at=started.isoformat())
        start_mono = self.monotonic_clock()
        last_error = None
        last_trace = None
        for attempt in range(1, spec.max_attempts + 1):
            if self._cancel[job_id].is_set():
                status = JobStatus.CANCELLED
                break
            try:
                context = JobContext(job_id, attempt, self._cancel[job_id], self.shared)
                context.check_cancelled()
                result = spec.task(context)
                elapsed = self.monotonic_clock() - start_mono
                if spec.timeout_seconds is not None and elapsed > spec.timeout_seconds:
                    raise TimeoutError(f"job exceeded timeout after {elapsed:.3f}s")
                finished = self.clock()
                final = JobRecord(job_id, JobStatus.SUCCEEDED, attempt, record.submitted_at,
                    started.isoformat(), finished.isoformat(), elapsed, result)
                with self._lock:
                    self._records[job_id] = final
                return final
            except JobCancelled as exc:
                last_error = str(exc)
                status = JobStatus.CANCELLED
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_trace = traceback.format_exc(limit=20)
                if attempt < spec.max_attempts and not self._cancel[job_id].is_set():
                    if spec.retry_delay_seconds:
                        # Bounded delay with cooperative cancel checks.
                        deadline = self.monotonic_clock() + spec.retry_delay_seconds
                        while self.monotonic_clock() < deadline:
                            if self._cancel[job_id].is_set():
                                break
                            sleep(min(0.02, max(0.0, deadline - self.monotonic_clock())))
                        if self._cancel[job_id].is_set():
                            status = JobStatus.CANCELLED
                            break
                    continue
                status = JobStatus.FAILED
                break
        finished = self.clock()
        final = JobRecord(job_id, status, attempt if 'attempt' in locals() else 0,
            record.submitted_at, started.isoformat(), finished.isoformat(),
            self.monotonic_clock() - start_mono, error=last_error, trace=last_trace)
        with self._lock:
            self._records[job_id] = final
        return final

    def run_ready(self, *, limit: int | None = None) -> tuple[JobRecord, ...]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        output = []
        while limit is None or len(output) < limit:
            with self._lock:
                candidates = [job_id for job_id, record in self._records.items()
                              if record.status is JobStatus.PENDING]
            if not candidates:
                break
            progressed = False
            for job_id in candidates:
                before = self.record(job_id).status
                record = self.run(job_id)
                if record.status is not before:
                    output.append(record)
                    progressed = True
                    if limit is not None and len(output) >= limit:
                        break
            if not progressed:
                break
        return tuple(output)

    def summary(self) -> dict[str, int]:
        with self._lock:
            rows = tuple(self._records.values())
        return {status.value: sum(r.status is status for r in rows) for status in JobStatus}
