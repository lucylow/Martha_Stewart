"""In-process event bus with idempotency, retry policy, and dead-letter capture."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import monotonic, sleep
from typing import Any, Callable, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_type: str
    payload: Mapping[str, Any]
    event_id: str = field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    source: str = "martha_plus"

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type is required")
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class DeliveryAttempt:
    event_id: str
    handler_name: str
    attempt: int
    success: bool
    duration_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DeadLetter:
    event: EventEnvelope
    handler_name: str
    attempts: int
    error: str
    failed_at: datetime


@dataclass(frozen=True, slots=True)
class PublishResult:
    event_id: str
    delivered: int
    failed: int
    skipped_duplicate: bool
    attempts: tuple[DeliveryAttempt, ...]


Handler = Callable[[EventEnvelope], Any]


class EventBus:
    """A synchronous bus suited to app-local workflows, not a distributed queue.

    Handler exceptions are isolated. Retries are bounded and opt-in per publish.
    The idempotency cache is in-memory, bounded, and lost on process restart.
    """
    def __init__(self, *, idempotency_capacity: int = 10_000,
                 dead_letter_capacity: int = 1_000) -> None:
        if idempotency_capacity < 1 or dead_letter_capacity < 1:
            raise ValueError("capacities must be >= 1")
        self.idempotency_capacity = idempotency_capacity
        self.dead_letter_capacity = dead_letter_capacity
        self._handlers: dict[str, list[tuple[str, Handler]]] = {}
        self._processed: dict[tuple[str, str], None] = {}
        self._dead_letters: list[DeadLetter] = []
        self._lock = RLock()
        self._attempts: list[DeliveryAttempt] = []

    def subscribe(self, event_type: str, handler: Handler, *, name: str | None = None) -> Callable[[], None]:
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type is required")
        if not callable(handler):
            raise TypeError("handler must be callable")
        handler_name = name or getattr(handler, "__name__", handler.__class__.__name__)
        key = (event_type, handler_name)
        with self._lock:
            entries = self._handlers.setdefault(event_type, [])
            if any(existing_name == handler_name for existing_name, _ in entries):
                raise ValueError(f"handler name already subscribed for {event_type}: {handler_name}")
            entries.append((handler_name, handler))
        def unsubscribe() -> None:
            with self._lock:
                current = self._handlers.get(event_type, [])
                self._handlers[event_type] = [(n, fn) for n, fn in current if n != handler_name]
        return unsubscribe

    def publish(self, event: EventEnvelope, *, retries: int = 0,
                backoff_seconds: float = 0.0, idempotent: bool = True) -> PublishResult:
        if retries < 0 or retries > 10:
            raise ValueError("retries must be between zero and ten")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        with self._lock:
            if idempotent and (event.event_type, event.event_id) in self._processed:
                return PublishResult(event.event_id, 0, 0, True, ())
            handlers = tuple(self._handlers.get(event.event_type, ()))
        attempts: list[DeliveryAttempt] = []
        delivered = failed = 0
        for handler_name, handler in handlers:
            handler_key = (event.event_type, f"{event.event_id}:{handler_name}")
            with self._lock:
                already_delivered = idempotent and handler_key in self._processed
            if already_delivered:
                delivered += 1
                continue
            handler_success = False
            last_error = "unknown error"
            for attempt_no in range(retries + 1):
                started = monotonic()
                try:
                    handler(event)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    result = DeliveryAttempt(event.event_id, handler_name, attempt_no + 1, False,
                                             monotonic() - started, last_error)
                    attempts.append(result)
                    if attempt_no < retries and backoff_seconds:
                        sleep(backoff_seconds * (2 ** attempt_no))
                    continue
                attempts.append(DeliveryAttempt(event.event_id, handler_name, attempt_no + 1, True,
                                                monotonic() - started))
                handler_success = True
                break
            if handler_success:
                delivered += 1
                if idempotent:
                    with self._lock:
                        self._processed[handler_key] = None
                        self._trim_processed()
            else:
                failed += 1
                with self._lock:
                    self._dead_letters.append(DeadLetter(event, handler_name, retries + 1, last_error,
                                                         datetime.now(timezone.utc)))
                    self._dead_letters = self._dead_letters[-self.dead_letter_capacity:]
        with self._lock:
            self._attempts.extend(attempts)
            self._attempts = self._attempts[-self.idempotency_capacity:]
            if idempotent and failed == 0:
                self._processed[(event.event_type, event.event_id)] = None
                self._trim_processed()
        return PublishResult(event.event_id, delivered, failed, False, tuple(attempts))

    def _trim_processed(self) -> None:
        while len(self._processed) > self.idempotency_capacity:
            self._processed.pop(next(iter(self._processed)))

    @property
    def dead_letters(self) -> tuple[DeadLetter, ...]:
        with self._lock:
            return tuple(self._dead_letters)

    @property
    def attempts(self) -> tuple[DeliveryAttempt, ...]:
        with self._lock:
            return tuple(self._attempts)

    def handler_counts(self) -> dict[str, int]:
        with self._lock:
            return {name: len(handlers) for name, handlers in self._handlers.items()}

    def clear_dead_letters(self) -> int:
        with self._lock:
            count = len(self._dead_letters)
            self._dead_letters.clear()
            return count

    def replay_dead_letters(self, *, handler_names: set[str] | None = None,
                            retries: int = 0) -> list[PublishResult]:
        if retries < 0 or retries > 10:
            raise ValueError("retries must be between zero and ten")
        with self._lock:
            selected = [d for d in self._dead_letters if handler_names is None or d.handler_name in handler_names]
            self._dead_letters = [d for d in self._dead_letters if d not in selected]
        results: list[PublishResult] = []
        for dead in selected:
            # Deliver only to the handler that failed originally; other subscribers already
            # received the event and must not have their side effects repeated.
            with self._lock:
                handler = next((fn for name, fn in self._handlers.get(dead.event.event_type, ())
                                if name == dead.handler_name), None)
            replayed = EventEnvelope(dead.event.event_type, dead.event.payload,
                                     event_id=f"replay-{uuid4().hex}",
                                     correlation_id=dead.event.correlation_id or dead.event.event_id,
                                     source=f"replay:{dead.event.source}")
            if handler is None:
                with self._lock:
                    self._dead_letters = (self._dead_letters + [dead])[-self.dead_letter_capacity:]
                results.append(PublishResult(replayed.event_id, 0, 1, False, ()))
                continue
            attempts: list[DeliveryAttempt] = []
            succeeded = False
            last_error = "unknown error"
            for attempt_no in range(retries + 1):
                started = monotonic()
                try:
                    handler(replayed)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempts.append(DeliveryAttempt(replayed.event_id, dead.handler_name, attempt_no + 1,
                                                   False, monotonic() - started, last_error))
                else:
                    attempts.append(DeliveryAttempt(replayed.event_id, dead.handler_name, attempt_no + 1,
                                                   True, monotonic() - started))
                    succeeded = True
                    break
            with self._lock:
                self._attempts.extend(attempts)
                self._attempts = self._attempts[-self.idempotency_capacity:]
                if not succeeded:
                    self._dead_letters = (self._dead_letters + [DeadLetter(replayed, dead.handler_name,
                        retries + 1, last_error, datetime.now(timezone.utc))])[-self.dead_letter_capacity:]
            results.append(PublishResult(replayed.event_id, int(succeeded), int(not succeeded), False, tuple(attempts)))
        return results
