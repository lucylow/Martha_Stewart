"""Circuit breaker for repeated provider or execution failures."""
from __future__ import annotations
from enum import Enum
from threading import RLock
from time import monotonic

class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0):
        if failure_threshold < 1 or recovery_seconds <= 0:
            raise ValueError("threshold and recovery time must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = RLock()

    @property
    def state(self) -> BreakerState:
        with self._lock:
            if self._state is BreakerState.OPEN and self._opened_at is not None:
                if monotonic() - self._opened_at >= self.recovery_seconds:
                    self._state = BreakerState.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = monotonic()

    def reset(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._opened_at = None
