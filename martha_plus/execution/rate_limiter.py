"""Thread-safe token bucket for limiting request/order attempt rate.

This limits client-side order submissions only; it does not represent a broker's
regulatory, exchange, or account-specific throttling rules.
"""
from __future__ import annotations
from threading import Condition
from time import monotonic
from typing import Callable

class TokenBucket:
    def __init__(self, capacity: float, refill_rate_per_second: float,
                 clock: Callable[[], float] = monotonic):
        if capacity <= 0 or refill_rate_per_second <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated_at = clock()
        self._condition = Condition()

    @property
    def available_tokens(self) -> float:
        with self._condition:
            self._refill_locked()
            return self._tokens

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._updated_at = max(self._updated_at, now)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        if tokens <= 0 or tokens > self.capacity:
            raise ValueError("requested tokens must be in (0, capacity]")
        with self._condition:
            self._refill_locked()
            if self._tokens + 1e-12 < tokens:
                return False
            self._tokens -= tokens
            return True

    def acquire(self, tokens: float = 1.0, timeout_seconds: float | None = None) -> bool:
        """Wait for tokens, or return False if timeout expires.

        Uses condition waits rather than a busy loop. With a custom frozen clock,
        use ``try_acquire`` in unit tests unless the clock advances externally.
        """
        if tokens <= 0 or tokens > self.capacity:
            raise ValueError("requested tokens must be in (0, capacity]")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        deadline = None if timeout_seconds is None else monotonic() + timeout_seconds
        with self._condition:
            while True:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                wait_for_tokens = (tokens - self._tokens) / self.refill_rate
                wait_for_timeout = remaining
                wait_for = wait_for_tokens if wait_for_timeout is None else min(wait_for_tokens, wait_for_timeout)
                self._condition.wait(timeout=max(0.001, wait_for))

    def reset(self) -> None:
        with self._condition:
            self._tokens = self.capacity
            self._updated_at = self._clock()
            self._condition.notify_all()
