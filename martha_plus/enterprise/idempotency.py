"""In-memory idempotency registry with payload conflict detection and TTL cleanup.

For multi-process deployments use a shared durable store with a unique key
constraint; this implementation is intentionally process-local.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping


class IdempotencyConflict(ValueError):
    """A reused idempotency key was submitted with a different payload."""


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    request_digest: str
    response: Any
    created_at: float
    expires_at: float
    status: str = "completed"


@dataclass(frozen=True, slots=True)
class IdempotencyResult:
    response: Any
    replayed: bool
    key: str
    request_digest: str


def canonical_digest(payload: Any) -> str:
    """Produce a stable SHA-256 digest for JSON-compatible request data."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class IdempotencyRegistry:
    def __init__(self, *, ttl_seconds: float = 86400.0,
                 clock: Callable[[], float] = monotonic,
                 max_entries: int = 100_000) -> None:
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max_entries
        self._clock = clock
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = RLock()

    def _purge_locked(self, now: float) -> int:
        expired = [k for k, v in self._records.items() if v.expires_at <= now]
        for key in expired:
            del self._records[key]
        return len(expired)

    def execute(self, key: str, payload: Any, operation: Callable[[], Any]) -> IdempotencyResult:
        normalized = key.strip()
        if not normalized or len(normalized) > 256:
            raise ValueError("idempotency key must be 1..256 characters")
        if not callable(operation):
            raise TypeError("operation must be callable")
        digest = canonical_digest(payload)
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            existing = self._records.get(normalized)
            if existing:
                if existing.request_digest != digest:
                    raise IdempotencyConflict("key was already used for a different request")
                return IdempotencyResult(existing.response, True, normalized, digest)
            if len(self._records) >= self.max_entries:
                # Remove earliest-expiring item, avoiding unbounded memory growth.
                oldest = min(self._records, key=lambda k: self._records[k].expires_at)
                del self._records[oldest]
        # Run user code outside lock to avoid blocking unrelated requests.
        response = operation()
        completed_at = self._clock()
        record = IdempotencyRecord(normalized, digest, response, completed_at,
                                   completed_at + self.ttl_seconds)
        with self._lock:
            self._purge_locked(completed_at)
            raced = self._records.get(normalized)
            if raced:
                if raced.request_digest != digest:
                    raise IdempotencyConflict("concurrent request reused key with different payload")
                return IdempotencyResult(raced.response, True, normalized, digest)
            self._records[normalized] = record
        return IdempotencyResult(response, False, normalized, digest)

    def inspect(self, key: str) -> IdempotencyRecord | None:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            return self._records.get(key)

    def forget(self, key: str) -> bool:
        with self._lock:
            return self._records.pop(key, None) is not None

    def cleanup(self) -> int:
        with self._lock:
            return self._purge_locked(self._clock())

    def stats(self) -> dict[str, int | float]:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            return {"entries": len(self._records), "max_entries": self.max_entries,
                    "ttl_seconds": self.ttl_seconds}
