"""Framework-neutral request validation, token-bucket limiting, and circuit protection."""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping, Sequence
import re


class RequestRejected(ValueError):
    def __init__(self, reason: str, status_code: int = 400) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    max_body_bytes: int = 1_000_000
    max_query_params: int = 100
    max_header_count: int = 100
    allowed_methods: tuple[str, ...] = ("GET", "POST")
    required_content_types: Mapping[str, tuple[str, ...]] | None = None
    allow_unknown_query_params: bool = False

    def __post_init__(self) -> None:
        if min(self.max_body_bytes, self.max_query_params, self.max_header_count) < 1:
            raise ValueError("request limits must be positive")
        if any(method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
               for method in self.allowed_methods):
            raise ValueError("unsupported HTTP method in policy")


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    method: str
    path: str
    body_bytes: int
    query_count: int
    header_count: int
    content_type: str | None
    client_id: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining_tokens: float
    retry_after_seconds: float
    reset_after_seconds: float


class TokenBucketLimiter:
    """Per-key token bucket; caller chooses the identity key and enforces result."""
    def __init__(self, *, capacity: float, refill_per_second: float,
                 clock: Callable[[], float] = monotonic, max_keys: int = 100_000) -> None:
        if capacity <= 0 or refill_per_second <= 0 or max_keys < 1:
            raise ValueError("capacity, refill rate, and max_keys must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_per_second)
        self.clock = clock
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = RLock()

    def consume(self, key: str, *, cost: float = 1.0) -> RateLimitResult:
        if not key.strip() or cost <= 0 or cost > self.capacity:
            raise ValueError("key and valid cost are required")
        now = self.clock()
        with self._lock:
            tokens, updated = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + max(0.0, now - updated) * self.refill_rate)
            if tokens >= cost:
                tokens -= cost
                allowed, retry = True, 0.0
            else:
                allowed = False
                retry = (cost - tokens) / self.refill_rate
            if key not in self._buckets and len(self._buckets) >= self.max_keys:
                # Evict the least recently updated identity. Deployments needing strict
                # enforcement across workers should use a shared atomic limiter.
                oldest = min(self._buckets, key=lambda k: self._buckets[k][1])
                del self._buckets[oldest]
            self._buckets[key] = (tokens, now)
            return RateLimitResult(allowed, tokens, retry,
                                   (self.capacity - tokens) / self.refill_rate)

    def reset(self, key: str | None = None) -> int:
        with self._lock:
            if key is not None:
                return int(self._buckets.pop(key, None) is not None)
            count = len(self._buckets)
            self._buckets.clear()
            return count

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {"keys": len(self._buckets), "max_keys": self.max_keys,
                    "capacity": self.capacity, "refill_per_second": self.refill_rate}


class RequestGuard:
    """Validate basic request metadata before parsing or routing a request body."""
    def __init__(self, policy: RequestPolicy | None = None) -> None:
        self.policy = policy or RequestPolicy()

    def validate(self, metadata: RequestMetadata, *, known_query_params: Iterable[str] | None = None) -> None:
        method = metadata.method.upper()
        if method not in {m.upper() for m in self.policy.allowed_methods}:
            raise RequestRejected("method_not_allowed", 405)
        if not metadata.path.startswith("/") or "\r" in metadata.path or "\n" in metadata.path:
            raise RequestRejected("invalid_path")
        if metadata.body_bytes < 0 or metadata.body_bytes > self.policy.max_body_bytes:
            raise RequestRejected("body_too_large", 413)
        if metadata.query_count < 0 or metadata.query_count > self.policy.max_query_params:
            raise RequestRejected("too_many_query_parameters", 414)
        if metadata.header_count < 0 or metadata.header_count > self.policy.max_header_count:
            raise RequestRejected("too_many_headers", 431)
        if self.policy.required_content_types and method in self.policy.required_content_types:
            accepted = self.policy.required_content_types[method]
            if metadata.content_type not in accepted:
                raise RequestRejected("unsupported_media_type", 415)
        if known_query_params is not None and not self.policy.allow_unknown_query_params:
            # The metadata object intentionally holds counts only; adapters should pass
            # the actual parsed query keys to validate_query for strict allow-listing.
            if metadata.query_count and not tuple(known_query_params):
                raise RequestRejected("query_allowlist_missing")

    def validate_query(self, query: Mapping[str, Any], *, allowed: Iterable[str],
                       required: Iterable[str] = ()) -> None:
        allowed_set, required_set = set(allowed), set(required)
        extras = set(query) - allowed_set
        missing = required_set - set(query)
        if extras:
            raise RequestRejected("unknown_query_parameters:" + ",".join(sorted(extras)))
        if missing:
            raise RequestRejected("missing_query_parameters:" + ",".join(sorted(missing)))

    def validate_json_object(self, body: Any, *, allowed_fields: Iterable[str],
                             required_fields: Iterable[str] = (),
                             max_string_length: int = 4096) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise RequestRejected("json_body_must_be_object")
        allowed, required = set(allowed_fields), set(required_fields)
        if set(body) - allowed:
            raise RequestRejected("unknown_fields:" + ",".join(sorted(set(body) - allowed)))
        if required - set(body):
            raise RequestRejected("missing_fields:" + ",".join(sorted(required - set(body))))
        for key, value in body.items():
            if isinstance(value, str) and len(value) > max_string_length:
                raise RequestRejected(f"field_too_long:{key}")
            if isinstance(value, float) and value != value:
                raise RequestRejected(f"non_finite_number:{key}")
        return dict(body)


@dataclass(frozen=True, slots=True)
class BreakerSnapshot:
    state: str
    failure_count: int
    opened_at: float | None
    retry_after_seconds: float


class CircuitGuard:
    """Tiny circuit breaker to prevent repeatedly calling an unhealthy dependency."""
    def __init__(self, *, failure_threshold: int = 5, reset_timeout: float = 30.0,
                 clock: Callable[[], float] = monotonic) -> None:
        if failure_threshold < 1 or reset_timeout <= 0:
            raise ValueError("invalid circuit settings")
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open = False
        self._lock = RLock()

    def allow_call(self) -> bool:
        now = self.clock()
        with self._lock:
            if self._opened_at is None:
                return True
            elapsed = now - self._opened_at
            if elapsed >= self.reset_timeout and not self._half_open:
                self._half_open = True
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._half_open = False

    def failure(self) -> None:
        now = self.clock()
        with self._lock:
            self._failures += 1
            if self._half_open or self._failures >= self.threshold:
                self._opened_at = now
                self._half_open = False

    def snapshot(self) -> BreakerSnapshot:
        now = self.clock()
        with self._lock:
            retry = max(0.0, self.reset_timeout - (now - self._opened_at)) if self._opened_at is not None else 0.0
            state = "closed" if self._opened_at is None else ("half_open" if retry == 0 else "open")
            return BreakerSnapshot(state, self._failures, self._opened_at, retry)


def sanitize_log_value(value: Any, *, max_length: int = 256) -> str:
    """Bound and flatten untrusted request values before logging; not HTML escaping."""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text[:max_length]
