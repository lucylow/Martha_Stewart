"""Deterministic event-time buffering, watermarking, and bar aggregation.

This module accepts events from any source. It deliberately contains no socket
client: transport/reconnection belongs in an adapter around these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import json


class LateEventPolicy(str, Enum):
    DROP = "drop"
    ACCEPT = "accept"
    SIDE_OUTPUT = "side_output"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event_id: str
    topic: str
    symbol: str
    event_time: datetime
    received_time: datetime
    payload: Mapping[str, float | str | int | bool | None] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.topic.strip() or not self.symbol.strip():
            raise ValueError("event_id, topic and symbol are required")
        if self.event_time.tzinfo is None or self.received_time.tzinfo is None:
            raise ValueError("event times must be timezone-aware")

    @property
    def event_digest(self) -> str:
        payload = {"id": self.event_id, "topic": self.topic, "symbol": self.symbol,
                   "event_time": self.event_time.isoformat(), "payload": dict(self.payload),
                   "source": self.source}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StreamStats:
    accepted: int
    duplicates: int
    dropped_late: int
    side_output: int
    invalid: int
    watermark: datetime | None
    buffered: int


class EventTimeBuffer:
    """Reorder bounded lateness while remembering duplicates by event identifier."""
    def __init__(self, *, allowed_lateness: timedelta = timedelta(seconds=5),
                 late_policy: LateEventPolicy = LateEventPolicy.DROP,
                 max_seen: int = 500_000) -> None:
        if allowed_lateness < timedelta(0) or max_seen < 1:
            raise ValueError("lateness cannot be negative and max_seen must be positive")
        self.allowed_lateness = allowed_lateness
        self.late_policy = late_policy
        self.max_seen = max_seen
        self._max_event_time: datetime | None = None
        self._heap: list[tuple[datetime, int, StreamEvent]] = []
        self._seen: dict[str, str] = {}
        self._late: list[StreamEvent] = []
        self._counter = 0
        self._accepted = self._duplicates = self._dropped = self._invalid = 0
        self._lock = RLock()

    @property
    def watermark(self) -> datetime | None:
        with self._lock:
            return self._max_event_time - self.allowed_lateness if self._max_event_time else None

    @property
    def late_events(self) -> tuple[StreamEvent, ...]:
        with self._lock:
            return tuple(self._late)

    def push(self, event: StreamEvent) -> tuple[StreamEvent, ...]:
        with self._lock:
            prior = self._seen.get(event.event_id)
            if prior is not None:
                if prior != event.event_digest:
                    self._invalid += 1
                    raise ValueError("event_id was reused with different content")
                self._duplicates += 1
                return ()
            watermark = self._max_event_time - self.allowed_lateness if self._max_event_time else None
            if watermark and event.event_time < watermark:
                if self.late_policy is LateEventPolicy.DROP:
                    self._dropped += 1
                    self._remember(event)
                    return ()
                if self.late_policy is LateEventPolicy.SIDE_OUTPUT:
                    self._late.append(event)
                    self._remember(event)
                    return ()
            self._remember(event)
            if self._max_event_time is None or event.event_time > self._max_event_time:
                self._max_event_time = event.event_time
            import heapq
            heapq.heappush(self._heap, (event.event_time, self._counter, event))
            self._counter += 1
            self._accepted += 1
            return self._drain_locked(self._max_event_time - self.allowed_lateness)

    def _remember(self, event: StreamEvent) -> None:
        self._seen[event.event_id] = event.event_digest
        if len(self._seen) > self.max_seen:
            # Dict insertion order is stable: evict oldest-seen identifiers.
            for key in list(self._seen)[:max(1, self.max_seen // 10)]:
                del self._seen[key]

    def _drain_locked(self, watermark: datetime | None) -> tuple[StreamEvent, ...]:
        import heapq
        output = []
        while self._heap and (watermark is None or self._heap[0][0] <= watermark):
            output.append(heapq.heappop(self._heap)[2])
        return tuple(output)

    def flush(self) -> tuple[StreamEvent, ...]:
        with self._lock:
            import heapq
            output = []
            while self._heap:
                output.append(heapq.heappop(self._heap)[2])
            return tuple(output)

    def stats(self) -> StreamStats:
        with self._lock:
            watermark = self._max_event_time - self.allowed_lateness if self._max_event_time else None
            return StreamStats(self._accepted, self._duplicates, self._dropped, len(self._late),
                               self._invalid, watermark, len(self._heap))


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    symbol: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None or self.end <= self.start:
            raise ValueError("bar interval must be positive and timezone-aware")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(not __import__("math").isfinite(v) for v in values):
            raise ValueError("bar values must be finite")
        if self.volume < 0 or self.trade_count < 0:
            raise ValueError("volume and trade count cannot be negative")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError("inconsistent OHLC values")


class BarAggregator:
    """Aggregate trade-like stream events into fixed-duration bars."""
    def __init__(self, interval: timedelta, *, price_key: str = "price",
                 size_key: str = "size", timestamp_floor: bool = True) -> None:
        if interval <= timedelta(0):
            raise ValueError("interval must be positive")
        self.interval = interval
        self.price_key = price_key
        self.size_key = size_key
        self.timestamp_floor = timestamp_floor
        self._groups: dict[tuple[str, datetime], list[tuple[datetime, float, float]]] = {}
        self._lock = RLock()

    def _start(self, timestamp: datetime) -> datetime:
        seconds = int(self.interval.total_seconds())
        if seconds <= 0:
            raise ValueError("sub-second bar aggregation is not supported")
        epoch = int(timestamp.timestamp())
        floored = epoch - epoch % seconds if self.timestamp_floor else epoch
        return datetime.fromtimestamp(floored, tz=timestamp.tzinfo)

    def add(self, event: StreamEvent) -> None:
        if self.price_key not in event.payload:
            raise ValueError(f"event is missing {self.price_key}")
        price = float(event.payload[self.price_key])
        size = float(event.payload.get(self.size_key, 0.0))
        if not __import__("math").isfinite(price) or price <= 0:
            raise ValueError("price must be finite and positive")
        if not __import__("math").isfinite(size) or size < 0:
            raise ValueError("size must be finite and non-negative")
        bucket = self._start(event.event_time)
        with self._lock:
            self._groups.setdefault((event.symbol, bucket), []).append((event.event_time, price, size))

    def close_before(self, watermark: datetime) -> tuple[OHLCVBar, ...]:
        if watermark.tzinfo is None:
            raise ValueError("watermark must be timezone-aware")
        with self._lock:
            keys = sorted((k for k in self._groups if k[1] + self.interval <= watermark),
                          key=lambda k: (k[1], k[0]))
            bars = tuple(self._make_bar(symbol, start, self._groups.pop((symbol, start)))
                         for symbol, start in keys)
            return bars

    def flush(self) -> tuple[OHLCVBar, ...]:
        with self._lock:
            keys = sorted(self._groups, key=lambda k: (k[1], k[0]))
            return tuple(self._make_bar(s, t, self._groups.pop((s, t))) for s, t in keys)

    def _make_bar(self, symbol: str, start: datetime,
                  points: Sequence[tuple[datetime, float, float]]) -> OHLCVBar:
        ordered = sorted(points, key=lambda row: row[0])
        prices = [r[1] for r in ordered]
        return OHLCVBar(symbol, start, start + self.interval, prices[0], max(prices),
                        min(prices), prices[-1], sum(r[2] for r in ordered), len(ordered))

    def pending_buckets(self) -> int:
        with self._lock:
            return len(self._groups)


class StreamRouter:
    """Topic-to-handler router that isolates handler failures per subscription."""
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[StreamEvent], None]]] = {}
        self._history: list[dict[str, str]] = []
        self._lock = RLock()

    def subscribe(self, topic: str, handler: Callable[[StreamEvent], None]) -> None:
        if not topic.strip() or not callable(handler):
            raise ValueError("topic and handler are required")
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[StreamEvent], None]) -> bool:
        with self._lock:
            handlers = self._handlers.get(topic, [])
            if handler not in handlers:
                return False
            handlers.remove(handler)
            if not handlers:
                self._handlers.pop(topic, None)
            return True

    def dispatch(self, event: StreamEvent) -> tuple[Exception, ...]:
        with self._lock:
            handlers = tuple(self._handlers.get(event.topic, ()))
        errors: list[Exception] = []
        for index, handler in enumerate(handlers):
            try:
                handler(event)
                self._history.append({"event_id": event.event_id, "topic": event.topic,
                                      "handler": str(index), "status": "ok"})
            except Exception as exc:  # handler boundary: record, don't starve other subscribers
                errors.append(exc)
                self._history.append({"event_id": event.event_id, "topic": event.topic,
                                      "handler": str(index), "status": type(exc).__name__})
        return tuple(errors)

    @property
    def history(self) -> tuple[Mapping[str, str], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._history)
