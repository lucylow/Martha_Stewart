"""Minimal thread-safe metrics registry; values are local process metrics only."""
from __future__ import annotations
from collections import defaultdict, deque
from threading import RLock
from time import monotonic
from typing import Mapping

class MetricsRegistry:
    def __init__(self, recent_sample_limit: int = 10_000):
        if recent_sample_limit < 1:
            raise ValueError("recent_sample_limit must be positive")
        self.recent_sample_limit = recent_sample_limit
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=recent_sample_limit))
        self._started_at = monotonic()
        self._lock = RLock()

    def inc(self, name: str, amount: float = 1.0) -> None:
        if not name or amount < 0:
            raise ValueError("metric name required and amount must be non-negative")
        with self._lock:
            self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        if not name:
            raise ValueError("metric name required")
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        if not name:
            raise ValueError("metric name required")
        with self._lock:
            self._samples[name].append(float(value))

    def snapshot(self) -> dict:
        with self._lock:
            summaries = {}
            for name, values in self._samples.items():
                seq = sorted(values)
                n = len(seq)
                summaries[name] = {
                    "count": n,
                    "min": seq[0] if n else 0.0,
                    "max": seq[-1] if n else 0.0,
                    "mean": sum(seq) / n if n else 0.0,
                    "p50": seq[min(n-1, int((n-1)*0.50))] if n else 0.0,
                    "p95": seq[min(n-1, int((n-1)*0.95))] if n else 0.0,
                }
            return {"uptime_seconds": monotonic() - self._started_at,
                    "counters": dict(self._counters), "gauges": dict(self._gauges),
                    "observations": summaries}
