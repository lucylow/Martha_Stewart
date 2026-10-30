"""Deterministic clocks, latency distributions, and timestamp utilities."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable


NANOSECONDS_PER_SECOND = 1_000_000_000


class LatencyKind(str, Enum):
    MARKET_DATA = "market_data"
    ORDER_ENTRY = "order_entry"
    ORDER_ACK = "order_ack"
    CANCEL = "cancel"
    REPLACE = "replace"


@dataclass(slots=True)
class SimClock:
    """Monotonic virtual clock; advancing backwards is always an error."""

    now_ns: int = 0

    def __post_init__(self) -> None:
        if self.now_ns < 0:
            raise ValueError("clock must start at a non-negative timestamp")

    def advance_to(self, ts_ns: int) -> int:
        if ts_ns < self.now_ns:
            raise ValueError(f"cannot move clock backwards ({ts_ns} < {self.now_ns})")
        self.now_ns = ts_ns
        return self.now_ns

    def advance_by(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError("clock delta must be non-negative")
        self.now_ns += delta_ns
        return self.now_ns

    def reset(self, ts_ns: int = 0) -> None:
        if ts_ns < 0:
            raise ValueError("reset timestamp must be non-negative")
        self.now_ns = ts_ns


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Per-message latency parameters in nanoseconds.

    The model is a clipped normal distribution.  The deterministic seed lives
    on ``LatencyModel`` so multiple scenarios can be reproduced exactly.
    """

    base_ns: int = 50_000
    jitter_ns: int = 10_000
    floor_ns: int = 0
    ceiling_ns: int = 1_000_000

    def __post_init__(self) -> None:
        if min(self.base_ns, self.jitter_ns, self.floor_ns) < 0:
            raise ValueError("latency values must be non-negative")
        if self.ceiling_ns < self.floor_ns:
            raise ValueError("latency ceiling must be >= floor")


class LatencyModel:
    """Reproducible per-channel latency sampler."""

    def __init__(self, profiles: dict[LatencyKind, LatencyProfile] | None = None, seed: int = 7):
        self.profiles = profiles or {
            LatencyKind.MARKET_DATA: LatencyProfile(40_000, 8_000),
            LatencyKind.ORDER_ENTRY: LatencyProfile(55_000, 12_000),
            LatencyKind.ORDER_ACK: LatencyProfile(45_000, 9_000),
            LatencyKind.CANCEL: LatencyProfile(60_000, 15_000),
            LatencyKind.REPLACE: LatencyProfile(65_000, 15_000),
        }
        self._random = random.Random(seed)
        self.seed = seed
        self.samples: dict[LatencyKind, list[int]] = {kind: [] for kind in LatencyKind}

    def sample(self, kind: LatencyKind) -> int:
        profile = self.profiles[kind]
        value = int(round(self._random.gauss(profile.base_ns, profile.jitter_ns)))
        value = min(profile.ceiling_ns, max(profile.floor_ns, value))
        self.samples[kind].append(value)
        return value

    def sample_many(self, kind: LatencyKind, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.sample(kind) for _ in range(count)]

    def summary(self, kind: LatencyKind) -> dict[str, float | int]:
        values = self.samples[kind]
        if not values:
            return {"count": 0, "mean_ns": 0, "median_ns": 0, "p90_ns": 0, "p99_ns": 0, "max_ns": 0}
        ordered = sorted(values)
        def percentile(p: float) -> int:
            return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))]
        return {
            "count": len(values),
            "mean_ns": statistics.fmean(values),
            "median_ns": statistics.median(values),
            "p90_ns": percentile(0.90),
            "p99_ns": percentile(0.99),
            "max_ns": max(values),
        }


def datetime_to_epoch_ns(value: datetime) -> int:
    """Convert an aware datetime to integer UTC nanoseconds."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    return int(utc.timestamp() * NANOSECONDS_PER_SECOND)


def epoch_ns_to_datetime(value: int) -> datetime:
    if value < 0:
        raise ValueError("timestamp must be non-negative")
    return datetime.fromtimestamp(value / NANOSECONDS_PER_SECOND, tz=timezone.utc)


def percentile(values: Iterable[int], probability: float) -> int:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sequence")
    index = int(round((len(ordered) - 1) * probability))
    return ordered[index]
