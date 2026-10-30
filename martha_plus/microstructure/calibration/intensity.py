"""Poisson-style event-intensity estimates for replay design."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from ..types import EventKind, MarketEvent, Side


@dataclass(frozen=True, slots=True)
class IntensityEstimate:
    event_count: int
    exposure_seconds: Decimal
    events_per_second: Decimal
    events_per_minute: Decimal
    interarrival_mean_ns: Decimal | None
    interarrival_median_ns: int | None


def estimate_intensity(events: Iterable[MarketEvent], *, kind: EventKind | None = None, side: Side | None = None) -> IntensityEstimate:
    rows = sorted((x for x in events if (kind is None or x.kind is kind) and (side is None or x.side is side)),
                  key=lambda x: (x.ts_ns, x.sequence))
    if len(rows) < 2:
        raise ValueError("at least two matching events are required to estimate intensity")
    exposure_ns = rows[-1].ts_ns - rows[0].ts_ns
    if exposure_ns <= 0:
        raise ValueError("events must span a positive time interval")
    exposure_seconds = Decimal(exposure_ns) / Decimal(1_000_000_000)
    rate = Decimal(len(rows) - 1) / exposure_seconds
    gaps = [rows[i].ts_ns - rows[i - 1].ts_ns for i in range(1, len(rows))]
    gaps_sorted = sorted(gaps)
    middle = len(gaps_sorted) // 2
    median_gap = gaps_sorted[middle] if len(gaps_sorted) % 2 else (gaps_sorted[middle - 1] + gaps_sorted[middle]) // 2
    return IntensityEstimate(len(rows), exposure_seconds, rate, rate * 60,
                             Decimal(exposure_ns) / Decimal(len(gaps)), median_gap)
