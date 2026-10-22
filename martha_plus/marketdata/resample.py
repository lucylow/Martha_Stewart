"""OHLCV bar resampling for local research data."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Sequence
from ..types import Bar


def resample_bars(bars: Sequence[Bar], interval_seconds: int) -> list[Bar]:
    """Aggregate bars into fixed UTC epoch buckets.

    All source bars must use the same symbol and be time-ordered. A bucket's
    timestamp is the timestamp of its final observed source bar; incomplete
    buckets are retained and should be treated carefully in analysis.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if not bars:
        return []
    symbol = bars[0].symbol
    if any(bar.symbol != symbol for bar in bars):
        raise ValueError("resampling requires a single symbol")
    if any(a.timestamp >= b.timestamp for a, b in zip(bars, bars[1:])):
        raise ValueError("bars must be strictly time-ordered")
    groups: dict[int, list[Bar]] = defaultdict(list)
    for bar in bars:
        bucket = int(bar.timestamp.astimezone(timezone.utc).timestamp()) // interval_seconds
        groups[bucket].append(bar)
    output: list[Bar] = []
    for bucket in sorted(groups):
        samples = groups[bucket]
        output.append(Bar(symbol, samples[-1].timestamp, samples[0].open,
                          max(item.high for item in samples), min(item.low for item in samples),
                          samples[-1].close, sum(item.volume for item in samples)))
    return output
