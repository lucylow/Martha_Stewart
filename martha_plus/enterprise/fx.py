"""FX conversion graph with timestamp freshness and path diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from threading import RLock
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class FXQuote:
    base: str
    quote: str
    rate: float
    timestamp: datetime
    source: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.base.strip()) != 3 or len(self.quote.strip()) != 3:
            raise ValueError("currency codes must be 3-character codes")
        if self.base.upper() == self.quote.upper():
            raise ValueError("base and quote currencies must differ")
        if not isfinite(self.rate) or self.rate <= 0:
            raise ValueError("FX rate must be finite and positive")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FXConversion:
    amount: float
    source_currency: str
    target_currency: str
    converted_amount: float
    rate: float
    path: tuple[str, ...]
    quote_timestamps: tuple[str, ...]
    oldest_age_seconds: float


class FXRateUnavailable(LookupError):
    pass


class FXBook:
    """Maintain latest known FX quotes and convert using a simple shortest path."""
    def __init__(self, *, now=lambda: datetime.now(timezone.utc),
                 max_age: timedelta = timedelta(minutes=15)) -> None:
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self._now = now
        self.max_age = max_age
        self._quotes: dict[tuple[str, str], FXQuote] = {}
        self._lock = RLock()

    def update(self, quote: FXQuote) -> None:
        base, term = quote.base.upper(), quote.quote.upper()
        normalized = FXQuote(base, term, quote.rate, quote.timestamp, quote.source)
        with self._lock:
            old = self._quotes.get((base, term))
            if old is None or quote.timestamp >= old.timestamp:
                self._quotes[(base, term)] = normalized

    def extend(self, quotes: Iterable[FXQuote]) -> None:
        for quote in quotes:
            self.update(quote)

    def _edges(self, as_of: datetime, allow_stale: bool) -> dict[str, list[tuple[str, float, FXQuote]]]:
        adjacency: dict[str, list[tuple[str, float, FXQuote]]] = {}
        with self._lock:
            quotes = tuple(self._quotes.values())
        for q in quotes:
            if q.timestamp > as_of:
                continue
            age = as_of - q.timestamp
            if not allow_stale and age > self.max_age:
                continue
            base, term = q.base, q.quote
            adjacency.setdefault(base, []).append((term, q.rate, q))
            adjacency.setdefault(term, []).append((base, 1.0 / q.rate, q))
        for currency in adjacency:
            adjacency[currency].sort(key=lambda item: item[0])
        return adjacency

    def convert(self, amount: float, source: str, target: str, *,
                as_of: datetime | None = None, allow_stale: bool = False) -> FXConversion:
        if not isfinite(amount):
            raise ValueError("amount must be finite")
        src, dst = source.upper(), target.upper()
        if len(src) != 3 or len(dst) != 3:
            raise ValueError("currencies must have 3 characters")
        cutoff = as_of or self._now()
        if cutoff.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if src == dst:
            return FXConversion(amount, src, dst, amount, 1.0, (src,), (), 0.0)
        graph = self._edges(cutoff, allow_stale)
        queue: list[tuple[str, float, tuple[str, ...], tuple[FXQuote, ...]]] = [(src, 1.0, (src,), ())]
        visited = {src}
        head = 0
        while head < len(queue):
            currency, rate, path, quotes = queue[head]
            head += 1
            for neighbor, edge_rate, quote in graph.get(currency, []):
                if neighbor in visited:
                    continue
                next_rate = rate * edge_rate
                next_path = path + (neighbor,)
                next_quotes = quotes + (quote,)
                if neighbor == dst:
                    ages = [(cutoff - q.timestamp).total_seconds() for q in next_quotes]
                    return FXConversion(amount, src, dst, amount * next_rate, next_rate,
                                        next_path, tuple(q.timestamp.isoformat() for q in next_quotes),
                                        max(ages, default=0.0))
                visited.add(neighbor)
                queue.append((neighbor, next_rate, next_path, next_quotes))
        raise FXRateUnavailable(f"no {'fresh ' if not allow_stale else ''}FX path from {src} to {dst}")

    def quote_snapshot(self) -> tuple[FXQuote, ...]:
        with self._lock:
            return tuple(sorted(self._quotes.values(), key=lambda q: (q.base, q.quote)))

    def health(self) -> dict[str, int | float]:
        now = self._now()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        with self._lock:
            quotes = tuple(self._quotes.values())
        ages = [(now - q.timestamp).total_seconds() for q in quotes if q.timestamp <= now]
        stale = sum(age > self.max_age.total_seconds() for age in ages)
        return {"quotes": len(quotes), "stale_quotes": stale,
                "oldest_age_seconds": max(ages, default=0.0)}
