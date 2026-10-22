"""Thread-safe quote cache with age and spread validation."""
from __future__ import annotations
from datetime import datetime, timezone
from threading import RLock
from ..errors import DataValidationError
from ..types import Quote

class QuoteCache:
    def __init__(self, max_age_seconds: float = 5.0, max_spread_bps: float = 250.0):
        if max_age_seconds <= 0 or max_spread_bps <= 0:
            raise ValueError("max quote age and spread must be positive")
        self.max_age_seconds = max_age_seconds
        self.max_spread_bps = max_spread_bps
        self._quotes: dict[str, Quote] = {}
        self._lock = RLock()

    def update(self, quote: Quote) -> None:
        with self._lock:
            prior = self._quotes.get(quote.symbol)
            if prior and quote.timestamp < prior.timestamp:
                raise DataValidationError("out-of-order quote rejected")
            if quote.spread_bps > self.max_spread_bps:
                raise DataValidationError("quote spread exceeds configured limit")
            self._quotes[quote.symbol] = quote

    def get(self, symbol: str, now: datetime | None = None) -> Quote:
        key = symbol.strip().upper()
        with self._lock:
            quote = self._quotes.get(key)
        if quote is None:
            raise KeyError(key)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (current - quote.timestamp.astimezone(timezone.utc)).total_seconds()
        if age < -1.0:
            raise DataValidationError("quote timestamp is too far in the future")
        if age > self.max_age_seconds:
            raise DataValidationError(f"stale quote: age={age:.3f}s")
        return quote

    def snapshot(self) -> dict[str, Quote]:
        with self._lock:
            return dict(self._quotes)
