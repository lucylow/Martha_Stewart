"""Market-data source contracts, stale-data checks, and deterministic failover."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Protocol, Sequence
from ..types import Bar, Quote


class MarketDataUnavailable(RuntimeError):
    pass


class MarketDataProvider(Protocol):
    name: str
    def get_bars(self, symbol: str, start: datetime | None = None,
                 end: datetime | None = None) -> Sequence[Bar]: ...
    def get_quote(self, symbol: str) -> Quote: ...


@dataclass(frozen=True, slots=True)
class SourceAttempt:
    provider: str
    operation: str
    symbol: str
    ok: bool
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class DataQualityPolicy:
    max_quote_age: timedelta = timedelta(seconds=10)
    max_bar_age: timedelta = timedelta(days=3)
    require_sorted_bars: bool = True
    reject_duplicate_timestamps: bool = True
    allow_empty_bars: bool = False

    def __post_init__(self) -> None:
        if self.max_quote_age.total_seconds() <= 0 or self.max_bar_age.total_seconds() <= 0:
            raise ValueError("maximum ages must be positive")


class FailoverMarketDataProvider:
    """Consult providers in order; validate output before returning any data.

    Provider errors are retained for observability. This adapter has no network
    client by itself and cannot silently route orders or execute trades.
    """
    def __init__(self, providers: Sequence[MarketDataProvider], *,
                 policy: DataQualityPolicy | None = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        names = [getattr(p, "name", p.__class__.__name__) for p in providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        self.providers = tuple(providers)
        self.policy = policy or DataQualityPolicy()
        self._now = now
        self._lock = RLock()
        self._history: list[SourceAttempt] = []

    @property
    def history(self) -> tuple[SourceAttempt, ...]:
        with self._lock:
            return tuple(self._history)

    def get_quote(self, symbol: str) -> Quote:
        errors: list[str] = []
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            started = datetime.now(timezone.utc)
            try:
                quote = provider.get_quote(symbol.strip().upper())
                if quote.symbol != symbol.strip().upper():
                    raise ValueError("provider returned a quote for the wrong symbol")
                age = self._now() - quote.timestamp
                if age < timedelta(seconds=-1):
                    raise ValueError("quote timestamp is in the future")
                if age > self.policy.max_quote_age:
                    raise ValueError(f"quote is stale ({age.total_seconds():.3f}s old)")
                self._record(SourceAttempt(name, "quote", symbol, True,
                                           elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds()))
                return quote
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(message)
                self._record(SourceAttempt(name, "quote", symbol, False, message,
                                           (datetime.now(timezone.utc) - started).total_seconds()))
        raise MarketDataUnavailable("all quote providers failed: " + " | ".join(errors))

    def get_bars(self, symbol: str, start: datetime | None = None,
                 end: datetime | None = None) -> list[Bar]:
        errors: list[str] = []
        normalized = symbol.strip().upper()
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            started = datetime.now(timezone.utc)
            try:
                bars = list(provider.get_bars(normalized, start, end))
                if not bars and not self.policy.allow_empty_bars:
                    raise ValueError("provider returned no bars")
                if any(bar.symbol != normalized for bar in bars):
                    raise ValueError("provider returned bars for another symbol")
                times = [bar.timestamp for bar in bars]
                if self.policy.require_sorted_bars and times != sorted(times):
                    raise ValueError("bars are not sorted chronologically")
                if self.policy.reject_duplicate_timestamps and len(set(times)) != len(times):
                    raise ValueError("duplicate bar timestamps")
                if end is not None and any(t > end for t in times):
                    raise ValueError("provider returned bars after requested end")
                if start is not None and any(t < start for t in times):
                    raise ValueError("provider returned bars before requested start")
                now = self._now()
                if bars and now - bars[-1].timestamp > self.policy.max_bar_age:
                    raise ValueError("latest bar exceeds maximum allowed age")
                self._record(SourceAttempt(name, "bars", normalized, True,
                                           elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds()))
                return bars
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(message)
                self._record(SourceAttempt(name, "bars", normalized, False, message,
                                           (datetime.now(timezone.utc) - started).total_seconds()))
        raise MarketDataUnavailable("all bar providers failed: " + " | ".join(errors))

    def _record(self, attempt: SourceAttempt) -> None:
        with self._lock:
            self._history.append(attempt)
            self._history = self._history[-5000:]

    def availability(self) -> dict[str, object]:
        with self._lock:
            recent = self._history[-100:]
        by_provider: dict[str, dict[str, int]] = {}
        for record in recent:
            counts = by_provider.setdefault(record.provider, {"success": 0, "failure": 0})
            counts["success" if record.ok else "failure"] += 1
        return {"provider_count": len(self.providers), "providers": by_provider,
                "recent_attempts": len(recent), "recent_failures": sum(not item.ok for item in recent)}


class StaticMarketDataProvider:
    """Deterministic adapter useful for replay, fixtures, and local integration."""
    def __init__(self, bars: Sequence[Bar] = (), quotes: Sequence[Quote] = (), name: str = "static") -> None:
        self.name = name
        self._bars: dict[str, list[Bar]] = {}
        for bar in bars:
            self._bars.setdefault(bar.symbol, []).append(bar)
        for symbol in self._bars:
            self._bars[symbol].sort(key=lambda b: b.timestamp)
        self._quotes = {quote.symbol: quote for quote in quotes}

    def get_bars(self, symbol: str, start: datetime | None = None,
                 end: datetime | None = None) -> list[Bar]:
        rows = list(self._bars.get(symbol.strip().upper(), ()))
        if start is not None:
            rows = [row for row in rows if row.timestamp >= start]
        if end is not None:
            rows = [row for row in rows if row.timestamp <= end]
        return rows

    def get_quote(self, symbol: str) -> Quote:
        key = symbol.strip().upper()
        if key not in self._quotes:
            raise KeyError(f"no quote for {key}")
        return self._quotes[key]

    def replace_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote

    def append_bar(self, bar: Bar) -> None:
        series = self._bars.setdefault(bar.symbol, [])
        if series and bar.timestamp <= series[-1].timestamp:
            raise ValueError("appended bar must be later than existing history")
        series.append(bar)


class CachedMarketDataProvider:
    """TTL cache wrapper with per-key locking and no stale-on-error behavior by default."""
    def __init__(self, provider: MarketDataProvider, *, quote_ttl: float = 2.0,
                 bar_ttl: float = 30.0, allow_stale_on_error: bool = False,
                 monotonic_clock: Callable[[], float] | None = None) -> None:
        from time import monotonic
        if quote_ttl <= 0 or bar_ttl <= 0:
            raise ValueError("cache TTLs must be positive")
        self.provider = provider
        self.name = f"cache({getattr(provider, 'name', 'provider')})"
        self.quote_ttl = quote_ttl
        self.bar_ttl = bar_ttl
        self.allow_stale_on_error = allow_stale_on_error
        self._clock = monotonic_clock or monotonic
        self._lock = RLock()
        self._quotes: dict[str, tuple[float, Quote]] = {}
        self._bars: dict[tuple[str, str | None, str | None], tuple[float, list[Bar]]] = {}
        self.hits = 0
        self.misses = 0

    def get_quote(self, symbol: str) -> Quote:
        key = symbol.strip().upper()
        now = self._clock()
        with self._lock:
            item = self._quotes.get(key)
            if item and now - item[0] < self.quote_ttl:
                self.hits += 1
                return item[1]
            self.misses += 1
            prior = item[1] if item else None
        try:
            quote = self.provider.get_quote(key)
        except Exception:
            if self.allow_stale_on_error and prior is not None:
                return prior
            raise
        with self._lock:
            self._quotes[key] = (self._clock(), quote)
        return quote

    def get_bars(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> list[Bar]:
        key = (symbol.strip().upper(), start.isoformat() if start else None, end.isoformat() if end else None)
        now = self._clock()
        with self._lock:
            item = self._bars.get(key)
            if item and now - item[0] < self.bar_ttl:
                self.hits += 1
                return list(item[1])
            self.misses += 1
            prior = list(item[1]) if item else None
        try:
            bars = list(self.provider.get_bars(key[0], start, end))
        except Exception:
            if self.allow_stale_on_error and prior is not None:
                return prior
            raise
        with self._lock:
            self._bars[key] = (self._clock(), list(bars))
        return bars

    def clear(self) -> None:
        with self._lock:
            self._quotes.clear()
            self._bars.clear()

    def cache_info(self) -> dict[str, int | bool]:
        with self._lock:
            return {"quote_entries": len(self._quotes), "bar_entries": len(self._bars),
                    "hits": self.hits, "misses": self.misses, "allow_stale_on_error": self.allow_stale_on_error}
