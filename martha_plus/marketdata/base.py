"""Market data provider contracts."""
from __future__ import annotations
from typing import Protocol, Sequence
from ..types import Bar, Quote

class MarketDataProvider(Protocol):
    def latest_quote(self, symbol: str) -> Quote: ...
    def bars(self, symbol: str, limit: int | None = None) -> Sequence[Bar]: ...

class InMemoryMarketData:
    """Simple injectable market-data provider useful for tests and integrations."""
    def __init__(self) -> None:
        self._quotes: dict[str, Quote] = {}
        self._bars: dict[str, list[Bar]] = {}

    def put_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote

    def put_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        key = symbol.strip().upper()
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        if any(b.symbol != key for b in ordered):
            raise ValueError("all bars must match symbol")
        if any(a.timestamp == b.timestamp for a, b in zip(ordered, ordered[1:])):
            raise ValueError("duplicate bar timestamps")
        self._bars[key] = list(ordered)

    def latest_quote(self, symbol: str) -> Quote:
        return self._quotes[symbol.strip().upper()]

    def bars(self, symbol: str, limit: int | None = None) -> Sequence[Bar]:
        values = self._bars.get(symbol.strip().upper(), [])
        return values[-limit:] if limit else tuple(values)
