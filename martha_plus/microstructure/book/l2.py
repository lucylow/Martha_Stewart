"""Aggregated depth book and quote-derived microstructure features."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from ..types import BookLevelView, BookTop, Side


@dataclass(frozen=True, slots=True)
class DepthLevel:
    price_ticks: int
    quantity: int

    def __post_init__(self) -> None:
        if self.price_ticks <= 0 or self.quantity <= 0:
            raise ValueError("depth prices and quantities must be positive")


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    symbol: str
    ts_ns: int
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        if not self.symbol or self.ts_ns < 0:
            raise ValueError("invalid symbol or timestamp")
        if any(a.price_ticks <= b.price_ticks for a, b in zip(self.bids, self.bids[1:])):
            raise ValueError("bids must be strictly descending")
        if any(a.price_ticks >= b.price_ticks for a, b in zip(self.asks, self.asks[1:])):
            raise ValueError("asks must be strictly ascending")


class AggregatedBook:
    """Level-2 book; deltas update a price's aggregate displayed quantity."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.strip().upper()
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.ts_ns = 0
        self.sequence = -1

    def apply_snapshot(self, snapshot: DepthSnapshot) -> None:
        if snapshot.symbol != self.symbol:
            raise ValueError("snapshot symbol mismatch")
        self.bids = {level.price_ticks: level.quantity for level in snapshot.bids}
        self.asks = {level.price_ticks: level.quantity for level in snapshot.asks}
        self.ts_ns = snapshot.ts_ns
        self.sequence = -1
        self.validate()

    def update(self, side: Side, price_ticks: int, quantity: int, ts_ns: int, sequence: int) -> None:
        if price_ticks <= 0 or quantity < 0 or ts_ns < self.ts_ns:
            raise ValueError("invalid price, quantity, or out-of-order timestamp")
        if sequence <= self.sequence:
            raise ValueError("sequence must increase strictly")
        levels = self.bids if side is Side.BUY else self.asks
        if quantity == 0:
            levels.pop(price_ticks, None)
        else:
            levels[price_ticks] = quantity
        self.ts_ns = ts_ns
        self.sequence = sequence
        self.validate()

    def _view(self, side: Side, count: int) -> tuple[BookLevelView, ...]:
        data = self.bids if side is Side.BUY else self.asks
        prices = sorted(data, reverse=(side is Side.BUY))[:count]
        return tuple(BookLevelView(price, data[price], 1) for price in prices)

    def top(self) -> BookTop:
        bid = self._view(Side.BUY, 1)
        ask = self._view(Side.SELL, 1)
        return BookTop(self.symbol, self.ts_ns, bid[0] if bid else None, ask[0] if ask else None)

    def depth(self, count: int = 10) -> DepthSnapshot:
        if count <= 0:
            raise ValueError("count must be positive")
        bids = tuple(DepthLevel(x.price_ticks, x.quantity) for x in self._view(Side.BUY, count))
        asks = tuple(DepthLevel(x.price_ticks, x.quantity) for x in self._view(Side.SELL, count))
        return DepthSnapshot(self.symbol, self.ts_ns, bids, asks)

    def imbalance(self, levels: int = 1) -> Decimal | None:
        depth = self.depth(levels)
        bid_qty = sum(x.quantity for x in depth.bids)
        ask_qty = sum(x.quantity for x in depth.asks)
        denominator = bid_qty + ask_qty
        if denominator == 0:
            return None
        return Decimal(bid_qty - ask_qty) / Decimal(denominator)

    def microprice_ticks(self) -> Decimal | None:
        top = self.top()
        if top.bid is None or top.ask is None:
            return None
        bid_q, ask_q = Decimal(top.bid.quantity), Decimal(top.ask.quantity)
        total = bid_q + ask_q
        if total == 0:
            return None
        return (Decimal(top.ask.price_ticks) * bid_q + Decimal(top.bid.price_ticks) * ask_q) / total

    def validate(self) -> None:
        if any(price <= 0 or qty <= 0 for price, qty in (*self.bids.items(), *self.asks.items())):
            raise AssertionError("invalid aggregated level")
        bid = max(self.bids, default=None)
        ask = min(self.asks, default=None)
        if bid is not None and ask is not None and bid >= ask:
            raise ValueError("snapshot/delta creates a locked or crossed aggregated book")
