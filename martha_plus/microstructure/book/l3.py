"""Order-by-order limit order book with price-time priority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import BookLevelView, BookTop, Side
from .level import PriceLevel, QueueOrder


@dataclass(frozen=True, slots=True)
class BookDepth:
    bids: tuple[BookLevelView, ...]
    asks: tuple[BookLevelView, ...]
    ts_ns: int


class LimitOrderBook:
    """Deterministic L3 book; all order identifiers are unique within a book."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.strip().upper()
        if not self.symbol:
            raise ValueError("symbol must not be blank")
        self._bids: dict[int, PriceLevel] = {}
        self._asks: dict[int, PriceLevel] = {}
        self._orders: dict[str, QueueOrder] = {}
        self._sequence = 0
        self.last_update_ns = 0

    def __len__(self) -> int:
        return len(self._orders)

    def _levels(self, side: Side) -> dict[int, PriceLevel]:
        return self._bids if side is Side.BUY else self._asks

    def _ordered_prices(self, side: Side) -> list[int]:
        prices = self._levels(side).keys()
        return sorted(prices, reverse=(side is Side.BUY))

    def add(
        self,
        order_id: str,
        owner: str,
        side: Side,
        price_ticks: int,
        quantity: int,
        ts_ns: int,
        *,
        is_external: bool = True,
        priority_sequence: int | None = None,
    ) -> QueueOrder:
        if order_id in self._orders:
            raise ValueError(f"duplicate order id: {order_id}")
        if price_ticks <= 0 or quantity <= 0 or ts_ns < 0:
            raise ValueError("price, quantity, and timestamp must be valid")
        self._sequence = max(self._sequence + 1, priority_sequence or 0)
        order = QueueOrder(order_id, owner, side, price_ticks, quantity, self._sequence, is_external)
        levels = self._levels(side)
        level = levels.setdefault(price_ticks, PriceLevel(price_ticks))
        level.add(order)
        self._orders[order_id] = order
        self.last_update_ns = max(self.last_update_ns, ts_ns)
        return order

    def get(self, order_id: str) -> QueueOrder | None:
        return self._orders.get(order_id)

    def cancel(self, order_id: str, ts_ns: int) -> QueueOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        level = self._levels(order.side)[order.price_ticks]
        removed = level.remove(order_id)
        if len(level) == 0:
            del self._levels(order.side)[order.price_ticks]
        del self._orders[order_id]
        self.last_update_ns = max(self.last_update_ns, ts_ns)
        return removed

    def reduce(self, order_id: str, quantity: int, ts_ns: int) -> QueueOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        level = self._levels(order.side)[order.price_ticks]
        level.reduce(order_id, quantity)
        if order.quantity == 0:
            del self._orders[order_id]
            if len(level) == 0:
                del self._levels(order.side)[order.price_ticks]
        self.last_update_ns = max(self.last_update_ns, ts_ns)
        return order

    def amend(self, order_id: str, new_quantity: int, ts_ns: int, *, preserve_priority_on_decrease: bool = True) -> QueueOrder:
        """Amend quantity; size increases lose priority, decreases may retain it."""
        if new_quantity <= 0:
            raise ValueError("new_quantity must be positive")
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        old_quantity = order.quantity
        if new_quantity == old_quantity:
            return order
        if new_quantity < old_quantity and preserve_priority_on_decrease:
            self.reduce(order_id, old_quantity - new_quantity, ts_ns)
            return order
        side, price, owner, external = order.side, order.price_ticks, order.owner, order.is_external
        self.cancel(order_id, ts_ns)
        return self.add(order_id, owner, side, price, new_quantity, ts_ns, is_external=external)

    def best_order(self, side: Side) -> QueueOrder | None:
        prices = self._ordered_prices(side)
        if not prices:
            return None
        return self._levels(side)[prices[0]].first()

    def best_price(self, side: Side) -> int | None:
        order = self.best_order(side)
        return order.price_ticks if order else None

    def best_level(self, side: Side) -> PriceLevel | None:
        price = self.best_price(side)
        return self._levels(side).get(price) if price is not None else None

    def top(self, ts_ns: int | None = None) -> BookTop:
        bid = self.best_level(Side.BUY)
        ask = self.best_level(Side.SELL)
        bid_view = BookLevelView(bid.price_ticks, bid.total_quantity, len(bid)) if bid else None
        ask_view = BookLevelView(ask.price_ticks, ask.total_quantity, len(ask)) if ask else None
        return BookTop(self.symbol, self.last_update_ns if ts_ns is None else ts_ns, bid_view, ask_view)

    def depth(self, levels: int = 10, ts_ns: int | None = None) -> BookDepth:
        if levels <= 0:
            raise ValueError("levels must be positive")
        def convert(side: Side) -> tuple[BookLevelView, ...]:
            data = self._levels(side)
            result = []
            for price in self._ordered_prices(side)[:levels]:
                level = data[price]
                result.append(BookLevelView(price, level.total_quantity, len(level)))
            return tuple(result)
        return BookDepth(convert(Side.BUY), convert(Side.SELL), self.last_update_ns if ts_ns is None else ts_ns)

    def queue_ahead(self, order_id: str) -> tuple[int, int]:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        level = self._levels(order.side)[order.price_ticks]
        return level.quantity_ahead(order_id), level.order_count_ahead(order_id)

    def pop_best(self, side: Side, ts_ns: int) -> QueueOrder:
        level = self.best_level(side)
        if level is None:
            raise IndexError("book side is empty")
        order = level.pop_first()
        self._orders.pop(order.order_id)
        if len(level) == 0:
            del self._levels(side)[level.price_ticks]
        self.last_update_ns = max(self.last_update_ns, ts_ns)
        return order

    def iter_orders(self, side: Side | None = None) -> Iterable[QueueOrder]:
        sides = (Side.BUY, Side.SELL) if side is None else (side,)
        for current_side in sides:
            for price in self._ordered_prices(current_side):
                yield from self._levels(current_side)[price]

    def crossed(self) -> bool:
        bid = self.best_price(Side.BUY)
        ask = self.best_price(Side.SELL)
        return bid is not None and ask is not None and bid >= ask

    def validate(self, *, require_uncrossed: bool = False) -> None:
        for side in (Side.BUY, Side.SELL):
            for price, level in self._levels(side).items():
                if price != level.price_ticks:
                    raise AssertionError("price-level key mismatch")
                level.validate()
                for order in level:
                    if order.side is not side or order.price_ticks != price:
                        raise AssertionError("order stored on wrong side or level")
                    if self._orders.get(order.order_id) is not order:
                        raise AssertionError("global order index mismatch")
        if len(self._orders) != sum(len(level) for levels in (self._bids, self._asks) for level in levels.values()):
            raise AssertionError("global order count mismatch")
        if require_uncrossed and self.crossed():
            raise AssertionError("book is locked or crossed")

    def digest_rows(self) -> tuple[tuple[str, int, int, str], ...]:
        """Stable rows suitable for replay comparison and fixture hashes."""
        rows = []
        for side in (Side.BUY, Side.SELL):
            for order in self.iter_orders(side):
                rows.append((side.value, order.price_ticks, order.quantity, order.order_id))
        return tuple(rows)
