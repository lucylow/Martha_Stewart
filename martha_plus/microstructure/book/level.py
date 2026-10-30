"""FIFO price-level implementation used by the simulated exchange."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(slots=True)
class QueueOrder:
    order_id: str
    owner: str
    side: object
    price_ticks: int
    quantity: int
    priority_sequence: int
    is_external: bool = True

    def __post_init__(self) -> None:
        if not self.order_id or not self.owner:
            raise ValueError("order_id and owner must not be blank")
        if self.price_ticks <= 0 or self.quantity <= 0:
            raise ValueError("price and quantity must be positive")
        if self.priority_sequence < 0:
            raise ValueError("priority_sequence must be non-negative")


class PriceLevel:
    """An ordered queue at one price; earliest accepted order is first."""

    def __init__(self, price_ticks: int) -> None:
        if price_ticks <= 0:
            raise ValueError("price_ticks must be positive")
        self.price_ticks = price_ticks
        self._orders: OrderedDict[str, QueueOrder] = OrderedDict()
        self.total_quantity = 0

    def __len__(self) -> int:
        return len(self._orders)

    def __iter__(self) -> Iterator[QueueOrder]:
        return iter(self._orders.values())

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def add(self, order: QueueOrder) -> None:
        if order.price_ticks != self.price_ticks:
            raise ValueError("order price does not match level")
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id at price level: {order.order_id}")
        self._orders[order.order_id] = order
        self.total_quantity += order.quantity

    def get(self, order_id: str) -> QueueOrder | None:
        return self._orders.get(order_id)

    def first(self) -> QueueOrder | None:
        return next(iter(self._orders.values()), None)

    def pop_first(self) -> QueueOrder:
        if not self._orders:
            raise IndexError("cannot pop an empty price level")
        order_id, order = self._orders.popitem(last=False)
        self.total_quantity -= order.quantity
        return order

    def reduce(self, order_id: str, quantity: int) -> int:
        """Reduce a resting order and return its remaining quantity."""
        if quantity <= 0:
            raise ValueError("reduction must be positive")
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        if quantity > order.quantity:
            raise ValueError("reduction exceeds leaves quantity")
        order.quantity -= quantity
        self.total_quantity -= quantity
        if order.quantity == 0:
            del self._orders[order_id]
            return 0
        return order.quantity

    def remove(self, order_id: str) -> QueueOrder:
        try:
            order = self._orders.pop(order_id)
        except KeyError:
            raise KeyError(order_id) from None
        self.total_quantity -= order.quantity
        return order

    def quantity_ahead(self, order_id: str) -> int:
        total = 0
        for candidate in self._orders.values():
            if candidate.order_id == order_id:
                return total
            total += candidate.quantity
        raise KeyError(order_id)

    def order_count_ahead(self, order_id: str) -> int:
        count = 0
        for candidate in self._orders.values():
            if candidate.order_id == order_id:
                return count
            count += 1
        raise KeyError(order_id)

    def snapshot(self) -> tuple[tuple[str, int, str], ...]:
        return tuple((order.order_id, order.quantity, order.owner) for order in self._orders.values())

    def validate(self) -> None:
        if self.total_quantity < 0:
            raise AssertionError("negative aggregate quantity")
        actual = sum(order.quantity for order in self._orders.values())
        if actual != self.total_quantity:
            raise AssertionError(f"level quantity mismatch: {actual} != {self.total_quantity}")
        if any(order.quantity <= 0 for order in self._orders.values()):
            raise AssertionError("non-positive order in queue")
