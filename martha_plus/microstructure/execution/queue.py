"""Conservative queue-position models for resting paper orders."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from math import exp
from typing import Iterable

from ..types import Side


class QueueEventType(str, Enum):
    TRADE = "trade"
    CANCEL_AHEAD = "cancel_ahead"
    CANCEL_BEHIND = "cancel_behind"
    ADD_AHEAD = "add_ahead"
    ADD_BEHIND = "add_behind"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class QueueEvent:
    ts_ns: int
    kind: QueueEventType
    quantity: int
    order_id: str = ""

    def __post_init__(self) -> None:
        if self.ts_ns < 0 or self.quantity <= 0:
            raise ValueError("queue event timestamp and quantity must be positive")


@dataclass(frozen=True, slots=True)
class QueueEstimate:
    order_id: str
    quantity_ahead: int
    own_quantity: int
    quantity_behind: int
    expected_fill_quantity: Decimal
    fill_probability: Decimal
    expected_wait_ns: Decimal | None
    updated_at_ns: int


class QueuePosition:
    """Track one simulated order's location inside a single price queue.

    The default behavior is deliberately pessimistic: unclassified cancels are
    assumed to happen behind the simulated order, so they do not improve its
    rank.  Use explicit `cancel_ahead` only when historical order-level data
    proves the cancellation belonged to a queue ahead of the hypothetical order.
    """

    def __init__(self, order_id: str, quantity_ahead: int, own_quantity: int,
                 quantity_behind: int = 0, *, timestamp_ns: int = 0) -> None:
        if not order_id.strip() or min(quantity_ahead, quantity_behind, timestamp_ns) < 0 or own_quantity <= 0:
            raise ValueError("invalid initial queue state")
        self.order_id = order_id
        self.quantity_ahead = quantity_ahead
        self.own_quantity = own_quantity
        self.quantity_behind = quantity_behind
        self.updated_at_ns = timestamp_ns
        self.traded_ahead = 0
        self.filled_quantity = 0
        self.cancelled_quantity = 0
        self._initial_own_quantity = own_quantity
        self.events = 0

    @property
    def remaining_quantity(self) -> int:
        return self.own_quantity

    @property
    def is_filled(self) -> bool:
        return self.own_quantity == 0

    def _validate(self) -> None:
        if min(self.quantity_ahead, self.own_quantity, self.quantity_behind) < 0:
            raise AssertionError("queue quantities cannot be negative")
        if self.filled_quantity + self.cancelled_quantity + self.own_quantity != self._initial_own_quantity:
            raise AssertionError("queue own-quantity conservation failed")

    def apply(self, event: QueueEvent) -> int:
        """Apply an event and return simulated fill quantity."""
        if event.ts_ns < self.updated_at_ns:
            raise ValueError("queue events must be timestamp ordered")
        if self.is_filled:
            return 0
        fill = 0
        if event.kind is QueueEventType.TRADE:
            ahead_consumed = min(self.quantity_ahead, event.quantity)
            self.quantity_ahead -= ahead_consumed
            self.traded_ahead += ahead_consumed
            remaining_trade = event.quantity - ahead_consumed
            fill = min(self.own_quantity, remaining_trade)
            self.own_quantity -= fill
            self.filled_quantity += fill
            # Remaining trade is assumed to consume queue behind this order.
            self.quantity_behind = max(0, self.quantity_behind - max(0, remaining_trade - fill))
        elif event.kind is QueueEventType.CANCEL_AHEAD:
            self.quantity_ahead = max(0, self.quantity_ahead - event.quantity)
        elif event.kind is QueueEventType.CANCEL_BEHIND:
            self.quantity_behind = max(0, self.quantity_behind - event.quantity)
        elif event.kind is QueueEventType.ADD_AHEAD:
            self.quantity_ahead += event.quantity
        elif event.kind is QueueEventType.ADD_BEHIND:
            self.quantity_behind += event.quantity
        elif event.kind is QueueEventType.REPLACE:
            # A replacement has no priority in most price-time venues.
            self.quantity_ahead += self.quantity_behind
            self.quantity_behind = 0
        self.updated_at_ns = event.ts_ns
        self.events += 1
        self._validate()
        return fill

    def cancel(self, quantity: int, ts_ns: int) -> int:
        if quantity <= 0 or ts_ns < self.updated_at_ns:
            raise ValueError("invalid cancel")
        cancelled = min(self.own_quantity, quantity)
        self.own_quantity -= cancelled
        self.cancelled_quantity += cancelled
        self.updated_at_ns = ts_ns
        self._validate()
        return cancelled

    def resize(self, new_quantity: int, ts_ns: int, *, preserve_priority: bool = False) -> None:
        if new_quantity <= 0 or ts_ns < self.updated_at_ns:
            raise ValueError("invalid resize")
        old_quantity = self.own_quantity
        if new_quantity > old_quantity and not preserve_priority:
            self.quantity_ahead += self.quantity_behind
            self.quantity_behind = 0
        elif new_quantity < old_quantity:
            self.cancelled_quantity += old_quantity - new_quantity
        self.own_quantity = new_quantity
        self._initial_own_quantity = self.filled_quantity + self.cancelled_quantity + new_quantity
        self.updated_at_ns = ts_ns

    def estimate(self, *, expected_trade_volume: int, horizon_ns: int,
                 cancellation_ahead_rate: float = 0.0) -> QueueEstimate:
        if expected_trade_volume < 0 or horizon_ns < 0 or not 0 <= cancellation_ahead_rate <= 1:
            raise ValueError("invalid queue estimate parameters")
        expected_cancel = Decimal(str(cancellation_ahead_rate)) * Decimal(self.quantity_ahead)
        effective_ahead = max(Decimal(0), Decimal(self.quantity_ahead) - expected_cancel)
        flow = Decimal(expected_trade_volume)
        expected_fill = min(Decimal(self.own_quantity), max(Decimal(0), flow - effective_ahead))
        if expected_trade_volume == 0:
            probability = Decimal(0)
        elif effective_ahead <= 0:
            probability = Decimal(1)
        else:
            rate = float(flow / max(Decimal(1), effective_ahead + self.own_quantity))
            probability = Decimal(str(1 - exp(-max(0.0, rate))))
        wait = None
        if expected_trade_volume > 0 and self.quantity_ahead + self.own_quantity > 0:
            wait = Decimal(horizon_ns) * Decimal(self.quantity_ahead + self.own_quantity) / Decimal(expected_trade_volume)
        return QueueEstimate(self.order_id, self.quantity_ahead, self.own_quantity,
                             self.quantity_behind, expected_fill, probability, wait,
                             self.updated_at_ns)


@dataclass(frozen=True, slots=True)
class QueueFill:
    ts_ns: int
    quantity: int
    price_ticks: int
    maker: bool
    confidence: Decimal


class ProbabilisticQueueModel:
    """Estimate fills from aggregate depth events with a tunable cancel split."""

    def __init__(self, *, cancellation_ahead_fraction: Decimal = Decimal("0.0"),
                 trade_participation: Decimal = Decimal("1.0")) -> None:
        if not 0 <= cancellation_ahead_fraction <= 1:
            raise ValueError("cancellation_ahead_fraction must lie in [0, 1]")
        if not 0 <= trade_participation <= 1:
            raise ValueError("trade_participation must lie in [0, 1]")
        self.cancellation_ahead_fraction = cancellation_ahead_fraction
        self.trade_participation = trade_participation

    def expected_fill(self, queue: QueuePosition, *, traded_quantity: int,
                      cancelled_quantity: int = 0) -> Decimal:
        if min(traded_quantity, cancelled_quantity) < 0:
            raise ValueError("flow quantities cannot be negative")
        ahead_cancel = Decimal(cancelled_quantity) * self.cancellation_ahead_fraction
        ahead = max(Decimal(0), Decimal(queue.quantity_ahead) - ahead_cancel)
        eligible_trade = Decimal(traded_quantity) * self.trade_participation
        return min(Decimal(queue.own_quantity), max(Decimal(0), eligible_trade - ahead))

    def apply_aggregate_update(self, queue: QueuePosition, ts_ns: int, *,
                               traded_quantity: int, cancelled_quantity: int = 0) -> QueueFill | None:
        expected = self.expected_fill(queue, traded_quantity=traded_quantity,
                                      cancelled_quantity=cancelled_quantity)
        ahead_cancel = int(Decimal(cancelled_quantity) * self.cancellation_ahead_fraction)
        if ahead_cancel:
            queue.apply(QueueEvent(ts_ns, QueueEventType.CANCEL_AHEAD, ahead_cancel))
        if traded_quantity <= 0:
            return None
        filled = queue.apply(QueueEvent(ts_ns, QueueEventType.TRADE,
                                        max(1, int(Decimal(traded_quantity) * self.trade_participation))))
        if expected <= 0 and filled <= 0:
            return None
        if not filled:
            return None
        return QueueFill(ts_ns, filled, 0, True, Decimal("0.5"))


class QueueRiskGuard:
    """Reject queue estimates that rely on optimistic data assumptions."""

    def __init__(self, *, max_queue_age_ns: int, min_confidence: Decimal = Decimal("0.5")) -> None:
        if max_queue_age_ns < 0 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid queue guard")
        self.max_queue_age_ns = max_queue_age_ns
        self.min_confidence = min_confidence

    def check(self, estimate: QueueEstimate, now_ns: int, confidence: Decimal) -> tuple[str, ...]:
        reasons = []
        if now_ns < estimate.updated_at_ns:
            reasons.append("clock_regression")
        elif now_ns - estimate.updated_at_ns > self.max_queue_age_ns:
            reasons.append("stale_queue_estimate")
        if confidence < self.min_confidence:
            reasons.append("low_queue_confidence")
        if estimate.fill_probability < 0 or estimate.fill_probability > 1:
            reasons.append("invalid_fill_probability")
        return tuple(reasons)
