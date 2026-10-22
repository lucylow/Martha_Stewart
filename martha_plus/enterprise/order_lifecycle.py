"""Validated order lifecycle state machine and append-only transition journal."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Callable, Mapping


class OrderState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderEvent(str, Enum):
    VALIDATE = "validate"
    REJECT = "reject"
    ACCEPT = "accept"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    REQUEST_CANCEL = "request_cancel"
    CANCEL = "cancel"
    EXPIRE = "expire"
    FAIL = "fail"


TRANSITIONS: dict[OrderState, dict[OrderEvent, OrderState]] = {
    OrderState.CREATED: {
        OrderEvent.VALIDATE: OrderState.VALIDATED,
        OrderEvent.REJECT: OrderState.REJECTED,
        OrderEvent.FAIL: OrderState.FAILED,
    },
    OrderState.VALIDATED: {
        OrderEvent.ACCEPT: OrderState.ACCEPTED,
        OrderEvent.REJECT: OrderState.REJECTED,
        OrderEvent.FAIL: OrderState.FAILED,
        OrderEvent.EXPIRE: OrderState.EXPIRED,
    },
    OrderState.ACCEPTED: {
        OrderEvent.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
        OrderEvent.FILL: OrderState.FILLED,
        OrderEvent.REQUEST_CANCEL: OrderState.CANCEL_PENDING,
        OrderEvent.EXPIRE: OrderState.EXPIRED,
        OrderEvent.FAIL: OrderState.FAILED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderEvent.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
        OrderEvent.FILL: OrderState.FILLED,
        OrderEvent.REQUEST_CANCEL: OrderState.CANCEL_PENDING,
        OrderEvent.FAIL: OrderState.FAILED,
    },
    OrderState.CANCEL_PENDING: {
        OrderEvent.CANCEL: OrderState.CANCELLED,
        OrderEvent.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
        OrderEvent.FILL: OrderState.FILLED,
        OrderEvent.FAIL: OrderState.FAILED,
    },
}

TERMINAL_STATES = frozenset({OrderState.REJECTED, OrderState.FILLED, OrderState.CANCELLED,
                             OrderState.EXPIRED, OrderState.FAILED})


@dataclass(frozen=True, slots=True)
class OrderTransition:
    order_id: str
    sequence: int
    from_state: OrderState
    event: OrderEvent
    to_state: OrderState
    occurred_at: str
    actor: str
    correlation_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRecord:
    order_id: str
    state: OrderState = OrderState.CREATED
    version: int = 0
    filled_quantity: float = 0.0
    requested_quantity: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("order_id is required")
        if self.requested_quantity < 0 or self.filled_quantity < 0:
            raise ValueError("quantities cannot be negative")
        if self.filled_quantity > self.requested_quantity > 0:
            raise ValueError("filled quantity cannot exceed requested quantity")


class InvalidTransition(ValueError):
    pass


class OrderStateMachine:
    """Thread-safe state machine; transition journal is immutable to callers."""
    def __init__(self, *, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._clock = clock
        self._orders: dict[str, OrderRecord] = {}
        self._journal: list[OrderTransition] = []
        self._lock = RLock()

    @property
    def journal(self) -> tuple[OrderTransition, ...]:
        with self._lock:
            return tuple(self._journal)

    def create(self, order_id: str, *, requested_quantity: float = 0.0,
               metadata: Mapping[str, str] | None = None) -> OrderRecord:
        record = OrderRecord(order_id, requested_quantity=requested_quantity,
                             metadata=dict(metadata or {}))
        with self._lock:
            if order_id in self._orders:
                raise ValueError(f"order already exists: {order_id}")
            self._orders[order_id] = record
            return self._copy_record(record)

    @staticmethod
    def _copy_record(record: OrderRecord) -> OrderRecord:
        return OrderRecord(record.order_id, record.state, record.version,
                           record.filled_quantity, record.requested_quantity,
                           dict(record.metadata))

    def get(self, order_id: str) -> OrderRecord:
        with self._lock:
            try:
                return self._copy_record(self._orders[order_id])
            except KeyError as exc:
                raise KeyError(f"unknown order: {order_id}") from exc

    def transition(self, order_id: str, event: OrderEvent, *, actor: str,
                   expected_version: int | None = None,
                   fill_quantity: float | None = None,
                   correlation_id: str | None = None,
                   metadata: Mapping[str, str] | None = None) -> OrderRecord:
        if not actor.strip():
            raise ValueError("actor is required")
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        with self._lock:
            if order_id not in self._orders:
                raise KeyError(f"unknown order: {order_id}")
            rec = self._orders[order_id]
            if expected_version is not None and rec.version != expected_version:
                raise RuntimeError(f"stale order version: expected {expected_version}, got {rec.version}")
            to_state = TRANSITIONS.get(rec.state, {}).get(event)
            if to_state is None:
                raise InvalidTransition(f"cannot apply {event.value} to {rec.state.value}")
            new_filled = rec.filled_quantity
            if event in (OrderEvent.PARTIAL_FILL, OrderEvent.FILL):
                if fill_quantity is None or fill_quantity <= 0:
                    raise ValueError("positive fill_quantity is required for fill events")
                new_filled += fill_quantity
                if rec.requested_quantity and new_filled > rec.requested_quantity + 1e-9:
                    raise ValueError("fill would exceed requested quantity")
                if event is OrderEvent.FILL and rec.requested_quantity and abs(new_filled - rec.requested_quantity) > 1e-9:
                    raise ValueError("fill event must complete the requested quantity")
                if event is OrderEvent.PARTIAL_FILL and rec.requested_quantity and new_filled >= rec.requested_quantity - 1e-9:
                    raise ValueError("use fill event when quantity completes order")
            elif fill_quantity is not None:
                raise ValueError("fill_quantity is valid only for fill events")
            before = rec.state
            seq = len(self._journal) + 1
            transition_metadata = dict(metadata or {})
            if fill_quantity is not None:
                transition_metadata["fill_quantity"] = repr(float(fill_quantity))
            transition = OrderTransition(order_id, seq, before, event, to_state,
                                         now.isoformat(), actor, correlation_id,
                                         transition_metadata)
            rec.state = to_state
            rec.version += 1
            rec.filled_quantity = new_filled
            self._journal.append(transition)
            return self._copy_record(rec)

    def replay(self, order_id: str) -> OrderRecord:
        """Rebuild state from journal and detect sequence/transition corruption."""
        with self._lock:
            events = [e for e in self._journal if e.order_id == order_id]
            record = self._orders.get(order_id)
            if record is None:
                raise KeyError(order_id)
            # The journal starts at CREATED; quantity is replayed from event metadata
            # only as a lifecycle check. Exact filled quantity remains in the live record.
            state = OrderState.CREATED
            expected_version = 0
            replayed_filled = 0.0
            for transition in events:
                if transition.sequence <= 0 or transition.from_state is not state:
                    raise RuntimeError("transition journal is inconsistent")
                target = TRANSITIONS.get(state, {}).get(transition.event)
                if target is not transition.to_state:
                    raise RuntimeError("transition target does not match state machine")
                if transition.event in (OrderEvent.PARTIAL_FILL, OrderEvent.FILL):
                    try:
                        replayed_filled += float(transition.metadata["fill_quantity"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError("fill transition missing valid quantity metadata") from exc
                state = target
                expected_version += 1
            if state is not record.state or expected_version != record.version:
                raise RuntimeError("journal state differs from stored order state")
            if abs(replayed_filled - record.filled_quantity) > 1e-9:
                raise RuntimeError("journal fills differ from stored filled quantity")
            return self._copy_record(record)

    def open_orders(self) -> tuple[OrderRecord, ...]:
        with self._lock:
            return tuple(self._copy_record(r) for r in self._orders.values()
                         if r.state not in TERMINAL_STATES)

    def counts(self) -> dict[str, int]:
        with self._lock:
            result = {state.value: 0 for state in OrderState}
            for record in self._orders.values():
                result[record.state.value] += 1
            return result
