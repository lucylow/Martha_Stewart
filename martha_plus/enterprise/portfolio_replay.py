"""Event-sourced portfolio projection and deterministic state replay utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import json


class PortfolioEventType(str, Enum):
    OPENING_POSITION = "opening_position"
    FILL = "fill"
    CASH_ADJUSTMENT = "cash_adjustment"
    MARK = "mark"
    SPLIT = "split"
    DIVIDEND = "dividend"
    FEE = "fee"
    TRANSFER = "transfer"


@dataclass(frozen=True, slots=True)
class PortfolioEvent:
    event_id: str
    sequence: int
    occurred_at: datetime
    event_type: PortfolioEventType
    account: str
    symbol: str | None
    quantity_delta: float = 0.0
    price: float | None = None
    cash_delta: float = 0.0
    currency: str = "USD"
    reference: str = ""
    metadata: Mapping[str, str | float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip() or self.sequence < 1 or not self.account.strip():
            raise ValueError("event_id, positive sequence, and account are required")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if not isfinite(self.quantity_delta) or not isfinite(self.cash_delta):
            raise ValueError("event deltas must be finite")
        if self.price is not None and (not isfinite(self.price) or self.price < 0):
            raise ValueError("price must be finite and non-negative")
        if len(self.currency) != 3:
            raise ValueError("currency must have three characters")

    @property
    def digest(self) -> str:
        data = {"event_id": self.event_id, "sequence": self.sequence,
            "occurred_at": self.occurred_at.isoformat(), "event_type": self.event_type.value,
            "account": self.account, "symbol": self.symbol, "quantity_delta": self.quantity_delta,
            "price": self.price, "cash_delta": self.cash_delta, "currency": self.currency,
            "reference": self.reference, "metadata": dict(self.metadata)}
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PositionProjection:
    account: str
    symbol: str
    quantity: float
    average_cost: float
    market_price: float | None
    market_value: float | None
    unrealized_pnl: float | None
    currency: str
    last_event_sequence: int


@dataclass(frozen=True, slots=True)
class PortfolioProjection:
    account: str
    cash_by_currency: Mapping[str, float]
    positions: tuple[PositionProjection, ...]
    event_count: int
    last_sequence: int
    last_event_at: str | None
    projection_digest: str

    @property
    def net_market_value(self) -> float:
        known_values = [p.market_value for p in self.positions if p.market_value is not None]
        return sum(known_values) + sum(self.cash_by_currency.values())


class PortfolioReplayError(RuntimeError):
    pass


class PortfolioEventStore:
    """Append-only account event log with contiguous sequence validation."""
    def __init__(self) -> None:
        self._events: dict[str, list[PortfolioEvent]] = {}
        self._by_id: dict[str, PortfolioEvent] = {}
        self._lock = RLock()

    def append(self, event: PortfolioEvent) -> None:
        with self._lock:
            existing = self._by_id.get(event.event_id)
            if existing:
                if existing.digest != event.digest:
                    raise PortfolioReplayError("event id reused with different content")
                return
            events = self._events.setdefault(event.account, [])
            expected = events[-1].sequence + 1 if events else 1
            if event.sequence != expected:
                raise PortfolioReplayError(f"expected sequence {expected}, got {event.sequence}")
            if events and event.occurred_at < events[-1].occurred_at:
                raise PortfolioReplayError("event timestamps must not move backward")
            events.append(event)
            self._by_id[event.event_id] = event

    def append_many(self, events: Iterable[PortfolioEvent]) -> None:
        # Validate into a temporary copy before mutating, to keep batch append atomic.
        ordered = list(events)
        with self._lock:
            temp = {account: list(rows) for account, rows in self._events.items()}
            temp_ids = dict(self._by_id)
            for event in ordered:
                prior = temp_ids.get(event.event_id)
                if prior:
                    if prior.digest != event.digest:
                        raise PortfolioReplayError("event id reused with different content")
                    continue
                rows = temp.setdefault(event.account, [])
                expected = rows[-1].sequence + 1 if rows else 1
                if event.sequence != expected:
                    raise PortfolioReplayError(f"expected sequence {expected}, got {event.sequence}")
                if rows and event.occurred_at < rows[-1].occurred_at:
                    raise PortfolioReplayError("event timestamps must not move backward")
                rows.append(event)
                temp_ids[event.event_id] = event
            self._events, self._by_id = temp, temp_ids

    def events(self, account: str, *, through_sequence: int | None = None) -> tuple[PortfolioEvent, ...]:
        with self._lock:
            rows = tuple(self._events.get(account, ()))
        if through_sequence is not None:
            rows = tuple(e for e in rows if e.sequence <= through_sequence)
        return rows

    def verify(self, account: str) -> tuple[str, ...]:
        errors = []
        rows = self.events(account)
        prior_time = None
        for index, event in enumerate(rows, start=1):
            if event.sequence != index:
                errors.append(f"sequence discontinuity at {event.event_id}")
            if prior_time is not None and event.occurred_at < prior_time:
                errors.append(f"timestamp reversal at {event.event_id}")
            if self._by_id.get(event.event_id) != event:
                errors.append(f"id index mismatch at {event.event_id}")
            prior_time = event.occurred_at
        return tuple(errors)


class PortfolioProjector:
    """Rebuild positions and cash from ordered events; pricing is externally supplied."""
    def __init__(self, *, market_prices: Mapping[tuple[str, str], float] | None = None) -> None:
        self.market_prices = dict(market_prices or {})

    def project(self, events: Sequence[PortfolioEvent], *, account: str,
                as_of: datetime | None = None) -> PortfolioProjection:
        if any(e.account != account for e in events):
            raise ValueError("all events must belong to requested account")
        selected = [e for e in events if as_of is None or e.occurred_at <= as_of]
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        selected.sort(key=lambda e: e.sequence)
        expected = 1
        quantities: dict[tuple[str, str], float] = {}
        basis: dict[tuple[str, str], float] = {}
        prices = dict(self.market_prices)
        cash: dict[str, float] = {}
        last_seq = 0
        for event in selected:
            if event.sequence != expected:
                raise PortfolioReplayError(f"expected event sequence {expected}, got {event.sequence}")
            expected += 1
            last_seq = event.sequence
            ccy = event.currency.upper()
            if event.cash_delta:
                cash[ccy] = cash.get(ccy, 0.0) + event.cash_delta
            if event.symbol:
                key = (event.symbol.upper(), ccy)
                if event.event_type is PortfolioEventType.OPENING_POSITION:
                    quantities[key] = quantities.get(key, 0.0) + event.quantity_delta
                    if event.price is not None:
                        basis[key] = basis.get(key, 0.0) + event.quantity_delta * event.price
                elif event.event_type is PortfolioEventType.FILL:
                    old_qty = quantities.get(key, 0.0)
                    old_basis = basis.get(key, 0.0)
                    new_qty = old_qty + event.quantity_delta
                    if event.price is None:
                        raise PortfolioReplayError("fill events require price")
                    if old_qty == 0 or old_qty * event.quantity_delta >= 0:
                        new_basis = old_basis + event.quantity_delta * event.price
                    else:
                        # Reduce existing position basis proportionally, carrying any
                        # residual through zero at the execution price.
                        close_qty = min(abs(old_qty), abs(event.quantity_delta))
                        remaining_old = abs(old_qty) - close_qty
                        old_avg = old_basis / abs(old_qty) if old_qty else 0.0
                        residual_basis = remaining_old * old_avg
                        opening_qty = abs(event.quantity_delta) - close_qty
                        new_basis = residual_basis + opening_qty * event.price * (1 if new_qty >= 0 else -1)
                    quantities[key], basis[key] = new_qty, new_basis
                    prices[key] = event.price
                elif event.event_type is PortfolioEventType.MARK:
                    if event.price is None:
                        raise PortfolioReplayError("mark events require price")
                    prices[key] = event.price
                elif event.event_type is PortfolioEventType.SPLIT:
                    ratio = event.quantity_delta
                    if ratio <= 0:
                        raise PortfolioReplayError("split ratio must be positive")
                    quantities[key] = quantities.get(key, 0.0) * ratio
                    if key in basis:
                        basis[key] = basis[key]  # split preserves total basis
                    if key in prices:
                        prices[key] = prices[key] / ratio
        projections = []
        for (symbol, ccy), qty in sorted(quantities.items()):
            total_basis = basis.get((symbol, ccy), 0.0)
            avg_cost = abs(total_basis / qty) if qty else 0.0
            price = prices.get((symbol, ccy))
            market_value = qty * price if price is not None else None
            unrealized = (qty * price - total_basis) if price is not None else None
            if abs(qty) > 1e-12 or abs(total_basis) > 1e-12:
                projections.append(PositionProjection(account, symbol, qty, avg_cost, price,
                    market_value, unrealized, ccy, last_seq))
        digest_data = {"account": account, "cash": cash,
            "positions": [{"symbol": p.symbol, "quantity": p.quantity,
                "average_cost": p.average_cost, "market_price": p.market_price,
                "market_value": p.market_value, "unrealized_pnl": p.unrealized_pnl,
                "currency": p.currency} for p in projections], "last_sequence": last_seq}
        digest = hashlib.sha256(json.dumps(digest_data, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        return PortfolioProjection(account, dict(cash), tuple(projections), len(selected), last_seq,
            selected[-1].occurred_at.isoformat() if selected else None, digest)

    def verify_replay_determinism(self, events: Sequence[PortfolioEvent], *, account: str) -> bool:
        left = self.project(events, account=account)
        right = self.project(tuple(events), account=account)
        return left.projection_digest == right.projection_digest


def create_event(account: str, sequence: int, event_type: PortfolioEventType, *,
                 symbol: str | None = None, quantity_delta: float = 0.0,
                 price: float | None = None, cash_delta: float = 0.0,
                 currency: str = "USD", reference: str = "",
                 occurred_at: datetime | None = None, event_id: str | None = None,
                 metadata: Mapping[str, str | float | int] | None = None) -> PortfolioEvent:
    import uuid
    return PortfolioEvent(event_id or str(uuid.uuid4()), sequence,
        occurred_at or datetime.now(timezone.utc), event_type, account, symbol,
        quantity_delta, price, cash_delta, currency, reference, dict(metadata or {}))
