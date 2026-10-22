"""Cash balances, reservations, and conservative settlement-aware cash forecasting."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Iterable, Mapping, Sequence
import uuid


class CashFlowKind(str, Enum):
    TRADE_SETTLEMENT = "trade_settlement"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    FEE = "fee"
    TAX = "tax"
    MANUAL = "manual"


class ReservationStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    CONSUMED = "consumed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class CashFlowItem:
    flow_id: str
    account: str
    currency: str
    settlement_date: date
    amount: float
    kind: CashFlowKind
    reference: str = ""
    confirmed: bool = True
    source: str = "manual"

    def __post_init__(self) -> None:
        if not self.flow_id.strip() or not self.account.strip() or len(self.currency.strip()) != 3:
            raise ValueError("flow_id, account, and a 3-character currency are required")
        if not isfinite(self.amount):
            raise ValueError("cash flow amount must be finite")


@dataclass(frozen=True, slots=True)
class CashReservation:
    reservation_id: str
    account: str
    currency: str
    amount: float
    purpose: str
    created_at: str
    expires_at: str | None
    status: ReservationStatus = ReservationStatus.ACTIVE
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.reservation_id.strip() or not self.account.strip() or not self.purpose.strip():
            raise ValueError("reservation identity and purpose are required")
        if not isfinite(self.amount) or self.amount <= 0:
            raise ValueError("reservation amount must be positive and finite")
        if datetime.fromisoformat(self.created_at).tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.expires_at and datetime.fromisoformat(self.expires_at).tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class CashBalance:
    account: str
    currency: str
    ledger_balance: float
    pending_credits: float
    pending_debits: float
    reserved: float
    available: float
    projected_after_pending: float


@dataclass(frozen=True, slots=True)
class CashForecastPoint:
    day: date
    opening_balance: float
    net_flows: float
    closing_balance: float
    confirmed_net_flows: float
    unconfirmed_net_flows: float
    minimum_reserve: float
    reserve_breach: bool


class InsufficientCash(RuntimeError):
    pass


class CashBook:
    """Track ledger cash, future cash flows, and explicitly reserved balances."""
    def __init__(self, *, clock=lambda: datetime.now(timezone.utc),
                 id_factory=lambda: str(uuid.uuid4())) -> None:
        self.clock = clock
        self.id_factory = id_factory
        self._balances: dict[tuple[str, str], float] = {}
        self._flows: dict[str, CashFlowItem] = {}
        self._reservations: dict[str, CashReservation] = {}
        self._journal: list[dict[str, str | float]] = []
        self._lock = RLock()

    def set_balance(self, account: str, currency: str, amount: float, *, actor: str,
                    reason: str) -> None:
        if not account.strip() or len(currency.strip()) != 3 or not actor.strip() or not reason.strip():
            raise ValueError("account, currency, actor, and reason are required")
        if not isfinite(amount):
            raise ValueError("amount must be finite")
        key = (account, currency.upper())
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        with self._lock:
            before = self._balances.get(key, 0.0)
            self._balances[key] = amount
            self._journal.append({"action": "set_balance", "account": account,
                "currency": currency.upper(), "before": before, "after": amount,
                "actor": actor, "reason": reason, "at": now.isoformat()})

    def post_flow(self, flow: CashFlowItem) -> None:
        with self._lock:
            if flow.flow_id in self._flows:
                if self._flows[flow.flow_id] != flow:
                    raise ValueError("flow id reused for a different cash flow")
                return
            self._flows[flow.flow_id] = flow

    def cancel_flow(self, flow_id: str, *, reason: str) -> bool:
        if not reason.strip():
            raise ValueError("reason is required")
        with self._lock:
            if flow_id not in self._flows:
                return False
            del self._flows[flow_id]
            return True

    def _active_reservations(self, account: str, currency: str, now: datetime) -> list[CashReservation]:
        return [r for r in self._reservations.values() if r.account == account
            and r.currency == currency and r.status is ReservationStatus.ACTIVE
            and (not r.expires_at or datetime.fromisoformat(r.expires_at) > now)]

    def balance(self, account: str, currency: str, *, at: datetime | None = None) -> CashBalance:
        now = at or self.clock()
        if now.tzinfo is None:
            raise ValueError("at must be timezone-aware")
        ccy = currency.upper()
        key = (account, ccy)
        with self._lock:
            ledger = self._balances.get(key, 0.0)
            flows = [f for f in self._flows.values() if (f.account, f.currency) == key
                     and f.settlement_date >= now.date()]
            reserved = sum(r.amount for r in self._active_reservations(account, ccy, now))
        credits = sum(f.amount for f in flows if f.amount > 0 and f.confirmed)
        debits = -sum(f.amount for f in flows if f.amount < 0 and f.confirmed)
        pending_net = sum(f.amount for f in flows)
        return CashBalance(account, ccy, ledger, credits, debits, reserved,
                           ledger - reserved, ledger + pending_net)

    def reserve(self, account: str, currency: str, amount: float, *, purpose: str,
                expires_at: datetime | None = None, reference: str | None = None,
                allow_pending_credits: bool = False) -> CashReservation:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        if expires_at is not None and (expires_at.tzinfo is None or expires_at <= now):
            raise ValueError("expiry must be timezone-aware and in the future")
        ccy = currency.upper()
        with self._lock:
            ledger = self._balances.get((account, ccy), 0.0)
            active = self._active_reservations(account, ccy, now)
            reserved = sum(r.amount for r in active)
            pending_credits = 0.0
            if allow_pending_credits:
                pending_credits = sum(f.amount for f in self._flows.values()
                    if f.account == account and f.currency == ccy and f.amount > 0
                    and f.confirmed and f.settlement_date >= now.date())
            available = ledger + pending_credits - reserved
            if amount <= 0 or not isfinite(amount):
                raise ValueError("reservation amount must be positive and finite")
            if amount > available + 1e-9:
                raise InsufficientCash(f"available cash {available:.2f} is below reservation {amount:.2f}")
            reservation = CashReservation(self.id_factory(), account, ccy, amount, purpose,
                now.isoformat(), expires_at.isoformat() if expires_at else None,
                ReservationStatus.ACTIVE, reference)
            self._reservations[reservation.reservation_id] = reservation
            self._journal.append({"action": "reserve", "reservation_id": reservation.reservation_id,
                "account": account, "currency": ccy, "amount": amount, "at": now.isoformat()})
            return reservation

    def update_reservation(self, reservation_id: str, status: ReservationStatus, *,
                           actor: str, reason: str) -> CashReservation:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required")
        now = self.clock()
        with self._lock:
            old = self._reservations[reservation_id]
            if old.status is not ReservationStatus.ACTIVE:
                raise ValueError("only active reservations may transition")
            if status is ReservationStatus.ACTIVE:
                raise ValueError("reservation cannot transition back to active")
            new = CashReservation(old.reservation_id, old.account, old.currency, old.amount,
                old.purpose, old.created_at, old.expires_at, status, old.reference)
            self._reservations[reservation_id] = new
            self._journal.append({"action": "reservation_transition", "reservation_id": reservation_id,
                "from": old.status.value, "to": status.value, "actor": actor,
                "reason": reason, "at": now.isoformat()})
            return new

    def expire_reservations(self, *, at: datetime | None = None) -> int:
        now = at or self.clock()
        if now.tzinfo is None:
            raise ValueError("at must be timezone-aware")
        count = 0
        with self._lock:
            for key, old in list(self._reservations.items()):
                if old.status is ReservationStatus.ACTIVE and old.expires_at and datetime.fromisoformat(old.expires_at) <= now:
                    self._reservations[key] = CashReservation(old.reservation_id, old.account,
                        old.currency, old.amount, old.purpose, old.created_at, old.expires_at,
                        ReservationStatus.EXPIRED, old.reference)
                    count += 1
        return count

    def forecast(self, account: str, currency: str, *, start: date, end: date,
                 minimum_reserve: float = 0.0, include_unconfirmed: bool = True) -> tuple[CashForecastPoint, ...]:
        if end < start or minimum_reserve < 0 or not isfinite(minimum_reserve):
            raise ValueError("invalid date range or reserve")
        with self._lock:
            opening = self._balances.get((account, currency.upper()), 0.0)
            flows = [f for f in self._flows.values() if f.account == account
                     and f.currency == currency.upper() and start <= f.settlement_date <= end]
        running = opening
        output = []
        for day_offset in range((end - start).days + 1):
            day = start + timedelta(days=day_offset)
            today = [f for f in flows if f.settlement_date == day and (include_unconfirmed or f.confirmed)]
            net = sum(f.amount for f in today)
            confirmed = sum(f.amount for f in today if f.confirmed)
            unconfirmed = net - confirmed
            closing = running + net
            output.append(CashForecastPoint(day, running, net, closing, confirmed, unconfirmed,
                                            minimum_reserve, closing < minimum_reserve))
            running = closing
        return tuple(output)

    def reservations(self, *, account: str | None = None) -> tuple[CashReservation, ...]:
        with self._lock:
            rows = tuple(self._reservations.values())
        return tuple(r for r in rows if account is None or r.account == account)

    @property
    def journal(self) -> tuple[Mapping[str, str | float], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._journal)

    def audit(self) -> tuple[str, ...]:
        errors = []
        with self._lock:
            for key, value in self._balances.items():
                if len(key[1]) != 3 or not isfinite(value):
                    errors.append(f"invalid balance entry: {key}")
            for key, reservation in self._reservations.items():
                if key != reservation.reservation_id:
                    errors.append(f"reservation key mismatch: {key}")
            for key, flow in self._flows.items():
                if key != flow.flow_id:
                    errors.append(f"flow key mismatch: {key}")
        return tuple(errors)
