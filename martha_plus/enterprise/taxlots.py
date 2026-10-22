"""Tax-lot accounting primitives with deterministic allocation policies.

These estimates are accounting aids only, not tax advice. Jurisdiction-specific
basis adjustments, wash-sale rules, FX treatment, and reporting requirements
must be reviewed by a qualified professional before production use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Iterable, Mapping
import uuid


class LotMethod(str, Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    HIFO = "hifo"
    LOFO = "lofo"


class LotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TaxLot:
    lot_id: str
    account: str
    symbol: str
    quantity: float
    unit_cost: float
    acquired_at: date
    currency: str = "USD"
    fees_per_unit: float = 0.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.lot_id, self.account, self.symbol, self.currency)):
            raise ValueError("lot id, account, symbol, and currency are required")
        if not isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("lot quantity must be finite and positive")
        if not isfinite(self.unit_cost) or self.unit_cost < 0:
            raise ValueError("unit cost must be finite and non-negative")
        if not isfinite(self.fees_per_unit) or self.fees_per_unit < 0:
            raise ValueError("fees per unit must be finite and non-negative")

    @property
    def adjusted_unit_cost(self) -> float:
        return self.unit_cost + self.fees_per_unit

    @property
    def basis(self) -> float:
        return self.quantity * self.adjusted_unit_cost


@dataclass(frozen=True, slots=True)
class LotDisposal:
    disposal_id: str
    lot_id: str
    account: str
    symbol: str
    quantity: float
    unit_proceeds: float
    unit_basis: float
    fees_per_unit: float
    realized_pnl: float
    acquired_at: date
    disposed_at: date
    currency: str
    method: str

    @property
    def proceeds(self) -> float:
        return self.quantity * (self.unit_proceeds - self.fees_per_unit)

    @property
    def basis(self) -> float:
        return self.quantity * self.unit_basis

    @property
    def holding_days(self) -> int:
        return (self.disposed_at - self.acquired_at).days


@dataclass(frozen=True, slots=True)
class LotPosition:
    account: str
    symbol: str
    quantity: float
    cost_basis: float
    average_cost: float
    lot_count: int
    currency: str


class TaxLotBook:
    """Book buys and sells against tax lots; each sell is atomic under a lock."""
    def __init__(self, *, id_factory=lambda: str(uuid.uuid4())) -> None:
        self._id_factory = id_factory
        self._lots: dict[str, TaxLot] = {}
        self._disposals: list[LotDisposal] = []
        self._lock = RLock()

    @property
    def disposals(self) -> tuple[LotDisposal, ...]:
        with self._lock:
            return tuple(self._disposals)

    def add_lot(self, account: str, symbol: str, quantity: float, unit_cost: float,
                acquired_at: date, *, currency: str = "USD", fees: float = 0.0,
                lot_id: str | None = None, tags: Iterable[str] = ()) -> TaxLot:
        lot = TaxLot(lot_id or self._id_factory(), account, symbol.upper(), quantity,
                     unit_cost, acquired_at, currency.upper(), fees, tuple(tags))
        with self._lock:
            if lot.lot_id in self._lots:
                raise LotError(f"lot already exists: {lot.lot_id}")
            # A security's lots in a single account should not silently mix currencies.
            currencies = {v.currency for v in self._lots.values()
                          if v.account == lot.account and v.symbol == lot.symbol}
            if currencies and lot.currency not in currencies:
                raise LotError("cannot mix currencies within a security position")
            self._lots[lot.lot_id] = lot
            return lot

    def available_lots(self, account: str, symbol: str) -> tuple[TaxLot, ...]:
        with self._lock:
            return tuple(sorted((lot for lot in self._lots.values()
                                 if lot.account == account and lot.symbol == symbol.upper()),
                                key=lambda lot: (lot.acquired_at, lot.lot_id)))

    def position(self, account: str, symbol: str) -> LotPosition:
        lots = self.available_lots(account, symbol)
        qty = sum(l.quantity for l in lots)
        basis = sum(l.basis for l in lots)
        currency = lots[0].currency if lots else "USD"
        return LotPosition(account, symbol.upper(), qty, basis, basis / qty if qty else 0.0,
                           len(lots), currency)

    def dispose(self, account: str, symbol: str, quantity: float, unit_proceeds: float,
                disposed_at: date, *, method: LotMethod = LotMethod.FIFO,
                fees: float = 0.0, lot_ids: Iterable[str] | None = None) -> tuple[LotDisposal, ...]:
        sym = symbol.upper()
        if not isfinite(quantity) or quantity <= 0:
            raise ValueError("disposal quantity must be positive and finite")
        if not isfinite(unit_proceeds) or unit_proceeds < 0 or not isfinite(fees) or fees < 0:
            raise ValueError("proceeds and fees must be finite and non-negative")
        with self._lock:
            candidates = [lot for lot in self._lots.values()
                          if lot.account == account and lot.symbol == sym]
            if lot_ids is not None:
                selected = set(lot_ids)
                unknown = selected - {lot.lot_id for lot in candidates}
                if unknown:
                    raise LotError(f"unknown lot ids for position: {sorted(unknown)}")
                candidates = [lot for lot in candidates if lot.lot_id in selected]
            if method is LotMethod.FIFO:
                candidates.sort(key=lambda x: (x.acquired_at, x.lot_id))
            elif method is LotMethod.LIFO:
                candidates.sort(key=lambda x: (x.acquired_at, x.lot_id), reverse=True)
            elif method is LotMethod.HIFO:
                candidates.sort(key=lambda x: (-x.adjusted_unit_cost, x.acquired_at, x.lot_id))
            elif method is LotMethod.LOFO:
                candidates.sort(key=lambda x: (x.adjusted_unit_cost, x.acquired_at, x.lot_id))
            else:
                raise ValueError(f"unsupported lot method: {method}")
            available = sum(l.quantity for l in candidates)
            if quantity > available + 1e-9:
                raise LotError(f"insufficient shares: requested {quantity}, available {available}")
            remaining = quantity
            planned: list[tuple[TaxLot, float]] = []
            for lot in candidates:
                take = min(lot.quantity, remaining)
                if take > 1e-9:
                    planned.append((lot, take))
                    remaining -= take
                if remaining <= 1e-9:
                    break
            if remaining > 1e-9:
                raise LotError("short lots are not represented by this book")
            # Validate the whole sale before mutating the book.
            disposal_rows: list[LotDisposal] = []
            replacements: dict[str, TaxLot | None] = {}
            for lot, take in planned:
                net_proceeds = unit_proceeds - fees
                unit_basis = lot.adjusted_unit_cost
                disposal_rows.append(LotDisposal(
                    self._id_factory(), lot.lot_id, account, sym, take, unit_proceeds,
                    unit_basis, fees, (net_proceeds - unit_basis) * take,
                    lot.acquired_at, disposed_at, lot.currency, method.value))
                residual = lot.quantity - take
                replacements[lot.lot_id] = (TaxLot(lot.lot_id, lot.account, lot.symbol, residual,
                    lot.unit_cost, lot.acquired_at, lot.currency, lot.fees_per_unit, lot.tags)
                    if residual > 1e-9 else None)
            for lot_id, replacement in replacements.items():
                if replacement is None:
                    del self._lots[lot_id]
                else:
                    self._lots[lot_id] = replacement
            self._disposals.extend(disposal_rows)
            return tuple(disposal_rows)

    def apply_split(self, account: str, symbol: str, ratio: float, effective_on: date) -> int:
        """Apply a positive split ratio; total basis stays constant."""
        if not isfinite(ratio) or ratio <= 0:
            raise ValueError("split ratio must be positive")
        sym = symbol.upper()
        count = 0
        with self._lock:
            updated = {}
            for key, lot in self._lots.items():
                if lot.account == account and lot.symbol == sym and lot.acquired_at <= effective_on:
                    updated[key] = TaxLot(lot.lot_id, lot.account, lot.symbol,
                        lot.quantity * ratio, lot.unit_cost / ratio, lot.acquired_at,
                        lot.currency, lot.fees_per_unit / ratio, lot.tags)
                    count += 1
                else:
                    updated[key] = lot
            self._lots = updated
        return count

    def adjustment(self, lot_id: str, *, unit_cost_delta: float) -> TaxLot:
        """Record an explicit per-unit basis adjustment, preserving quantity."""
        if not isfinite(unit_cost_delta):
            raise ValueError("unit cost delta must be finite")
        with self._lock:
            lot = self._lots[lot_id]
            new_cost = lot.unit_cost + unit_cost_delta
            if new_cost < 0:
                raise LotError("basis adjustment would make unit cost negative")
            updated = TaxLot(lot.lot_id, lot.account, lot.symbol, lot.quantity,
                             new_cost, lot.acquired_at, lot.currency,
                             lot.fees_per_unit, lot.tags + ("basis_adjusted",))
            self._lots[lot_id] = updated
            return updated

    def realized_summary(self, *, account: str | None = None,
                         symbol: str | None = None) -> dict[str, float | int]:
        with self._lock:
            rows = [d for d in self._disposals if (account is None or d.account == account)
                    and (symbol is None or d.symbol == symbol.upper())]
        return {"disposals": len(rows), "quantity": sum(d.quantity for d in rows),
                "proceeds": sum(d.proceeds for d in rows), "basis": sum(d.basis for d in rows),
                "realized_pnl": sum(d.realized_pnl for d in rows),
                "short_term_quantity": sum(d.quantity for d in rows if d.holding_days < 365),
                "long_term_quantity": sum(d.quantity for d in rows if d.holding_days >= 365)}

    def integrity_check(self) -> tuple[str, ...]:
        errors = []
        with self._lock:
            for key, lot in self._lots.items():
                if key != lot.lot_id:
                    errors.append(f"key mismatch for lot {key}")
                if lot.quantity <= 0:
                    errors.append(f"non-positive quantity for lot {key}")
            for disposal in self._disposals:
                expected = (disposal.unit_proceeds - disposal.fees_per_unit - disposal.unit_basis) * disposal.quantity
                if abs(expected - disposal.realized_pnl) > 1e-8:
                    errors.append(f"realized PnL mismatch for disposal {disposal.disposal_id}")
        return tuple(errors)
