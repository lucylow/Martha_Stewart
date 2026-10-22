"""Three-way reconciliation for internal ledger, fills, and broker statements."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from math import isclose, isfinite
from typing import Any, Iterable, Mapping, Sequence


class DifferenceKind(str, Enum):
    MISSING_INTERNAL = "missing_internal"
    MISSING_EXTERNAL = "missing_external"
    QUANTITY_MISMATCH = "quantity_mismatch"
    PRICE_MISMATCH = "price_mismatch"
    CASH_MISMATCH = "cash_mismatch"
    SYMBOL_MISMATCH = "symbol_mismatch"
    TIMESTAMP_MISMATCH = "timestamp_mismatch"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"


class ReconStatus(str, Enum):
    MATCHED = "matched"
    BREAK = "break"
    REVIEWED = "reviewed"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    execution_id: str
    order_id: str
    account: str
    symbol: str
    side: str
    quantity: float
    price: float
    fees: float
    timestamp: datetime
    currency: str = "USD"
    source: str = "internal"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.execution_id, self.order_id, self.account, self.symbol)):
            raise ValueError("execution identity fields are required")
        if self.timestamp.tzinfo is None:
            raise ValueError("execution timestamp must be timezone-aware")
        if any(not isfinite(v) for v in (self.quantity, self.price, self.fees)):
            raise ValueError("execution numeric values must be finite")
        if self.quantity <= 0 or self.price < 0 or self.fees < 0:
            raise ValueError("quantity must be positive; price and fees non-negative")

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def signed_cash(self) -> float:
        side = self.side.lower()
        if side in {"buy", "cover"}:
            return -self.notional - self.fees
        if side in {"sell", "short"}:
            return self.notional - self.fees
        raise ValueError(f"unsupported side: {self.side}")


@dataclass(frozen=True, slots=True)
class ReconBreak:
    break_id: str
    kind: DifferenceKind
    status: ReconStatus
    execution_id: str | None
    internal_id: str | None
    external_id: str | None
    account: str
    symbol: str
    summary: str
    internal_value: float | str | None
    external_value: float | str | None
    detected_at: str
    severity: str = "medium"
    assigned_to: str | None = None
    resolution_note: str | None = None


@dataclass(frozen=True, slots=True)
class ReconSummary:
    internal_count: int
    external_count: int
    matched_count: int
    break_count: int
    duplicate_count: int
    absolute_quantity_difference: float
    absolute_cash_difference: float
    breaks: tuple[ReconBreak, ...]


class ExecutionReconciler:
    """Match execution records by externalizable execution id, then compare values."""
    def __init__(self, *, quantity_tolerance: float = 1e-6,
                 price_tolerance: float = 1e-4, cash_tolerance: float = 0.01,
                 timestamp_tolerance_seconds: float = 300.0,
                 clock=lambda: datetime.now(timezone.utc)) -> None:
        if min(quantity_tolerance, price_tolerance, cash_tolerance, timestamp_tolerance_seconds) < 0:
            raise ValueError("tolerances must be non-negative")
        self.quantity_tolerance = quantity_tolerance
        self.price_tolerance = price_tolerance
        self.cash_tolerance = cash_tolerance
        self.timestamp_tolerance = timestamp_tolerance_seconds
        self.clock = clock
        self._breaks: dict[str, ReconBreak] = {}

    def reconcile(self, internal: Sequence[ExecutionRecord], external: Sequence[ExecutionRecord]) -> ReconSummary:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        breaks: list[ReconBreak] = []
        internal_groups: dict[str, list[ExecutionRecord]] = {}
        external_groups: dict[str, list[ExecutionRecord]] = {}
        for row in internal:
            internal_groups.setdefault(row.execution_id, []).append(row)
        for row in external:
            external_groups.setdefault(row.execution_id, []).append(row)
        matched_count = duplicate_count = 0
        total_qty_diff = total_cash_diff = 0.0
        for ident in sorted(set(internal_groups) | set(external_groups)):
            left = internal_groups.get(ident, [])
            right = external_groups.get(ident, [])
            if len(left) > 1 or len(right) > 1:
                duplicate_count += max(0, len(left) - 1) + max(0, len(right) - 1)
                row = (left or right)[0]
                breaks.append(self._make_break(DifferenceKind.DUPLICATE, row, left[0] if left else None,
                    right[0] if right else None, now, f"duplicate execution id: {ident}",
                    len(left), len(right), "high"))
                # Never arbitrarily treat duplicate matches as clean.
                continue
            if not left:
                row = right[0]
                breaks.append(self._make_break(DifferenceKind.MISSING_INTERNAL, row, None, row, now,
                    "execution exists externally but not in internal ledger", None, row.quantity, "high"))
                continue
            if not right:
                row = left[0]
                breaks.append(self._make_break(DifferenceKind.MISSING_EXTERNAL, row, row, None, now,
                    "internal execution missing from external statement", row.quantity, None, "high"))
                continue
            a, b = left[0], right[0]
            found = []
            if a.account != b.account or a.symbol.upper() != b.symbol.upper() or a.side.lower() != b.side.lower():
                found.append(self._make_break(DifferenceKind.SYMBOL_MISMATCH, a, a, b, now,
                    "identity fields differ between systems", f"{a.account}/{a.symbol}/{a.side}",
                    f"{b.account}/{b.symbol}/{b.side}", "high"))
            qdiff = abs(a.quantity - b.quantity)
            pdiff = abs(a.price - b.price)
            cdiff = abs(a.signed_cash - b.signed_cash)
            total_qty_diff += qdiff
            total_cash_diff += cdiff
            if qdiff > self.quantity_tolerance:
                found.append(self._make_break(DifferenceKind.QUANTITY_MISMATCH, a, a, b, now,
                    "execution quantity differs", a.quantity, b.quantity, "high"))
            if pdiff > self.price_tolerance:
                found.append(self._make_break(DifferenceKind.PRICE_MISMATCH, a, a, b, now,
                    "execution price differs", a.price, b.price, "medium"))
            if cdiff > self.cash_tolerance:
                found.append(self._make_break(DifferenceKind.CASH_MISMATCH, a, a, b, now,
                    "net cash movement differs", a.signed_cash, b.signed_cash, "high"))
            tdiff = abs((a.timestamp - b.timestamp).total_seconds())
            if tdiff > self.timestamp_tolerance:
                found.append(self._make_break(DifferenceKind.TIMESTAMP_MISMATCH, a, a, b, now,
                    "execution timestamps differ beyond tolerance", a.timestamp.isoformat(),
                    b.timestamp.isoformat(), "low"))
            if found:
                breaks.extend(found)
            else:
                matched_count += 1
        for row in breaks:
            self._breaks[row.break_id] = row
        return ReconSummary(len(internal), len(external), matched_count, len(breaks),
                            duplicate_count, total_qty_diff, total_cash_diff, tuple(breaks))

    def _make_break(self, kind: DifferenceKind, base: ExecutionRecord,
                    internal: ExecutionRecord | None, external: ExecutionRecord | None,
                    now: datetime, summary: str, internal_value: Any, external_value: Any,
                    severity: str) -> ReconBreak:
        import hashlib
        token = f"{kind.value}|{base.execution_id}|{internal_value}|{external_value}|{summary}"
        ident = hashlib.sha256(token.encode()).hexdigest()[:20]
        return ReconBreak(ident, kind, ReconStatus.BREAK, base.execution_id,
            internal.execution_id if internal else None, external.execution_id if external else None,
            base.account, base.symbol, summary, internal_value, external_value, now.isoformat(), severity)

    def review(self, break_id: str, *, reviewer: str, note: str,
               resolved: bool = False) -> ReconBreak:
        if not reviewer.strip() or not note.strip():
            raise ValueError("reviewer and note are required")
        prior = self._breaks[break_id]
        updated = ReconBreak(prior.break_id, prior.kind,
            ReconStatus.RESOLVED if resolved else ReconStatus.REVIEWED,
            prior.execution_id, prior.internal_id, prior.external_id, prior.account,
            prior.symbol, prior.summary, prior.internal_value, prior.external_value,
            prior.detected_at, prior.severity, reviewer, note)
        self._breaks[break_id] = updated
        return updated

    def open_breaks(self, *, severity: str | None = None) -> tuple[ReconBreak, ...]:
        rows = [row for row in self._breaks.values() if row.status is ReconStatus.BREAK]
        if severity:
            rows = [row for row in rows if row.severity == severity]
        return tuple(sorted(rows, key=lambda row: (row.detected_at, row.break_id)))

    def summary(self) -> dict[str, int]:
        rows = tuple(self._breaks.values())
        return {"total": len(rows), "open": sum(r.status is ReconStatus.BREAK for r in rows),
                "reviewed": sum(r.status is ReconStatus.REVIEWED for r in rows),
                "resolved": sum(r.status is ReconStatus.RESOLVED for r in rows),
                **{kind.value: sum(r.kind is kind and r.status is ReconStatus.BREAK for r in rows)
                   for kind in DifferenceKind}}


def account_cash_control(records: Iterable[ExecutionRecord]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in records:
        key = (row.account, row.currency.upper())
        result[key] = result.get(key, 0.0) + row.signed_cash
    return result


def compare_cash_controls(internal: Mapping[tuple[str, str], float],
                          external: Mapping[tuple[str, str], float], *,
                          tolerance: float = 0.01) -> tuple[dict[str, Any], ...]:
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    output = []
    for key in sorted(set(internal) | set(external)):
        left, right = internal.get(key, 0.0), external.get(key, 0.0)
        diff = left - right
        if not isclose(left, right, abs_tol=tolerance, rel_tol=0.0):
            output.append({"account": key[0], "currency": key[1], "internal": left,
                           "external": right, "difference": diff})
    return tuple(output)
