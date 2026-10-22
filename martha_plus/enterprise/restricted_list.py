"""Versioned restricted instrument list and pre-trade eligibility decisions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Callable, Iterable, Mapping


class RestrictionLevel(str, Enum):
    BLOCK = "block"
    REDUCE_ONLY = "reduce_only"
    REVIEW = "review"
    WATCH = "watch"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Restriction:
    symbol: str
    level: RestrictionLevel
    reason: str
    effective_from: datetime
    effective_to: datetime | None = None
    source: str = "manual"
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.reason.strip():
            raise ValueError("symbol and restriction reason are required")
        if self.effective_from.tzinfo is None:
            raise ValueError("effective_from must be timezone-aware")
        if self.effective_to is not None:
            if self.effective_to.tzinfo is None or self.effective_to <= self.effective_from:
                raise ValueError("effective_to must be after effective_from")

    def active(self, now: datetime) -> bool:
        return self.effective_from <= now and (self.effective_to is None or now < self.effective_to)


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    allowed: bool
    requires_review: bool
    symbol: str
    side: Side
    quantity: float
    reasons: tuple[str, ...]
    restriction_ids: tuple[str, ...]
    checked_at: str


class RestrictedInstrumentRegistry:
    def __init__(self, *, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.clock = clock
        self._rows: dict[str, Restriction] = {}
        self._history: list[dict[str, str]] = []
        self._lock = RLock()

    def upsert(self, restriction_id: str, restriction: Restriction, *, actor: str) -> None:
        if not restriction_id.strip() or not actor.strip():
            raise ValueError("restriction_id and actor are required")
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        symbol = restriction.symbol.upper()
        normalized = Restriction(symbol, restriction.level, restriction.reason,
            restriction.effective_from, restriction.effective_to, restriction.source, restriction.case_id)
        with self._lock:
            operation = "updated" if restriction_id in self._rows else "created"
            self._rows[restriction_id] = normalized
            self._history.append({"id": restriction_id, "actor": actor, "operation": operation,
                                  "symbol": symbol, "at": now.isoformat()})

    def remove(self, restriction_id: str, *, actor: str, reason: str) -> bool:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and removal reason are required")
        now = self.clock()
        with self._lock:
            row = self._rows.pop(restriction_id, None)
            if row is None:
                return False
            self._history.append({"id": restriction_id, "actor": actor, "operation": "removed",
                                  "symbol": row.symbol, "reason": reason, "at": now.isoformat()})
            return True

    def active(self, symbol: str | None = None, *, at: datetime | None = None) -> tuple[tuple[str, Restriction], ...]:
        now = at or self.clock()
        if now.tzinfo is None:
            raise ValueError("at must be timezone-aware")
        with self._lock:
            items = tuple(self._rows.items())
        return tuple((key, value) for key, value in items if value.active(now)
                     and (symbol is None or value.symbol == symbol.upper()))

    def check_order(self, symbol: str, side: Side, quantity: float, *,
                    current_position: float = 0.0, at: datetime | None = None) -> EligibilityDecision:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        now = at or self.clock()
        restrictions = self.active(symbol, at=now)
        reasons: list[str] = []
        ids: list[str] = []
        blocked = False
        review = False
        for rid, restriction in restrictions:
            ids.append(rid)
            if restriction.level is RestrictionLevel.BLOCK:
                blocked = True
                reasons.append(f"blocked:{restriction.reason}")
            elif restriction.level is RestrictionLevel.REDUCE_ONLY:
                increases_exposure = (side is Side.BUY and current_position >= 0) or (
                    side is Side.SELL and current_position <= 0)
                if increases_exposure:
                    blocked = True
                    reasons.append(f"reduce_only:{restriction.reason}")
            elif restriction.level is RestrictionLevel.REVIEW:
                review = True
                reasons.append(f"review_required:{restriction.reason}")
            else:
                reasons.append(f"watch:{restriction.reason}")
        return EligibilityDecision(not blocked, review, symbol.upper(), side, quantity,
                                   tuple(reasons), tuple(ids), now.isoformat())

    @property
    def audit_history(self) -> tuple[Mapping[str, str], ...]:
        with self._lock:
            return tuple(dict(item) for item in self._history)

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = tuple(self._rows.values())
        return {level.value: sum(r.level is level for r in rows) for level in RestrictionLevel}
