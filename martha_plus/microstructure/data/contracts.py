"""Typed feed contracts, validation results, and data-quality diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..types import EventKind, MarketEvent, Side


class QualityCode(str, Enum):
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_SEQUENCE = "invalid_sequence"
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    MISSING_SIDE = "missing_side"
    MISSING_ORDER_ID = "missing_order_id"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    SEQUENCE_GAP = "sequence_gap"
    OUT_OF_ORDER = "out_of_order"
    UNKNOWN_EVENT = "unknown_event"
    SYMBOL_MISMATCH = "symbol_mismatch"
    STALE_EVENT = "stale_event"
    CROSSED_BOOK = "crossed_book"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: QualityCode
    message: str
    ts_ns: int | None = None
    sequence: int | None = None
    severity: str = "warning"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    event: MarketEvent | None
    issues: tuple[QualityIssue, ...] = ()

    @property
    def errors(self) -> tuple[QualityIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")


@dataclass(frozen=True, slots=True)
class FeedEnvelope:
    """Canonical, provider-neutral representation of one feed update."""
    provider: str
    channel: str
    received_at_ns: int
    event: MarketEvent
    provider_sequence: int | None = None
    venue: str = "SIM"
    source_timestamp_ns: int | None = None
    checksum: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.channel.strip():
            raise ValueError("provider and channel must not be blank")
        if self.received_at_ns < 0:
            raise ValueError("received_at_ns must be non-negative")
        if self.provider_sequence is not None and self.provider_sequence < 0:
            raise ValueError("provider_sequence must be non-negative")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("source_timestamp_ns must be non-negative")

    @property
    def latency_ns(self) -> int | None:
        if self.source_timestamp_ns is None:
            return None
        return self.received_at_ns - self.source_timestamp_ns


@dataclass(frozen=True, slots=True)
class FeedHealth:
    provider: str
    channel: str
    received: int
    accepted: int
    rejected: int
    duplicates: int
    gaps: int
    out_of_order: int
    last_sequence: int | None
    last_event_ns: int | None
    last_received_ns: int | None
    mean_latency_ns: float | None


class MarketEventValidator:
    """Boundary validator for canonical market events.

    Validation is intentionally conservative.  It does not repair unknown
    messages or silently coerce floating point prices into integer ticks.
    """

    def __init__(self, symbol: str | None = None, *, max_age_ns: int | None = None) -> None:
        self.symbol = symbol.strip().upper() if symbol else None
        if max_age_ns is not None and max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        self.max_age_ns = max_age_ns

    def validate(self, raw: Mapping[str, Any], *, now_ns: int | None = None) -> ValidationResult:
        issues: list[QualityIssue] = []
        try:
            kind = EventKind(str(raw["kind"]).lower())
            symbol = str(raw["symbol"]).strip().upper()
            ts_ns = int(raw["ts_ns"])
            sequence = int(raw["sequence"])
            side_raw = raw.get("side")
            side = Side(str(side_raw).lower()) if side_raw is not None else None
            price = raw.get("price_ticks")
            quantity = raw.get("quantity")
            if price is not None and (isinstance(price, bool) or not isinstance(price, int)):
                raise TypeError("price_ticks must be an integer number of ticks")
            if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, int)):
                raise TypeError("quantity must be an integer number of lots")
            event = MarketEvent(ts_ns, sequence, symbol, kind, side, price,
                                quantity, str(raw.get("order_id", "")),
                                raw.get("reference_price_ticks"))
        except (KeyError, ValueError, TypeError) as exc:
            issues.append(QualityIssue(QualityCode.UNKNOWN_EVENT, str(exc), severity="error"))
            return ValidationResult(False, None, tuple(issues))
        if self.symbol and event.symbol != self.symbol:
            issues.append(QualityIssue(QualityCode.SYMBOL_MISMATCH,
                                       f"expected {self.symbol}, received {event.symbol}",
                                       event.ts_ns, event.sequence, "error"))
        if now_ns is not None and self.max_age_ns is not None:
            if now_ns - event.ts_ns > self.max_age_ns:
                issues.append(QualityIssue(QualityCode.STALE_EVENT,
                                           "event exceeds configured age limit",
                                           event.ts_ns, event.sequence))
        errors = any(x.severity == "error" for x in issues)
        return ValidationResult(not errors, None if errors else event, tuple(issues))

    def validate_event(self, event: MarketEvent) -> ValidationResult:
        issues = []
        if self.symbol and event.symbol != self.symbol:
            issues.append(QualityIssue(QualityCode.SYMBOL_MISMATCH,
                                       f"expected {self.symbol}, received {event.symbol}",
                                       event.ts_ns, event.sequence, "error"))
        return ValidationResult(not issues, None if issues else event, tuple(issues))


class FeedSequenceMonitor:
    """Detect sequence duplicates, gaps, and timestamp regressions."""

    def __init__(self, *, allow_reset: bool = False) -> None:
        self.allow_reset = allow_reset
        self.last_sequence: int | None = None
        self.last_timestamp_ns: int | None = None
        self.received = self.accepted = self.rejected = 0
        self.duplicates = self.gaps = self.out_of_order = 0

    def observe(self, envelope: FeedEnvelope) -> tuple[QualityIssue, ...]:
        self.received += 1
        issues: list[QualityIssue] = []
        seq = envelope.provider_sequence
        ts = envelope.source_timestamp_ns or envelope.event.ts_ns
        if seq is not None and self.last_sequence is not None:
            if seq == self.last_sequence:
                self.duplicates += 1
                issues.append(QualityIssue(QualityCode.DUPLICATE_SEQUENCE,
                                           "provider sequence repeated", ts, seq))
            elif seq < self.last_sequence:
                self.out_of_order += 1
                issues.append(QualityIssue(QualityCode.OUT_OF_ORDER,
                                           "provider sequence regressed", ts, seq))
                if self.allow_reset:
                    self.last_sequence = seq
            elif seq > self.last_sequence + 1:
                self.gaps += seq - self.last_sequence - 1
                issues.append(QualityIssue(QualityCode.SEQUENCE_GAP,
                                           f"missing {seq - self.last_sequence - 1} sequence(s)", ts, seq))
                self.last_sequence = seq
            else:
                self.last_sequence = seq
        elif seq is not None:
            self.last_sequence = seq
        if self.last_timestamp_ns is not None and ts < self.last_timestamp_ns:
            self.out_of_order += 1
            issues.append(QualityIssue(QualityCode.OUT_OF_ORDER,
                                       "source timestamp regressed", ts, seq))
        self.last_timestamp_ns = ts if self.last_timestamp_ns is None else max(self.last_timestamp_ns, ts)
        if issues:
            self.rejected += 1
        else:
            self.accepted += 1
        return tuple(issues)

    def health(self, provider: str, channel: str, *, last_received_ns: int | None = None,
               mean_latency_ns: float | None = None) -> FeedHealth:
        return FeedHealth(provider, channel, self.received, self.accepted, self.rejected,
                          self.duplicates, self.gaps, self.out_of_order,
                          self.last_sequence, self.last_timestamp_ns,
                          last_received_ns, mean_latency_ns)


@dataclass(frozen=True, slots=True)
class SnapshotLevel:
    side: Side
    price_ticks: int
    quantity: int
    order_count: int = 1

    def __post_init__(self) -> None:
        if self.price_ticks <= 0 or self.quantity <= 0 or self.order_count <= 0:
            raise ValueError("snapshot level values must be positive")


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    symbol: str
    ts_ns: int
    sequence: int
    bids: tuple[SnapshotLevel, ...]
    asks: tuple[SnapshotLevel, ...]
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip() or min(self.ts_ns, self.sequence) < 0:
            raise ValueError("invalid snapshot header")
        bid_prices = [x.price_ticks for x in self.bids]
        ask_prices = [x.price_ticks for x in self.asks]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise ValueError("bids must be ordered best to worst")
        if ask_prices != sorted(ask_prices):
            raise ValueError("asks must be ordered best to worst")
        if self.bids and self.asks and self.bids[0].price_ticks >= self.asks[0].price_ticks:
            raise ValueError("snapshot must not be locked or crossed")


@dataclass(frozen=True, slots=True)
class NormalizationAudit:
    input_count: int
    output_count: int
    dropped_count: int
    reordered_count: int
    duplicate_count: int
    issues: tuple[QualityIssue, ...] = field(default_factory=tuple)
