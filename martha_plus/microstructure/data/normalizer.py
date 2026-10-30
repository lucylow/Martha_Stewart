"""Canonicalize provider-like dictionaries without hiding malformed data."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Iterable, Mapping

from ..types import EventKind, MarketEvent, Side, SymbolSpec
from .contracts import NormalizationAudit, QualityCode, QualityIssue, ValidationResult


@dataclass(frozen=True, slots=True)
class NormalizedBatch:
    events: tuple[MarketEvent, ...]
    audit: NormalizationAudit


class PriceConverter:
    """Convert decimal prices to integer instrument ticks with strict tolerance."""

    def __init__(self, spec: SymbolSpec, *, tolerance: Decimal = Decimal("0.0000001")) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.spec = spec
        self.tolerance = tolerance

    def to_ticks(self, price: Decimal | str | int | float) -> int:
        try:
            value = Decimal(str(price))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("price is not decimal-compatible") from exc
        if not value.is_finite() or value <= 0:
            raise ValueError("price must be finite and positive")
        quotient = value / self.spec.tick_size
        rounded = quotient.to_integral_value(rounding=ROUND_HALF_EVEN)
        if abs(quotient - rounded) > self.tolerance:
            raise ValueError("price is not aligned to the configured tick size")
        return int(rounded)

    def from_ticks(self, price_ticks: int) -> Decimal:
        if price_ticks <= 0:
            raise ValueError("price_ticks must be positive")
        return self.spec.tick_size * price_ticks


class TimestampConverter:
    """Normalize timestamp units to nanoseconds using explicit unit labels."""

    MULTIPLIERS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}

    @classmethod
    def to_ns(cls, value: int | float | Decimal, unit: str) -> int:
        multiplier = cls.MULTIPLIERS.get(unit.lower())
        if multiplier is None:
            raise ValueError(f"unsupported timestamp unit: {unit}")
        decimal_value = Decimal(str(value))
        if not decimal_value.is_finite() or decimal_value < 0:
            raise ValueError("timestamp must be finite and non-negative")
        ns = decimal_value * multiplier
        if ns != ns.to_integral_value():
            raise ValueError("timestamp has sub-nanosecond precision")
        return int(ns)


class MarketDataNormalizer:
    """Normalize a stream of provider records to stable ordered events.

    Duplicate identity is `(symbol, sequence, kind, order_id)`.  A caller can
    choose strict mode to reject an entire batch when any malformed row exists.
    """

    def __init__(self, spec: SymbolSpec, *, strict: bool = False,
                 timestamp_unit: str = "ns") -> None:
        self.spec = spec
        self.strict = strict
        self.timestamp_unit = timestamp_unit
        self.prices = PriceConverter(spec)

    def _convert(self, row: Mapping[str, Any], sequence_fallback: int) -> MarketEvent:
        kind = EventKind(str(row["kind"]).lower())
        symbol = str(row.get("symbol", self.spec.symbol)).strip().upper()
        if symbol != self.spec.symbol:
            raise ValueError(f"symbol mismatch: expected {self.spec.symbol}, received {symbol}")
        ts_value = row["ts_ns"] if "ts_ns" in row else row["timestamp"]
        ts_unit = "ns" if "ts_ns" in row else str(row.get("timestamp_unit", self.timestamp_unit))
        ts_ns = TimestampConverter.to_ns(ts_value, ts_unit)
        sequence = int(row.get("sequence", sequence_fallback))
        side = Side(str(row["side"]).lower()) if row.get("side") is not None else None
        price = row.get("price_ticks")
        if price is None and row.get("price") is not None:
            price = self.prices.to_ticks(row["price"])
        quantity = row.get("quantity")
        if quantity is not None:
            quantity = int(quantity)
        reference = row.get("reference_price_ticks")
        if reference is None and row.get("reference_price") is not None:
            reference = self.prices.to_ticks(row["reference_price"])
        return MarketEvent(ts_ns, sequence, symbol, kind, side,
                           int(price) if price is not None else None,
                           quantity, str(row.get("order_id", "")),
                           int(reference) if reference is not None else None)

    def normalize(self, records: Iterable[Mapping[str, Any]]) -> NormalizedBatch:
        input_rows = list(records)
        parsed: list[tuple[int, MarketEvent]] = []
        issues: list[QualityIssue] = []
        dropped = 0
        for index, row in enumerate(input_rows):
            try:
                parsed.append((index, self._convert(row, index)))
            except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
                dropped += 1
                issues.append(QualityIssue(QualityCode.UNKNOWN_EVENT,
                                           f"row {index}: {exc}", severity="error"))
        if self.strict and dropped:
            raise ValueError(f"normalization rejected {dropped} malformed row(s)")
        original_valid_order = [old_index for old_index, _ in parsed]
        ordered = sorted(parsed, key=lambda pair: (pair[1].ts_ns, pair[1].sequence, pair[0]))
        sorted_valid_order = [old_index for old_index, _ in ordered]
        reordered = sum(1 for before, after in zip(original_valid_order, sorted_valid_order)
                        if before != after)
        seen = set()
        events: list[MarketEvent] = []
        duplicates = 0
        for _, event in ordered:
            key = (event.symbol, event.sequence, event.kind, event.order_id)
            if key in seen:
                duplicates += 1
                issues.append(QualityIssue(QualityCode.DUPLICATE_SEQUENCE,
                                           f"duplicate event key {key}", event.ts_ns, event.sequence))
                continue
            seen.add(key)
            events.append(event)
        audit = NormalizationAudit(len(input_rows), len(events), dropped,
                                   reordered, duplicates, tuple(issues))
        return NormalizedBatch(tuple(events), audit)


def event_to_dict(event: MarketEvent) -> dict[str, Any]:
    """JSON-compatible canonical event dictionary."""
    return {
        "ts_ns": event.ts_ns,
        "sequence": event.sequence,
        "symbol": event.symbol,
        "kind": event.kind.value,
        "side": event.side.value if event.side else None,
        "price_ticks": event.price_ticks,
        "quantity": event.quantity,
        "order_id": event.order_id,
        "reference_price_ticks": event.reference_price_ticks,
    }


def event_from_dict(row: Mapping[str, Any]) -> MarketEvent:
    """Strictly parse a canonical dictionary; no unit inference is performed."""
    side = Side(row["side"]) if row.get("side") is not None else None
    return MarketEvent(int(row["ts_ns"]), int(row["sequence"]), str(row["symbol"]),
                       EventKind(row["kind"]), side,
                       int(row["price_ticks"]) if row.get("price_ticks") is not None else None,
                       int(row["quantity"]) if row.get("quantity") is not None else None,
                       str(row.get("order_id", "")),
                       int(row["reference_price_ticks"]) if row.get("reference_price_ticks") is not None else None)
