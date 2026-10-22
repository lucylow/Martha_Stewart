"""Strict, dependency-free data contracts for time-series and portfolio inputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


class ContractViolation(ValueError):
    """A row or dataset violates an explicit input contract."""


@dataclass(frozen=True, slots=True)
class FieldRule:
    name: str
    required: bool = True
    kind: type | tuple[type, ...] | None = None
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: frozenset[Any] | None = None

    def validate(self, value: Any, *, row_number: int) -> list[str]:
        errors: list[str] = []
        prefix = f"row {row_number} field {self.name!r}"
        if value is None:
            if not self.nullable:
                errors.append(f"{prefix} may not be null")
            return errors
        if self.kind and not isinstance(value, self.kind):
            errors.append(f"{prefix} must be {self.kind}")
            return errors
        numeric_kind = self.kind in (int, float) or (isinstance(self.kind, tuple) and any(t in (int, float) for t in self.kind))
        if isinstance(value, bool) and numeric_kind:
            errors.append(f"{prefix} may not be boolean")
            return errors
        if isinstance(value, (int, float)):
            if not isfinite(float(value)):
                errors.append(f"{prefix} must be finite")
            if self.minimum is not None and value < self.minimum:
                errors.append(f"{prefix} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                errors.append(f"{prefix} must be <= {self.maximum}")
        if self.allowed is not None and value not in self.allowed:
            errors.append(f"{prefix} must be one of {sorted(map(str, self.allowed))}")
        return errors


@dataclass(frozen=True, slots=True)
class ContractReport:
    valid: bool
    row_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_at: datetime | None = None

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ContractViolation("; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class DataContract:
    name: str
    version: int
    fields: tuple[FieldRule, ...]
    strict_extra_fields: bool = True
    row_predicates: tuple[Any, ...] = field(default_factory=tuple, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractViolation("contract name is required")
        if self.version < 1:
            raise ContractViolation("contract version must be >= 1")
        names = [rule.name for rule in self.fields]
        if len(names) != len(set(names)):
            raise ContractViolation("contract field names must be unique")

    @property
    def field_names(self) -> frozenset[str]:
        return frozenset(rule.name for rule in self.fields)

    def validate_row(self, row: Mapping[str, Any], row_number: int = 1) -> tuple[str, ...]:
        errors: list[str] = []
        for rule in self.fields:
            if rule.name not in row:
                if rule.required:
                    errors.append(f"row {row_number} missing required field {rule.name!r}")
            else:
                errors.extend(rule.validate(row[rule.name], row_number=row_number))
        if self.strict_extra_fields:
            for key in sorted(set(row) - self.field_names):
                errors.append(f"row {row_number} has unexpected field {key!r}")
        for predicate in self.row_predicates:
            try:
                result = predicate(row)
            except Exception as exc:
                errors.append(f"row {row_number} predicate failed: {type(exc).__name__}")
            else:
                if result is False:
                    errors.append(f"row {row_number} failed predicate {getattr(predicate, '__name__', 'custom')}")
                elif isinstance(result, str):
                    errors.append(f"row {row_number}: {result}")
        return tuple(errors)

    def validate(self, rows: Iterable[Mapping[str, Any]], *, max_errors: int = 100) -> ContractReport:
        if max_errors < 1:
            raise ValueError("max_errors must be >= 1")
        errors: list[str] = []
        count = 0
        for count, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                errors.append(f"row {count} must be a mapping")
            else:
                errors.extend(self.validate_row(row, count))
            if len(errors) >= max_errors:
                errors = errors[:max_errors]
                errors.append(f"validation stopped after {max_errors} errors")
                break
        return ContractReport(valid=not errors, row_count=count, errors=tuple(errors))


def _increasing_timestamp(rows: Sequence[Mapping[str, Any]], field_name: str = "timestamp") -> list[str]:
    previous: Any = None
    errors: list[str] = []
    for i, row in enumerate(rows, start=1):
        current = row.get(field_name)
        if previous is not None and current is not None and current <= previous:
            errors.append(f"row {i} timestamp must be strictly increasing")
        previous = current
    return errors


def validate_time_series(rows: Sequence[Mapping[str, Any]], *, timestamp_field: str = "timestamp",
                          unique_key_fields: Sequence[str] = ()) -> ContractReport:
    """Validate chronological order and optional compound-key uniqueness."""
    errors = _increasing_timestamp(rows, timestamp_field)
    seen: set[tuple[Any, ...]] = set()
    for i, row in enumerate(rows, start=1):
        if timestamp_field not in row:
            errors.append(f"row {i} missing {timestamp_field!r}")
        if unique_key_fields:
            key = tuple(row.get(name) for name in unique_key_fields)
            if key in seen:
                errors.append(f"row {i} duplicates key {key!r}")
            seen.add(key)
    return ContractReport(not errors, len(rows), tuple(errors))


def ohlcv_contract() -> DataContract:
    """Return a conservative raw OHLCV bar schema."""
    fields = (
        FieldRule("symbol", kind=str), FieldRule("timestamp"),
        FieldRule("open", kind=(int, float), minimum=0.0000001),
        FieldRule("high", kind=(int, float), minimum=0.0000001),
        FieldRule("low", kind=(int, float), minimum=0.0000001),
        FieldRule("close", kind=(int, float), minimum=0.0000001),
        FieldRule("volume", kind=(int, float), minimum=0),
    )
    def valid_ohlc(row: Mapping[str, Any]) -> bool:
        return (row["low"] <= min(row["open"], row["close"]) and
                row["high"] >= max(row["open"], row["close"]) and
                row["low"] <= row["high"])
    return DataContract("ohlcv", 1, fields, row_predicates=(valid_ohlc,))


def validate_no_future_timestamps(rows: Iterable[Mapping[str, Any]], now: datetime,
                                  field_name: str = "timestamp") -> ContractReport:
    errors: list[str] = []
    count = 0
    for count, row in enumerate(rows, start=1):
        value = row.get(field_name)
        if not isinstance(value, datetime):
            errors.append(f"row {count} {field_name!r} must be datetime")
        elif value.tzinfo is None or now.tzinfo is None:
            errors.append(f"row {count} timestamps must be timezone-aware")
        elif value > now:
            errors.append(f"row {count} timestamp is in the future")
    return ContractReport(not errors, count, tuple(errors))


def profile_numeric_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Summarize numeric fields with explicit missing counts and stable output."""
    field_names = sorted({key for row in rows for key in row})
    profile: dict[str, dict[str, float | int | None]] = {}
    for name in field_names:
        observed = [float(row[name]) for row in rows
                    if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
                    and isfinite(float(row[name]))]
        profile[name] = {
            "count": len(observed), "missing": sum(1 for row in rows if row.get(name) is None),
            "minimum": min(observed) if observed else None,
            "maximum": max(observed) if observed else None,
            "mean": sum(observed) / len(observed) if observed else None,
        }
    return profile
