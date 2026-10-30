"""Point-in-time data controls for systematic research.

All times are non-negative integer nanoseconds. A feature becomes usable at
``available_ts_ns`` (publication/ingestion time), not merely its economic
observation date. Missing or stale matches fail closed by default.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


class DataIntegrityError(ValueError):
    """Raised when timestamps, keys, or point-in-time constraints are invalid."""


@dataclass(frozen=True, slots=True)
class PointInTimeMatch:
    """A left decision row and its latest eligible feature observation."""

    decision: Mapping[str, Any]
    observation: Mapping[str, Any] | None
    age_ns: int | None


@dataclass(frozen=True, slots=True)
class UniverseInterval:
    """A membership interval, including when that fact became knowable.

    ``valid_from_ns`` is inclusive and ``valid_to_ns`` is exclusive. The row
    may affect a decision only at or after ``known_from_ns``. This distinction
    prevents retrospective constituent lists from leaking into earlier dates.
    """

    instrument: str
    valid_from_ns: int
    valid_to_ns: int | None
    known_from_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise DataIntegrityError("instrument must be a non-empty identifier")
        _timestamp(self.valid_from_ns, "valid_from_ns")
        _timestamp(self.known_from_ns, "known_from_ns")
        if self.valid_to_ns is not None:
            _timestamp(self.valid_to_ns, "valid_to_ns")
            if self.valid_to_ns <= self.valid_from_ns:
                raise DataIntegrityError("valid_to_ns must be later than valid_from_ns")


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataIntegrityError(f"{name} must be a non-negative integer nanosecond timestamp")
    return value


def _key(row: Mapping[str, Any], fields: Sequence[str], row_number: int) -> tuple[Any, ...]:
    try:
        values = tuple(row[field] for field in fields)
    except KeyError as exc:
        raise DataIntegrityError(f"row {row_number} missing key field {exc.args[0]!r}") from exc
    if any(value is None for value in values):
        raise DataIntegrityError(f"row {row_number} has a null join key")
    try:
        hash(values)
    except TypeError as exc:
        raise DataIntegrityError(f"row {row_number} has an unhashable join key") from exc
    return values


def point_in_time_join(
    decisions: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str] = ("symbol",),
    decision_time: str = "decision_ts_ns",
    available_time: str = "available_ts_ns",
    max_age_ns: int | None = None,
    require_match: bool = True,
) -> list[PointInTimeMatch]:
    """Backward as-of join with a strict availability-time barrier.

    For each decision, selects the latest right row with matching keys and
    ``available_time <= decision_time``. Ties select the last row in input
    order, giving deterministic precedence to the final same-time correction.
    If a matched row exceeds ``max_age_ns``, it is treated as unavailable.
    With ``require_match=False``, missing/stale matches are returned as
    ``observation=None``; values are never forward-filled from the future.
    """
    if isinstance(key_fields, str):
        raise DataIntegrityError("key_fields must be a sequence of field names, not one string")
    fields = tuple(key_fields)
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise DataIntegrityError("at least one key field is required")
    if max_age_ns is not None and (
        isinstance(max_age_ns, bool) or not isinstance(max_age_ns, int) or max_age_ns < 0
    ):
        raise DataIntegrityError("max_age_ns must be a non-negative integer or None")
    left = list(decisions)
    right = list(observations)
    grouped: dict[tuple[Any, ...], list[tuple[int, int, Mapping[str, Any]]]] = {}
    for i, row in enumerate(right):
        if not isinstance(row, Mapping):
            raise DataIntegrityError(f"observation {i} must be a mapping")
        key = _key(row, fields, i)
        if available_time not in row:
            raise DataIntegrityError(f"observation {i} missing {available_time!r}")
        ts = _timestamp(row[available_time], f"observation[{i}].{available_time}")
        grouped.setdefault(key, []).append((ts, i, dict(row)))
    for rows in grouped.values():
        rows.sort(key=lambda item: (item[0], item[1]))

    result: list[PointInTimeMatch] = []
    for i, row in enumerate(left):
        if not isinstance(row, Mapping):
            raise DataIntegrityError(f"decision {i} must be a mapping")
        key = _key(row, fields, i)
        if decision_time not in row:
            raise DataIntegrityError(f"decision {i} missing {decision_time!r}")
        asof = _timestamp(row[decision_time], f"decision[{i}].{decision_time}")
        candidates = grouped.get(key, [])
        times = [entry[0] for entry in candidates]
        pos = bisect_right(times, asof) - 1
        observation = candidates[pos][2] if pos >= 0 else None
        age = asof - candidates[pos][0] if pos >= 0 else None
        if age is not None and max_age_ns is not None and age > max_age_ns:
            observation, age = None, None
        if observation is None and require_match:
            raise DataIntegrityError(
                f"no fresh point-in-time match for decision {i} at {asof}"
            )
        result.append(PointInTimeMatch(dict(row), observation, age))
    return result


def assert_available_by(available_ts_ns: int, decision_ts_ns: int) -> None:
    """Raise if a feature was not yet available at the decision timestamp."""
    available = _timestamp(available_ts_ns, "available_ts_ns")
    decision = _timestamp(decision_ts_ns, "decision_ts_ns")
    if available > decision:
        raise DataIntegrityError(
            f"future information: available at {available}, decision at {decision}"
        )


def members_at(intervals: Iterable[UniverseInterval], decision_ts_ns: int) -> tuple[str, ...]:
    """Return instruments valid and already known at the decision time."""
    decision = _timestamp(decision_ts_ns, "decision_ts_ns")
    members: set[str] = set()
    for item in intervals:
        if not isinstance(item, UniverseInterval):
            raise DataIntegrityError("intervals must contain UniverseInterval records")
        known = item.known_from_ns <= decision
        valid = item.valid_from_ns <= decision and (
            item.valid_to_ns is None or decision < item.valid_to_ns
        )
        if known and valid:
            members.add(item.instrument)
    return tuple(sorted(members))


def delisting_return(last_mark: float, recovery_value: float) -> float:
    """Compute explicit terminal return from the last mark to recovery value.

    Recovery value is cash or equivalent per same unit. It must be supplied
    explicitly; missing recovery is not silently converted into zero or a
    guessed return. Corporate-action adjustments remain the caller's duty.
    """
    if isinstance(last_mark, bool) or isinstance(recovery_value, bool):
        raise DataIntegrityError("delisting values must be numeric, not boolean")
    try:
        last_mark = float(last_mark)
        recovery_value = float(recovery_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataIntegrityError("delisting values must be numeric") from exc
    if not isfinite(last_mark) or last_mark <= 0:
        raise DataIntegrityError("last_mark must be finite and positive")
    if not isfinite(recovery_value) or recovery_value < 0:
        raise DataIntegrityError("recovery_value must be finite and non-negative")
    return recovery_value / last_mark - 1.0
