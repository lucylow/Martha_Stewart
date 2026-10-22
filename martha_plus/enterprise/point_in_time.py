"""Point-in-time joins and feature availability checks to reduce look-ahead bias."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class VersionedValue:
    key: str
    value: Any
    valid_from: datetime
    published_at: datetime
    valid_to: datetime | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key is required")
        if self.valid_from.tzinfo is None or self.published_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.valid_to is not None:
            if self.valid_to.tzinfo is None or self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be aware and later than valid_from")

    def available_at(self, as_of: datetime) -> bool:
        return self.published_at <= as_of and self.valid_from <= as_of and (
            self.valid_to is None or as_of < self.valid_to)


@dataclass(frozen=True, slots=True)
class JoinResult:
    key: str
    as_of: datetime
    value: Any | None
    matched: bool
    published_at: datetime | None = None
    source: str | None = None
    age_seconds: float | None = None
    reason: str = "matched"


class PointInTimeIndex:
    """Keep immutable revisions and select the latest version known at a cutoff."""
    def __init__(self) -> None:
        self._values: dict[str, list[VersionedValue]] = {}

    def add(self, row: VersionedValue) -> None:
        entries = self._values.setdefault(row.key, [])
        # Conflicting duplicates at same publication instant are dangerous.
        for item in entries:
            if item.published_at == row.published_at and item.valid_from == row.valid_from:
                if item.value != row.value:
                    raise ValueError("conflicting revisions share publication and valid times")
                return
        entries.append(row)
        entries.sort(key=lambda item: (item.published_at, item.valid_from))

    def extend(self, rows: Iterable[VersionedValue]) -> None:
        for row in rows:
            self.add(row)

    def lookup(self, key: str, as_of: datetime, *, max_age: timedelta | None = None) -> JoinResult:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        candidates = [r for r in self._values.get(key, ()) if r.available_at(as_of)]
        if not candidates:
            return JoinResult(key, as_of, None, False, reason="no_point_in_time_match")
        selected = max(candidates, key=lambda r: (r.valid_from, r.published_at))
        age = (as_of - selected.published_at).total_seconds()
        if max_age is not None and age > max_age.total_seconds():
            return JoinResult(key, as_of, None, False, selected.published_at,
                              selected.source, age, "stale_value")
        return JoinResult(key, as_of, selected.value, True, selected.published_at,
                          selected.source, age, "matched")

    def bulk_lookup(self, keys: Iterable[str], as_of: datetime, *,
                    max_age: timedelta | None = None) -> dict[str, JoinResult]:
        return {key: self.lookup(key, as_of, max_age=max_age) for key in keys}

    def history(self, key: str) -> tuple[VersionedValue, ...]:
        return tuple(self._values.get(key, ()))

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def remove_source(self, source: str) -> int:
        removed = 0
        for key in list(self._values):
            before = len(self._values[key])
            self._values[key] = [r for r in self._values[key] if r.source != source]
            removed += before - len(self._values[key])
            if not self._values[key]:
                del self._values[key]
        return removed


def asof_join(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]], *,
              left_time: str, right_time: str, by: str,
              fields: Sequence[str], allow_exact: bool = True,
              tolerance: timedelta | None = None,
              suffix: str = "_pit") -> list[dict[str, Any]]:
    """Backward as-of join keyed by entity, with optional maximum age.

    Right-side rows must be available no later than each left timestamp. This
    explicit backward-only join avoids accidentally leaking future observations.
    """
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in right_rows:
        ts = row[right_time]
        if not isinstance(ts, datetime) or ts.tzinfo is None:
            raise ValueError("right timestamps must be timezone-aware datetimes")
        groups.setdefault(row[by], []).append(row)
    prepared: dict[Any, tuple[list[datetime], list[Mapping[str, Any]]]] = {}
    for entity, values in groups.items():
        ordered = sorted(values, key=lambda r: r[right_time])
        times = [r[right_time] for r in ordered]
        prepared[entity] = times, ordered
    result: list[dict[str, Any]] = []
    for left in left_rows:
        cutoff = left[left_time]
        if not isinstance(cutoff, datetime) or cutoff.tzinfo is None:
            raise ValueError("left timestamps must be timezone-aware datetimes")
        entity = left[by]
        match = None
        if entity in prepared:
            times, rows = prepared[entity]
            idx = bisect_right(times, cutoff) - 1 if allow_exact else bisect_left(times, cutoff) - 1
            if idx >= 0:
                candidate = rows[idx]
                age = cutoff - candidate[right_time]
                if tolerance is None or age <= tolerance:
                    match = candidate
        joined = dict(left)
        for field in fields:
            joined[field + suffix] = match.get(field) if match else None
        joined["_pit_matched"] = match is not None
        joined["_pit_age_seconds"] = (cutoff - match[right_time]).total_seconds() if match else None
        result.append(joined)
    return result


def assert_no_future_features(feature_times: Sequence[datetime], decision_times: Sequence[datetime]) -> None:
    if len(feature_times) != len(decision_times):
        raise ValueError("feature and decision time sequences must have equal lengths")
    for index, (feature, decision) in enumerate(zip(feature_times, decision_times)):
        if feature.tzinfo is None or decision.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if feature > decision:
            raise ValueError(f"future leakage at row {index}: feature={feature}, decision={decision}")
