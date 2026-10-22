"""Composable data-quality checks with severities, quarantining, and summaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence


class RuleSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule_id: str
    passed: bool
    severity: RuleSeverity
    message: str
    row_indices: tuple[int, ...] = ()
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QualityReport:
    dataset: str
    checked_at: str
    row_count: int
    results: tuple[RuleResult, ...]

    @property
    def passed(self) -> bool:
        return not any(not item.passed and item.severity in (RuleSeverity.ERROR, RuleSeverity.FATAL)
                       for item in self.results)

    @property
    def fatal(self) -> bool:
        return any(not item.passed and item.severity is RuleSeverity.FATAL for item in self.results)

    def summary(self) -> dict[str, int | bool]:
        return {"dataset": self.dataset, "row_count": self.row_count, "passed": self.passed,
                "fatal": self.fatal, "rules": len(self.results),
                "failed": sum(not r.passed for r in self.results),
                **{severity.value: sum(not r.passed and r.severity is severity for r in self.results)
                   for severity in RuleSeverity}}


@dataclass(frozen=True, slots=True)
class QualityRule:
    rule_id: str
    severity: RuleSeverity
    evaluate: Callable[[Sequence[Mapping[str, Any]]], RuleResult]

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not callable(self.evaluate):
            raise ValueError("rule_id and callable evaluator are required")


class QualityRuleBook:
    def __init__(self, rules: Iterable[QualityRule] = (), *,
                 clock=lambda: datetime.now(timezone.utc)) -> None:
        self.clock = clock
        self._rules: dict[str, QualityRule] = {}
        for rule in rules:
            self.add(rule)

    def add(self, rule: QualityRule, *, replace: bool = False) -> None:
        if rule.rule_id in self._rules and not replace:
            raise ValueError(f"duplicate rule: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def remove(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    def run(self, dataset: str, rows: Sequence[Mapping[str, Any]]) -> QualityReport:
        if not dataset.strip():
            raise ValueError("dataset name is required")
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        results = []
        for rule in self._rules.values():
            try:
                result = rule.evaluate(rows)
                if result.rule_id != rule.rule_id:
                    raise ValueError("rule result id differs from registered rule id")
                results.append(result)
            except Exception as exc:
                results.append(RuleResult(rule.rule_id, False, RuleSeverity.FATAL,
                    f"rule evaluator raised {type(exc).__name__}: {exc}"))
        return QualityReport(dataset, now.isoformat(), len(rows), tuple(results))


def required_fields(fields: Iterable[str], *, rule_id: str = "required_fields",
                    severity: RuleSeverity = RuleSeverity.ERROR) -> QualityRule:
    required = tuple(fields)
    if not required:
        raise ValueError("at least one field is required")
    def evaluate(rows: Sequence[Mapping[str, Any]]) -> RuleResult:
        failures = []
        for idx, row in enumerate(rows):
            missing = [field for field in required if field not in row or row[field] is None]
            if missing:
                failures.append(idx)
        return RuleResult(rule_id, not failures, severity,
            "all required fields present" if not failures else "rows missing required fields",
            tuple(failures), {"required_field_count": len(required), "failed_rows": len(failures)})
    return QualityRule(rule_id, severity, evaluate)


def numeric_range(field: str, *, minimum: float | None = None, maximum: float | None = None,
                  rule_id: str | None = None, severity: RuleSeverity = RuleSeverity.ERROR) -> QualityRule:
    if minimum is None and maximum is None:
        raise ValueError("at least one bound is required")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    ident = rule_id or f"range_{field}"
    def evaluate(rows: Sequence[Mapping[str, Any]]) -> RuleResult:
        failures = []
        values = []
        for idx, row in enumerate(rows):
            raw = row.get(field)
            try:
                number = float(raw)
            except (TypeError, ValueError):
                failures.append(idx)
                continue
            if not isfinite(number):
                failures.append(idx)
                continue
            values.append(number)
            if minimum is not None and number < minimum or maximum is not None and number > maximum:
                failures.append(idx)
        return RuleResult(ident, not failures, severity,
            f"{field} within configured range" if not failures else f"{field} has invalid/out-of-range values",
            tuple(sorted(set(failures))), {"observed": len(values), "failures": len(set(failures))})
    return QualityRule(ident, severity, evaluate)


def uniqueness(fields: Sequence[str], *, rule_id: str = "unique_keys",
               severity: RuleSeverity = RuleSeverity.ERROR) -> QualityRule:
    if not fields:
        raise ValueError("uniqueness requires at least one field")
    def evaluate(rows: Sequence[Mapping[str, Any]]) -> RuleResult:
        seen = {}
        failures = set()
        for idx, row in enumerate(rows):
            key = tuple(row.get(field) for field in fields)
            try:
                prior = seen.get(key)
            except TypeError as exc:
                raise ValueError("uniqueness key values must be hashable") from exc
            if prior is not None:
                failures.update((prior, idx))
            else:
                seen[key] = idx
        return RuleResult(rule_id, not failures, severity,
            "keys are unique" if not failures else "duplicate keys found",
            tuple(sorted(failures)), {"key_fields": len(fields), "duplicate_rows": len(failures)})
    return QualityRule(rule_id, severity, evaluate)


def monotonic_datetime(field: str, *, rule_id: str = "monotonic_time",
                       severity: RuleSeverity = RuleSeverity.ERROR) -> QualityRule:
    def evaluate(rows: Sequence[Mapping[str, Any]]) -> RuleResult:
        failures = []
        prior = None
        for idx, row in enumerate(rows):
            value = row.get(field)
            if not isinstance(value, datetime) or value.tzinfo is None:
                failures.append(idx)
                continue
            if prior is not None and value < prior:
                failures.append(idx)
            prior = value
        return RuleResult(rule_id, not failures, severity,
            "timestamps are monotonic and timezone-aware" if not failures else "invalid or out-of-order timestamps",
            tuple(failures), {"rows_checked": len(rows), "failures": len(failures)})
    return QualityRule(rule_id, severity, evaluate)


def no_stale_values(field: str, timestamp_field: str, *, max_age_seconds: float,
                    now: datetime, rule_id: str = "freshness",
                    severity: RuleSeverity = RuleSeverity.WARNING) -> QualityRule:
    if max_age_seconds < 0 or now.tzinfo is None:
        raise ValueError("max age must be non-negative and now timezone-aware")
    def evaluate(rows: Sequence[Mapping[str, Any]]) -> RuleResult:
        failures, ages = [], []
        for idx, row in enumerate(rows):
            if row.get(field) is None:
                failures.append(idx)
                continue
            ts = row.get(timestamp_field)
            if not isinstance(ts, datetime) or ts.tzinfo is None:
                failures.append(idx)
                continue
            age = (now - ts).total_seconds()
            ages.append(age)
            if age < 0 or age > max_age_seconds:
                failures.append(idx)
        return RuleResult(rule_id, not failures, severity,
            "values within freshness limit" if not failures else "missing, future, or stale values found",
            tuple(failures), {"max_age_seconds": max_age_seconds,
                              "median_age_seconds": median(ages) if ages else 0.0})
    return QualityRule(rule_id, severity, evaluate)


def quarantine_indices(report: QualityReport, *, severities: Iterable[RuleSeverity] =
                       (RuleSeverity.ERROR, RuleSeverity.FATAL)) -> tuple[int, ...]:
    selected = set(severities)
    return tuple(sorted({idx for result in report.results if not result.passed and result.severity in selected
                         for idx in result.row_indices}))
