"""Portfolio weight constraint validation and deterministic projection."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    total_weight: float = 1.0
    max_gross_exposure: float = 1.0
    max_single_name: float = 0.25
    min_single_name: float = 0.0
    long_only: bool = True
    max_turnover: float | None = None
    min_trade_weight: float = 0.0
    group_by_symbol: Mapping[str, str] = field(default_factory=dict)
    max_group_weight: Mapping[str, float] = field(default_factory=dict)
    banned_symbols: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0 <= self.total_weight <= self.max_gross_exposure <= 2.0:
            raise ValueError("require 0 <= total_weight <= max_gross_exposure <= 2")
        if not 0 < self.max_single_name <= 1:
            raise ValueError("max_single_name must be in (0, 1]")
        if not 0 <= self.min_single_name <= self.max_single_name:
            raise ValueError("min_single_name must be between zero and max_single_name")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative")
        if self.min_trade_weight < 0:
            raise ValueError("min_trade_weight must be non-negative")
        if any(not 0 <= value <= self.max_gross_exposure for value in self.max_group_weight.values()):
            raise ValueError("group caps must be within gross exposure")


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    feasible: bool
    violations: tuple[str, ...]
    net_exposure: float
    gross_exposure: float
    turnover: float
    group_exposures: Mapping[str, float]


class ConstraintProjector:
    """Project desired weights onto a practical long-only or bounded-short set.

    The method prioritizes hard caps and approximate feasibility. It is not a
    general-purpose convex optimizer; callers should inspect `report` after projection.
    """
    def __init__(self, constraints: AllocationConstraints | None = None) -> None:
        self.constraints = constraints or AllocationConstraints()

    def validate(self, weights: Mapping[str, float], previous: Mapping[str, float] | None = None) -> ConstraintReport:
        current = self._normalize_symbols(weights)
        old = self._normalize_symbols(previous or {})
        c = self.constraints
        violations: list[str] = []
        if any(not isfinite(v) for v in current.values()):
            violations.append("weights must be finite")
        if c.long_only and any(v < -1e-10 for v in current.values()):
            violations.append("negative weights disallowed by long_only")
        if any(abs(v) > c.max_single_name + 1e-10 for v in current.values()):
            violations.append("single-name cap exceeded")
        gross = sum(abs(v) for v in current.values())
        net = sum(current.values())
        if gross > c.max_gross_exposure + 1e-8:
            violations.append("gross exposure cap exceeded")
        if abs(net - c.total_weight) > 1e-6:
            violations.append("net exposure does not equal configured target")
        groups = self._group_exposure(current)
        for group, cap in c.max_group_weight.items():
            if groups.get(group, 0.0) > cap + 1e-8:
                violations.append(f"group cap exceeded: {group}")
        turnover = sum(abs(current.get(s, 0.0) - old.get(s, 0.0)) for s in set(current) | set(old))
        if c.max_turnover is not None and turnover > c.max_turnover + 1e-8:
            violations.append("turnover cap exceeded")
        if set(current) & c.banned_symbols:
            violations.append("banned symbol has non-zero weight")
        return ConstraintReport(not violations, tuple(violations), net, gross, turnover, groups)

    def project(self, desired: Mapping[str, float], previous: Mapping[str, float] | None = None,
                *, iterations: int = 50) -> dict[str, float]:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        c = self.constraints
        values = self._normalize_symbols(desired)
        symbols = sorted(set(values) | set(previous or {}))
        if not symbols:
            if c.total_weight:
                raise ValueError("cannot project empty universe to non-zero net exposure")
            return {}
        if not c.long_only:
            raise NotImplementedError("short-enabled projection is validate-only in this release")
        target = c.total_weight
        for s in symbols:
            values.setdefault(s, 0.0)
            if s in c.banned_symbols:
                values[s] = 0.0
            values[s] = max(0.0, min(c.max_single_name, values[s]))
        for _ in range(iterations):
            self._scale_to_total(values, target)
            self._apply_group_caps(values)
            if previous is not None and c.max_turnover is not None:
                values = self._apply_turnover_cap(values, previous, c.max_turnover)
            for symbol in symbols:
                if symbol in c.banned_symbols or abs(values[symbol]) < c.min_trade_weight:
                    values[symbol] = 0.0
                values[symbol] = max(0.0, min(c.max_single_name, values[symbol]))
        result = {s: w for s, w in sorted(values.items()) if w > 1e-12}
        report = self.validate(result, previous)
        if not report.feasible:
            raise ValueError("constraints are infeasible after projection: " + "; ".join(report.violations))
        return result

    @staticmethod
    def _normalize_symbols(weights: Mapping[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol, raw in weights.items():
            name = str(symbol).strip().upper()
            if not name:
                raise ValueError("symbol may not be blank")
            if name in result:
                raise ValueError(f"duplicate normalized symbol: {name}")
            result[name] = float(raw)
        return result

    def _group_exposure(self, weights: Mapping[str, float]) -> dict[str, float]:
        groups: dict[str, float] = {}
        for symbol, weight in weights.items():
            group_map = {key.strip().upper(): value for key, value in self.constraints.group_by_symbol.items()}
            group = group_map.get(symbol)
            if group:
                groups[group] = groups.get(group, 0.0) + abs(weight)
        return groups

    @staticmethod
    def _scale_to_total(values: dict[str, float], target: float) -> None:
        active = [s for s, w in values.items() if w > 0]
        if not active:
            if target > 0:
                raise ValueError("all names are capped/banned; target exposure cannot be reached")
            return
        # Water-filling respecting each symbol's max cap is done in caller loops.
        total = sum(values.values())
        if total > 0:
            scale = target / total
            for s in values:
                values[s] *= scale

    def _apply_group_caps(self, values: dict[str, float]) -> None:
        for group, cap in self.constraints.max_group_weight.items():
            group_map = {key.strip().upper(): value for key, value in self.constraints.group_by_symbol.items()}
            members = [s for s in values if group_map.get(s) == group]
            total = sum(abs(values[s]) for s in members)
            if total > cap and total > 0:
                scale = cap / total
                for s in members:
                    values[s] *= scale

    @staticmethod
    def _apply_turnover_cap(values: dict[str, float], previous: Mapping[str, float], cap: float) -> dict[str, float]:
        symbols = set(values) | set(previous)
        turnover = sum(abs(values.get(s, 0) - float(previous.get(s, 0))) for s in symbols)
        if turnover <= cap or turnover <= 0:
            return values
        scale = cap / turnover
        return {s: float(previous.get(s, 0)) + (values.get(s, 0) - float(previous.get(s, 0))) * scale for s in symbols}
