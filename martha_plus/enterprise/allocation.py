"""Constraint-aware allocation utilities and transparent allocation diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AllocationBounds:
    lower: Mapping[str, float] = field(default_factory=dict)
    upper: Mapping[str, float] = field(default_factory=dict)
    sector_max: Mapping[str, float] = field(default_factory=dict)
    max_positions: int | None = None
    long_only: bool = True

    def __post_init__(self) -> None:
        symbols = set(self.lower) | set(self.upper)
        for symbol in symbols:
            lo = self.lower.get(symbol, 0.0 if self.long_only else -1.0)
            hi = self.upper.get(symbol, 1.0)
            if not isfinite(lo) or not isfinite(hi) or lo > hi:
                raise ValueError(f"invalid bounds for {symbol}")
        if any(not 0 <= value <= 1 for value in self.sector_max.values()):
            raise ValueError("sector max weights must be within [0, 1]")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError("max_positions must be positive")


@dataclass(frozen=True, slots=True)
class AllocationDiagnostic:
    feasible: bool
    total_weight: float
    gross_exposure: float
    net_exposure: float
    concentration_hhi: float
    active_positions: int
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AllocationResult:
    weights: Mapping[str, float]
    converged: bool
    iterations: int
    objective: float
    diagnostic: AllocationDiagnostic
    message: str


def diagnose_allocation(weights: Mapping[str, float], *, sectors: Mapping[str, str] | None = None,
                        bounds: AllocationBounds | None = None, target_total: float = 1.0,
                        tolerance: float = 1e-8) -> AllocationDiagnostic:
    sectors = sectors or {}
    violations: list[str] = []
    for symbol, weight in weights.items():
        if not isfinite(weight):
            violations.append(f"{symbol}: non-finite weight")
        if bounds:
            lo = bounds.lower.get(symbol, 0.0 if bounds.long_only else -1.0)
            hi = bounds.upper.get(symbol, 1.0)
            if weight < lo - tolerance or weight > hi + tolerance:
                violations.append(f"{symbol}: weight outside [{lo}, {hi}]")
            if bounds.long_only and weight < -tolerance:
                violations.append(f"{symbol}: negative weight in long-only allocation")
    total = sum(weights.values())
    gross = sum(abs(w) for w in weights.values())
    active = sum(abs(w) > tolerance for w in weights.values())
    if abs(total - target_total) > tolerance:
        violations.append(f"total weight {total:.8f} differs from target {target_total:.8f}")
    if bounds and bounds.max_positions is not None and active > bounds.max_positions:
        violations.append(f"active positions {active} exceeds max {bounds.max_positions}")
    if bounds:
        exposure: dict[str, float] = {}
        for symbol, weight in weights.items():
            exposure[sectors.get(symbol, "unknown")] = exposure.get(sectors.get(symbol, "unknown"), 0.0) + weight
        for sector, max_weight in bounds.sector_max.items():
            if exposure.get(sector, 0.0) > max_weight + tolerance:
                violations.append(f"sector {sector} weight exceeds {max_weight}")
    positive = [abs(w) / gross for w in weights.values() if abs(w) > tolerance] if gross else []
    hhi = sum(w * w for w in positive)
    return AllocationDiagnostic(not violations, total, gross, total, hhi, active, tuple(violations))


def project_to_simplex(values: Mapping[str, float], *, total: float = 1.0,
                       lower: Mapping[str, float] | None = None,
                       upper: Mapping[str, float] | None = None) -> dict[str, float]:
    """Project a vector onto box constraints plus a fixed sum using bisection.

    This solves x_i = clip(v_i - lambda, lower_i, upper_i) and finds lambda.
    It supports per-asset bounds but not sector constraints or cardinality limits.
    """
    if not values or not isfinite(total):
        raise ValueError("values must be non-empty and total finite")
    lower, upper = lower or {}, upper or {}
    keys = tuple(values)
    v = {k: float(values[k]) for k in keys}
    lo = {k: float(lower.get(k, 0.0)) for k in keys}
    hi = {k: float(upper.get(k, 1.0)) for k in keys}
    if any(not isfinite(v[k]) or not isfinite(lo[k]) or not isfinite(hi[k]) or lo[k] > hi[k] for k in keys):
        raise ValueError("invalid values or bounds")
    if sum(lo.values()) > total + 1e-10 or sum(hi.values()) < total - 1e-10:
        raise ValueError("sum target is infeasible under bounds")
    left = min(v[k] - hi[k] for k in keys) - abs(total) - 1.0
    right = max(v[k] - lo[k] for k in keys) + abs(total) + 1.0
    for _ in range(200):
        mid = (left + right) / 2
        candidate = {k: min(hi[k], max(lo[k], v[k] - mid)) for k in keys}
        current = sum(candidate.values())
        if abs(current - total) <= 1e-12:
            return candidate
        if current > total:
            left = mid
        else:
            right = mid
    mid = (left + right) / 2
    projected = {k: min(hi[k], max(lo[k], v[k] - mid)) for k in keys}
    # Correct only tiny numeric residue, using available slack deterministically.
    delta = total - sum(projected.values())
    for key in keys:
        if abs(delta) < 1e-10:
            break
        slack = (hi[key] - projected[key]) if delta > 0 else (projected[key] - lo[key])
        adjustment = (1 if delta > 0 else -1) * min(abs(delta), max(0.0, slack))
        projected[key] += adjustment
        delta -= adjustment
    if abs(sum(projected.values()) - total) > 1e-7:
        raise RuntimeError("simplex projection did not converge to requested total")
    return projected


def inverse_volatility_weights(volatility: Mapping[str, float], *, target_total: float = 1.0,
                               floor: float = 1e-6, caps: Mapping[str, float] | None = None) -> dict[str, float]:
    if not volatility or floor <= 0 or not isfinite(target_total):
        raise ValueError("volatility map, positive floor and finite total required")
    raw = {}
    for symbol, vol in volatility.items():
        if not isfinite(vol) or vol < 0:
            raise ValueError("volatilities must be finite and non-negative")
        raw[symbol] = 1.0 / max(vol, floor)
    total = sum(raw.values())
    weights = {k: target_total * value / total for k, value in raw.items()}
    if caps:
        weights = project_to_simplex(weights, total=target_total,
                                     lower={k: 0.0 for k in weights}, upper=caps)
    return weights


def equal_risk_budget(covariance: Sequence[Sequence[float]], *, iterations: int = 5_000,
                      learning_rate: float = 0.01, tolerance: float = 1e-8) -> dict[str, float]:
    """Approximate long-only equal risk-contribution weights from a covariance matrix."""
    n = len(covariance)
    if n == 0 or any(len(row) != n for row in covariance):
        raise ValueError("covariance must be a non-empty square matrix")
    if any(not isfinite(v) for row in covariance for v in row):
        raise ValueError("covariance entries must be finite")
    for i in range(n):
        if covariance[i][i] < 0:
            raise ValueError("variance cannot be negative")
        for j in range(n):
            if abs(covariance[i][j] - covariance[j][i]) > 1e-8:
                raise ValueError("covariance matrix must be symmetric")
    weights = [1 / n] * n
    for _ in range(iterations):
        marginal = [sum(covariance[i][j] * weights[j] for j in range(n)) for i in range(n)]
        variance = sum(weights[i] * marginal[i] for i in range(n))
        if variance <= 1e-18:
            return {str(i): 1 / n for i in range(n)}
        risk = [weights[i] * marginal[i] / variance for i in range(n)]
        error = [risk[i] - 1 / n for i in range(n)]
        if max(abs(x) for x in error) <= tolerance:
            break
        adjusted = [max(1e-8, weights[i] * (1 - learning_rate * error[i] * n)) for i in range(n)]
        total = sum(adjusted)
        weights = [value / total for value in adjusted]
    return {str(i): value for i, value in enumerate(weights)}


def round_lot_quantities(target_weights: Mapping[str, float], prices: Mapping[str, float],
                         equity: float, *, lot_sizes: Mapping[str, int] | None = None) -> dict[str, int]:
    if equity < 0 or not isfinite(equity):
        raise ValueError("equity must be finite and non-negative")
    result = {}
    for symbol, weight in target_weights.items():
        price = prices.get(symbol)
        if price is None or not isfinite(price) or price <= 0:
            raise ValueError(f"missing or invalid price for {symbol}")
        if not isfinite(weight):
            raise ValueError("weights must be finite")
        lot = (lot_sizes or {}).get(symbol, 1)
        if lot < 1:
            raise ValueError("lot sizes must be positive integers")
        shares = int((equity * weight / price) // lot) * lot
        result[symbol] = shares
    return result
