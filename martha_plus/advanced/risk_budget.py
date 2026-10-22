"""Risk budgeting and liquidity-aware sizing helpers."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RiskBudgetResult:
    weights: Mapping[str, float]
    risk_contributions: Mapping[str, float]
    portfolio_volatility: float
    converged: bool
    iterations: int


def risk_contributions(weights: Mapping[str, float],
                       covariance: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    symbols = sorted(weights)
    w = [float(weights[s]) for s in symbols]
    cov_w: list[float] = []
    for a in symbols:
        row = covariance.get(a)
        if row is None:
            raise ValueError(f"missing covariance row {a}")
        cov_w.append(sum(float(row[b]) * weights[b] for b in symbols if b in row))
        if any(b not in row for b in symbols):
            raise ValueError(f"incomplete covariance row {a}")
    variance = sum(wi * cwi for wi, cwi in zip(w, cov_w))
    if variance <= 0 or not isfinite(variance):
        raise ValueError("portfolio variance must be positive and finite")
    vol = sqrt(variance)
    # Signed marginal contributions sum to portfolio volatility (before weight normalization).
    return {symbol: w[i] * cov_w[i] / vol for i, symbol in enumerate(symbols)}


def inverse_volatility_weights(volatilities: Mapping[str, float], total: float = 1.0,
                               floor: float = 1e-4, cap: float = 1.0) -> dict[str, float]:
    if not volatilities or not 0 < total <= 1 or floor <= 0 or not 0 < cap <= 1:
        raise ValueError("invalid inputs for inverse-volatility weights")
    if cap * len(volatilities) < total - 1e-12:
        raise ValueError("cap is infeasible for requested total")
    scores = {s.strip().upper(): 1.0 / max(float(v), floor) for s, v in volatilities.items()}
    if any(not isfinite(v) or v < 0 for v in volatilities.values()):
        raise ValueError("volatilities must be finite and non-negative")
    if len(scores) != len(volatilities):
        raise ValueError("symbols collide after normalization")
    result = {s: 0.0 for s in scores}
    remaining = set(scores)
    left = total
    for _ in range(len(scores) + 1):
        denom = sum(scores[s] for s in remaining)
        if not remaining or denom <= 0 or left <= 1e-12:
            break
        allocations = {s: left * scores[s] / denom for s in remaining}
        capped = [s for s, w in allocations.items() if w > cap]
        if not capped:
            for s, w in allocations.items():
                result[s] += w
            left = 0
            break
        for s in capped:
            result[s] += cap
            left -= cap
            remaining.remove(s)
    if left > 1e-7:
        raise ValueError("could not satisfy allocation caps")
    return result


def risk_parity_weights(covariance: Mapping[str, Mapping[str, float]],
                        budgets: Mapping[str, float] | None = None,
                        *, max_iterations: int = 2000, tolerance: float = 1e-8) -> RiskBudgetResult:
    symbols = sorted(covariance)
    if not symbols:
        raise ValueError("covariance matrix must not be empty")
    for s in symbols:
        if any(t not in covariance[s] for t in symbols):
            raise ValueError("covariance matrix is incomplete")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid convergence settings")
    budget = {s: float((budgets or {}).get(s, 1.0)) for s in symbols}
    if any(not isfinite(v) or v <= 0 for v in budget.values()):
        raise ValueError("risk budgets must be finite and positive")
    total_budget = sum(budget.values())
    budget = {s: v / total_budget for s, v in budget.items()}
    matrix = [[float(covariance[a][b]) for b in symbols] for a in symbols]
    if any(not isfinite(v) for row in matrix for v in row):
        raise ValueError("covariance must be finite")
    if any(matrix[i][i] <= 0 for i in range(len(symbols))):
        raise ValueError("variance diagonal must be positive")
    weights = [1 / sqrt(matrix[i][i]) for i in range(len(symbols))]
    total = sum(weights)
    weights = [v / total for v in weights]
    converged = False
    for iteration in range(1, max_iterations + 1):
        variance = sum(weights[i] * sum(matrix[i][j] * weights[j] for j in range(len(symbols)))
                       for i in range(len(symbols)))
        if variance <= 0:
            raise ValueError("covariance produced non-positive variance")
        sigma = sqrt(variance)
        contributions = [weights[i] * sum(matrix[i][j] * weights[j] for j in range(len(symbols))) / sigma
                         for i in range(len(symbols))]
        target = [budget[s] * sigma for s in symbols]
        relative_errors = [abs(contributions[i] - target[i]) / max(sigma, 1e-12) for i in range(len(symbols))]
        if max(relative_errors, default=0.0) <= tolerance:
            converged = True
            break
        # Multiplicative update remains positive and is renormalized each iteration.
        for i in range(len(symbols)):
            if contributions[i] > 1e-12:
                weights[i] *= (target[i] / contributions[i]) ** 0.5
        total = sum(weights)
        weights = [w / total for w in weights]
    final_variance = sum(weights[i] * sum(matrix[i][j] * weights[j] for j in range(len(symbols)))
                          for i in range(len(symbols)))
    vol = sqrt(max(0.0, final_variance))
    contributions = {s: weights[i] * sum(matrix[i][j] * weights[j] for j in range(len(symbols))) / vol
                     for i, s in enumerate(symbols)} if vol else {s: 0.0 for s in symbols}
    return RiskBudgetResult(dict(zip(symbols, weights)), contributions, vol, converged, iteration)


def liquidity_capped_quantity(*, equity: float, risk_fraction: float, stop_distance: float,
                              price: float, average_daily_volume: float,
                              max_adv_fraction: float = 0.05, lot_size: int = 1) -> int:
    """Return floor-rounded long quantity capped by risk and participation limits."""
    values = (equity, risk_fraction, stop_distance, price, average_daily_volume, max_adv_fraction)
    if any(not isfinite(float(v)) for v in values):
        raise ValueError("sizing inputs must be finite")
    if equity <= 0 or not 0 < risk_fraction < 1 or stop_distance <= 0 or price <= 0:
        raise ValueError("invalid risk sizing inputs")
    if average_daily_volume < 0 or not 0 < max_adv_fraction <= 1 or lot_size < 1:
        raise ValueError("invalid liquidity sizing inputs")
    by_risk = int(equity * risk_fraction / stop_distance)
    by_adv = int(average_daily_volume * max_adv_fraction)
    max_qty = min(by_risk, by_adv)
    return (max_qty // lot_size) * lot_size
