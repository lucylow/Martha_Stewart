"""Small-matrix portfolio optimization with explicit convergence diagnostics.

The optimizer uses projected gradient descent and only standard-library math.
For high-dimensional or production allocation workloads, validate against an
independent numerical implementation before deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    weights: Mapping[str, float]
    objective: float
    converged: bool
    iterations: int
    constraint_violation: float
    warnings: tuple[str, ...] = ()


def _project_simplex(values: Sequence[float], total: float = 1.0) -> list[float]:
    if total < 0 or not isfinite(total):
        raise ValueError("simplex total must be finite and non-negative")
    if not values:
        return []
    ordered = sorted(values, reverse=True)
    cumulative = 0.0
    theta = 0.0
    for j, value in enumerate(ordered, start=1):
        cumulative += value
        candidate = (cumulative - total) / j
        if j == len(ordered) or ordered[j] <= candidate:
            theta = candidate
            break
    projected = [max(0.0, value - theta) for value in values]
    sum_projected = sum(projected)
    if sum_projected <= 0:
        return [total / len(values)] * len(values)
    return [x * total / sum_projected for x in projected]


def _matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(a) * float(b) for a, b in zip(row, vector)) for row in matrix]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def validate_covariance(symbols: Sequence[str], covariance: Mapping[str, Mapping[str, float]],
                        tolerance: float = 1e-8) -> list[list[float]]:
    matrix: list[list[float]] = []
    for a in symbols:
        row: list[float] = []
        for b in symbols:
            if a not in covariance or b not in covariance[a]:
                raise ValueError(f"missing covariance {a}/{b}")
            value = float(covariance[a][b])
            if not isfinite(value):
                raise ValueError("covariance values must be finite")
            if a == b and value < -tolerance:
                raise ValueError("covariance diagonal must be non-negative")
            if b in covariance and a in covariance[b] and abs(value - float(covariance[b][a])) > tolerance:
                raise ValueError(f"covariance must be symmetric for {a}/{b}")
            row.append(value)
        matrix.append(row)
    return matrix


class PortfolioOptimizer:
    """Long-only mean-variance optimizer with turnover-aware regularization."""
    def __init__(self, *, risk_aversion: float = 4.0, max_weight: float = 0.35,
                 max_iterations: int = 5000, tolerance: float = 1e-9,
                 turnover_penalty: float = 0.0) -> None:
        if risk_aversion < 0 or not isfinite(risk_aversion):
            raise ValueError("risk_aversion must be finite and non-negative")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if max_iterations < 1 or tolerance <= 0 or turnover_penalty < 0:
            raise ValueError("invalid optimizer iteration/tolerance/penalty")
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.turnover_penalty = turnover_penalty

    def optimize(self, expected_returns: Mapping[str, float],
                 covariance: Mapping[str, Mapping[str, float]],
                 previous_weights: Mapping[str, float] | None = None,
                 *, total_weight: float = 1.0) -> OptimizationResult:
        symbols = sorted(expected_returns)
        if not symbols:
            raise ValueError("expected_returns must not be empty")
        if not 0 < total_weight <= 1:
            raise ValueError("total_weight must be in (0, 1]")
        if self.max_weight * len(symbols) + 1e-12 < total_weight:
            raise ValueError("max_weight is too small to satisfy total_weight")
        mu = [float(expected_returns[s]) for s in symbols]
        if any(not isfinite(x) for x in mu):
            raise ValueError("expected returns must be finite")
        sigma = validate_covariance(symbols, covariance)
        previous = [float((previous_weights or {}).get(s, 0.0)) for s in symbols]
        if any(not isfinite(x) or x < 0 for x in previous):
            raise ValueError("previous_weights must be finite and non-negative")
        # Conservative step size based on an absolute row-sum bound for the spectral norm.
        spectral_bound = max(sum(abs(v) for v in row) for row in sigma)
        lipschitz = max(2 * self.risk_aversion * spectral_bound + self.turnover_penalty, 1e-9)
        step = 0.8 / lipschitz
        weights = _project_simplex(previous if sum(previous) else [total_weight / len(symbols)] * len(symbols), total_weight)
        weights = self._cap_weights(weights, total_weight)
        converged = False
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            sigma_w = _matrix_vector(sigma, weights)
            gradient = [2 * self.risk_aversion * sigma_w[i] - mu[i] +
                        self.turnover_penalty * (weights[i] - previous[i]) for i in range(len(symbols))]
            candidate = [weights[i] - step * gradient[i] for i in range(len(symbols))]
            candidate = self._cap_weights(_project_simplex(candidate, total_weight), total_weight)
            movement = max(abs(candidate[i] - weights[i]) for i in range(len(symbols)))
            weights = candidate
            if movement <= self.tolerance:
                converged = True
                break
        variance = _dot(weights, _matrix_vector(sigma, weights))
        turnover = sum(abs(weights[i] - previous[i]) for i in range(len(symbols)))
        objective = self.risk_aversion * variance - _dot(mu, weights) + self.turnover_penalty * turnover
        violations = [abs(sum(weights) - total_weight), max(0.0, max(weights) - self.max_weight),
                      max(0.0, -min(weights))]
        warning = () if converged else ("optimizer did not converge within iteration limit",)
        return OptimizationResult(dict(zip(symbols, weights)), objective, converged, iteration,
                                  max(violations), warning)

    def _cap_weights(self, weights: Sequence[float], total: float) -> list[float]:
        values = [max(0.0, min(self.max_weight, float(w))) for w in weights]
        # Redistribute excess/deficit across available capacity until stable.
        for _ in range(len(values) + 2):
            diff = total - sum(values)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                active = [i for i, v in enumerate(values) if v < self.max_weight - 1e-12]
                if not active:
                    break
                increment = diff / len(active)
                for i in active:
                    values[i] = min(self.max_weight, values[i] + increment)
            else:
                active = [i for i, v in enumerate(values) if v > 1e-12]
                if not active:
                    break
                decrement = -diff / len(active)
                for i in active:
                    values[i] = max(0.0, values[i] - decrement)
        return values

    @staticmethod
    def portfolio_volatility(weights: Mapping[str, float],
                             covariance: Mapping[str, Mapping[str, float]]) -> float:
        symbols = sorted(weights)
        matrix = validate_covariance(symbols, covariance)
        vector = [float(weights[s]) for s in symbols]
        variance = _dot(vector, _matrix_vector(matrix, vector))
        if variance < -1e-8:
            raise ValueError("covariance produced negative portfolio variance")
        return sqrt(max(0.0, variance))


def shrink_covariance(covariance: Mapping[str, Mapping[str, float]], shrinkage: float = 0.1
                      ) -> dict[str, dict[str, float]]:
    """Shrink off-diagonal estimates toward zero while keeping diagonals intact."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one")
    symbols = sorted(covariance)
    result: dict[str, dict[str, float]] = {}
    for a in symbols:
        if a not in covariance:
            raise ValueError(f"missing covariance row {a}")
        result[a] = {}
        for b in symbols:
            if b not in covariance[a]:
                raise ValueError(f"missing covariance {a}/{b}")
            value = float(covariance[a][b])
            if not isfinite(value):
                raise ValueError("covariance values must be finite")
            result[a][b] = value if a == b else value * (1.0 - shrinkage)
    return result


def covariance_from_returns(returns: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Sample covariance matrix. Series must share the same aligned observations."""
    symbols = sorted(returns)
    if not symbols:
        return {}
    lengths = {len(returns[s]) for s in symbols}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("return series must have equal length >= 2")
    count = next(iter(lengths))
    means = {s: sum(float(x) for x in returns[s]) / count for s in symbols}
    if any(not isfinite(float(x)) for s in symbols for x in returns[s]):
        raise ValueError("returns must be finite")
    result: dict[str, dict[str, float]] = {}
    for a in symbols:
        result[a] = {}
        for b in symbols:
            result[a][b] = sum((float(x) - means[a]) * (float(y) - means[b])
                               for x, y in zip(returns[a], returns[b])) / (count - 1)
    return result
