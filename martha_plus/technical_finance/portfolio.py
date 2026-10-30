"""Portfolio construction, covariance estimation and risk decomposition."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .core import FinanceInputError, Matrix, Number, Vector, cholesky, finite, mean, solve_linear, variance, vector


def covariance_matrix(returns: Sequence[Vector]) -> tuple[tuple[float, ...], ...]:
    """Sample covariance matrix from asset return columns."""
    from .core import covariance
    columns = [vector(col, f"returns[{i}]", min_length=2) for i, col in enumerate(returns)]
    if not columns or any(len(col) != len(columns[0]) for col in columns):
        raise FinanceInputError("return columns must be non-empty and aligned")
    return tuple(tuple(covariance(columns[i], columns[j]) for j in range(len(columns)))
                 for i in range(len(columns)))


def correlation_matrix(returns: Sequence[Vector]) -> tuple[tuple[float, ...], ...]:
    """Sample correlation matrix from aligned asset return columns."""
    from .core import correlation
    columns = [vector(col, f"returns[{i}]", min_length=2) for i, col in enumerate(returns)]
    if not columns or any(len(col) != len(columns[0]) for col in columns):
        raise FinanceInputError("return columns must be non-empty and aligned")
    return tuple(tuple(1.0 if i == j else correlation(columns[i], columns[j])
                       for j in range(len(columns))) for i in range(len(columns)))


def ewma_covariance(returns: Sequence[Vector], decay: Number = 0.94,
                    *, demean: bool = True) -> tuple[tuple[float, ...], ...]:
    """RiskMetrics-style exponentially weighted covariance estimate."""
    columns = [vector(col, f"returns[{i}]", min_length=2) for i, col in enumerate(returns)]
    if not columns or any(len(col) != len(columns[0]) for col in columns):
        raise FinanceInputError("return columns must align")
    lam = finite(decay, "decay")
    if not 0 < lam < 1:
        raise FinanceInputError("decay must be between zero and one")
    n, p = len(columns[0]), len(columns)
    centers = [mean(col) if demean else 0.0 for col in columns]
    weights = [(1-lam)*lam**(n-1-t) for t in range(n)]
    norm = sum(weights)
    weights = [w/norm for w in weights]
    result = [[0.0]*p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            result[i][j] = sum(weights[t]*(columns[i][t]-centers[i])*(columns[j][t]-centers[j])
                               for t in range(n))
    return tuple(tuple(row) for row in result)


def shrink_covariance(cov: Matrix, intensity: Number = 0.2,
                      *, target: str = "diagonal") -> tuple[tuple[float, ...], ...]:
    """Shrink a sample covariance toward a diagonal or constant-correlation target."""
    values = tuple(tuple(finite(v) for v in row) for row in cov)
    n = len(values)
    if n == 0 or any(len(row) != n for row in values):
        raise FinanceInputError("covariance must be a non-empty square matrix")
    if any(abs(values[i][j]-values[j][i]) > 1e-8 for i in range(n) for j in range(n)):
        raise FinanceInputError("covariance must be symmetric")
    alpha = finite(intensity, "intensity")
    if not 0 <= alpha <= 1:
        raise FinanceInputError("intensity must be in [0, 1]")
    variances = [values[i][i] for i in range(n)]
    if any(v < 0 for v in variances):
        raise FinanceInputError("variances cannot be negative")
    if target == "diagonal":
        tgt = [[variances[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    elif target == "constant_correlation":
        corr_values = []
        for i in range(n):
            for j in range(i):
                denom = math.sqrt(variances[i]*variances[j])
                if denom > 0:
                    corr_values.append(values[i][j]/denom)
        rho = sum(corr_values)/len(corr_values) if corr_values else 0.0
        tgt = [[variances[i] if i == j else rho*math.sqrt(variances[i]*variances[j])
                for j in range(n)] for i in range(n)]
    else:
        raise FinanceInputError("target must be diagonal or constant_correlation")
    return tuple(tuple((1-alpha)*values[i][j]+alpha*tgt[i][j] for j in range(n))
                 for i in range(n))


def portfolio_variance(weights: Vector, cov: Matrix) -> float:
    """Compute w'Σw, allowing long and short weights."""
    w = vector(weights)
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    if len(sigma) != len(w) or any(len(row) != len(w) for row in sigma):
        raise FinanceInputError("weight and covariance dimensions do not match")
    value = sum(w[i]*sigma[i][j]*w[j] for i in range(len(w)) for j in range(len(w)))
    if value < -1e-10:
        raise FinanceInputError("covariance matrix produced negative portfolio variance")
    return max(0.0, value)


def portfolio_volatility(weights: Vector, cov: Matrix, *, annualization: Number = 1.0) -> float:
    """Annualized portfolio volatility from a covariance matrix."""
    return math.sqrt(portfolio_variance(weights, cov))*math.sqrt(finite(annualization, "annualization"))


def portfolio_return(weights: Vector, expected_returns: Vector) -> float:
    """Linear expected portfolio return; weights need not sum to one."""
    w, mu = vector(weights), vector(expected_returns)
    if len(w) != len(mu):
        raise FinanceInputError("weights and expected returns must align")
    return sum(a*b for a, b in zip(w, mu))


def marginal_risk_contribution(weights: Vector, cov: Matrix) -> tuple[float, ...]:
    """Derivative of portfolio volatility with respect to each weight."""
    w = vector(weights)
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    vol = math.sqrt(portfolio_variance(w, sigma))
    if vol == 0:
        raise FinanceInputError("marginal risk is undefined at zero volatility")
    return tuple(sum(sigma[i][j]*w[j] for j in range(len(w)))/vol for i in range(len(w)))


def component_risk_contribution(weights: Vector, cov: Matrix) -> tuple[float, ...]:
    """Euler component contributions to portfolio standard deviation."""
    w = vector(weights)
    marginal = marginal_risk_contribution(w, cov)
    return tuple(wi*mi for wi, mi in zip(w, marginal))


def risk_contribution_percent(weights: Vector, cov: Matrix) -> tuple[float, ...]:
    """Normalize component risk contributions to percentages."""
    components = component_risk_contribution(weights, cov)
    total = sum(components)
    if abs(total) < 1e-15:
        raise FinanceInputError("risk contribution total is zero")
    return tuple(x/total for x in components)


def minimum_variance_weights(cov: Matrix, *, long_only: bool = True,
                             max_iterations: int = 10000, tolerance: float = 1e-10) -> tuple[float, ...]:
    """Projected-gradient minimum variance portfolio on the unit simplex.

    This is a deterministic research optimizer, not a substitute for a
    constrained convex solver where tight tolerances or many constraints matter.
    """
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    n = len(sigma)
    if n == 0 or any(len(row) != n for row in sigma):
        raise FinanceInputError("covariance must be square")
    if not long_only:
        ones = [1.0]*n
        solved = solve_linear(sigma, ones)
        total = sum(solved)
        if abs(total) < tolerance:
            raise FinanceInputError("unconstrained solution cannot be normalized")
        return tuple(x/total for x in solved)
    _validate_psd_shape(sigma)
    max_row_sum = max(sum(abs(x) for x in row) for row in sigma)
    step = 0.5/max(max_row_sum, 1e-12)
    w = [1.0/n]*n
    for _ in range(max_iterations):
        grad = [2*sum(sigma[i][j]*w[j] for j in range(n)) for i in range(n)]
        proposal = _project_simplex([w[i]-step*grad[i] for i in range(n)])
        change = max(abs(proposal[i]-w[i]) for i in range(n))
        w = proposal
        if change <= tolerance:
            break
    return tuple(w)


def risk_parity_weights(cov: Matrix, *, max_iterations: int = 10000,
                        tolerance: float = 1e-9) -> tuple[float, ...]:
    """Equalize long-only component variance contributions by iterative updates."""
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    n = len(sigma)
    if n == 0 or any(len(row) != n for row in sigma):
        raise FinanceInputError("covariance must be square")
    if any(sigma[i][i] <= 0 for i in range(n)):
        raise FinanceInputError("risk parity requires positive asset variances")
    w = [1.0/n]*n
    target = 1.0/n
    for _ in range(max_iterations):
        marginal = [sum(sigma[i][j]*w[j] for j in range(n)) for i in range(n)]
        contributions = [w[i]*marginal[i] for i in range(n)]
        total = sum(contributions)
        if total <= 0:
            raise FinanceInputError("portfolio variance must be positive")
        shares = [c/total for c in contributions]
        error = max(abs(s-target) for s in shares)
        if error <= tolerance:
            break
        for i in range(n):
            if shares[i] > 0:
                w[i] *= math.sqrt(target/shares[i])
        w = list(_project_simplex(w))
    return tuple(w)


def tangency_weights(expected_returns: Vector, cov: Matrix,
                     risk_free: Number = 0.0) -> tuple[float, ...]:
    """Unconstrained maximum-Sharpe direction normalized to unit net exposure."""
    mu = vector(expected_returns)
    rf = finite(risk_free, "risk_free")
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    if len(mu) != len(sigma) or any(len(row) != len(mu) for row in sigma):
        raise FinanceInputError("expected returns and covariance dimensions differ")
    excess = [m-rf for m in mu]
    direction = solve_linear(sigma, excess)
    total = sum(direction)
    if abs(total) < 1e-12:
        raise FinanceInputError("tangency portfolio has zero net exposure")
    return tuple(x/total for x in direction)


def tracking_error_weights(weights: Vector, benchmark_weights: Vector,
                           cov: Matrix) -> float:
    """Tracking error for active weights against the supplied covariance."""
    w, b = vector(weights), vector(benchmark_weights)
    if len(w) != len(b):
        raise FinanceInputError("portfolio and benchmark weights must align")
    active = [x-y for x, y in zip(w, b)]
    return math.sqrt(portfolio_variance(active, cov))


def active_risk_contributions(weights: Vector, benchmark_weights: Vector,
                              cov: Matrix) -> tuple[float, ...]:
    """Euler contributions to ex-ante tracking error by asset."""
    w, b = vector(weights), vector(benchmark_weights)
    if len(w) != len(b):
        raise FinanceInputError("portfolio and benchmark weights must align")
    return component_risk_contribution([x-y for x, y in zip(w, b)], cov)


def effective_number_of_positions(weights: Vector) -> float:
    """Inverse Herfindahl concentration for non-negative normalized weights."""
    w = vector(weights)
    if any(x < 0 for x in w) or abs(sum(w)-1.0) > 1e-6:
        raise FinanceInputError("weights must be long-only and sum to one")
    hhi = sum(x*x for x in w)
    return 1.0/hhi if hhi else 0.0


def herfindahl_index(weights: Vector) -> float:
    """Sum of squared portfolio weights (long-only concentration measure)."""
    w = vector(weights)
    if any(x < 0 for x in w) or abs(sum(w)-1.0) > 1e-6:
        raise FinanceInputError("weights must be long-only and sum to one")
    return sum(x*x for x in w)


def _project_simplex(values: Sequence[Number]) -> tuple[float, ...]:
    """Euclidean projection of a vector onto {w >= 0, sum(w)=1}."""
    data = [finite(v) for v in values]
    ordered = sorted(data, reverse=True)
    cumulative = 0.0
    rho = 0
    theta = 0.0
    for j, value in enumerate(ordered, start=1):
        cumulative += value
        candidate = (cumulative-1.0)/j
        if value-candidate > 0:
            rho, theta = j, candidate
    if rho == 0:
        return tuple(1.0/len(data) for _ in data)
    return tuple(max(v-theta, 0.0) for v in data)


def _validate_psd_shape(cov: Sequence[Sequence[float]]) -> None:
    for i, row in enumerate(cov):
        for j, value in enumerate(row):
            if abs(value-cov[j][i]) > 1e-8:
                raise FinanceInputError("covariance must be symmetric")
            if i == j and value < 0:
                raise FinanceInputError("diagonal variances cannot be negative")
