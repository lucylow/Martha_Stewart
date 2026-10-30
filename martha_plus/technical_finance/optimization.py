"""Constrained portfolio optimization and Bayesian Black-Litterman blending."""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .core import FinanceInputError, Matrix, Number, finite, positive, solve_linear, vector
from .portfolio import portfolio_return, portfolio_variance, minimum_variance_weights, tangency_weights
from .regression import inverse, matmul


def mean_variance_weights(expected_returns: Sequence[Number], covariance: Matrix,
                          risk_aversion: Number = 3.0, *,
                          lower_bounds: Sequence[Number] | None = None,
                          upper_bounds: Sequence[Number] | None = None,
                          max_iterations: int = 20000,
                          tolerance: Number = 1e-10) -> tuple[float, ...]:
    """Maximize mu'w - lambda/2*w'Σw on a bounded fully-invested simplex.

    Projected gradient with backtracking is used for deterministic small
    portfolios. It supports box bounds and sum(weights)=1 but not turnover,
    group, integer or cardinality constraints.
    """
    mu = vector(expected_returns)
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    n = len(mu)
    if len(sigma) != n or any(len(row) != n for row in sigma):
        raise FinanceInputError("expected returns and covariance dimensions differ")
    risk = positive(risk_aversion, "risk_aversion")
    lo = [0.0]*n if lower_bounds is None else list(vector(lower_bounds))
    hi = [1.0]*n if upper_bounds is None else list(vector(upper_bounds))
    if len(lo) != n or len(hi) != n or any(a > b for a,b in zip(lo,hi)):
        raise FinanceInputError("invalid lower/upper bounds")
    if sum(lo) > 1+1e-10 or sum(hi) < 1-1e-10:
        raise FinanceInputError("bounds must permit fully invested weights")
    # Feasible initial point from bounded-simplex projection of equal weights.
    weights = _project_bounded_simplex([1/n]*n, lo, hi)
    lipschitz = max(sum(abs(v) for v in row) for row in sigma)*risk
    step = 1/max(lipschitz, 1e-8)
    tol = positive(tolerance, "tolerance")
    for _ in range(max_iterations):
        gradient = [mu[i]-risk*sum(sigma[i][j]*weights[j] for j in range(n)) for i in range(n)]
        proposal = _project_bounded_simplex([weights[i]+step*gradient[i] for i in range(n)], lo, hi)
        change = max(abs(proposal[i]-weights[i]) for i in range(n))
        weights = proposal
        if change <= tol:
            break
    return tuple(weights)


def _project_bounded_simplex(values: Sequence[Number], lower: Sequence[float],
                             upper: Sequence[float]) -> tuple[float, ...]:
    """Project onto a box-constrained unit simplex by scalar threshold search."""
    vals = [finite(v) for v in values]
    if len(vals) != len(lower) or len(vals) != len(upper):
        raise FinanceInputError("projection dimensions do not match")
    lo_t, hi_t = min(v-u for v,u in zip(vals,upper))-1, max(v-l for v,l in zip(vals,lower))+1
    for _ in range(120):
        mid = (lo_t+hi_t)/2
        projected = [min(upper[i], max(lower[i], vals[i]-mid)) for i in range(len(vals))]
        if sum(projected) > 1:
            lo_t = mid
        else:
            hi_t = mid
    result = [min(upper[i], max(lower[i], vals[i]-(lo_t+hi_t)/2)) for i in range(len(vals))]
    total = sum(result)
    if abs(total-1) > 1e-8:
        # Distribute residual only across coordinates with remaining capacity.
        residual = 1-total
        for i in range(len(result)):
            capacity = (upper[i]-result[i]) if residual > 0 else (result[i]-lower[i])
            adjustment = math.copysign(min(abs(residual), capacity), residual)
            result[i] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-12:
                break
    return tuple(result)


def efficient_frontier(expected_returns: Sequence[Number], covariance: Matrix,
                       *, points: int = 25, risk_free_rate: Number = 0.0,
                       long_only: bool = True) -> tuple[dict[str, object], ...]:
    """Approximate efficient frontier by varying mean-variance risk aversion."""
    mu = vector(expected_returns)
    if not isinstance(points, int) or points < 2:
        raise FinanceInputError("points must be an integer >= 2")
    outputs = []
    for i in range(points):
        risk_aversion = 10**(-2+4*i/(points-1))
        weights = mean_variance_weights(mu, covariance, risk_aversion) if long_only else _unconstrained_mean_variance(mu, covariance, risk_aversion)
        ret = portfolio_return(weights, mu)
        vol = math.sqrt(portfolio_variance(weights, covariance))
        sharpe = (ret-finite(risk_free_rate))/vol if vol else math.nan
        outputs.append({"risk_aversion": risk_aversion, "expected_return": ret,
                        "volatility": vol, "sharpe": sharpe, "weights": weights})
    return tuple(outputs)


def _unconstrained_mean_variance(mu: Sequence[float], cov: Matrix,
                                 risk_aversion: float) -> tuple[float, ...]:
    sigma = tuple(tuple(finite(v) for v in row) for row in cov)
    inv_mu = solve_linear(sigma, mu)
    return tuple(v/risk_aversion for v in inv_mu)


def black_litterman_posterior(covariance: Matrix, market_weights: Sequence[Number],
                              risk_aversion: Number,
                              view_matrix: Matrix,
                              view_returns: Sequence[Number],
                              view_uncertainty: Matrix,
                              *, tau: Number = 0.05) -> dict[str, object]:
    """Compute Black-Litterman posterior expected returns and covariance.

    ``view_matrix`` P has one row per view and one column per asset; ``view_returns``
    is Q, and ``view_uncertainty`` is Ω. The implementation uses small dense
    linear algebra and expects positive-definite inputs.
    """
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    weights = vector(market_weights)
    p = tuple(tuple(finite(v) for v in row) for row in view_matrix)
    q = vector(view_returns)
    omega = tuple(tuple(finite(v) for v in row) for row in view_uncertainty)
    n, views = len(weights), len(q)
    if len(sigma) != n or any(len(row) != n for row in sigma):
        raise FinanceInputError("covariance must align with market weights")
    if len(p) != views or any(len(row) != n for row in p):
        raise FinanceInputError("view matrix shape must be views x assets")
    if len(omega) != views or any(len(row) != views for row in omega):
        raise FinanceInputError("view uncertainty must be square with one row per view")
    delta, tau_value = positive(risk_aversion), positive(tau)
    equilibrium = [delta*sum(sigma[i][j]*weights[j] for j in range(n)) for i in range(n)]
    tau_sigma = [[tau_value*sigma[i][j] for j in range(n)] for i in range(n)]
    prior_precision = inverse(tau_sigma)
    omega_inverse = inverse(omega)
    p_t = [list(row) for row in zip(*p)]
    pt_omega_inv = matmul(p_t, omega_inverse)
    view_precision = matmul(pt_omega_inv, p)
    posterior_precision = [[prior_precision[i][j]+view_precision[i][j]
                            for j in range(n)] for i in range(n)]
    posterior_cov = inverse(posterior_precision)
    prior_term = [sum(prior_precision[i][j]*equilibrium[j] for j in range(n)) for i in range(n)]
    view_term = matmul(pt_omega_inv, [[v] for v in q])
    rhs = [prior_term[i]+view_term[i][0] for i in range(n)]
    posterior_mean = [sum(posterior_cov[i][j]*rhs[j] for j in range(n)) for i in range(n)]
    return {"equilibrium_returns": tuple(equilibrium),
            "posterior_returns": tuple(posterior_mean),
            "posterior_covariance": tuple(tuple(row) for row in posterior_cov),
            "tau": tau_value, "risk_aversion": delta}


def proportional_transaction_cost(weights_before: Sequence[Number],
                                  weights_after: Sequence[Number],
                                  cost_bps: Number) -> float:
    """Estimate one-way rebalance cost as half gross turnover times bps."""
    before, after = vector(weights_before), vector(weights_after)
    if len(before) != len(after):
        raise FinanceInputError("weight vectors must align")
    rate = positive(cost_bps, "cost_bps", allow_zero=True)/10000
    turnover = sum(abs(a-b) for a,b in zip(after,before))/2
    return turnover*rate


def rebalance_with_turnover_penalty(expected_returns: Sequence[Number],
                                    covariance: Matrix,
                                    previous_weights: Sequence[Number],
                                    risk_aversion: Number = 3.0,
                                    cost_bps: Number = 10.0,
                                    *, max_iterations: int = 20000,
                                    tolerance: Number = 1e-9) -> tuple[float, ...]:
    """Mean-variance allocation with an L2 penalty on weight changes.

    The penalty approximates proportional turnover cost with a smooth convex
    surrogate; it is not an exact model of bid/ask costs or market impact.
    """
    mu, old = vector(expected_returns), vector(previous_weights)
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    if len(mu) != len(old) or len(sigma) != len(mu) or any(len(row) != len(mu) for row in sigma):
        raise FinanceInputError("portfolio dimensions differ")
    risk, cost = positive(risk_aversion), positive(cost_bps, allow_zero=True)/10000
    n = len(mu)
    lipschitz = max(sum(abs(v) for v in row) for row in sigma)*risk+2*cost
    step = 1/max(lipschitz, 1e-8)
    weights = _project_bounded_simplex(old, [0.0]*n, [1.0]*n)
    for _ in range(max_iterations):
        gradient = [mu[i]-risk*sum(sigma[i][j]*weights[j] for j in range(n))
                    -2*cost*(weights[i]-old[i]) for i in range(n)]
        candidate = _project_bounded_simplex([weights[i]+step*gradient[i] for i in range(n)],
                                              [0.0]*n, [1.0]*n)
        change = max(abs(candidate[i]-weights[i]) for i in range(n))
        weights = candidate
        if change < tolerance:
            break
    return tuple(weights)


def portfolio_constraints_report(weights: Sequence[Number],
                                  lower: Sequence[Number] | None = None,
                                  upper: Sequence[Number] | None = None,
                                  groups: Mapping[str, Sequence[int]] | None = None) -> dict[str, object]:
    """Report simplex, bound and group exposures for an allocation."""
    w = vector(weights)
    lo = [0.0]*len(w) if lower is None else list(vector(lower))
    hi = [1.0]*len(w) if upper is None else list(vector(upper))
    if len(lo) != len(w) or len(hi) != len(w):
        raise FinanceInputError("bounds must align with weights")
    violations = []
    if abs(sum(w)-1.0) > 1e-6:
        violations.append("weights do not sum to one")
    for i, value in enumerate(w):
        if value < lo[i]-1e-8 or value > hi[i]+1e-8:
            violations.append(f"weight[{i}] outside bounds")
    group_exposure = {}
    for name, indices in (groups or {}).items():
        if any(i < 0 or i >= len(w) for i in indices):
            raise FinanceInputError(f"group {name} contains invalid asset index")
        group_exposure[name] = sum(w[i] for i in indices)
    return {"feasible": not violations, "violations": tuple(violations),
            "gross_exposure": sum(abs(x) for x in w),
            "net_exposure": sum(w), "group_exposures": group_exposure}


def equal_risk_contribution(weights: Sequence[Number], covariance: Matrix) -> dict[str, object]:
    """Diagnostics for distance from equal component risk contribution."""
    from .portfolio import risk_contribution_percent
    shares = risk_contribution_percent(weights, covariance)
    target = 1/len(shares)
    return {"risk_shares": shares, "target_share": target,
            "max_absolute_deviation": max(abs(x-target) for x in shares),
            "sum_squared_deviation": sum((x-target)**2 for x in shares)}


def random_portfolios(expected_returns: Sequence[Number], covariance: Matrix,
                      *, samples: int = 1000, seed: int = 7,
                      risk_free_rate: Number = 0.0) -> tuple[dict[str, object], ...]:
    """Generate reproducible long-only fully-invested random portfolio diagnostics."""
    mu = vector(expected_returns)
    if not isinstance(samples, int) or samples < 1:
        raise FinanceInputError("samples must be positive")
    rng = random.Random(seed)
    outputs = []
    for _ in range(samples):
        raw = [rng.expovariate(1.0) for _ in mu]
        total = sum(raw)
        weights = [v/total for v in raw]
        ret = portfolio_return(weights, mu)
        vol = math.sqrt(portfolio_variance(weights, covariance))
        outputs.append({"weights": tuple(weights), "return": ret,
                        "volatility": vol,
                        "sharpe": (ret-finite(risk_free_rate))/vol if vol else math.nan})
    return tuple(outputs)
