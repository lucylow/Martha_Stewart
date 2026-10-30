"""Value-at-risk, expected shortfall, stress testing and risk budgeting."""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .core import (FinanceInputError, Matrix, Number, Vector, cholesky, finite,
                   mean, normal_ppf, probability, quantile, variance, vector)
from .portfolio import portfolio_variance
from .statistics import moments, tail_mean


def historical_var(returns: Vector, confidence: Number = 0.99,
                   *, portfolio_value: Number = 1.0) -> float:
    """Historical one-period loss VaR as a positive currency amount."""
    data = vector(returns)
    alpha = probability(confidence, "confidence")
    if not 0.5 < alpha < 1:
        raise FinanceInputError("confidence must be between 0.5 and 1")
    value = finite(portfolio_value, "portfolio_value")
    if value < 0:
        raise FinanceInputError("portfolio_value must be non-negative")
    return max(0.0, -quantile(data, 1.0-alpha)*value)


def historical_expected_shortfall(returns: Vector, confidence: Number = 0.99,
                                  *, portfolio_value: Number = 1.0) -> float:
    """Historical expected shortfall using observations at/below the VaR quantile."""
    data = vector(returns)
    alpha = probability(confidence, "confidence")
    if not 0.5 < alpha < 1:
        raise FinanceInputError("confidence must be between 0.5 and 1")
    value = finite(portfolio_value, "portfolio_value")
    cutoff = quantile(data, 1.0-alpha)
    tail = [x for x in data if x <= cutoff]
    if not tail:
        raise FinanceInputError("no observations in the selected tail")
    return max(0.0, -mean(tail)*value)


def gaussian_var(returns: Vector, confidence: Number = 0.99,
                 *, portfolio_value: Number = 1.0, mean_adjusted: bool = True) -> float:
    """Normal VaR based on sample mean and volatility."""
    data = vector(returns, min_length=2)
    alpha = probability(confidence, "confidence")
    if not 0.5 < alpha < 1:
        raise FinanceInputError("confidence must be between 0.5 and 1")
    value = finite(portfolio_value, "portfolio_value")
    z = normal_ppf(alpha)
    mu = mean(data) if mean_adjusted else 0.0
    sigma = math.sqrt(variance(data))
    return max(0.0, (z*sigma-mu)*value)


def gaussian_expected_shortfall(returns: Vector, confidence: Number = 0.99,
                                 *, portfolio_value: Number = 1.0,
                                 mean_adjusted: bool = True) -> float:
    """Closed-form Gaussian expected shortfall for a one-period loss."""
    data = vector(returns, min_length=2)
    alpha = probability(confidence, "confidence")
    if not 0.5 < alpha < 1:
        raise FinanceInputError("confidence must be between 0.5 and 1")
    value = finite(portfolio_value, "portfolio_value")
    mu = mean(data) if mean_adjusted else 0.0
    sigma = math.sqrt(variance(data))
    z = normal_ppf(alpha)
    density = math.exp(-0.5*z*z)/math.sqrt(2*math.pi)
    return max(0.0, (sigma*density/(1-alpha)-mu)*value)


def cornish_fisher_var(returns: Vector, confidence: Number = 0.99,
                       *, portfolio_value: Number = 1.0) -> float:
    """Cornish-Fisher adjusted VaR using sample skewness and excess kurtosis."""
    data = vector(returns, min_length=4)
    alpha = probability(confidence, "confidence")
    if not 0.5 < alpha < 1:
        raise FinanceInputError("confidence must be between 0.5 and 1")
    stats = moments(data)
    z = normal_ppf(alpha)
    s, k = stats["skewness"], stats["excess_kurtosis"]
    z_cf = (z + (z*z-1)*s/6 + (z**3-3*z)*k/24
            - (2*z**3-5*z)*s*s/36)
    return max(0.0, (z_cf*stats["volatility"]-stats["mean"])*finite(portfolio_value))


def parametric_var_from_covariance(weights: Vector, covariance: Matrix,
                                   confidence: Number = 0.99,
                                   *, portfolio_value: Number = 1.0,
                                   expected_returns: Vector | None = None) -> float:
    """One-period Gaussian portfolio VaR from a covariance matrix."""
    w = vector(weights)
    sigma2 = portfolio_variance(w, covariance)
    mu = 0.0 if expected_returns is None else sum(a*b for a, b in zip(w, vector(expected_returns)))
    if expected_returns is not None and len(vector(expected_returns)) != len(w):
        raise FinanceInputError("expected return vector must align with weights")
    z = normal_ppf(probability(confidence, "confidence"))
    if z <= 0:
        raise FinanceInputError("confidence must exceed 50%")
    return max(0.0, (z*math.sqrt(sigma2)-mu)*finite(portfolio_value))


def monte_carlo_portfolio_returns(weights: Vector, covariance: Matrix,
                                  expected_returns: Vector, *, simulations: int = 10000,
                                  seed: int = 7) -> tuple[float, ...]:
    """Draw correlated Gaussian portfolio returns with a reproducible seed."""
    w, mu = vector(weights), vector(expected_returns)
    if len(w) != len(mu) or len(w) != len(covariance):
        raise FinanceInputError("portfolio dimensions do not match")
    if not isinstance(simulations, int) or simulations < 2:
        raise FinanceInputError("simulations must be an integer >= 2")
    lower = cholesky(covariance)
    rng = random.Random(seed)
    result = []
    for _ in range(simulations):
        z = [rng.gauss(0.0, 1.0) for _ in w]
        correlated = [sum(lower[i][j]*z[j] for j in range(i+1)) for i in range(len(w))]
        asset_returns = [mu[i]+correlated[i] for i in range(len(w))]
        result.append(sum(wi*ri for wi, ri in zip(w, asset_returns)))
    return tuple(result)


def monte_carlo_var(weights: Vector, covariance: Matrix, expected_returns: Vector,
                    confidence: Number = 0.99, *, simulations: int = 10000,
                    seed: int = 7, portfolio_value: Number = 1.0) -> dict[str, float]:
    """Monte Carlo Gaussian VaR/ES and simulation diagnostics."""
    samples = monte_carlo_portfolio_returns(weights, covariance, expected_returns,
                                             simulations=simulations, seed=seed)
    return {"var": historical_var(samples, confidence, portfolio_value=portfolio_value),
            "expected_shortfall": historical_expected_shortfall(samples, confidence,
                                                                   portfolio_value=portfolio_value),
            "simulations": float(simulations), "seed": float(seed)}


def volatility_scaled_var(base_var: Number, current_vol: Number,
                          reference_vol: Number) -> float:
    """Scale an existing VaR estimate by a current/reference volatility ratio."""
    var = finite(base_var, "base_var")
    current = finite(current_vol, "current_vol")
    reference = finite(reference_vol, "reference_vol")
    if var < 0 or current < 0 or reference <= 0:
        raise FinanceInputError("VaR and current volatility must be non-negative; reference positive")
    return var*current/reference


def stress_pnl(exposures: Mapping[str, Number], shocks: Mapping[str, Number]) -> dict[str, float]:
    """Linear factor stress with per-factor and total P&L."""
    if set(exposures) != set(shocks):
        raise FinanceInputError("exposure and shock factor names must match")
    contributions = {key: finite(exposures[key])*finite(shocks[key]) for key in exposures}
    contributions["total"] = sum(contributions.values())
    return contributions


def historical_scenario_pnl(exposures: Mapping[str, Number],
                            scenario_returns: Mapping[str, Mapping[str, Number]]) -> dict[str, dict[str, float]]:
    """Apply a named set of historical factor shocks to linear exposures."""
    result = {}
    for scenario, shocks in scenario_returns.items():
        result[scenario] = stress_pnl(exposures, shocks)
    return result


def risk_limit_utilization(current: Number, limit: Number) -> float:
    """Fraction of a positive risk budget consumed by a non-negative metric."""
    value, cap = finite(current, "current"), finite(limit, "limit")
    if value < 0 or cap <= 0:
        raise FinanceInputError("current must be non-negative and limit positive")
    return value/cap


def risk_budget_scale(current_risk: Number, target_risk: Number,
                      *, minimum_scale: Number = 0.0,
                      maximum_scale: Number = 2.0) -> float:
    """Linear exposure scale required to move current risk to target risk."""
    current, target = finite(current_risk), finite(target_risk)
    lo, hi = finite(minimum_scale), finite(maximum_scale)
    if current <= 0 or target < 0 or lo < 0 or hi < lo:
        raise FinanceInputError("invalid risk-budget inputs")
    return min(hi, max(lo, target/current))


def component_var(weights: Vector, covariance: Matrix,
                  confidence: Number = 0.99) -> tuple[float, ...]:
    """Euler component contributions to zero-mean Gaussian VaR."""
    w = vector(weights)
    sigma = math.sqrt(portfolio_variance(w, covariance))
    if sigma <= 0:
        raise FinanceInputError("component VaR undefined at zero volatility")
    z = normal_ppf(probability(confidence, "confidence"))
    marginal = [sum(covariance[i][j]*w[j] for j in range(len(w)))/sigma for i in range(len(w))]
    return tuple(z*wi*mi for wi, mi in zip(w, marginal))


def expected_shortfall_backtest(returns: Vector, var_series: Vector,
                                confidence: Number = 0.99) -> dict[str, float]:
    """Simple VaR exception and average-tail-loss diagnostics.

    This is descriptive only; formal model validation should use longer samples,
    conditional coverage tests and dependence-aware inference.
    """
    data, forecasts = vector(returns), vector(var_series)
    if len(data) != len(forecasts):
        raise FinanceInputError("returns and VaR forecasts must align")
    alpha = probability(confidence)
    breaches = [r < -v for r, v in zip(data, forecasts)]
    observed = sum(breaches)/len(data)
    errors = [max(0.0, -r-v) for r, v in zip(data, forecasts)]
    return {"expected_exception_rate": 1-alpha, "observed_exception_rate": observed,
            "exception_count": float(sum(breaches)), "mean_excess_loss": mean(errors)}


def parametric_delta_normal_var(delta_exposures: Vector, factor_cov: Matrix,
                                confidence: Number = 0.99,
                                *, portfolio_value: Number = 1.0) -> float:
    """Delta-normal VaR for factor sensitivities and factor covariance."""
    return parametric_var_from_covariance(delta_exposures, factor_cov, confidence,
                                          portfolio_value=portfolio_value)


def drawdown_risk(returns: Vector) -> dict[str, float]:
    """Drawdown depth, duration and episode count."""
    data = vector(returns, min_length=0)
    episodes = []
    from .returns import drawdown_episodes
    episodes = drawdown_episodes(data)
    return {"maximum_drawdown": min((e.depth for e in episodes), default=0.0),
            "longest_duration": float(max((e.duration for e in episodes), default=0)),
            "episode_count": float(len(episodes))}
