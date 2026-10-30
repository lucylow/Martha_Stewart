"""Cost of capital, leverage, capital structure and position-sizing formulas."""
from __future__ import annotations

import math
from typing import Mapping

from .core import FinanceInputError, Number, finite, positive


def capm_cost_of_equity(risk_free_rate: Number, beta: Number,
                        market_risk_premium: Number) -> float:
    """CAPM expected return: risk-free + beta * market risk premium."""
    return finite(risk_free_rate)+finite(beta)*finite(market_risk_premium)


def fama_french_cost_of_equity(risk_free_rate: Number,
                               factor_betas: Mapping[str, Number],
                               factor_premia: Mapping[str, Number]) -> float:
    """Multi-factor expected return from aligned beta and premium mappings."""
    if set(factor_betas) != set(factor_premia):
        raise FinanceInputError("factor beta and premium keys must match")
    return finite(risk_free_rate)+sum(finite(factor_betas[k])*finite(factor_premia[k])
                                      for k in factor_betas)


def after_tax_cost_of_debt(pre_tax_cost: Number, marginal_tax_rate: Number) -> float:
    """After-tax debt cost under full interest deductibility assumption."""
    cost, tax = finite(pre_tax_cost), finite(marginal_tax_rate)
    if not 0 <= tax <= 1:
        raise FinanceInputError("marginal tax rate must be in [0, 1]")
    return cost*(1-tax)


def weighted_average_cost_of_capital(cost_of_equity: Number,
                                     pre_tax_cost_of_debt: Number,
                                     equity_value: Number, debt_value: Number,
                                     tax_rate: Number,
                                     preferred_cost: Number = 0.0,
                                     preferred_value: Number = 0.0) -> dict[str, float]:
    """Market-value weighted WACC with tax shield on debt only."""
    equity, debt, preferred = (positive(equity_value, "equity_value", allow_zero=True),
                               positive(debt_value, "debt_value", allow_zero=True),
                               positive(preferred_value, "preferred_value", allow_zero=True))
    total = equity+debt+preferred
    if total <= 0:
        raise FinanceInputError("capital structure must have positive market value")
    tax = finite(tax_rate)
    if not 0 <= tax <= 1:
        raise FinanceInputError("tax_rate must be in [0, 1]")
    ke, kd, kp = map(finite, (cost_of_equity, pre_tax_cost_of_debt, preferred_cost))
    weights = {"equity": equity/total, "debt": debt/total, "preferred": preferred/total}
    debt_after_tax = kd*(1-tax)
    wacc = weights["equity"]*ke+weights["debt"]*debt_after_tax+weights["preferred"]*kp
    return {"wacc": wacc, "equity_weight": weights["equity"],
            "debt_weight": weights["debt"], "preferred_weight": weights["preferred"],
            "after_tax_cost_of_debt": debt_after_tax, "total_capital": total}


def levered_beta(unlevered_beta: Number, debt_to_equity: Number,
                 tax_rate: Number) -> float:
    """Hamada relevering beta under a simplified permanent-debt assumption."""
    beta, leverage, tax = finite(unlevered_beta), finite(debt_to_equity), finite(tax_rate)
    if leverage < 0 or not 0 <= tax <= 1:
        raise FinanceInputError("leverage must be non-negative and tax in [0,1]")
    return beta*(1+(1-tax)*leverage)


def unlevered_beta(levered: Number, debt_to_equity: Number,
                   tax_rate: Number) -> float:
    """Hamada unlevering inverse of :func:`levered_beta`."""
    beta, leverage, tax = finite(levered), finite(debt_to_equity), finite(tax_rate)
    if leverage < 0 or not 0 <= tax <= 1:
        raise FinanceInputError("leverage must be non-negative and tax in [0,1]")
    return beta/(1+(1-tax)*leverage)


def weighted_average_life(cashflows: Mapping[float, Number]) -> float:
    """Cash-flow-weighted average maturity, requiring non-negative flows."""
    normalized = [(positive(t, "time", allow_zero=True), positive(v, "cashflow", allow_zero=True))
                  for t, v in cashflows.items()]
    total = sum(value for _, value in normalized)
    if total <= 0:
        raise FinanceInputError("cashflows must have positive total")
    return sum(t*v for t, v in normalized)/total


def kelly_fraction(win_probability: Number, payoff_ratio: Number,
                   *, fraction: Number = 1.0) -> float:
    """Binary-outcome Kelly fraction, optionally scaled down by a fraction.

    This is a theoretical growth-optimal sizing result, not a recommendation.
    Estimation error, fat tails, liquidity and ruin risk are not included.
    """
    p, b, scale = finite(win_probability), positive(payoff_ratio), finite(fraction)
    if not 0 <= p <= 1 or not 0 <= scale <= 1:
        raise FinanceInputError("win_probability and fraction must be in [0, 1]")
    q = 1-p
    full = (b*p-q)/b
    return max(0.0, full)*scale


def fractional_kelly_from_moments(expected_return: Number,
                                  variance_value: Number,
                                  *, fraction: Number = 0.5) -> float:
    """Quadratic-utility approximation f = fraction * mean / variance."""
    mu, var, scale = finite(expected_return), positive(variance_value), finite(fraction)
    if not 0 <= scale <= 1:
        raise FinanceInputError("fraction must be in [0,1]")
    return max(0.0, mu/var*scale)


def leverage_metrics(total_assets: Number, equity: Number,
                     interest_expense: Number, ebit: Number) -> dict[str, float]:
    """Balance-sheet and interest-coverage ratios."""
    assets, capital = positive(total_assets), positive(equity)
    interest, operating = finite(interest_expense), finite(ebit)
    return {"assets_to_equity": assets/capital,
            "equity_multiplier": assets/capital,
            "interest_coverage": operating/interest if interest > 0 else math.inf,
            "debt_to_assets_proxy": max(0.0, assets-capital)/assets,
            "equity_to_assets": capital/assets}


def portfolio_leverage(gross_exposure: Number, net_asset_value: Number) -> dict[str, float]:
    """Gross and net leverage ratios for absolute exposure and NAV inputs."""
    gross, nav = positive(gross_exposure, allow_zero=True), positive(net_asset_value)
    return {"gross_leverage": gross/nav,
            "net_asset_value": nav,
            "excess_gross_exposure": max(0.0, gross-nav)}
