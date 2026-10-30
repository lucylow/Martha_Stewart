"""Discounted cash flow, dividend discount and relative valuation utilities."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .core import FinanceInputError, Number, finite, positive


def present_value(cashflows: Sequence[tuple[float, float]], discount_rate: Number,
                  *, continuous: bool = False) -> float:
    """Present value of cash flows at year fractions from valuation date."""
    rate = finite(discount_rate)
    result = 0.0
    for time, amount in cashflows:
        t, cf = positive(time, "time", allow_zero=True), finite(amount, "cashflow")
        base = math.exp(-rate*t) if continuous else (1+rate)**(-t)
        if not continuous and 1+rate <= 0:
            raise FinanceInputError("discount rate must exceed -100%")
        result += cf*base
    return result


def net_present_value(initial_investment: Number, cashflows: Sequence[tuple[float, float]],
                      discount_rate: Number) -> float:
    """NPV with the initial investment treated as a time-zero outflow."""
    initial = finite(initial_investment)
    if initial < 0:
        raise FinanceInputError("initial_investment should be non-negative")
    return present_value(cashflows, discount_rate)-initial


def terminal_value_perpetuity(next_year_cashflow: Number, discount_rate: Number,
                              growth_rate: Number) -> float:
    """Gordon growth terminal value at the end of a forecast period."""
    cf, discount, growth = finite(next_year_cashflow), finite(discount_rate), finite(growth_rate)
    if discount <= growth:
        raise FinanceInputError("discount rate must exceed perpetual growth rate")
    return cf/(discount-growth)


def dcf_valuation(forecast_cashflows: Sequence[Number], discount_rate: Number,
                  terminal_growth: Number, *, terminal_multiple: Number | None = None,
                  terminal_metric: Number | None = None,
                  net_debt: Number = 0.0, shares_outstanding: Number | None = None) -> dict[str, float]:
    """Enterprise and equity value from explicit forecast cash flows.

    Supply either Gordon-growth terminal value or an exit multiple (not both).
    Cash flows are assumed to occur at integer year ends.
    """
    flows = tuple(finite(v, f"forecast_cashflows[{i}]") for i, v in enumerate(forecast_cashflows))
    if not flows:
        raise FinanceInputError("forecast_cashflows cannot be empty")
    rate, growth, debt = finite(discount_rate), finite(terminal_growth), finite(net_debt)
    if rate <= growth or rate <= -1:
        raise FinanceInputError("discount rate must exceed terminal growth and -100%")
    if (terminal_multiple is None) != (terminal_metric is None):
        raise FinanceInputError("terminal_multiple and terminal_metric must be supplied together")
    explicit_pv = sum(cf/(1+rate)**(i+1) for i, cf in enumerate(flows))
    last = flows[-1]
    if terminal_multiple is not None:
        multiple, metric = positive(terminal_multiple), finite(terminal_metric)
        terminal = multiple*metric
    else:
        terminal = terminal_value_perpetuity(last*(1+growth), rate, growth)
    terminal_pv = terminal/(1+rate)**len(flows)
    enterprise = explicit_pv+terminal_pv
    equity = enterprise-debt
    result = {"explicit_pv": explicit_pv, "terminal_value": terminal,
              "terminal_pv": terminal_pv, "enterprise_value": enterprise,
              "net_debt": debt, "equity_value": equity}
    if shares_outstanding is not None:
        shares = positive(shares_outstanding)
        result["value_per_share"] = equity/shares
    return result


def dividend_discount_model(next_dividend: Number, cost_of_equity: Number,
                            growth_rate: Number) -> float:
    """Single-stage Gordon dividend discount value."""
    dividend, cost, growth = finite(next_dividend), finite(cost_of_equity), finite(growth_rate)
    if dividend < 0 or cost <= growth:
        raise FinanceInputError("dividend must be non-negative and cost of equity exceed growth")
    return dividend/(cost-growth)


def two_stage_ddm(current_dividend: Number, near_growth: Number,
                  terminal_growth: Number, cost_of_equity: Number,
                  high_growth_years: int) -> dict[str, float]:
    """Two-stage dividend model with a finite high-growth phase."""
    dividend, g1, g2, r = map(finite, (current_dividend, near_growth, terminal_growth, cost_of_equity))
    if dividend < 0 or not isinstance(high_growth_years, int) or high_growth_years < 1:
        raise FinanceInputError("invalid dividend or high-growth horizon")
    if r <= g2 or r <= -1 or g1 <= -1:
        raise FinanceInputError("invalid growth/cost-of-equity relationship")
    pv_explicit, next_div = 0.0, dividend
    for year in range(1, high_growth_years+1):
        next_div *= 1+g1
        pv_explicit += next_div/(1+r)**year
    terminal = next_div*(1+g2)/(r-g2)
    terminal_pv = terminal/(1+r)**high_growth_years
    return {"explicit_pv": pv_explicit, "terminal_value": terminal,
            "terminal_pv": terminal_pv, "equity_value": pv_explicit+terminal_pv}


def enterprise_value_to_equity(enterprise_value: Number, debt: Number,
                               cash: Number, minority_interest: Number = 0.0,
                               preferred_equity: Number = 0.0) -> float:
    """Bridge enterprise value to common equity value."""
    return (finite(enterprise_value)-finite(debt)+finite(cash)
            -finite(minority_interest)-finite(preferred_equity))


def valuation_multiples(price: Number, earnings_per_share: Number,
                        book_value_per_share: Number,
                        sales_per_share: Number,
                        dividend_per_share: Number = 0.0) -> dict[str, float]:
    """Compute P/E, P/B, P/S and dividend yield with sign checks."""
    p = positive(price)
    eps, book, sales, dividend = map(finite, (earnings_per_share, book_value_per_share,
                                              sales_per_share, dividend_per_share))
    return {"pe": p/eps if eps > 0 else math.nan,
            "pb": p/book if book > 0 else math.nan,
            "ps": p/sales if sales > 0 else math.nan,
            "dividend_yield": dividend/p}


def implied_growth_from_multiple(price: Number, next_year_cashflow: Number,
                                 discount_rate: Number) -> float:
    """Invert the Gordon multiple to solve for the implied perpetual growth rate."""
    p, cf, r = positive(price), positive(next_year_cashflow), finite(discount_rate)
    growth = r-cf/p
    if growth <= -1:
        raise FinanceInputError("implied growth is outside admissible range")
    return growth


def sensitivity_grid(base_value: float, discount_rates: Sequence[Number],
                     growth_rates: Sequence[Number],
                     terminal_cashflow: Number) -> dict[tuple[float, float], float]:
    """Gordon terminal value sensitivity grid, discounted one period."""
    cf = finite(terminal_cashflow)
    grid = {}
    for rate in discount_rates:
        for growth in growth_rates:
            r, g = finite(rate), finite(growth)
            if r <= g or r <= -1:
                grid[(r, g)] = math.nan
            else:
                grid[(r, g)] = base_value+cf/(r-g)
    return grid


def implied_cost_of_equity(price: Number, next_dividend: Number,
                           growth_rate: Number) -> float:
    """Solve the Gordon model for cost of equity r = D1/P + g."""
    p, d, g = positive(price), finite(next_dividend), finite(growth_rate)
    if d < 0:
        raise FinanceInputError("next dividend must be non-negative")
    return d/p+g


def cash_conversion_metrics(net_income: Number, operating_cash_flow: Number,
                            capital_expenditure: Number,
                            average_assets: Number | None = None) -> dict[str, float]:
    """Free cash flow and cash conversion diagnostics from aggregate statements."""
    ni, cfo, capex = map(finite, (net_income, operating_cash_flow, capital_expenditure))
    result = {"free_cash_flow": cfo-abs(capex),
              "cash_conversion": cfo/ni if ni else math.nan,
              "capex_to_cfo": abs(capex)/cfo if cfo else math.nan}
    if average_assets is not None:
        assets = positive(average_assets)
        result["cash_return_on_assets"] = cfo/assets
    return result
