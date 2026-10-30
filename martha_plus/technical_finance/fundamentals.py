"""Financial-statement ratios and quantitative fundamental quality diagnostics."""
from __future__ import annotations

import math
from typing import Mapping

from .core import FinanceInputError, Number, finite, positive


def dupont_analysis(net_income: Number, revenue: Number,
                    average_assets: Number, average_equity: Number) -> dict[str, float]:
    """Three-step DuPont decomposition of return on average equity."""
    ni, sales, assets, equity = map(finite, (net_income, revenue, average_assets, average_equity))
    if sales == 0 or assets <= 0 or equity <= 0:
        raise FinanceInputError("revenue must be non-zero and assets/equity positive")
    margin, turnover, leverage = ni/sales, sales/assets, assets/equity
    return {"net_margin": margin, "asset_turnover": turnover,
            "equity_multiplier": leverage, "roe": margin*turnover*leverage}


def altman_z_score(working_capital: Number, retained_earnings: Number,
                   ebit: Number, market_equity: Number,
                   total_liabilities: Number, sales: Number,
                   total_assets: Number) -> float:
    """Classic public-manufacturer Altman Z formula.

    The original coefficients are not appropriate for every industry or private
    company. Treat the result as a screening statistic only.
    """
    wc, re, ebit_value, equity, liabilities, revenue, assets = map(finite,
        (working_capital, retained_earnings, ebit, market_equity,
         total_liabilities, sales, total_assets))
    if assets <= 0 or liabilities <= 0:
        raise FinanceInputError("assets and liabilities must be positive")
    return (1.2*wc/assets+1.4*re/assets+3.3*ebit_value/assets
            +0.6*equity/liabilities+1.0*revenue/assets)


def accruals_ratio(net_income: Number, operating_cash_flow: Number,
                   average_assets: Number) -> float:
    """Cash-flow accruals scaled by average assets; lower may indicate quality."""
    ni, cfo, assets = map(finite, (net_income, operating_cash_flow, average_assets))
    if assets <= 0:
        raise FinanceInputError("average_assets must be positive")
    return (ni-cfo)/assets


def piotroski_f_score(current: Mapping[str, Number],
                      prior: Mapping[str, Number]) -> dict[str, object]:
    """Compute a transparent nine-point Piotroski-style score from two periods.

    Required keys: net_income, operating_cash_flow, roa, long_term_debt,
    current_ratio, shares_outstanding, gross_margin and asset_turnover.
    ``roa`` is assumed supplied as a ratio, and ``gross_margin`` likewise.
    """
    required = {"net_income", "operating_cash_flow", "roa", "long_term_debt",
                "current_ratio", "shares_outstanding", "gross_margin", "asset_turnover"}
    if not required.issubset(current) or not required.issubset(prior):
        missing = sorted(required-set(current) | required-set(prior))
        raise FinanceInputError(f"missing financial inputs: {missing}")
    c, p = ({k: finite(data[k]) for k in required} for data in (current, prior))
    tests = {
        "positive_net_income": c["net_income"] > 0,
        "positive_operating_cash_flow": c["operating_cash_flow"] > 0,
        "improving_roa": c["roa"] > p["roa"],
        "cash_flow_exceeds_income": c["operating_cash_flow"] > c["net_income"],
        "lower_leverage": c["long_term_debt"] < p["long_term_debt"],
        "improving_liquidity": c["current_ratio"] > p["current_ratio"],
        "no_dilution": c["shares_outstanding"] <= p["shares_outstanding"],
        "improving_gross_margin": c["gross_margin"] > p["gross_margin"],
        "improving_asset_turnover": c["asset_turnover"] > p["asset_turnover"]}
    return {"score": sum(tests.values()), "maximum_score": 9,
            "checks": {k: bool(v) for k,v in tests.items()}}


def quality_metrics(net_income: Number, operating_cash_flow: Number,
                    revenue: Number, total_assets: Number,
                    total_debt: Number, cash: Number,
                    capital_expenditure: Number) -> dict[str, float]:
    """Compact profitability, cash conversion, FCF and leverage metrics."""
    ni, cfo, sales, assets, debt, cash_value, capex = map(finite,
        (net_income, operating_cash_flow, revenue, total_assets,
         total_debt, cash, capital_expenditure))
    if assets <= 0 or sales == 0:
        raise FinanceInputError("assets must be positive and revenue non-zero")
    fcf = cfo-abs(capex)
    return {"net_margin": ni/sales, "cash_conversion": cfo/ni if ni else math.nan,
            "free_cash_flow": fcf, "fcf_margin": fcf/sales,
            "net_debt": debt-cash_value, "net_debt_to_assets": (debt-cash_value)/assets,
            "cash_to_debt": cash_value/debt if debt else math.inf}


def net_working_capital(current_assets: Number, current_liabilities: Number,
                        cash: Number = 0.0, short_term_debt: Number = 0.0) -> dict[str, float]:
    """Traditional and operating net working capital approximations."""
    ca, cl, cash_value, debt = map(finite, (current_assets, current_liabilities,
                                            cash, short_term_debt))
    return {"net_working_capital": ca-cl,
            "operating_working_capital": ca-cash_value-(cl-debt),
            "current_ratio": ca/cl if cl else math.inf,
            "quick_ratio_proxy": (ca-cash_value)/cl if cl else math.inf}


def cash_flow_yield(free_cash_flow: Number, market_capitalization: Number) -> float:
    """FCF yield using equity market capitalization as denominator."""
    cap = positive(market_capitalization)
    return finite(free_cash_flow)/cap


def earnings_yield(earnings: Number, market_capitalization: Number) -> float:
    """Earnings yield using market capitalization; inverse of positive P/E."""
    return finite(earnings)/positive(market_capitalization)


def enterprise_value( market_cap: Number, total_debt: Number,
                      cash: Number, preferred_equity: Number = 0.0,
                      minority_interest: Number = 0.0) -> float:
    """Enterprise value bridge from market cap and balance-sheet claims."""
    return (positive(market_cap)+finite(total_debt)-finite(cash)
            +finite(preferred_equity)+finite(minority_interest))


def ev_to_ebitda(ev: Number, ebitda: Number) -> float:
    """EV/EBITDA multiple; undefined for non-positive EBITDA."""
    denominator = finite(ebitda)
    return finite(ev)/denominator if denominator > 0 else math.nan


def interest_coverage(ebit: Number, interest_expense: Number) -> float:
    """EBIT / interest expense, with infinite coverage for zero expense."""
    interest = finite(interest_expense)
    if interest < 0:
        raise FinanceInputError("interest expense must be non-negative")
    return finite(ebit)/interest if interest else math.inf


def dividend_payout_ratio(dividends_per_share: Number,
                          earnings_per_share: Number) -> float:
    """Dividend payout ratio where EPS is positive."""
    dividend, earnings = map(finite, (dividends_per_share, earnings_per_share))
    if dividend < 0:
        raise FinanceInputError("dividend must be non-negative")
    return dividend/earnings if earnings > 0 else math.nan
