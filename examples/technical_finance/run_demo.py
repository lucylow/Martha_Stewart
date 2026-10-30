"""End-to-end synthetic technical-finance example; no external data required."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from martha_plus.technical_finance.performance import performance_report
from martha_plus.technical_finance.portfolio import covariance_matrix, risk_parity_weights
from martha_plus.technical_finance.risk import historical_var, historical_expected_shortfall
from martha_plus.technical_finance.options.black_scholes import price, implied_volatility
from martha_plus.technical_finance.fixed_income.bonds import bond_price, bond_risk
from martha_plus.technical_finance.fixed_income.curves import CurvePoint, DiscountCurve
from martha_plus.technical_finance.credit import hazard_from_spread, default_probability
from martha_plus.technical_finance.ledger import PortfolioLedger


def main() -> None:
    # Small deterministic synthetic daily-return panel (not historical prices).
    returns = (
        (0.006, -0.004, 0.009, 0.002, -0.003, 0.007, 0.004, -0.002, 0.005, 0.003),
        (0.003, -0.001, 0.004, 0.001, 0.000, 0.003, 0.002, -0.001, 0.002, 0.001),
        (0.008, -0.007, 0.011, 0.003, -0.006, 0.009, 0.004, -0.004, 0.006, 0.002),
    )
    covariance = covariance_matrix(returns)
    weights = risk_parity_weights(covariance)
    portfolio_returns = tuple(sum(weights[i]*returns[i][t] for i in range(3))
                              for t in range(len(returns[0])))
    performance = performance_report(portfolio_returns, periods_per_year=252)
    risk = {
        "historical_var_95": historical_var(portfolio_returns, 0.95),
        "historical_es_95": historical_expected_shortfall(portfolio_returns, 0.95),
    }

    market_call = price(100, 100, 0.75, 0.04, 0.24, option_type="call")
    implied = implied_volatility(market_call, 100, 100, 0.75, 0.04, option_type="call")

    bond_value = bond_price(100, 0.045, 0.04, 5, 2)
    bond_sensitivities = bond_risk(100, 0.045, 0.04, 5, 2)
    curve = DiscountCurve([CurvePoint(0.5, 0.98), CurvePoint(1.0, 0.955),
                           CurvePoint(2.0, 0.91), CurvePoint(5.0, 0.80)])

    hazard = hazard_from_spread(0.012, 0.4)
    pd_5y = default_probability(hazard, 5)

    book = PortfolioLedger("USD", 10_000)
    book.trade("DEMO", "buy", 20, 100, "USD", fee=1.0)
    book.mark({"DEMO": 103.5})
    nav = book.net_asset_value()

    report = {
        "warning": "Synthetic illustration only; not a forecast, recommendation or profitability claim.",
        "portfolio": {"risk_parity_weights": weights, "performance": performance, "risk": risk},
        "option": {"call_price": market_call, "recovered_implied_volatility": implied},
        "fixed_income": {"bond_price": bond_value,
                          "modified_duration": bond_sensitivities.modified_duration,
                          "convexity": bond_sensitivities.convexity,
                          "two_year_zero_rate": curve.zero_rate(2)},
        "credit": {"flat_hazard": hazard, "five_year_risk_neutral_pd": pd_5y},
        "paper_ledger": {"nav": nav, "positions": book.positions(),
                         "unrealized_pnl": book.unrealized_pnl()},
    }
    print(json.dumps(report, indent=2, default=lambda value: value.__dict__ if hasattr(value, "__dict__") else str(value)))


if __name__ == "__main__":
    main()
