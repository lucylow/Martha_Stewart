"""Technical-finance research extension for the Martha_Stewart project.

The package is standard-library-only and intentionally separates analytical
models from data vendors, brokerage connectivity and order routing.
"""
from .core import CashFlow, DayCount, Estimate, FinanceInputError, ModelResult, ModelWarning, NumericalError
from .returns import (annualized_return, annualized_volatility, compound,
                      drawdown_episodes, drawdown_series, log_returns,
                      maximum_drawdown, simple_returns, wealth_index)
from .performance import (calmar_ratio, information_ratio, omega_ratio,
                          performance_report, sharpe_ratio, sortino_ratio,
                          tracking_error)
from .risk import (cornish_fisher_var, gaussian_expected_shortfall,
                   gaussian_var, historical_expected_shortfall, historical_var)
from .portfolio import (component_risk_contribution, covariance_matrix,
                        minimum_variance_weights, portfolio_variance,
                        risk_parity_weights)

__version__ = "0.1.0"

__all__ = [
    "CashFlow", "DayCount", "Estimate", "FinanceInputError", "ModelResult",
    "ModelWarning", "NumericalError", "annualized_return",
    "annualized_volatility", "compound", "drawdown_episodes",
    "drawdown_series", "log_returns", "maximum_drawdown", "simple_returns",
    "wealth_index", "calmar_ratio", "information_ratio", "omega_ratio",
    "performance_report", "sharpe_ratio", "sortino_ratio", "tracking_error",
    "cornish_fisher_var", "gaussian_expected_shortfall", "gaussian_var",
    "historical_expected_shortfall", "historical_var", "component_risk_contribution",
    "covariance_matrix", "minimum_variance_weights", "portfolio_variance",
    "risk_parity_weights",
]
