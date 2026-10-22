"""Backtest metrics with explicit handling of empty and degenerate samples."""
from __future__ import annotations
from math import isfinite
from statistics import mean
from typing import Sequence
from ..math_utils import annualized_volatility, max_drawdown, sharpe_ratio, percentile


def calculate_metrics(equity_curve: Sequence[float], periods_per_year: float = 252.0,
                      risk_free_rate_per_period: float = 0.0) -> dict[str, float | int]:
    if not equity_curve:
        return {"observations": 0, "total_return": 0.0, "annualized_return": 0.0,
                "annualized_volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "best_period": 0.0, "worst_period": 0.0}
    clean = [float(v) for v in equity_curve]
    if any(not isfinite(v) or v <= 0 for v in clean):
        raise ValueError("equity values must be finite and positive")
    period_returns = [b / a - 1 for a, b in zip(clean, clean[1:])]
    total_return = clean[-1] / clean[0] - 1
    annualized_return = (clean[-1] / clean[0]) ** (periods_per_year / max(1, len(clean) - 1)) - 1
    volatility = annualized_volatility(period_returns, periods_per_year) if period_returns else 0.0
    drawdown, peak, trough = max_drawdown(clean)
    calmar = annualized_return / drawdown if drawdown > 0 else 0.0
    return {
        "observations": len(clean), "total_return": total_return,
        "annualized_return": annualized_return, "annualized_volatility": volatility,
        "sharpe": sharpe_ratio(period_returns, risk_free_rate_per_period, periods_per_year),
        "max_drawdown": drawdown, "drawdown_peak_index": peak,
        "drawdown_trough_index": trough, "calmar": calmar,
        "best_period": max(period_returns, default=0.0),
        "worst_period": min(period_returns, default=0.0),
        "average_period_return": mean(period_returns) if period_returns else 0.0,
    }


def trade_statistics(realized_trade_pnls: Sequence[float]) -> dict[str, float | int]:
    pnls = [float(p) for p in realized_trade_pnls if isfinite(float(p))]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    gross_profit = sum(winners)
    gross_loss = -sum(losers)
    return {
        "trade_count": len(pnls), "win_count": len(winners), "loss_count": len(losers),
        "win_rate": len(winners) / len(pnls) if pnls else 0.0,
        "average_trade_pnl": mean(pnls) if pnls else 0.0,
        "median_trade_pnl": percentile(pnls, 50) if pnls else 0.0,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "largest_win": max(pnls, default=0.0), "largest_loss": min(pnls, default=0.0),
    }
