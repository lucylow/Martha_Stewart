"""Exposure and concentration measurements from portfolio snapshots."""
from __future__ import annotations
from math import sqrt
from typing import Mapping
from ..types import PortfolioSnapshot


def exposure_report(snapshot: PortfolioSnapshot) -> dict:
    positions = snapshot.positions
    equity = snapshot.equity
    values = {symbol: position.market_value for symbol, position in positions.items()}
    gross = sum(abs(value) for value in values.values())
    net = sum(values.values())
    # Concentration is computed over invested capital, independent of cash deployment.
    # Exposure versus equity is reported separately in the top-level fields.
    gross_weights = {symbol: abs(value) / gross for symbol, value in values.items()} if gross > 0 else {}
    net_weights = {symbol: value / equity for symbol, value in values.items()} if equity > 0 else {}
    concentration = sum(weight * weight for weight in gross_weights.values())
    return {
        "position_count": len(positions), "gross_market_value": gross,
        "net_market_value": net, "gross_exposure_fraction": gross / equity if equity > 0 else float("inf"),
        "net_exposure_fraction": net / equity if equity > 0 else float("inf"),
        "largest_position_fraction": max(gross_weights.values(), default=0.0),
        "herfindahl_concentration": concentration,
        "effective_position_count": 1.0 / concentration if concentration > 0 else 0.0,
        "gross_weights": gross_weights, "net_weights": net_weights,
    }


def covariance_portfolio_volatility(weights: Mapping[str, float],
                                    covariance_matrix: Mapping[str, Mapping[str, float]]) -> float:
    """Compute sqrt(w' C w) from decimal return covariance values."""
    variance = 0.0
    for left, left_weight in weights.items():
        for right, right_weight in weights.items():
            try:
                covariance = covariance_matrix[left][right]
            except KeyError as exc:
                raise ValueError(f"covariance matrix missing {left}/{right}") from exc
            variance += left_weight * right_weight * covariance
    if variance < -1e-10:
        raise ValueError("covariance matrix produced negative portfolio variance")
    return sqrt(max(0.0, variance))
