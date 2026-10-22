"""Deterministic portfolio stress scenarios; excludes liquidity and market-impact modeling."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from ..types import PortfolioSnapshot

@dataclass(frozen=True, slots=True)
class StressResult:
    scenario: str
    pnl: float
    pnl_fraction_of_equity: float
    shocked_values: Mapping[str, float]


def run_stress_scenario(snapshot: PortfolioSnapshot, shocks: Mapping[str, float],
                        scenario: str = "custom") -> StressResult:
    """Apply symbol-specific simple percentage price shocks to open positions.

    Shock values use decimal returns (e.g. -0.10 for -10%). Unknown symbols are
    ignored. This is a transparent linear stress, not VaR or a probability forecast.
    """
    if any(shock < -1 for shock in shocks.values()):
        raise ValueError("a price shock cannot be below -100%")
    shocked_values: dict[str, float] = {}
    pnl = 0.0
    for symbol, position in snapshot.positions.items():
        shock = float(shocks.get(symbol, 0.0))
        shocked = position.market_value * shock
        shocked_values[symbol] = position.market_value + shocked
        pnl += shocked
    fraction = pnl / snapshot.equity if snapshot.equity > 0 else 0.0
    return StressResult(scenario, pnl, fraction, shocked_values)


def run_stress_suite(snapshot: PortfolioSnapshot,
                     scenarios: Mapping[str, Mapping[str, float]]) -> tuple[StressResult, ...]:
    return tuple(run_stress_scenario(snapshot, shocks, name)
                 for name, shocks in scenarios.items())


def standard_scenarios(symbols: list[str] | tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Create simple broad-market shock scenarios keyed to a supplied universe."""
    keys = {symbol.strip().upper() for symbol in symbols}
    return {
        "broad_selloff_5pct": {symbol: -0.05 for symbol in keys},
        "broad_selloff_10pct": {symbol: -0.10 for symbol in keys},
        "mixed_rotation": {symbol: (-0.08 if i % 2 == 0 else 0.04)
                            for i, symbol in enumerate(sorted(keys))},
        "broad_rally_5pct": {symbol: 0.05 for symbol in keys},
    }
