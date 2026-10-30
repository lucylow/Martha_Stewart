"""Deterministic portfolio and market stress scenario definitions."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping

from ..types import Side


class ShockType(str, Enum):
    PRICE_RETURN = "price_return"
    SPREAD_MULTIPLIER = "spread_multiplier"
    DEPTH_MULTIPLIER = "depth_multiplier"
    VOLATILITY_MULTIPLIER = "volatility_multiplier"
    FEE_BPS = "fee_bps"
    FUNDING_BPS = "funding_bps"


@dataclass(frozen=True, slots=True)
class MarketShock:
    symbol: str
    shock_type: ShockType
    magnitude: Decimal

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.magnitude.is_finite():
            raise ValueError("invalid shock")
        if self.shock_type in {ShockType.SPREAD_MULTIPLIER, ShockType.DEPTH_MULTIPLIER,
                               ShockType.VOLATILITY_MULTIPLIER} and self.magnitude < 0:
            raise ValueError("multipliers must be non-negative")


@dataclass(frozen=True, slots=True)
class StressScenario:
    name: str
    description: str
    shocks: tuple[MarketShock, ...]
    global_cash_haircut: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.name.strip() or self.global_cash_haircut < 0:
            raise ValueError("invalid scenario")


@dataclass(frozen=True, slots=True)
class StressResult:
    scenario: str
    base_value: Decimal
    stressed_value: Decimal
    pnl_change: Decimal
    cash_haircut: Decimal
    per_symbol: tuple[tuple[str, Decimal], ...]


class StressEngine:
    """Apply simple first-order shocks to marked positions.

    Price-return shocks are applied directly to marked position values.  Spread,
    depth, volatility, fees, and funding shocks are returned as market-state
    diagnostics and do not silently become P&L assumptions.
    """

    def run(self, positions: Mapping[str, int], marks: Mapping[str, Decimal],
            scenario: StressScenario, *, cash: Decimal = Decimal(0)) -> StressResult:
        if cash < 0:
            raise ValueError("cash must be non-negative")
        base = Decimal(0)
        shocked = Decimal(0)
        per_symbol = []
        shock_by_symbol = {x.symbol.upper(): x for x in scenario.shocks
                           if x.shock_type is ShockType.PRICE_RETURN}
        for symbol, quantity in positions.items():
            key = symbol.upper()
            if key not in marks:
                raise KeyError(f"missing mark for {key}")
            mark = Decimal(marks[key])
            if mark <= 0:
                raise ValueError("marks must be positive")
            value = mark * quantity
            shock = shock_by_symbol.get(key)
            return_shock = shock.magnitude if shock else Decimal(0)
            if return_shock <= -1:
                raise ValueError("price-return shock cannot be <= -100%")
            stressed = value * (Decimal(1) + return_shock)
            base += value
            shocked += stressed
            per_symbol.append((key, stressed - value))
        cash_haircut = cash * scenario.global_cash_haircut
        total_base = base + cash
        total_stressed = shocked + cash - cash_haircut
        return StressResult(scenario.name, total_base, total_stressed,
                            total_stressed - total_base, cash_haircut,
                            tuple(sorted(per_symbol)))


class ScenarioLibrary:
    """Small transparent scenario library; magnitudes are illustrative only."""

    @staticmethod
    def equity_gap_down(symbols: tuple[str, ...], return_shock: Decimal = Decimal("-0.08")) -> StressScenario:
        return StressScenario("equity_gap_down", "Illustrative broad-market gap lower",
                              tuple(MarketShock(symbol, ShockType.PRICE_RETURN, return_shock)
                                    for symbol in symbols))

    @staticmethod
    def liquidity_withdrawal(symbols: tuple[str, ...], spread_multiple: Decimal = Decimal("4"),
                             depth_multiple: Decimal = Decimal("0.2")) -> StressScenario:
        shocks = []
        for symbol in symbols:
            shocks.extend((MarketShock(symbol, ShockType.SPREAD_MULTIPLIER, spread_multiple),
                           MarketShock(symbol, ShockType.DEPTH_MULTIPLIER, depth_multiple)))
        return StressScenario("liquidity_withdrawal", "Wider spreads and reduced displayed depth",
                              tuple(shocks))

    @staticmethod
    def volatility_spike(symbols: tuple[str, ...], multiplier: Decimal = Decimal("3")) -> StressScenario:
        return StressScenario("volatility_spike", "Illustrative volatility regime change",
                              tuple(MarketShock(symbol, ShockType.VOLATILITY_MULTIPLIER, multiplier)
                                    for symbol in symbols))


@dataclass(frozen=True, slots=True)
class ReverseStressResult:
    target_loss: Decimal
    required_return: Decimal
    feasible: bool
    limiting_symbol: str | None


def reverse_stress_single_asset(position_value: Decimal, target_loss: Decimal) -> ReverseStressResult:
    """Solve the single-asset return shock required to reach a loss threshold."""
    if position_value == 0 or target_loss < 0:
        raise ValueError("position_value must be non-zero and loss target non-negative")
    required = -target_loss / position_value
    return ReverseStressResult(target_loss, required, required > -1, None)


def factor_shock_pnl(exposures: Mapping[str, Decimal], factor_returns: Mapping[str, Decimal]) -> Decimal:
    """Compute a linear factor P&L approximation for supplied exposures."""
    pnl = Decimal(0)
    for factor, exposure in exposures.items():
        if factor not in factor_returns:
            raise KeyError(f"missing factor return for {factor}")
        pnl += Decimal(exposure) * Decimal(factor_returns[factor])
    return pnl
