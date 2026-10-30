"""Transparent transaction-cost and market-impact models for research."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import Iterable

from ..types import Side, SymbolSpec


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    spread_bps: Decimal
    commission_bps: Decimal
    temporary_impact_bps: Decimal
    permanent_impact_bps: Decimal
    delay_bps: Decimal
    total_bps: Decimal
    notional: Decimal
    total_cost: Decimal


@dataclass(frozen=True, slots=True)
class ImpactParameters:
    eta: Decimal = Decimal("0.10")
    gamma: Decimal = Decimal("0.05")
    volatility_floor: Decimal = Decimal("0.0001")
    max_impact_bps: Decimal = Decimal("500")

    def __post_init__(self) -> None:
        if self.eta < 0 or self.gamma < 0 or self.volatility_floor <= 0 or self.max_impact_bps <= 0:
            raise ValueError("invalid impact parameters")


class SquareRootImpactModel:
    """Estimate impact from participation, volatility, and a liquidity scale.

    The square-root relationship is a stylized model, not a calibrated forecast.
    Inputs are explicit so users can stress rather than mistake estimates for
    measured market impact.
    """

    def __init__(self, parameters: ImpactParameters | None = None) -> None:
        self.parameters = parameters or ImpactParameters()

    def temporary_bps(self, *, quantity: int, average_daily_volume: int,
                      volatility: Decimal | float) -> Decimal:
        if quantity < 0 or average_daily_volume <= 0:
            raise ValueError("quantity must be non-negative and ADV positive")
        sigma = max(Decimal(str(volatility)), self.parameters.volatility_floor)
        participation = Decimal(quantity) / Decimal(average_daily_volume)
        value = self.parameters.eta * sigma * Decimal(str(sqrt(float(participation))) ) * Decimal(10000)
        return min(self.parameters.max_impact_bps, value)

    def permanent_bps(self, *, quantity: int, average_daily_volume: int,
                      volatility: Decimal | float) -> Decimal:
        if quantity < 0 or average_daily_volume <= 0:
            raise ValueError("quantity must be non-negative and ADV positive")
        sigma = max(Decimal(str(volatility)), self.parameters.volatility_floor)
        participation = Decimal(quantity) / Decimal(average_daily_volume)
        value = self.parameters.gamma * sigma * Decimal(str(sqrt(float(participation)))) * Decimal(10000)
        return min(self.parameters.max_impact_bps, value)


class LinearImpactModel:
    """Alternative linear participation model for sensitivity checks."""

    def __init__(self, coefficient_bps: Decimal = Decimal("10")) -> None:
        if coefficient_bps < 0:
            raise ValueError("coefficient must be non-negative")
        self.coefficient_bps = coefficient_bps

    def impact_bps(self, quantity: int, average_daily_volume: int) -> Decimal:
        if quantity < 0 or average_daily_volume <= 0:
            raise ValueError("invalid quantity or ADV")
        return self.coefficient_bps * Decimal(quantity) / Decimal(average_daily_volume)


@dataclass(frozen=True, slots=True)
class ExecutionCostInputs:
    side: Side
    quantity: int
    reference_price_ticks: int
    arrival_mid_ticks: Decimal
    spread_ticks: Decimal
    average_daily_volume: int
    volatility: Decimal
    commission_per_share: Decimal = Decimal("0")
    delay_bps: Decimal = Decimal("0")
    permanent_impact_fraction: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        if self.quantity < 0 or self.reference_price_ticks <= 0 or self.arrival_mid_ticks <= 0:
            raise ValueError("invalid quantity or reference price")
        if self.spread_ticks < 0 or self.average_daily_volume <= 0:
            raise ValueError("spread must be non-negative and ADV positive")
        if self.commission_per_share < 0 or self.delay_bps < 0:
            raise ValueError("fees and delay costs must be non-negative")
        if not 0 <= self.permanent_impact_fraction <= 1:
            raise ValueError("permanent impact fraction must be in [0, 1]")


class ExecutionCostEstimator:
    """Compose spread, fees, impact, and delay into one auditable estimate."""

    def __init__(self, spec: SymbolSpec, impact: SquareRootImpactModel | None = None) -> None:
        self.spec = spec
        self.impact = impact or SquareRootImpactModel()

    def estimate(self, inputs: ExecutionCostInputs) -> CostBreakdown:
        if inputs.quantity == 0:
            zero = Decimal(0)
            return CostBreakdown(zero, zero, zero, zero, zero, zero, zero, zero)
        notional = self.spec.money(inputs.reference_price_ticks, inputs.quantity)
        spread_bps = (inputs.spread_ticks / inputs.arrival_mid_ticks) * Decimal(5000)
        commission = inputs.commission_per_share * Decimal(inputs.quantity * self.spec.lot_size)
        commission_bps = commission / notional * Decimal(10000) if notional else Decimal(0)
        temporary = self.impact.temporary_bps(quantity=inputs.quantity,
                                               average_daily_volume=inputs.average_daily_volume,
                                               volatility=inputs.volatility)
        permanent = self.impact.permanent_bps(quantity=inputs.quantity,
                                               average_daily_volume=inputs.average_daily_volume,
                                               volatility=inputs.volatility) * inputs.permanent_impact_fraction
        total = spread_bps + commission_bps + temporary + permanent + inputs.delay_bps
        cost = notional * total / Decimal(10000)
        return CostBreakdown(spread_bps, commission_bps, temporary,
                             permanent, inputs.delay_bps, total, notional, cost)


@dataclass(frozen=True, slots=True)
class SlippageSample:
    ts_ns: int
    side: Side
    arrival_mid_ticks: Decimal
    fill_price_ticks: Decimal
    quantity: int

    def __post_init__(self) -> None:
        if self.ts_ns < 0 or self.quantity <= 0 or min(self.arrival_mid_ticks, self.fill_price_ticks) <= 0:
            raise ValueError("invalid slippage sample")

    @property
    def signed_slippage_bps(self) -> Decimal:
        return ((self.fill_price_ticks - self.arrival_mid_ticks) * self.side.sign /
                self.arrival_mid_ticks * Decimal(10000))


@dataclass(frozen=True, slots=True)
class SlippageSummary:
    samples: int
    quantity: int
    mean_bps: Decimal | None
    weighted_bps: Decimal | None
    worst_bps: Decimal | None
    best_bps: Decimal | None


class SlippageAccumulator:
    """Accumulate order-level implementation-shortfall observations."""

    def __init__(self) -> None:
        self._samples: list[SlippageSample] = []

    def add(self, sample: SlippageSample) -> None:
        self._samples.append(sample)

    def extend(self, samples: Iterable[SlippageSample]) -> None:
        self._samples.extend(samples)

    def summary(self) -> SlippageSummary:
        if not self._samples:
            return SlippageSummary(0, 0, None, None, None, None)
        values = [x.signed_slippage_bps for x in self._samples]
        quantity = sum(x.quantity for x in self._samples)
        weighted = sum((x.signed_slippage_bps * x.quantity for x in self._samples), Decimal(0)) / quantity
        return SlippageSummary(len(values), quantity,
                               sum(values, Decimal(0)) / len(values), weighted,
                               max(values), min(values))

    def clear(self) -> None:
        self._samples.clear()
