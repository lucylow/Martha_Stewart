"""Portfolio scenario analysis using explicit shocks and deterministic outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: float
    price: float
    currency: str = "USD"
    beta: float = 1.0
    sector: str = "unknown"

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.currency.strip():
            raise ValueError("symbol and currency are required")
        if any(not isfinite(v) for v in (self.quantity, self.price, self.beta)):
            raise ValueError("position inputs must be finite")
        if self.price < 0:
            raise ValueError("price cannot be negative")

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    shocks: Mapping[str, float] = field(default_factory=dict)
    sector_shocks: Mapping[str, float] = field(default_factory=dict)
    default_shock: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id is required")
        for value in list(self.shocks.values()) + list(self.sector_shocks.values()) + [self.default_shock]:
            if not isfinite(value) or value < -1:
                raise ValueError("price shocks must be finite and no less than -100%")


@dataclass(frozen=True, slots=True)
class ScenarioLine:
    symbol: str
    base_value: float
    shock: float
    shocked_value: float
    pnl: float
    pnl_percent: float


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario_id: str
    base_value: float
    shocked_value: float
    pnl: float
    pnl_percent: float
    lines: tuple[ScenarioLine, ...]
    worst_position: str | None
    best_position: str | None


class ScenarioEngine:
    def evaluate(self, positions: Sequence[Position], scenario: Scenario) -> ScenarioResult:
        lines = []
        for pos in positions:
            shock = scenario.shocks.get(pos.symbol, scenario.sector_shocks.get(pos.sector, scenario.default_shock))
            pnl = pos.market_value * shock
            shocked = pos.market_value + pnl
            lines.append(ScenarioLine(pos.symbol, pos.market_value, shock, shocked, pnl,
                                      pnl / abs(pos.market_value) if pos.market_value else 0.0))
        base = sum(p.market_value for p in positions)
        shocked_total = sum(line.shocked_value for line in lines)
        pnl_total = shocked_total - base
        worst = min(lines, key=lambda x: x.pnl).symbol if lines else None
        best = max(lines, key=lambda x: x.pnl).symbol if lines else None
        return ScenarioResult(scenario.scenario_id, base, shocked_total, pnl_total,
            pnl_total / abs(base) if base else 0.0, tuple(lines), worst, best)

    def evaluate_many(self, positions: Sequence[Position], scenarios: Sequence[Scenario]) -> tuple[ScenarioResult, ...]:
        ids = [s.scenario_id for s in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return tuple(self.evaluate(positions, scenario) for scenario in scenarios)

    def parametric_shock(self, positions: Sequence[Position], *, market_shock: float,
                         beta_adjusted: bool = True, scenario_id: str = "market_shock") -> ScenarioResult:
        if market_shock < -1 or not isfinite(market_shock):
            raise ValueError("market shock must be finite and no less than -100%")
        adjusted = []
        shocks = {}
        for pos in positions:
            beta = pos.beta if beta_adjusted else 1.0
            shocks[pos.symbol] = max(-1.0, market_shock * beta)
        return self.evaluate(positions, Scenario(scenario_id, shocks=shocks,
                                                  description="beta-adjusted market shock"))

    def historical_replay(self, positions: Sequence[Position],
                          price_returns: Sequence[Mapping[str, float]]) -> tuple[ScenarioResult, ...]:
        scenarios = []
        for index, returns in enumerate(price_returns):
            scenarios.append(Scenario(f"historical-{index:05d}", shocks=dict(returns),
                                      description="historical cross-sectional return replay"))
        return self.evaluate_many(positions, scenarios)


def expected_shortfall(pnls: Sequence[float], *, confidence: float = 0.95) -> float:
    """Historical expected shortfall expressed as a positive loss number."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not pnls or any(not isfinite(v) for v in pnls):
        raise ValueError("pnls must contain finite values")
    ordered = sorted(pnls)
    count = max(1, int((1.0 - confidence) * len(ordered) + 0.999999))
    tail = ordered[:count]
    return max(0.0, -sum(tail) / len(tail))


def scenario_breach(result: ScenarioResult, *, max_loss_fraction: float) -> bool:
    if not 0 <= max_loss_fraction <= 1:
        raise ValueError("max_loss_fraction must be between 0 and 1")
    return result.pnl_percent < -max_loss_fraction
