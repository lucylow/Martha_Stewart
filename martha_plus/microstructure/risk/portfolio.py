"""Portfolio-level exposure, P&L, and drawdown controls for research runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping

from ..types import Side, SymbolSpec


class RiskSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: int
    average_cost_ticks: Decimal
    realized_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.symbol.strip() or self.average_cost_ticks < 0 or self.fees_paid < 0:
            raise ValueError("invalid position")

    @property
    def side(self) -> Side | None:
        return Side.BUY if self.quantity > 0 else Side.SELL if self.quantity < 0 else None


@dataclass(frozen=True, slots=True)
class Mark:
    symbol: str
    price_ticks: Decimal
    ts_ns: int

    def __post_init__(self) -> None:
        if not self.symbol.strip() or self.price_ticks <= 0 or self.ts_ns < 0:
            raise ValueError("invalid mark")


@dataclass(frozen=True, slots=True)
class PositionValuation:
    symbol: str
    quantity: int
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal
    total_pnl: Decimal


class PositionBook:
    """Track signed positions and weighted-average cost in integer ticks."""

    def __init__(self, specs: Mapping[str, SymbolSpec]) -> None:
        if not specs:
            raise ValueError("at least one symbol spec is required")
        self.specs = {key.upper(): value for key, value in specs.items()}
        self._positions: dict[str, Position] = {}
        self._marks: dict[str, Mark] = {}

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol.upper(), Position(symbol.upper(), 0, Decimal(0)))

    def mark(self, mark: Mark) -> None:
        symbol = mark.symbol.upper()
        if symbol not in self.specs:
            raise KeyError(symbol)
        previous = self._marks.get(symbol)
        if previous and mark.ts_ns < previous.ts_ns:
            raise ValueError("mark timestamps must be non-decreasing")
        self._marks[symbol] = mark

    def apply_fill(self, symbol: str, side: Side, quantity: int, price_ticks: int,
                   fee: Decimal = Decimal(0)) -> Position:
        key = symbol.upper()
        if key not in self.specs:
            raise KeyError(key)
        if quantity <= 0 or price_ticks <= 0 or fee < 0:
            raise ValueError("invalid fill")
        old = self.position(key)
        signed = quantity * side.sign
        new_qty = old.quantity + signed
        realized = old.realized_pnl
        average = old.average_cost_ticks
        if old.quantity == 0 or old.quantity * signed > 0:
            total_abs = abs(old.quantity) + quantity
            average = (average * abs(old.quantity) + Decimal(price_ticks * quantity)) / Decimal(total_abs)
        else:
            closed = min(abs(old.quantity), quantity)
            direction = 1 if old.quantity > 0 else -1
            realized += Decimal(closed) * (Decimal(price_ticks) - average) * direction * self.specs[key].tick_size * self.specs[key].lot_size
            if new_qty == 0:
                average = Decimal(0)
            elif old.quantity * new_qty < 0:
                average = Decimal(price_ticks)
        updated = Position(key, new_qty, average, realized, old.fees_paid + fee)
        self._positions[key] = updated
        return updated

    def valuation(self, symbol: str, *, require_mark: bool = True) -> PositionValuation:
        key = symbol.upper()
        position = self.position(key)
        mark = self._marks.get(key)
        if mark is None:
            if require_mark and position.quantity != 0:
                raise KeyError(f"missing mark for {key}")
            price = position.average_cost_ticks
        else:
            price = mark.price_ticks
        spec = self.specs[key]
        market_value = price * Decimal(position.quantity) * spec.tick_size * spec.lot_size
        unrealized = (price - position.average_cost_ticks) * Decimal(position.quantity) * spec.tick_size * spec.lot_size
        total = position.realized_pnl + unrealized - position.fees_paid
        return PositionValuation(key, position.quantity, market_value, unrealized,
                                 position.realized_pnl, position.fees_paid, total)

    def valuations(self) -> tuple[PositionValuation, ...]:
        symbols = set(self._positions) | set(self._marks)
        return tuple(self.valuation(symbol) for symbol in sorted(symbols))

    def gross_exposure(self) -> Decimal:
        return sum((abs(x.market_value) for x in self.valuations()), Decimal(0))

    def net_exposure(self) -> Decimal:
        return sum((x.market_value for x in self.valuations()), Decimal(0))

    def total_pnl(self) -> Decimal:
        return sum((x.total_pnl for x in self.valuations()), Decimal(0))

    def snapshot(self) -> dict[str, dict[str, str | int]]:
        return {
            symbol: {
                "quantity": row.quantity,
                "average_cost_ticks": str(row.average_cost_ticks),
                "realized_pnl": str(row.realized_pnl),
                "fees_paid": str(row.fees_paid),
                "mark_ticks": str(self._marks[symbol].price_ticks) if symbol in self._marks else "",
            }
            for symbol, row in sorted(self._positions.items())
        }


@dataclass(frozen=True, slots=True)
class RiskLimit:
    name: str
    threshold: Decimal
    severity: RiskSeverity = RiskSeverity.BLOCK
    comparison: str = "lte"

    def __post_init__(self) -> None:
        if not self.name.strip() or self.threshold < 0:
            raise ValueError("invalid risk limit")
        if self.comparison not in {"lte", "gte"}:
            raise ValueError("comparison must be lte or gte")


@dataclass(frozen=True, slots=True)
class RiskBreach:
    limit_name: str
    observed: Decimal
    threshold: Decimal
    severity: RiskSeverity
    message: str


class PortfolioRiskMonitor:
    """Evaluate a set of transparent gross/net/loss limits."""

    def __init__(self, limits: tuple[RiskLimit, ...]) -> None:
        names = [x.name for x in limits]
        if len(names) != len(set(names)):
            raise ValueError("risk limit names must be unique")
        self.limits = limits

    def evaluate(self, measures: Mapping[str, Decimal | int | float]) -> tuple[RiskBreach, ...]:
        breaches = []
        for limit in self.limits:
            if limit.name not in measures:
                continue
            value = Decimal(str(measures[limit.name]))
            failed = value > limit.threshold if limit.comparison == "lte" else value < limit.threshold
            if failed:
                breaches.append(RiskBreach(limit.name, value, limit.threshold,
                                           limit.severity,
                                           f"{limit.name} observed {value} violates {limit.comparison} {limit.threshold}"))
        return tuple(breaches)


@dataclass(frozen=True, slots=True)
class DrawdownObservation:
    ts_ns: int
    equity: Decimal
    peak_equity: Decimal
    drawdown: Decimal
    drawdown_fraction: Decimal


class DrawdownTracker:
    """Track peak-to-trough equity drawdown without assuming annualized returns."""

    def __init__(self, initial_equity: Decimal = Decimal(0)) -> None:
        if initial_equity < 0:
            raise ValueError("initial_equity must be non-negative")
        self.peak = initial_equity
        self.trough = initial_equity
        self.maximum_drawdown = Decimal(0)
        self.maximum_fraction = Decimal(0)
        self.history: list[DrawdownObservation] = []

    def update(self, ts_ns: int, equity: Decimal | int | float) -> DrawdownObservation:
        value = Decimal(str(equity))
        if ts_ns < 0 or value < 0:
            raise ValueError("timestamp and equity must be non-negative")
        self.peak = max(self.peak, value)
        self.trough = min(self.trough, value) if value < self.peak else self.peak
        amount = self.peak - value
        fraction = amount / self.peak if self.peak > 0 else Decimal(0)
        self.maximum_drawdown = max(self.maximum_drawdown, amount)
        self.maximum_fraction = max(self.maximum_fraction, fraction)
        row = DrawdownObservation(ts_ns, value, self.peak, amount, fraction)
        self.history.append(row)
        return row

    def reset(self, initial_equity: Decimal = Decimal(0)) -> None:
        if initial_equity < 0:
            raise ValueError("initial equity must be non-negative")
        self.peak = self.trough = initial_equity
        self.maximum_drawdown = self.maximum_fraction = Decimal(0)
        self.history.clear()


@dataclass(frozen=True, slots=True)
class ConcentrationReport:
    total_absolute_value: Decimal
    largest_weight: Decimal
    top_three_weight: Decimal
    weights: tuple[tuple[str, Decimal], ...]


def concentration_report(exposures: Mapping[str, Decimal]) -> ConcentrationReport:
    cleaned = {key: abs(Decimal(str(value))) for key, value in exposures.items()}
    total = sum(cleaned.values(), Decimal(0))
    ordered = sorted(cleaned.items(), key=lambda item: item[1], reverse=True)
    weights = tuple((key, value / total if total else Decimal(0)) for key, value in ordered)
    return ConcentrationReport(total, weights[0][1] if weights else Decimal(0),
                               sum((x[1] for x in weights[:3]), Decimal(0)), weights)
