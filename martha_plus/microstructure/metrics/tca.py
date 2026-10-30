"""Transaction-cost and execution-quality analysis for simulated fills."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean
from typing import Iterable, Mapping

from ..types import Execution, Side, SymbolSpec


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    order_id: str
    strategy_id: str
    side: Side
    requested_quantity: int
    filled_quantity: int
    cancelled_quantity: int
    average_fill_ticks: Decimal | None
    arrival_mid_ticks: Decimal | None
    fees: Decimal
    first_fill_ns: int | None
    last_fill_ns: int | None

    @property
    def fill_ratio(self) -> Decimal:
        if self.requested_quantity <= 0:
            return Decimal("0")
        return Decimal(self.filled_quantity) / Decimal(self.requested_quantity)


@dataclass(frozen=True, slots=True)
class ExecutionQualityReport:
    order_count: int
    filled_order_count: int
    fill_ratio: Decimal
    total_filled_quantity: int
    total_notional: Decimal
    fees: Decimal
    maker_quantity: int
    taker_quantity: int
    maker_fraction: Decimal
    mean_slippage_bps: Decimal | None
    median_slippage_bps: Decimal | None
    outcomes: tuple[OrderOutcome, ...]

    def to_dict(self) -> dict:
        return {
            "order_count": self.order_count,
            "filled_order_count": self.filled_order_count,
            "fill_ratio": str(self.fill_ratio),
            "total_filled_quantity": self.total_filled_quantity,
            "total_notional": str(self.total_notional),
            "fees": str(self.fees),
            "maker_quantity": self.maker_quantity,
            "taker_quantity": self.taker_quantity,
            "maker_fraction": str(self.maker_fraction),
            "mean_slippage_bps": str(self.mean_slippage_bps) if self.mean_slippage_bps is not None else None,
            "median_slippage_bps": str(self.median_slippage_bps) if self.median_slippage_bps is not None else None,
            "outcomes": [{
                "order_id": row.order_id, "strategy_id": row.strategy_id, "side": row.side.value,
                "requested_quantity": row.requested_quantity, "filled_quantity": row.filled_quantity,
                "cancelled_quantity": row.cancelled_quantity,
                "average_fill_ticks": str(row.average_fill_ticks) if row.average_fill_ticks is not None else None,
                "arrival_mid_ticks": str(row.arrival_mid_ticks) if row.arrival_mid_ticks is not None else None,
                "fees": str(row.fees), "first_fill_ns": row.first_fill_ns, "last_fill_ns": row.last_fill_ns,
            } for row in self.outcomes],
        }


class ExecutionQualityAnalyzer:
    """Joins order lifecycle records, fills, and arrival midpoints."""

    def __init__(self, spec: SymbolSpec) -> None:
        self.spec = spec

    def analyze(self, orders: Iterable, executions: Iterable[Execution], arrival_midpoints: Mapping[str, Decimal | int] | None = None) -> ExecutionQualityReport:
        rows = list(orders)
        fills = list(executions)
        arrival_midpoints = arrival_midpoints or {}
        by_order: dict[str, list[Execution]] = {}
        for execution in fills:
            by_order.setdefault(execution.maker_order_id, []).append(execution)
            by_order.setdefault(execution.taker_order_id, []).append(execution)
        outcomes: list[OrderOutcome] = []
        slippage_values: list[Decimal] = []
        total_filled = 0
        fees = Decimal("0")
        maker_qty = 0
        taker_qty = 0
        total_notional = Decimal("0")
        filled_order_count = 0
        for record in rows:
            request = record.request
            order_fills = [x for x in by_order.get(request.client_order_id, ())
                           if x.maker_order_id == request.client_order_id or x.taker_order_id == request.client_order_id]
            # A client order can be maker or taker, but the above explicit OR
            # protects against accidental inclusion when ids overlap in a bad fixture.
            filled_qty = sum(x.quantity for x in order_fills)
            if filled_qty:
                filled_order_count += 1
            notional_ticks = sum(x.price_ticks * x.quantity for x in order_fills)
            average = Decimal(notional_ticks) / Decimal(filled_qty) if filled_qty else None
            arrival = arrival_midpoints.get(request.client_order_id)
            arrival_decimal = Decimal(arrival) if arrival is not None else None
            order_fees = Decimal("0")
            first_fill = min((x.ts_ns for x in order_fills), default=None)
            last_fill = max((x.ts_ns for x in order_fills), default=None)
            for execution in order_fills:
                if execution.maker_order_id == request.client_order_id:
                    order_fees += execution.maker_fee
                    maker_qty += execution.quantity
                else:
                    order_fees += execution.taker_fee
                    taker_qty += execution.quantity
                total_notional += self.spec.money(execution.price_ticks, execution.quantity)
            fees += order_fees
            total_filled += filled_qty
            if average is not None and arrival_decimal is not None and arrival_decimal > 0:
                signed_move = (average - arrival_decimal) * Decimal(request.side.sign)
                slippage_values.append(signed_move / arrival_decimal * Decimal("10000"))
            outcomes.append(OrderOutcome(
                request.client_order_id, request.strategy_id, request.side,
                request.quantity, filled_qty, record.cancelled_quantity,
                average, arrival_decimal, order_fees, first_fill, last_fill,
            ))
        requested = sum(x.request.quantity for x in rows)
        fill_ratio = Decimal(total_filled) / Decimal(requested) if requested else Decimal("0")
        total_maker_taker = maker_qty + taker_qty
        maker_fraction = Decimal(maker_qty) / Decimal(total_maker_taker) if total_maker_taker else Decimal("0")
        sorted_slippage = sorted(slippage_values)
        median = None
        if sorted_slippage:
            middle = len(sorted_slippage) // 2
            median = sorted_slippage[middle] if len(sorted_slippage) % 2 else (sorted_slippage[middle - 1] + sorted_slippage[middle]) / 2
        return ExecutionQualityReport(
            order_count=len(rows), filled_order_count=filled_order_count,
            fill_ratio=fill_ratio, total_filled_quantity=total_filled,
            total_notional=total_notional, fees=fees,
            maker_quantity=maker_qty, taker_quantity=taker_qty,
            maker_fraction=maker_fraction,
            mean_slippage_bps=Decimal(str(fmean(slippage_values))) if slippage_values else None,
            median_slippage_bps=median, outcomes=tuple(outcomes),
        )


def markout_bps(side: Side, fill_price_ticks: Decimal | int, future_mid_ticks: Decimal | int) -> Decimal:
    """Signed post-fill markout; negative means adverse selection for the fill."""
    fill = Decimal(fill_price_ticks)
    future = Decimal(future_mid_ticks)
    if fill <= 0:
        raise ValueError("fill price must be positive")
    return (future - fill) * Decimal(side.sign) / fill * Decimal("10000")


def implementation_shortfall_bps(side: Side, arrival_mid_ticks: Decimal | int, fill_price_ticks: Decimal | int) -> Decimal:
    arrival, fill = Decimal(arrival_mid_ticks), Decimal(fill_price_ticks)
    if arrival <= 0:
        raise ValueError("arrival midpoint must be positive")
    return (fill - arrival) * Decimal(side.sign) / arrival * Decimal("10000")
