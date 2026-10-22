"""Portfolio/order reconciliation invariants for periodic operator review."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
from ..types import Fill, Order, OrderStatus, Position

@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    code: str
    message: str
    severity: str = "error"

@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    ok: bool
    issues: tuple[ReconciliationIssue, ...]
    order_count: int
    fill_count: int
    position_count: int


def reconcile(orders: Sequence[Order], fills: Sequence[Fill],
              positions: Mapping[str, Position], tolerance: float = 1e-8) -> ReconciliationReport:
    issues: list[ReconciliationIssue] = []
    order_by_id = {order.order_id: order for order in orders}
    if len(order_by_id) != len(orders):
        issues.append(ReconciliationIssue("duplicate_order_id", "duplicate internal order id"))
    fill_qty: dict[str, float] = {}
    fill_notional: dict[str, float] = {}
    seen_fills: set[str] = set()
    for fill in fills:
        if fill.fill_id in seen_fills:
            issues.append(ReconciliationIssue("duplicate_fill_id", f"duplicate fill id {fill.fill_id}"))
        seen_fills.add(fill.fill_id)
        order = order_by_id.get(fill.order_id)
        if order is None:
            issues.append(ReconciliationIssue("orphan_fill", f"fill {fill.fill_id} references unknown order"))
            continue
        if fill.symbol != order.request.symbol or fill.side != order.request.side:
            issues.append(ReconciliationIssue("fill_order_mismatch", f"fill {fill.fill_id} differs from order"))
        fill_qty[fill.order_id] = fill_qty.get(fill.order_id, 0.0) + fill.quantity
        fill_notional[fill.order_id] = fill_notional.get(fill.order_id, 0.0) + fill.quantity * fill.price
    for order in orders:
        quantity = fill_qty.get(order.order_id, 0.0)
        if quantity - order.request.quantity > tolerance:
            issues.append(ReconciliationIssue("overfill", f"order {order.order_id} overfilled"))
        if abs(quantity - order.filled_quantity) > tolerance:
            issues.append(ReconciliationIssue("filled_quantity_mismatch",
                                              f"order {order.order_id} fill ledger differs from order state"))
        if order.status is OrderStatus.FILLED and abs(quantity - order.request.quantity) > tolerance:
            issues.append(ReconciliationIssue("filled_status_mismatch",
                                              f"filled order {order.order_id} lacks complete fills"))
    for symbol, position in positions.items():
        if symbol != position.symbol:
            issues.append(ReconciliationIssue("position_symbol_mismatch", f"position key differs for {symbol}"))
        if abs(position.quantity) <= tolerance:
            issues.append(ReconciliationIssue("zero_position_retained", f"zero quantity retained for {symbol}", "warning"))
        if position.average_cost <= 0 or position.last_price <= 0:
            issues.append(ReconciliationIssue("invalid_position_price", f"invalid price for {symbol}"))
    errors = [issue for issue in issues if issue.severity == "error"]
    return ReconciliationReport(not errors, tuple(issues), len(orders), len(fills), len(positions))
