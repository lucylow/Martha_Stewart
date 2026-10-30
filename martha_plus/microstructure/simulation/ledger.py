"""Per-strategy cash, inventory, average-cost basis, and realized P&L."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from ..types import Execution, Side, SymbolSpec


@dataclass(slots=True)
class InventoryState:
    quantity: int = 0
    average_price_ticks: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    fills: int = 0

    @property
    def net_realized_pnl(self) -> Decimal:
        return self.realized_pnl - self.fees_paid


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    owner: str
    quantity: int
    average_price_ticks: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal
    cash: Decimal
    mark_to_market: Decimal
    equity: Decimal


class PortfolioLedger:
    """Average-cost ledger updated from simulated executions, not order intent."""

    def __init__(self, spec: SymbolSpec, starting_cash: Decimal = Decimal("0")) -> None:
        self.spec = spec
        self.starting_cash = starting_cash
        self._states: dict[str, InventoryState] = {}
        self.execution_ids: set[str] = set()

    def state(self, owner: str) -> InventoryState:
        return self._states.setdefault(owner, InventoryState(cash=self.starting_cash))

    def _apply_fill(self, owner: str, side: Side, quantity: int, price_ticks: int, fee: Decimal) -> None:
        state = self.state(owner)
        signed = side.sign * quantity
        old_qty = state.quantity
        old_avg = state.average_price_ticks
        if old_qty == 0 or (old_qty > 0 and signed > 0) or (old_qty < 0 and signed < 0):
            new_qty = old_qty + signed
            total = abs(old_qty) * old_avg + quantity * Decimal(price_ticks)
            state.average_price_ticks = total / Decimal(abs(new_qty))
        else:
            closing = min(abs(old_qty), quantity)
            if old_qty > 0:
                state.realized_pnl += Decimal(closing) * (Decimal(price_ticks) - old_avg) * self.spec.tick_size * self.spec.lot_size
            else:
                state.realized_pnl += Decimal(closing) * (old_avg - Decimal(price_ticks)) * self.spec.tick_size * self.spec.lot_size
            new_qty = old_qty + signed
            if new_qty == 0:
                state.average_price_ticks = Decimal("0")
            elif (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
                state.average_price_ticks = Decimal(price_ticks)
        notional = self.spec.money(price_ticks, quantity)
        state.cash += -notional if side is Side.BUY else notional
        state.cash -= fee
        state.fees_paid += fee
        state.quantity = old_qty + signed
        state.fills += 1

    def apply(self, execution: Execution) -> None:
        if execution.execution_id in self.execution_ids:
            raise ValueError(f"duplicate execution: {execution.execution_id}")
        self.execution_ids.add(execution.execution_id)
        if execution.maker_owner not in {"market", "external-flow"}:
            self._apply_fill(execution.maker_owner, execution.maker_side, execution.quantity,
                             execution.price_ticks, execution.maker_fee)
        if execution.taker_owner not in {"market", "external-flow"}:
            self._apply_fill(execution.taker_owner, execution.taker_side, execution.quantity,
                             execution.price_ticks, execution.taker_fee)

    def mark(self, owner: str, mark_price_ticks: Decimal | int | None) -> LedgerSnapshot:
        state = self.state(owner)
        mark = Decimal(mark_price_ticks or 0)
        mtm = Decimal(state.quantity) * mark * self.spec.tick_size * self.spec.lot_size
        equity = state.cash + mtm
        return LedgerSnapshot(owner, state.quantity, state.average_price_ticks, state.realized_pnl,
                              state.fees_paid, state.cash, mtm, equity)

    def owners(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def summary(self, mark_price_ticks: Decimal | int | None = None) -> dict[str, dict[str, str | int]]:
        rows = {}
        for owner in self.owners():
            snapshot = self.mark(owner, mark_price_ticks)
            rows[owner] = {
                "quantity": snapshot.quantity,
                "average_price_ticks": str(snapshot.average_price_ticks),
                "realized_pnl": str(snapshot.realized_pnl),
                "fees_paid": str(snapshot.fees_paid),
                "cash": str(snapshot.cash),
                "mark_to_market": str(snapshot.mark_to_market),
                "equity": str(snapshot.equity),
            }
        return rows
