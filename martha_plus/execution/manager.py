"""Order manager with idempotency, risk gates, and explicit lifecycle tracking."""
from __future__ import annotations
from threading import RLock
from typing import Callable
from ..errors import DuplicateOrder, RiskRejected
from ..risk.limits import RiskContext, RiskLimits
from ..types import Order, OrderRequest, OrderStatus, Quote
from ..portfolio.ledger import PortfolioLedger
from .base import ExecutionAdapter, ExecutionResult

class OrderManager:
    def __init__(self, adapter: ExecutionAdapter, risk_limits: RiskLimits,
                 ledger: PortfolioLedger, persist_order: Callable[[Order], None] | None = None,
                 persist_fill: Callable[[object], None] | None = None):
        self.adapter = adapter
        self.risk_limits = risk_limits
        self.ledger = ledger
        self.persist_order = persist_order or (lambda order: None)
        self.persist_fill = persist_fill or (lambda fill: None)
        self._orders_by_client_id: dict[str, Order] = {}
        self._request_fingerprints: dict[str, tuple] = {}
        self._open: dict[str, Order] = {}
        self._lock = RLock()

    @property
    def open_orders(self) -> tuple[Order, ...]:
        with self._lock:
            return tuple(self._open.values())

    @property
    def all_orders(self) -> tuple[Order, ...]:
        with self._lock:
            return tuple(self._orders_by_client_id.values())

    @staticmethod
    def _fingerprint(request: OrderRequest) -> tuple:
        return (request.symbol, request.side, request.quantity, request.order_type,
                request.limit_price, request.strategy_id)

    def submit(self, request: OrderRequest, quote: Quote, context: RiskContext) -> ExecutionResult:
        key = request.client_order_id
        fingerprint = self._fingerprint(request)
        with self._lock:
            if key in self._orders_by_client_id:
                if self._request_fingerprints[key] != fingerprint:
                    raise DuplicateOrder("client_order_id reused with different order parameters")
                prior = self._orders_by_client_id[key]
                return ExecutionResult(prior, ())
            adjusted_context = RiskContext(
                portfolio=context.portfolio, quote=quote,
                open_order_count=len(self._open), daily_realized_pnl=context.daily_realized_pnl,
                current_drawdown_fraction=context.current_drawdown_fraction,
                trading_halted=context.trading_halted, now=context.now,
            )
            decision = self.risk_limits.evaluate(request, adjusted_context)
            if not decision.approved:
                rejected = Order("risk-" + key, request, OrderStatus.REJECTED,
                                 reason=decision.reason)
                self._orders_by_client_id[key] = rejected
                self._request_fingerprints[key] = fingerprint
                self.persist_order(rejected)
                return ExecutionResult(rejected, ())
            result = self.adapter.execute(request, quote)
            self._orders_by_client_id[key] = result.order
            self._request_fingerprints[key] = fingerprint
            if not result.order.status.terminal:
                self._open[result.order.order_id] = result.order
            # Persist the parent before fills because the SQLite schema intentionally
            # enforces a foreign key from fills to orders.
            self.persist_order(result.order)
            for fill in result.fills:
                self.ledger.apply_fill(fill)
                self.persist_fill(fill)
            if result.order.status.terminal:
                self._open.pop(result.order.order_id, None)
            return result

    def cancel(self, client_order_id: str, reason: str = "cancelled by operator") -> Order:
        with self._lock:
            order = self._orders_by_client_id.get(client_order_id)
            if order is None:
                raise KeyError(client_order_id)
            if not order.status.terminal:
                order.status = OrderStatus.CANCELLED
                order.reason = reason
                self._open.pop(order.order_id, None)
                self.persist_order(order)
            return order
