"""Deterministic paper broker with fees, slippage, and optional missed fills.

This adapter never sends orders to a market or any third party.
"""
from __future__ import annotations
import random
from datetime import datetime, timezone
from uuid import uuid4
from ..config import Settings
from ..types import Fill, Order, OrderRequest, OrderStatus, OrderType, Quote
from .base import ExecutionResult

class PaperBroker:
    def __init__(self, settings: Settings, seed: int = 19, use_quote_timestamp: bool = False):
        self.settings = settings
        self._random = random.Random(seed)
        self.use_quote_timestamp = use_quote_timestamp

    def execute(self, request: OrderRequest, quote: Quote) -> ExecutionResult:
        if request.symbol != quote.symbol:
            order = Order(uuid4().hex, request, OrderStatus.REJECTED, reason="quote symbol mismatch")
            return ExecutionResult(order)
        order = Order(uuid4().hex, request, OrderStatus.ACCEPTED)
        if self._random.random() > self.settings.paper_fill_probability:
            order.status = OrderStatus.EXPIRED
            order.reason = "paper simulation: no fill generated"
            order.updated_at = datetime.now(timezone.utc)
            return ExecutionResult(order)
        if request.side.value == "buy":
            base_price = quote.ask
        else:
            base_price = quote.bid
        if request.order_type is OrderType.LIMIT:
            assert request.limit_price is not None
            if request.side.value == "buy" and quote.ask > request.limit_price:
                order.status = OrderStatus.ACCEPTED
                order.reason = "buy limit is not marketable"
                return ExecutionResult(order)
            if request.side.value == "sell" and quote.bid < request.limit_price:
                order.status = OrderStatus.ACCEPTED
                order.reason = "sell limit is not marketable"
                return ExecutionResult(order)
            base_price = min(base_price, request.limit_price) if request.side.value == "buy" else max(base_price, request.limit_price)
        slippage = self.settings.slippage_bps / 10_000.0
        fill_price = base_price * (1 + slippage if request.side.value == "buy" else 1 - slippage)
        notional = request.quantity * fill_price
        commission = notional * self.settings.commission_bps / 10_000.0
        fill_time = quote.timestamp if self.use_quote_timestamp else datetime.now(timezone.utc)
        fill = Fill(uuid4().hex, order.order_id, request.symbol, request.side,
                    request.quantity, fill_price, commission, fill_time)
        order.status = OrderStatus.FILLED
        order.filled_quantity = request.quantity
        order.average_fill_price = fill_price
        order.updated_at = fill.timestamp
        order.reason = "paper fill; not a real execution"
        return ExecutionResult(order, (fill,))
