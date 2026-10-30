"""Deterministic price-time-priority matching engine for simulation only."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from ..book.l3 import LimitOrderBook
from ..book.level import QueueOrder
from ..types import (
    Execution,
    LiquidityRole,
    OrderAck,
    OrderRecord,
    OrderRequest,
    OrderStatus,
    OrderType,
    RejectCode,
    Side,
    SymbolSpec,
    TimeInForce,
)
from .fees import FeeSchedule


@dataclass(frozen=True, slots=True)
class MatchResult:
    ack: OrderAck
    executions: tuple[Execution, ...]
    record: OrderRecord | None

    @property
    def filled_quantity(self) -> int:
        return sum(execution.quantity for execution in self.executions)


@dataclass(frozen=True, slots=True)
class CancelResult:
    accepted: bool
    order_id: str
    cancelled_quantity: int
    ts_ns: int
    reason: str = ""


class MatchingEngine:
    """Single-symbol matching engine with explicit external liquidity.

    The engine does not connect to a broker, venue, or network. Market orders
    require a protection price (max for buys, min for sells) to prevent an
    unbounded sweep in pathological or empty books.
    """

    def __init__(
        self,
        spec: SymbolSpec,
        *,
        fees: FeeSchedule | None = None,
        max_book_levels: int = 10_000,
    ) -> None:
        if max_book_levels <= 0:
            raise ValueError("max_book_levels must be positive")
        self.spec = spec
        self.book = LimitOrderBook(spec.symbol)
        self.fees = fees or FeeSchedule()
        self.max_book_levels = max_book_levels
        self.orders: dict[str, OrderRecord] = {}
        self.executions: list[Execution] = []
        self._execution_sequence = 0
        self._external_sequence = 0
        self.halted = False
        self.halt_reason = ""

    def _reject(self, request: OrderRequest, ts_ns: int, code: RejectCode, reason: str) -> MatchResult:
        ack = OrderAck(request.client_order_id, False, ts_ns, OrderStatus.REJECTED, reason, code)
        return MatchResult(ack, (), None)

    def _validate_request(self, request: OrderRequest, ts_ns: int) -> RejectCode | None:
        if request.symbol != self.spec.symbol:
            return RejectCode.INVALID_SYMBOL
        if request.quantity <= 0:
            return RejectCode.INVALID_QUANTITY
        if request.order_type is OrderType.LIMIT:
            price = request.limit_price_ticks
            if price is None or price <= 0:
                return RejectCode.INVALID_PRICE
            if price * self.spec.tick_size % self.spec.tick_size != 0:
                return RejectCode.INVALID_PRICE
        else:
            if request.side is Side.BUY and (request.max_price_ticks is None or request.max_price_ticks <= 0):
                return RejectCode.MARKET_ORDER_WITHOUT_PROTECTION
            if request.side is Side.SELL and (request.min_price_ticks is None or request.min_price_ticks <= 0):
                return RejectCode.MARKET_ORDER_WITHOUT_PROTECTION
        if request.time_in_force is TimeInForce.POST_ONLY and request.order_type is not OrderType.LIMIT:
            return RejectCode.INVALID_PRICE
        if self.halted:
            return RejectCode.HALTED
        return None

    def _protection_price(self, request: OrderRequest) -> int | None:
        if request.order_type is OrderType.LIMIT:
            return request.limit_price_ticks
        return request.max_price_ticks if request.side is Side.BUY else request.min_price_ticks

    def _crosses(self, taker_side: Side, maker_price: int, protection: int | None) -> bool:
        if protection is None:
            return False
        return maker_price <= protection if taker_side is Side.BUY else maker_price >= protection

    def _available_quantity(self, side: Side, protection: int | None) -> int:
        maker_side = side.opposite
        total = 0
        for order in self.book.iter_orders(maker_side):
            if not self._crosses(side, order.price_ticks, protection):
                break
            total += order.quantity
        return total

    def _new_execution(
        self,
        *,
        ts_ns: int,
        price_ticks: int,
        quantity: int,
        maker: QueueOrder,
        taker_order_id: str,
        taker_owner: str,
        taker_side: Side,
    ) -> Execution:
        self._execution_sequence += 1
        maker_fee = self.fees.fee(LiquidityRole.MAKER, self.spec, price_ticks, quantity)
        taker_fee = self.fees.fee(LiquidityRole.TAKER, self.spec, price_ticks, quantity)
        execution = Execution(
            execution_id=f"{self.spec.symbol}-EXEC-{self._execution_sequence:012d}",
            ts_ns=ts_ns,
            symbol=self.spec.symbol,
            price_ticks=price_ticks,
            quantity=quantity,
            maker_order_id=maker.order_id,
            taker_order_id=taker_order_id,
            maker_owner=maker.owner,
            taker_owner=taker_owner,
            maker_side=maker.side,
            taker_side=taker_side,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
        )
        self.executions.append(execution)
        maker_record = self.orders.get(maker.order_id)
        if maker_record is not None:
            maker_record.apply_fill(quantity, price_ticks, maker_fee, ts_ns)
        taker_record = self.orders.get(taker_order_id)
        if taker_record is not None:
            taker_record.apply_fill(quantity, price_ticks, taker_fee, ts_ns)
        return execution

    def _match_incoming(
        self,
        *,
        side: Side,
        order_id: str,
        owner: str,
        quantity: int,
        protection: int,
        ts_ns: int,
    ) -> tuple[Execution, ...]:
        remaining = quantity
        executions: list[Execution] = []
        while remaining > 0:
            maker = self.book.best_order(side.opposite)
            if maker is None or not self._crosses(side, maker.price_ticks, protection):
                break
            fill_quantity = min(remaining, maker.quantity)
            # Consume maker queue before applying fill updates. The returned
            # order object retains its pre-fill quantity for the execution.
            if fill_quantity == maker.quantity:
                self.book.cancel(maker.order_id, ts_ns)
            else:
                self.book.reduce(maker.order_id, fill_quantity, ts_ns)
            execution = self._new_execution(
                ts_ns=ts_ns,
                price_ticks=maker.price_ticks,
                quantity=fill_quantity,
                maker=maker,
                taker_order_id=order_id,
                taker_owner=owner,
                taker_side=side,
            )
            executions.append(execution)
            remaining -= fill_quantity
        return tuple(executions)

    def submit(self, request: OrderRequest, ts_ns: int, *, submitted_at_ns: int | None = None) -> MatchResult:
        """Accept and match an order at exchange-arrival time."""
        submitted_at_ns = ts_ns if submitted_at_ns is None else submitted_at_ns
        if ts_ns < 0 or submitted_at_ns < 0 or submitted_at_ns > ts_ns:
            raise ValueError("invalid submit timestamps")
        if request.client_order_id in self.orders or self.book.get(request.client_order_id):
            return self._reject(request, ts_ns, RejectCode.DUPLICATE_ORDER_ID, "duplicate order id")
        code = self._validate_request(request, ts_ns)
        if code is not None:
            return self._reject(request, ts_ns, code, code.value)
        protection = self._protection_price(request)
        assert protection is not None
        available = self._available_quantity(request.side, protection)
        if request.time_in_force is TimeInForce.POST_ONLY and available > 0:
            return self._reject(request, ts_ns, RejectCode.POST_ONLY_WOULD_TAKE, "post-only order would execute immediately")
        if request.time_in_force is TimeInForce.FOK and available < request.quantity:
            return self._reject(request, ts_ns, RejectCode.FOK_NOT_FILLABLE, "insufficient displayed contra quantity")

        record = OrderRecord(request, submitted_at_ns, ts_ns)
        self.orders[request.client_order_id] = record
        executions = self._match_incoming(
            side=request.side,
            order_id=request.client_order_id,
            owner=request.strategy_id,
            quantity=request.quantity,
            protection=protection,
            ts_ns=ts_ns,
        )
        if record.leaves_quantity > 0:
            if request.order_type is OrderType.LIMIT and request.time_in_force in {TimeInForce.GTC, TimeInForce.POST_ONLY}:
                self.book.add(
                    request.client_order_id,
                    request.strategy_id,
                    request.side,
                    request.limit_price_ticks or 0,
                    record.leaves_quantity,
                    ts_ns,
                    is_external=False,
                )
                record.status = OrderStatus.PARTIALLY_FILLED if record.filled_quantity else OrderStatus.ACCEPTED
            else:
                record.cancel_leaves(ts_ns)
        ack = OrderAck(request.client_order_id, True, ts_ns, record.status)
        self.book.validate()
        return MatchResult(ack, executions, record)

    def add_external_liquidity(
        self,
        order_id: str,
        side: Side,
        price_ticks: int,
        quantity: int,
        ts_ns: int,
        *,
        owner: str = "market",
    ) -> QueueOrder:
        if self.halted:
            raise RuntimeError("cannot add liquidity while venue is halted")
        if order_id in self.orders or self.book.get(order_id):
            raise ValueError("duplicate order id")
        self._external_sequence += 1
        return self.book.add(order_id, owner, side, price_ticks, quantity, ts_ns, is_external=True)

    def cancel(self, order_id: str, ts_ns: int) -> CancelResult:
        record = self.orders.get(order_id)
        queue_order = self.book.get(order_id)
        if record is None or queue_order is None or not record.is_open:
            return CancelResult(False, order_id, 0, ts_ns, "order is not resting")
        removed = self.book.cancel(order_id, ts_ns)
        cancelled = record.cancel_leaves(ts_ns)
        if removed.quantity != cancelled:
            raise AssertionError("cancelled quantity disagrees with book leaves")
        return CancelResult(True, order_id, cancelled, ts_ns)

    def external_aggressor(self, side: Side, quantity: int, ts_ns: int, *, price_limit_ticks: int | None = None) -> tuple[Execution, ...]:
        """Apply an external marketable flow event to currently resting orders."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.halted:
            return ()
        if price_limit_ticks is None:
            best = self.book.best_price(side.opposite)
            if best is None:
                return ()
            price_limit_ticks = best
        self._external_sequence += 1
        order_id = f"EXT-{self._external_sequence:012d}"
        return self._match_incoming(
            side=side,
            order_id=order_id,
            owner="external-flow",
            quantity=quantity,
            protection=price_limit_ticks,
            ts_ns=ts_ns,
        )

    def halt(self, reason: str = "manual halt") -> None:
        self.halted = True
        self.halt_reason = reason

    def resume(self) -> None:
        self.halted = False
        self.halt_reason = ""

    def open_orders(self, *, owner: str | None = None) -> tuple[OrderRecord, ...]:
        rows = [record for record in self.orders.values() if record.is_open]
        if owner is not None:
            rows = [record for record in rows if record.request.strategy_id == owner]
        return tuple(sorted(rows, key=lambda x: (x.accepted_at_ns, x.request.client_order_id)))

    def order_status(self, order_id: str) -> OrderStatus | None:
        record = self.orders.get(order_id)
        return record.status if record is not None else None

    def snapshot(self, depth: int = 10):
        return self.book.depth(depth)

    def validate(self) -> None:
        self.book.validate()
        for order_id, record in self.orders.items():
            if record.filled_quantity + record.leaves_quantity + record.cancelled_quantity != record.total_quantity:
                raise AssertionError(f"order quantity conservation failed for {order_id}")
            resting = self.book.get(order_id)
            if record.is_open and record.leaves_quantity > 0:
                if resting is None or resting.quantity != record.leaves_quantity:
                    raise AssertionError(f"book/order leaves mismatch for {order_id}")
            elif resting is not None:
                raise AssertionError(f"terminal order still rests: {order_id}")
