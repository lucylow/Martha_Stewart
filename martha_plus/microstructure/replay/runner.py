"""Event-time market replay with delayed strategy observations and order entry."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Sequence

from ..book.l2 import DepthLevel, DepthSnapshot
from ..clock import LatencyKind, LatencyModel
from ..matching.engine import MatchingEngine, MatchResult
from ..risk.controls import RiskContext, RiskController
from ..simulation.ledger import PortfolioLedger
from ..strategies.base import Strategy, StrategyContext
from ..types import EventKind, MarketEvent, OrderAck, OrderRequest, Side
from .scheduler import EventScheduler, ScheduledTask


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    source_event_sequence: int
    source_event_ts_ns: int
    decision_ts_ns: int
    order_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RiskRejection:
    client_order_id: str
    ts_ns: int
    codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    symbol: str
    start_ns: int
    end_ns: int
    market_events: int
    scheduled_tasks: int
    strategy_decisions: tuple[StrategyDecision, ...]
    venue_acks: tuple[OrderAck, ...]
    delivered_acks: tuple[OrderAck, ...]
    risk_rejections: tuple[RiskRejection, ...]
    executions: tuple
    ledger: dict
    latency: dict
    final_book: object
    anomalies: tuple[str, ...]

    @property
    def accepted_orders(self) -> int:
        return sum(1 for ack in self.venue_acks if ack.accepted)

    @property
    def rejected_orders(self) -> int:
        return sum(1 for ack in self.venue_acks if not ack.accepted) + len(self.risk_rejections)

    @property
    def filled_quantity(self) -> int:
        return sum(x.quantity for x in self.executions)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "market_events": self.market_events,
            "scheduled_tasks": self.scheduled_tasks,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "filled_quantity": self.filled_quantity,
            "strategy_decisions": [decision.__dict__ if hasattr(decision, "__dict__") else {
                "source_event_sequence": decision.source_event_sequence,
                "source_event_ts_ns": decision.source_event_ts_ns,
                "decision_ts_ns": decision.decision_ts_ns,
                "order_ids": list(decision.order_ids),
            } for decision in self.strategy_decisions],
            "venue_acks": [_ack_dict(x) for x in self.venue_acks],
            "delivered_acks": [_ack_dict(x) for x in self.delivered_acks],
            "risk_rejections": [{"client_order_id": x.client_order_id, "ts_ns": x.ts_ns, "codes": list(x.codes)} for x in self.risk_rejections],
            "executions": [_execution_dict(x) for x in self.executions],
            "ledger": self.ledger,
            "latency": self.latency,
            "final_book": {
                "bids": [{"price_ticks": x.price_ticks, "quantity": x.quantity, "order_count": x.order_count} for x in self.final_book.bids],
                "asks": [{"price_ticks": x.price_ticks, "quantity": x.quantity, "order_count": x.order_count} for x in self.final_book.asks],
            },
            "anomalies": list(self.anomalies),
        }


def _ack_dict(ack: OrderAck) -> dict:
    return {"client_order_id": ack.client_order_id, "accepted": ack.accepted, "ts_ns": ack.ts_ns,
            "status": ack.status.value, "reason": ack.reason,
            "reject_code": ack.reject_code.value if ack.reject_code else None}


def _execution_dict(execution) -> dict:
    return {
        "execution_id": execution.execution_id,
        "ts_ns": execution.ts_ns,
        "symbol": execution.symbol,
        "price_ticks": execution.price_ticks,
        "quantity": execution.quantity,
        "maker_order_id": execution.maker_order_id,
        "taker_order_id": execution.taker_order_id,
        "maker_owner": execution.maker_owner,
        "taker_owner": execution.taker_owner,
        "maker_side": execution.maker_side.value,
        "taker_side": execution.taker_side.value,
        "maker_fee": str(execution.maker_fee),
        "taker_fee": str(execution.taker_fee),
    }


class MarketReplay:
    """Simulates exchange arrival, delayed feed delivery, and delayed order entry.

    Input market events are applied to the exchange at their source timestamps.
    A strategy receives a point-in-time book snapshot after a sampled feed delay;
    orders then arrive after a separately sampled order-entry delay. Ties are
    resolved by stable scheduler insertion sequence, never wall-clock time.
    """

    def __init__(
        self,
        exchange: MatchingEngine,
        strategy: Strategy,
        *,
        latency: LatencyModel | None = None,
        risk: RiskController | None = None,
        starting_cash: Decimal = Decimal("0"),
        max_scheduled_tasks: int = 5_000_000,
    ) -> None:
        if max_scheduled_tasks <= 0:
            raise ValueError("max_scheduled_tasks must be positive")
        self.exchange = exchange
        self.strategy = strategy
        self.latency = latency or LatencyModel()
        self.risk = risk
        self.ledger = PortfolioLedger(exchange.spec, starting_cash)
        self.max_scheduled_tasks = max_scheduled_tasks
        self.scheduler = EventScheduler()
        self.decisions: list[StrategyDecision] = []
        self.venue_acks: list[OrderAck] = []
        self.delivered_acks: list[OrderAck] = []
        self.risk_rejections: list[RiskRejection] = []
        self.anomalies: list[str] = []
        self.market_event_count = 0
        self._last_mark: Decimal | None = None
        # A generated feed is unaware of simulated client fills. Remember
        # external makers consumed by the strategy so later feed cancels or
        # reductions for those orders can be reconciled as stale updates.
        self._client_consumed_external: set[str] = set()

    def _remember_external_fills(self, executions: Iterable) -> None:
        for execution in executions:
            if execution.maker_owner == "market":
                self._client_consumed_external.add(execution.maker_order_id)

    def _capture_depth(self, ts_ns: int) -> tuple[object, DepthSnapshot]:
        book_depth = self.exchange.book.depth(20, ts_ns)
        bids = tuple(DepthLevel(row.price_ticks, row.quantity) for row in book_depth.bids)
        asks = tuple(DepthLevel(row.price_ticks, row.quantity) for row in book_depth.asks)
        return self.exchange.book.top(ts_ns), DepthSnapshot(self.exchange.spec.symbol, ts_ns, bids, asks)

    def _apply_market_event(self, event: MarketEvent) -> tuple:
        book = self.exchange.book
        if event.kind is EventKind.ADD:
            book.add(event.order_id, "market", event.side, event.price_ticks, event.quantity, event.ts_ns, is_external=True)
            executions = ()
        elif event.kind is EventKind.CANCEL:
            existing = book.get(event.order_id)
            if existing is None:
                if event.order_id not in self._client_consumed_external:
                    self.anomalies.append(f"{event.ts_ns}:cancel_unknown:{event.order_id}")
            elif event.order_id in self.exchange.orders:
                result = self.exchange.cancel(event.order_id, event.ts_ns)
                if not result.accepted:
                    self.anomalies.append(f"{event.ts_ns}:cancel_rejected:{event.order_id}")
            else:
                book.cancel(event.order_id, event.ts_ns)
            executions = ()
        elif event.kind is EventKind.REDUCE:
            existing = book.get(event.order_id)
            if existing is None:
                if event.order_id not in self._client_consumed_external:
                    self.anomalies.append(f"{event.ts_ns}:reduce_unknown:{event.order_id}")
                executions = ()
            elif event.order_id in self.exchange.orders:
                self.anomalies.append(f"{event.ts_ns}:external_reduce_client_order:{event.order_id}")
                executions = ()
            elif event.quantity > existing.quantity:
                if event.order_id in self._client_consumed_external:
                    book.reduce(event.order_id, existing.quantity, event.ts_ns)
                    executions = ()
                else:
                    self.anomalies.append(f"{event.ts_ns}:reduce_exceeds_order:{event.order_id}")
                    executions = ()
            else:
                book.reduce(event.order_id, event.quantity, event.ts_ns)
                executions = ()
        elif event.kind is EventKind.TRADE:
            executions = self.exchange.external_aggressor(event.side, event.quantity, event.ts_ns,
                                                          price_limit_ticks=event.price_ticks)
        elif event.kind is EventKind.HALT:
            self.exchange.halt("market-data halt event")
            executions = ()
        elif event.kind is EventKind.RESUME:
            self.exchange.resume()
            executions = ()
        elif event.kind is EventKind.REPLACE:
            existing = book.get(event.order_id)
            if existing is None or existing.order_id in self.exchange.orders:
                self.anomalies.append(f"{event.ts_ns}:replace_unknown_or_client:{event.order_id}")
                executions = ()
            else:
                book.cancel(event.order_id, event.ts_ns)
                book.add(event.order_id, existing.owner, existing.side, event.price_ticks,
                         event.quantity, event.ts_ns, is_external=True)
                executions = ()
        else:
            self.anomalies.append(f"{event.ts_ns}:unsupported_event:{event.kind.value}")
            executions = ()
        for execution in executions:
            self.ledger.apply(execution)
        self._remember_external_fills(executions)
        return executions

    def _context(self, event: MarketEvent, decision_ts_ns: int, top, depth) -> StrategyContext:
        state = self.ledger.state(self.strategy.strategy_id)
        position = state.quantity
        mark = top.mid_ticks
        cash = state.cash
        mtm = Decimal(position) * (mark or Decimal("0")) * self.exchange.spec.tick_size * self.exchange.spec.lot_size
        return StrategyContext(
            symbol=self.exchange.spec.symbol,
            decision_ts_ns=decision_ts_ns,
            source_event_ts_ns=event.ts_ns,
            top=top,
            depth=depth,
            position_quantity=position,
            cash=cash,
            equity=cash + mtm,
            open_order_count=len(self.exchange.open_orders(owner=self.strategy.strategy_id)),
        )

    def _handle(self, task: ScheduledTask) -> None:
        if task.kind == "market":
            event: MarketEvent = task.payload
            executions = self._apply_market_event(event)
            self.market_event_count += 1
            top, depth = self._capture_depth(event.ts_ns)
            if top.mid_ticks is not None:
                self._last_mark = top.mid_ticks
            # Do not invent strategy messages while halted; strategy receives
            # the resume event so it can re-establish a fresh quote afterwards.
            if not self.exchange.halted or event.kind is EventKind.RESUME:
                feed_delay = self.latency.sample(LatencyKind.MARKET_DATA)
                self.scheduler.schedule(task.due_ns + feed_delay, "strategy",
                                        (event, top, depth), source_ns=event.ts_ns)
        elif task.kind == "strategy":
            event, top, depth = task.payload
            context = self._context(event, task.due_ns, top, depth)
            intents = tuple(self.strategy.on_market_data(event, context) or ())
            order_ids = tuple(request.client_order_id for request in intents)
            self.decisions.append(StrategyDecision(event.sequence, event.ts_ns, task.due_ns, order_ids))
            for request in intents:
                if request.strategy_id != self.strategy.strategy_id:
                    self.risk_rejections.append(RiskRejection(request.client_order_id, task.due_ns, ("strategy_id_mismatch",)))
                    continue
                if self.risk is not None:
                    decision = self.risk.evaluate(request, RiskContext(
                        ts_ns=task.due_ns,
                        position_quantity=self.ledger.state(self.strategy.strategy_id).quantity,
                        open_order_count=len(self.exchange.open_orders(owner=self.strategy.strategy_id)),
                        realized_pnl=self.ledger.state(self.strategy.strategy_id).net_realized_pnl,
                        reference_price_ticks=request.limit_price_ticks or request.max_price_ticks or request.min_price_ticks,
                        spread_ticks=top.spread_ticks,
                    ))
                    if not decision.allowed:
                        self.risk_rejections.append(RiskRejection(request.client_order_id, task.due_ns, decision.codes))
                        continue
                send_delay = self.latency.sample(LatencyKind.ORDER_ENTRY)
                self.scheduler.schedule(task.due_ns + send_delay, "submit", (request, task.due_ns), source_ns=task.due_ns)
        elif task.kind == "submit":
            request, decision_ts_ns = task.payload
            result: MatchResult = self.exchange.submit(request, task.due_ns, submitted_at_ns=decision_ts_ns)
            self.venue_acks.append(result.ack)
            for execution in result.executions:
                self.ledger.apply(execution)
            self._remember_external_fills(result.executions)
            ack_delay = self.latency.sample(LatencyKind.ORDER_ACK)
            self.scheduler.schedule(task.due_ns + ack_delay, "ack_delivery", result.ack, source_ns=task.due_ns)
        elif task.kind == "ack_delivery":
            self.delivered_acks.append(task.payload)
        else:
            raise ValueError(f"unknown scheduled task kind: {task.kind}")

    def run(self, events: Iterable[MarketEvent]) -> RunResult:
        rows = list(events)
        if not rows:
            raise ValueError("at least one market event is required")
        rows.sort(key=lambda event: (event.ts_ns, event.sequence))
        seen: set[tuple[int, int]] = set()
        previous = (-1, -1)
        for event in rows:
            key = (event.ts_ns, event.sequence)
            if key in seen:
                raise ValueError(f"duplicate event timestamp/sequence: {key}")
            seen.add(key)
            if event.symbol != self.exchange.spec.symbol:
                raise ValueError(f"event symbol mismatch: {event.symbol}")
            self.scheduler.schedule(event.ts_ns, "market", event, source_ns=event.ts_ns)
            previous = key
        start_ns = rows[0].ts_ns
        max_market_ns = rows[-1].ts_ns
        self.scheduler.run(self._handle, max_tasks=self.max_scheduled_tasks)
        self.exchange.validate()
        self.exchange.book.validate()
        final_depth = self.exchange.book.depth(20, self.scheduler.now_ns)
        mark = self.exchange.book.top(self.scheduler.now_ns).mid_ticks or self._last_mark
        ledger = self.ledger.summary(mark)
        latency_summary = {kind.value: self.latency.summary(kind) for kind in LatencyKind}
        return RunResult(
            symbol=self.exchange.spec.symbol,
            start_ns=start_ns,
            end_ns=self.scheduler.now_ns,
            market_events=self.market_event_count,
            scheduled_tasks=self.scheduler.processed,
            strategy_decisions=tuple(self.decisions),
            venue_acks=tuple(self.venue_acks),
            delivered_acks=tuple(self.delivered_acks),
            risk_rejections=tuple(self.risk_rejections),
            executions=tuple(self.exchange.executions),
            ledger=ledger,
            latency=latency_summary,
            final_book=final_depth,
            anomalies=tuple(self.anomalies),
        )
