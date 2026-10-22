"""Orchestrates signal generation, risk checks, paper execution, and accounting."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Sequence
from ..clock import Clock, SystemClock
from ..config import Settings
from ..execution.manager import OrderManager
from ..execution.paper import PaperBroker
from ..marketdata.base import MarketDataProvider
from ..persistence.sqlite_store import SQLiteStore
from ..portfolio.ledger import PortfolioLedger
from ..risk.kill_switch import KillSwitch
from ..risk.limits import RiskContext, RiskLimits
from ..risk.sizing import target_weight_quantity, cap_quantity_by_limits
from ..strategies.base import Strategy
from ..types import OrderRequest, OrderType, Quote, Side, Signal

@dataclass(frozen=True, slots=True)
class TickResult:
    symbol: str
    signal: Signal
    decision: str
    reason: str
    order_id: str | None = None
    filled_quantity: float = 0.0
    fill_price: float | None = None

class TradingResearchEngine:
    """Single-process research/paper engine. No background thread is started implicitly."""
    def __init__(self, settings: Settings, market_data: MarketDataProvider,
                 strategy: Strategy, ledger: PortfolioLedger | None = None,
                 store: SQLiteStore | None = None, clock: Clock | None = None,
                 kill_switch: KillSwitch | None = None):
        settings.validate()
        self.settings = settings
        self.market_data = market_data
        self.strategy = strategy
        self.ledger = ledger or PortfolioLedger(settings.starting_cash)
        self.store = store
        self.clock = clock or SystemClock()
        self.kill_switch = kill_switch or KillSwitch(settings.kill_switch_file)
        self.risk_limits = RiskLimits(settings, self.clock)
        broker = PaperBroker(settings)
        self.order_manager = OrderManager(
            broker, self.risk_limits, self.ledger,
            persist_order=store.save_order if store else None,
            persist_fill=store.save_fill if store else None,
        )
        self.high_water_mark = settings.starting_cash
        self.day_start_equity: dict[date, float] = {}

    def status(self) -> dict:
        now = self.clock.now()
        snapshot = self.ledger.snapshot(now, self.high_water_mark)
        return {
            "mode": self.settings.mode,
            "paper_only": True,
            "kill_switch_active": self.kill_switch.active,
            "strategy_id": getattr(self.strategy, "strategy_id", self.strategy.__class__.__name__),
            "cash": round(snapshot.cash, 4),
            "equity": round(snapshot.equity, 4),
            "gross_exposure": round(snapshot.gross_exposure, 6),
            "net_exposure": round(snapshot.net_exposure, 6),
            "open_orders": len(self.order_manager.open_orders),
            "positions": {symbol: {"quantity": p.quantity, "average_cost": p.average_cost,
                                    "last_price": p.last_price, "market_value": p.market_value,
                                    "unrealized_pnl": p.unrealized_pnl}
                          for symbol, p in snapshot.positions.items()},
            "timestamp": now.isoformat(),
        }

    def process(self, symbol: str, history_limit: int | None = None) -> TickResult:
        symbol = symbol.strip().upper()
        bars = tuple(self.market_data.bars(symbol, history_limit))
        if not bars:
            raise ValueError(f"no bars available for {symbol}")
        signal = self.strategy.generate(symbol, bars)
        quote = self.market_data.latest_quote(symbol)
        now = self.clock.now()
        self.ledger.mark({symbol: quote.midpoint})
        snapshot = self.ledger.snapshot(now, self.high_water_mark)
        day = now.astimezone(timezone.utc).date()
        self.day_start_equity.setdefault(day, snapshot.equity)
        current_pos = snapshot.positions.get(symbol)
        current_value = current_pos.market_value if current_pos else 0.0
        price = quote.midpoint
        delta_value = snapshot.equity * signal.target_weight - current_value
        raw_quantity = target_weight_quantity(snapshot.equity, current_value, signal.target_weight,
                                               price, self.settings.allow_fractional_shares)
        quantity = cap_quantity_by_limits(raw_quantity, price, self.settings.max_order_notional,
                                          self.settings.max_position_notional, current_value,
                                          self.settings.allow_fractional_shares)
        if quantity <= 0 or abs(delta_value) < price:
            reason = "target already met or below one share"
            if self.store:
                self.store.append_event("strategy_tick", {
                    "symbol": symbol, "strategy_id": getattr(self.strategy, "strategy_id", "strategy"),
                    "target_weight": signal.target_weight, "strength": signal.strength,
                    "signal_reason": signal.reason, "decision": "no_order", "reason": reason,
                }, timestamp=now)
                self.store.save_snapshot(self.ledger.snapshot(now, self.high_water_mark))
            return TickResult(symbol, signal, "no_order", reason)
        side = Side.BUY if delta_value > 0 else Side.SELL
        request = OrderRequest(symbol=symbol, side=side, quantity=quantity,
                               order_type=OrderType.MARKET, created_at=now,
                               strategy_id=getattr(self.strategy, "strategy_id", "strategy"),
                               metadata={"signal_reason": signal.reason})
        context = RiskContext(
            portfolio=snapshot, quote=quote,
            open_order_count=len(self.order_manager.open_orders),
            daily_realized_pnl=self.ledger.realized_pnl(day),
            current_drawdown_fraction=max(0.0, (self.high_water_mark - snapshot.equity) / self.high_water_mark),
            trading_halted=self.kill_switch.active, now=now,
        )
        result = self.order_manager.submit(request, quote, context)
        self.high_water_mark = max(self.high_water_mark, self.ledger.equity())
        if self.store:
            self.store.append_event("strategy_tick", {
                "symbol": symbol, "strategy_id": getattr(self.strategy, "strategy_id", "strategy"),
                "target_weight": signal.target_weight, "strength": signal.strength,
                "signal_reason": signal.reason, "decision": result.order.status.value,
                "order_id": result.order.order_id, "client_order_id": request.client_order_id,
            }, timestamp=now)
            self.store.save_snapshot(self.ledger.snapshot(now, self.high_water_mark))
        first_fill = result.fills[0] if result.fills else None
        decision = result.order.status.value
        return TickResult(symbol, signal, decision, result.order.reason or decision,
                          result.order.order_id, result.order.filled_quantity,
                          first_fill.price if first_fill else None)

    def run_once(self, symbols: Sequence[str]) -> list[TickResult]:
        """Process an explicit list once; callers control scheduling and shutdown."""
        return [self.process(symbol) for symbol in symbols]
