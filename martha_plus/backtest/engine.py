"""Bar-based backtest engine with signal lag, fees, slippage, and risk checks."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence
from ..config import Settings
from ..execution.paper import PaperBroker
from ..indicators import returns
from ..portfolio.ledger import PortfolioLedger
from ..risk.limits import RiskContext, RiskLimits
from ..risk.sizing import target_weight_quantity, cap_quantity_by_limits
from ..types import Bar, OrderRequest, OrderType, Quote, Side
from ..strategies.base import Strategy
from .metrics import calculate_metrics, trade_statistics

@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float

@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_id: str
    symbol: str
    equity_curve: tuple[EquityPoint, ...]
    orders: tuple[dict, ...]
    metrics: Mapping[str, float | int]
    trades: Mapping[str, float | int]
    rejected_orders: int

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id, "symbol": self.symbol,
            "equity_curve": [p.__dict__ if hasattr(p, "__dict__") else {
                "timestamp": p.timestamp.isoformat(), "cash": p.cash, "equity": p.equity,
                "gross_exposure": p.gross_exposure, "net_exposure": p.net_exposure} for p in self.equity_curve],
            "orders": list(self.orders), "metrics": dict(self.metrics),
            "trades": dict(self.trades), "rejected_orders": self.rejected_orders,
        }

class BacktestEngine:
    def __init__(self, settings: Settings | None = None, spread_bps: float = 2.0,
                 periods_per_year: float = 252.0):
        self.settings = settings or Settings(mode="research")
        self.settings.validate()
        if spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")
        self.spread_bps = spread_bps
        self.periods_per_year = periods_per_year

    def run(self, bars: Sequence[Bar], strategy: Strategy, symbol: str | None = None,
            minimum_history: int = 35) -> BacktestResult:
        if not bars:
            raise ValueError("backtest requires bars")
        key = (symbol or bars[0].symbol).strip().upper()
        data = sorted((b for b in bars if b.symbol == key), key=lambda b: b.timestamp)
        if len(data) < minimum_history + 1:
            raise ValueError(f"backtest requires at least {minimum_history + 1} bars")
        if any(a.timestamp == b.timestamp for a, b in zip(data, data[1:])):
            raise ValueError("duplicate timestamps are not supported")
        ledger = PortfolioLedger(self.settings.starting_cash)
        broker = PaperBroker(self.settings, seed=1234, use_quote_timestamp=True)
        risk = RiskLimits(self.settings)
        points: list[EquityPoint] = []
        order_log: list[dict] = []
        rejected = 0
        high_water = self.settings.starting_cash
        day_start_equity: dict[object, float] = {}
        completed_pnl: list[float] = []
        before_fill_count = 0

        # The signal sees bars strictly before the execution bar. This avoids using
        # the execution bar's close to predict an order filled at that bar's open.
        for index in range(minimum_history, len(data)):
            execution_bar = data[index]
            history = data[:index]
            prior_equity = ledger.equity()
            day = execution_bar.timestamp.astimezone(timezone.utc).date()
            day_start_equity.setdefault(day, prior_equity)
            signal = strategy.generate(key, history)
            ledger.mark({key: execution_bar.open})
            snapshot = ledger.snapshot(execution_bar.timestamp, high_water, day_start_equity[day])
            current_pos = snapshot.positions.get(key)
            current_value = current_pos.quantity * execution_bar.open if current_pos else 0.0
            desired_qty_abs = target_weight_quantity(snapshot.equity, current_value,
                                                       signal.target_weight, execution_bar.open,
                                                       fractional=self.settings.allow_fractional_shares)
            desired_value = snapshot.equity * signal.target_weight
            delta_value = desired_value - current_value
            quantity = cap_quantity_by_limits(
                desired_qty_abs, execution_bar.open, self.settings.max_order_notional,
                self.settings.max_position_notional, current_value,
                self.settings.allow_fractional_shares)
            if quantity > 0 and abs(delta_value) >= execution_bar.open:
                side = Side.BUY if delta_value > 0 else Side.SELL
                half_spread = self.spread_bps / 20_000.0
                quote = Quote(key, execution_bar.open * (1-half_spread),
                              execution_bar.open * (1+half_spread), execution_bar.timestamp,
                              venue="BACKTEST_PROXY")
                request = OrderRequest(key, side, quantity, OrderType.MARKET,
                                       created_at=execution_bar.timestamp,
                                       strategy_id=getattr(strategy, "strategy_id", "strategy"))
                context = RiskContext(snapshot, quote, daily_realized_pnl=ledger.realized_pnl(day),
                                      current_drawdown_fraction=max(0.0, (high_water - snapshot.equity) / high_water),
                                      now=execution_bar.timestamp)
                decision = risk.evaluate(request, context)
                if decision.approved:
                    result = broker.execute(request, quote)
                    order_log.append({"timestamp": execution_bar.timestamp.isoformat(),
                                      "side": side.value, "quantity": quantity,
                                      "status": result.order.status.value,
                                      "signal_weight": signal.target_weight,
                                      "reason": signal.reason,
                                      "fill_price": result.order.average_fill_price})
                    for fill in result.fills:
                        ledger.apply_fill(fill)
                else:
                    rejected += 1
                    order_log.append({"timestamp": execution_bar.timestamp.isoformat(),
                                      "side": side.value, "quantity": quantity,
                                      "status": "rejected", "reason": decision.reason,
                                      "signal_weight": signal.target_weight})
            # At the close mark the portfolio. This is the equity point that belongs
            # to the current period, including transaction costs from the open.
            ledger.mark({key: execution_bar.close})
            snapshot = ledger.snapshot(execution_bar.timestamp, high_water, day_start_equity[day])
            high_water = max(high_water, snapshot.equity)
            points.append(EquityPoint(execution_bar.timestamp, snapshot.cash, snapshot.equity,
                                      snapshot.gross_exposure, snapshot.net_exposure))
            fills = ledger.fills
            if len(fills) > before_fill_count:
                # Approximate realized P&L records by changes in day-level realized pnl;
                # transaction-level FIFO attribution belongs in a more advanced ledger.
                day_pnl = ledger.realized_pnl(day)
                if day_pnl != 0:
                    completed_pnl.append(day_pnl)
                before_fill_count = len(fills)
        curve = [point.equity for point in points]
        metrics = calculate_metrics(curve, self.periods_per_year)
        trades = trade_statistics(completed_pnl)
        return BacktestResult(getattr(strategy, "strategy_id", "strategy"), key,
                              tuple(points), tuple(order_log), metrics, trades, rejected)
