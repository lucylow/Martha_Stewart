"""Shared-capital multi-asset backtesting with timestamp-aligned execution bars."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Sequence
from ..config import Settings
from ..execution.paper import PaperBroker
from ..portfolio.ledger import PortfolioLedger
from ..risk.limits import RiskContext, RiskLimits
from ..risk.sizing import target_weight_quantity, cap_quantity_by_limits
from ..strategies.base import Strategy
from ..types import Bar, OrderRequest, OrderType, Quote, Side
from .engine import EquityPoint
from .metrics import calculate_metrics, trade_statistics

@dataclass(frozen=True, slots=True)
class MultiAssetBacktestResult:
    symbols: tuple[str, ...]
    equity_curve: tuple[EquityPoint, ...]
    orders: tuple[dict, ...]
    metrics: Mapping[str, float | int]
    trades: Mapping[str, float | int]
    rejected_orders: int

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "equity_curve": [{"timestamp": point.timestamp.isoformat(),
                "cash": point.cash, "equity": point.equity,
                "gross_exposure": point.gross_exposure, "net_exposure": point.net_exposure}
                for point in self.equity_curve],
            "orders": list(self.orders), "metrics": dict(self.metrics),
            "trades": dict(self.trades), "rejected_orders": self.rejected_orders,
        }

class MultiAssetBacktestEngine:
    """Backtest several symbols against one shared cash/position ledger.

    Every signal is generated using only bars strictly earlier than that symbol's
    current execution bar. Execution uses the current bar's open with a synthetic
    spread and fixed slippage. Symbols are processed lexicographically at the same
    timestamp for deterministic allocation order; this is not a venue simulator.
    """
    def __init__(self, settings: Settings | None = None, spread_bps: float = 2.0,
                 periods_per_year: float = 252.0):
        self.settings = settings or Settings(mode="research")
        self.settings.validate()
        if spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")
        self.spread_bps = spread_bps
        self.periods_per_year = periods_per_year

    def run(self, bars_by_symbol: Mapping[str, Sequence[Bar]],
            strategies: Mapping[str, Strategy], minimum_history: int = 35) -> MultiAssetBacktestResult:
        if not bars_by_symbol:
            raise ValueError("at least one symbol is required")
        normalized: dict[str, list[Bar]] = {}
        for raw_symbol, raw_bars in bars_by_symbol.items():
            symbol = raw_symbol.strip().upper()
            values = sorted((bar for bar in raw_bars if bar.symbol == symbol), key=lambda bar: bar.timestamp)
            if not values:
                raise ValueError(f"no bars for symbol {symbol}")
            if len(values) < minimum_history + 1:
                raise ValueError(f"symbol {symbol} requires at least {minimum_history + 1} bars")
            if any(a.timestamp == b.timestamp for a, b in zip(values, values[1:])):
                raise ValueError(f"duplicate timestamps for symbol {symbol}")
            if symbol not in {key.strip().upper() for key in strategies}:
                raise ValueError(f"missing strategy for symbol {symbol}")
            normalized[symbol] = values
        strategy_by_symbol = {key.strip().upper(): value for key, value in strategies.items()}
        all_times = sorted({bar.timestamp for values in normalized.values() for bar in values})
        by_symbol_at_time = {symbol: {bar.timestamp: bar for bar in values}
                             for symbol, values in normalized.items()}
        histories: dict[str, list[Bar]] = {symbol: [] for symbol in normalized}
        ledger = PortfolioLedger(self.settings.starting_cash)
        broker = PaperBroker(self.settings, seed=991, use_quote_timestamp=True)
        risk = RiskLimits(self.settings)
        points: list[EquityPoint] = []
        orders: list[dict] = []
        rejected = 0
        high_water = self.settings.starting_cash
        day_start: dict[date, float] = {}
        realized_markers: dict[date, float] = {}
        realized_pnl_samples: list[float] = []
        last_realized_count = 0

        for stamp in all_times:
            current_bars = {symbol: by_symbol_at_time[symbol][stamp]
                            for symbol in normalized if stamp in by_symbol_at_time[symbol]}
            if not current_bars:
                continue
            ledger.mark({symbol: bar.open for symbol, bar in current_bars.items()})
            day = stamp.astimezone(timezone.utc).date()
            day_start.setdefault(day, ledger.equity())
            # Process only symbols with enough strictly prior observations.
            eligible = [symbol for symbol in sorted(current_bars)
                        if len(histories[symbol]) >= minimum_history]
            for symbol in eligible:
                bar = current_bars[symbol]
                strategy = strategy_by_symbol[symbol]
                signal = strategy.generate(symbol, tuple(histories[symbol]))
                snapshot = ledger.snapshot(stamp, high_water, day_start[day])
                position = snapshot.positions.get(symbol)
                current_value = position.market_value if position else 0.0
                delta_value = snapshot.equity * signal.target_weight - current_value
                raw_quantity = target_weight_quantity(snapshot.equity, current_value,
                    signal.target_weight, bar.open, self.settings.allow_fractional_shares)
                quantity = cap_quantity_by_limits(raw_quantity, bar.open,
                    self.settings.max_order_notional, self.settings.max_position_notional,
                    current_value, self.settings.allow_fractional_shares)
                if quantity > 0 and abs(delta_value) >= bar.open:
                    side = Side.BUY if delta_value > 0 else Side.SELL
                    half_spread = self.spread_bps / 20_000.0
                    quote = Quote(symbol, bar.open * (1-half_spread), bar.open * (1+half_spread),
                                  stamp, venue="MULTI_ASSET_BACKTEST_PROXY")
                    request = OrderRequest(symbol, side, quantity, OrderType.MARKET,
                                           created_at=stamp,
                                           strategy_id=getattr(strategy, "strategy_id", "strategy"))
                    context = RiskContext(snapshot, quote,
                        daily_realized_pnl=ledger.realized_pnl(day),
                        current_drawdown_fraction=max(0.0, (high_water - snapshot.equity) / high_water),
                        now=stamp)
                    decision = risk.evaluate(request, context)
                    if decision.approved:
                        result = broker.execute(request, quote)
                        for fill in result.fills:
                            ledger.apply_fill(fill)
                        orders.append({"timestamp": stamp.isoformat(), "symbol": symbol,
                            "side": side.value, "quantity": quantity, "status": result.order.status.value,
                            "target_weight": signal.target_weight, "signal_reason": signal.reason,
                            "fill_price": result.order.average_fill_price})
                    else:
                        rejected += 1
                        orders.append({"timestamp": stamp.isoformat(), "symbol": symbol,
                            "side": side.value, "quantity": quantity, "status": "rejected",
                            "reason": decision.reason, "target_weight": signal.target_weight})
            # Close marks belong to the current timestamp after all open-price fills.
            ledger.mark({symbol: bar.close for symbol, bar in current_bars.items()})
            snapshot = ledger.snapshot(stamp, high_water, day_start[day])
            high_water = max(high_water, snapshot.equity)
            points.append(EquityPoint(stamp, snapshot.cash, snapshot.equity,
                                      snapshot.gross_exposure, snapshot.net_exposure))
            fill_count = len(ledger.fills)
            if fill_count > last_realized_count:
                realized = ledger.realized_pnl(day)
                if realized != 0:
                    realized_pnl_samples.append(realized)
                last_realized_count = fill_count
            for symbol, bar in current_bars.items():
                histories[symbol].append(bar)
        metrics = calculate_metrics([point.equity for point in points], self.periods_per_year)
        return MultiAssetBacktestResult(tuple(sorted(normalized)), tuple(points), tuple(orders), metrics,
                                        trade_statistics(realized_pnl_samples), rejected)
