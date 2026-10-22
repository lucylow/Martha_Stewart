"""Pre-trade risk checks with a reason for every approval or rejection."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from ..config import Settings
from ..types import OrderRequest, PortfolioSnapshot, Quote, RiskDecision

@dataclass(frozen=True, slots=True)
class RiskContext:
    portfolio: PortfolioSnapshot
    quote: Quote
    open_order_count: int = 0
    daily_realized_pnl: float = 0.0
    current_drawdown_fraction: float = 0.0
    trading_halted: bool = False
    now: datetime | None = None

class RiskLimits:
    """Reject invalid/stale/oversized orders before they reach execution."""
    def __init__(self, settings: Settings, clock=None):
        self.settings = settings
        self.clock = clock

    def evaluate(self, request: OrderRequest, context: RiskContext) -> RiskDecision:
        checks: list[str] = []
        reject: list[str] = []
        quote = context.quote
        now = context.now or (self.clock.now() if self.clock else datetime.now(timezone.utc))
        if request.symbol != quote.symbol:
            reject.append("symbol_mismatch")
        age = (now.astimezone(timezone.utc) - quote.timestamp.astimezone(timezone.utc)).total_seconds()
        if age < -1:
            reject.append("future_quote")
        elif age > self.settings.max_quote_age_seconds:
            reject.append("stale_quote")
        else:
            checks.append("quote_fresh")
        if context.trading_halted:
            reject.append("trading_halted")
        else:
            checks.append("kill_switch_clear")
        if context.open_order_count >= self.settings.max_open_orders:
            reject.append("open_order_limit")
        else:
            checks.append("open_order_limit_ok")
        if context.daily_realized_pnl <= -self.settings.max_daily_loss:
            reject.append("daily_loss_limit")
        else:
            checks.append("daily_loss_limit_ok")
        if context.current_drawdown_fraction >= self.settings.max_drawdown_fraction:
            reject.append("drawdown_limit")
        else:
            checks.append("drawdown_limit_ok")
        if request.side.value == "sell" and not self.settings.allow_short:
            held = context.portfolio.positions.get(request.symbol)
            held_qty = max(0.0, held.quantity if held else 0.0)
            if request.quantity > held_qty:
                reject.append("short_selling_disabled")
            else:
                checks.append("sell_within_long_position")
        elif request.side.value == "buy":
            checks.append("long_side_order")
        price = request.limit_price or (quote.ask if request.side.value == "buy" else quote.bid)
        notional = request.quantity * price
        if notional > self.settings.max_order_notional:
            reject.append("max_order_notional")
        else:
            checks.append("max_order_notional_ok")
        if not self.settings.allow_fractional_shares and not float(request.quantity).is_integer():
            reject.append("fractional_quantity_disabled")
        position = context.portfolio.positions.get(request.symbol)
        existing = position.market_value if position else 0.0
        signed = request.side.sign * notional
        prospective_symbol_exposure = abs(existing + signed)
        if prospective_symbol_exposure > self.settings.max_position_notional:
            reject.append("max_position_notional")
        else:
            checks.append("max_position_notional_ok")
        gross = context.portfolio.gross_exposure * context.portfolio.equity
        prospective_gross = gross + (notional if request.side.value == "buy" else max(0.0, notional - abs(existing)))
        if context.portfolio.equity <= 0:
            reject.append("nonpositive_equity")
        elif prospective_gross / context.portfolio.equity > self.settings.max_gross_exposure:
            reject.append("gross_exposure_limit")
        else:
            checks.append("gross_exposure_limit_ok")
        if notional > max(0.0, context.portfolio.cash) and request.side.value == "buy":
            reject.append("insufficient_cash")
        else:
            checks.append("cash_check_ok")
        return RiskDecision(not reject, ",".join(reject) if reject else "approved",
                            notional, 0.0 if reject else request.quantity, tuple(checks))

    def approve_or_raise(self, request: OrderRequest, context: RiskContext) -> RiskDecision:
        from ..errors import RiskRejected
        decision = self.evaluate(request, context)
        if not decision.approved:
            raise RiskRejected(decision.reason)
        return decision
