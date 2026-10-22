"""Turnover-aware target-weight to order-delta planning without broker side effects."""
from __future__ import annotations
from dataclasses import dataclass
from math import floor, isfinite
from typing import Mapping
from ..types import Side


@dataclass(frozen=True, slots=True)
class RebalanceTrade:
    symbol: str
    side: Side
    quantity: float
    current_weight: float
    target_weight: float
    delta_notional: float
    estimated_cost: float
    reason: str


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    equity: float
    trades: tuple[RebalanceTrade, ...]
    gross_turnover: float
    estimated_cost: float
    symbols_skipped: Mapping[str, str]

    @property
    def order_count(self) -> int:
        return len(self.trades)


def build_rebalance_plan(*, equity: float, current_quantities: Mapping[str, float],
                         prices: Mapping[str, float], target_weights: Mapping[str, float],
                         current_cash: float = 0.0, minimum_trade_notional: float = 50.0,
                         no_trade_band: float = 0.0025, commission_bps: float = 1.0,
                         slippage_bps: float = 2.0, lot_size: int = 1,
                         allow_fractional: bool = False) -> RebalancePlan:
    """Generate a sorted rebalance plan; risk approval/execution is a separate step."""
    scalars = (equity, current_cash, minimum_trade_notional, no_trade_band, commission_bps, slippage_bps)
    if any(not isfinite(float(x)) for x in scalars):
        raise ValueError("rebalance inputs must be finite")
    if equity <= 0 or minimum_trade_notional < 0 or no_trade_band < 0 or commission_bps < 0 or slippage_bps < 0:
        raise ValueError("invalid rebalance thresholds")
    if lot_size < 1:
        raise ValueError("lot_size must be >= 1")
    symbols = sorted(set(current_quantities) | set(prices) | set(target_weights))
    target_normalized = {s.strip().upper(): float(w) for s, w in target_weights.items()}
    qty_normalized = {s.strip().upper(): float(q) for s, q in current_quantities.items()}
    price_normalized = {s.strip().upper(): float(p) for s, p in prices.items()}
    if len(target_normalized) != len(target_weights) or len(qty_normalized) != len(current_quantities) or len(price_normalized) != len(prices):
        raise ValueError("symbol names collide after normalization")
    trades: list[RebalanceTrade] = []
    skipped: dict[str, str] = {}
    gross = cost_total = 0.0
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        target = target_normalized.get(symbol, 0.0)
        current_qty = qty_normalized.get(symbol, 0.0)
        price = price_normalized.get(symbol)
        if not isfinite(target) or not isfinite(current_qty) or current_qty < 0:
            raise ValueError(f"invalid target/current quantity for {symbol}")
        if price is None or not isfinite(price) or price <= 0:
            if abs(target) > 0 or current_qty:
                skipped[symbol] = "missing_or_invalid_price"
            continue
        if target < 0:
            raise ValueError("short target weights are not supported by this long-only rebalance planner")
        current_value = current_qty * price
        current_weight = current_value / equity
        weight_delta = target - current_weight
        if abs(weight_delta) < no_trade_band:
            skipped[symbol] = "within_no_trade_band"
            continue
        dollar_delta = weight_delta * equity
        raw_qty = abs(dollar_delta) / price
        quantity = raw_qty if allow_fractional else floor(raw_qty / lot_size) * lot_size
        if quantity <= 0:
            skipped[symbol] = "below_one_lot"
            continue
        notional = quantity * price
        if notional < minimum_trade_notional:
            skipped[symbol] = "below_minimum_trade_notional"
            continue
        side = Side.BUY if weight_delta > 0 else Side.SELL
        commission = notional * commission_bps / 10_000
        slippage = notional * slippage_bps / 10_000
        estimated_cost = commission + slippage
        trades.append(RebalanceTrade(symbol, side, float(quantity), current_weight, target,
                                     (1 if side is Side.BUY else -1) * notional,
                                     estimated_cost, "target_weight_delta"))
        gross += notional
        cost_total += estimated_cost
    trades.sort(key=lambda t: (t.side is Side.SELL, -abs(t.delta_notional), t.symbol))
    return RebalancePlan(equity, tuple(trades), gross / equity, cost_total, skipped)


def apply_trade_weights(current_weights: Mapping[str, float], trades: Mapping[str, float]) -> dict[str, float]:
    """Pure helper applying signed weight deltas and validating finite output."""
    result = {str(s).strip().upper(): float(w) for s, w in current_weights.items()}
    for symbol, delta in trades.items():
        key = str(symbol).strip().upper()
        result[key] = result.get(key, 0.0) + float(delta)
    if any(not isfinite(v) for v in result.values()):
        raise ValueError("weights and deltas must be finite")
    return result
