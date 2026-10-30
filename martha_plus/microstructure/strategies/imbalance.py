"""Simple order-book imbalance strategy for controlled experiments."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from ..types import MarketEvent, OrderRequest, OrderType, Side, TimeInForce
from .base import StrategyContext


@dataclass(frozen=True, slots=True)
class ImbalanceConfig:
    threshold: Decimal = Decimal("0.65")
    quantity: int = 1
    protection_ticks: int = 2
    max_spread_ticks: int = 4
    depth_levels: int = 3
    cooldown_events: int = 5

    def __post_init__(self) -> None:
        if not Decimal("0.5") < self.threshold <= Decimal("1"):
            raise ValueError("threshold must be in (0.5, 1]")
        if min(self.quantity, self.protection_ticks, self.max_spread_ticks,
               self.depth_levels, self.cooldown_events) <= 0:
            raise ValueError("imbalance configuration values must be positive")


class ImbalanceTaker:
    """Submits protected IOC orders when displayed depth is strongly skewed."""

    def __init__(self, strategy_id: str = "imbalance", config: ImbalanceConfig | None = None) -> None:
        self.strategy_id = strategy_id
        self.config = config or ImbalanceConfig()
        self._last_signal_event = -10**12
        self._event_count = 0
        self._order_sequence = 0

    def on_market_data(self, event: MarketEvent, context: StrategyContext) -> Sequence[OrderRequest]:
        self._event_count += 1
        spread = context.top.spread_ticks
        if spread is None or spread > self.config.max_spread_ticks:
            return ()
        bid_qty = sum(x.quantity for x in context.depth.bids[:self.config.depth_levels])
        ask_qty = sum(x.quantity for x in context.depth.asks[:self.config.depth_levels])
        total = bid_qty + ask_qty
        if total == 0:
            return ()
        imbalance = Decimal(bid_qty - ask_qty) / Decimal(total)
        if abs(imbalance) < self.config.threshold:
            return ()
        if self._event_count - self._last_signal_event < self.config.cooldown_events:
            return ()
        side = Side.BUY if imbalance > 0 else Side.SELL
        if side is Side.BUY and context.top.ask is None:
            return ()
        if side is Side.SELL and context.top.bid is None:
            return ()
        self._last_signal_event = self._event_count
        self._order_sequence += 1
        if side is Side.BUY:
            protection = context.top.ask.price_ticks + self.config.protection_ticks
            req = OrderRequest(
                f"{self.strategy_id}-{self._order_sequence:08d}", context.symbol, side,
                OrderType.MARKET, self.config.quantity,
                max_price_ticks=protection, time_in_force=TimeInForce.IOC,
                strategy_id=self.strategy_id,
            )
        else:
            protection = context.top.bid.price_ticks - self.config.protection_ticks
            if protection <= 0:
                return ()
            req = OrderRequest(
                f"{self.strategy_id}-{self._order_sequence:08d}", context.symbol, side,
                OrderType.MARKET, self.config.quantity,
                min_price_ticks=protection, time_in_force=TimeInForce.IOC,
                strategy_id=self.strategy_id,
            )
        return (req,)
