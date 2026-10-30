"""Inventory-aware, quote-refreshing illustrative market-making strategy."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Sequence

from ..types import MarketEvent, OrderRequest, OrderType, Side, TimeInForce
from .base import StrategyContext


@dataclass(frozen=True, slots=True)
class MarketMakerConfig:
    quote_quantity: int = 5
    half_spread_ticks: int = 2
    inventory_skew_ticks: Decimal = Decimal("0.25")
    max_inventory: int = 50
    refresh_every_events: int = 1
    max_quote_distance_ticks: int = 20

    def __post_init__(self) -> None:
        if min(self.quote_quantity, self.half_spread_ticks, self.max_inventory,
               self.refresh_every_events, self.max_quote_distance_ticks) <= 0:
            raise ValueError("market maker configuration values must be positive")


class InventoryAwareMarketMaker:
    """Posts one bid and one ask around a skewed fair value.

    This strategy is pedagogical: it does not model adverse selection, queue
    cancellation, hidden liquidity, or venue-specific post-only semantics.
    """

    def __init__(self, strategy_id: str = "mm", config: MarketMakerConfig | None = None) -> None:
        self.strategy_id = strategy_id
        self.config = config or MarketMakerConfig()
        self._counter = 0

    def on_market_data(self, event: MarketEvent, context: StrategyContext) -> Sequence[OrderRequest]:
        self._counter += 1
        if self._counter % self.config.refresh_every_events:
            return ()
        mid = context.mid_ticks
        if mid is None:
            return ()
        inventory = context.position_quantity
        skew = Decimal(inventory) * self.config.inventory_skew_ticks
        fair = mid - skew
        bid = int((fair - self.config.half_spread_ticks).to_integral_value(rounding=ROUND_FLOOR))
        ask = int((fair + self.config.half_spread_ticks).to_integral_value(rounding=ROUND_FLOOR))
        best_bid = context.top.bid.price_ticks if context.top.bid else None
        best_ask = context.top.ask.price_ticks if context.top.ask else None
        bid = min(bid, (best_ask - 1) if best_ask is not None else bid)
        ask = max(ask, (best_bid + 1) if best_bid is not None else ask)
        if bid <= 0 or ask <= bid:
            return ()
        intents: list[OrderRequest] = []
        if inventory < self.config.max_inventory and (best_bid is None or abs(bid - best_bid) <= self.config.max_quote_distance_ticks):
            intents.append(OrderRequest(
                client_order_id=self._next_id("B"), symbol=context.symbol, side=Side.BUY,
                order_type=OrderType.LIMIT, quantity=self.config.quote_quantity,
                limit_price_ticks=bid, time_in_force=TimeInForce.POST_ONLY,
                strategy_id=self.strategy_id,
            ))
        if inventory > -self.config.max_inventory and (best_ask is None or abs(ask - best_ask) <= self.config.max_quote_distance_ticks):
            intents.append(OrderRequest(
                client_order_id=self._next_id("S"), symbol=context.symbol, side=Side.SELL,
                order_type=OrderType.LIMIT, quantity=self.config.quote_quantity,
                limit_price_ticks=ask, time_in_force=TimeInForce.POST_ONLY,
                strategy_id=self.strategy_id,
            ))
        return tuple(intents)

    def _next_id(self, side: str) -> str:
        return f"{self.strategy_id}-{side}-{self._counter:08d}"
