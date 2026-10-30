"""Strategy interface and immutable observation context."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, Sequence

from ..book.l2 import DepthSnapshot
from ..types import BookTop, MarketEvent, OrderRequest


@dataclass(frozen=True, slots=True)
class StrategyContext:
    symbol: str
    decision_ts_ns: int
    source_event_ts_ns: int
    top: BookTop
    depth: DepthSnapshot
    position_quantity: int = 0
    cash: Decimal = Decimal("0")
    equity: Decimal = Decimal("0")
    open_order_count: int = 0

    @property
    def mid_ticks(self) -> Decimal | None:
        return self.top.mid_ticks


class Strategy(Protocol):
    """A deterministic strategy receives one market-data observation at a time."""

    strategy_id: str

    def on_market_data(self, event: MarketEvent, context: StrategyContext) -> Sequence[OrderRequest]: ...


class NoOpStrategy:
    strategy_id = "noop"

    def on_market_data(self, event: MarketEvent, context: StrategyContext) -> Sequence[OrderRequest]:
        return ()
