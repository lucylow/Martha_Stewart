"""Quote-based order-flow imbalance and realized-volatility features."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from statistics import fmean, stdev
from typing import Deque

from ..types import BookTop


@dataclass(frozen=True, slots=True)
class OrderFlowObservation:
    ts_ns: int
    bid_price_ticks: int | None
    bid_quantity: int
    ask_price_ticks: int | None
    ask_quantity: int
    order_flow_imbalance: int
    microprice_ticks: Decimal | None
    mid_ticks: Decimal | None
    spread_ticks: int | None


class OrderFlowAccumulator:
    """Compute Cont-style top-of-book OFI and bounded rolling statistics."""

    def __init__(self, window: int = 100) -> None:
        if window <= 1:
            raise ValueError("window must be at least 2")
        self.window = window
        self.previous: BookTop | None = None
        self.observations: Deque[OrderFlowObservation] = deque(maxlen=window)
        self._mid_returns: Deque[Decimal] = deque(maxlen=window)

    def update(self, top: BookTop) -> OrderFlowObservation:
        bid_px = top.bid.price_ticks if top.bid else None
        ask_px = top.ask.price_ticks if top.ask else None
        bid_qty = top.bid.quantity if top.bid else 0
        ask_qty = top.ask.quantity if top.ask else 0
        ofi = 0
        if self.previous is not None:
            prev_bid_px = self.previous.bid.price_ticks if self.previous.bid else None
            prev_ask_px = self.previous.ask.price_ticks if self.previous.ask else None
            prev_bid_qty = self.previous.bid.quantity if self.previous.bid else 0
            prev_ask_qty = self.previous.ask.quantity if self.previous.ask else 0
            if bid_px is not None and prev_bid_px is not None:
                if bid_px > prev_bid_px:
                    ofi += bid_qty
                elif bid_px == prev_bid_px:
                    ofi += bid_qty - prev_bid_qty
                else:
                    ofi -= prev_bid_qty
            if ask_px is not None and prev_ask_px is not None:
                if ask_px < prev_ask_px:
                    ofi -= ask_qty
                elif ask_px == prev_ask_px:
                    ofi += prev_ask_qty - ask_qty
                else:
                    ofi += prev_ask_qty
            prev_mid = self.previous.mid_ticks
            current_mid = top.mid_ticks
            if prev_mid and current_mid and prev_mid > 0:
                self._mid_returns.append((current_mid - prev_mid) / prev_mid)
        microprice = None
        if top.bid and top.ask:
            denominator = Decimal(top.bid.quantity + top.ask.quantity)
            if denominator:
                microprice = (Decimal(top.ask.price_ticks * top.bid.quantity) +
                              Decimal(top.bid.price_ticks * top.ask.quantity)) / denominator
        observation = OrderFlowObservation(
            top.ts_ns, bid_px, bid_qty, ask_px, ask_qty, ofi,
            microprice, top.mid_ticks, top.spread_ticks,
        )
        self.observations.append(observation)
        self.previous = top
        return observation

    @property
    def cumulative_ofi(self) -> int:
        return sum(row.order_flow_imbalance for row in self.observations)

    @property
    def normalized_ofi(self) -> Decimal | None:
        denominator = sum(row.bid_quantity + row.ask_quantity for row in self.observations)
        if denominator == 0:
            return None
        return Decimal(self.cumulative_ofi) / Decimal(denominator)

    def realized_volatility(self, *, annualization_factor: Decimal = Decimal("1")) -> Decimal | None:
        if len(self._mid_returns) < 2:
            return None
        value = Decimal(str(stdev(self._mid_returns))) * annualization_factor.sqrt()
        return value

    def summary(self) -> dict[str, object]:
        spreads = [row.spread_ticks for row in self.observations if row.spread_ticks is not None]
        microprice_edges = [row.microprice_ticks - row.mid_ticks for row in self.observations
                            if row.microprice_ticks is not None and row.mid_ticks is not None]
        return {
            "observations": len(self.observations),
            "cumulative_ofi": self.cumulative_ofi,
            "normalized_ofi": str(self.normalized_ofi) if self.normalized_ofi is not None else None,
            "mean_spread_ticks": fmean(spreads) if spreads else None,
            "mean_microprice_edge_ticks": str(sum(microprice_edges, Decimal("0")) / len(microprice_edges)) if microprice_edges else None,
            "realized_volatility": str(self.realized_volatility()) if self.realized_volatility() is not None else None,
        }
