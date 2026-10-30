"""Seeded synthetic L3 event feed and stress transformations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from ..types import EventKind, MarketEvent, Side


@dataclass(frozen=True, slots=True)
class SyntheticFeedConfig:
    symbol: str = "SIM"
    start_ns: int = 1_000_000_000
    start_mid_ticks: int = 10_000
    spread_ticks: int = 2
    initial_levels: int = 5
    initial_quantity: int = 100
    events: int = 2_000
    interval_ns: int = 1_000_000
    price_step_ticks: int = 1
    min_quantity: int = 1
    max_quantity: int = 20
    cancel_probability: float = 0.25
    trade_probability: float = 0.35
    price_drift_probability: float = 0.5

    def __post_init__(self) -> None:
        if not self.symbol.strip() or self.start_ns < 0 or self.start_mid_ticks <= 0:
            raise ValueError("invalid symbol, start time, or start price")
        integer_values = (self.spread_ticks, self.initial_levels, self.initial_quantity,
                          self.interval_ns, self.price_step_ticks, self.min_quantity, self.max_quantity)
        if any(x <= 0 for x in integer_values) or self.events < 0:
            raise ValueError("feed integer parameters must be positive (events may be zero)")
        if self.min_quantity > self.max_quantity:
            raise ValueError("min_quantity must not exceed max_quantity")
        probabilities = (self.cancel_probability, self.trade_probability, self.price_drift_probability)
        if any(not 0 <= p <= 1 for p in probabilities):
            raise ValueError("probabilities must be between zero and one")


class SyntheticMarketFeed:
    """Generates a reproducible valid sequence of external book messages."""

    def __init__(self, config: SyntheticFeedConfig | None = None, *, seed: int = 7) -> None:
        self.config = config or SyntheticFeedConfig()
        self.random = random.Random(seed)
        self.seed = seed
        self._next_id = 0
        self._next_sequence = 0
        self._active: dict[str, tuple[Side, int, int]] = {}
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}

    def _id(self) -> str:
        self._next_id += 1
        return f"SYN-{self._next_id:010d}"

    def _event(self, ts_ns: int, kind: EventKind, *, side: Side | None = None,
               price: int | None = None, quantity: int | None = None, order_id: str = "") -> MarketEvent:
        event = MarketEvent(ts_ns, self._next_sequence, self.config.symbol, kind,
                            side, price, quantity, order_id)
        self._next_sequence += 1
        return event

    def _add(self, ts_ns: int, side: Side, price: int, quantity: int) -> MarketEvent:
        order_id = self._id()
        self._active[order_id] = (side, price, quantity)
        levels = self._bids if side is Side.BUY else self._asks
        levels[price] = levels.get(price, 0) + quantity
        return self._event(ts_ns, EventKind.ADD, side=side, price=price, quantity=quantity, order_id=order_id)

    def _remove(self, order_id: str, quantity: int | None = None) -> None:
        side, price, old_quantity = self._active[order_id]
        removed = old_quantity if quantity is None else quantity
        if removed > old_quantity:
            raise ValueError("cannot remove more than active quantity")
        levels = self._bids if side is Side.BUY else self._asks
        remaining = old_quantity - removed
        levels[price] -= removed
        if levels[price] <= 0:
            del levels[price]
        if remaining:
            self._active[order_id] = (side, price, remaining)
        else:
            del self._active[order_id]

    def _best(self, side: Side) -> int | None:
        levels = self._bids if side is Side.BUY else self._asks
        if not levels:
            return None
        return max(levels) if side is Side.BUY else min(levels)

    def _trade(self, ts_ns: int, aggressor: Side, quantity: int) -> MarketEvent | None:
        maker_side = aggressor.opposite
        limit = self._best(maker_side)
        if limit is None:
            return None
        # Sweep up to a small number of levels, but keep the print limit explicit.
        if aggressor is Side.BUY:
            price_limit = limit + self.config.price_step_ticks * self.random.randint(0, 2)
            prices = sorted((p for p in self._asks if p <= price_limit))
        else:
            price_limit = limit - self.config.price_step_ticks * self.random.randint(0, 2)
            prices = sorted((p for p in self._bids if p >= price_limit), reverse=True)
        remaining = quantity
        for price in prices:
            ids = [order_id for order_id, (side, px, _) in self._active.items() if side is maker_side and px == price]
            for order_id in ids:
                if remaining <= 0:
                    break
                available = self._active[order_id][2]
                take = min(available, remaining)
                self._remove(order_id, take)
                remaining -= take
            if remaining <= 0:
                break
        traded = quantity - remaining
        if traded <= 0:
            return None
        return self._event(ts_ns, EventKind.TRADE, side=aggressor, price=price_limit, quantity=traded)

    def generate(self) -> tuple[MarketEvent, ...]:
        cfg = self.config
        events: list[MarketEvent] = []
        mid = cfg.start_mid_ticks
        bid_start = mid - cfg.spread_ticks // 2
        ask_start = mid + (cfg.spread_ticks - cfg.spread_ticks // 2)
        for level in range(cfg.initial_levels):
            events.append(self._add(cfg.start_ns, Side.BUY, bid_start - level * cfg.price_step_ticks, cfg.initial_quantity))
            events.append(self._add(cfg.start_ns, Side.SELL, ask_start + level * cfg.price_step_ticks, cfg.initial_quantity))
        for index in range(cfg.events):
            ts = cfg.start_ns + (index + 1) * cfg.interval_ns
            if self.random.random() < cfg.price_drift_probability:
                mid += self.random.choice((-1, 0, 1)) * cfg.price_step_ticks
                mid = max(10, mid)
            choice = self.random.random()
            quantity = self.random.randint(cfg.min_quantity, cfg.max_quantity)
            if choice < cfg.cancel_probability and self._active:
                order_id = self.random.choice(tuple(self._active))
                side, price, old_qty = self._active[order_id]
                if old_qty <= quantity:
                    self._remove(order_id)
                    events.append(self._event(ts, EventKind.CANCEL, order_id=order_id))
                else:
                    self._remove(order_id, quantity)
                    events.append(self._event(ts, EventKind.REDUCE, side=side, quantity=quantity, order_id=order_id))
            elif choice < cfg.cancel_probability + cfg.trade_probability:
                aggressor = self.random.choice((Side.BUY, Side.SELL))
                event = self._trade(ts, aggressor, quantity)
                if event is not None:
                    events.append(event)
            else:
                side = self.random.choice((Side.BUY, Side.SELL))
                best = self._best(side)
                if best is None:
                    price = mid - cfg.spread_ticks if side is Side.BUY else mid + cfg.spread_ticks
                else:
                    price = best - cfg.price_step_ticks if side is Side.BUY else best + cfg.price_step_ticks
                opposite = self._best(side.opposite)
                if opposite is not None:
                    if side is Side.BUY:
                        price = min(price, opposite - 1)
                    else:
                        price = max(price, opposite + 1)
                events.append(self._add(ts, side, max(1, price), quantity))
        events.sort(key=lambda x: (x.ts_ns, x.sequence))
        return tuple(events)


class ScenarioTransforms:
    """Pure deterministic transforms for liquidity, volume, and feed stress."""

    @staticmethod
    def scale_quantities(events: Iterable[MarketEvent], factor: float) -> tuple[MarketEvent, ...]:
        if factor <= 0:
            raise ValueError("factor must be positive")
        output = []
        for event in events:
            quantity = event.quantity
            if quantity is not None:
                quantity = max(1, int(round(quantity * factor)))
            output.append(MarketEvent(event.ts_ns, event.sequence, event.symbol, event.kind,
                                     event.side, event.price_ticks, quantity, event.order_id,
                                     event.reference_price_ticks))
        return tuple(output)

    @staticmethod
    def delay_feed(events: Iterable[MarketEvent], delay_ns: int, *, jitter_ns: int = 0, seed: int = 7) -> tuple[MarketEvent, ...]:
        if delay_ns < 0 or jitter_ns < 0:
            raise ValueError("delay and jitter must be non-negative")
        rng = random.Random(seed)
        rows = []
        for index, event in enumerate(events):
            jitter = rng.randint(-jitter_ns, jitter_ns) if jitter_ns else 0
            ts = max(0, event.ts_ns + delay_ns + jitter)
            rows.append((ts, index, event))
        rows.sort(key=lambda item: (item[0], item[1]))
        return tuple(MarketEvent(ts, seq, event.symbol, event.kind, event.side,
                                 event.price_ticks, event.quantity, event.order_id,
                                 event.reference_price_ticks)
                     for seq, (ts, _, event) in enumerate(rows))

    @staticmethod
    def drop_fraction(events: Iterable[MarketEvent], fraction: float, *, seed: int = 7) -> tuple[MarketEvent, ...]:
        if not 0 <= fraction < 1:
            raise ValueError("fraction must be in [0, 1)")
        rng = random.Random(seed)
        kept = [event for event in events if rng.random() >= fraction]
        return tuple(MarketEvent(event.ts_ns, seq, event.symbol, event.kind, event.side,
                                 event.price_ticks, event.quantity, event.order_id,
                                 event.reference_price_ticks)
                     for seq, event in enumerate(kept))

    @staticmethod
    def liquidity_pulse(events: Iterable[MarketEvent], *, start_ns: int, end_ns: int, side: Side,
                        cancel_fraction: float = 1.0) -> tuple[MarketEvent, ...]:
        if end_ns < start_ns or not 0 <= cancel_fraction <= 1:
            raise ValueError("invalid pulse range or fraction")
        rows = list(events)
        active: dict[str, tuple[Side, int]] = {}
        additions: list[MarketEvent] = []
        sequence = max((x.sequence for x in rows), default=-1) + 1
        for event in sorted(rows, key=lambda x: (x.ts_ns, x.sequence)):
            if event.kind is EventKind.ADD:
                active[event.order_id] = (event.side, event.ts_ns)
            elif event.kind is EventKind.CANCEL:
                active.pop(event.order_id, None)
            elif event.kind is EventKind.REDUCE and event.order_id in active:
                pass
        candidates = [order_id for order_id, (order_side, added_ns) in active.items()
                      if order_side is side and added_ns <= start_ns]
        count = int(len(candidates) * cancel_fraction)
        for order_id in candidates[:count]:
            # The simulator removes resting size; quantity is intentionally
            # omitted for CANCEL events and the event timestamp is the shock start.
            additions.append(MarketEvent(start_ns, sequence, rows[0].symbol if rows else "SIM",
                                         EventKind.CANCEL, order_id=order_id))
            sequence += 1
        return tuple(sorted(rows + additions, key=lambda x: (x.ts_ns, x.sequence)))
