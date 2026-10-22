"""In-memory portfolio ledger with realized/unrealized P&L accounting."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone, date
from threading import RLock
from typing import Mapping
from ..errors import ExecutionError
from ..types import Fill, Position, PortfolioSnapshot

class PortfolioLedger:
    def __init__(self, starting_cash: float = 100_000.0):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self._quantities: dict[str, float] = defaultdict(float)
        self._average_costs: dict[str, float] = defaultdict(float)
        self._marks: dict[str, float] = {}
        self._realized_by_day: dict[date, float] = defaultdict(float)
        self._processed_fills: set[str] = set()
        self._fills: list[Fill] = []
        self._lock = RLock()

    @property
    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._fills)

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill exactly once. Replayed fills with the same fill_id are ignored."""
        with self._lock:
            if fill.fill_id in self._processed_fills:
                return
            symbol = fill.symbol.strip().upper()
            old_qty = self._quantities[symbol]
            old_cost = self._average_costs[symbol]
            signed_qty = fill.quantity * fill.side.sign
            self.cash -= signed_qty * fill.price
            self.cash -= fill.commission
            realized = 0.0
            if old_qty == 0 or old_qty * signed_qty > 0:
                new_qty = old_qty + signed_qty
                if new_qty:
                    self._average_costs[symbol] = (abs(old_qty) * old_cost + abs(signed_qty) * fill.price) / abs(new_qty)
                self._quantities[symbol] = new_qty
            else:
                closed_qty = min(abs(old_qty), abs(signed_qty))
                direction = 1.0 if old_qty > 0 else -1.0
                realized = closed_qty * (fill.price - old_cost) * direction
                new_qty = old_qty + signed_qty
                if new_qty == 0:
                    self._average_costs[symbol] = 0.0
                elif old_qty * new_qty < 0:
                    self._average_costs[symbol] = fill.price
                self._quantities[symbol] = new_qty
            # Attribute fees to realized P&L even for opening fills.
            realized -= fill.commission
            self._realized_by_day[fill.timestamp.astimezone(timezone.utc).date()] += realized
            self._marks[symbol] = fill.price
            self._processed_fills.add(fill.fill_id)
            self._fills.append(fill)

    def mark(self, prices: Mapping[str, float]) -> None:
        with self._lock:
            for raw_symbol, price in prices.items():
                if price <= 0:
                    raise ValueError("mark prices must be positive")
                symbol = raw_symbol.strip().upper()
                if self._quantities.get(symbol, 0.0):
                    self._marks[symbol] = float(price)

    def positions(self) -> dict[str, Position]:
        with self._lock:
            result = {}
            for symbol, quantity in self._quantities.items():
                if abs(quantity) < 1e-12:
                    continue
                mark = self._marks.get(symbol, self._average_costs[symbol])
                result[symbol] = Position(symbol, quantity, self._average_costs[symbol], mark)
            return result

    def realized_pnl(self, on_date: date | None = None) -> float:
        with self._lock:
            if on_date is not None:
                return self._realized_by_day.get(on_date, 0.0)
            return sum(self._realized_by_day.values())

    def unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.positions().values())

    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions().values())

    def snapshot(self, now: datetime | None = None, high_water_mark: float | None = None,
                 day_start_equity: float | None = None) -> PortfolioSnapshot:
        with self._lock:
            timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            positions = self.positions()
            market_values = [p.market_value for p in positions.values()]
            equity = self.cash + sum(market_values)
            gross = sum(abs(v) for v in market_values)
            net = sum(market_values)
            gross_fraction = gross / equity if equity > 0 else float("inf")
            net_fraction = net / equity if equity > 0 else float("inf")
            daily_realized = self.realized_pnl(timestamp.date())
            if day_start_equity is not None:
                daily_pnl = equity - day_start_equity
            else:
                daily_pnl = daily_realized + sum(p.unrealized_pnl for p in positions.values())
            return PortfolioSnapshot(timestamp, self.cash, equity, gross_fraction,
                                     net_fraction, daily_pnl, high_water_mark or max(self.starting_cash, equity),
                                     positions)

    def reset(self) -> None:
        """Clear all in-memory state. Intended for test setup, not production recovery."""
        with self._lock:
            self.cash = self.starting_cash
            self._quantities.clear()
            self._average_costs.clear()
            self._marks.clear()
            self._realized_by_day.clear()
            self._processed_fills.clear()
            self._fills.clear()
