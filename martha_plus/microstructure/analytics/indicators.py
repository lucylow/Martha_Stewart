"""Microstructure indicators computed from integer-tick book observations."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from statistics import fmean
from typing import Deque, Iterable

from ..types import BookTop, Side


@dataclass(frozen=True, slots=True)
class BookFeatures:
    ts_ns: int
    mid_ticks: Decimal | None
    microprice_ticks: Decimal | None
    spread_ticks: int | None
    relative_spread_bps: Decimal | None
    top_imbalance: Decimal | None
    weighted_imbalance: Decimal | None
    depth_slope_bid: Decimal | None
    depth_slope_ask: Decimal | None


def microprice(top: BookTop) -> Decimal | None:
    """Queue-size weighted midpoint; None when either side is absent."""
    if top.bid is None or top.ask is None:
        return None
    total = top.bid.quantity + top.ask.quantity
    if total <= 0:
        return None
    return (Decimal(top.ask.price_ticks * top.bid.quantity) +
            Decimal(top.bid.price_ticks * top.ask.quantity)) / Decimal(total)


def top_imbalance(top: BookTop) -> Decimal | None:
    if top.bid is None or top.ask is None:
        return None
    total = top.bid.quantity + top.ask.quantity
    return Decimal(top.bid.quantity - top.ask.quantity) / Decimal(total) if total else None


def relative_spread_bps(top: BookTop) -> Decimal | None:
    mid = top.mid_ticks
    if mid is None or mid <= 0 or top.spread_ticks is None:
        return None
    return Decimal(top.spread_ticks) / mid * Decimal("10000")


class BookFeatureExtractor:
    """Build stateless top-of-book features and maintain optional history."""

    def __init__(self, *, history: int = 256, decay: float = 0.8) -> None:
        if history <= 1 or not 0 < decay < 1:
            raise ValueError("invalid feature history or decay")
        self._history: Deque[BookFeatures] = deque(maxlen=history)
        self.decay = decay
        self._weighted_imbalance: float | None = None
        self._last_mid: Decimal | None = None
        self._returns: Deque[float] = deque(maxlen=history)

    def update(self, top: BookTop) -> BookFeatures:
        mid = top.mid_ticks
        imbalance = top_imbalance(top)
        if imbalance is not None:
            raw = float(imbalance)
            self._weighted_imbalance = raw if self._weighted_imbalance is None else (
                self.decay * self._weighted_imbalance + (1 - self.decay) * raw
            )
        if mid is not None and self._last_mid is not None and self._last_mid > 0:
            self._returns.append(float(mid / self._last_mid - 1))
        if mid is not None:
            self._last_mid = mid
        weighted = Decimal(str(self._weighted_imbalance)) if self._weighted_imbalance is not None else None
        row = BookFeatures(top.ts_ns, mid, microprice(top), top.spread_ticks,
                           relative_spread_bps(top), imbalance, weighted, None, None)
        self._history.append(row)
        return row

    @property
    def history(self) -> tuple[BookFeatures, ...]:
        return tuple(self._history)

    def realized_volatility(self, *, periods_per_year: float = 1.0) -> Decimal | None:
        if periods_per_year <= 0 or len(self._returns) < 2:
            return None
        values = list(self._returns)
        mean = fmean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return Decimal(str(sqrt(variance * periods_per_year)))

    def mean_spread(self) -> Decimal | None:
        values = [Decimal(row.spread_ticks) for row in self._history if row.spread_ticks is not None]
        return sum(values, Decimal(0)) / len(values) if values else None

    def mean_imbalance(self) -> Decimal | None:
        values = [row.top_imbalance for row in self._history if row.top_imbalance is not None]
        return sum(values, Decimal(0)) / len(values) if values else None


@dataclass(frozen=True, slots=True)
class TradePrint:
    ts_ns: int
    price_ticks: int
    quantity: int
    aggressor: Side

    def __post_init__(self) -> None:
        if self.ts_ns < 0 or self.price_ticks <= 0 or self.quantity <= 0:
            raise ValueError("invalid trade print")


@dataclass(frozen=True, slots=True)
class FlowToxicity:
    buckets: int
    buy_volume: int
    sell_volume: int
    imbalance_volume: int
    total_volume: int
    normalized_imbalance: Decimal | None


class VolumeImbalanceTracker:
    """Simplified volume-imbalance toxicity proxy.

    This is not a canonical VPIN implementation: it uses fixed event buckets
    rather than equal-volume buckets and labels the result accordingly.
    """

    def __init__(self, bucket_trades: int = 100, max_buckets: int = 50) -> None:
        if bucket_trades <= 0 or max_buckets <= 0:
            raise ValueError("bucket sizes must be positive")
        self.bucket_trades = bucket_trades
        self._buckets: Deque[tuple[int, int]] = deque(maxlen=max_buckets)
        self._buy = 0
        self._sell = 0
        self._events = 0

    def update(self, trade: TradePrint) -> FlowToxicity:
        if trade.aggressor is Side.BUY:
            self._buy += trade.quantity
        else:
            self._sell += trade.quantity
        self._events += 1
        if self._events >= self.bucket_trades:
            self._buckets.append((self._buy, self._sell))
            self._buy = self._sell = self._events = 0
        return self.snapshot()

    def snapshot(self) -> FlowToxicity:
        buy = sum(x[0] for x in self._buckets) + self._buy
        sell = sum(x[1] for x in self._buckets) + self._sell
        total = buy + sell
        imbalance = abs(buy - sell)
        return FlowToxicity(len(self._buckets) + bool(self._events), buy, sell,
                            imbalance, total,
                            Decimal(imbalance) / Decimal(total) if total else None)


class KyleLambdaEstimator:
    """Rolling OLS estimate of price change per signed traded quantity."""

    def __init__(self, window: int = 200) -> None:
        if window < 3:
            raise ValueError("window must be at least 3")
        self._pairs: Deque[tuple[float, float]] = deque(maxlen=window)
        self._last_mid: float | None = None

    def update(self, mid_ticks: int | Decimal, signed_volume: int) -> Decimal | None:
        mid = float(mid_ticks)
        if mid <= 0:
            raise ValueError("mid must be positive")
        if self._last_mid is not None:
            self._pairs.append((float(signed_volume), mid - self._last_mid))
        self._last_mid = mid
        return self.value

    @property
    def value(self) -> Decimal | None:
        if len(self._pairs) < 3:
            return None
        xs = [x for x, _ in self._pairs]
        ys = [y for _, y in self._pairs]
        xbar, ybar = fmean(xs), fmean(ys)
        denominator = sum((x - xbar) ** 2 for x in xs)
        if denominator <= 0:
            return None
        slope = sum((x - xbar) * (y - ybar) for x, y in self._pairs) / denominator
        return Decimal(str(slope))


class AmihudIlliquidity:
    """Rolling mean of absolute return divided by traded notional proxy."""

    def __init__(self, window: int = 100) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._observations: Deque[float] = deque(maxlen=window)
        self._last_price: float | None = None

    def update(self, price_ticks: int, quantity: int) -> Decimal | None:
        if price_ticks <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        current = float(price_ticks)
        if self._last_price is not None:
            ret = abs(current / self._last_price - 1.0)
            notional = current * quantity
            self._observations.append(ret / notional if notional else 0.0)
        self._last_price = current
        return self.value

    @property
    def value(self) -> Decimal | None:
        return Decimal(str(fmean(self._observations))) if self._observations else None


class AdverseSelectionTracker:
    """Measure signed post-fill markouts over caller-supplied horizons."""

    def __init__(self, horizons_ns: Iterable[int] = (1_000_000, 10_000_000, 100_000_000)) -> None:
        horizons = tuple(sorted(set(horizons_ns)))
        if not horizons or horizons[0] <= 0:
            raise ValueError("horizons must be positive")
        self.horizons_ns = horizons
        self._pending: Deque[tuple[int, int, Side, int, set[int]]] = deque()
        self._results: dict[int, list[Decimal]] = {h: [] for h in horizons}

    def record_fill(self, ts_ns: int, price_ticks: int, side: Side, quantity: int) -> None:
        if min(ts_ns, quantity) < 0 or price_ticks <= 0 or quantity <= 0:
            raise ValueError("invalid fill")
        self._pending.append((ts_ns, price_ticks, side, quantity, set()))

    def observe_mid(self, ts_ns: int, mid_ticks: int | Decimal) -> None:
        mid = Decimal(mid_ticks)
        if ts_ns < 0 or mid <= 0:
            raise ValueError("invalid midpoint observation")
        remaining = deque()
        while self._pending:
            fill_ts, fill_px, side, qty, completed = self._pending.popleft()
            for horizon in self.horizons_ns:
                if horizon not in completed and ts_ns >= fill_ts + horizon:
                    markout = (mid - Decimal(fill_px)) * side.sign / Decimal(fill_px) * Decimal(10000)
                    self._results[horizon].append(markout)
                    completed.add(horizon)
            if len(completed) < len(self.horizons_ns):
                remaining.append((fill_ts, fill_px, side, qty, completed))
        self._pending = remaining

    def summary(self) -> dict[int, dict[str, str | int | None]]:
        result = {}
        for horizon, values in self._results.items():
            result[horizon] = {
                "count": len(values),
                "mean_markout_bps": str(sum(values, Decimal(0)) / len(values)) if values else None,
                "positive_fraction": str(Decimal(sum(x > 0 for x in values)) / len(values)) if values else None,
            }
        return result
