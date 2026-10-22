"""Liquidity budgets, participation constraints, and execution schedule diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LiquiditySnapshot:
    symbol: str
    timestamp: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    average_daily_volume: float
    spread_bps: float | None = None
    volatility: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.timestamp.strip():
            raise ValueError("symbol and timestamp are required")
        if any(not isfinite(v) for v in (self.bid, self.ask, self.bid_size, self.ask_size, self.average_daily_volume)):
            raise ValueError("liquidity fields must be finite")
        if self.bid < 0 or self.ask < self.bid or min(self.bid_size, self.ask_size, self.average_daily_volume) < 0:
            raise ValueError("invalid bid/ask, size, or volume")
        if self.spread_bps is not None and (not isfinite(self.spread_bps) or self.spread_bps < 0):
            raise ValueError("spread_bps must be finite and non-negative")
        if self.volatility is not None and (not isfinite(self.volatility) or self.volatility < 0):
            raise ValueError("volatility must be finite and non-negative")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def computed_spread_bps(self) -> float:
        return ((self.ask - self.bid) / self.midpoint * 10_000) if self.midpoint else 0.0

    @property
    def top_of_book_imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        return (self.bid_size - self.ask_size) / total if total else 0.0


@dataclass(frozen=True, slots=True)
class LiquidityPolicy:
    max_participation: float = 0.10
    max_spread_bps: float = 50.0
    min_adv: float = 1.0
    max_notional_fraction_of_adv: float = 0.20
    min_top_of_book_size: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        if self.max_spread_bps < 0 or self.min_adv < 0 or not 0 < self.max_notional_fraction_of_adv <= 1:
            raise ValueError("invalid liquidity policy")


@dataclass(frozen=True, slots=True)
class LiquidityDecision:
    allowed: bool
    symbol: str
    requested_quantity: float
    recommended_max_quantity: float
    participation_rate: float
    spread_bps: float
    reasons: tuple[str, ...]


class LiquidityGuard:
    def __init__(self, policy: LiquidityPolicy | None = None) -> None:
        self.policy = policy or LiquidityPolicy()

    def check(self, snapshot: LiquiditySnapshot, *, requested_quantity: float,
              side: str = "buy", reference_price: float | None = None) -> LiquidityDecision:
        if not isfinite(requested_quantity) or requested_quantity <= 0:
            raise ValueError("requested quantity must be positive and finite")
        if side.lower() not in {"buy", "sell", "short", "cover"}:
            raise ValueError("unsupported side")
        price = reference_price if reference_price is not None else snapshot.midpoint
        if not isfinite(price) or price < 0:
            raise ValueError("reference price must be finite and non-negative")
        policy = self.policy
        reasons = []
        if snapshot.average_daily_volume < policy.min_adv:
            reasons.append("average_daily_volume_below_minimum")
        spread = snapshot.spread_bps if snapshot.spread_bps is not None else snapshot.computed_spread_bps
        if spread > policy.max_spread_bps:
            reasons.append("spread_above_maximum")
        if max(snapshot.bid_size, snapshot.ask_size) < policy.min_top_of_book_size:
            reasons.append("top_of_book_size_below_minimum")
        max_by_participation = snapshot.average_daily_volume * policy.max_participation
        max_by_notional = (snapshot.average_daily_volume * price * policy.max_notional_fraction_of_adv / price
                           if price > 0 else max_by_participation)
        max_qty = min(max_by_participation, max_by_notional)
        participation = requested_quantity / snapshot.average_daily_volume if snapshot.average_daily_volume else float("inf")
        if requested_quantity > max_qty + 1e-9:
            reasons.append("requested_quantity_exceeds_liquidity_budget")
        return LiquidityDecision(not reasons, snapshot.symbol, requested_quantity, max(0.0, max_qty),
                                 participation, spread, tuple(reasons))


def vwap_schedule(total_quantity: float, buckets: Sequence[float], *,
                  max_participation: float = 0.10) -> tuple[float, ...]:
    """Allocate quantity proportionally to bucket volumes under per-bucket cap.

    If caps make the order infeasible, the function raises rather than silently
    oversizing a bucket or discarding residual quantity.
    """
    if not isfinite(total_quantity) or total_quantity <= 0:
        raise ValueError("total_quantity must be positive and finite")
    if not buckets or any(not isfinite(v) or v < 0 for v in buckets):
        raise ValueError("bucket volumes must be non-empty and non-negative")
    if not 0 < max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")
    total_volume = sum(buckets)
    if total_volume <= 0:
        raise ValueError("sum of bucket volumes must be positive")
    caps = [v * max_participation for v in buckets]
    if sum(caps) + 1e-9 < total_quantity:
        raise ValueError("requested quantity exceeds aggregate participation capacity")
    weights = [v / total_volume for v in buckets]
    allocation = [0.0] * len(buckets)
    remaining = total_quantity
    open_indices = {i for i, volume in enumerate(buckets) if volume > 0}
    # Iterative waterfill: enforce caps, then redistribute residual across free buckets.
    for _ in range(len(buckets) + 2):
        if remaining <= 1e-9 or not open_indices:
            break
        denom = sum(weights[i] for i in open_indices)
        if denom == 0:
            share = remaining / len(open_indices)
            proposals = {i: share for i in open_indices}
        else:
            proposals = {i: remaining * weights[i] / denom for i in open_indices}
        saturated = [i for i in open_indices if proposals[i] + allocation[i] > caps[i] + 1e-9]
        if not saturated:
            for i, value in proposals.items():
                allocation[i] += value
            remaining = 0.0
            break
        for i in saturated:
            available = max(0.0, caps[i] - allocation[i])
            allocation[i] += available
            remaining -= available
            open_indices.remove(i)
    if remaining > 1e-6:
        raise RuntimeError("could not allocate requested quantity within per-bucket caps")
    # Correct tiny floating residue against the final bucket with slack.
    residue = total_quantity - sum(allocation)
    for i in range(len(allocation)):
        slack = caps[i] - allocation[i]
        if abs(residue) <= 1e-9:
            break
        adjust = min(max(slack, 0.0), residue) if residue > 0 else max(-allocation[i], residue)
        allocation[i] += adjust
        residue -= adjust
    return tuple(allocation)


def liquidity_summary(snapshots: Sequence[LiquiditySnapshot]) -> dict[str, float | int]:
    if not snapshots:
        return {"symbols": 0, "median_spread_bps": 0.0, "total_adv": 0.0,
                "mean_imbalance": 0.0, "low_liquidity_symbols": 0}
    spreads = sorted(s.spread_bps if s.spread_bps is not None else s.computed_spread_bps for s in snapshots)
    mid = len(spreads) // 2
    med = spreads[mid] if len(spreads) % 2 else (spreads[mid - 1] + spreads[mid]) / 2
    return {"symbols": len(snapshots), "median_spread_bps": med,
            "total_adv": sum(s.average_daily_volume for s in snapshots),
            "mean_imbalance": sum(s.top_of_book_imbalance for s in snapshots) / len(snapshots),
            "low_liquidity_symbols": sum(s.average_daily_volume <= 0 or (s.spread_bps or s.computed_spread_bps) > 50
                                          for s in snapshots)}
