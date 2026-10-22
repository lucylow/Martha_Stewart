"""Deterministic paper-order slicing plans (TWAP, VWAP, and participation caps).

These functions generate schedules only. They do not submit orders, connect to
venues, promise fills, or replace the base package's risk checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Sequence
from ..types import Side


@dataclass(frozen=True, slots=True)
class Slice:
    sequence: int
    scheduled_at: datetime
    quantity: float
    cumulative_quantity: float
    fraction: float
    method: str


@dataclass(frozen=True, slots=True)
class SlicePlan:
    symbol: str
    side: Side
    total_quantity: float
    start_at: datetime
    end_at: datetime
    method: str
    slices: tuple[Slice, ...]
    metadata: dict[str, float | int | str]

    def validate(self, tolerance: float = 1e-8) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.slices:
            return ("plan must contain at least one slice",)
        if any(s.quantity <= 0 or not isfinite(s.quantity) for s in self.slices):
            errors.append("slice quantities must be positive finite values")
        if abs(sum(s.quantity for s in self.slices) - self.total_quantity) > tolerance:
            errors.append("slice quantities do not sum to total quantity")
        if [s.sequence for s in self.slices] != list(range(1, len(self.slices) + 1)):
            errors.append("slice sequence numbers are not contiguous")
        times = [s.scheduled_at for s in self.slices]
        if times != sorted(times):
            errors.append("slice timestamps are not sorted")
        if any(t < self.start_at or t > self.end_at for t in times):
            errors.append("slice timestamp outside plan interval")
        if self.start_at > self.end_at:
            errors.append("plan start_at must not be after end_at")
        return tuple(errors)


def _validate_common(symbol: str, side: Side, quantity: float, start_at: datetime,
                     end_at: datetime, slice_count: int) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    if not isinstance(side, Side):
        raise ValueError("side must be a Side enum")
    if not isfinite(quantity) or quantity <= 0:
        raise ValueError("quantity must be positive and finite")
    if start_at.tzinfo is None or end_at.tzinfo is None:
        raise ValueError("schedule timestamps must be timezone-aware")
    if end_at < start_at:
        raise ValueError("end_at must not be before start_at")
    if slice_count < 1:
        raise ValueError("slice_count must be >= 1")
    return normalized


def _make_plan(symbol: str, side: Side, quantity: float, start_at: datetime, end_at: datetime,
               method: str, weights: Sequence[float], metadata: dict[str, float | int | str]) -> SlicePlan:
    if any(not isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("slice weights must be finite, non-negative, with positive sum")
    normalized_weights = [w / sum(weights) for w in weights]
    slices: list[Slice] = []
    cumulative = 0.0
    duration = (end_at - start_at).total_seconds()
    for i, fraction in enumerate(normalized_weights, start=1):
        part = quantity * fraction
        # Ensure the final slice absorbs floating point residual without changing the intended total.
        if i == len(normalized_weights):
            part = quantity - cumulative
            fraction = part / quantity
        cumulative += part
        at = start_at + timedelta(seconds=duration * (i - 1) / max(1, len(normalized_weights) - 1))
        slices.append(Slice(i, at, part, cumulative, fraction, method))
    plan = SlicePlan(symbol, side, quantity, start_at, end_at, method, tuple(slices), metadata)
    errors = plan.validate(tolerance=max(1e-8, quantity * 1e-10))
    if errors:
        raise ValueError("invalid slice plan: " + "; ".join(errors))
    return plan


def build_twap(symbol: str, side: Side, quantity: float, start_at: datetime, end_at: datetime,
               slice_count: int = 10) -> SlicePlan:
    """Distribute quantity equally over evenly spaced schedule points."""
    normalized = _validate_common(symbol, side, quantity, start_at, end_at, slice_count)
    return _make_plan(normalized, side, quantity, start_at, end_at, "twap",
                      [1.0] * slice_count, {"slice_count": slice_count})


def build_vwap(symbol: str, side: Side, quantity: float, start_at: datetime, end_at: datetime,
               historical_bucket_volumes: Sequence[float]) -> SlicePlan:
    """Weight slices by an externally supplied volume curve with positive total."""
    count = len(historical_bucket_volumes)
    normalized = _validate_common(symbol, side, quantity, start_at, end_at, count)
    volumes = [float(v) for v in historical_bucket_volumes]
    if any(not isfinite(v) or v < 0 for v in volumes) or sum(volumes) <= 0:
        raise ValueError("historical bucket volumes must be finite, non-negative and sum positive")
    return _make_plan(normalized, side, quantity, start_at, end_at, "vwap", volumes,
                      {"slice_count": count, "volume_sum": sum(volumes)})


def build_participation(symbol: str, side: Side, quantity: float, start_at: datetime,
                        bucket_volumes: Sequence[float], max_participation: float = 0.05,
                        bucket_interval: timedelta = timedelta(minutes=1)) -> SlicePlan:
    """Cap scheduled slice size to a fixed fraction of each bucket's expected volume.

    Raises if the supplied horizon cannot schedule the requested total. This is a
    pre-trade schedule check, not a forecast that the volume will actually print.
    """
    count = len(bucket_volumes)
    normalized = symbol.strip().upper()
    if not normalized or not isinstance(side, Side):
        raise ValueError("symbol and valid side are required")
    if not isfinite(quantity) or quantity <= 0 or start_at.tzinfo is None:
        raise ValueError("quantity must be positive and start_at timezone-aware")
    if not 0 < max_participation <= 1 or bucket_interval.total_seconds() <= 0:
        raise ValueError("invalid participation rate or interval")
    volumes = [float(v) for v in bucket_volumes]
    if not volumes or any(not isfinite(v) or v < 0 for v in volumes):
        raise ValueError("bucket volumes must be a non-empty finite non-negative series")
    capacities = [v * max_participation for v in volumes]
    if sum(capacities) + 1e-9 < quantity:
        raise ValueError("requested quantity exceeds available simulated participation capacity")
    remaining = quantity
    parts: list[float] = []
    for capacity in capacities:
        amount = min(capacity, remaining)
        parts.append(amount)
        remaining -= amount
        if remaining <= 1e-10:
            break
    end_at = start_at + bucket_interval * (len(parts) - 1)
    plan = _make_plan(normalized, side, quantity, start_at, end_at, "participation", parts,
                      {"max_participation": max_participation,
                       "bucket_interval_seconds": bucket_interval.total_seconds(),
                       "used_buckets": len(parts)})
    if any(plan.slices[i].quantity > volumes[i] * max_participation + 1e-8 for i in range(len(parts))):
        raise ValueError("internal participation-cap verification failed")
    return plan


def round_plan_to_lot(plan: SlicePlan, lot_size: int = 1) -> SlicePlan:
    """Round fractional slices down to lots and rebalance the final residual.

    Lot rounding cannot increase the order above the original quantity. A fully
    rounded-away request is rejected rather than outputting zero-sized slices.
    """
    if lot_size < 1:
        raise ValueError("lot_size must be >= 1")
    parts = [int(s.quantity // lot_size) * lot_size for s in plan.slices]
    target = int(plan.total_quantity // lot_size) * lot_size
    delta = target - sum(parts)
    for i in range(len(parts)):
        if delta <= 0:
            break
        parts[i] += min(delta, lot_size)
        delta = target - sum(parts)
    parts = [p for p in parts if p > 0]
    if not parts or target <= 0:
        raise ValueError("quantity is below one lot")
    weights = [float(p) for p in parts]
    return _make_plan(plan.symbol, plan.side, float(target), plan.start_at, plan.end_at,
                      plan.method + "_lot_rounded", weights,
                      {**plan.metadata, "lot_size": lot_size, "original_quantity": plan.total_quantity})


def schedule_summary(plan: SlicePlan) -> dict[str, float | int | str]:
    intervals = [(plan.slices[i].scheduled_at - plan.slices[i - 1].scheduled_at).total_seconds()
                 for i in range(1, len(plan.slices))]
    return {"symbol": plan.symbol, "side": plan.side.value, "method": plan.method,
            "total_quantity": plan.total_quantity, "slice_count": len(plan.slices),
            "largest_slice": max((s.quantity for s in plan.slices), default=0.0),
            "smallest_slice": min((s.quantity for s in plan.slices), default=0.0),
            "average_interval_seconds": sum(intervals) / len(intervals) if intervals else 0.0,
            "valid": not plan.validate()}
