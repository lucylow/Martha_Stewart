"""Deterministic parent-order slicing schedules for paper execution research."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Iterable, Sequence

from ..types import Side


class ScheduleKind(str, Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"


@dataclass(frozen=True, slots=True)
class VolumeBucket:
    start_ns: int
    end_ns: int
    forecast_volume: int

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns or self.forecast_volume < 0:
            raise ValueError("invalid volume bucket")


@dataclass(frozen=True, slots=True)
class ChildOrderPlan:
    parent_id: str
    child_index: int
    scheduled_ns: int
    side: Side
    quantity: int
    cumulative_quantity: int
    remaining_quantity: int
    participation_rate: Decimal
    rationale: str


@dataclass(frozen=True, slots=True)
class ParentOrder:
    parent_id: str
    symbol: str
    side: Side
    quantity: int
    start_ns: int
    end_ns: int
    schedule: ScheduleKind
    participation_rate: Decimal = Decimal("0.10")
    min_child_quantity: int = 1
    max_child_quantity: int | None = None

    def __post_init__(self) -> None:
        if not self.parent_id.strip() or not self.symbol.strip():
            raise ValueError("parent id and symbol must not be blank")
        if self.quantity <= 0 or self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("invalid parent quantity or time interval")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must lie in (0, 1]")
        if self.min_child_quantity <= 0:
            raise ValueError("min_child_quantity must be positive")
        if self.max_child_quantity is not None and self.max_child_quantity < self.min_child_quantity:
            raise ValueError("max_child_quantity must be >= min_child_quantity")


class ScheduleBuilder:
    """Build integer-quantity child plans while conserving parent quantity."""

    def _allocate(self, total: int, weights: Sequence[Decimal]) -> list[int]:
        if total <= 0 or not weights or any(x < 0 for x in weights):
            raise ValueError("invalid allocation inputs")
        weight_sum = sum(weights, Decimal(0))
        if weight_sum <= 0:
            weights = [Decimal(1)] * len(weights)
            weight_sum = Decimal(len(weights))
        raw = [Decimal(total) * x / weight_sum for x in weights]
        base = [int(x) for x in raw]
        remainder = total - sum(base)
        order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
        for index in order[:remainder]:
            base[index] += 1
        return base

    def _weights(self, parent: ParentOrder, buckets: Sequence[VolumeBucket]) -> list[Decimal]:
        n = len(buckets)
        if parent.schedule is ScheduleKind.TWAP:
            return [Decimal(1)] * n
        if parent.schedule is ScheduleKind.VWAP:
            return [Decimal(max(0, x.forecast_volume)) for x in buckets]
        if parent.schedule is ScheduleKind.POV:
            # Equal time spacing, with the plan quantity capped by the target
            # participation against forecast volume in each interval.
            return [Decimal(max(1, x.forecast_volume)) for x in buckets]
        if parent.schedule is ScheduleKind.FRONT_LOADED:
            return [Decimal(n - i) for i in range(n)]
        if parent.schedule is ScheduleKind.BACK_LOADED:
            return [Decimal(i + 1) for i in range(n)]
        raise ValueError(f"unsupported schedule: {parent.schedule}")

    def build(self, parent: ParentOrder, buckets: Sequence[VolumeBucket]) -> tuple[ChildOrderPlan, ...]:
        rows = tuple(buckets)
        if not rows:
            raise ValueError("at least one schedule bucket is required")
        if rows[0].start_ns < parent.start_ns or rows[-1].end_ns > parent.end_ns:
            raise ValueError("volume buckets must lie inside the parent time interval")
        previous = parent.start_ns
        for row in rows:
            if row.start_ns < previous:
                raise ValueError("volume buckets must be ordered and non-overlapping")
            previous = row.end_ns
        quantities = self._allocate(parent.quantity, self._weights(parent, rows))
        plans: list[ChildOrderPlan] = []
        cumulative = 0
        for index, (bucket, qty) in enumerate(zip(rows, quantities)):
            if parent.schedule is ScheduleKind.POV:
                cap = int(Decimal(bucket.forecast_volume) * parent.participation_rate)
                qty = min(qty, cap)
            if qty == 0:
                continue
            if parent.max_child_quantity is not None:
                chunks = [min(parent.max_child_quantity, qty)]
                while sum(chunks) < qty:
                    chunks.append(min(parent.max_child_quantity, qty - sum(chunks)))
            else:
                chunks = [qty]
            for chunk in chunks:
                # Preserve exact parent quantity even when the final residual
                # is below min_child_quantity; the minimum is a target, not a
                # reason to silently discard a scheduled share/lot.
                cumulative += chunk
                plans.append(ChildOrderPlan(parent.parent_id, len(plans), bucket.start_ns,
                                            parent.side, chunk, cumulative,
                                            max(0, parent.quantity - cumulative),
                                            parent.participation_rate,
                                            f"{parent.schedule.value} allocation"))
        # POV can intentionally leave a residual; all other schedules must
        # conserve the requested quantity exactly.
        if parent.schedule is not ScheduleKind.POV and cumulative != parent.quantity:
            raise AssertionError("schedule allocation did not conserve parent quantity")
        if parent.schedule is ScheduleKind.POV and cumulative > parent.quantity:
            raise AssertionError("POV allocation exceeded parent quantity")
        return tuple(plans)


class AdaptivePOVController:
    """Adjust a target rate using only a bounded, caller-supplied urgency score."""

    def __init__(self, base_rate: Decimal = Decimal("0.10"),
                 minimum: Decimal = Decimal("0.02"), maximum: Decimal = Decimal("0.25"),
                 gain: Decimal = Decimal("0.05")) -> None:
        if not 0 < minimum <= base_rate <= maximum <= 1 or gain < 0:
            raise ValueError("invalid adaptive POV parameters")
        self.base_rate = base_rate
        self.minimum = minimum
        self.maximum = maximum
        self.gain = gain
        self.last_rate = base_rate

    def update(self, urgency: Decimal | float) -> Decimal:
        value = Decimal(str(urgency))
        if not -1 <= value <= 1:
            raise ValueError("urgency must lie in [-1, 1]")
        rate = self.base_rate + self.gain * value
        self.last_rate = min(self.maximum, max(self.minimum, rate))
        return self.last_rate


@dataclass(slots=True)
class ParentOrderTracker:
    """Track fills, cancellations, and child IDs for one parent order."""
    parent: ParentOrder
    filled_quantity: int = 0
    cancelled_quantity: int = 0
    child_ids: list[str] | None = None
    fees_paid: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.child_ids is None:
            self.child_ids = []

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.parent.quantity - self.filled_quantity - self.cancelled_quantity)

    @property
    def fill_ratio(self) -> Decimal:
        return Decimal(self.filled_quantity) / Decimal(self.parent.quantity)

    def add_child(self, child_id: str) -> None:
        if not child_id or child_id in (self.child_ids or []):
            raise ValueError("child id must be unique and non-empty")
        assert self.child_ids is not None
        self.child_ids.append(child_id)

    def apply_fill(self, quantity: int, fee: Decimal = Decimal(0)) -> None:
        if quantity <= 0 or quantity > self.remaining_quantity or fee < 0:
            raise ValueError("invalid parent fill")
        self.filled_quantity += quantity
        self.fees_paid += fee

    def cancel(self, quantity: int) -> None:
        if quantity <= 0 or quantity > self.remaining_quantity:
            raise ValueError("invalid parent cancellation")
        self.cancelled_quantity += quantity

    def validate(self) -> None:
        if self.filled_quantity + self.cancelled_quantity > self.parent.quantity:
            raise AssertionError("parent order quantity conservation failed")


def evenly_spaced_buckets(start_ns: int, end_ns: int, count: int,
                          forecast_volume: int = 0) -> tuple[VolumeBucket, ...]:
    if start_ns < 0 or end_ns <= start_ns or count <= 0 or forecast_volume < 0:
        raise ValueError("invalid bucket parameters")
    width = end_ns - start_ns
    result = []
    for index in range(count):
        left = start_ns + width * index // count
        right = start_ns + width * (index + 1) // count
        result.append(VolumeBucket(left, right, forecast_volume // count + (1 if index < forecast_volume % count else 0)))
    return tuple(result)
