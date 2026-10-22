"""Execution-quality diagnostics for paper fills and offline trade records."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from statistics import mean
from typing import Mapping, Sequence
from ..types import Side


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    order_id: str
    symbol: str
    side: Side
    requested_quantity: float
    filled_quantity: float
    decision_price: float
    arrival_price: float
    fill_price: float | None
    fees: float = 0.0
    timestamp: datetime | None = None
    benchmark_price: float | None = None

    def __post_init__(self) -> None:
        if not self.order_id.strip() or not self.symbol.strip():
            raise ValueError("order_id and symbol are required")
        if not isinstance(self.side, Side):
            raise ValueError("side must be a Side enum")
        for name in ("requested_quantity", "filled_quantity", "decision_price", "arrival_price", "fees"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0 or (name.endswith("price") and value <= 0):
                raise ValueError(f"invalid {name}")
        if self.filled_quantity > self.requested_quantity + 1e-8:
            raise ValueError("filled quantity cannot exceed requested quantity")
        if self.fill_price is not None and (not isfinite(self.fill_price) or self.fill_price <= 0):
            raise ValueError("fill_price must be positive finite when provided")
        if self.benchmark_price is not None and (not isfinite(self.benchmark_price) or self.benchmark_price <= 0):
            raise ValueError("benchmark_price must be positive finite when provided")


@dataclass(frozen=True, slots=True)
class ExecutionQualityReport:
    orders: int
    filled_orders: int
    fill_rate: float
    quantity_fill_rate: float
    average_slippage_bps: float
    weighted_slippage_bps: float
    implementation_shortfall: float
    fees: float
    average_latency_seconds: float | None
    by_symbol: Mapping[str, Mapping[str, float | int]]


def slippage_bps(observation: ExecutionObservation, benchmark: str = "arrival") -> float | None:
    if observation.filled_quantity <= 0 or observation.fill_price is None:
        return None
    if benchmark == "arrival":
        reference = observation.arrival_price
    elif benchmark == "decision":
        reference = observation.decision_price
    elif benchmark == "custom":
        reference = observation.benchmark_price
        if reference is None:
            return None
    else:
        raise ValueError("benchmark must be arrival, decision, or custom")
    signed = (observation.fill_price - reference) * observation.side.sign
    return signed / reference * 10_000


def implementation_shortfall(observation: ExecutionObservation) -> float:
    """Positive values mean worse than decision price; includes recorded fees."""
    if observation.filled_quantity <= 0 or observation.fill_price is None:
        return observation.fees
    price_cost = (observation.fill_price - observation.decision_price) * observation.side.sign
    return price_cost * observation.filled_quantity + observation.fees


def analyze_execution(observations: Sequence[ExecutionObservation]) -> ExecutionQualityReport:
    if not observations:
        return ExecutionQualityReport(0, 0, 0, 0, 0, 0, 0, 0, None, {})
    for obs in observations:
        if not isinstance(obs, ExecutionObservation):
            raise TypeError("all observations must be ExecutionObservation instances")
    filled = [o for o in observations if o.filled_quantity > 0 and o.fill_price is not None]
    rates = [slippage_bps(o) for o in observations]
    valid_rates = [x for x in rates if x is not None]
    total_requested = sum(o.requested_quantity for o in observations)
    total_filled = sum(o.filled_quantity for o in observations)
    cost = sum(implementation_shortfall(o) for o in observations)
    weighted_slippage = sum((slippage_bps(o) or 0.0) * o.filled_quantity for o in observations) / total_filled if total_filled else 0.0
    latencies = []
    for observation in observations:
        if observation.timestamp is not None:
            if observation.timestamp.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")
            # Latency is not inferable without a separate submission timestamp.
    by_symbol: dict[str, dict[str, float | int]] = {}
    for symbol in sorted({o.symbol.strip().upper() for o in observations}):
        rows = [o for o in observations if o.symbol.strip().upper() == symbol]
        sym_rates = [slippage_bps(o) for o in rows]
        sym_rates = [v for v in sym_rates if v is not None]
        requested = sum(o.requested_quantity for o in rows)
        sym_filled = sum(o.filled_quantity for o in rows)
        by_symbol[symbol] = {"orders": len(rows), "filled_orders": sum(o.filled_quantity > 0 for o in rows),
                             "quantity_fill_rate": sym_filled / requested if requested else 0.0,
                             "average_slippage_bps": mean(sym_rates) if sym_rates else 0.0,
                             "fees": sum(o.fees for o in rows),
                             "implementation_shortfall": sum(implementation_shortfall(o) for o in rows)}
    return ExecutionQualityReport(
        len(observations), len(filled), len(filled) / len(observations),
        total_filled / total_requested if total_requested else 0.0,
        mean(valid_rates) if valid_rates else 0.0, weighted_slippage, cost,
        sum(o.fees for o in observations), mean(latencies) if latencies else None, by_symbol)


def compare_benchmarks(observation: ExecutionObservation) -> dict[str, float | None]:
    return {"decision_bps": slippage_bps(observation, "decision"),
            "arrival_bps": slippage_bps(observation, "arrival"),
            "custom_bps": slippage_bps(observation, "custom"),
            "implementation_shortfall": implementation_shortfall(observation)}


def aggregate_daily(observations: Sequence[ExecutionObservation]) -> dict[str, dict[str, float | int]]:
    """Group by UTC calendar date using caller-supplied event timestamps."""
    grouped: dict[str, list[ExecutionObservation]] = {}
    for obs in observations:
        if obs.timestamp is None:
            continue
        if obs.timestamp.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        key = obs.timestamp.astimezone(timezone.utc).date().isoformat()
        grouped.setdefault(key, []).append(obs)
    return {day: {"orders": len(rows), "fills": sum(o.filled_quantity > 0 for o in rows),
                  "fees": sum(o.fees for o in rows),
                  "shortfall": sum(implementation_shortfall(o) for o in rows),
                  "quantity_fill_rate": sum(o.filled_quantity for o in rows) / max(1e-12, sum(o.requested_quantity for o in rows))}
            for day, rows in sorted(grouped.items())}
