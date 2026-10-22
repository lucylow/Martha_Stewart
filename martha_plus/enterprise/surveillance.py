"""Explainable heuristic trading surveillance and order behavior flags.

Flags are triage signals only. They do not determine intent or establish a
regulatory violation, and they must not replace human compliance review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from threading import RLock
from typing import Iterable, Mapping, Sequence


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class OrderObservation:
    order_id: str
    account: str
    symbol: str
    side: str
    quantity: float
    price: float
    event: str
    timestamp: datetime
    strategy_id: str = "unknown"
    venue: str = "unknown"
    parent_order_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id.strip() or not self.account.strip() or not self.symbol.strip():
            raise ValueError("order_id, account and symbol are required")
        if not isfinite(self.quantity) or self.quantity < 0 or not isfinite(self.price) or self.price < 0:
            raise ValueError("quantity and price must be finite and non-negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.side.lower() not in {"buy", "sell", "short", "cover"}:
            raise ValueError("unsupported order side")


@dataclass(frozen=True, slots=True)
class SurveillanceFlag:
    flag_id: str
    rule: str
    severity: AlertSeverity
    account: str
    symbol: str
    order_ids: tuple[str, ...]
    detected_at: str
    summary: str
    metrics: Mapping[str, float | int | str]
    disposition: str = "open"
    reviewer: str | None = None
    review_note: str | None = None


class SurveillanceEngine:
    """Evaluate order-event sequences against configurable, explainable heuristics."""
    def __init__(self, *, clock=lambda: datetime.now(timezone.utc),
                 repeated_cancel_window_seconds: float = 30.0,
                 repeated_cancel_threshold: int = 5,
                 quote_to_trade_threshold: float = 10.0,
                 max_single_order_notional: float = 1_000_000.0) -> None:
        if repeated_cancel_window_seconds <= 0 or repeated_cancel_threshold < 2:
            raise ValueError("invalid cancel rule settings")
        if quote_to_trade_threshold <= 0 or max_single_order_notional <= 0:
            raise ValueError("notional thresholds must be positive")
        self.clock = clock
        self.cancel_window = repeated_cancel_window_seconds
        self.cancel_threshold = repeated_cancel_threshold
        self.quote_to_trade_threshold = quote_to_trade_threshold
        self.max_notional = max_single_order_notional
        self._observations: list[OrderObservation] = []
        self._flags: list[SurveillanceFlag] = []
        self._counter = 0
        self._lock = RLock()

    def ingest(self, observation: OrderObservation) -> tuple[SurveillanceFlag, ...]:
        with self._lock:
            if any(o.order_id == observation.order_id and o.event == observation.event
                   and o.timestamp == observation.timestamp for o in self._observations):
                return ()
            prior_count = len(self._flags)
            self._observations.append(observation)
            self._observations.sort(key=lambda row: row.timestamp)
            self._evaluate_locked(observation)
            return tuple(self._flags[prior_count:])

    def _flag(self, rule: str, severity: AlertSeverity, obs: OrderObservation,
              ids: Sequence[str], summary: str, metrics: Mapping[str, float | int | str]) -> None:
        self._counter += 1
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        self._flags.append(SurveillanceFlag(f"surv-{self._counter:08d}", rule, severity,
            obs.account, obs.symbol.upper(), tuple(dict.fromkeys(ids)), now.isoformat(),
            summary, dict(metrics)))

    def _evaluate_locked(self, obs: OrderObservation) -> None:
        notional = obs.quantity * obs.price
        if notional >= self.max_notional and obs.event.lower() in {"submitted", "accepted", "new"}:
            self._flag("large_order_notional", AlertSeverity.MEDIUM, obs, [obs.order_id],
                       "Order notional meets or exceeds configured triage threshold",
                       {"notional": notional, "threshold": self.max_notional})
        relevant = [o for o in self._observations if o.account == obs.account
                    and o.symbol.upper() == obs.symbol.upper()
                    and (obs.timestamp - o.timestamp).total_seconds() <= self.cancel_window
                    and (obs.timestamp - o.timestamp).total_seconds() >= 0]
        cancels = [o for o in relevant if o.event.lower() in {"cancelled", "canceled", "cancel"}]
        if len(cancels) >= self.cancel_threshold:
            latest_ids = [o.order_id for o in cancels[-self.cancel_threshold:]]
            # Avoid emitting the same same-window flag repeatedly.
            existing = [f for f in self._flags if f.rule == "repeated_cancellations"
                        and f.account == obs.account and f.symbol == obs.symbol.upper()
                        and set(latest_ids).issubset(set(f.order_ids))]
            if not existing:
                self._flag("repeated_cancellations", AlertSeverity.MEDIUM, obs, latest_ids,
                    "Repeated cancellations in a short window; review order intent and venue behavior",
                    {"cancel_count": len(cancels), "window_seconds": self.cancel_window})
        submitted = [o for o in relevant if o.event.lower() in {"submitted", "accepted", "new"}]
        fills = [o for o in relevant if o.event.lower() in {"filled", "fill", "partial_fill"}]
        if len(submitted) >= 3 and len(fills) > 0 and len(submitted) / len(fills) >= self.quote_to_trade_threshold:
            existing = [f for f in self._flags if f.rule == "high_order_to_fill_ratio"
                        and f.account == obs.account and f.symbol == obs.symbol.upper()]
            if not existing:
                self._flag("high_order_to_fill_ratio", AlertSeverity.LOW, obs,
                    [o.order_id for o in submitted], "High order-to-fill count ratio; triage only",
                    {"order_events": len(submitted), "fill_events": len(fills),
                     "ratio": len(submitted) / len(fills)})

    @property
    def observations(self) -> tuple[OrderObservation, ...]:
        with self._lock:
            return tuple(self._observations)

    @property
    def flags(self) -> tuple[SurveillanceFlag, ...]:
        with self._lock:
            return tuple(self._flags)

    def review(self, flag_id: str, *, reviewer: str, disposition: str, note: str) -> SurveillanceFlag:
        if not reviewer.strip() or not note.strip():
            raise ValueError("reviewer and note are required")
        if disposition not in {"open", "investigating", "cleared", "escalated", "confirmed"}:
            raise ValueError("unsupported disposition")
        with self._lock:
            for idx, flag in enumerate(self._flags):
                if flag.flag_id == flag_id:
                    updated = SurveillanceFlag(flag.flag_id, flag.rule, flag.severity, flag.account,
                        flag.symbol, flag.order_ids, flag.detected_at, flag.summary, flag.metrics,
                        disposition, reviewer, note)
                    self._flags[idx] = updated
                    return updated
        raise KeyError(flag_id)

    def summary(self) -> dict[str, int]:
        with self._lock:
            flags = tuple(self._flags)
        return {"observations": len(self._observations), "flags": len(flags),
                "open_flags": sum(f.disposition == "open" for f in flags),
                **{severity.value: sum(f.severity is severity for f in flags) for severity in AlertSeverity}}


def zscore_anomalies(values: Sequence[float], *, threshold: float = 3.0) -> tuple[int, ...]:
    """Return indices with population-standard-deviation z-scores above threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if any(not isfinite(v) for v in values):
        raise ValueError("values must be finite")
    if len(values) < 3:
        return ()
    center = mean(values)
    scale = pstdev(values)
    if scale == 0:
        return ()
    return tuple(i for i, value in enumerate(values) if abs(value - center) / scale >= threshold)
