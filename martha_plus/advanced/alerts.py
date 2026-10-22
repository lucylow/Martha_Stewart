"""Alert deduplication, cooldowns, and notification adapter contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30
    CRITICAL = 40


@dataclass(frozen=True, slots=True)
class Alert:
    title: str
    message: str
    severity: Severity = Severity.WARNING
    key: str = "general"
    tags: Mapping[str, str] = field(default_factory=dict)
    alert_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.message.strip() or not self.key.strip():
            raise ValueError("alert title, message, and key are required")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("alert timestamp must be timezone-aware")


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


@dataclass(frozen=True, slots=True)
class AlertDelivery:
    alert_id: str
    notifier: str
    sent: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    minimum_severity: Severity = Severity.INFO
    cooldown_seconds: float = 300.0
    repeat_after_seconds: float = 3600.0
    deduplicate: bool = True
    max_history: int = 5000

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0 or self.repeat_after_seconds < self.cooldown_seconds:
            raise ValueError("alert repeat time must be >= non-negative cooldown")
        if self.max_history < 1:
            raise ValueError("max_history must be >= 1")


class InMemoryNotifier:
    """Notifier for tests and local development; stores messages in memory only."""
    def __init__(self, name: str = "memory") -> None:
        self.name = name
        self.messages: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.messages.append(alert)


class AlertManager:
    """Emit deduplicated alerts with failure isolation and delivery receipts."""
    def __init__(self, notifiers: list[Notifier] | None = None,
                 policy: AlertPolicy | None = None,
                 monotonic_clock: Callable[[], float] = monotonic) -> None:
        self.notifiers = list(notifiers or [])
        self.policy = policy or AlertPolicy()
        self._clock = monotonic_clock
        self._lock = RLock()
        self._last_sent: dict[str, float] = {}
        self._active_keys: set[str] = set()
        self._history: list[Alert] = []
        self._delivery: list[AlertDelivery] = []

    def emit(self, alert: Alert, *, force: bool = False) -> tuple[AlertDelivery, ...]:
        if alert.severity < self.policy.minimum_severity and not force:
            return ()
        now = self._clock()
        key = f"{alert.key}:{int(alert.severity)}"
        with self._lock:
            last = self._last_sent.get(key)
            if not force and self.policy.deduplicate and key in self._active_keys:
                if last is not None and now - last < self.policy.repeat_after_seconds:
                    return ()
            if not force and last is not None and now - last < self.policy.cooldown_seconds:
                return ()
            self._last_sent[key] = now
            self._active_keys.add(key)
            self._history.append(alert)
            self._history = self._history[-self.policy.max_history:]
        receipts: list[AlertDelivery] = []
        for notifier in self.notifiers:
            name = getattr(notifier, "name", notifier.__class__.__name__)
            try:
                notifier.send(alert)
                receipt = AlertDelivery(alert.alert_id, name, True)
            except Exception as exc:
                receipt = AlertDelivery(alert.alert_id, name, False, f"{type(exc).__name__}: {exc}")
            receipts.append(receipt)
        with self._lock:
            self._delivery.extend(receipts)
            self._delivery = self._delivery[-self.policy.max_history:]
        return tuple(receipts)

    def resolve(self, key: str) -> bool:
        """Clear a dedupe key so a future alert can create a new incident."""
        with self._lock:
            matching = [item for item in self._active_keys if item.startswith(key + ":")]
            for item in matching:
                self._active_keys.remove(item)
            return bool(matching)

    @property
    def history(self) -> tuple[Alert, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def deliveries(self) -> tuple[AlertDelivery, ...]:
        with self._lock:
            return tuple(self._delivery)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"history_count": len(self._history), "active_keys": sorted(self._active_keys),
                    "delivery_count": len(self._delivery),
                    "failed_deliveries": sum(not d.sent for d in self._delivery),
                    "last_alert_at": self._history[-1].created_at.isoformat() if self._history else None}


def drawdown_alert(drawdown_fraction: float, threshold: float, *, portfolio: str = "default") -> Alert | None:
    if drawdown_fraction < 0 or threshold <= 0:
        raise ValueError("drawdown must be non-negative and threshold positive")
    if drawdown_fraction < threshold:
        return None
    severity = Severity.CRITICAL if drawdown_fraction >= threshold * 1.5 else Severity.ERROR
    return Alert("Portfolio drawdown threshold breached",
                 f"Observed drawdown {drawdown_fraction:.2%} is at or above {threshold:.2%}.",
                 severity, f"drawdown:{portfolio}", {"portfolio": portfolio})


def health_alert(component: str, healthy: bool, detail: str = "") -> Alert | None:
    if healthy:
        return None
    return Alert(f"Component unhealthy: {component}", detail or "Health probe failed.",
                 Severity.ERROR, f"health:{component}", {"component": component})
