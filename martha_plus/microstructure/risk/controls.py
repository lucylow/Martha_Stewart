"""Pre-trade limits and a filesystem kill switch for paper simulations."""
from __future__ import annotations

import os
import tempfile
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Deque

from ..types import OrderRequest, OrderType, Side, SymbolSpec


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_order_quantity: int = 1_000
    max_order_notional: Decimal = Decimal("100000")
    max_position_quantity: int = 5_000
    max_open_orders: int = 100
    max_messages_per_second: int = 50
    max_daily_loss: Decimal = Decimal("5000")
    max_spread_ticks: int = 100
    allow_market_orders: bool = True

    def __post_init__(self) -> None:
        integers = (self.max_order_quantity, self.max_position_quantity, self.max_open_orders,
                    self.max_messages_per_second, self.max_spread_ticks)
        if any(value <= 0 for value in integers):
            raise ValueError("risk limits must be positive")
        if self.max_order_notional <= 0 or self.max_daily_loss <= 0:
            raise ValueError("currency limits must be positive")


@dataclass(frozen=True, slots=True)
class RiskContext:
    ts_ns: int
    position_quantity: int = 0
    open_order_count: int = 0
    realized_pnl: Decimal = Decimal("0")
    reference_price_ticks: int | None = None
    spread_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    codes: tuple[str, ...] = ()
    checked_at_ns: int = 0

    @property
    def reason(self) -> str:
        return ",".join(self.codes) if self.codes else "approved"


class FileKillSwitch:
    """Atomic marker-file kill switch. The caller must check it on each action."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def active(self) -> bool:
        return self.path.exists()

    def activate(self, reason: str, ts_ns: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"timestamp_ns={ts_ns}\nreason={reason.replace(chr(10), ' ')[:300]}\n"
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def deactivate(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class RiskController:
    """Fail-closed order gate with a rolling message-rate limit."""

    def __init__(self, spec: SymbolSpec, limits: RiskLimits | None = None, kill_switch: FileKillSwitch | None = None) -> None:
        self.spec = spec
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch
        self._message_times: Deque[int] = deque()
        self.rejection_counts: dict[str, int] = {}

    def evaluate(self, request: OrderRequest, context: RiskContext) -> RiskDecision:
        codes: list[str] = []
        now = context.ts_ns
        while self._message_times and self._message_times[0] <= now - 1_000_000_000:
            self._message_times.popleft()
        if self.kill_switch is not None and self.kill_switch.active:
            codes.append("kill_switch_active")
        self._message_times.append(now)
        if len(self._message_times) > self.limits.max_messages_per_second:
            codes.append("message_rate_limit")
        if request.symbol != self.spec.symbol:
            codes.append("symbol_mismatch")
        if request.quantity > self.limits.max_order_quantity:
            codes.append("max_order_quantity")
        if request.order_type is OrderType.MARKET and not self.limits.allow_market_orders:
            codes.append("market_orders_disabled")
        reference = request.limit_price_ticks
        if reference is None:
            reference = request.max_price_ticks if request.side is Side.BUY else request.min_price_ticks
        if reference is not None and reference > 0:
            notional = self.spec.money(reference, request.quantity)
            if notional > self.limits.max_order_notional:
                codes.append("max_order_notional")
        signed_position = context.position_quantity + request.side.sign * request.quantity
        if abs(signed_position) > self.limits.max_position_quantity:
            codes.append("max_position_quantity")
        if context.open_order_count >= self.limits.max_open_orders:
            codes.append("max_open_orders")
        if context.realized_pnl <= -self.limits.max_daily_loss:
            codes.append("daily_loss_limit")
        if context.spread_ticks is not None and context.spread_ticks > self.limits.max_spread_ticks:
            codes.append("spread_too_wide")
        for code in codes:
            self.rejection_counts[code] = self.rejection_counts.get(code, 0) + 1
        return RiskDecision(not codes, tuple(codes), now)

    def summary(self) -> dict[str, object]:
        return {
            "message_count_in_window": len(self._message_times),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "kill_switch_active": bool(self.kill_switch and self.kill_switch.active),
        }
