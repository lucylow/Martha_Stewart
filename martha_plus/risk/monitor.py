"""Stateful portfolio guard that can activate a file-based kill switch.

This monitor is intentionally pull-driven: the application must call ``update``
with a fresh equity reading. It does not create background threads.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from threading import RLock
from ..config import Settings
from .kill_switch import KillSwitch

@dataclass(frozen=True, slots=True)
class MonitorState:
    equity: float
    high_water_mark: float
    day_start_equity: float
    daily_return: float
    drawdown_fraction: float
    halted: bool
    reasons: tuple[str, ...]
    timestamp: datetime

class PortfolioRiskMonitor:
    def __init__(self, settings: Settings, kill_switch: KillSwitch):
        self.settings = settings
        self.kill_switch = kill_switch
        self._high_water = settings.starting_cash
        self._day_start = settings.starting_cash
        self._day: date | None = None
        self._last_equity = settings.starting_cash
        self._last_state: MonitorState | None = None
        self._lock = RLock()

    def update(self, equity: float, timestamp: datetime | None = None) -> MonitorState:
        if equity <= 0:
            # A nonpositive equity reading is sufficiently severe to halt immediately.
            self.kill_switch.activate("nonpositive equity reading")
            raise ValueError("equity must be positive; kill switch activated")
        now = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        day = now.date()
        with self._lock:
            if self._day != day:
                self._day = day
                self._day_start = equity
            self._high_water = max(self._high_water, equity)
            daily_pnl = equity - self._day_start
            daily_return = daily_pnl / self._day_start if self._day_start > 0 else 0.0
            drawdown = max(0.0, (self._high_water - equity) / self._high_water)
            reasons: list[str] = []
            if daily_pnl <= -self.settings.max_daily_loss:
                reasons.append("daily_loss_limit")
            if drawdown >= self.settings.max_drawdown_fraction:
                reasons.append("drawdown_limit")
            if reasons:
                self.kill_switch.activate(",".join(reasons))
            state = MonitorState(equity, self._high_water, self._day_start, daily_return,
                                 drawdown, self.kill_switch.active, tuple(reasons), now)
            self._last_equity = equity
            self._last_state = state
            return state

    @property
    def last_state(self) -> MonitorState | None:
        with self._lock:
            return self._last_state

    def reset_high_water_mark(self, equity: float, operator_acknowledged: bool = False) -> None:
        if not operator_acknowledged:
            raise PermissionError("explicit operator_acknowledged=True is required")
        if equity <= 0:
            raise ValueError("equity must be positive")
        with self._lock:
            self._high_water = equity
            self._day_start = equity
            self._last_equity = equity
            self._last_state = None
