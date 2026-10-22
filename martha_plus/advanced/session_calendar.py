"""Minimal configurable trading-session calendar for deterministic research schedules."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class MarketSession:
    session_date: date
    open_at: datetime
    close_at: datetime
    early_close: bool = False

    def __post_init__(self) -> None:
        if self.open_at.tzinfo is None or self.close_at.tzinfo is None:
            raise ValueError("session times must be timezone-aware")
        if self.close_at <= self.open_at:
            raise ValueError("session close must be after open")
        if self.open_at.date() != self.session_date or self.close_at.date() != self.session_date:
            raise ValueError("session timestamps must be on session_date")


class SessionCalendar:
    """Weekday-based session calendar with caller-supplied holidays and early closes.

    This is intentionally not an exchange-official holiday feed. Keep the holiday
    set versioned and source-reviewed before using this for production scheduling.
    """
    def __init__(self, *, timezone_name: str = "America/New_York", open_time: time = time(9, 30),
                 close_time: time = time(16, 0), holidays: Iterable[date] = (),
                 early_closes: dict[date, time] | None = None, weekmask: frozenset[int] = frozenset({0,1,2,3,4})) -> None:
        try:
            self.tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
        if open_time.tzinfo is not None or close_time.tzinfo is not None:
            raise ValueError("open_time and close_time must be local naive times")
        if close_time <= open_time:
            raise ValueError("close time must be after open time")
        if any(day not in range(7) for day in weekmask):
            raise ValueError("weekmask values must be weekday integers 0..6")
        self.open_time = open_time
        self.close_time = close_time
        self.holidays = frozenset(holidays)
        self.early_closes = dict(early_closes or {})
        self.weekmask = weekmask
        if any(t <= self.open_time or t > self.close_time for t in self.early_closes.values()):
            raise ValueError("early close times must be after open and no later than regular close")

    def is_session(self, day: date) -> bool:
        return day.weekday() in self.weekmask and day not in self.holidays

    def session(self, day: date) -> MarketSession | None:
        if not self.is_session(day):
            return None
        close_time = self.early_closes.get(day, self.close_time)
        return MarketSession(day, datetime.combine(day, self.open_time, self.tz),
                             datetime.combine(day, close_time, self.tz), day in self.early_closes)

    def sessions(self, start: date, end: date) -> list[MarketSession]:
        if end < start:
            raise ValueError("end must not precede start")
        result = []
        current = start
        while current <= end:
            item = self.session(current)
            if item is not None:
                result.append(item)
            current += timedelta(days=1)
        return result

    def next_session(self, day: date, *, include_day: bool = False, max_search_days: int = 370) -> MarketSession:
        start = day if include_day else day + timedelta(days=1)
        for offset in range(max_search_days):
            item = self.session(start + timedelta(days=offset))
            if item:
                return item
        raise RuntimeError("no session found within max_search_days")

    def previous_session(self, day: date, *, include_day: bool = False, max_search_days: int = 370) -> MarketSession:
        start = day if include_day else day - timedelta(days=1)
        for offset in range(max_search_days):
            item = self.session(start - timedelta(days=offset))
            if item:
                return item
        raise RuntimeError("no session found within max_search_days")

    def is_open(self, timestamp: datetime) -> bool:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        local = timestamp.astimezone(self.tz)
        session = self.session(local.date())
        return bool(session and session.open_at <= local < session.close_at)

    def session_progress(self, timestamp: datetime) -> dict[str, float | bool | str | None]:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        local = timestamp.astimezone(self.tz)
        session = self.session(local.date())
        if session is None:
            return {"is_session": False, "is_open": False, "fraction_complete": 0.0,
                    "session_date": local.date().isoformat(), "open_at": None, "close_at": None}
        total = (session.close_at - session.open_at).total_seconds()
        elapsed = max(0.0, min(total, (local - session.open_at).total_seconds()))
        return {"is_session": True, "is_open": session.open_at <= local < session.close_at,
                "fraction_complete": elapsed / total, "session_date": session.session_date.isoformat(),
                "open_at": session.open_at.isoformat(), "close_at": session.close_at.isoformat(),
                "early_close": session.early_close}

    def bar_timestamps(self, day: date, interval: timedelta, *, include_close: bool = False) -> list[datetime]:
        if interval.total_seconds() <= 0:
            raise ValueError("interval must be positive")
        session = self.session(day)
        if session is None:
            return []
        values = []
        current = session.open_at
        while current < session.close_at or (include_close and current == session.close_at):
            values.append(current)
            current += interval
        return values
