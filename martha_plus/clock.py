"""Clock abstractions make time-based logic deterministic in tests."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Protocol

class Clock(Protocol):
    def now(self) -> datetime: ...

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

class FrozenClock:
    def __init__(self, value: datetime):
        if value.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime")
        self._value = value.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._value

    def advance_seconds(self, seconds: float) -> None:
        from datetime import timedelta
        self._value += timedelta(seconds=seconds)
