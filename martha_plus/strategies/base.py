"""Common strategy interface and shared validation helpers."""
from __future__ import annotations
from typing import Protocol, Sequence
from ..types import Bar, Signal

class Strategy(Protocol):
    strategy_id: str
    def generate(self, symbol: str, bars: Sequence[Bar]) -> Signal: ...


def require_bars(bars: Sequence[Bar], minimum: int, strategy_id: str) -> None:
    if len(bars) < minimum:
        raise ValueError(f"{strategy_id} requires at least {minimum} bars; got {len(bars)}")
    symbols = {bar.symbol for bar in bars}
    if len(symbols) != 1:
        raise ValueError("strategy bars must contain one symbol")
    if any(a.timestamp >= b.timestamp for a, b in zip(bars, bars[1:])):
        raise ValueError("strategy bars must be strictly time-ordered")
