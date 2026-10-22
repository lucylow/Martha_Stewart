"""Seeded synthetic prices for repeatable research and safe demos."""
from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone
from typing import Sequence
from ..types import Bar, Quote
from .base import InMemoryMarketData

class SyntheticMarketData(InMemoryMarketData):
    def __init__(self, seed: int = 7, start: datetime | None = None):
        super().__init__()
        self.random = random.Random(seed)
        self.start = (start or datetime(2024, 1, 1, tzinfo=timezone.utc)).astimezone(timezone.utc)

    def generate(self, symbol: str, count: int = 500, start_price: float = 100.0,
                 drift: float = 0.0002, volatility: float = 0.012,
                 interval_seconds: int = 86_400) -> Sequence[Bar]:
        if count < 1 or start_price <= 0 or volatility < 0:
            raise ValueError("count/start_price must be positive and volatility non-negative")
        key = symbol.strip().upper()
        price = start_price
        bars: list[Bar] = []
        for index in range(count):
            open_price = price
            shock = self.random.gauss(drift, volatility)
            close = max(0.01, open_price * (1 + shock))
            wick = abs(self.random.gauss(0, volatility / 2))
            high = max(open_price, close) * (1 + wick)
            low = max(0.01, min(open_price, close) * (1 - wick))
            volume = max(1.0, self.random.lognormvariate(10, 0.6))
            stamp = self.start + timedelta(seconds=interval_seconds * index)
            bars.append(Bar(key, stamp, open_price, high, low, close, volume))
            price = close
        self.put_bars(key, bars)
        self.put_quote(Quote(key, price * 0.9995, price * 1.0005, bars[-1].timestamp,
                             venue="SYNTHETIC"))
        return tuple(bars)
