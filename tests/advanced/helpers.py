from datetime import datetime, timedelta, timezone
from martha_plus.types import Bar, Quote, Side

NOW = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)

def make_bars(symbol="AAA", closes=None, *, start=NOW, interval=timedelta(days=1)):
    closes = closes or [100 + i for i in range(80)]
    rows = []
    for i, close in enumerate(closes):
        rows.append(Bar(symbol, start + interval * i, close * 0.99, close * 1.01,
                        close * 0.98, close, 1_000 + 10 * i))
    return rows

def make_quote(symbol="AAA", *, at=NOW, bid=99.9, ask=100.1, bid_size=500, ask_size=700):
    return Quote(symbol, bid, ask, at, bid_size, ask_size)
