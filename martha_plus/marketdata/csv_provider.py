"""CSV OHLCV reader. Expected columns: timestamp,symbol,open,high,low,close,volume."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from ..errors import DataValidationError
from ..types import Bar, Quote
from .base import InMemoryMarketData

class CSVBarProvider(InMemoryMarketData):
    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        grouped: dict[str, list[Bar]] = {}
        try:
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"timestamp", "symbol", "open", "high", "low", "close"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise DataValidationError(f"CSV missing columns: {', '.join(sorted(missing))}")
                for row_number, row in enumerate(reader, start=2):
                    try:
                        stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                        if stamp.tzinfo is None:
                            stamp = stamp.replace(tzinfo=timezone.utc)
                        bar = Bar(symbol=row["symbol"], timestamp=stamp,
                                  open=float(row["open"]), high=float(row["high"]),
                                  low=float(row["low"]), close=float(row["close"]),
                                  volume=float(row.get("volume") or 0.0))
                    except Exception as exc:
                        raise DataValidationError(f"invalid CSV row {row_number}: {exc}") from exc
                    grouped.setdefault(bar.symbol, []).append(bar)
        except UnicodeDecodeError as exc:
            raise DataValidationError("CSV must be UTF-8 encoded") from exc
        for symbol, bars in grouped.items():
            self.put_bars(symbol, bars)
            if bars:
                last = bars[-1]
                # The local CSV reader synthesizes a zero-spread close quote. Clearly
                # label it as simulated: it is not executable market data.
                self.put_quote(Quote(symbol, last.close, last.close, last.timestamp,
                                     venue="CSV_CLOSE_PROXY"))

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._bars))
