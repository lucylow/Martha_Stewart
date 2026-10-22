"""Data-quality checks for OHLCV feeds before strategy evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from typing import Sequence
from ..types import Bar

@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    code: str
    message: str
    index: int | None = None

@dataclass(frozen=True, slots=True)
class DataQualityReport:
    symbol: str
    bar_count: int
    issues: tuple[DataQualityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def error_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "bar_count": self.bar_count, "ok": self.ok,
                "issues": [{"code": issue.code, "message": issue.message, "index": issue.index}
                           for issue in self.issues]}


def inspect_bars(bars: Sequence[Bar], expected_interval_seconds: float | None = None,
                 max_absolute_return: float | None = 0.50) -> DataQualityReport:
    issues: list[DataQualityIssue] = []
    if expected_interval_seconds is not None and expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be positive")
    if max_absolute_return is not None and max_absolute_return <= 0:
        raise ValueError("max_absolute_return must be positive")
    symbol = bars[0].symbol if bars else ""
    if not bars:
        issues.append(DataQualityIssue("empty_series", "no bars supplied"))
        return DataQualityReport(symbol, 0, tuple(issues))
    seen: set[tuple[str, object]] = set()
    for index, bar in enumerate(bars):
        if bar.symbol != symbol:
            issues.append(DataQualityIssue("mixed_symbols", f"expected {symbol}, got {bar.symbol}", index))
        key = (bar.symbol, bar.timestamp)
        if key in seen:
            issues.append(DataQualityIssue("duplicate_timestamp", "duplicate bar timestamp", index))
        seen.add(key)
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if not all(isfinite(v) for v in values):
            issues.append(DataQualityIssue("non_finite", "bar contains non-finite values", index))
        if bar.low <= 0 or bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
            issues.append(DataQualityIssue("invalid_ohlc", "OHLC bounds are inconsistent", index))
        if bar.volume < 0:
            issues.append(DataQualityIssue("negative_volume", "volume must be non-negative", index))
        if index > 0:
            previous = bars[index-1]
            delta = (bar.timestamp - previous.timestamp).total_seconds()
            if delta <= 0:
                issues.append(DataQualityIssue("time_not_increasing", "timestamps must increase", index))
            elif expected_interval_seconds and abs(delta - expected_interval_seconds) > max(1.0, expected_interval_seconds * 0.05):
                issues.append(DataQualityIssue("interval_gap", f"observed {delta}s interval", index))
            if max_absolute_return is not None and previous.close > 0:
                move = abs(bar.close / previous.close - 1.0)
                if move > max_absolute_return:
                    issues.append(DataQualityIssue("extreme_price_move", f"absolute return {move:.3f} exceeds threshold", index))
    return DataQualityReport(symbol, len(bars), tuple(issues))


def require_clean_bars(bars: Sequence[Bar], **kwargs) -> DataQualityReport:
    report = inspect_bars(bars, **kwargs)
    if not report.ok:
        summary = ", ".join(f"{issue.code}@{issue.index}" for issue in report.issues[:8])
        raise ValueError(f"data quality validation failed: {summary}")
    return report
