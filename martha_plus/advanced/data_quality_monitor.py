"""Rolling market-data anomaly detection with actionable, explainable findings."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, log
from statistics import median
from typing import Mapping, Sequence
from ..types import Bar, Quote


@dataclass(frozen=True, slots=True)
class QualityFinding:
    code: str
    severity: str
    message: str
    symbol: str | None = None
    timestamp: str | None = None
    value: float | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class QualitySnapshot:
    symbol: str
    observation_count: int
    checked_at: str
    findings: tuple[QualityFinding, ...]
    scores: Mapping[str, float]

    @property
    def healthy(self) -> bool:
        return not any(f.severity == "critical" for f in self.findings)

    @property
    def score(self) -> float:
        return max(0.0, 1.0 - sum({"info": 0.01, "warning": 0.05, "error": 0.15, "critical": 0.5}.get(f.severity, 0.05)
                                  for f in self.findings))


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    max_abs_return: float = 0.35
    max_gap_return: float = 0.25
    zero_volume_window: int = 5
    price_jump_mad_multiple: float = 12.0
    spread_warning_bps: float = 50.0
    spread_critical_bps: float = 200.0
    stale_after_seconds: float = 120.0
    min_history: int = 10

    def __post_init__(self) -> None:
        if any(not isfinite(x) or x <= 0 for x in (self.max_abs_return, self.max_gap_return,
                self.price_jump_mad_multiple, self.spread_warning_bps, self.spread_critical_bps,
                self.stale_after_seconds)):
            raise ValueError("quality thresholds must be positive and finite")
        if self.zero_volume_window < 1 or self.min_history < 2:
            raise ValueError("quality window sizes are invalid")
        if self.spread_critical_bps < self.spread_warning_bps:
            raise ValueError("critical spread threshold must be >= warning threshold")


class DataQualityMonitor:
    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def analyze_bars(self, bars: Sequence[Bar], *, expected_interval_seconds: float | None = None,
                     now: datetime | None = None) -> QualitySnapshot:
        if not bars:
            return QualitySnapshot("UNKNOWN", 0, (now or datetime.now(timezone.utc)).isoformat(),
                                   (QualityFinding("no_data", "critical", "No bars supplied."),), {})
        symbol = bars[0].symbol
        findings: list[QualityFinding] = []
        if any(b.symbol != symbol for b in bars):
            raise ValueError("bars must contain one symbol")
        times = [b.timestamp for b in bars]
        if times != sorted(times):
            findings.append(QualityFinding("out_of_order", "critical", "Bars are not chronological.", symbol))
        if len(times) != len(set(times)):
            findings.append(QualityFinding("duplicate_timestamp", "critical", "Duplicate timestamps found.", symbol))
        returns: list[float] = []
        for i in range(1, len(bars)):
            prior, bar = bars[i - 1], bars[i]
            ret = bar.close / prior.close - 1
            returns.append(ret)
            if abs(ret) > self.policy.max_abs_return:
                findings.append(QualityFinding("extreme_return", "error", "Close-to-close return exceeds configured bound.",
                                               symbol, bar.timestamp.isoformat(), ret, self.policy.max_abs_return))
            gap = bar.open / prior.close - 1
            if abs(gap) > self.policy.max_gap_return:
                findings.append(QualityFinding("extreme_gap", "warning", "Open gap exceeds configured bound.",
                                               symbol, bar.timestamp.isoformat(), gap, self.policy.max_gap_return))
            if expected_interval_seconds is not None:
                interval = (bar.timestamp - prior.timestamp).total_seconds()
                if interval > expected_interval_seconds * 1.5:
                    findings.append(QualityFinding("bar_gap", "warning", "Time gap exceeds expected interval.",
                                                   symbol, bar.timestamp.isoformat(), interval, expected_interval_seconds))
                elif interval <= 0:
                    findings.append(QualityFinding("invalid_interval", "critical", "Bar interval is non-positive.",
                                                   symbol, bar.timestamp.isoformat(), interval, 0.0))
        zero_tail = bars[-self.policy.zero_volume_window:]
        if len(zero_tail) == self.policy.zero_volume_window and all(b.volume == 0 for b in zero_tail):
            findings.append(QualityFinding("zero_volume_run", "warning", "Consecutive bars have zero volume.", symbol,
                                           bars[-1].timestamp.isoformat(), float(len(zero_tail)),
                                           float(self.policy.zero_volume_window)))
        if len(returns) >= self.policy.min_history:
            center = median(returns)
            mad = median([abs(r - center) for r in returns])
            scale = max(mad * 1.4826, 1e-12)
            for bar, ret in zip(bars[1:], returns):
                z = abs(ret - center) / scale
                if z > self.policy.price_jump_mad_multiple:
                    findings.append(QualityFinding("robust_return_outlier", "warning",
                                                   "Return is a robust-MAD outlier; verify news and data adjustments.",
                                                   symbol, bar.timestamp.isoformat(), z,
                                                   self.policy.price_jump_mad_multiple))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or bars[-1].timestamp.tzinfo is None:
            raise ValueError("quality monitoring timestamps must be timezone-aware")
        age = (current - bars[-1].timestamp).total_seconds()
        if age > self.policy.stale_after_seconds:
            findings.append(QualityFinding("stale_bars", "error", "Latest bar is older than configured freshness window.",
                                           symbol, bars[-1].timestamp.isoformat(), age,
                                           self.policy.stale_after_seconds))
        scores = {"return_outlier_rate": sum(abs(r) > self.policy.max_abs_return for r in returns) / max(1, len(returns)),
                  "zero_volume_fraction": sum(b.volume == 0 for b in bars) / len(bars),
                  "duplicate_timestamp_fraction": 1 - len(set(times)) / len(times)}
        return QualitySnapshot(symbol, len(bars), current.isoformat(), tuple(findings), scores)

    def analyze_quote(self, quote: Quote, *, now: datetime | None = None) -> tuple[QualityFinding, ...]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        findings: list[QualityFinding] = []
        if quote.spread_bps > self.policy.spread_critical_bps:
            findings.append(QualityFinding("wide_spread", "critical", "Quote spread above critical threshold.",
                                           quote.symbol, quote.timestamp.isoformat(), quote.spread_bps,
                                           self.policy.spread_critical_bps))
        elif quote.spread_bps > self.policy.spread_warning_bps:
            findings.append(QualityFinding("wide_spread", "warning", "Quote spread above warning threshold.",
                                           quote.symbol, quote.timestamp.isoformat(), quote.spread_bps,
                                           self.policy.spread_warning_bps))
        age = (current - quote.timestamp).total_seconds()
        if age > self.policy.stale_after_seconds:
            findings.append(QualityFinding("stale_quote", "error", "Quote age exceeds freshness threshold.",
                                           quote.symbol, quote.timestamp.isoformat(), age,
                                           self.policy.stale_after_seconds))
        if quote.bid_size == 0 and quote.ask_size == 0:
            findings.append(QualityFinding("empty_book_sizes", "info", "Bid and ask sizes are both zero.",
                                           quote.symbol, quote.timestamp.isoformat()))
        return tuple(findings)

    def compare_snapshots(self, left: QualitySnapshot, right: QualitySnapshot) -> dict[str, object]:
        if left.symbol != right.symbol:
            raise ValueError("cannot compare snapshots for different symbols")
        left_codes = {f.code for f in left.findings}
        right_codes = {f.code for f in right.findings}
        return {"symbol": left.symbol, "score_delta": right.score - left.score,
                "new_findings": sorted(right_codes - left_codes),
                "resolved_findings": sorted(left_codes - right_codes),
                "persistent_findings": sorted(left_codes & right_codes)}


def return_distribution_summary(bars: Sequence[Bar]) -> dict[str, float | int]:
    returns = [bars[i].close / bars[i - 1].close - 1 for i in range(1, len(bars))]
    if any(not isfinite(x) for x in returns):
        raise ValueError("returns contain non-finite values")
    values = sorted(returns)
    def quantile(q: float) -> float:
        if not values: return 0.0
        pos = (len(values) - 1) * q
        lo = int(pos); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"observations": len(values), "mean": sum(values) / len(values) if values else 0.0,
            "median": median(values) if values else 0.0, "min": min(values, default=0.0),
            "max": max(values, default=0.0), "p01": quantile(.01), "p05": quantile(.05),
            "p95": quantile(.95), "p99": quantile(.99),
            "positive_fraction": sum(v > 0 for v in values) / len(values) if values else 0.0}
