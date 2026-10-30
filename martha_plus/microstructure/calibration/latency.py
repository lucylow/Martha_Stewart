"""Empirical latency summaries and deterministic fitted profiles."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean, median, pstdev
from typing import Iterable

from ..clock import LatencyProfile


@dataclass(frozen=True, slots=True)
class LatencyFit:
    sample_count: int
    mean_ns: Decimal
    median_ns: int
    p50_ns: int
    p90_ns: int
    p95_ns: int
    p99_ns: int
    standard_deviation_ns: Decimal
    minimum_ns: int
    maximum_ns: int

    def profile(self, *, ceiling_multiplier: Decimal = Decimal("3")) -> LatencyProfile:
        base = int(self.median_ns)
        jitter = max(0, int(self.standard_deviation_ns))
        ceiling = max(base, int(Decimal(self.p99_ns) * ceiling_multiplier))
        return LatencyProfile(base_ns=base, jitter_ns=jitter, floor_ns=self.minimum_ns, ceiling_ns=ceiling)


def fit_latency(samples_ns: Iterable[int]) -> LatencyFit:
    samples = sorted(int(x) for x in samples_ns)
    if not samples:
        raise ValueError("at least one latency sample is required")
    if any(x < 0 for x in samples):
        raise ValueError("latency samples must be non-negative")

    def quantile(p: float) -> int:
        index = min(len(samples) - 1, max(0, int(round((len(samples) - 1) * p))))
        return samples[index]

    return LatencyFit(
        sample_count=len(samples), mean_ns=Decimal(str(fmean(samples))),
        median_ns=int(median(samples)), p50_ns=quantile(0.50),
        p90_ns=quantile(0.90), p95_ns=quantile(0.95), p99_ns=quantile(0.99),
        standard_deviation_ns=Decimal(str(pstdev(samples))) if len(samples) > 1 else Decimal("0"),
        minimum_ns=samples[0], maximum_ns=samples[-1],
    )
