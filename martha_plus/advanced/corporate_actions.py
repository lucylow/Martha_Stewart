"""Explicit corporate-action records and point-in-time split/dividend adjustments."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Iterable, Sequence
from ..types import Bar


@dataclass(frozen=True, slots=True)
class SplitEvent:
    symbol: str
    effective_date: date
    new_shares_per_old_share: float
    source: str = "manual"

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not isfinite(self.new_shares_per_old_share) or self.new_shares_per_old_share <= 0:
            raise ValueError("split requires symbol and positive finite share ratio")


@dataclass(frozen=True, slots=True)
class DividendEvent:
    symbol: str
    ex_date: date
    cash_per_share: float
    currency: str = "USD"
    source: str = "manual"

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not isfinite(self.cash_per_share) or self.cash_per_share < 0:
            raise ValueError("dividend requires symbol and non-negative finite amount")
        if len(self.currency) != 3:
            raise ValueError("currency must be a three-letter code")


@dataclass(frozen=True, slots=True)
class ActionAdjustmentResult:
    bars: tuple[Bar, ...]
    action_count: int
    applied_actions: tuple[str, ...]
    warnings: tuple[str, ...]


def adjust_for_splits(bars: Sequence[Bar], events: Iterable[SplitEvent], *,
                      as_of: date | None = None, mode: str = "backward") -> ActionAdjustmentResult:
    """Adjust historical OHLC prices and volume for split events.

    `backward` modifies pre-effective bars to current share units. `forward` modifies
    bars on/after split events to the original share unit. This helper assumes raw
    unadjusted input; applying it to already adjusted history would double-adjust.
    """
    if mode not in {"backward", "forward"}:
        raise ValueError("mode must be backward or forward")
    if not bars:
        return ActionAdjustmentResult((), 0, (), ())
    symbol = bars[0].symbol
    if any(bar.symbol != symbol for bar in bars):
        raise ValueError("all bars must be for one symbol")
    times = [bar.timestamp for bar in bars]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError("bars must be strictly chronological")
    actions = sorted([e for e in events if e.symbol.strip().upper() == symbol and
                      (as_of is None or e.effective_date <= as_of)], key=lambda e: e.effective_date)
    if len({e.effective_date for e in actions}) != len(actions):
        raise ValueError("multiple split actions on same effective date require manual consolidation")
    result: list[Bar] = []
    applied: list[str] = []
    for bar in bars:
        factor = 1.0
        for action in actions:
            bar_date = bar.timestamp.date()
            if mode == "backward" and bar_date < action.effective_date:
                factor /= action.new_shares_per_old_share
            elif mode == "forward" and bar_date >= action.effective_date:
                factor *= action.new_shares_per_old_share
        if factor != 1.0:
            result.append(Bar(bar.symbol, bar.timestamp, bar.open * factor, bar.high * factor,
                              bar.low * factor, bar.close * factor, bar.volume / factor))
        else:
            result.append(bar)
    applied = [f"split:{a.effective_date.isoformat()}:{a.new_shares_per_old_share:g}" for a in actions]
    return ActionAdjustmentResult(tuple(result), len(actions), tuple(applied), ())


def total_return_series(bars: Sequence[Bar], dividends: Iterable[DividendEvent], *,
                        reinvest: bool = True) -> list[float | None]:
    """Compute simple close-to-close total returns with ex-date dividend cash yield.

    This is a close-based approximation. It assumes each dividend is paid to a
    holder who owned shares at the prior close, and excludes tax and settlement timing.
    """
    if not bars:
        return []
    symbol = bars[0].symbol
    if any(b.symbol != symbol for b in bars):
        raise ValueError("all bars must be for one symbol")
    if [b.timestamp for b in bars] != sorted(b.timestamp for b in bars):
        raise ValueError("bars must be chronological")
    event_map: dict[date, float] = {}
    for event in dividends:
        if event.symbol.strip().upper() != symbol:
            continue
        event_map[event.ex_date] = event_map.get(event.ex_date, 0.0) + event.cash_per_share
    result: list[float | None] = [None]
    for i in range(1, len(bars)):
        previous, current = bars[i - 1], bars[i]
        cash = event_map.get(current.timestamp.date(), 0.0)
        r = (current.close + cash) / previous.close - 1.0
        result.append(r)
    return result


def action_coverage(bars: Sequence[Bar], split_events: Sequence[SplitEvent],
                    dividend_events: Sequence[DividendEvent]) -> dict[str, object]:
    if not bars:
        return {"symbol": None, "start": None, "end": None,
                "split_events": len(split_events), "dividend_events": len(dividend_events),
                "within_history": 0}
    symbol = bars[0].symbol
    start, end = bars[0].timestamp.date(), bars[-1].timestamp.date()
    events = [e.effective_date for e in split_events if e.symbol.strip().upper() == symbol]
    dividends = [e.ex_date for e in dividend_events if e.symbol.strip().upper() == symbol]
    relevant = sum(start <= d <= end for d in events + dividends)
    return {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat(),
            "split_events": len(events), "dividend_events": len(dividends), "within_history": relevant,
            "warning": "No action provider is included; manually verify completeness and prior adjustments."}
