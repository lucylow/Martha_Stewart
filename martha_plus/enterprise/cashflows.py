"""Portfolio cash-flow analytics: XIRR, money-weighted and time-weighted returns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class CashFlow:
    when: date | datetime
    amount: float
    label: str = "cashflow"

    def __post_init__(self) -> None:
        if not isfinite(self.amount):
            raise ValueError("cash flow amount must be finite")
        if isinstance(self.when, datetime) and self.when.tzinfo is None:
            raise ValueError("datetime cash flows must be timezone-aware")

    @property
    def day(self) -> date:
        return self.when.date() if isinstance(self.when, datetime) else self.when


def _normalized(cashflows: Iterable[CashFlow]) -> list[CashFlow]:
    rows = sorted(cashflows, key=lambda row: row.day)
    if len(rows) < 2:
        raise ValueError("at least two dated cash flows are required")
    if len({row.day for row in rows}) == 1:
        raise ValueError("cash flows must span more than one date")
    if not any(row.amount < 0 for row in rows) or not any(row.amount > 0 for row in rows):
        raise ValueError("cash flows must include at least one negative and one positive amount")
    return rows


def xnpv(rate: float, cashflows: Iterable[CashFlow]) -> float:
    """Net present value using actual day counts and annual effective rate."""
    if not isfinite(rate) or rate <= -1:
        raise ValueError("rate must be finite and greater than -1")
    rows = _normalized(cashflows)
    origin = rows[0].day
    return sum(row.amount / ((1.0 + rate) ** ((row.day - origin).days / 365.0)) for row in rows)


def xirr(cashflows: Iterable[CashFlow], *, low: float = -0.9999, high: float = 10.0,
         tolerance: float = 1e-9, max_iterations: int = 500) -> float:
    """Find a bracketed XIRR root using bisection; fail rather than guess if unbracketed."""
    rows = _normalized(cashflows)
    if low <= -1 or high <= low or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid root search bounds or iteration settings")
    f_low, f_high = xnpv(low, rows), xnpv(high, rows)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        # Expand the upper bracket, but do not imply every cashflow series has a root.
        candidate = high
        for _ in range(12):
            candidate = candidate * 2 + 1
            f_candidate = xnpv(candidate, rows)
            if f_low * f_candidate <= 0:
                high, f_high = candidate, f_candidate
                break
        else:
            raise ValueError("could not bracket XIRR root; cash flows may have multiple or no roots")
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        value = xnpv(mid, rows)
        if abs(value) <= tolerance or abs(high - low) <= tolerance:
            return mid
        if f_low * value <= 0:
            high, f_high = mid, value
        else:
            low, f_low = mid, value
    raise RuntimeError("XIRR did not converge within max_iterations")


def money_weighted_return(cashflows: Iterable[CashFlow]) -> float:
    return xirr(cashflows)


def time_weighted_return(period_returns: Sequence[float]) -> float:
    """Compound subperiod returns after external cash flows have been neutralized."""
    value = 1.0
    for result in period_returns:
        if not isfinite(result) or result < -1:
            raise ValueError("period returns must be finite and no less than -100%")
        value *= 1.0 + result
    return value - 1.0


def annualize_return(total_return: float, periods: int, periods_per_year: float) -> float:
    if not isfinite(total_return) or total_return < -1:
        raise ValueError("total return must be finite and no less than -100%")
    if periods <= 0 or not isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods and periods_per_year must be positive")
    if total_return == -1:
        return -1.0
    return (1.0 + total_return) ** (periods_per_year / periods) - 1.0


def drawdown_series(equity: Sequence[float]) -> list[float]:
    peak = float("-inf")
    values: list[float] = []
    for amount in equity:
        if not isfinite(amount) or amount < 0:
            raise ValueError("equity values must be finite and non-negative")
        peak = max(peak, amount)
        values.append(0.0 if peak == 0 else amount / peak - 1.0)
    return values


def period_summary(cashflows: Sequence[CashFlow]) -> dict[str, float | int | str]:
    rows = _normalized(cashflows)
    positives = sum(r.amount for r in rows if r.amount > 0)
    negatives = -sum(r.amount for r in rows if r.amount < 0)
    return {"count": len(rows), "start": rows[0].day.isoformat(), "end": rows[-1].day.isoformat(),
            "inflows": positives, "outflows": negatives, "net": positives - negatives,
            "net_flow_ratio": (positives - negatives) / negatives if negatives else 0.0}


def split_cashflows_by_year(cashflows: Iterable[CashFlow]) -> dict[int, tuple[CashFlow, ...]]:
    groups: dict[int, list[CashFlow]] = {}
    for row in cashflows:
        groups.setdefault(row.day.year, []).append(row)
    return {year: tuple(sorted(rows, key=lambda x: x.day)) for year, rows in sorted(groups.items())}
