"""Discount curves, zero/forward rates and simple curve bootstrapping."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..core import FinanceInputError, Number, finite, positive


@dataclass(frozen=True)
class CurvePoint:
    maturity: float
    discount_factor: float

    def __post_init__(self) -> None:
        positive(self.maturity, "maturity")
        positive(self.discount_factor, "discount_factor")


class DiscountCurve:
    """Log-linear discount curve with flat-zero extrapolation by default."""
    def __init__(self, points: Sequence[CurvePoint], *, extrapolation: str = "flat_forward"):
        if not points:
            raise FinanceInputError("at least one curve point is required")
        ordered = sorted(points, key=lambda p: p.maturity)
        if len({p.maturity for p in ordered}) != len(ordered):
            raise FinanceInputError("curve maturities must be unique")
        if any(a.discount_factor < b.discount_factor-1e-10 for a, b in zip(ordered, ordered[1:])):
            # Negative rates can legitimately cause increasing discount factors.
            pass
        if extrapolation not in {"flat_forward", "flat_zero", "error"}:
            raise FinanceInputError("unsupported extrapolation policy")
        self.points = tuple(ordered)
        self.extrapolation = extrapolation

    @property
    def maturities(self) -> tuple[float, ...]:
        return tuple(p.maturity for p in self.points)

    def discount_factor(self, maturity: Number) -> float:
        """Interpolate log discount factors, with t=0 fixed at one."""
        t = finite(maturity, "maturity")
        if t < 0:
            raise FinanceInputError("maturity must be non-negative")
        if t == 0:
            return 1.0
        xs, ys = self.maturities, tuple(math.log(p.discount_factor) for p in self.points)
        if t < xs[0]:
            if self.extrapolation == "error":
                raise FinanceInputError("maturity before curve range")
            slope = ys[0]/xs[0] if self.extrapolation == "flat_zero" else (ys[1]-ys[0])/(xs[1]-xs[0]) if len(xs)>1 else ys[0]/xs[0]
            return math.exp(slope*t)
        if t > xs[-1]:
            if self.extrapolation == "error":
                raise FinanceInputError("maturity beyond curve range")
            if self.extrapolation == "flat_zero" or len(xs) == 1:
                slope = ys[-1]/xs[-1]
            else:
                slope = (ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            return math.exp(ys[-1]+slope*(t-xs[-1]))
        for i, (left, right) in enumerate(zip(xs, xs[1:])):
            if left <= t <= right:
                weight = (t-left)/(right-left)
                return math.exp(ys[i]*(1-weight)+ys[i+1]*weight)
        return self.points[-1].discount_factor

    def zero_rate(self, maturity: Number, *, compounding: str = "continuous") -> float:
        """Zero rate implied by a discount factor."""
        t = positive(maturity, "maturity")
        df = self.discount_factor(t)
        if compounding == "continuous":
            return -math.log(df)/t
        if compounding == "annual":
            return df**(-1/t)-1
        raise FinanceInputError("compounding must be continuous or annual")

    def forward_rate(self, start: Number, end: Number,
                     *, compounding: str = "simple") -> float:
        """Implied forward rate from start to end using curve discount factors."""
        t0, t1 = finite(start), positive(end, "end")
        if t0 < 0 or t1 <= t0:
            raise FinanceInputError("require 0 <= start < end")
        ratio = self.discount_factor(t0)/self.discount_factor(t1)
        tau = t1-t0
        if compounding == "continuous":
            return math.log(ratio)/tau
        if compounding == "simple":
            return (ratio-1)/tau
        if compounding == "annual":
            return ratio**(1/tau)-1
        raise FinanceInputError("unsupported forward compounding")

    def parallel_shift(self, basis_points: Number) -> "DiscountCurve":
        """Return a curve shifted in continuously compounded zero-rate space."""
        shift = finite(basis_points, "basis_points")/10000.0
        shifted = [CurvePoint(p.maturity, p.discount_factor*math.exp(-shift*p.maturity))
                   for p in self.points]
        return DiscountCurve(shifted, extrapolation=self.extrapolation)

    def key_rate_shift(self, node_index: int, basis_points: Number) -> "DiscountCurve":
        """Shift a single quoted zero node and rebuild discount factors."""
        if not 0 <= node_index < len(self.points):
            raise FinanceInputError("node_index outside curve")
        shift = finite(basis_points)/10000
        rebuilt = []
        for i, point in enumerate(self.points):
            zero = self.zero_rate(point.maturity)+(shift if i == node_index else 0.0)
            rebuilt.append(CurvePoint(point.maturity, math.exp(-zero*point.maturity)))
        return DiscountCurve(rebuilt, extrapolation=self.extrapolation)


def bootstrap_deposit_curve(deposits: Sequence[tuple[float, float]]) -> DiscountCurve:
    """Bootstrap discount factors from simple money-market deposit rates."""
    points = []
    for maturity, rate in deposits:
        t, r = positive(maturity, "maturity"), finite(rate, "rate")
        if 1+r*t <= 0:
            raise FinanceInputError("deposit quote implies non-positive discount factor")
        points.append(CurvePoint(t, 1/(1+r*t)))
    return DiscountCurve(points)


def bootstrap_par_swap_curve(deposits: Sequence[tuple[float, float]],
                             swaps: Sequence[tuple[float, float, int]]) -> DiscountCurve:
    """Bootstrap a simple annualized fixed-leg par swap curve.

    ``swaps`` contains (maturity_years, par_rate, fixed_payment_frequency).
    All schedules are regular and begin at time zero; business-day calendars,
    stubs, collateral and payment lags must be layered on by the caller.
    """
    known: dict[float, float] = {}
    for maturity, rate in deposits:
        t, r = positive(maturity), finite(rate)
        df = 1/(1+r*t)
        known[t] = df
    for maturity, par_rate, frequency in sorted(swaps, key=lambda x: x[0]):
        t, rate = positive(maturity), finite(par_rate)
        if not isinstance(frequency, int) or frequency < 1:
            raise FinanceInputError("frequency must be positive integer")
        periods = int(round(t*frequency))
        if abs(periods/frequency-t) > 1e-8 or periods < 1:
            raise FinanceInputError("swap maturity must align with fixed payment frequency")
        accrual = 1/frequency
        earlier = [known.get(i*accrual) for i in range(1, periods)]
        if any(df is None for df in earlier):
            raise FinanceInputError("missing earlier discount factors for swap bootstrap")
        final_df = (1-rate*accrual*sum(earlier))/(1+rate*accrual)
        if final_df <= 0:
            raise FinanceInputError("swap quote implies non-positive discount factor")
        known[t] = final_df
    return DiscountCurve([CurvePoint(t, df) for t, df in sorted(known.items())])


def par_swap_rate(curve: DiscountCurve, maturity: Number, frequency: int = 2) -> float:
    """Compute fixed coupon rate that makes a vanilla par swap worth zero."""
    t = positive(maturity, "maturity")
    if not isinstance(frequency, int) or frequency < 1:
        raise FinanceInputError("frequency must be positive")
    periods = int(round(t*frequency))
    if abs(periods/frequency-t) > 1e-8:
        raise FinanceInputError("maturity must align with payment frequency")
    annuity = sum(curve.discount_factor(i/frequency)/frequency for i in range(1, periods+1))
    return (1-curve.discount_factor(t))/annuity


def discount_cashflows(curve: DiscountCurve, cashflows: Sequence[tuple[float, float]]) -> float:
    """Discount arbitrary deterministic cash flows using a term structure."""
    return sum(finite(amount)*curve.discount_factor(positive(t, "time", allow_zero=True))
               for t, amount in cashflows)
