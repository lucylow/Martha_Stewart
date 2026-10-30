"""Parametric yield-curve models and term-structure diagnostics."""
from __future__ import annotations

import math
from typing import Sequence

from ..core import FinanceInputError, Number, finite, positive
from ..regression import ols
from .curves import CurvePoint, DiscountCurve


def nelson_siegel_yield(maturity: Number, beta0: Number, beta1: Number,
                        beta2: Number, tau: Number) -> float:
    """Evaluate the Nelson-Siegel zero curve at a maturity in years."""
    t, b0, b1, b2, decay = map(finite, (maturity, beta0, beta1, beta2, tau))
    if t < 0 or decay <= 0:
        raise FinanceInputError("maturity must be non-negative and tau positive")
    if t == 0:
        return b0+b1
    x = t/decay
    loading1 = (1-math.exp(-x))/x
    loading2 = loading1-math.exp(-x)
    return b0+b1*loading1+b2*loading2


def nelson_siegel_forward(maturity: Number, beta0: Number, beta1: Number,
                          beta2: Number, tau: Number) -> float:
    """Instantaneous forward rate implied by Nelson-Siegel parameters."""
    t, b0, b1, b2, decay = map(finite, (maturity, beta0, beta1, beta2, tau))
    if t < 0 or decay <= 0:
        raise FinanceInputError("maturity must be non-negative and tau positive")
    return b0+b1*math.exp(-t/decay)+b2*(t/decay)*math.exp(-t/decay)


def svensson_yield(maturity: Number, beta0: Number, beta1: Number,
                   beta2: Number, beta3: Number,
                   tau1: Number, tau2: Number) -> float:
    """Evaluate the six-parameter Nelson-Siegel-Svensson zero curve."""
    t, b0, b1, b2, b3, d1, d2 = map(finite, (maturity, beta0, beta1,
        beta2, beta3, tau1, tau2))
    if t < 0 or d1 <= 0 or d2 <= 0:
        raise FinanceInputError("maturity must be non-negative and taus positive")
    if t == 0:
        return b0+b1
    x1, x2 = t/d1, t/d2
    l1 = (1-math.exp(-x1))/x1
    l2 = l1-math.exp(-x1)
    l3 = (1-math.exp(-x2))/x2-math.exp(-x2)
    return b0+b1*l1+b2*l2+b3*l3


def fit_nelson_siegel(maturities: Sequence[Number], yields: Sequence[Number],
                      tau_grid: Sequence[Number] | None = None) -> dict[str, object]:
    """Fit Nelson-Siegel betas by OLS for each candidate tau, selecting SSE minimum.

    This profile-grid calibration is transparent and stable for small curve
    datasets. It is not a global nonlinear optimizer for all six Svensson
    parameters; use it as a benchmark or initialization.
    """
    times = tuple(map(finite, maturities))
    rates = tuple(map(finite, yields))
    if len(times) != len(rates) or len(times) < 4 or any(t <= 0 for t in times):
        raise FinanceInputError("at least four aligned positive maturities and yields required")
    candidates = tuple(map(positive, tau_grid or (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0)))
    best = None
    for tau in candidates:
        x1, x2 = [], []
        for t in times:
            x = t/tau
            l1 = (1-math.exp(-x))/x
            x1.append(l1)
            x2.append(l1-math.exp(-x))
        fit = ols(rates, (x1, x2), intercept=True)
        sse = sum(v*v for v in fit.residuals)
        if best is None or sse < best[0]:
            best = (sse, tau, fit)
    assert best is not None
    sse, tau, fit = best
    b0, b1, b2 = fit.coefficients
    return {"beta0": b0, "beta1": b1, "beta2": b2, "tau": tau,
            "sse": sse, "r_squared": fit.r_squared,
            "fitted_yields": tuple(nelson_siegel_yield(t, b0, b1, b2, tau) for t in times)}


def curve_from_zero_rates(maturities: Sequence[Number],
                          zero_rates: Sequence[Number],
                          *, compounding: str = "continuous") -> DiscountCurve:
    """Build a log-linear discount curve from quoted zero rates."""
    times, rates = tuple(map(finite, maturities)), tuple(map(finite, zero_rates))
    if len(times) != len(rates) or not times:
        raise FinanceInputError("maturity and rate arrays must be non-empty and aligned")
    points = []
    for t, rate in zip(times, rates):
        maturity = positive(t, "maturity")
        if compounding == "continuous":
            df = math.exp(-rate*maturity)
        elif compounding == "annual":
            if 1+rate <= 0:
                raise FinanceInputError("annual zero rate must exceed -100%")
            df = (1+rate)**(-maturity)
        else:
            raise FinanceInputError("compounding must be continuous or annual")
        points.append(CurvePoint(maturity, df))
    return DiscountCurve(points)


def curve_slope(curve: DiscountCurve, short_maturity: Number = 2.0,
                long_maturity: Number = 10.0) -> float:
    """Long zero yield minus short zero yield."""
    short, long = positive(short_maturity), positive(long_maturity)
    if long <= short:
        raise FinanceInputError("long_maturity must exceed short_maturity")
    return curve.zero_rate(long)-curve.zero_rate(short)


def curve_curvature(curve: DiscountCurve, short_maturity: Number = 2.0,
                    belly_maturity: Number = 5.0,
                    long_maturity: Number = 10.0) -> float:
    """Twist/curvature proxy: twice belly yield less short and long yields."""
    short, belly, long = map(positive, (short_maturity, belly_maturity, long_maturity))
    if not short < belly < long:
        raise FinanceInputError("require short < belly < long maturities")
    return 2*curve.zero_rate(belly)-curve.zero_rate(short)-curve.zero_rate(long)


def curve_roll_down_pnl(curve: DiscountCurve, cashflows: Sequence[tuple[float, float]],
                        horizon: Number) -> float:
    """Reprice cash flows after they roll down the unchanged curve by horizon."""
    from .curves import discount_cashflows
    shift = positive(horizon, "horizon")
    if any(t <= shift for t, _ in cashflows):
        raise FinanceInputError("all cash flows must remain beyond roll horizon")
    initial = discount_cashflows(curve, cashflows)
    rolled = [(t-shift, cf) for t, cf in cashflows]
    future_value_at_horizon = discount_cashflows(curve, rolled)
    return future_value_at_horizon-initial


def forward_zero_spread(curve_a: DiscountCurve, curve_b: DiscountCurve,
                        start: Number, end: Number) -> float:
    """Difference between two continuously compounded forward rates."""
    return (curve_a.forward_rate(start, end, compounding="continuous")
            -curve_b.forward_rate(start, end, compounding="continuous"))


def key_rate_exposure(curve: DiscountCurve,
                      reprice, bump_bps: Number = 1.0) -> tuple[dict[str, float], ...]:
    """One-node-at-a-time price sensitivity by curve maturity node."""
    bump = finite(bump_bps)
    if bump == 0:
        raise FinanceInputError("bump_bps cannot be zero")
    base_price = finite(reprice(curve))
    exposures = []
    for i, point in enumerate(curve.points):
        shifted = curve.key_rate_shift(i, bump)
        shocked_price = finite(reprice(shifted))
        exposures.append({"maturity": point.maturity,
                          "dv01": (base_price-shocked_price)/bump})
    return tuple(exposures)


def par_coupon_from_discount_factors(discount_factors: Sequence[Number],
                                     accrual_fractions: Sequence[Number],
                                     redemption: Number = 1.0) -> float:
    """Par coupon rate for a fixed leg from discount factors and accruals."""
    dfs, accruals = tuple(map(positive, discount_factors)), tuple(map(positive, accrual_fractions))
    if len(dfs) != len(accruals) or not dfs:
        raise FinanceInputError("discount factors and accrual fractions must align")
    principal = positive(redemption)
    annuity = sum(df*tau for df, tau in zip(dfs, accruals))
    if annuity <= 0:
        raise FinanceInputError("fixed leg annuity must be positive")
    return principal*(1-dfs[-1])/annuity
