"""Analytic short-rate model utilities: Vasicek and Cox-Ingersoll-Ross."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..core import FinanceInputError, Number, finite, positive


@dataclass(frozen=True)
class ShortRateBond:
    price: float
    yield_rate: float
    duration: float


def vasicek_bond_price(current_rate: Number, mean_reversion: Number,
                       long_run_rate: Number, volatility: Number,
                       maturity: Number) -> float:
    """Closed-form zero-coupon bond price under the Vasicek model."""
    r, a, b, sigma, t = map(finite, (current_rate, mean_reversion,
                                     long_run_rate, volatility, maturity))
    if a <= 0 or sigma < 0 or t < 0:
        raise FinanceInputError("invalid Vasicek inputs")
    B = (1-math.exp(-a*t))/a
    A = math.exp((b-sigma*sigma/(2*a*a))*(B-t)-sigma*sigma*B*B/(4*a))
    return A*math.exp(-B*r)


def vasicek_zero_rate(current_rate: Number, mean_reversion: Number,
                      long_run_rate: Number, volatility: Number,
                      maturity: Number) -> float:
    """Continuously compounded zero yield from Vasicek zero-coupon price."""
    t = positive(maturity)
    price = vasicek_bond_price(current_rate, mean_reversion, long_run_rate, volatility, t)
    return -math.log(price)/t


def vasicek_bond_duration(current_rate: Number, mean_reversion: Number,
                          long_run_rate: Number, volatility: Number,
                          maturity: Number) -> float:
    """Analytic modified duration of the Vasicek zero coupon bond."""
    r, a, b, sigma, t = map(finite, (current_rate, mean_reversion,
                                     long_run_rate, volatility, maturity))
    if a <= 0 or sigma < 0 or t < 0:
        raise FinanceInputError("invalid Vasicek inputs")
    return (1-math.exp(-a*t))/a


def cir_bond_price(current_rate: Number, mean_reversion: Number,
                   long_run_rate: Number, volatility: Number,
                   maturity: Number) -> float:
    """Closed-form zero-coupon bond price under the CIR model."""
    r, a, b, sigma, t = map(finite, (current_rate, mean_reversion,
                                     long_run_rate, volatility, maturity))
    if min(r, a, b, sigma, t) < 0 or a <= 0:
        raise FinanceInputError("invalid CIR inputs")
    if sigma == 0:
        # Deterministic mean-reverting short-rate path.
        integral = b*t+(r-b)*(1-math.exp(-a*t))/a
        return math.exp(-integral)
    gamma = math.sqrt(a*a+2*sigma*sigma)
    exp_gt = math.exp(gamma*t)
    denominator = (gamma+a)*(exp_gt-1)+2*gamma
    B = 2*(exp_gt-1)/denominator
    A = (2*gamma*math.exp((a+gamma)*t/2)/denominator)**(2*a*b/(sigma*sigma))
    return A*math.exp(-B*r)


def cir_zero_rate(current_rate: Number, mean_reversion: Number,
                  long_run_rate: Number, volatility: Number,
                  maturity: Number) -> float:
    """Continuously compounded CIR zero rate."""
    t = positive(maturity)
    return -math.log(cir_bond_price(current_rate, mean_reversion,
                                    long_run_rate, volatility, t))/t


def cir_feller_condition(mean_reversion: Number, long_run_rate: Number,
                         volatility: Number) -> dict[str, float]:
    """Check the CIR Feller condition 2*kappa*theta >= sigma^2."""
    kappa, theta, sigma = map(finite, (mean_reversion, long_run_rate, volatility))
    if min(kappa, theta, sigma) < 0:
        raise FinanceInputError("CIR parameters must be non-negative")
    lhs, rhs = 2*kappa*theta, sigma*sigma
    return {"lhs": lhs, "rhs": rhs, "satisfied": float(lhs >= rhs)}


def vasicek_caplet_price(notional: Number, accrual: Number,
                         strike: Number, forward_rate: Number,
                         forward_volatility: Number,
                         discount_factor: Number,
                         *, payer: bool = True) -> float:
    """Black-76 caplet/floorlet approximation on a forward rate."""
    from ..core import normal_cdf
    n, tau, k, f, vol, df = (positive(notional), positive(accrual), finite(strike),
                             finite(forward_rate), positive(forward_volatility),
                             positive(discount_factor))
    d1 = (math.log(f/k)+0.5*vol*vol)/(vol) if k > 0 and f > 0 else None
    if d1 is None:
        raise FinanceInputError("Black caplet requires positive strike and forward")
    d2 = d1-vol
    if payer:
        payoff = f*normal_cdf(d1)-k*normal_cdf(d2)
    else:
        payoff = k*normal_cdf(-d2)-f*normal_cdf(-d1)
    return n*tau*df*payoff


def swaption_black_price(notional: Number, annuity: Number,
                         forward_swap_rate: Number, strike: Number,
                         volatility: Number, expiry: Number,
                         *, payer: bool = True) -> float:
    """Black-76 payer/receiver swaption price for lognormal swap rates."""
    from ..core import normal_cdf
    n, a, f, k, vol, t = (positive(notional), positive(annuity), positive(forward_swap_rate),
                          positive(strike), positive(volatility), positive(expiry))
    sigma_t = vol*math.sqrt(t)
    d1 = (math.log(f/k)+0.5*sigma_t*sigma_t)/sigma_t
    d2 = d1-sigma_t
    value = f*normal_cdf(d1)-k*normal_cdf(d2) if payer else k*normal_cdf(-d2)-f*normal_cdf(-d1)
    return n*a*value


def forward_rate_agreement_curve(curve_points: Sequence[tuple[float, float]],
                                 start: Number, end: Number) -> float:
    """Compute simple forward rate from a sorted (time, discount-factor) curve."""
    from .curves import CurvePoint, DiscountCurve
    curve = DiscountCurve([CurvePoint(t, df) for t, df in curve_points])
    return curve.forward_rate(start, end, compounding="simple")
