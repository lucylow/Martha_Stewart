"""Fixed-income cash flows, bond pricing, yield and risk measures."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

from ..core import CashFlow, DayCount, FinanceInputError, Number, bisect, finite, positive, year_fraction


@dataclass(frozen=True)
class BondRisk:
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    dollar_duration: float


def coupon_schedule(maturity_years: Number, frequency: int = 2) -> tuple[float, ...]:
    """Build regular coupon times in years, including final maturity."""
    maturity = positive(maturity_years, "maturity_years")
    if not isinstance(frequency, int) or frequency < 1:
        raise FinanceInputError("frequency must be a positive integer")
    periods = int(round(maturity*frequency))
    if periods < 1 or abs(periods/frequency-maturity) > 1e-8:
        raise FinanceInputError("maturity must be an exact coupon-period multiple")
    return tuple(i/frequency for i in range(1, periods+1))


def bond_cashflows(face: Number, coupon_rate: Number, maturity_years: Number,
                   frequency: int = 2) -> tuple[tuple[float, float], ...]:
    """Return (time, amount) cash flows for a plain fixed-rate bullet bond."""
    principal = positive(face, "face")
    coupon = finite(coupon_rate, "coupon_rate")
    if coupon < 0:
        raise FinanceInputError("coupon_rate must be non-negative")
    times = coupon_schedule(maturity_years, frequency)
    coupon_amount = principal*coupon/frequency
    flows = [(t, coupon_amount) for t in times]
    last_t, last_amount = flows[-1]
    flows[-1] = (last_t, last_amount+principal)
    return tuple(flows)


def bond_price(face: Number, coupon_rate: Number, yield_rate: Number,
               maturity_years: Number, frequency: int = 2,
               *, compounding: str = "periodic") -> float:
    """Present value of fixed coupon and principal cash flows."""
    flows = bond_cashflows(face, coupon_rate, maturity_years, frequency)
    y = finite(yield_rate, "yield_rate")
    if compounding == "periodic":
        if 1+y/frequency <= 0:
            raise FinanceInputError("yield creates a non-positive discount base")
        return sum(amount/(1+y/frequency)**(time*frequency) for time, amount in flows)
    if compounding == "continuous":
        return sum(amount*math.exp(-y*time) for time, amount in flows)
    raise FinanceInputError("compounding must be periodic or continuous")


def yield_to_maturity(market_price: Number, face: Number, coupon_rate: Number,
                      maturity_years: Number, frequency: int = 2,
                      *, lower: Number = -0.95, upper: Number = 2.0,
                      tolerance: Number = 1e-10) -> float:
    """Solve nominal annual yield compounded at coupon frequency."""
    target = positive(market_price, "market_price")
    low, high = finite(lower), finite(upper)
    if 1+low/frequency <= 0 or low >= high:
        raise FinanceInputError("invalid yield bracket")
    objective = lambda y: bond_price(face, coupon_rate, y, maturity_years, frequency)
    # Bond price is decreasing in yield; bracket sign is handled by bisection.
    return bisect(objective, target, low, high, tolerance=float(tolerance))


def macaulay_duration(face: Number, coupon_rate: Number, yield_rate: Number,
                      maturity_years: Number, frequency: int = 2) -> float:
    """Present-value-weighted average time of the bond cash flows."""
    flows = bond_cashflows(face, coupon_rate, maturity_years, frequency)
    y = finite(yield_rate)
    if 1+y/frequency <= 0:
        raise FinanceInputError("invalid yield")
    pv = [amount/(1+y/frequency)**(time*frequency) for time, amount in flows]
    price = sum(pv)
    return sum(time*value for (time, _), value in zip(flows, pv))/price


def modified_duration(face: Number, coupon_rate: Number, yield_rate: Number,
                       maturity_years: Number, frequency: int = 2) -> float:
    """Macaulay duration divided by one plus periodic yield."""
    y = finite(yield_rate)
    if 1+y/frequency <= 0:
        raise FinanceInputError("invalid yield")
    return macaulay_duration(face, coupon_rate, y, maturity_years, frequency)/(1+y/frequency)


def convexity(face: Number, coupon_rate: Number, yield_rate: Number,
              maturity_years: Number, frequency: int = 2) -> float:
    """Discrete-compounding convexity measured against annual yield changes."""
    flows = bond_cashflows(face, coupon_rate, maturity_years, frequency)
    y = finite(yield_rate)
    base = 1+y/frequency
    if base <= 0:
        raise FinanceInputError("invalid yield")
    price = bond_price(face, coupon_rate, y, maturity_years, frequency)
    weighted = sum(amount*time*(time+1/frequency)/(base**(time*frequency+2))
                   for time, amount in flows)
    return weighted/price


def bond_risk(face: Number, coupon_rate: Number, yield_rate: Number,
              maturity_years: Number, frequency: int = 2) -> BondRisk:
    """Bundle standard interest-rate sensitivities for a plain bond."""
    price_value = bond_price(face, coupon_rate, yield_rate, maturity_years, frequency)
    mac = macaulay_duration(face, coupon_rate, yield_rate, maturity_years, frequency)
    mod = modified_duration(face, coupon_rate, yield_rate, maturity_years, frequency)
    conv = convexity(face, coupon_rate, yield_rate, maturity_years, frequency)
    dollar_duration = mod*price_value
    return BondRisk(mac, mod, conv, dollar_duration*1e-4, dollar_duration)


def duration_convexity_pnl(market_value: Number, modified_duration_value: Number,
                           convexity_value: Number, yield_change: Number) -> float:
    """Approximate bond P&L for a parallel yield shift."""
    value, duration, conv, dy = map(finite, (market_value, modified_duration_value,
                                             convexity_value, yield_change))
    return value*(-duration*dy+0.5*conv*dy*dy)


def accrued_interest(face: Number, coupon_rate: Number, frequency: int,
                     days_accrued: int, days_period: int) -> float:
    """Simple accrued coupon using actual day counts within current period."""
    principal = positive(face)
    coupon = finite(coupon_rate)
    if coupon < 0 or frequency < 1 or days_period <= 0 or not 0 <= days_accrued <= days_period:
        raise FinanceInputError("invalid accrued interest inputs")
    return principal*coupon/frequency*days_accrued/days_period


def clean_dirty_prices(clean_price: Number, accrued: Number) -> dict[str, float]:
    """Convert quoted clean price to invoice (dirty) price."""
    clean, ai = finite(clean_price), finite(accrued)
    if clean < 0 or ai < 0:
        raise FinanceInputError("clean price and accrued interest must be non-negative")
    return {"clean": clean, "accrued_interest": ai, "dirty": clean+ai}


def z_spread(price_value: Number, cashflows: Sequence[tuple[float, float]],
             discount_factors: Sequence[float], *, lower: Number = -0.5,
             upper: Number = 2.0) -> float:
    """Solve constant continuously compounded spread over supplied discount factors."""
    target = positive(price_value, "price")
    if len(cashflows) != len(discount_factors) or not cashflows:
        raise FinanceInputError("cashflows and discount factors must align")
    flows = [(positive(t, "time", allow_zero=True), finite(cf)) for t, cf in cashflows]
    dfs = [positive(df, "discount_factor") for df in discount_factors]
    if any(t <= 0 for t, _ in flows):
        raise FinanceInputError("cash flow times must be positive")
    objective = lambda spread: sum(cf*df*math.exp(-spread*t)
                                   for (t, cf), df in zip(flows, dfs))
    return bisect(objective, target, finite(lower), finite(upper))


def key_rate_dv01(reprice, curve_nodes: Sequence[float], bump: Number = 1e-4) -> tuple[float, ...]:
    """Finite-difference key-rate DV01 for a caller-supplied curve reprice function.

    ``reprice`` receives a complete tuple of zero rates. Output is positive for
    a price decline under a +1bp node bump.
    """
    nodes = tuple(finite(x) for x in curve_nodes)
    bp = positive(bump, "bump")
    base = finite(reprice(nodes), "base price")
    result = []
    for i in range(len(nodes)):
        shocked = list(nodes)
        shocked[i] += bp
        result.append((base-finite(reprice(tuple(shocked))))*1e-4/bp)
    return tuple(result)
