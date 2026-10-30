"""Forwards, futures, FX carry, FRAs and vanilla interest-rate swaps."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .core import FinanceInputError, Number, finite, positive
from .fixed_income.curves import DiscountCurve


@dataclass(frozen=True)
class ForwardQuote:
    forward: float
    spot: float
    carry_factor: float
    maturity: float


@dataclass(frozen=True)
class SwapCashflow:
    payment_time: float
    fixed_amount: float
    floating_amount: float
    net_receive_fixed: float


def equity_forward(spot: Number, rate: Number, dividend_yield: Number,
                   maturity: Number, *, compounding: str = "continuous") -> ForwardQuote:
    """Fair equity forward with continuous or annual carry convention."""
    s, t = positive(spot), positive(maturity)
    r, q = finite(rate), finite(dividend_yield)
    carry = r-q
    factor = math.exp(carry*t) if compounding == "continuous" else (1+carry)**t if compounding == "annual" else None
    if factor is None or factor <= 0:
        raise FinanceInputError("unsupported compounding or non-positive carry factor")
    return ForwardQuote(s*factor, s, factor, t)


def commodity_forward(spot: Number, rate: Number, storage_cost: Number,
                      convenience_yield: Number, maturity: Number) -> ForwardQuote:
    """Cost-of-carry commodity forward: r + storage - convenience yield."""
    s, t = positive(spot), positive(maturity)
    carry = finite(rate)+finite(storage_cost)-finite(convenience_yield)
    factor = math.exp(carry*t)
    return ForwardQuote(s*factor, s, factor, t)


def fx_forward(spot_domestic_per_foreign: Number, domestic_rate: Number,
               foreign_rate: Number, maturity: Number,
               *, compounding: str = "continuous") -> ForwardQuote:
    """Covered-interest-parity FX forward in domestic currency per foreign unit."""
    spot, t = positive(spot_domestic_per_foreign), positive(maturity)
    rd, rf = finite(domestic_rate), finite(foreign_rate)
    if compounding == "continuous":
        factor = math.exp((rd-rf)*t)
    elif compounding == "simple":
        factor = (1+rd*t)/(1+rf*t)
        if 1+rf*t <= 0:
            raise FinanceInputError("foreign money-market discount denominator must be positive")
    else:
        raise FinanceInputError("compounding must be continuous or simple")
    return ForwardQuote(spot*factor, spot, factor, t)


def forward_value(long_position: bool, spot: Number, strike_forward: Number,
                  discount_factor: Number, carry_factor: Number) -> float:
    """Value an existing forward as discounted prepaid-forward difference."""
    s, k = positive(spot), positive(strike_forward)
    df, carry = positive(discount_factor), positive(carry_factor)
    value = df*(s*carry-k)
    return value if long_position else -value


def futures_fair_value(spot: Number, rate: Number, dividend_yield: Number,
                       maturity: Number, *, known_income_pv: Number = 0.0) -> float:
    """Equity-index futures fair value with known income PV paid before expiry."""
    s, t = positive(spot), positive(maturity)
    r, q, income = finite(rate), finite(dividend_yield), finite(known_income_pv)
    if income < 0 or income >= s:
        raise FinanceInputError("known_income_pv must be in [0, spot)")
    prepaid = s-income
    return prepaid*math.exp((r-q)*t)


def basis(futures_price: Number, spot_price: Number) -> float:
    """Absolute futures basis, futures less spot."""
    return finite(futures_price)-positive(spot_price)


def annualized_basis(futures_price: Number, spot_price: Number,
                     maturity: Number) -> float:
    """Simple annualized futures basis relative to spot."""
    return basis(futures_price, spot_price)/positive(spot_price)/positive(maturity)


def implied_repo_rate(spot_price: Number, futures_price: Number,
                      maturity: Number, *, income_pv: Number = 0.0,
                      continuous: bool = False) -> float:
    """Repo/carry rate implied by spot, futures and known income PV."""
    spot, future, t = positive(spot_price), positive(futures_price), positive(maturity)
    income = finite(income_pv)
    if income < 0 or income >= spot:
        raise FinanceInputError("income_pv must be in [0, spot)")
    ratio = future/(spot-income)
    return math.log(ratio)/t if continuous else (ratio-1)/t


def fra_value(notional: Number, fixed_rate: Number, forward_rate: Number,
              accrual_fraction: Number, discount_factor: Number,
              *, pay_fixed: bool = True) -> float:
    """Value a forward-rate agreement at reset, with simple settlement discount."""
    n = finite(notional)
    if n < 0:
        raise FinanceInputError("notional must be non-negative")
    fixed, floating = finite(fixed_rate), finite(forward_rate)
    tau, df = positive(accrual_fraction), positive(discount_factor)
    numerator = (floating-fixed)*tau
    amount = n*numerator/(1+floating*tau)*df
    return -amount if pay_fixed else amount


def par_swap_rate(curve: DiscountCurve, maturity: Number,
                  fixed_frequency: int = 2) -> float:
    """Par fixed rate for a vanilla single-curve interest-rate swap."""
    from .fixed_income.curves import par_swap_rate as _par
    return _par(curve, maturity, fixed_frequency)


def interest_rate_swap_value(notional: Number, fixed_rate: Number,
                             curve: DiscountCurve, maturity: Number,
                             fixed_frequency: int = 2,
                             *, receive_fixed: bool = True) -> float:
    """Single-curve payer/receiver swap PV for a regular schedule.

    Floating leg PV is approximated as N*(1-DF(T)); this is valid for a
    standard single-curve par floating leg under the stated simplified setup.
    """
    n, k, t = positive(notional), finite(fixed_rate), positive(maturity)
    if not isinstance(fixed_frequency, int) or fixed_frequency < 1:
        raise FinanceInputError("fixed_frequency must be positive")
    periods = int(round(t*fixed_frequency))
    if abs(periods/fixed_frequency-t) > 1e-8:
        raise FinanceInputError("maturity must align with fixed schedule")
    annuity = sum(curve.discount_factor(i/fixed_frequency)/fixed_frequency
                  for i in range(1, periods+1))
    fixed_pv = n*k*annuity
    floating_pv = n*(1-curve.discount_factor(t))
    receive_fixed_value = fixed_pv-floating_pv
    return receive_fixed_value if receive_fixed else -receive_fixed_value


def swap_cashflows(notional: Number, fixed_rate: Number,
                   floating_rates: Sequence[Number], frequency: int = 2,
                   *, receive_fixed: bool = True) -> tuple[SwapCashflow, ...]:
    """Generate deterministic net swap cash flows from supplied reset rates."""
    n, fixed = positive(notional), finite(fixed_rate)
    if not isinstance(frequency, int) or frequency < 1:
        raise FinanceInputError("frequency must be positive")
    output = []
    for i, raw_rate in enumerate(floating_rates, start=1):
        floating = finite(raw_rate)
        fixed_amount = n*fixed/frequency
        float_amount = n*floating/frequency
        net = fixed_amount-float_amount if receive_fixed else float_amount-fixed_amount
        output.append(SwapCashflow(i/frequency, fixed_amount, float_amount, net))
    return tuple(output)


def cross_currency_basis_notional(domestic_notional: Number,
                                  spot_domestic_per_foreign: Number) -> float:
    """Translate domestic principal into foreign principal at spot FX."""
    return positive(domestic_notional)/positive(spot_domestic_per_foreign)


def carry_roll_down_pnl(position_value: Number, carry_return: Number,
                        roll_down_return: Number) -> dict[str, float]:
    """Separate first-order carry and roll-down P&L contributions."""
    value = finite(position_value)
    carry = value*finite(carry_return)
    roll = value*finite(roll_down_return)
    return {"carry_pnl": carry, "roll_down_pnl": roll, "total_pnl": carry+roll}


def futures_calendar_spread_near_far(near_price: Number, far_price: Number) -> float:
    """Calendar spread quote as near minus far."""
    return finite(near_price)-finite(far_price)


def futures_curve_slope(prices: Sequence[tuple[float, float]]) -> float:
    """Linear slope of futures price against maturity from first to last contract."""
    if len(prices) < 2:
        raise FinanceInputError("at least two futures contracts are required")
    ordered = sorted((positive(t, "maturity"), finite(p, "price")) for t, p in prices)
    if ordered[-1][0] == ordered[0][0]:
        raise FinanceInputError("contract maturities must differ")
    return (ordered[-1][1]-ordered[0][1])/(ordered[-1][0]-ordered[0][0])
