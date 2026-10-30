"""Black-Scholes-Merton European option pricing and implied-volatility tools."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import FinanceInputError, Number, finite, normal_cdf, normal_pdf, positive


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float


@dataclass(frozen=True)
class OptionQuote:
    price: float
    intrinsic_value: float
    time_value: float
    greeks: OptionGreeks


def _validate(spot: Number, strike: Number, time: Number, rate: Number,
              volatility: Number, dividend_yield: Number) -> tuple[float, ...]:
    s = positive(spot, "spot")
    k = positive(strike, "strike")
    t = positive(time, "time", allow_zero=True)
    r = finite(rate, "rate")
    sigma = positive(volatility, "volatility", allow_zero=True)
    q = finite(dividend_yield, "dividend_yield")
    return s, k, t, r, sigma, q


def d1_d2(spot: Number, strike: Number, time: Number, rate: Number,
          volatility: Number, dividend_yield: Number = 0.0) -> tuple[float, float]:
    """Return standard Black-Scholes d1 and d2."""
    s, k, t, r, sigma, q = _validate(spot, strike, time, rate, volatility, dividend_yield)
    if t <= 0 or sigma <= 0:
        raise FinanceInputError("d1/d2 require positive time and volatility")
    d1 = (math.log(s/k)+(r-q+0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
    return d1, d1-sigma*math.sqrt(t)


def price(spot: Number, strike: Number, time: Number, rate: Number,
          volatility: Number, *, option_type: str = "call",
          dividend_yield: Number = 0.0) -> float:
    """Price a European call or put under Black-Scholes-Merton."""
    s, k, t, r, sigma, q = _validate(spot, strike, time, rate, volatility, dividend_yield)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise FinanceInputError("option_type must be call or put")
    if t == 0:
        return max(0.0, s-k) if kind == "call" else max(0.0, k-s)
    discounted_s, discounted_k = s*math.exp(-q*t), k*math.exp(-r*t)
    if sigma == 0:
        forward_intrinsic = discounted_s-discounted_k
        return max(0.0, forward_intrinsic) if kind == "call" else max(0.0, -forward_intrinsic)
    d_1, d_2 = d1_d2(s, k, t, r, sigma, q)
    if kind == "call":
        return discounted_s*normal_cdf(d_1)-discounted_k*normal_cdf(d_2)
    return discounted_k*normal_cdf(-d_2)-discounted_s*normal_cdf(-d_1)


def greeks(spot: Number, strike: Number, time: Number, rate: Number,
           volatility: Number, *, option_type: str = "call",
           dividend_yield: Number = 0.0) -> OptionGreeks:
    """Return analytic first- and selected second-order Greeks per share."""
    s, k, t, r, sigma, q = _validate(spot, strike, time, rate, volatility, dividend_yield)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise FinanceInputError("option_type must be call or put")
    if t <= 0 or sigma <= 0:
        raise FinanceInputError("Greeks require positive time and volatility")
    d_1, d_2 = d1_d2(s, k, t, r, sigma, q)
    disc_q, disc_r = math.exp(-q*t), math.exp(-r*t)
    pdf = normal_pdf(d_1)
    delta_call = disc_q*normal_cdf(d_1)
    delta = delta_call if kind == "call" else delta_call-disc_q
    gamma = disc_q*pdf/(s*sigma*math.sqrt(t))
    vega = s*disc_q*pdf*math.sqrt(t)
    common_theta = -s*disc_q*pdf*sigma/(2*math.sqrt(t))
    if kind == "call":
        theta = common_theta - r*k*disc_r*normal_cdf(d_2) + q*s*disc_q*normal_cdf(d_1)
        rho = k*t*disc_r*normal_cdf(d_2)
    else:
        theta = common_theta + r*k*disc_r*normal_cdf(-d_2) - q*s*disc_q*normal_cdf(-d_1)
        rho = -k*t*disc_r*normal_cdf(-d_2)
    vanna = -disc_q*pdf*d_2/sigma
    volga = vega*d_1*d_2/sigma
    return OptionGreeks(delta, gamma, vega, theta, rho, vanna, volga)


def quote(spot: Number, strike: Number, time: Number, rate: Number,
          volatility: Number, *, option_type: str = "call",
          dividend_yield: Number = 0.0) -> OptionQuote:
    """Bundle model price, intrinsic value, time value and Greeks."""
    s, k, t, r, sigma, q = _validate(spot, strike, time, rate, volatility, dividend_yield)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise FinanceInputError("option_type must be call or put")
    model_price = price(s, k, t, r, sigma, option_type=kind, dividend_yield=q)
    intrinsic = max(0.0, s-k) if kind == "call" else max(0.0, k-s)
    g = greeks(s, k, t, r, sigma, option_type=kind, dividend_yield=q) if t > 0 and sigma > 0 else OptionGreeks(0,0,0,0,0,0,0)
    return OptionQuote(model_price, intrinsic, max(0.0, model_price-intrinsic), g)


def implied_volatility(market_price: Number, spot: Number, strike: Number,
                       time: Number, rate: Number, *, option_type: str = "call",
                       dividend_yield: Number = 0.0, lower: Number = 1e-6,
                       upper: Number = 5.0, tolerance: Number = 1e-9,
                       max_iterations: int = 200) -> float:
    """Invert Black-Scholes price with a safeguarded bisection solver."""
    from ..core import bisect
    target = positive(market_price, "market_price", allow_zero=True)
    s, k, t, r, _, q = _validate(spot, strike, time, rate, 0.2, dividend_yield)
    if t <= 0:
        raise FinanceInputError("implied volatility requires positive time")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise FinanceInputError("option_type must be call or put")
    low, high = positive(lower, "lower"), positive(upper, "upper")
    if low >= high:
        raise FinanceInputError("lower volatility must be below upper")
    intrinsic_bound = max(0.0, s*math.exp(-q*t)-k*math.exp(-r*t)) if kind == "call" else max(0.0, k*math.exp(-r*t)-s*math.exp(-q*t))
    upper_bound = s*math.exp(-q*t) if kind == "call" else k*math.exp(-r*t)
    if target < intrinsic_bound-1e-8 or target > upper_bound+1e-8:
        raise FinanceInputError("market price violates European option bounds")
    def objective(vol: float) -> float:
        return price(s, k, t, r, vol, option_type=kind, dividend_yield=q)
    lo_price, hi_price = objective(low), objective(high)
    if target <= lo_price:
        return low
    if target > hi_price:
        raise FinanceInputError("upper volatility bound does not bracket market price")
    return bisect(objective, target, low, high, tolerance=float(tolerance), max_iterations=max_iterations)


def put_call_parity_residual(call: Number, put: Number, spot: Number,
                             strike: Number, time: Number, rate: Number,
                             dividend_yield: Number = 0.0) -> float:
    """Call-put parity residual; zero indicates parity under model assumptions."""
    c, p = finite(call), finite(put)
    s, k, t, r, q = positive(spot), positive(strike), positive(time, allow_zero=True), finite(rate), finite(dividend_yield)
    return (c-p) - (s*math.exp(-q*t)-k*math.exp(-r*t))


def delta_hedge_shares(option_contracts: Number, contract_multiplier: Number,
                       option_delta: Number, *, underlying_position: Number = 0.0) -> float:
    """Underlying shares needed to neutralize option delta exposure."""
    contracts = finite(option_contracts)
    multiplier = positive(contract_multiplier)
    delta = finite(option_delta)
    position = finite(underlying_position)
    return -(contracts*multiplier*delta+position)


def gamma_pnl(spot_change: Number, gamma: Number,
              *, underlying_units: Number = 1.0) -> float:
    """Second-order option P&L approximation, 0.5 * gamma * dS^2."""
    move, g, units = finite(spot_change), finite(gamma), finite(underlying_units)
    return 0.5*g*move*move*units


def delta_gamma_vega_pnl(spot_change: Number, vol_change: Number,
                         *, delta: Number, gamma: Number, vega: Number) -> float:
    """Second-order P&L approximation using delta, gamma and vega."""
    ds, dv = finite(spot_change), finite(vol_change)
    return finite(delta)*ds + 0.5*finite(gamma)*ds*ds + finite(vega)*dv
