"""CRR binomial pricing for European and American vanilla options."""
from __future__ import annotations

import math
from ..core import FinanceInputError, Number, finite, positive


def crr_price(spot: Number, strike: Number, time: Number, rate: Number,
              volatility: Number, *, steps: int = 200, option_type: str = "call",
              american: bool = False, dividend_yield: Number = 0.0) -> float:
    """Cox-Ross-Rubinstein tree with optional early exercise at each node."""
    s, k, t = positive(spot), positive(strike), positive(time, allow_zero=True)
    r, sigma, q = finite(rate), positive(volatility, allow_zero=True), finite(dividend_yield)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise FinanceInputError("option_type must be call or put")
    if not isinstance(steps, int) or steps < 1:
        raise FinanceInputError("steps must be a positive integer")
    if t == 0:
        return max(0.0, s-k) if kind == "call" else max(0.0, k-s)
    if sigma == 0:
        forward = s*math.exp((r-q)*t)
        discounted = max(0.0, forward-k) if kind == "call" else max(0.0, k-forward)
        return math.exp(-r*t)*discounted
    dt = t/steps
    up = math.exp(sigma*math.sqrt(dt))
    down = 1.0/up
    growth = math.exp((r-q)*dt)
    probability = (growth-down)/(up-down)
    if not 0 <= probability <= 1:
        raise FinanceInputError("CRR risk-neutral probability outside [0, 1]; adjust steps or inputs")
    discount = math.exp(-r*dt)
    values = []
    for j in range(steps+1):
        terminal = s*(up**j)*(down**(steps-j))
        values.append(max(0.0, terminal-k) if kind == "call" else max(0.0, k-terminal))
    for level in range(steps-1, -1, -1):
        updated = []
        for j in range(level+1):
            continuation = discount*(probability*values[j+1]+(1-probability)*values[j])
            if american:
                node_spot = s*(up**j)*(down**(level-j))
                exercise = max(0.0, node_spot-k) if kind == "call" else max(0.0, k-node_spot)
                continuation = max(continuation, exercise)
            updated.append(continuation)
        values = updated
    return values[0]


def binomial_delta(spot: Number, strike: Number, time: Number, rate: Number,
                   volatility: Number, *, steps: int = 200,
                   option_type: str = "call", american: bool = False,
                   dividend_yield: Number = 0.0, bump: Number = 1e-4) -> float:
    """Finite-difference spot delta using a symmetric relative bump."""
    s, h = positive(spot), positive(bump)
    up = crr_price(s*(1+h), strike, time, rate, volatility, steps=steps,
                   option_type=option_type, american=american, dividend_yield=dividend_yield)
    down = crr_price(s*(1-h), strike, time, rate, volatility, steps=steps,
                     option_type=option_type, american=american, dividend_yield=dividend_yield)
    return (up-down)/(2*s*h)


def binomial_gamma(spot: Number, strike: Number, time: Number, rate: Number,
                   volatility: Number, *, steps: int = 200,
                   option_type: str = "call", american: bool = False,
                   dividend_yield: Number = 0.0, bump: Number = 1e-3) -> float:
    """Finite-difference spot gamma using a symmetric relative bump."""
    s, h = positive(spot), positive(bump)
    center = crr_price(s, strike, time, rate, volatility, steps=steps, option_type=option_type,
                       american=american, dividend_yield=dividend_yield)
    up = crr_price(s*(1+h), strike, time, rate, volatility, steps=steps, option_type=option_type,
                   american=american, dividend_yield=dividend_yield)
    down = crr_price(s*(1-h), strike, time, rate, volatility, steps=steps, option_type=option_type,
                     american=american, dividend_yield=dividend_yield)
    return (up-2*center+down)/(s*h)**2
