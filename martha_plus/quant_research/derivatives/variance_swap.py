"""Variance-swap, forward-variance and variance-risk-premium analytics.

Option-strip variance is a discrete approximation to the model-free variance
integral. Results depend on strike coverage, quote quality, rate/forward inputs,
and market-specific filtering rules; this module does not calculate an official
index volatility measure.
"""
from __future__ import annotations
from math import exp, log, sqrt
from typing import Sequence

from ..core import finite_series


def annualized_realized_variance(log_returns: Sequence[float],
                                 observations_per_year: float = 252.0) -> float:
    """Annualize close-to-close realized variance from log returns."""
    r = finite_series(log_returns, name="log_returns", min_length=2)
    if observations_per_year <= 0:
        raise ValueError("observations_per_year must be positive")
    return observations_per_year*sum(v*v for v in r)/len(r)


def realized_variance_path(log_returns: Sequence[float], window: int,
                            observations_per_year: float = 252.0) -> tuple[float | None, ...]:
    """Rolling annualized realized variance, aligned to the right edge of each window."""
    r = finite_series(log_returns, name="log_returns")
    if window < 2 or window > len(r) or observations_per_year <= 0:
        raise ValueError("invalid window or annualization")
    out: list[float | None] = [None]*(window-1)
    rolling = sum(v*v for v in r[:window])
    out.append(observations_per_year*rolling/window)
    for i in range(window, len(r)):
        rolling += r[i]*r[i]-r[i-window]*r[i-window]
        out.append(observations_per_year*rolling/window)
    return tuple(out)


def forward_variance(near_maturity: float, far_maturity: float,
                     near_total_variance: float, far_total_variance: float) -> float:
    """Extract forward variance between two expiries from total variances.

    Maturities are year fractions and total variance is maturity times annual
    variance. A negative result indicates inconsistent/noisy term-structure inputs.
    """
    t1, t2 = float(near_maturity), float(far_maturity)
    v1, v2 = float(near_total_variance), float(far_total_variance)
    if t1 < 0 or t2 <= t1 or v1 < 0 or v2 < 0:
        raise ValueError("require 0 <= near maturity < far maturity and non-negative variances")
    result = (t2*v2-t1*v1)/(t2-t1)
    if result < -1e-10:
        raise ValueError("far total variance is below near total variance; check inputs")
    return max(0.0, result)


def variance_risk_premium(implied_variance: float,
                           expected_realized_variance: float) -> dict:
    """Return implied-minus-expected variance and volatility-scaled premium."""
    iv, rv = float(implied_variance), float(expected_realized_variance)
    if iv < 0 or rv < 0:
        raise ValueError("variance inputs must be non-negative")
    spread = iv-rv
    return {"variance_risk_premium": spread,
            "implied_variance": iv,
            "expected_realized_variance": rv,
            "implied_volatility": sqrt(iv),
            "expected_realized_volatility": sqrt(rv),
            "premium_sign_convention": "positive means implied variance exceeds expected realized variance"}


def variance_swap_strike_from_quotes(forward: float, maturity: float,
                                     strikes: Sequence[float],
                                     option_prices: Sequence[float],
                                     option_types: Sequence[str],
                                     risk_free_rate: float = 0.0) -> dict:
    """Estimate a discrete model-free variance strike from an OTM option strip.

    Strikes must be ascending and unique. Supply puts below K0 and calls above
    K0; the option at K0 may be either type. Endpoint integration weights use
    adjacent strike spacing and interior weights use half-neighbour spans.
    Sparse strips and stale quotes can dominate the numerical approximation.
    """
    f, t, r = float(forward), float(maturity), float(risk_free_rate)
    k = finite_series(strikes, name="strikes", min_length=3)
    q = finite_series(option_prices, name="option_prices", min_length=3)
    kinds = [str(v).lower() for v in option_types]
    if f <= 0 or t <= 0 or len(k) != len(q) or len(k) != len(kinds):
        raise ValueError("invalid forward, maturity or strip dimensions")
    if any(v <= 0 for v in k) or any(k[i] >= k[i+1] for i in range(len(k)-1)):
        raise ValueError("strikes must be positive, strictly ascending and unique")
    if any(v < 0 for v in q) or any(v not in {"call", "put"} for v in kinds):
        raise ValueError("option prices must be non-negative and types call/put")
    k0_candidates = [i for i, strike in enumerate(k) if strike <= f]
    k0_index = k0_candidates[-1] if k0_candidates else 0
    k0 = k[k0_index]
    for i, kind in enumerate(kinds):
        if i < k0_index and kind != "put":
            raise ValueError("strikes below K0 must use OTM put quotes")
        if i > k0_index and kind != "call":
            raise ValueError("strikes above K0 must use OTM call quotes")
    widths = []
    for i in range(len(k)):
        if i == 0:
            widths.append(k[1]-k[0])
        elif i == len(k)-1:
            widths.append(k[-1]-k[-2])
        else:
            widths.append(0.5*(k[i+1]-k[i-1]))
    integral = sum(widths[i]*q[i]/(k[i]*k[i]) for i in range(len(k)))
    correction = (f/k0-1.0)**2/t
    raw = 2.0*exp(r*t)*integral/t-correction
    if raw < -1e-8:
        raise ValueError("option strip produced negative variance; inspect quotes and strike grid")
    strike_var = max(0.0, raw)
    return {"variance_strike": strike_var,
            "volatility_strike": sqrt(strike_var),
            "forward": f, "k0": k0, "maturity": t,
            "option_strip_integral": integral,
            "forward_correction": correction,
            "integration": "discrete trapezoid-like strike weights",
            "limitations": "sensitive to strike truncation, quote staleness and option convention"}


def variance_swap_pnl(realized_variance: float, strike_variance: float,
                       variance_notional: float) -> dict:
    """Calculate linear variance-swap settlement using variance notional."""
    rv, kv, notional = float(realized_variance), float(strike_variance), float(variance_notional)
    if rv < 0 or kv < 0:
        raise ValueError("realized and strike variance must be non-negative")
    payoff = notional*(rv-kv)
    return {"payoff": payoff, "realized_variance": rv,
            "strike_variance": kv, "variance_notional": notional,
            "receiver_position": "long variance receives positive payoff when realized variance exceeds strike"}


def volatility_swap_approximation(variance_strike: float,
                                  expected_variance: float) -> dict:
    """Approximate volatility-swap strike using a second-order convexity correction.

    This approximation uses Var(V) only through the supplied difference between
    variance strike and expected variance; it is not an exact replication formula.
    """
    kv, ev = float(variance_strike), float(expected_variance)
    if kv < 0 or ev < 0:
        raise ValueError("variance inputs must be non-negative")
    if ev == 0:
        return {"approximate_volatility_strike": 0.0, "convexity_adjustment": 0.0}
    base = sqrt(ev)
    adjustment = (kv-ev)/(8.0*ev**1.5)
    return {"approximate_volatility_strike": base+adjustment,
            "convexity_adjustment": adjustment,
            "method": "second-order approximation; not exact volatility-swap replication"}


def variance_term_structure(maturities: Sequence[float],
                            annualized_variances: Sequence[float]) -> dict:
    """Summarize total-variance monotonicity and adjacent forward variances."""
    t = finite_series(maturities, name="maturities", min_length=2)
    v = finite_series(annualized_variances, name="annualized_variances", min_length=2)
    if len(t) != len(v) or t[0] <= 0 or any(t[i] >= t[i+1] for i in range(len(t)-1)) or any(x < 0 for x in v):
        raise ValueError("maturities must increase and variance inputs must be non-negative and aligned")
    total = [a*b for a, b in zip(t, v)]
    forwards = [forward_variance(t[i], t[i+1], total[i], total[i+1])
                for i in range(len(t)-1)]
    return {"maturities": tuple(t), "annualized_variances": tuple(v),
            "total_variances": tuple(total), "forward_variances": tuple(forwards),
            "nondecreasing_total_variance": all(total[i+1] >= total[i]-1e-10 for i in range(len(total)-1))}
