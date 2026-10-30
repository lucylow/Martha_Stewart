"""Monte Carlo derivatives pricing and stochastic-volatility research tools.

GBM estimators use antithetic paths when requested. Barrier results monitor only
discrete simulation dates; continuous barrier crossing is not corrected unless a
user applies a bridge correction externally. Heston uses full-truncation Euler.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log, sqrt, isfinite
import random
from typing import Sequence

from ..core import mean, variance, normal_ppf


def finite(value: float, name: str = "value") -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def positive(value: float, name: str = "value", allow_zero: bool = False) -> float:
    out = finite(value, name)
    if out < 0 or (out == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return out


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    paths: int
    time_steps: int
    variance_reduction: str
    diagnostics: dict


def _validate_market(spot: float, rate: float, dividend_yield: float,
                     volatility: float, maturity: float, paths: int, steps: int) -> tuple[float, ...]:
    s = positive(spot, "spot")
    r = finite(rate, "rate")
    q = finite(dividend_yield, "dividend_yield")
    sigma = positive(volatility, "volatility", allow_zero=True)
    t = positive(maturity, "maturity")
    if paths < 2 or steps < 1:
        raise ValueError("paths must be >=2 and steps >=1")
    return s, r, q, sigma, t


def _ci(values: Sequence[float], confidence: float, *, antithetic: bool = False) -> tuple[float, float, float, float]:
    """Confidence interval with cluster-robust SE for paired antithetic paths."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    x = list(map(float, values))
    if len(x) < 2:
        raise ValueError("at least two payoff observations are required")
    m = mean(x)
    clusters = [x[i:i+2] for i in range(0, len(x), 2)] if antithetic else [[v] for v in x]
    if len(clusters) < 2:
        raise ValueError("at least two independent path clusters are required")
    # Pair-level centered sums preserve the covariance induced by antithetic
    # pairing. A final unpaired path (when paths is odd) forms a singleton.
    cluster_sums = [sum(v-m for v in cluster) for cluster in clusters]
    se2 = len(clusters)/(len(clusters)-1)*sum(v*v for v in cluster_sums)/(len(x)**2)
    se = sqrt(max(0.0, se2))
    z = normal_ppf(0.5+confidence/2.0)
    return m, se, m-z*se, m+z*se


def simulate_gbm_paths(spot: float, rate: float, dividend_yield: float,
                       volatility: float, maturity: float, *, paths: int = 10000,
                       steps: int = 252, seed: int = 7,
                       antithetic: bool = True) -> tuple[tuple[float, ...], ...]:
    """Simulate risk-neutral GBM paths, including initial spot at index zero."""
    s, r, q, sigma, t = _validate_market(spot, rate, dividend_yield, volatility, maturity, paths, steps)
    rng = random.Random(seed)
    dt = t/steps
    drift = (r-q-0.5*sigma*sigma)*dt
    diffusion = sigma*sqrt(dt)
    out: list[tuple[float, ...]] = []
    generated = 0
    while generated < paths:
        z_path = [rng.gauss(0.0, 1.0) for _ in range(steps)]
        for sign in ((1.0, -1.0) if antithetic and generated+1 < paths else (1.0,)):
            path = [s]
            value = s
            for z in z_path:
                value *= exp(drift + sign*diffusion*z)
                path.append(value)
            out.append(tuple(path))
            generated += 1
            if generated >= paths:
                break
    return tuple(out)


def monte_carlo_european(spot: float, strike: float, rate: float,
                         dividend_yield: float, volatility: float, maturity: float,
                         *, option_type: str = "call", paths: int = 10000,
                         steps: int = 1, seed: int = 7, antithetic: bool = True,
                         confidence: float = 0.95) -> MonteCarloEstimate:
    """Price a European option by risk-neutral GBM terminal-payoff simulation."""
    k = positive(strike, "strike")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    s, r, q, sigma, t = _validate_market(spot, rate, dividend_yield, volatility, maturity, paths, steps)
    samples = simulate_gbm_paths(s, r, q, sigma, t, paths=paths, steps=steps,
                                 seed=seed, antithetic=antithetic)
    discount = exp(-r*t)
    payoffs = [discount*max(0.0, path[-1]-k) if kind == "call"
               else discount*max(0.0, k-path[-1]) for path in samples]
    price, se, lo, hi = _ci(payoffs, confidence, antithetic=antithetic)
    return MonteCarloEstimate(price, se, (lo, hi), paths, steps,
                              "antithetic" if antithetic else "plain",
                              {"option_type": kind, "discount_factor": discount,
                               "risk_neutral_drift": r-q, "seed": seed})


def monte_carlo_asian(spot: float, strike: float, rate: float,
                      dividend_yield: float, volatility: float, maturity: float,
                      *, option_type: str = "call", paths: int = 20000,
                      steps: int = 252, seed: int = 7, antithetic: bool = True,
                      geometric_average: bool = False,
                      confidence: float = 0.95) -> MonteCarloEstimate:
    """Price a discretely monitored average-price Asian option under GBM."""
    k = positive(strike, "strike")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    s, r, q, sigma, t = _validate_market(spot, rate, dividend_yield, volatility, maturity, paths, steps)
    paths_data = simulate_gbm_paths(s, r, q, sigma, t, paths=paths, steps=steps,
                                     seed=seed, antithetic=antithetic)
    discount = exp(-r*t)
    payoffs = []
    for path in paths_data:
        monitored = path[1:]
        avg = exp(sum(log(v) for v in monitored)/len(monitored)) if geometric_average else mean(monitored)
        payoff = max(0.0, avg-k) if kind == "call" else max(0.0, k-avg)
        payoffs.append(discount*payoff)
    price, se, lo, hi = _ci(payoffs, confidence, antithetic=antithetic)
    return MonteCarloEstimate(price, se, (lo, hi), paths, steps,
                              "antithetic" if antithetic else "plain",
                              {"option_type": kind, "average": "geometric" if geometric_average else "arithmetic",
                               "monitoring_dates": steps, "seed": seed})


def monte_carlo_barrier(spot: float, strike: float, barrier: float,
                        rate: float, dividend_yield: float, volatility: float,
                        maturity: float, *, option_type: str = "call",
                        barrier_type: str = "up_out", rebate: float = 0.0,
                        paths: int = 20000, steps: int = 252, seed: int = 7,
                        antithetic: bool = True, confidence: float = 0.95) -> MonteCarloEstimate:
    """Price a discretely monitored single-barrier European option.

    ``barrier_type`` accepts up/down-and-in/out. An untriggered knock-in pays
    its rebate; an activated knock-out receives the rebate instead of payoff.
    The barrier is checked at simulation dates, not continuously between them.
    """
    k = positive(strike, "strike")
    h = positive(barrier, "barrier")
    rb = finite(rebate, "rebate")
    kind = option_type.lower()
    bt = barrier_type.lower()
    if kind not in {"call", "put"} or bt not in {"up_out", "down_out", "up_in", "down_in"} or rb < 0:
        raise ValueError("invalid option type, barrier type or rebate")
    s, r, q, sigma, t = _validate_market(spot, rate, dividend_yield, volatility, maturity, paths, steps)
    paths_data = simulate_gbm_paths(s, r, q, sigma, t, paths=paths, steps=steps,
                                     seed=seed, antithetic=antithetic)
    discount = exp(-r*t)
    payoffs = []
    for path in paths_data:
        hit = (max(path) >= h) if bt.startswith("up") else (min(path) <= h)
        active = hit if bt.endswith("in") else not hit
        terminal = path[-1]
        intrinsic = max(0.0, terminal-k) if kind == "call" else max(0.0, k-terminal)
        payoff = intrinsic if active else rb
        payoffs.append(discount*payoff)
    price, se, lo, hi = _ci(payoffs, confidence, antithetic=antithetic)
    return MonteCarloEstimate(price, se, (lo, hi), paths, steps,
                              "antithetic" if antithetic else "plain",
                              {"option_type": kind, "barrier_type": bt, "barrier": h,
                               "rebate": rb, "monitoring": "discrete", "seed": seed})


def monte_carlo_heston(spot: float, strike: float, rate: float,
                       dividend_yield: float, variance0: float, kappa: float,
                       theta: float, vol_of_vol: float, rho: float, maturity: float,
                       *, option_type: str = "call", paths: int = 20000,
                       steps: int = 252, seed: int = 7,
                       confidence: float = 0.95) -> MonteCarloEstimate:
    """Heston European option via full-truncation Euler simulation.

    The variance state is floored at zero for diffusion and drift evaluation.
    Euler discretization introduces time-step bias; compare results across
    increasing `steps` and use a specialized solver for production valuation.
    """
    s = positive(spot, "spot")
    k = positive(strike, "strike")
    r, q = finite(rate), finite(dividend_yield)
    v0, kap, th, eta = positive(variance0, "variance0", allow_zero=True), positive(kappa, "kappa"), positive(theta, "theta", allow_zero=True), positive(vol_of_vol, "vol_of_vol", allow_zero=True)
    corr = finite(rho)
    t = positive(maturity, "maturity")
    if not -1 <= corr <= 1 or paths < 2 or steps < 1:
        raise ValueError("rho must be in [-1,1]; paths >=2 and steps >=1")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    rng = random.Random(seed)
    dt = t/steps
    root_dt = sqrt(dt)
    orthogonal = sqrt(max(0.0, 1.0-corr*corr))
    discount = exp(-r*t)
    payoffs = []
    for _ in range(paths):
        stock, var = s, v0
        for _step in range(steps):
            z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
            zv = corr*z1 + orthogonal*z2
            vp = max(var, 0.0)
            stock *= exp((r-q-0.5*vp)*dt + sqrt(vp)*root_dt*z1)
            var = var + kap*(th-vp)*dt + eta*sqrt(vp)*root_dt*zv
        intrinsic = max(0.0, stock-k) if kind == "call" else max(0.0, k-stock)
        payoffs.append(discount*intrinsic)
    price, se, lo, hi = _ci(payoffs, confidence)
    return MonteCarloEstimate(price, se, (lo, hi), paths, steps, "full_truncation_euler",
                              {"option_type": kind, "variance0": v0, "kappa": kap,
                               "theta": th, "vol_of_vol": eta, "rho": corr,
                               "feller_condition": 2*kap*th >= eta*eta, "seed": seed})


def common_random_number_greeks(pricer, spot: float, volatility: float, *,
                                spot_bump: float = 0.01, vol_bump: float = 0.001) -> dict:
    """Finite-difference delta/vega estimates using a caller-supplied seeded pricer.

    `pricer(spot, volatility)` must return a scalar and should reuse a fixed seed
    internally. The common-random-number design reduces finite-difference noise.
    """
    s = positive(spot, "spot")
    v = positive(volatility, "volatility")
    ds, dv = positive(spot_bump, "spot_bump"), positive(vol_bump, "vol_bump")
    base = finite(pricer(s, v), "base price")
    up_s = finite(pricer(s+ds, v), "up-spot price")
    dn_s = finite(pricer(max(1e-10, s-ds), v), "down-spot price")
    up_v = finite(pricer(s, v+dv), "up-vol price")
    dn_v = finite(pricer(s, max(1e-10, v-dv)), "down-vol price")
    return {"price": base, "delta": (up_s-dn_s)/(2*ds),
            "gamma": (up_s-2*base+dn_s)/(ds*ds),
            "vega_per_vol_point": (up_v-dn_v)/(2*dv),
            "spot_bump": ds, "volatility_bump": dv,
            "method": "central difference; caller must reuse random seed"}


def antithetic_variance_reduction(samples_plain: Sequence[float],
                                  samples_antithetic: Sequence[float]) -> dict:
    """Compare estimator variances and report empirical reduction fraction."""
    plain = list(map(float, samples_plain))
    anti = list(map(float, samples_antithetic))
    if len(plain) < 2 or len(anti) < 2:
        raise ValueError("at least two observations are required in each sample")
    vp, va = variance(plain), variance(anti)
    return {"plain_variance": vp, "antithetic_variance": va,
            "variance_reduction_fraction": 1.0-va/vp if vp > 0 else 0.0,
            "plain_mean": mean(plain), "antithetic_mean": mean(anti),
            "note": "variance comparison is empirical and depends on paired sampling"}
