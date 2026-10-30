"""Stochastic-process simulators and Monte Carlo derivative pricing."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from .core import FinanceInputError, Number, finite, positive


def simulate_gbm(spot: Number, drift: Number, volatility: Number,
                 horizon: Number, steps: int, paths: int = 1,
                 *, seed: int = 7, antithetic: bool = False) -> tuple[tuple[float, ...], ...]:
    """Simulate geometric Brownian motion paths with exact time-step solution."""
    s, mu, sigma, t = positive(spot), finite(drift), positive(volatility, allow_zero=True), positive(horizon)
    if not isinstance(steps, int) or steps < 1 or not isinstance(paths, int) or paths < 1:
        raise FinanceInputError("steps and paths must be positive integers")
    rng = random.Random(seed)
    dt = t/steps
    drift_step = (mu-0.5*sigma*sigma)*dt
    vol_step = sigma*math.sqrt(dt)
    output = []
    for path_index in range(paths):
        current = s
        path = [s]
        for _ in range(steps):
            z = rng.gauss(0, 1)
            if antithetic and path_index % 2 == 1:
                z = -z
            current *= math.exp(drift_step+vol_step*z)
            path.append(current)
        output.append(tuple(path))
    return tuple(output)


def simulate_arithmetic_brownian(initial: Number, drift: Number,
                                 volatility: Number, horizon: Number,
                                 steps: int, paths: int = 1,
                                 *, seed: int = 7) -> tuple[tuple[float, ...], ...]:
    """Simulate arithmetic Brownian motion with Euler-exact increments."""
    x0, mu, sigma, t = map(finite, (initial, drift, volatility, horizon))
    if sigma < 0 or t <= 0 or steps < 1 or paths < 1:
        raise FinanceInputError("invalid Brownian simulation parameters")
    rng, dt = random.Random(seed), t/steps
    output = []
    for _ in range(paths):
        x, path = x0, [x0]
        for _ in range(steps):
            x += mu*dt+sigma*math.sqrt(dt)*rng.gauss(0, 1)
            path.append(x)
        output.append(tuple(path))
    return tuple(output)


def simulate_ou(initial: Number, mean_level: Number, speed: Number,
                volatility: Number, horizon: Number, steps: int,
                paths: int = 1, *, seed: int = 7) -> tuple[tuple[float, ...], ...]:
    """Simulate exact OU transitions on a regular grid."""
    x0, theta, kappa, sigma, t = map(finite, (initial, mean_level, speed, volatility, horizon))
    if kappa <= 0 or sigma < 0 or t <= 0 or steps < 1 or paths < 1:
        raise FinanceInputError("invalid OU simulation parameters")
    dt = t/steps
    decay = math.exp(-kappa*dt)
    shock = sigma*math.sqrt((1-decay*decay)/(2*kappa))
    rng = random.Random(seed)
    output = []
    for _ in range(paths):
        x, path = x0, [x0]
        for _ in range(steps):
            x = theta+(x-theta)*decay+shock*rng.gauss(0, 1)
            path.append(x)
        output.append(tuple(path))
    return tuple(output)


def simulate_vasicek(initial_rate: Number, mean_reversion: Number,
                     long_run_rate: Number, volatility: Number,
                     horizon: Number, steps: int, paths: int = 1,
                     *, seed: int = 7) -> tuple[tuple[float, ...], ...]:
    """Simulate Vasicek short rates using exact Gaussian transitions."""
    return simulate_ou(initial_rate, long_run_rate, mean_reversion,
                       volatility, horizon, steps, paths, seed=seed)


def simulate_cir(initial_rate: Number, mean_reversion: Number,
                 long_run_rate: Number, volatility: Number,
                 horizon: Number, steps: int, paths: int = 1,
                 *, seed: int = 7) -> tuple[tuple[float, ...], ...]:
    """Simulate CIR short rates with full-truncation Euler discretization."""
    r0, kappa, theta, sigma, t = map(finite, (initial_rate, mean_reversion,
                                              long_run_rate, volatility, horizon))
    if min(r0, kappa, theta, sigma) < 0 or t <= 0 or steps < 1 or paths < 1:
        raise FinanceInputError("invalid CIR inputs")
    rng, dt = random.Random(seed), t/steps
    output = []
    for _ in range(paths):
        r, path = r0, [r0]
        for _ in range(steps):
            root = math.sqrt(max(0.0, r))
            r = r+kappa*(theta-max(0.0, r))*dt+sigma*root*math.sqrt(dt)*rng.gauss(0, 1)
            r = max(0.0, r)
            path.append(r)
        output.append(tuple(path))
    return tuple(output)


def simulate_heston(spot: Number, initial_variance: Number,
                    drift: Number, mean_reversion: Number,
                    long_run_variance: Number, vol_of_vol: Number,
                    correlation: Number, horizon: Number,
                    steps: int, paths: int = 1,
                    *, seed: int = 7) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Full-truncation Euler Heston paths with correlated Brownian shocks.

    Returns (spot_path, variance_path) for each simulation. Discretization bias
    can be material, especially near zero variance or at coarse time steps.
    """
    s0, v0, mu, kappa, theta, xi, rho, t = map(finite, (spot, initial_variance,
        drift, mean_reversion, long_run_variance, vol_of_vol, correlation, horizon))
    if s0 <= 0 or v0 < 0 or kappa < 0 or theta < 0 or xi < 0 or abs(rho) > 1 or t <= 0 or steps < 1 or paths < 1:
        raise FinanceInputError("invalid Heston parameters")
    rng, dt = random.Random(seed), t/steps
    sqrt_dt, orthogonal = math.sqrt(dt), math.sqrt(max(0.0, 1-rho*rho))
    output = []
    for _ in range(paths):
        s, v, sp, vp = s0, v0, [s0], [v0]
        for _ in range(steps):
            z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
            zv = rho*z1+orthogonal*z2
            v_pos = max(v, 0.0)
            s *= math.exp((mu-0.5*v_pos)*dt+math.sqrt(v_pos)*sqrt_dt*z1)
            v = max(0.0, v+kappa*(theta-v_pos)*dt+xi*math.sqrt(v_pos)*sqrt_dt*zv)
            sp.append(s); vp.append(v)
        output.append((tuple(sp), tuple(vp)))
    return tuple(output)


def monte_carlo_european_option(spot: Number, strike: Number,
                                rate: Number, volatility: Number,
                                maturity: Number, *, option_type: str = "call",
                                dividend_yield: Number = 0.0,
                                simulations: int = 100000, seed: int = 7,
                                antithetic: bool = True) -> dict[str, float]:
    """Price European vanilla option under risk-neutral GBM with MC standard error."""
    s, k, r, sigma, t, q = (positive(spot), positive(strike), finite(rate),
                            positive(volatility), positive(maturity), finite(dividend_yield))
    kind = option_type.lower()
    if kind not in {"call", "put"} or not isinstance(simulations, int) or simulations < 2:
        raise FinanceInputError("invalid option type or simulation count")
    rng = random.Random(seed)
    count = simulations//2 if antithetic else simulations
    payoffs = []
    drift = (r-q-0.5*sigma*sigma)*t
    diffusion = sigma*math.sqrt(t)
    for _ in range(count):
        z = rng.gauss(0, 1)
        terminal_up = s*math.exp(drift+diffusion*z)
        values = [max(0.0, terminal_up-k) if kind == "call" else max(0.0, k-terminal_up)]
        if antithetic:
            terminal_down = s*math.exp(drift-diffusion*z)
            values.append(max(0.0, terminal_down-k) if kind == "call" else max(0.0, k-terminal_down))
        payoffs.append(sum(values)/len(values))
    estimate = sum(payoffs)/len(payoffs)*math.exp(-r*t)
    sample_var = sum((p-sum(payoffs)/len(payoffs))**2 for p in payoffs)/(len(payoffs)-1)
    se = math.sqrt(sample_var/len(payoffs))*math.exp(-r*t)
    return {"price": estimate, "standard_error": se,
            "confidence_95_low": estimate-1.96*se,
            "confidence_95_high": estimate+1.96*se,
            "simulation_batches": float(len(payoffs)), "antithetic": float(antithetic)}


def monte_carlo_asian_call(spot: Number, strike: Number,
                           rate: Number, volatility: Number,
                           maturity: Number, *, observations: int = 52,
                           simulations: int = 20000,
                           dividend_yield: Number = 0.0,
                           seed: int = 7) -> dict[str, float]:
    """Price arithmetic-average Asian call by risk-neutral GBM Monte Carlo."""
    s, k, r, sigma, t, q = (positive(spot), positive(strike), finite(rate),
                            positive(volatility), positive(maturity), finite(dividend_yield))
    if observations < 1 or simulations < 2:
        raise FinanceInputError("observations >= 1 and simulations >= 2 required")
    rng, dt = random.Random(seed), t/observations
    batch = []
    for _ in range(simulations):
        current, total = s, s
        for _ in range(observations):
            current *= math.exp((r-q-0.5*sigma*sigma)*dt+sigma*math.sqrt(dt)*rng.gauss(0, 1))
            total += current
        average = total/(observations+1)
        batch.append(max(average-k, 0.0))
    discount = math.exp(-r*t)
    avg = sum(batch)/simulations
    se = math.sqrt(sum((x-avg)**2 for x in batch)/(simulations-1)/simulations)
    return {"price": avg*discount, "standard_error": se*discount,
            "observations_per_path": float(observations), "simulations": float(simulations)}


def monte_carlo_control_variate(payoffs: Sequence[Number],
                                control_payoffs: Sequence[Number],
                                known_control_value: Number) -> dict[str, float]:
    """Estimate a payoff mean with a linear control-variate adjustment."""
    y, x = tuple(map(finite, payoffs)), tuple(map(finite, control_payoffs))
    if len(y) != len(x) or len(y) < 3:
        raise FinanceInputError("payoff samples must align and contain at least three observations")
    my, mx = sum(y)/len(y), sum(x)/len(x)
    vx = sum((v-mx)**2 for v in x)/(len(x)-1)
    if vx <= 0:
        raise FinanceInputError("control variate has zero variance")
    cov = sum((a-my)*(b-mx) for a,b in zip(y,x))/(len(y)-1)
    beta = cov/vx
    adjusted = [a-beta*(b-finite(known_control_value)) for a,b in zip(y,x)]
    estimate = sum(adjusted)/len(adjusted)
    se = math.sqrt(sum((v-estimate)**2 for v in adjusted)/(len(adjusted)-1)/len(adjusted))
    return {"estimate": estimate, "standard_error": se,
            "control_coefficient": beta}
