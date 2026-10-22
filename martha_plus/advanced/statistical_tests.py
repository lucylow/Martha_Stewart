"""Resampling and multiple-testing diagnostics for strategy research."""
from __future__ import annotations
from math import erf, exp, isfinite, log, sqrt
from statistics import mean, median
from random import Random
from typing import Sequence


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _validate(values: Sequence[float], min_count: int = 1) -> list[float]:
    result = [float(x) for x in values]
    if len(result) < min_count:
        raise ValueError(f"at least {min_count} observations are required")
    if any(not isfinite(x) for x in result):
        raise ValueError("observations must be finite")
    return result


def percentile(values: Sequence[float], q: float) -> float:
    data = sorted(_validate(values))
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    if len(data) == 1:
        return data[0]
    index = (len(data) - 1) * q / 100
    lower = int(index)
    upper = min(lower + 1, len(data) - 1)
    frac = index - lower
    return data[lower] * (1 - frac) + data[upper] * frac


def bootstrap_mean(values: Sequence[float], *, repetitions: int = 2000, seed: int = 7,
                   confidence: float = 0.95) -> dict[str, float | int]:
    data = _validate(values, 2)
    if repetitions < 100:
        raise ValueError("repetitions must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = Random(seed)
    estimates = [mean(rng.choices(data, k=len(data))) for _ in range(repetitions)]
    alpha = (1 - confidence) * 100
    return {"mean": mean(data), "lower": percentile(estimates, alpha / 2),
            "upper": percentile(estimates, 100 - alpha / 2), "confidence": confidence,
            "repetitions": repetitions, "seed": seed}


def block_bootstrap_mean(values: Sequence[float], *, block_size: int = 5,
                         repetitions: int = 2000, seed: int = 7,
                         confidence: float = 0.95) -> dict[str, float | int]:
    """Circular block bootstrap preserving some local serial dependence."""
    data = _validate(values, 2)
    if not 1 <= block_size <= len(data):
        raise ValueError("block_size must be in [1, sample length]")
    if repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("invalid repetitions or confidence")
    rng = Random(seed)
    estimates: list[float] = []
    n = len(data)
    for _ in range(repetitions):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(data[(start + j) % n] for j in range(block_size))
        estimates.append(mean(sample[:n]))
    alpha = (1 - confidence) * 100
    return {"mean": mean(data), "lower": percentile(estimates, alpha / 2),
            "upper": percentile(estimates, 100 - alpha / 2), "confidence": confidence,
            "block_size": block_size, "repetitions": repetitions, "seed": seed}


def probabilistic_sharpe_ratio(returns: Sequence[float], benchmark_sharpe: float = 0.0,
                               periods_per_year: float = 252.0) -> float:
    """Approximate PSR using sample skewness/kurtosis and the Sharpe standard error."""
    data = _validate(returns, 3)
    if periods_per_year <= 0 or not isfinite(benchmark_sharpe):
        raise ValueError("invalid benchmark Sharpe or annualization")
    avg = mean(data)
    centered = [x - avg for x in data]
    m2 = mean([x * x for x in centered])
    if m2 <= 0:
        return 0.5 if benchmark_sharpe == 0 else float(benchmark_sharpe < 0)
    m3 = mean([x ** 3 for x in centered])
    m4 = mean([x ** 4 for x in centered])
    skew = m3 / m2 ** 1.5
    kurtosis = m4 / (m2 * m2)
    sr = avg / sqrt(m2)
    threshold = benchmark_sharpe / sqrt(periods_per_year)
    variance_sr = max(1e-15, (1 - skew * sr + ((kurtosis - 1) / 4) * sr * sr) / (len(data) - 1))
    return _normal_cdf((sr - threshold) / sqrt(variance_sr))


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> dict[str, object]:
    """False-discovery-rate flags and adjusted p-values in original input order."""
    values = [float(p) for p in p_values]
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    if any(not isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("p-values must be finite and between zero and one")
    n = len(values)
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [1.0] * n
    running = 1.0
    for rank_from_end in range(n, 0, -1):
        index, p = ordered[rank_from_end - 1]
        running = min(running, p * n / rank_from_end)
        adjusted[index] = min(1.0, running)
    critical_rank = max((rank for rank, (_, p) in enumerate(ordered, start=1) if p <= q * rank / n), default=0)
    discoveries = [False] * n
    for index, _ in ordered[:critical_rank]:
        discoveries[index] = True
    return {"discoveries": discoveries, "adjusted_p_values": adjusted,
            "critical_rank": critical_rank, "q": q, "tests": n}


def deflated_sharpe_probability(observed_sharpe: float, trial_count: int,
                                sample_size: int, sharpe_std: float = 1.0) -> float:
    """Approximate probability a Sharpe exceeds a multiple-testing noise hurdle.

    This simplified diagnostic assumes approximately independent trials. It is
    intentionally conservative only when its assumptions are reasonable; it is
    not a replacement for holdout data or a complete deflated-Sharpe treatment.
    """
    if not isfinite(observed_sharpe) or trial_count < 1 or sample_size < 3 or sharpe_std <= 0:
        raise ValueError("invalid deflated Sharpe inputs")
    if trial_count == 1:
        hurdle = 0.0
    else:
        # Extreme-value approximation for the maximum of independent standard normals.
        z = sqrt(2 * log(trial_count))
        correction = (log(log(trial_count)) + log(4 * 3.141592653589793)) / (2 * z) if trial_count > 2 else 0.0
        hurdle = sharpe_std * max(0.0, z - correction) / sqrt(sample_size)
    se = sharpe_std / sqrt(sample_size)
    return _normal_cdf((observed_sharpe - hurdle) / se)


def permutation_p_value(a: Sequence[float], b: Sequence[float], *, repetitions: int = 5000,
                        seed: int = 11, alternative: str = "two-sided") -> float:
    """Permutation test for a difference in means; samples are exchangeable under H0."""
    left, right = _validate(a, 1), _validate(b, 1)
    if repetitions < 100:
        raise ValueError("repetitions must be >= 100")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid alternative")
    observed = mean(left) - mean(right)
    pooled = left + right
    n_left = len(left)
    rng = Random(seed)
    extreme = 0
    for _ in range(repetitions):
        shuffled = list(pooled)
        rng.shuffle(shuffled)
        diff = mean(shuffled[:n_left]) - mean(shuffled[n_left:])
        if alternative == "greater":
            extreme += diff >= observed
        elif alternative == "less":
            extreme += diff <= observed
        else:
            extreme += abs(diff) >= abs(observed)
    return (extreme + 1) / (repetitions + 1)


def rolling_mean(values: Sequence[float], window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    output: list[float | None] = []
    running = 0.0
    for i, value in enumerate(values):
        value = float(value)
        if not isfinite(value):
            raise ValueError("values must be finite")
        running += value
        if i >= window:
            running -= float(values[i - window])
        output.append(running / window if i + 1 >= window else None)
    return output


def robust_scale(values: Sequence[float], minimum_scale: float = 1e-12) -> dict[str, object]:
    data = _validate(values, 1)
    center = median(data)
    deviations = [abs(x - center) for x in data]
    mad = median(deviations)
    scale = max(1.4826 * mad, minimum_scale)
    return {"center": center, "scale": scale, "scaled": [(x - center) / scale for x in data],
            "degenerate": mad < minimum_scale}
