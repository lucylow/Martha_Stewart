"""Benchmark, factor, and allocation attribution utilities."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RelativePerformance:
    observations: int
    active_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    correlation: float
    up_capture: float
    down_capture: float
    hit_rate: float


def _paired(portfolio: Sequence[float], benchmark: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(portfolio) != len(benchmark):
        raise ValueError("portfolio and benchmark series must be aligned and equal length")
    if any(not isfinite(float(x)) for x in list(portfolio) + list(benchmark)):
        raise ValueError("return series must be finite")
    return [float(x) for x in portfolio], [float(x) for x in benchmark]


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / (len(values) - 1)


def _covariance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)


def relative_performance(portfolio_returns: Sequence[float], benchmark_returns: Sequence[float],
                         periods_per_year: float = 252.0) -> RelativePerformance:
    p, b = _paired(portfolio_returns, benchmark_returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    active = [x - y for x, y in zip(p, b)]
    active_mean = mean(active) if active else 0.0
    te = sqrt(_sample_variance(active)) * sqrt(periods_per_year)
    var_b = _sample_variance(b)
    cov_pb = _covariance(p, b)
    corr_denom = sqrt(_sample_variance(p) * var_b)
    up = [i for i, x in enumerate(b) if x > 0]
    down = [i for i, x in enumerate(b) if x < 0]
    def capture(indices: list[int]) -> float:
        if not indices or abs(mean([b[i] for i in indices])) < 1e-15:
            return 0.0
        # Arithmetic conditional mean capture is easy to interpret and stable in small samples.
        return mean([p[i] for i in indices]) / mean([b[i] for i in indices])
    return RelativePerformance(
        observations=len(p), active_return=sum(active), tracking_error=te,
        information_ratio=(active_mean * periods_per_year / te if te > 0 else 0.0),
        beta=(cov_pb / var_b if var_b > 0 else 0.0),
        correlation=(cov_pb / corr_denom if corr_denom > 0 else 0.0),
        up_capture=capture(up), down_capture=capture(down),
        hit_rate=sum(1 for x in active if x > 0) / len(active) if active else 0.0,
    )


def arithmetic_linked_contribution(contributions: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Sum per-period additive contributions; useful for arithmetic attribution."""
    result: dict[str, float] = {}
    lengths = {len(values) for values in contributions.values()}
    if len(lengths) > 1:
        raise ValueError("contribution series must have equal lengths")
    for name, values in contributions.items():
        if any(not isfinite(float(x)) for x in values):
            raise ValueError(f"contribution {name} contains non-finite values")
        result[name] = sum(float(x) for x in values)
    return result


def brinson_fachler(allocations: Mapping[str, float], portfolio_returns: Mapping[str, float],
                    benchmark_weights: Mapping[str, float], benchmark_returns: Mapping[str, float],
                    benchmark_total_return: float | None = None) -> dict[str, dict[str, float]]:
    """Single-period Brinson-Fachler allocation, selection and interaction.

    Inputs are sector weights as fractions and sector returns as arithmetic returns.
    Sector universes are unioned; absent weights/returns are treated as zero.
    """
    sectors = sorted(set(allocations) | set(portfolio_returns) | set(benchmark_weights) | set(benchmark_returns))
    if not sectors:
        return {}
    for name, mapping in (("allocations", allocations), ("portfolio_returns", portfolio_returns),
                          ("benchmark_weights", benchmark_weights), ("benchmark_returns", benchmark_returns)):
        if any(not isfinite(float(v)) for _, v in mapping.items()):
            raise ValueError(f"{name} contains non-finite values")
    r_b = benchmark_total_return
    if r_b is None:
        r_b = sum(float(benchmark_weights.get(s, 0.0)) * float(benchmark_returns.get(s, 0.0)) for s in sectors)
    result: dict[str, dict[str, float]] = {}
    for sector in sectors:
        wp = float(allocations.get(sector, 0.0))
        wb = float(benchmark_weights.get(sector, 0.0))
        rp = float(portfolio_returns.get(sector, 0.0))
        rb = float(benchmark_returns.get(sector, 0.0))
        result[sector] = {
            "allocation": (wp - wb) * (rb - float(r_b)),
            "selection": wb * (rp - rb),
            "interaction": (wp - wb) * (rp - rb),
        }
    return result


def factor_regression(portfolio_returns: Sequence[float], factors: Mapping[str, Sequence[float]],
                      ridge: float = 0.0) -> dict[str, object]:
    """Small-factor ordinary least squares using a guarded normal-equation solver.

    Ridge regularization is available to stabilize highly collinear factors; the
    intercept is not penalized. This is diagnostic research, not inference-grade
    econometrics for arbitrary ill-conditioned designs.
    """
    names = sorted(factors)
    y = [float(x) for x in portfolio_returns]
    if not y or len(y) < len(names) + 2:
        raise ValueError("insufficient observations for factor regression")
    if ridge < 0 or not isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if any(len(factors[name]) != len(y) for name in names):
        raise ValueError("all factors must align with portfolio returns")
    if any(not isfinite(x) for x in y) or any(not isfinite(float(x)) for n in names for x in factors[n]):
        raise ValueError("regression data must be finite")
    x = [[1.0] + [float(factors[name][i]) for name in names] for i in range(len(y))]
    cols = len(names) + 1
    xtx = [[sum(row[a] * row[b] for row in x) + (ridge if a == b and a else 0.0)
            for b in range(cols)] for a in range(cols)]
    xty = [sum(row[a] * target for row, target in zip(x, y)) for a in range(cols)]
    beta = _solve_linear(xtx, xty)
    predicted = [sum(row[j] * beta[j] for j in range(cols)) for row in x]
    residuals = [actual - fitted for actual, fitted in zip(y, predicted)]
    sse = sum(e * e for e in residuals)
    ybar = mean(y)
    sst = sum((value - ybar) ** 2 for value in y)
    return {"alpha_per_period": beta[0], "betas": dict(zip(names, beta[1:])),
            "r_squared": 1.0 - sse / sst if sst > 0 else 0.0,
            "residual_volatility": sqrt(_sample_variance(residuals)),
            "observations": len(y), "ridge": ridge, "residuals": residuals}


def _solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; raises on a singular design."""
    n = len(rhs)
    a = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("factor design is singular; add ridge or remove collinear factors")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [a[row][j] - factor * a[col][j] for j in range(n + 1)]
    return [a[i][-1] for i in range(n)]
