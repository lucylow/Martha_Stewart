"""Cross-sectional factor research, score construction and attribution."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .core import FinanceInputError, Number, Vector, finite, mean, vector
from .regression import ols
from .statistics import mad, median, rank, winsorize, zscore


def cross_sectional_zscores(values: Mapping[str, Number], *, robust: bool = False) -> dict[str, float]:
    """Standardize a cross-section while preserving asset identifiers."""
    if len(values) < 2:
        raise FinanceInputError("at least two assets are required")
    keys = list(values)
    scores = zscore([values[k] for k in keys], robust=robust)
    return dict(zip(keys, scores))


def winsorized_zscores(values: Mapping[str, Number], lower_q: Number = 0.02,
                       upper_q: Number = 0.98) -> dict[str, float]:
    """Winsorize a cross section and then z-score it."""
    keys = list(values)
    clipped = winsorize([values[k] for k in keys], lower_q, upper_q)
    scores = zscore(clipped)
    return dict(zip(keys, scores))


def percentile_scores(values: Mapping[str, Number], *, higher_is_better: bool = True) -> dict[str, float]:
    """Map cross-sectional ranks to [0, 1], averaging tied ranks."""
    if len(values) < 2:
        raise FinanceInputError("at least two assets are required")
    keys = list(values)
    ranks = rank([values[k] for k in keys], ascending=higher_is_better, pct=True)
    if not higher_is_better:
        ranks = tuple(1-r for r in ranks)
    return dict(zip(keys, ranks))


def composite_score(factors: Mapping[str, Mapping[str, Number]],
                    weights: Mapping[str, Number], *, robust: bool = True) -> dict[str, float]:
    """Build a weighted standardized score from named cross-sectional factors.

    All factor maps must contain the same asset keys. Positive weights reward
    larger standardized values; reverse direction by using a negative weight.
    """
    if not factors or set(factors) != set(weights):
        raise FinanceInputError("factor names and weight names must match")
    assets = set(next(iter(factors.values())))
    if not assets or any(set(values) != assets for values in factors.values()):
        raise FinanceInputError("all factor maps must contain identical asset keys")
    total_weight = math.fsum(abs(finite(w, f"weights[{name}]")) for name, w in weights.items())
    if total_weight == 0:
        raise FinanceInputError("at least one factor weight must be non-zero")
    normalized = {name: finite(w)/total_weight for name, w in weights.items()}
    scores = {asset: 0.0 for asset in assets}
    for name, values in factors.items():
        standardized = cross_sectional_zscores(values, robust=robust)
        for asset in assets:
            scores[asset] += normalized[name]*standardized[asset]
    return scores


def information_coefficient(signal: Vector, forward_returns: Vector) -> float:
    """Pearson information coefficient for a single cross-section."""
    from .core import correlation
    return correlation(signal, forward_returns)


def rank_information_coefficient(signal: Vector, forward_returns: Vector) -> float:
    """Spearman rank IC, using average ranks for ties."""
    from .core import correlation
    x, y = vector(signal), vector(forward_returns)
    if len(x) != len(y):
        raise FinanceInputError("signal and forward returns must align")
    return correlation(rank(x), rank(y))


def factor_exposures(asset_returns: Vector, factor_returns: Mapping[str, Vector],
                     *, intercept: bool = True) -> dict[str, float]:
    """Estimate factor loadings for one asset using time-series OLS."""
    names = list(factor_returns)
    if not names:
        raise FinanceInputError("at least one factor series is required")
    result = ols(asset_returns, [factor_returns[n] for n in names], intercept=intercept)
    output: dict[str, float] = {}
    offset = 0
    if intercept:
        output["alpha"] = result.coefficients[0]
        offset = 1
    output.update({name: result.coefficients[i+offset] for i, name in enumerate(names)})
    output["r_squared"] = result.r_squared
    return output


def factor_attribution(portfolio_returns: Vector, factor_returns: Mapping[str, Vector]) -> dict[str, float]:
    """Estimate realized factor contribution using fitted loadings and mean returns."""
    names = list(factor_returns)
    result = ols(portfolio_returns, [factor_returns[n] for n in names], intercept=True)
    contributions = {name: result.coefficients[i+1]*mean(factor_returns[name])
                     for i, name in enumerate(names)}
    contributions["alpha"] = result.coefficients[0]
    contributions["realized_mean"] = mean(portfolio_returns)
    contributions["residual_mean"] = mean(result.residuals)
    contributions["r_squared"] = result.r_squared
    return contributions


def rolling_information_coefficient(signal_history: Sequence[Vector],
                                    return_history: Sequence[Vector]) -> tuple[float, ...]:
    """Calculate cross-sectional IC at each timestamp from aligned snapshots."""
    if len(signal_history) != len(return_history) or not signal_history:
        raise FinanceInputError("signal and return histories must align")
    return tuple(information_coefficient(x, y) for x, y in zip(signal_history, return_history))


def ic_summary(ic_series: Vector) -> dict[str, float]:
    """Summarize mean IC, IC volatility, t-statistic and positive fraction."""
    data = vector(ic_series, min_length=2)
    avg = mean(data)
    sd = math.sqrt(sum((x-avg)**2 for x in data)/(len(data)-1))
    return {"mean_ic": avg, "ic_volatility": sd,
            "ic_t_stat": avg/(sd/math.sqrt(len(data))) if sd else math.inf,
            "positive_fraction": sum(x > 0 for x in data)/len(data)}


def quantile_buckets(values: Mapping[str, Number], buckets: int = 5,
                     *, higher_is_better: bool = True) -> dict[int, tuple[str, ...]]:
    """Assign assets to approximately equal-sized factor buckets."""
    if not isinstance(buckets, int) or not 2 <= buckets <= len(values):
        raise FinanceInputError("buckets must be between 2 and number of assets")
    keys = sorted(values, key=lambda key: (finite(values[key]), key), reverse=higher_is_better)
    result = {i: [] for i in range(buckets)}
    for position, key in enumerate(keys):
        bucket = min(buckets-1, position*buckets//len(keys))
        result[bucket].append(key)
    return {key: tuple(items) for key, items in result.items()}


def factor_spread_returns(values: Mapping[str, Number], next_period_returns: Mapping[str, Number],
                          buckets: int = 5) -> dict[str, float]:
    """Compute equal-weight top, bottom and long-short bucket returns."""
    if set(values) != set(next_period_returns):
        raise FinanceInputError("factor values and returns must have identical assets")
    groups = quantile_buckets(values, buckets, higher_is_better=True)
    means = {bucket: mean([next_period_returns[k] for k in assets])
             for bucket, assets in groups.items()}
    return {"top": means[0], "bottom": means[buckets-1],
            "long_short": means[0]-means[buckets-1]}


def neutralize_by_groups(values: Mapping[str, Number], groups: Mapping[str, str]) -> dict[str, float]:
    """Remove group means, preserving each asset's within-group deviation."""
    if set(values) != set(groups):
        raise FinanceInputError("values and groups must have identical assets")
    grouped: dict[str, list[float]] = {}
    for asset, group in groups.items():
        grouped.setdefault(group, []).append(finite(values[asset]))
    centers = {group: mean(sample) for group, sample in grouped.items()}
    return {asset: finite(values[asset])-centers[group] for asset, group in groups.items()}


def neutralize_against_controls(values: Vector, controls: Sequence[Vector]) -> tuple[float, ...]:
    """Return residuals from cross-sectional OLS on industry/size or other controls."""
    if not controls:
        raise FinanceInputError("at least one control is required")
    return ols(values, controls, intercept=True).residuals


def factor_turnover(previous: Mapping[str, Number], current: Mapping[str, Number]) -> float:
    """One-way turnover of two non-negative normalized portfolio weight maps."""
    keys = set(previous) | set(current)
    old = {k: finite(previous.get(k, 0.0)) for k in keys}
    new = {k: finite(current.get(k, 0.0)) for k in keys}
    if any(v < 0 for v in (*old.values(), *new.values())):
        raise FinanceInputError("weights must be non-negative")
    if abs(sum(old.values())-1.0) > 1e-6 or abs(sum(new.values())-1.0) > 1e-6:
        raise FinanceInputError("long-only weights must sum to one")
    return 0.5*sum(abs(new[k]-old[k]) for k in keys)


def factor_decay(ic_by_horizon: Mapping[int, Number]) -> dict[int, float]:
    """Normalize absolute IC by the first supplied horizon for decay analysis."""
    if not ic_by_horizon:
        raise FinanceInputError("ic_by_horizon cannot be empty")
    ordered = sorted(ic_by_horizon)
    base = abs(finite(ic_by_horizon[ordered[0]]))
    if base == 0:
        raise FinanceInputError("first horizon IC must be non-zero")
    return {horizon: abs(finite(ic_by_horizon[horizon]))/base for horizon in ordered}
