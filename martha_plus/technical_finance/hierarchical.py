"""Hierarchical risk parity, correlation clustering and allocation diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import FinanceInputError, Number, finite, variance, vector
from .portfolio import covariance_matrix, portfolio_variance
from .regression import correlation_matrix


@dataclass(frozen=True)
class ClusterMerge:
    left: tuple[int, ...]
    right: tuple[int, ...]
    distance: float
    merged: tuple[int, ...]


def correlation_distance(correlation: Sequence[Sequence[Number]]) -> tuple[tuple[float, ...], ...]:
    """Convert correlation into the standard Euclidean distance proxy."""
    corr = tuple(tuple(finite(v) for v in row) for row in correlation)
    n = len(corr)
    if n == 0 or any(len(row) != n for row in corr):
        raise FinanceInputError("correlation matrix must be square and non-empty")
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            value = corr[i][j]
            if not -1 <= value <= 1:
                raise FinanceInputError("correlations must lie in [-1,1]")
            if abs(value-corr[j][i]) > 1e-8:
                raise FinanceInputError("correlation matrix must be symmetric")
            row.append(math.sqrt(max(0.0, 0.5*(1-value))))
        result.append(tuple(row))
    return tuple(result)


def single_linkage(distance: Sequence[Sequence[Number]]) -> tuple[ClusterMerge, ...]:
    """Agglomerative single-linkage clustering with deterministic tie breaking."""
    matrix = tuple(tuple(finite(v) for v in row) for row in distance)
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise FinanceInputError("distance matrix must be square and non-empty")
    if any(matrix[i][j] < 0 for i in range(n) for j in range(n)):
        raise FinanceInputError("distances must be non-negative")
    if any(abs(matrix[i][j]-matrix[j][i]) > 1e-8 for i in range(n) for j in range(n)):
        raise FinanceInputError("distance matrix must be symmetric")
    clusters = [(i,) for i in range(n)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                distance_value = min(matrix[a][b] for a in clusters[i] for b in clusters[j])
                candidate = (distance_value, clusters[i], clusters[j], i, j)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        dist, left, right, i, j = best
        merged = tuple(sorted(left+right))
        merges.append(ClusterMerge(left, right, dist, merged))
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in {i,j}]
        clusters.append(merged)
        clusters.sort(key=lambda value: value[0])
    return tuple(merges)


def quasi_diagonal_order(merges: Sequence[ClusterMerge], asset_count: int) -> tuple[int, ...]:
    """Return a leaf order consistent with a supplied agglomerative merge tree.

    The order follows merge chronology, retaining left/right subcluster order
    from the deterministic single-linkage construction.
    """
    if asset_count < 1:
        raise FinanceInputError("asset_count must be positive")
    order = [(i,) for i in range(asset_count)]
    for merge in merges:
        left_positions = [i for i, group in enumerate(order) if set(group).issubset(merge.left)]
        right_positions = [i for i, group in enumerate(order) if set(group).issubset(merge.right)]
        if left_positions and right_positions:
            left = tuple(x for i in left_positions for x in order[i])
            right = tuple(x for i in right_positions for x in order[i])
            combined = left+right
            remove = set(left_positions+right_positions)
            order = [group for i, group in enumerate(order) if i not in remove]
            order.append(combined)
    flattened = tuple(x for group in order for x in group)
    if len(flattened) != asset_count or set(flattened) != set(range(asset_count)):
        # Fallback to deterministic asset order if caller supplied inconsistent merges.
        return tuple(range(asset_count))
    return flattened


def inverse_variance_cluster_weights(covariance: Sequence[Sequence[Number]],
                                     cluster: Sequence[int]) -> tuple[float, ...]:
    """Compute normalized inverse-variance weights for one cluster."""
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    members = tuple(cluster)
    if not members or any(i < 0 or i >= len(sigma) for i in members):
        raise FinanceInputError("cluster has invalid asset indices")
    variances = [sigma[i][i] for i in members]
    if any(v <= 0 for v in variances):
        raise FinanceInputError("cluster variances must be positive")
    inv = [1/v for v in variances]
    total = sum(inv)
    return tuple(v/total for v in inv)


def cluster_variance(covariance: Sequence[Sequence[Number]],
                     cluster: Sequence[int]) -> float:
    """Variance of an inverse-variance-weighted cluster portfolio."""
    members = tuple(cluster)
    weights = inverse_variance_cluster_weights(covariance, members)
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    sub = [[sigma[i][j] for j in members] for i in members]
    return portfolio_variance(weights, sub)


def hierarchical_risk_parity(returns: Sequence[Sequence[Number]]) -> tuple[float, ...]:
    """Compute long-only HRP weights from return columns.

    This implementation uses single-linkage correlation clustering and
    recursive bisection with inverse-variance cluster risk. It has no SciPy or
    NumPy dependency and is suitable for modest research universes.
    """
    columns = [vector(col, f"returns[{i}]", min_length=3) for i, col in enumerate(returns)]
    if not columns or any(len(col) != len(columns[0]) for col in columns):
        raise FinanceInputError("return columns must be non-empty and aligned")
    n = len(columns)
    if n == 1:
        return (1.0,)
    corr = correlation_matrix(columns)
    distance = correlation_distance(corr)
    merges = single_linkage(distance)
    order = quasi_diagonal_order(merges, n)
    cov = covariance_matrix(columns)
    weights = [1.0]*n
    clusters = [tuple(order)]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster)//2
            left, right = tuple(cluster[:split]), tuple(cluster[split:])
            left_var, right_var = cluster_variance(cov, left), cluster_variance(cov, right)
            total = left_var+right_var
            if total <= 0:
                alpha = 0.5
            else:
                alpha = 1.0-left_var/total
            for index in left:
                weights[index] *= alpha
            for index in right:
                weights[index] *= 1-alpha
            next_clusters.extend((left,right))
        clusters = next_clusters
    total = sum(weights)
    if total <= 0:
        raise FinanceInputError("HRP produced zero total weight")
    return tuple(value/total for value in weights)


def cluster_correlation_summary(correlation: Sequence[Sequence[Number]],
                                clusters: Sequence[Sequence[int]]) -> tuple[dict[str, float], ...]:
    """Report average within-cluster and cross-cluster correlations."""
    corr = tuple(tuple(finite(v) for v in row) for row in correlation)
    n = len(corr)
    if any(len(row) != n for row in corr):
        raise FinanceInputError("correlation matrix must be square")
    result = []
    for index, raw in enumerate(clusters):
        group = tuple(raw)
        if not group or any(i < 0 or i >= n for i in group):
            raise FinanceInputError("invalid cluster")
        within = [corr[i][j] for pos, i in enumerate(group) for j in group[pos+1:]]
        outside = [corr[i][j] for i in group for j in range(n) if j not in group]
        result.append({"cluster": float(index),
                       "size": float(len(group)),
                       "average_within_correlation": sum(within)/len(within) if within else 1.0,
                       "average_cross_correlation": sum(outside)/len(outside) if outside else math.nan})
    return tuple(result)


def diversification_ratio(weights: Sequence[Number],
                           covariance: Sequence[Sequence[Number]]) -> float:
    """Weighted average asset volatility divided by portfolio volatility."""
    w = vector(weights)
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    if len(w) != len(sigma) or any(len(row) != len(w) for row in sigma):
        raise FinanceInputError("weights and covariance dimensions differ")
    if any(sigma[i][i] < 0 for i in range(len(w))):
        raise FinanceInputError("negative asset variance")
    numerator = sum(abs(w[i])*math.sqrt(sigma[i][i]) for i in range(len(w)))
    denominator = math.sqrt(portfolio_variance(w, sigma))
    if denominator <= 0:
        raise FinanceInputError("diversification ratio undefined at zero portfolio volatility")
    return numerator/denominator


def marginal_contribution_to_variance(weights: Sequence[Number],
                                      covariance: Sequence[Sequence[Number]]) -> tuple[float, ...]:
    """Derivative of portfolio variance with respect to each asset weight."""
    w = vector(weights)
    sigma = tuple(tuple(finite(v) for v in row) for row in covariance)
    if len(w) != len(sigma) or any(len(row) != len(w) for row in sigma):
        raise FinanceInputError("weights and covariance dimensions differ")
    return tuple(2*sum(sigma[i][j]*w[j] for j in range(len(w))) for i in range(len(w)))


def factor_concentration(weights: Sequence[Number],
                         factor_loadings: Sequence[Sequence[Number]]) -> dict[str, object]:
    """Compute aggregate exposures to each factor and gross factor concentration."""
    w = vector(weights)
    rows = tuple(tuple(finite(v) for v in row) for row in factor_loadings)
    if len(rows) != len(w) or not rows or any(len(row) != len(rows[0]) for row in rows):
        raise FinanceInputError("factor loadings must be asset-by-factor and align with weights")
    exposures = tuple(sum(w[i]*rows[i][j] for i in range(len(w))) for j in range(len(rows[0])))
    gross = sum(abs(value) for value in exposures)
    normalized = tuple(abs(value)/gross for value in exposures) if gross else tuple(0.0 for _ in exposures)
    hhi = sum(value*value for value in normalized)
    return {"factor_exposures": exposures, "gross_factor_exposure": gross,
            "factor_hhi": hhi, "effective_factors": 1/hhi if hhi else 0.0}


def correlation_cluster_returns(returns: Sequence[Sequence[Number]],
                                 weights: Sequence[Number]) -> tuple[float, ...]:
    """Aggregate assets into a weighted synthetic portfolio return series."""
    columns = [vector(col) for col in returns]
    w = vector(weights)
    if len(columns) != len(w) or any(len(c) != len(columns[0]) for c in columns):
        raise FinanceInputError("weights and aligned return columns required")
    return tuple(sum(w[i]*columns[i][t] for i in range(len(w))) for t in range(len(columns[0])))
