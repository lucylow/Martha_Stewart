"""Gaussian-copula scenarios with empirical marginal distributions."""
from __future__ import annotations
import random
from typing import Sequence
from ..core import finite_series, normal_cdf, quantile
from ..linalg import cholesky, matvec


def gaussian_copula_scenarios(historical_returns: Sequence[Sequence[float]],
                              correlation: Sequence[Sequence[float]], n_scenarios: int = 1000,
                              *, seed: int = 7, jitter: float = 1e-10) -> list[tuple[float,...]]:
    """Sample dependent uniforms then transform through empirical marginals.

    The Gaussian copula imposes zero asymptotic tail dependence; it can materially
    understate joint crashes. Compare with historical and stress scenarios.
    """
    rows=[finite_series(r,name="historical return column",min_length=5) for r in historical_returns]
    assets=len(rows)
    if any(len(r)!=len(rows[0]) for r in rows) or len(correlation)!=assets or any(len(r)!=assets for r in correlation):
        raise ValueError("marginal data and correlation dimensions mismatch")
    if n_scenarios<1: raise ValueError("n_scenarios must be positive")
    chol=cholesky(correlation,jitter=jitter); rng=random.Random(seed); result=[]
    for _ in range(n_scenarios):
        z=[rng.gauss(0,1) for _ in range(assets)]
        correlated=matvec(chol,z)
        result.append(tuple(quantile(rows[j],normal_cdf(correlated[j])) for j in range(assets)))
    return result


def empirical_copula_rank(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert time x asset panel to pseudo-observations in (0,1)."""
    if not returns: raise ValueError("returns must not be empty")
    cols=list(zip(*returns)); n=len(returns)
    if n<2: raise ValueError("at least two rows required")
    from ..core import rank
    ranked=[rank(col) for col in cols]
    return [[ranked[j][i]/(n+1.0) for j in range(len(cols))] for i in range(n)]
