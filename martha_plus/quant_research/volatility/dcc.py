"""Dynamic Conditional Correlation (DCC) multivariate volatility research model.

The estimator uses two-stage marginal volatility standardization followed by
Engle-style scalar DCC recursion. Its deterministic grid search is transparent
and intentionally modest; it is not a substitute for a fully constrained
maximum-likelihood implementation with robust inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

from ..core import finite_series, mean, variance
from ..linalg import inverse, matmul, matvec, quadratic_form, symmetrize
from .garch import fit_garch11, GARCHResult


@dataclass(frozen=True)
class DCCResult:
    a: float
    b: float
    qbar: tuple[tuple[float, ...], ...]
    q_last: tuple[tuple[float, ...], ...]
    correlations: tuple[tuple[tuple[float, ...], ...], ...]
    conditional_volatility: tuple[tuple[float, ...], ...]
    standardized_residuals: tuple[tuple[float, ...], ...]
    marginal_models: tuple[object, ...]
    log_likelihood: float
    observations: int
    assets: int
    converged: bool
    marginal_method: str


def _panel(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [finite_series(row, name="return row") for row in returns]
    if len(rows) < 30 or not rows or len(rows[0]) < 2:
        raise ValueError("DCC requires at least 30 rows and 2 assets")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("returns must be a rectangular time x asset matrix")
    return rows


def _sample_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    p = len(rows[0])
    mu = [mean([r[j] for r in rows]) for j in range(p)]
    return [[sum((r[i]-mu[i])*(r[j]-mu[j]) for r in rows)/(len(rows)-1)
             for j in range(p)] for i in range(p)]


def _correlation(q: Sequence[Sequence[float]]) -> list[list[float]]:
    p = len(q)
    scale = [sqrt(max(float(q[i][i]), 1e-14)) for i in range(p)]
    out = [[float(q[i][j])/(scale[i]*scale[j]) for j in range(p)] for i in range(p)]
    for i in range(p):
        out[i][i] = 1.0
    return symmetrize(out)


def _logdet_spd(matrix: Sequence[Sequence[float]]) -> float:
    """Stable log determinant through Cholesky, with tiny diagonal jitter."""
    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        lower = [[0.0]*n for _ in range(n)]
        ok = True
        for i in range(n):
            for j in range(i+1):
                value = a[i][j] + (jitter if i == j else 0.0)
                value -= sum(lower[i][k]*lower[j][k] for k in range(j))
                if i == j:
                    if value <= 0:
                        ok = False
                        break
                    lower[i][j] = sqrt(value)
                else:
                    lower[i][j] = value/lower[j][j]
            if not ok:
                break
        if ok:
            return 2.0*sum(log(lower[i][i]) for i in range(n))
    raise ValueError("correlation matrix is not positive definite")


def _dcc_path(z: Sequence[Sequence[float]], qbar: Sequence[Sequence[float]],
              a: float, b: float) -> tuple[float, list[list[list[float]]], list[list[float]]]:
    n, p = len(z), len(z[0])
    q = [[float(v) for v in row] for row in qbar]
    path: list[list[list[float]]] = []
    ll = 0.0
    for t in range(n):
        r = _correlation(q)
        path.append(r)
        # DCC Gaussian quasi-likelihood contribution, omitting constants.
        det = _logdet_spd(r)
        inv = inverse(r, ridge=1e-12)
        quad = quadratic_form(z[t], inv)
        iid = sum(v*v for v in z[t])
        ll += -0.5*(det + quad - iid)
        if t < n-1:
            zz = [[z[t][i]*z[t][j] for j in range(p)] for i in range(p)]
            q = [[(1.0-a-b)*qbar[i][j] + a*zz[i][j] + b*q[i][j]
                  for j in range(p)] for i in range(p)]
            q = symmetrize(q)
    return ll, path, q


def fit_dcc(returns: Sequence[Sequence[float]], *, marginal: str = "garch",
            decay: float = 0.94, a_values: Sequence[float] | None = None,
            b_values: Sequence[float] | None = None) -> DCCResult:
    """Fit scalar DCC parameters to a time x asset return panel.

    ``marginal='garch'`` fits one GARCH(1,1) per column. ``marginal='ewma'``
    uses a causal EWMA variance recursion. A sample-standardization option is
    provided for controlled experiments. Search grids are explicit parameters.
    """
    x = _panel(returns)
    if marginal not in {"garch", "ewma", "sample"}:
        raise ValueError("marginal must be garch, ewma or sample")
    if not 0 < decay < 1:
        raise ValueError("decay must be in (0,1)")
    n, p = len(x), len(x[0])
    columns = [[row[j] for row in x] for j in range(p)]
    z_cols: list[list[float]] = []
    vol_cols: list[list[float]] = []
    models: list[object] = []
    for col in columns:
        if marginal == "garch":
            model = fit_garch11(col)
            vols = [sqrt(max(v, 1e-14)) for v in model.conditional_variance]
            zc = list(model.standardized_residuals)
            models.append(model)
        elif marginal == "ewma":
            # Daily financial returns are centered at zero to avoid using a
            # full-sample mean estimate inside this causal variance recursion.
            mu = 0.0
            h = variance(col)
            vols, zc = [], []
            for value in col:
                vols.append(sqrt(max(h, 1e-14)))
                zc.append(value/sqrt(max(h, 1e-14)))
                h = decay*h + (1.0-decay)*value*value
            models.append({"mean": mu, "decay": decay, "last_variance": h})
        else:
            mu = mean(col)
            sd = sqrt(max(variance(col), 1e-14))
            vols = [sd]*n
            zc = [(value-mu)/sd for value in col]
            models.append({"mean": mu, "standard_deviation": sd})
        z_cols.append(zc)
        vol_cols.append(vols)
    z = [[z_cols[j][t] for j in range(p)] for t in range(n)]
    qbar = _sample_matrix(z)
    av = tuple(a_values) if a_values is not None else (0.01, 0.03, 0.05, 0.08, 0.12, 0.16)
    bv = tuple(b_values) if b_values is not None else (0.70, 0.78, 0.84, 0.88, 0.92, 0.95, 0.97)
    if not av or not bv or any(a < 0 for a in av) or any(b < 0 for b in bv):
        raise ValueError("DCC search grids must be non-empty and non-negative")
    best = (float("-inf"), 0.03, 0.90, None, None)
    for a in av:
        for b in bv:
            if a+b >= 0.995:
                continue
            try:
                ll, path, qlast = _dcc_path(z, qbar, a, b)
            except ValueError:
                continue
            if ll > best[0]:
                best = (ll, float(a), float(b), path, qlast)
    if best[3] is None:
        raise ValueError("no admissible DCC parameter pair produced positive-definite correlations")
    ll, a, b, path, qlast = best
    return DCCResult(a, b, tuple(tuple(r) for r in qbar), tuple(tuple(r) for r in qlast),
                     tuple(tuple(tuple(row) for row in mat) for mat in path),
                     tuple(tuple(vol_cols[j][t] for j in range(p)) for t in range(n)),
                     tuple(tuple(row) for row in z), tuple(models), ll, n, p, True, marginal)


def forecast_dcc(result: DCCResult, *, standardized_residual: Sequence[float] | None = None) -> dict:
    """One-step-ahead correlation and covariance forecast from the fitted DCC."""
    p = result.assets
    z = list(result.standardized_residuals[-1] if standardized_residual is None else map(float, standardized_residual))
    if len(z) != p:
        raise ValueError("standardized_residual must match asset count")
    qbar = [list(row) for row in result.qbar]
    qlast = [list(row) for row in result.q_last]
    qnext = [[(1-result.a-result.b)*qbar[i][j] + result.a*z[i]*z[j] + result.b*qlast[i][j]
              for j in range(p)] for i in range(p)]
    corr = _correlation(symmetrize(qnext))
    last_vol = list(result.conditional_volatility[-1])
    cov = [[corr[i][j]*last_vol[i]*last_vol[j] for j in range(p)] for i in range(p)]
    return {"correlation": tuple(tuple(r) for r in corr),
            "covariance": tuple(tuple(r) for r in cov),
            "q_next": tuple(tuple(r) for r in qnext),
            "assumption": "one-step DCC recursion; marginal volatility held at last filtered estimate"}


def dcc_portfolio_volatility(weights: Sequence[float], forecast: dict) -> float:
    """Compute portfolio volatility from a DCC covariance forecast."""
    from ..linalg import quadratic_form
    w = list(map(float, weights))
    cov = forecast["covariance"]
    if len(w) != len(cov) or any(len(row) != len(w) for row in cov):
        raise ValueError("weights and covariance dimensions do not match")
    return sqrt(max(0.0, quadratic_form(w, cov)))


def rolling_dcc_forecasts(returns: Sequence[Sequence[float]], window: int = 250,
                          step: int = 20, *, marginal: str = "ewma") -> tuple[dict, ...]:
    """Refit DCC on expanding-by-window slices and return out-of-window forecasts.

    This deliberately simple outer loop is useful for research walk-forward
    diagnostics; use purged validation when forecasts drive labelled outcomes.
    """
    x = _panel(returns)
    if window < 30 or step < 1 or window >= len(x):
        raise ValueError("window must be >=30 and smaller than sample; step >=1")
    out = []
    for end in range(window, len(x), step):
        fit = fit_dcc(x[end-window:end], marginal=marginal)
        out.append({"fit_end_index": end-1, "forecast_index": end,
                    "forecast": forecast_dcc(fit), "a": fit.a, "b": fit.b,
                    "log_likelihood": fit.log_likelihood})
    return tuple(out)
