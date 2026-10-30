"""OLS, robust covariance estimates, factor regression and rolling models."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .core import FinanceInputError, Number, Vector, covariance, finite, mean, solve_linear, variance, vector


@dataclass(frozen=True)
class OLSResult:
    """OLS coefficients and classic inference statistics."""
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    t_statistics: tuple[float, ...]
    r_squared: float
    adjusted_r_squared: float
    residuals: tuple[float, ...]
    fitted: tuple[float, ...]
    observations: int
    degrees_freedom: int
    intercept: bool

    def predict(self, rows: Sequence[Sequence[Number]]) -> tuple[float, ...]:
        """Predict on rows that contain one value per explanatory variable."""
        result = []
        for row in rows:
            values = vector(row, "row", min_length=0)
            expected = len(self.coefficients) - int(self.intercept)
            if len(values) != expected:
                raise FinanceInputError("prediction row has wrong feature count")
            result.append(self.coefficients[0] + sum(b*x for b, x in zip(self.coefficients[1:], values))
                          if self.intercept else sum(b*x for b, x in zip(self.coefficients, values)))
        return tuple(result)


def ols(y: Vector, x: Sequence[Vector], *, intercept: bool = True) -> OLSResult:
    """Fit OLS via normal equations with pivoted Gaussian elimination.

    Columns in ``x`` are features. This implementation is intentionally
    transparent; for large or ill-conditioned problems use a QR/SVD library.
    """
    target = vector(y, "y", min_length=3)
    columns = [vector(col, f"x[{i}]") for i, col in enumerate(x)]
    if not columns:
        raise FinanceInputError("at least one explanatory variable is required")
    if any(len(col) != len(target) for col in columns):
        raise FinanceInputError("all explanatory variables must align with y")
    n = len(target)
    design = [[1.0] if intercept else [] for _ in range(n)]
    for col in columns:
        for i, value in enumerate(col):
            design[i].append(value)
    p = len(design[0])
    if n <= p:
        raise FinanceInputError("observations must exceed estimated parameters")
    xtx = [[math.fsum(row[i]*row[j] for row in design) for j in range(p)] for i in range(p)]
    xty = [math.fsum(design[k][i]*target[k] for k in range(n)) for i in range(p)]
    beta = solve_linear(xtx, xty)
    fitted = tuple(math.fsum(a*b for a, b in zip(row, beta)) for row in design)
    residuals = tuple(a-b for a, b in zip(target, fitted))
    ybar = mean(target)
    sse = math.fsum(e*e for e in residuals)
    sst = math.fsum((v-ybar)**2 for v in target)
    r2 = 1.0-sse/sst if sst > 0 else (1.0 if sse <= 1e-20 else 0.0)
    df = n-p
    adj = 1.0-(1.0-r2)*(n-1)/df if intercept else 1.0-(1.0-r2)*n/df
    sigma2 = sse/df
    inv = inverse(xtx)
    ses = tuple(math.sqrt(max(0.0, sigma2*inv[i][i])) for i in range(p))
    tstats = tuple(beta[i]/ses[i] if ses[i] else math.copysign(math.inf, beta[i]) for i in range(p))
    return OLSResult(beta, ses, tstats, r2, adj, residuals, fitted, n, df, intercept)


def inverse(a: Sequence[Sequence[Number]]) -> tuple[tuple[float, ...], ...]:
    """Matrix inverse using one solve per identity column."""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise FinanceInputError("inverse requires a non-empty square matrix")
    cols = [solve_linear(a, tuple(1.0 if i == j else 0.0 for i in range(n))) for j in range(n)]
    return tuple(tuple(cols[j][i] for j in range(n)) for i in range(n))


def newey_west_covariance(x: Sequence[Vector], residuals: Vector,
                          max_lag: int | None = None, *, intercept: bool = True) -> tuple[tuple[float, ...], ...]:
    """Newey-West HAC covariance for an OLS design, Bartlett kernel.

    The returned matrix is for coefficients in the same order as ``ols``.
    """
    e = vector(residuals, "residuals", min_length=3)
    columns = [vector(col, f"x[{i}]") for i, col in enumerate(x)]
    n = len(e)
    if not columns or any(len(col) != n for col in columns):
        raise FinanceInputError("predictors must be non-empty and aligned")
    p = len(columns)+int(intercept)
    if n <= p:
        raise FinanceInputError("not enough observations for HAC covariance")
    lag = int(math.floor(4*(n/100)**(2/9))) if max_lag is None else max_lag
    if not isinstance(lag, int) or lag < 0 or lag >= n:
        raise FinanceInputError("max_lag must be an integer in [0, n)")
    design = [[1.0] if intercept else [] for _ in range(n)]
    for col in columns:
        for i, val in enumerate(col):
            design[i].append(val)
    bread = inverse([[math.fsum(row[i]*row[j] for row in design) for j in range(p)] for i in range(p)])
    score = [[design[t][i]*e[t] for i in range(p)] for t in range(n)]
    meat = [[math.fsum(score[t][i]*score[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    for k in range(1, lag+1):
        weight = 1.0-k/(lag+1.0)
        for i in range(p):
            for j in range(p):
                gamma = math.fsum(score[t][i]*score[t-k][j] + score[t-k][i]*score[t][j]
                                  for t in range(k, n))
                meat[i][j] += weight*gamma
    left = matmul(bread, meat)
    result = matmul(left, bread)
    return tuple(tuple(row) for row in result)


def matmul(a: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> list[list[float]]:
    """Dense matrix product for small research matrices."""
    if not a or not b or len(a[0]) != len(b):
        raise FinanceInputError("matrix dimensions do not conform")
    bt = list(zip(*b))
    return [[math.fsum(float(x)*float(y) for x, y in zip(row, col)) for col in bt] for row in a]


def hac_standard_errors(x: Sequence[Vector], residuals: Vector,
                        max_lag: int | None = None, *, intercept: bool = True) -> tuple[float, ...]:
    """Square roots of the Newey-West covariance diagonal."""
    cov = newey_west_covariance(x, residuals, max_lag, intercept=intercept)
    return tuple(math.sqrt(max(0.0, cov[i][i])) for i in range(len(cov)))


def rolling_ols(y: Vector, x: Sequence[Vector], window: int,
                *, intercept: bool = True) -> tuple[OLSResult, ...]:
    """Fit a sequence of trailing-window regressions."""
    target = vector(y)
    columns = [vector(c) for c in x]
    if any(len(c) != len(target) for c in columns):
        raise FinanceInputError("predictor columns must align with y")
    if not 2 <= window <= len(target):
        raise FinanceInputError("window must be in [2, n]")
    p = len(columns)+int(intercept)
    if window <= p:
        raise FinanceInputError("window must exceed number of parameters")
    results = []
    for end in range(window, len(target)+1):
        results.append(ols(target[end-window:end], [c[end-window:end] for c in columns], intercept=intercept))
    return tuple(results)


def variance_inflation_factors(x: Sequence[Vector]) -> tuple[float, ...]:
    """Compute VIF for each feature, with no intercept in target columns."""
    columns = [vector(col, f"x[{i}]", min_length=3) for i, col in enumerate(x)]
    if len(columns) < 2:
        raise FinanceInputError("VIF requires at least two features")
    if any(len(col) != len(columns[0]) for col in columns):
        raise FinanceInputError("features must have equal lengths")
    result = []
    for i, target in enumerate(columns):
        predictors = [col for j, col in enumerate(columns) if j != i]
        r2 = ols(target, predictors, intercept=True).r_squared
        if r2 >= 1.0-1e-12:
            result.append(math.inf)
        else:
            result.append(1.0/(1.0-r2))
    return tuple(result)


def correlation_matrix(columns: Sequence[Vector]) -> tuple[tuple[float, ...], ...]:
    """Compute a symmetric sample correlation matrix from column vectors."""
    data = [vector(col) for col in columns]
    if not data or any(len(col) != len(data[0]) for col in data):
        raise FinanceInputError("columns must be non-empty and equally sized")
    return tuple(tuple(1.0 if i == j else _corr(data[i], data[j])
                       for j in range(len(data))) for i in range(len(data)))


def _corr(x: Vector, y: Vector) -> float:
    vx, vy = variance(x), variance(y)
    if vx <= 0 or vy <= 0:
        raise FinanceInputError("correlation is undefined for constant columns")
    return covariance(x, y)/math.sqrt(vx*vy)


def regression_summary(result: OLSResult, names: Sequence[str] | None = None) -> dict[str, object]:
    """Create a JSON-friendly compact coefficient summary."""
    labels = list(names or [f"x{i}" for i in range(len(result.coefficients)-int(result.intercept))])
    if result.intercept:
        labels = ["intercept"]+labels
    if len(labels) != len(result.coefficients):
        raise FinanceInputError("coefficient names do not match parameter count")
    return {"n": result.observations, "df_resid": result.degrees_freedom,
            "r_squared": result.r_squared, "adjusted_r_squared": result.adjusted_r_squared,
            "coefficients": [{"name": name, "estimate": beta, "std_error": se, "t_stat": t}
                              for name, beta, se, t in zip(labels, result.coefficients,
                                                          result.standard_errors, result.t_statistics)]}
