"""Shared numerical primitives for the technical-finance extension.

The package deliberately uses only the Python standard library.  Functions
validate inputs at boundaries and use decimal-free floating point arithmetic
for analytical research; callers that need exchange settlement precision
should use Decimal in their own accounting boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math
from typing import Iterable, Sequence, TypeAlias

Number: TypeAlias = int | float
Vector: TypeAlias = Sequence[Number]
Matrix: TypeAlias = Sequence[Sequence[Number]]


class FinanceInputError(ValueError):
    """Raised when a financial input violates a model's domain."""


class NumericalError(ArithmeticError):
    """Raised when a numerical method cannot produce a stable result."""


class DayCount(str, Enum):
    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


def finite(value: Number, name: str = "value") -> float:
    """Return *value* as float, rejecting booleans, NaN and infinities."""
    if isinstance(value, bool):
        raise FinanceInputError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinanceInputError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise FinanceInputError(f"{name} must be finite")
    return result


def positive(value: Number, name: str = "value", *, allow_zero: bool = False) -> float:
    """Validate a strictly positive (or non-negative) scalar."""
    result = finite(value, name)
    if result < 0 or (result == 0 and not allow_zero):
        op = "non-negative" if allow_zero else "positive"
        raise FinanceInputError(f"{name} must be {op}")
    return result


def probability(value: Number, name: str = "probability") -> float:
    """Validate a probability in the closed interval [0, 1]."""
    result = finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise FinanceInputError(f"{name} must be between 0 and 1")
    return result


def vector(values: Iterable[Number], name: str = "values", *, min_length: int = 1) -> tuple[float, ...]:
    """Materialize and validate a one-dimensional numeric input."""
    try:
        result = tuple(finite(v, f"{name}[{i}]") for i, v in enumerate(values))
    except TypeError as exc:
        raise FinanceInputError(f"{name} must be iterable") from exc
    if len(result) < min_length:
        raise FinanceInputError(f"{name} requires at least {min_length} value(s)")
    return result


def paired(x: Iterable[Number], y: Iterable[Number], *, min_length: int = 2) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate two aligned vectors and return immutable float tuples."""
    left, right = vector(x, "x", min_length=min_length), vector(y, "y", min_length=min_length)
    if len(left) != len(right):
        raise FinanceInputError("paired vectors must have equal lengths")
    return left, right


def matrix(values: Matrix, name: str = "matrix", *, square: bool = False) -> tuple[tuple[float, ...], ...]:
    """Validate a rectangular matrix; optionally require a square shape."""
    rows = tuple(vector(row, f"{name}[{i}]") for i, row in enumerate(values))
    if not rows:
        raise FinanceInputError(f"{name} must not be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise FinanceInputError(f"{name} must be rectangular")
    if square and len(rows) != width:
        raise FinanceInputError(f"{name} must be square")
    return rows


def mean(values: Vector) -> float:
    data = vector(values)
    return math.fsum(data) / len(data)


def variance(values: Vector, *, sample: bool = True) -> float:
    data = vector(values, min_length=2 if sample else 1)
    center = math.fsum(data) / len(data)
    denom = len(data) - 1 if sample else len(data)
    return math.fsum((x - center) ** 2 for x in data) / denom


def covariance(x: Vector, y: Vector, *, sample: bool = True) -> float:
    left, right = paired(x, y, min_length=2 if sample else 1)
    mx, my = math.fsum(left) / len(left), math.fsum(right) / len(right)
    denom = len(left) - 1 if sample else len(left)
    return math.fsum((a - mx) * (b - my) for a, b in zip(left, right)) / denom


def correlation(x: Vector, y: Vector) -> float:
    left, right = paired(x, y)
    vx, vy = variance(left), variance(right)
    if vx <= 0 or vy <= 0:
        raise NumericalError("correlation is undefined for a constant vector")
    return covariance(left, right) / math.sqrt(vx * vy)


def quantile(values: Vector, q: Number) -> float:
    """Linear-interpolated sample quantile (Hyndman-Fan type 7)."""
    data = sorted(vector(values))
    qv = probability(q, "q")
    position = (len(data) - 1) * qv
    lo = int(math.floor(position))
    hi = min(lo + 1, len(data) - 1)
    weight = position - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def normal_pdf(x: Number) -> float:
    value = finite(x, "x")
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: Number) -> float:
    value = finite(x, "x")
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_ppf(p: Number) -> float:
    """Acklam rational approximation to the standard-normal quantile."""
    p = probability(p, "p")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1)
    c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > high:
        q = math.sqrt(-2.0 * math.log(1.0-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def bisect(function, target: float, lower: float, upper: float, *, tolerance: float = 1e-10, max_iterations: int = 200) -> float:
    """Solve function(x) == target on a bracketed monotone interval."""
    lo, hi = finite(lower, "lower"), finite(upper, "upper")
    if lo >= hi:
        raise FinanceInputError("lower must be less than upper")
    target = finite(target, "target")
    flo, fhi = finite(function(lo), "function(lower)"), finite(function(hi), "function(upper)")
    glo, ghi = flo - target, fhi - target
    if glo == 0:
        return lo
    if ghi == 0:
        return hi
    if glo * ghi > 0:
        raise NumericalError("root is not bracketed")
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        gm = finite(function(mid), "function(mid)") - target
        if abs(gm) <= tolerance or hi - lo <= tolerance:
            return mid
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    raise NumericalError("bisection did not converge")


def solve_linear(a: Matrix, b: Vector, *, tolerance: float = 1e-12) -> tuple[float, ...]:
    """Solve Ax=b using Gaussian elimination with partial pivoting."""
    rows = [list(row) for row in matrix(a, square=True)]
    rhs = list(vector(b))
    n = len(rows)
    if len(rhs) != n:
        raise FinanceInputError("A and b dimensions do not match")
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= tolerance:
            raise NumericalError("singular or ill-conditioned linear system")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        scale = rows[col][col]
        for j in range(col, n):
            rows[col][j] /= scale
        rhs[col] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col]
            if factor == 0:
                continue
            for j in range(col, n):
                rows[r][j] -= factor * rows[col][j]
            rhs[r] -= factor * rhs[col]
    return tuple(rhs)


def cholesky(a: Matrix, *, tolerance: float = 1e-12) -> tuple[tuple[float, ...], ...]:
    """Lower Cholesky factor for a symmetric positive-definite matrix."""
    values = matrix(a, square=True)
    n = len(values)
    for i in range(n):
        for j in range(n):
            if abs(values[i][j] - values[j][i]) > 1e-9:
                raise FinanceInputError("matrix must be symmetric")
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            residual = values[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual <= tolerance:
                    raise NumericalError("matrix is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    return tuple(tuple(row) for row in lower)


def year_fraction(start: date | datetime, end: date | datetime, convention: DayCount = DayCount.ACT_365F) -> float:
    """Compute a year fraction for common money-market and bond conventions."""
    d1, d2 = start.date() if isinstance(start, datetime) else start, end.date() if isinstance(end, datetime) else end
    if d2 < d1:
        raise FinanceInputError("end date must not precede start date")
    days = (d2 - d1).days
    if convention == DayCount.ACT_360:
        return days / 360.0
    if convention == DayCount.ACT_365F:
        return days / 365.0
    if convention == DayCount.ACT_ACT:
        if d1 == d2:
            return 0.0
        cursor, fraction = d1, 0.0
        while cursor < d2:
            year_end = date(cursor.year + 1, 1, 1)
            segment_end = min(d2, year_end)
            denominator = 366.0 if _is_leap(cursor.year) else 365.0
            fraction += (segment_end - cursor).days / denominator
            cursor = segment_end
        return fraction
    if convention == DayCount.THIRTY_360:
        d_start = min(d1.day, 30)
        d_end = min(d2.day, 30) if d_start == 30 else d2.day
        return ((d2.year-d1.year)*360 + (d2.month-d1.month)*30 + d_end-d_start) / 360.0
    raise FinanceInputError(f"unsupported day count convention: {convention}")


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@dataclass(frozen=True)
class Estimate:
    """Point estimate with optional standard error and metadata."""
    value: float
    standard_error: float | None = None
    observations: int | None = None
    method: str = "unspecified"

    def __post_init__(self) -> None:
        finite(self.value, "value")
        if self.standard_error is not None:
            positive(self.standard_error, "standard_error", allow_zero=True)
        if self.observations is not None and self.observations < 0:
            raise FinanceInputError("observations must be non-negative")


@dataclass(frozen=True)
class CashFlow:
    """Dated cash flow, where amount is from the investor's perspective."""
    payment_date: date
    amount: float
    label: str = "cashflow"

    def __post_init__(self) -> None:
        finite(self.amount, "amount")


@dataclass(frozen=True)
class ModelWarning:
    """Model limitation carried with outputs so caveats survive serialization."""
    code: str
    message: str
    severity: str = "info"


@dataclass(frozen=True)
class ModelResult:
    """Generic immutable result envelope for analytical components."""
    values: dict[str, float]
    warnings: tuple[ModelWarning, ...] = ()
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for key, value in self.values.items():
            finite(value, key)
