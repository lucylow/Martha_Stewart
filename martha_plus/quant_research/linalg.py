"""Small, dependency-free linear algebra primitives for research workloads.

The routines intentionally favour transparent algorithms over speed.  They are
suitable for small factor models and teaching/research examples, not large-scale
production linear algebra.  Inputs are copied so callers cannot mutate state
through a returned view.
"""
from __future__ import annotations

from math import sqrt, isfinite
from typing import Iterable, Sequence

Matrix = list[list[float]]
Vector = list[float]


def as_vector(values: Iterable[float], *, name: str = "vector") -> Vector:
    result = [float(x) for x in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(isfinite(x) for x in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def as_matrix(values: Sequence[Sequence[float]], *, name: str = "matrix") -> Matrix:
    if not values:
        raise ValueError(f"{name} must not be empty")
    width = len(values[0])
    if width == 0 or any(len(row) != width for row in values):
        raise ValueError(f"{name} must be rectangular and non-empty")
    result = [[float(x) for x in row] for row in values]
    if not all(isfinite(x) for row in result for x in row):
        raise ValueError(f"{name} must contain only finite values")
    return result


def shape(a: Sequence[Sequence[float]]) -> tuple[int, int]:
    return len(a), len(a[0]) if a else 0


def transpose(a: Sequence[Sequence[float]]) -> Matrix:
    if not a:
        return []
    return [list(col) for col in zip(*a)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError(f"matrix dimensions do not align: {(ar, ac)} x {(br, bc)}")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: Sequence[Sequence[float]], x: Sequence[float]) -> Vector:
    if not a or len(a[0]) != len(x):
        raise ValueError("matrix and vector dimensions do not align")
    return [sum(v * w for v, w in zip(row, x)) for row in a]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimensions do not align")
    return sum(x * y for x, y in zip(a, b))


def eye(n: int) -> Matrix:
    if n < 1:
        raise ValueError("identity size must be positive")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(rows: int, cols: int) -> Matrix:
    if rows < 0 or cols < 0:
        raise ValueError("matrix dimensions must be non-negative")
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def add(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    if shape(a) != shape(b):
        raise ValueError("matrix dimensions do not align")
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def subtract(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    if shape(a) != shape(b):
        raise ValueError("matrix dimensions do not align")
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(a: Sequence[Sequence[float]], scalar: float) -> Matrix:
    return [[float(scalar) * x for x in row] for row in a]


def symmetrize(a: Sequence[Sequence[float]]) -> Matrix:
    if shape(a)[0] != shape(a)[1]:
        raise ValueError("matrix must be square")
    at = transpose(a)
    return [[(x + y) / 2.0 for x, y in zip(row, other)] for row, other in zip(a, at)]


def solve(a: Sequence[Sequence[float]], b: Sequence[float], *, ridge: float = 0.0) -> Vector:
    """Solve Ax=b using partial-pivoted Gaussian elimination."""
    m = as_matrix(a)
    n, cols = shape(m)
    if n != cols or len(b) != n:
        raise ValueError("solve requires a square matrix and matching right-hand side")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    rhs = [float(v) for v in b]
    for i in range(n):
        m[i][i] += ridge
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("matrix is singular; increase ridge or remove collinearity")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        p = m[col][col]
        for j in range(col, n):
            m[col][j] /= p
        rhs[col] /= p
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            if factor:
                for j in range(col, n):
                    m[row][j] -= factor * m[col][j]
                rhs[row] -= factor * rhs[col]
    return rhs


def inverse(a: Sequence[Sequence[float]], *, ridge: float = 0.0) -> Matrix:
    n, m = shape(a)
    if n != m:
        raise ValueError("inverse requires a square matrix")
    columns = [solve(a, col, ridge=ridge) for col in transpose(eye(n))]
    return transpose(columns)


def cholesky(a: Sequence[Sequence[float]], *, jitter: float = 0.0) -> Matrix:
    """Lower Cholesky factor.  A small jitter may regularize near-PSD inputs."""
    m = symmetrize(as_matrix(a))
    n, cols = shape(m)
    if n != cols:
        raise ValueError("Cholesky requires a square matrix")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    out = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = m[i][j] + (jitter if i == j else 0.0)
            s -= sum(out[i][k] * out[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite; increase jitter")
                out[i][j] = sqrt(s)
            else:
                out[i][j] = s / out[j][j]
    return out


def eigen_symmetric(a: Sequence[Sequence[float]], *, tol: float = 1e-12,
                    max_iter: int = 10_000) -> tuple[Vector, Matrix]:
    """Jacobi eigensolver. Returns descending eigenvalues and eigenvectors as columns."""
    m = symmetrize(as_matrix(a))
    n, cols = shape(m)
    if n != cols:
        raise ValueError("eigendecomposition requires a square matrix")
    vectors = eye(n)
    for _ in range(max_iter):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(m[i][j]) > largest:
                    p, q, largest = i, j, abs(m[i][j])
        if largest < tol:
            break
        app, aqq, apq = m[p][p], m[q][q], m[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c
        m[p][p] = app - t * apq
        m[q][q] = aqq + t * apq
        m[p][q] = m[q][p] = 0.0
        for k in range(n):
            if k not in (p, q):
                mkp, mkq = m[k][p], m[k][q]
                m[k][p] = m[p][k] = c * mkp - s * mkq
                m[k][q] = m[q][k] = s * mkp + c * mkq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq
    else:
        raise RuntimeError("Jacobi eigensolver did not converge")
    order = sorted(range(n), key=lambda i: m[i][i], reverse=True)
    values = [m[i][i] for i in order]
    vecs = [[vectors[r][i] for i in order] for r in range(n)]
    return values, vecs


def quadratic_form(x: Sequence[float], a: Sequence[Sequence[float]]) -> float:
    return dot(x, matvec(a, x))


def normalize_weights(weights: Sequence[float], *, allow_short: bool = False) -> Vector:
    w = [float(x) for x in weights]
    if not w or not all(isfinite(x) for x in w):
        raise ValueError("weights must be finite and non-empty")
    if not allow_short and any(x < 0 for x in w):
        raise ValueError("short weights are not allowed")
    total = sum(w)
    if abs(total) < 1e-14:
        raise ValueError("weights sum to zero")
    return [x / total for x in w]
