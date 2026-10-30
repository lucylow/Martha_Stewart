"""Implied-volatility surface storage, interpolation and consistency checks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..core import FinanceInputError, Number, finite, positive


@dataclass(frozen=True, order=True)
class VolPoint:
    maturity: float
    strike: float
    volatility: float

    def __post_init__(self) -> None:
        positive(self.maturity, "maturity")
        positive(self.strike, "strike")
        positive(self.volatility, "volatility", allow_zero=True)


@dataclass(frozen=True)
class SurfaceCheck:
    passed: bool
    violations: tuple[str, ...]


class ImpliedVolSurface:
    """Rectangular or sparse volatility surface with bilinear interpolation.

    Coordinates are time-to-expiry in years and absolute strike. For better
    cross-asset comparability, callers can construct the surface in log-moneyness
    by supplying transformed strikes consistently.
    """
    def __init__(self, points: Sequence[VolPoint], *, spot: Number | None = None):
        if len(points) < 2:
            raise FinanceInputError("at least two volatility points are required")
        self._points = tuple(points)
        self.spot = None if spot is None else positive(spot, "spot")
        self._maturities = tuple(sorted({p.maturity for p in points}))
        self._strikes = tuple(sorted({p.strike for p in points}))
        grid = {(p.maturity, p.strike): p.volatility for p in points}
        if len(grid) != len(points):
            raise FinanceInputError("duplicate maturity/strike coordinates")
        self._grid = grid

    @property
    def points(self) -> tuple[VolPoint, ...]:
        return self._points

    @property
    def maturities(self) -> tuple[float, ...]:
        return self._maturities

    @property
    def strikes(self) -> tuple[float, ...]:
        return self._strikes

    def volatility(self, maturity: Number, strike: Number, *, extrapolate: bool = False) -> float:
        """Bilinearly interpolate a rectangular surface; sparse cells fail clearly."""
        t, k = positive(maturity, "maturity"), positive(strike, "strike")
        if not extrapolate and (t < self._maturities[0] or t > self._maturities[-1]
                                or k < self._strikes[0] or k > self._strikes[-1]):
            raise FinanceInputError("query lies outside surface bounds")
        t0, t1 = _bracket(self._maturities, t)
        k0, k1 = _bracket(self._strikes, k)
        q00, q01 = self._grid.get((t0, k0)), self._grid.get((t0, k1))
        q10, q11 = self._grid.get((t1, k0)), self._grid.get((t1, k1))
        if None in (q00, q01, q10, q11):
            raise FinanceInputError("bilinear interpolation requires a complete grid cell")
        wt = 0.0 if t0 == t1 else (t-t0)/(t1-t0)
        wk = 0.0 if k0 == k1 else (k-k0)/(k1-k0)
        lower = q00*(1-wk)+q01*wk
        upper = q10*(1-wk)+q11*wk
        return lower*(1-wt)+upper*wt

    def total_variance(self, maturity: Number, strike: Number) -> float:
        """Return implied total variance sigma^2 * T."""
        t = positive(maturity, "maturity")
        vol = self.volatility(t, strike)
        return vol*vol*t

    def term_structure(self, strike: Number) -> tuple[tuple[float, float], ...]:
        """Return maturity/volatility pairs at a fixed strike."""
        return tuple((t, self.volatility(t, strike)) for t in self._maturities)

    def smile(self, maturity: Number) -> tuple[tuple[float, float], ...]:
        """Return strike/volatility pairs at a fixed maturity."""
        return tuple((k, self.volatility(maturity, k)) for k in self._strikes)

    def check_calendar_monotonicity(self, *, tolerance: Number = 1e-8) -> SurfaceCheck:
        """Check total variance is non-decreasing with maturity at each strike."""
        tol = positive(tolerance, "tolerance", allow_zero=True)
        violations = []
        for strike in self._strikes:
            available = [(t, self._grid[(t, strike)]) for t in self._maturities
                         if (t, strike) in self._grid]
            for (t0, v0), (t1, v1) in zip(available, available[1:]):
                if v1*v1*t1 + tol < v0*v0*t0:
                    violations.append(f"calendar total variance decreases at strike={strike}: {t0}->{t1}")
        return SurfaceCheck(not violations, tuple(violations))

    def check_butterfly_convexity(self, *, tolerance: Number = 1e-8) -> SurfaceCheck:
        """Discrete convexity check on call prices is intentionally not inferred from IV alone.

        This method checks convexity of total variance across equally spaced
        strike nodes as a screening diagnostic, not a complete static-arbitrage
        proof. Use option prices and density constraints for production checks.
        """
        tol = positive(tolerance, "tolerance", allow_zero=True)
        violations = []
        if len(self._strikes) < 3:
            return SurfaceCheck(True, ())
        for maturity in self._maturities:
            grid = [self._grid.get((maturity, k)) for k in self._strikes]
            if any(v is None for v in grid):
                continue
            total = [float(v)**2*maturity for v in grid]
            for i in range(1, len(total)-1):
                left_dx = self._strikes[i]-self._strikes[i-1]
                right_dx = self._strikes[i+1]-self._strikes[i]
                left_slope = (total[i]-total[i-1])/left_dx
                right_slope = (total[i+1]-total[i])/right_dx
                if right_slope < left_slope-tol:
                    violations.append(f"curvature screen failed at maturity={maturity}, strike={self._strikes[i]}")
        return SurfaceCheck(not violations, tuple(violations))

    def to_dict(self) -> dict[str, object]:
        """Serialize the surface into primitive Python objects."""
        return {"spot": self.spot,
                "points": [{"maturity": p.maturity, "strike": p.strike,
                            "volatility": p.volatility} for p in self._points]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ImpliedVolSurface":
        """Construct a surface from the output of :meth:`to_dict`."""
        raw = payload.get("points")
        if not isinstance(raw, list):
            raise FinanceInputError("surface payload requires a points list")
        points = [VolPoint(float(item["maturity"]), float(item["strike"]), float(item["volatility"]))
                  for item in raw if isinstance(item, dict)]
        return cls(points, spot=payload.get("spot"))


def _bracket(axis: Sequence[float], value: float) -> tuple[float, float]:
    if value <= axis[0]:
        return axis[0], axis[0]
    if value >= axis[-1]:
        return axis[-1], axis[-1]
    for left, right in zip(axis, axis[1:]):
        if left <= value <= right:
            return left, right
    raise FinanceInputError("could not bracket surface coordinate")
