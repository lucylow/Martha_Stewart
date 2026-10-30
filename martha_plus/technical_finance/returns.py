"""Return construction, compounding, drawdown and rolling time-series tools."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import FinanceInputError, Number, Vector, finite, positive, vector


def simple_returns(prices: Vector) -> tuple[float, ...]:
    """Convert strictly positive prices into close-to-close simple returns."""
    data = tuple(positive(x, f"prices[{i}]") for i, x in enumerate(prices))
    if len(data) < 2:
        raise FinanceInputError("at least two prices are required")
    return tuple(data[i]/data[i-1]-1.0 for i in range(1, len(data)))


def log_returns(prices: Vector) -> tuple[float, ...]:
    """Convert strictly positive prices into continuously compounded returns."""
    data = tuple(positive(x, f"prices[{i}]") for i, x in enumerate(prices))
    if len(data) < 2:
        raise FinanceInputError("at least two prices are required")
    return tuple(math.log(data[i]/data[i-1]) for i in range(1, len(data)))


def compound(returns: Vector) -> float:
    """Compound simple returns to a total holding-period return."""
    data = vector(returns, min_length=0)
    wealth = 1.0
    for i, value in enumerate(data):
        if value < -1.0:
            raise FinanceInputError(f"return[{i}] cannot be below -100%")
        wealth *= 1.0 + value
    return wealth - 1.0


def wealth_index(returns: Vector, initial_value: Number = 1.0) -> tuple[float, ...]:
    """Return wealth after each simple-return observation."""
    start = positive(initial_value, "initial_value")
    data = vector(returns, min_length=0)
    wealth, result = start, []
    for i, value in enumerate(data):
        if value < -1.0:
            raise FinanceInputError(f"return[{i}] cannot be below -100%")
        wealth *= 1.0 + value
        result.append(wealth)
    return tuple(result)


def annualized_return(returns: Vector, periods_per_year: Number = 252.0) -> float:
    """Geometric annualized return over equally spaced simple returns."""
    data = vector(returns)
    periods = positive(periods_per_year, "periods_per_year")
    gross = 1.0 + compound(data)
    if gross <= 0:
        return -1.0
    return gross ** (periods/len(data)) - 1.0


def annualized_volatility(returns: Vector, periods_per_year: Number = 252.0) -> float:
    """Sample volatility annualized by the square root of periods per year."""
    from .core import variance
    data = vector(returns, min_length=2)
    return math.sqrt(variance(data))*math.sqrt(positive(periods_per_year, "periods_per_year"))


def arithmetic_annualized_return(returns: Vector, periods_per_year: Number = 252.0) -> float:
    """Arithmetic annualized return, useful as a first-order estimator."""
    from .core import mean
    return mean(returns)*positive(periods_per_year, "periods_per_year")


def excess_returns(returns: Vector, risk_free: Vector | Number) -> tuple[float, ...]:
    """Subtract a scalar or period-aligned risk-free return series."""
    data = vector(returns, min_length=0)
    if isinstance(risk_free, (int, float)) and not isinstance(risk_free, bool):
        rf = (finite(risk_free, "risk_free"),)*len(data)
    else:
        rf = vector(risk_free, "risk_free", min_length=0)
        if len(rf) != len(data):
            raise FinanceInputError("risk-free series must align with returns")
    return tuple(a-b for a, b in zip(data, rf))


def periodize_annual_rate(annual_rate: Number, periods_per_year: Number = 252.0, *, continuous: bool = False) -> float:
    """Convert an annual effective or continuously compounded rate to one period."""
    rate = finite(annual_rate, "annual_rate")
    periods = positive(periods_per_year, "periods_per_year")
    if rate <= -1.0 and not continuous:
        raise FinanceInputError("effective annual rate must exceed -100%")
    return math.expm1(rate/periods) if continuous else (1.0+rate)**(1.0/periods)-1.0


def drawdown_series(returns: Vector, initial_value: Number = 1.0) -> tuple[float, ...]:
    """Drawdown from the running high-water mark of a compounded wealth path."""
    wealth = wealth_index(returns, initial_value)
    peak = positive(initial_value, "initial_value")
    result = []
    for value in wealth:
        peak = max(peak, value)
        result.append(value/peak-1.0)
    return tuple(result)


@dataclass(frozen=True)
class DrawdownEpisode:
    """A peak-to-trough drawdown episode indexed by zero-based return periods."""
    peak_index: int
    trough_index: int
    recovery_index: int | None
    depth: float
    duration: int
    recovery_duration: int | None


def drawdown_episodes(returns: Vector, initial_value: Number = 1.0) -> tuple[DrawdownEpisode, ...]:
    """Identify non-overlapping drawdown episodes and their recovery dates."""
    data = vector(returns, min_length=0)
    wealth = (positive(initial_value, "initial_value"),) + wealth_index(data, initial_value)
    episodes: list[DrawdownEpisode] = []
    peak_idx, peak_value = 0, wealth[0]
    active_peak: int | None = None
    trough_idx, trough_dd = 0, 0.0
    for i in range(1, len(wealth)):
        if wealth[i] >= peak_value:
            if active_peak is not None:
                episodes.append(DrawdownEpisode(active_peak, trough_idx, i-1, trough_dd,
                                                trough_idx-active_peak,
                                                (i-1)-active_peak))
                active_peak = None
            peak_idx, peak_value = i, wealth[i]
        else:
            dd = wealth[i]/peak_value-1.0
            if active_peak is None:
                active_peak = peak_idx
                trough_idx, trough_dd = i, dd
            elif dd < trough_dd:
                trough_idx, trough_dd = i, dd
    if active_peak is not None:
        episodes.append(DrawdownEpisode(active_peak, trough_idx, None, trough_dd,
                                        len(wealth)-1-active_peak, None))
    return tuple(episodes)


def maximum_drawdown(returns: Vector) -> float:
    """Most negative peak-to-trough drawdown."""
    path = drawdown_series(returns)
    return min(path, default=0.0)


def ulcer_index(returns: Vector) -> float:
    """Root mean square percentage drawdown, a duration-sensitive risk measure."""
    drawdowns = drawdown_series(returns)
    return math.sqrt(math.fsum(x*x for x in drawdowns)/len(drawdowns)) if drawdowns else 0.0


def rolling_compound(returns: Vector, window: int) -> tuple[float, ...]:
    """Trailing compounded return for each complete window."""
    data = vector(returns)
    if not 1 <= window <= len(data):
        raise FinanceInputError("window must be in [1, n]")
    return tuple(compound(data[i-window:i]) for i in range(window, len(data)+1))


def rolling_beta(asset: Vector, benchmark: Vector, window: int) -> tuple[float, ...]:
    """Trailing OLS beta from aligned simple returns."""
    from .core import covariance, variance
    x, y = vector(asset), vector(benchmark)
    if len(x) != len(y):
        raise FinanceInputError("asset and benchmark returns must align")
    if not 2 <= window <= len(x):
        raise FinanceInputError("window must be in [2, n]")
    values = []
    for end in range(window, len(x)+1):
        xs, ys = x[end-window:end], y[end-window:end]
        denom = variance(ys)
        if denom <= 0:
            raise FinanceInputError("benchmark window has zero variance")
        values.append(covariance(xs, ys)/denom)
    return tuple(values)


def return_decomposition(price_return: Number, dividend_return: Number = 0.0,
                         fx_return: Number = 0.0) -> dict[str, float]:
    """Multiplicative decomposition of local price, dividend and FX gross returns."""
    p, d, fx = (finite(price_return, "price_return"),
                finite(dividend_return, "dividend_return"), finite(fx_return, "fx_return"))
    if min(p, d, fx) < -1:
        raise FinanceInputError("return components cannot be below -100%")
    gross = (1+p)*(1+d)*(1+fx)
    total = gross-1
    additive = p+d+fx
    return {"price": p, "dividend": d, "fx": fx, "total": total,
            "interaction": total-additive}


def annualization_factor(frequency: str) -> int:
    """Map common observation frequencies to conventional annual counts."""
    mapping = {"daily": 252, "business_daily": 252, "weekly": 52,
               "monthly": 12, "quarterly": 4, "annual": 1, "hourly": 252*6.5}
    key = frequency.lower().strip()
    if key not in mapping:
        raise FinanceInputError(f"unknown frequency: {frequency}")
    return int(mapping[key])
