"""Portfolio and strategy performance analytics with explicit conventions."""
from __future__ import annotations

import math
from typing import Sequence

from .core import FinanceInputError, Number, Vector, finite, mean, normal_cdf, positive, variance, vector
from .returns import (annualized_return, annualized_volatility, compound,
                      drawdown_episodes, maximum_drawdown, ulcer_index)
from .statistics import downside_deviation, tail_mean


def sharpe_ratio(returns: Vector, risk_free: Vector | Number = 0.0,
                 periods_per_year: Number = 252.0) -> float:
    """Annualized Sharpe based on arithmetic mean excess return / sample sigma."""
    from .returns import excess_returns
    excess = excess_returns(returns, risk_free)
    if len(excess) < 2:
        raise FinanceInputError("Sharpe ratio requires at least two observations")
    sigma = math.sqrt(variance(excess))
    if sigma == 0:
        raise FinanceInputError("Sharpe ratio is undefined for zero volatility")
    return mean(excess)/sigma*math.sqrt(positive(periods_per_year, "periods_per_year"))


def sortino_ratio(returns: Vector, target: Number = 0.0,
                  periods_per_year: Number = 252.0) -> float:
    """Annualized excess mean divided by downside deviation around target."""
    data = vector(returns, min_length=2)
    threshold = finite(target, "target")
    downside = downside_deviation(data, threshold, annualization=periods_per_year)
    if downside == 0:
        raise FinanceInputError("Sortino ratio is undefined for zero downside deviation")
    return (mean(data)-threshold)*positive(periods_per_year, "periods_per_year")/downside


def calmar_ratio(returns: Vector, periods_per_year: Number = 252.0) -> float:
    """Geometric annualized return divided by absolute maximum drawdown."""
    ann = annualized_return(returns, periods_per_year)
    dd = abs(maximum_drawdown(returns))
    if dd == 0:
        raise FinanceInputError("Calmar ratio is undefined without a drawdown")
    return ann/dd


def omega_ratio(returns: Vector, threshold: Number = 0.0) -> float:
    """Ratio of gains above a threshold to losses below it."""
    data = vector(returns)
    hurdle = finite(threshold, "threshold")
    gains = math.fsum(max(0.0, x-hurdle) for x in data)
    losses = math.fsum(max(0.0, hurdle-x) for x in data)
    if losses == 0:
        raise FinanceInputError("Omega ratio is undefined with no observations below threshold")
    return gains/losses


def information_ratio(returns: Vector, benchmark: Vector,
                      periods_per_year: Number = 252.0) -> float:
    """Annualized active mean / tracking error."""
    active = _active(returns, benchmark)
    if len(active) < 2:
        raise FinanceInputError("information ratio requires at least two observations")
    tracking = math.sqrt(variance(active))
    if tracking == 0:
        raise FinanceInputError("information ratio undefined at zero tracking error")
    return mean(active)/tracking*math.sqrt(positive(periods_per_year, "periods_per_year"))


def tracking_error(returns: Vector, benchmark: Vector,
                   periods_per_year: Number = 252.0) -> float:
    """Annualized sample standard deviation of active returns."""
    active = _active(returns, benchmark)
    if len(active) < 2:
        raise FinanceInputError("tracking error requires at least two observations")
    return math.sqrt(variance(active))*math.sqrt(positive(periods_per_year, "periods_per_year"))


def hit_rate(returns: Vector, *, include_zero: bool = False) -> float:
    """Fraction of non-zero (or all) periods with positive return."""
    data = vector(returns)
    sample = data if include_zero else tuple(x for x in data if x != 0)
    if not sample:
        raise FinanceInputError("hit rate has no eligible observations")
    return sum(x > 0 for x in sample)/len(sample)


def gain_to_pain(returns: Vector) -> float:
    """Sum of positive returns divided by absolute sum of negative returns."""
    data = vector(returns)
    gains = math.fsum(x for x in data if x > 0)
    pains = -math.fsum(x for x in data if x < 0)
    if pains == 0:
        raise FinanceInputError("gain-to-pain is undefined without losses")
    return gains/pains


def payoff_ratio(returns: Vector) -> float:
    """Average positive observation divided by average absolute negative one."""
    data = vector(returns)
    wins = [x for x in data if x > 0]
    losses = [-x for x in data if x < 0]
    if not wins or not losses:
        raise FinanceInputError("payoff ratio requires both wins and losses")
    return (math.fsum(wins)/len(wins))/(math.fsum(losses)/len(losses))


def profit_factor(returns: Vector) -> float:
    """Gross winning return divided by gross losing return."""
    return gain_to_pain(returns)


def annualized_skewness(returns: Vector) -> float:
    """Return sample skewness without annualization (provided for discoverability)."""
    from .statistics import moments
    return moments(returns)["skewness"]


def sterling_ratio(returns: Vector, periods_per_year: Number = 252.0,
                   top_n: int = 3) -> float:
    """Annualized return divided by the mean absolute depth of N worst episodes."""
    if top_n < 1:
        raise FinanceInputError("top_n must be at least one")
    episodes = sorted(drawdown_episodes(returns), key=lambda e: e.depth)[:top_n]
    if not episodes:
        raise FinanceInputError("Sterling ratio requires a drawdown episode")
    denominator = math.fsum(abs(e.depth) for e in episodes)/len(episodes)
    if denominator == 0:
        raise FinanceInputError("Sterling ratio is undefined at zero drawdown")
    return annualized_return(returns, periods_per_year)/denominator


def martin_ratio(returns: Vector, periods_per_year: Number = 252.0) -> float:
    """Annualized return divided by the Ulcer Index."""
    ui = ulcer_index(returns)
    if ui == 0:
        raise FinanceInputError("Martin ratio is undefined at zero Ulcer Index")
    return annualized_return(returns, periods_per_year)/ui


def tail_ratio(returns: Vector, q: Number = 0.05) -> float:
    """Upper-tail mean divided by absolute lower-tail mean."""
    upper, lower = tail_mean(returns, q, lower=False), tail_mean(returns, q, lower=True)
    if lower == 0:
        raise FinanceInputError("tail ratio is undefined when lower-tail mean is zero")
    return upper/abs(lower)


def trade_expectancy(trade_returns: Vector) -> dict[str, float]:
    """Per-trade arithmetic expectancy and win/loss decomposition."""
    data = vector(trade_returns)
    wins, losses = [x for x in data if x > 0], [x for x in data if x < 0]
    return {
        "trade_count": float(len(data)),
        "expectancy": mean(data),
        "win_rate": len(wins)/len(data),
        "average_win": math.fsum(wins)/len(wins) if wins else 0.0,
        "average_loss": math.fsum(losses)/len(losses) if losses else 0.0,
        "largest_win": max(wins, default=0.0),
        "largest_loss": min(losses, default=0.0),
    }


def performance_report(returns: Vector, *, benchmark: Vector | None = None,
                       risk_free: Vector | Number = 0.0,
                       periods_per_year: Number = 252.0) -> dict[str, float]:
    """Build a compact report; optional benchmark metrics require aligned data."""
    data = vector(returns, min_length=2)
    periods = positive(periods_per_year, "periods_per_year")
    result = {
        "total_return": compound(data),
        "annualized_return": annualized_return(data, periods),
        "annualized_volatility": annualized_volatility(data, periods),
        "max_drawdown": maximum_drawdown(data),
        "ulcer_index": ulcer_index(data),
        "sharpe": sharpe_ratio(data, risk_free, periods),
        "hit_rate": hit_rate(data),
        "gain_to_pain": gain_to_pain(data),
    }
    try:
        result["sortino"] = sortino_ratio(data, periods_per_year=periods)
    except FinanceInputError:
        result["sortino"] = math.nan
    if benchmark is not None:
        result["tracking_error"] = tracking_error(data, benchmark, periods)
        result["information_ratio"] = information_ratio(data, benchmark, periods)
    return result


def _active(returns: Vector, benchmark: Vector) -> tuple[float, ...]:
    left, right = vector(returns), vector(benchmark)
    if len(left) != len(right):
        raise FinanceInputError("returns and benchmark must have equal lengths")
    return tuple(a-b for a, b in zip(left, right))


def rolling_sharpe(returns: Vector, window: int, *, periods_per_year: Number = 252.0) -> tuple[float, ...]:
    """Trailing annualized Sharpe ratios for full windows."""
    from .core import variance
    data = vector(returns)
    if not 2 <= window <= len(data):
        raise FinanceInputError("window must be in [2, n]")
    scale = math.sqrt(positive(periods_per_year, "periods_per_year"))
    values = []
    for end in range(window, len(data)+1):
        sample = data[end-window:end]
        sd = math.sqrt(variance(sample))
        if sd == 0:
            values.append(math.nan)
        else:
            values.append(mean(sample)/sd*scale)
    return tuple(values)


def alpha_beta(returns: Vector, benchmark: Vector,
               periods_per_year: Number = 252.0) -> dict[str, float]:
    """Single-factor arithmetic regression expressed as annualized alpha and beta."""
    from .regression import ols
    result = ols(returns, (benchmark,), intercept=True)
    return {"alpha_period": result.coefficients[0],
            "alpha_annualized_arithmetic": result.coefficients[0]*positive(periods_per_year),
            "beta": result.coefficients[1], "r_squared": result.r_squared}
