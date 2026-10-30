"""Historical and conditional volatility estimators for quantitative finance."""
from __future__ import annotations

import math
from typing import Sequence

from .core import FinanceInputError, Number, finite, mean, positive, variance, vector


def realized_variance(returns: Sequence[Number], *, annualization: Number = 1.0) -> float:
    """Sum of squared intraperiod returns, annualized by supplied factor."""
    data = vector(returns)
    scale = positive(annualization, "annualization")
    return sum(x*x for x in data)*scale


def realized_volatility(returns: Sequence[Number], *, annualization: Number = 1.0) -> float:
    """Square root of annualized realized variance."""
    return math.sqrt(realized_variance(returns, annualization=annualization))


def realized_semivariance(returns: Sequence[Number], *, downside: bool = True,
                          annualization: Number = 1.0) -> float:
    """Annualized sum of squared negative or positive intraperiod returns."""
    data = vector(returns)
    scale = positive(annualization)
    selected = [x*x for x in data if (x < 0 if downside else x > 0)]
    return sum(selected)*scale


def bipower_variation(returns: Sequence[Number], *, annualization: Number = 1.0) -> float:
    """Realized bipower variation, a robust estimator of continuous variation."""
    data = vector(returns, min_length=2)
    mu = math.sqrt(2/math.pi)
    return (math.pi/2)*sum(abs(data[i])*abs(data[i-1]) for i in range(1, len(data)))*positive(annualization)


def jump_variation(returns: Sequence[Number], *, annualization: Number = 1.0) -> float:
    """Non-negative realized variance in excess of bipower variation."""
    rv = realized_variance(returns, annualization=annualization)
    bv = bipower_variation(returns, annualization=annualization)
    return max(0.0, rv-bv)


def parkinson_volatility(high: Sequence[Number], low: Sequence[Number],
                         *, annualization: Number = 252.0) -> float:
    """Parkinson range-based volatility estimator from high/low bars."""
    highs, lows = vector(high), vector(low)
    if len(highs) != len(lows) or any(h < l or l <= 0 for h, l in zip(highs, lows)):
        raise FinanceInputError("high/low series must align and satisfy high >= low > 0")
    factor = positive(annualization)
    daily_var = sum(math.log(h/l)**2 for h, l in zip(highs, lows))/(4*len(highs)*math.log(2))
    return math.sqrt(daily_var*factor)


def garman_klass_volatility(open_: Sequence[Number], high: Sequence[Number],
                            low: Sequence[Number], close: Sequence[Number],
                            *, annualization: Number = 252.0) -> float:
    """Garman-Klass OHLC volatility estimator; assumes no opening jumps."""
    o, h, l, c = map(vector, (open_, high, low, close))
    if not (len(o) == len(h) == len(l) == len(c)):
        raise FinanceInputError("OHLC series must align")
    if any(min(a,b,c_,d) <= 0 or b < max(a,d,c_) or c_ > min(a,d) for a,b,c_,d in zip(o,h,l,c)):
        raise FinanceInputError("invalid OHLC bar")
    terms = [0.5*math.log(hi/lo)**2-(2*math.log(2)-1)*math.log(cl/op)**2
             for op, hi, lo, cl in zip(o,h,l,c)]
    daily_var = max(0.0, sum(terms)/len(terms))
    return math.sqrt(daily_var*positive(annualization))


def rogers_satchell_variance(open_: Sequence[Number], high: Sequence[Number],
                             low: Sequence[Number], close: Sequence[Number],
                             *, annualization: Number = 1.0) -> float:
    """Rogers-Satchell per-period variance, robust to drift and opening jumps."""
    o, h, l, c = map(vector, (open_, high, low, close))
    if not (len(o) == len(h) == len(l) == len(c)):
        raise FinanceInputError("OHLC series must align")
    if any(min(a,b,c_,d) <= 0 or b < max(a,d,c_) or c_ > min(a,d) for a,b,c_,d in zip(o,h,l,c)):
        raise FinanceInputError("invalid OHLC bar")
    values = [math.log(hi/cl)*math.log(hi/op)+math.log(lo/cl)*math.log(lo/op)
              for op, hi, lo, cl in zip(o,h,l,c)]
    return max(0.0, sum(values)/len(values))*positive(annualization)


def yang_zhang_volatility(open_: Sequence[Number], high: Sequence[Number],
                          low: Sequence[Number], close: Sequence[Number],
                          *, annualization: Number = 252.0) -> float:
    """Yang-Zhang volatility estimator combining overnight and intraday ranges."""
    o, h, l, c = map(vector, (open_, high, low, close))
    if not (len(o) == len(h) == len(l) == len(c)) or len(o) < 2:
        raise FinanceInputError("aligned OHLC series with at least two bars required")
    if any(min(a,b,c_,d) <= 0 or b < max(a,d,c_) or c_ > min(a,d) for a,b,c_,d in zip(o,h,l,c)):
        raise FinanceInputError("invalid OHLC bar")
    overnight = [math.log(o[i]/c[i-1]) for i in range(1, len(o))]
    open_close = [math.log(c[i]/o[i]) for i in range(len(o))]
    rs = [math.log(h[i]/c[i])*math.log(h[i]/o[i])+math.log(l[i]/c[i])*math.log(l[i]/o[i])
          for i in range(len(o))]
    n = len(overnight)
    vo, vc = variance(overnight), variance(open_close[1:])
    vrs = sum(rs[1:])/n
    k = 0.34/(1.34+(n+1)/(n-1))
    yz = vo+k*vc+(1-k)*vrs
    return math.sqrt(max(0.0, yz)*positive(annualization))


def ewma_variance(returns: Sequence[Number], decay: Number = 0.94,
                  *, initial_variance: Number | None = None) -> tuple[float, ...]:
    """Recursive EWMA variance forecast, with newest observation weighted 1-lambda."""
    data = vector(returns)
    lam = finite(decay)
    if not 0 < lam < 1:
        raise FinanceInputError("decay must be in (0,1)")
    current = variance(data[:min(len(data), 20)]) if initial_variance is None else positive(initial_variance, allow_zero=True)
    result = []
    for value in data:
        current = lam*current+(1-lam)*value*value
        result.append(current)
    return tuple(result)


def ewma_volatility_forecast(returns: Sequence[Number], decay: Number = 0.94,
                             *, annualization: Number = 252.0) -> dict[str, float]:
    """Last EWMA daily sigma and annualized forecast."""
    values = ewma_variance(returns, decay)
    latest = math.sqrt(values[-1])
    return {"period_volatility": latest,
            "annualized_volatility": latest*math.sqrt(positive(annualization)),
            "variance": values[-1]}


def garch11_variance(returns: Sequence[Number], omega: Number,
                     alpha: Number, beta: Number,
                     *, initial_variance: Number | None = None) -> tuple[float, ...]:
    """Compute GARCH(1,1) conditional variances given fixed parameters."""
    data = vector(returns)
    w, a, b = map(finite, (omega, alpha, beta))
    if w <= 0 or a < 0 or b < 0 or a+b >= 1:
        raise FinanceInputError("require omega>0, alpha/beta>=0 and alpha+beta<1")
    current = variance(data) if initial_variance is None else positive(initial_variance)
    result = []
    for value in data:
        current = w+a*value*value+b*current
        result.append(current)
    return tuple(result)


def garch11_unconditional_variance(omega: Number, alpha: Number, beta: Number) -> float:
    """Long-run GARCH variance omega/(1-alpha-beta)."""
    w, a, b = map(finite, (omega, alpha, beta))
    if w <= 0 or a < 0 or b < 0 or a+b >= 1:
        raise FinanceInputError("require omega>0, alpha/beta>=0 and alpha+beta<1")
    return w/(1-a-b)


def garch11_forecast(last_variance: Number, last_return: Number,
                     omega: Number, alpha: Number, beta: Number,
                     horizon: int = 1) -> tuple[float, ...]:
    """Multi-step GARCH variance forecast using unconditional reversion."""
    variance_value, ret = positive(last_variance), finite(last_return)
    w, a, b = map(finite, (omega, alpha, beta))
    if w <= 0 or a < 0 or b < 0 or a+b >= 1 or not isinstance(horizon, int) or horizon < 1:
        raise FinanceInputError("invalid GARCH parameters or forecast horizon")
    long_run = w/(1-a-b)
    next_var = w+a*ret*ret+b*variance_value
    result = [next_var]
    for _ in range(1, horizon):
        next_var = w+(a+b)*next_var
        result.append(next_var)
    # The recursion naturally approaches long-run variance.
    return tuple(result)


def garch11_log_likelihood(returns: Sequence[Number], omega: Number,
                           alpha: Number, beta: Number) -> float:
    """Gaussian quasi-log-likelihood for fixed GARCH(1,1) parameters."""
    data = vector(returns, min_length=3)
    variances = garch11_variance(data, omega, alpha, beta)
    return -0.5*sum(math.log(2*math.pi*v)+r*r/v for r, v in zip(data, variances))


def garch_persistence(alpha: Number, beta: Number) -> float:
    """GARCH(1,1) persistence and approximate half-life in observations."""
    persistence = finite(alpha)+finite(beta)
    if not 0 <= persistence < 1:
        raise FinanceInputError("stationary GARCH persistence must be in [0,1)")
    half_life = math.log(0.5)/math.log(persistence) if persistence > 0 else 0.0
    return half_life


def volatility_cone(returns: Sequence[Number], windows: Sequence[int],
                    *, annualization: Number = 252.0) -> dict[int, dict[str, float]]:
    """Distribution of rolling annualized volatility at several horizons."""
    from .statistics import quantile
    data = vector(returns)
    result = {}
    for window in windows:
        if not isinstance(window, int) or not 2 <= window <= len(data):
            raise FinanceInputError("volatility cone windows must be in [2,n]")
        samples = []
        for end in range(window, len(data)+1):
            sample = data[end-window:end]
            samples.append(math.sqrt(variance(sample))*math.sqrt(positive(annualization)))
        result[window] = {"q10": quantile(samples, 0.10), "q25": quantile(samples, 0.25),
                          "median": quantile(samples, 0.50), "q75": quantile(samples, 0.75),
                          "q90": quantile(samples, 0.90), "latest": samples[-1]}
    return result


def variance_risk_premium(implied_volatility: Number,
                          realized_volatility_value: Number) -> float:
    """Variance premium proxy: implied variance less realized variance."""
    implied, realized = positive(implied_volatility), positive(realized_volatility_value)
    return implied*implied-realized*realized


def volatility_target_scale(current_volatility: Number, target_volatility: Number,
                            *, maximum_leverage: Number = 2.0,
                            minimum_leverage: Number = 0.0) -> float:
    """Exposure scale needed to target a volatility band, capped by leverage."""
    current, target = positive(current_volatility), positive(target_volatility)
    maximum, minimum = finite(maximum_leverage), finite(minimum_leverage)
    if minimum < 0 or maximum < minimum:
        raise FinanceInputError("invalid leverage bounds")
    return min(maximum, max(minimum, target/current))


def volatility_regime(returns: Sequence[Number], window: int = 20,
                      low_quantile: Number = 0.25,
                      high_quantile: Number = 0.75) -> tuple[str, ...]:
    """Classify each complete rolling window relative to in-sample vol quantiles."""
    from .statistics import quantile
    data = vector(returns)
    if not 2 <= window <= len(data):
        raise FinanceInputError("window must be in [2,n]")
    realized = [math.sqrt(variance(data[i-window:i])) for i in range(window, len(data)+1)]
    low, high = quantile(realized, low_quantile), quantile(realized, high_quantile)
    return tuple("low" if v <= low else "high" if v >= high else "normal" for v in realized)


def exponentially_weighted_correlation(x: Sequence[Number], y: Sequence[Number],
                                        decay: Number = 0.94) -> float:
    """EWMA correlation with normalized exponentially decaying weights."""
    left, right = vector(x), vector(y)
    if len(left) != len(right) or len(left) < 2:
        raise FinanceInputError("aligned series with at least two points required")
    lam = finite(decay)
    if not 0 < lam < 1:
        raise FinanceInputError("decay must be in (0,1)")
    n = len(left)
    weights = [(1-lam)*lam**(n-1-i) for i in range(n)]
    total = sum(weights)
    weights = [w/total for w in weights]
    mx = sum(w*xv for w, xv in zip(weights, left))
    my = sum(w*yv for w, yv in zip(weights, right))
    cov = sum(w*(xv-mx)*(yv-my) for w, xv, yv in zip(weights, left, right))
    vx = sum(w*(xv-mx)**2 for w, xv in zip(weights, left))
    vy = sum(w*(yv-my)**2 for w, yv in zip(weights, right))
    if vx <= 0 or vy <= 0:
        raise FinanceInputError("correlation undefined for zero variance")
    return cov/math.sqrt(vx*vy)
