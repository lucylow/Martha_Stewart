"""Extreme-tail diagnostics and peaks-over-threshold risk approximations."""
from __future__ import annotations

import math
from typing import Sequence

from .core import FinanceInputError, Number, finite, mean, positive, quantile, vector
from .statistics import median


def exceedances(losses: Sequence[Number], threshold: Number) -> tuple[float, ...]:
    """Return positive loss exceedances above a selected threshold."""
    data = vector(losses)
    cutoff = finite(threshold)
    if any(loss < 0 for loss in data):
        raise FinanceInputError("loss inputs should be non-negative loss magnitudes")
    return tuple(loss-cutoff for loss in data if loss > cutoff)


def hill_tail_index(losses: Sequence[Number], tail_fraction: Number = 0.1) -> float:
    """Hill estimator of the Pareto tail index from the largest observations.

    The estimator is sensitive to threshold choice and sample size. The
    returned value is alpha, where survival tail scales approximately x^-alpha.
    """
    data = sorted(vector(losses), reverse=True)
    fraction = finite(tail_fraction)
    if not 0 < fraction < 1:
        raise FinanceInputError("tail_fraction must be in (0,1)")
    count = max(2, int(math.ceil(len(data)*fraction)))
    tail = data[:count]
    threshold = data[count] if count < len(data) else data[-1]
    if threshold <= 0 or any(x <= 0 for x in tail):
        raise FinanceInputError("Hill estimator requires positive losses")
    average_log = sum(math.log(x/threshold) for x in tail)/len(tail)
    if average_log <= 0:
        raise FinanceInputError("selected tail has no positive log excess")
    return 1/average_log


def gpd_moments(excesses: Sequence[Number]) -> dict[str, float]:
    """Method-of-moments GPD parameter estimates for non-negative excesses.

    For shape xi < 0.5, Var(X)=beta^2/((1-xi)^2*(1-2xi)). The method-of-moments
    formulas are approximate and are not a substitute for constrained MLE.
    """
    data = vector(excesses, min_length=3)
    if any(x < 0 for x in data):
        raise FinanceInputError("GPD excesses must be non-negative")
    m1 = mean(data)
    m2 = sum((x-m1)**2 for x in data)/len(data)
    if m1 <= 0 or m2 <= 0:
        raise FinanceInputError("GPD moments require positive mean and variance")
    ratio = m1*m1/m2
    shape = 0.5*(1-ratio)
    if shape >= 0.5:
        raise FinanceInputError("moment estimate implies infinite GPD variance")
    scale = m1*(1-shape)
    if scale <= 0:
        raise FinanceInputError("invalid GPD scale estimate")
    return {"shape": shape, "scale": scale,
            "mean_excess": scale/(1-shape),
            "variance_excess": scale*scale/((1-shape)**2*(1-2*shape))}


def gpd_survival(excess: Number, shape: Number, scale: Number) -> float:
    """GPD survival probability for a non-negative threshold excess."""
    x, xi, beta = finite(excess), finite(shape), positive(scale)
    if x < 0:
        raise FinanceInputError("excess must be non-negative")
    if abs(xi) < 1e-10:
        return math.exp(-x/beta)
    support = 1+xi*x/beta
    if support <= 0:
        return 0.0
    return support**(-1/xi)


def gpd_quantile(probability: Number, shape: Number, scale: Number) -> float:
    """GPD quantile for a cumulative probability in (0,1)."""
    p, xi, beta = finite(probability), finite(shape), positive(scale)
    if not 0 < p < 1:
        raise FinanceInputError("probability must be in (0,1)")
    if abs(xi) < 1e-10:
        return -beta*math.log(1-p)
    return beta/xi*((1-p)**(-xi)-1)


def gpd_expected_shortfall(confidence: Number, threshold: Number,
                           exceedance_probability: Number,
                           shape: Number, scale: Number) -> float:
    """POT expected shortfall above a threshold for shape xi < 1."""
    p, u, pu, xi, beta = map(finite, (confidence, threshold,
                                      exceedance_probability, shape, scale))
    if not 0.5 < p < 1 or not 0 < pu < 1 or xi >= 1 or beta <= 0:
        raise FinanceInputError("invalid ES confidence, tail probability or GPD parameters")
    tail_probability = (1-p)/pu
    if not 0 < tail_probability < 1:
        raise FinanceInputError("confidence lies outside modeled exceedance tail")
    quantile_excess = gpd_quantile(1-tail_probability, xi, beta)
    var = u+quantile_excess
    return var+(beta+xi*(var-u))/(1-xi)


def peaks_over_threshold_var(losses: Sequence[Number],
                             confidence: Number = 0.99,
                             threshold_quantile: Number = 0.9) -> dict[str, float]:
    """Estimate VaR and ES with a method-of-moments GPD tail fit.

    The body is not modeled parametrically: empirical threshold exceedance
    frequency scales the conditional GPD survival probability.
    """
    data = vector(losses, min_length=10)
    if any(x < 0 for x in data):
        raise FinanceInputError("losses must be non-negative")
    p = finite(confidence)
    if not 0.9 < p < 1:
        raise FinanceInputError("confidence must be in (0.9,1)")
    threshold_q = finite(threshold_quantile)
    if not 0.5 < threshold_q < 1:
        raise FinanceInputError("threshold_quantile must be in (0.5,1)")
    threshold = quantile(data, threshold_q)
    excess = exceedances(data, threshold)
    if len(excess) < 3:
        raise FinanceInputError("not enough exceedances to fit a GPD")
    params = gpd_moments(excess)
    pu = len(excess)/len(data)
    tail_prob = (1-p)/pu
    if not 0 < tail_prob < 1:
        raise FinanceInputError("confidence level exceeds resolution of threshold sample")
    var = threshold+gpd_quantile(1-tail_prob, params["shape"], params["scale"])
    es = var+(params["scale"]+params["shape"]*(var-threshold))/(1-params["shape"])
    return {"var": var, "expected_shortfall": es,
            "threshold": threshold, "threshold_quantile": threshold_q,
            "exceedance_probability": pu, "exceedance_count": float(len(excess)),
            "shape": params["shape"], "scale": params["scale"]}


def conditional_drawdown_at_risk(returns: Sequence[Number],
                                 confidence: Number = 0.95) -> float:
    """Empirical quantile of drawdown depth across observed drawdown episodes."""
    from .returns import drawdown_episodes
    episodes = drawdown_episodes(returns)
    if not episodes:
        return 0.0
    depth_values = [-episode.depth for episode in episodes]
    return quantile(depth_values, confidence)


def expected_drawdown(returns: Sequence[Number]) -> float:
    """Mean absolute drawdown episode depth."""
    from .returns import drawdown_episodes
    episodes = drawdown_episodes(returns)
    return mean([-episode.depth for episode in episodes]) if episodes else 0.0


def tail_ratio_from_losses(losses: Sequence[Number],
                           confidence: Number = 0.95) -> dict[str, float]:
    """Summarize loss quantile, tail mean, median and maximum."""
    data = vector(losses)
    if any(x < 0 for x in data):
        raise FinanceInputError("losses must be non-negative")
    p = finite(confidence)
    if not 0.5 < p < 1:
        raise FinanceInputError("confidence must be in (0.5,1)")
    var = quantile(data, p)
    tail = [x for x in data if x >= var]
    return {"var": var, "tail_mean": mean(tail), "median_loss": median(data),
            "maximum_loss": max(data)}


def tail_dependence_count(x: Sequence[Number], y: Sequence[Number],
                          quantile_level: Number = 0.95) -> dict[str, float]:
    """Empirical joint and conditional upper-tail exceedance rates."""
    left, right = vector(x), vector(y)
    if len(left) != len(right):
        raise FinanceInputError("series must align")
    q = finite(quantile_level)
    if not 0.5 < q < 1:
        raise FinanceInputError("quantile_level must be in (0.5,1)")
    x_cut, y_cut = quantile(left, q), quantile(right, q)
    x_exc = [a > x_cut for a in left]
    y_exc = [b > y_cut for b in right]
    joint = sum(a and b for a,b in zip(x_exc,y_exc))/len(left)
    x_rate = sum(x_exc)/len(left)
    y_rate = sum(y_exc)/len(left)
    return {"joint_exceedance_rate": joint,
            "conditional_y_given_x": joint/x_rate if x_rate else math.nan,
            "conditional_x_given_y": joint/y_rate if y_rate else math.nan,
            "x_tail_rate": x_rate, "y_tail_rate": y_rate}


def tail_event_flag(losses: Sequence[Number], threshold: Number) -> tuple[bool, ...]:
    """Boolean flags indicating observations exceeding a fixed loss threshold."""
    data = vector(losses)
    cutoff = finite(threshold)
    return tuple(value > cutoff for value in data)
