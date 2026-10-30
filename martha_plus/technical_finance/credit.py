"""Credit curves, CDS valuation building blocks and structural credit metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import FinanceInputError, Number, finite, positive
from .fixed_income.curves import DiscountCurve


@dataclass(frozen=True)
class CreditMetrics:
    hazard_rate: float
    survival_probability: float
    default_probability: float
    expected_loss_rate: float


def survival_probability(hazard_rate: Number, maturity: Number) -> float:
    """Survival under a constant continuously compounded hazard rate."""
    hazard, t = finite(hazard_rate), positive(maturity, "maturity", allow_zero=True)
    if hazard < 0:
        raise FinanceInputError("hazard_rate must be non-negative")
    return math.exp(-hazard*t)


def default_probability(hazard_rate: Number, maturity: Number) -> float:
    """Cumulative default probability under constant hazard."""
    return 1-survival_probability(hazard_rate, maturity)


def hazard_from_spread(cds_spread: Number, recovery_rate: Number) -> float:
    """Approximate flat hazard = CDS spread / (1 - recovery)."""
    spread, recovery = positive(cds_spread, "cds_spread", allow_zero=True), finite(recovery_rate)
    if not 0 <= recovery < 1:
        raise FinanceInputError("recovery_rate must be in [0, 1)")
    if recovery == 1 and spread > 0:
        raise FinanceInputError("positive spread is inconsistent with full recovery")
    return 0.0 if recovery == 1 else spread/(1-recovery)


def credit_metrics(spread: Number, recovery: Number, maturity: Number,
                   exposure_at_default: Number) -> CreditMetrics:
    """Flat-hazard survival, default probability and simple expected loss."""
    hazard = hazard_from_spread(spread, recovery)
    t, ead = positive(maturity), positive(exposure_at_default, allow_zero=True)
    pd = default_probability(hazard, t)
    loss_rate = pd*(1-finite(recovery))
    return CreditMetrics(hazard, 1-pd, pd, ead*loss_rate)


def cds_risky_annuity(curve: DiscountCurve, maturity: Number,
                      payment_frequency: int = 4,
                      hazard_rate: Number = 0.0) -> float:
    """Quarterly/semiannual premium-leg risky annuity with midpoint survival."""
    t, hazard = positive(maturity), finite(hazard_rate)
    if hazard < 0 or not isinstance(payment_frequency, int) or payment_frequency < 1:
        raise FinanceInputError("invalid CDS hazard or payment frequency")
    periods = int(round(t*payment_frequency))
    if abs(periods/payment_frequency-t) > 1e-8:
        raise FinanceInputError("maturity must align with payment frequency")
    annuity = 0.0
    for i in range(1, periods+1):
        pay_t = i/payment_frequency
        mid_t = pay_t-0.5/payment_frequency
        annuity += curve.discount_factor(pay_t)*math.exp(-hazard*mid_t)/payment_frequency
    return annuity


def cds_protection_leg(curve: DiscountCurve, maturity: Number,
                       hazard_rate: Number, recovery_rate: Number,
                       payment_frequency: int = 4) -> float:
    """Approximate discounted protection-leg PV via midpoint default density."""
    t, hazard, recovery = positive(maturity), finite(hazard_rate), finite(recovery_rate)
    if hazard < 0 or not 0 <= recovery < 1:
        raise FinanceInputError("invalid hazard or recovery")
    if not isinstance(payment_frequency, int) or payment_frequency < 1:
        raise FinanceInputError("payment_frequency must be positive")
    periods = int(round(t*payment_frequency))
    if abs(periods/payment_frequency-t) > 1e-8:
        raise FinanceInputError("maturity must align with payment frequency")
    pv = 0.0
    for i in range(1, periods+1):
        start, end = (i-1)/payment_frequency, i/payment_frequency
        mid = (start+end)/2
        default_mass = math.exp(-hazard*start)-math.exp(-hazard*end)
        pv += (1-recovery)*curve.discount_factor(mid)*default_mass
    return pv


def fair_cds_spread(curve: DiscountCurve, maturity: Number,
                    hazard_rate: Number, recovery_rate: Number,
                    payment_frequency: int = 4) -> float:
    """Par CDS premium rate = protection PV / risky premium annuity."""
    protection = cds_protection_leg(curve, maturity, hazard_rate, recovery_rate, payment_frequency)
    annuity = cds_risky_annuity(curve, maturity, payment_frequency, hazard_rate)
    if annuity <= 0:
        raise FinanceInputError("risky annuity must be positive")
    return protection/annuity


def cds_spread_from_flat_hazard(curve: DiscountCurve, maturity: Number,
                                spread: Number, recovery_rate: Number,
                                payment_frequency: int = 4) -> float:
    """Convert a quoted spread into a flat hazard using a first-order approximation."""
    return hazard_from_spread(spread, recovery_rate)


def expected_credit_loss(pd: Number, loss_given_default: Number,
                         exposure_at_default: Number) -> float:
    """Expected credit loss = PD * LGD * EAD."""
    p, lgd, ead = finite(pd), finite(loss_given_default), finite(exposure_at_default)
    if not 0 <= p <= 1 or not 0 <= lgd <= 1 or ead < 0:
        raise FinanceInputError("PD/LGD must be in [0,1] and EAD non-negative")
    return p*lgd*ead


def merton_distance_to_default(asset_value: Number, debt_face: Number,
                               asset_volatility: Number, rate: Number,
                               maturity: Number) -> dict[str, float]:
    """Simplified Merton structural model distance to default and equity value.

    The equity is treated as a call on firm assets with debt face as strike.
    This is a model diagnostic, not a credit rating or default forecast.
    """
    a, d, sigma, r, t = positive(asset_value), positive(debt_face), positive(asset_volatility), finite(rate), positive(maturity)
    d1 = (math.log(a/d)+(r+0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
    d2 = d1-sigma*math.sqrt(t)
    from .core import normal_cdf
    equity = a*normal_cdf(d1)-d*math.exp(-r*t)*normal_cdf(d2)
    return {"distance_to_default": d2, "risk_neutral_default_probability": normal_cdf(-d2),
            "model_equity_value": equity, "asset_value": a}


def transition_matrix_check(matrix: Sequence[Sequence[Number]]) -> tuple[tuple[float, ...], ...]:
    """Validate a credit-rating transition matrix is square and row-stochastic."""
    rows = tuple(tuple(finite(v) for v in row) for row in matrix)
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise FinanceInputError("transition matrix must be non-empty and square")
    for i, row in enumerate(rows):
        if any(p < 0 or p > 1 for p in row) or abs(sum(row)-1.0) > 1e-8:
            raise FinanceInputError(f"transition row {i} must contain probabilities summing to one")
    return rows


def migrate_rating_probabilities(initial: Sequence[Number],
                                 transition: Sequence[Sequence[Number]],
                                 periods: int = 1) -> tuple[float, ...]:
    """Propagate rating state probabilities through a transition matrix."""
    state = tuple(finite(v) for v in initial)
    matrix = transition_matrix_check(transition)
    if len(state) != len(matrix) or any(v < 0 for v in state) or abs(sum(state)-1) > 1e-8:
        raise FinanceInputError("initial probabilities must align and sum to one")
    if not isinstance(periods, int) or periods < 0:
        raise FinanceInputError("periods must be a non-negative integer")
    for _ in range(periods):
        state = tuple(sum(state[i]*matrix[i][j] for i in range(len(state))) for j in range(len(state)))
    return state


def rating_expected_loss(initial: Sequence[Number], transition: Sequence[Sequence[Number]],
                         default_index: int, recovery: Number, ead: Number,
                         periods: int = 1) -> float:
    """Expected loss from a one-period rating migration distribution."""
    probabilities = migrate_rating_probabilities(initial, transition, periods)
    if not 0 <= default_index < len(probabilities):
        raise FinanceInputError("default_index outside rating states")
    return probabilities[default_index]*(1-finite(recovery))*positive(ead, allow_zero=True)
