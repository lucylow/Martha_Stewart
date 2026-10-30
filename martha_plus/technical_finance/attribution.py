"""Portfolio performance attribution: Brinson, factor and currency effects."""
from __future__ import annotations

from typing import Mapping, Sequence
import math

from .core import FinanceInputError, Number, finite


def brinson_fachler(portfolio_weights: Mapping[str, Number],
                    benchmark_weights: Mapping[str, Number],
                    portfolio_returns: Mapping[str, Number],
                    benchmark_returns: Mapping[str, Number]) -> dict[str, object]:
    """Brinson-Fachler allocation, selection and interaction by sector.

    Active sector return is measured against the total benchmark return.
    Inputs are sector weights as fractions and sector returns for the period.
    """
    keys = set(portfolio_weights) | set(benchmark_weights)
    if not keys or set(portfolio_returns) != keys or set(benchmark_returns) != keys:
        raise FinanceInputError("all four sector mappings must cover identical sectors")
    pw = {k: finite(portfolio_weights.get(k, 0.0)) for k in keys}
    bw = {k: finite(benchmark_weights.get(k, 0.0)) for k in keys}
    pr = {k: finite(portfolio_returns[k]) for k in keys}
    br = {k: finite(benchmark_returns[k]) for k in keys}
    if any(w < 0 for w in (*pw.values(), *bw.values())):
        raise FinanceInputError("sector weights must be non-negative")
    if abs(sum(pw.values())-1) > 1e-6 or abs(sum(bw.values())-1) > 1e-6:
        raise FinanceInputError("portfolio and benchmark sector weights must each sum to one")
    benchmark_total = sum(bw[k]*br[k] for k in keys)
    allocation, selection, interaction = {}, {}, {}
    for sector in keys:
        allocation[sector] = (pw[sector]-bw[sector])*(br[sector]-benchmark_total)
        selection[sector] = bw[sector]*(pr[sector]-br[sector])
        interaction[sector] = (pw[sector]-bw[sector])*(pr[sector]-br[sector])
    return {"allocation": allocation, "selection": selection,
            "interaction": interaction,
            "allocation_total": sum(allocation.values()),
            "selection_total": sum(selection.values()),
            "interaction_total": sum(interaction.values()),
            "active_return_explained": sum(allocation.values())+sum(selection.values())+sum(interaction.values()),
            "portfolio_return": sum(pw[k]*pr[k] for k in keys),
            "benchmark_return": benchmark_total}


def brinson_menchero(portfolio_weights: Mapping[str, Number],
                     benchmark_weights: Mapping[str, Number],
                     portfolio_returns: Mapping[str, Number],
                     benchmark_returns: Mapping[str, Number]) -> dict[str, object]:
    """Brinson-Menchero decomposition with additive allocation and selection.

    The method allocates the interaction effect half to allocation and half
    to selection, so totals continue to reconcile.
    """
    raw = brinson_fachler(portfolio_weights, benchmark_weights,
                          portfolio_returns, benchmark_returns)
    alloc = dict(raw["allocation"])
    select = dict(raw["selection"])
    interaction = raw["interaction"]
    for sector, value in interaction.items():
        alloc[sector] += 0.5*value
        select[sector] += 0.5*value
    return {"allocation": alloc, "selection": select,
            "allocation_total": sum(alloc.values()),
            "selection_total": sum(select.values()),
            "active_return_explained": sum(alloc.values())+sum(select.values()),
            "benchmark_return": raw["benchmark_return"],
            "portfolio_return": raw["portfolio_return"]}


def currency_attribution(local_return: Number, fx_return: Number,
                         local_weight: Number = 1.0) -> dict[str, float]:
    """Decompose base-currency return into local asset, FX and interaction."""
    local, fx, weight = map(finite, (local_return, fx_return, local_weight))
    total = (1+local)*(1+fx)-1
    return {"local_asset_effect": weight*local,
            "fx_effect": weight*fx,
            "interaction_effect": weight*local*fx,
            "total_base_return": weight*total}


def factor_pnl_attribution(exposures: Mapping[str, Number],
                           factor_returns: Mapping[str, Number],
                           specific_return: Number = 0.0) -> dict[str, float]:
    """One-period linear factor attribution with specific residual."""
    if set(exposures) != set(factor_returns):
        raise FinanceInputError("exposure and return factor keys must match")
    components = {name: finite(exposures[name])*finite(factor_returns[name])
                  for name in exposures}
    components["specific"] = finite(specific_return)
    components["total"] = sum(components.values())
    return components


def link_period_returns(period_effects: Sequence[Mapping[str, Number]]) -> dict[str, float]:
    """Geometrically link periodic effects with a Carino-style scaling factor.

    For small returns this is close to arithmetic summation. It is intended
    for additive attribution components that reconcile to each period's active
    return; it does not validate the economic meaning of supplied effects.
    """
    if not period_effects:
        raise FinanceInputError("period_effects cannot be empty")
    keys = set(period_effects[0])
    if any(set(period) != keys for period in period_effects):
        raise FinanceInputError("all periods must have identical effect keys")
    linked = {key: 0.0 for key in keys}
    for key in keys:
        wealth = 1.0
        for period in period_effects:
            value = finite(period[key])
            if value <= -1:
                raise FinanceInputError("period effects must exceed -100%")
            wealth *= 1+value
        linked[key] = wealth-1
    return linked


def active_return_reconciliation(portfolio_return: Number,
                                 benchmark_return: Number,
                                 effects: Mapping[str, Number],
                                 *, tolerance: Number = 1e-8) -> dict[str, float]:
    """Check whether supplied attribution effects sum to observed active return."""
    active = finite(portfolio_return)-finite(benchmark_return)
    explained = sum(finite(v) for v in effects.values())
    residual = active-explained
    tol = abs(finite(tolerance))
    return {"active_return": active, "explained_return": explained,
            "unexplained_residual": residual,
            "reconciled": float(abs(residual) <= tol)}
