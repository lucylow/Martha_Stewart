"""Aggregated option-book Greeks, scenario P&L and volatility exposures."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..core import FinanceInputError, Number, finite, positive
from .black_scholes import greeks, price


def aggregate_greeks(positions: Sequence[Mapping[str, Number | str]]) -> dict[str, float]:
    """Aggregate Black-Scholes Greeks for a list of option positions.

    Each position requires spot, strike, time, rate, volatility, quantity,
    multiplier and option_type. Quantity is signed: long positive, short negative.
    """
    total = {name: 0.0 for name in ("market_value", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga")}
    for i, item in enumerate(positions):
        required = ("spot", "strike", "time", "rate", "volatility", "quantity", "multiplier", "option_type")
        missing = [key for key in required if key not in item]
        if missing:
            raise FinanceInputError(f"position {i} missing: {', '.join(missing)}")
        qty, multiplier = finite(item["quantity"]), positive(item["multiplier"])
        kwargs = {"option_type": str(item["option_type"]),
                  "dividend_yield": finite(item.get("dividend_yield", 0.0))}
        value = price(item["spot"], item["strike"], item["time"], item["rate"], item["volatility"], **kwargs)
        g = greeks(item["spot"], item["strike"], item["time"], item["rate"], item["volatility"], **kwargs)
        scale = qty*multiplier
        total["market_value"] += value*scale
        for name in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga"):
            total[name] += getattr(g, name)*scale
    return total


def option_book_scenario(positions: Sequence[Mapping[str, Number | str]],
                         spot_shocks: Mapping[str, Number],
                         vol_shocks: Mapping[str, Number],
                         *, time_decay: Number = 0.0) -> dict[str, float]:
    """Aggregate first/second-order option P&L by underlying symbol.

    Position dictionaries must include the keys used by aggregate_greeks plus
    ``symbol``. Spot shocks are relative returns; volatility shocks are absolute
    volatility points. The report is a Greek approximation, not full repricing.
    """
    decay = finite(time_decay)
    result: dict[str, float] = {}
    for item in positions:
        symbol = str(item.get("symbol", "")).upper()
        if not symbol or symbol not in spot_shocks or symbol not in vol_shocks:
            raise FinanceInputError("each option position requires a symbol with spot and vol shocks")
        g = greeks(item["spot"], item["strike"], item["time"], item["rate"], item["volatility"],
                   option_type=str(item["option_type"]),
                   dividend_yield=finite(item.get("dividend_yield", 0.0)))
        ds = finite(item["spot"])*finite(spot_shocks[symbol])
        dv = finite(vol_shocks[symbol])
        dt = decay
        scale = finite(item["quantity"])*positive(item["multiplier"])
        pnl = (g.delta*ds+0.5*g.gamma*ds*ds+g.vega*dv+g.theta*dt)*scale
        result[symbol] = result.get(symbol, 0.0)+pnl
    result["total"] = sum(result.values())
    return result


def delta_dollars(delta: Number, spot: Number, quantity: Number,
                  multiplier: Number = 1.0) -> float:
    """Dollar delta exposure for an option or option position."""
    return finite(delta)*positive(spot)*finite(quantity)*positive(multiplier)


def vega_dollars(vega: Number, quantity: Number,
                 multiplier: Number = 1.0, vol_point: Number = 0.01) -> float:
    """P&L per one volatility point (default one percentage point)."""
    return finite(vega)*finite(quantity)*positive(multiplier)*positive(vol_point)


def gamma_dollars(gamma: Number, spot: Number, quantity: Number,
                  multiplier: Number = 1.0) -> float:
    """Dollar gamma exposure convention: gamma * spot squared * contract units."""
    return finite(gamma)*positive(spot)**2*finite(quantity)*positive(multiplier)


def put_call_skew(call_iv: Number, put_iv: Number) -> float:
    """Put implied volatility less call implied volatility at matched delta/tenor."""
    return finite(put_iv)-finite(call_iv)


def risk_reversal(call_iv: Number, put_iv: Number) -> float:
    """25-delta risk reversal convention: call IV minus put IV."""
    return finite(call_iv)-finite(put_iv)


def butterfly_volatility(put_wing_iv: Number, atm_iv: Number,
                         call_wing_iv: Number) -> float:
    """Symmetric smile butterfly: average wing volatility less ATM volatility."""
    return 0.5*(finite(put_wing_iv)+finite(call_wing_iv))-finite(atm_iv)


def delta_neutrality_report(greeks_by_underlying: Mapping[str, Number],
                            *, tolerance: Number = 1.0) -> dict[str, object]:
    """Report delta exposure by underlying and whether the book is neutral."""
    values = {k: finite(v) for k,v in greeks_by_underlying.items()}
    tol = positive(tolerance, "tolerance", allow_zero=True)
    total = sum(values.values())
    return {"delta_by_underlying": values, "net_delta": total,
            "gross_delta": sum(abs(v) for v in values.values()),
            "delta_neutral": abs(total) <= tol}
