"""Position sizing helpers. None of these functions implies an expected return."""
from __future__ import annotations
from math import floor, isfinite


def fixed_fractional_quantity(equity: float, risk_fraction: float,
                              entry_price: float, stop_price: float,
                              fractional: bool = False) -> float:
    if not all(isfinite(v) for v in (equity, risk_fraction, entry_price, stop_price)):
        raise ValueError("inputs must be finite")
    if equity <= 0 or not 0 < risk_fraction < 1 or entry_price <= 0 or stop_price <= 0:
        raise ValueError("invalid equity, risk fraction, entry, or stop")
    per_unit_risk = abs(entry_price - stop_price)
    if per_unit_risk == 0:
        raise ValueError("entry and stop prices must differ")
    raw = equity * risk_fraction / per_unit_risk
    raw = min(raw, equity / entry_price)
    return raw if fractional else float(floor(raw))


def volatility_scaled_quantity(equity: float, target_volatility: float,
                               estimated_daily_volatility: float, price: float,
                               max_notional_fraction: float = 0.20,
                               fractional: bool = False) -> float:
    values = (equity, target_volatility, estimated_daily_volatility, price, max_notional_fraction)
    if not all(isfinite(v) for v in values):
        raise ValueError("inputs must be finite")
    if equity <= 0 or price <= 0 or estimated_daily_volatility <= 0:
        raise ValueError("equity, price, and volatility must be positive")
    if target_volatility <= 0 or not 0 < max_notional_fraction <= 1:
        raise ValueError("target volatility and cap must be positive")
    target_notional = equity * min(max_notional_fraction, target_volatility / estimated_daily_volatility)
    quantity = target_notional / price
    return quantity if fractional else float(floor(quantity))


def target_weight_quantity(equity: float, current_position_value: float,
                           target_weight: float, price: float,
                           fractional: bool = False) -> float:
    if equity <= 0 or price <= 0 or not isfinite(target_weight):
        raise ValueError("equity and price must be positive and target_weight finite")
    delta_notional = equity * target_weight - current_position_value
    quantity = abs(delta_notional) / price
    return quantity if fractional else float(floor(quantity))


def cap_quantity_by_limits(quantity: float, price: float, max_order_notional: float,
                           max_position_notional: float, current_position_value: float = 0.0,
                           fractional: bool = False) -> float:
    if min(quantity, price, max_order_notional, max_position_notional) < 0 or price == 0:
        raise ValueError("quantity and limits must be non-negative and price positive")
    order_cap = max_order_notional / price
    position_cap = max(0.0, max_position_notional - abs(current_position_value)) / price
    result = min(quantity, order_cap, position_cap)
    return result if fractional else float(floor(result))
