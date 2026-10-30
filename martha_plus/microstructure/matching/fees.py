"""Transparent maker/taker fees with decimal currency arithmetic."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN

from ..types import LiquidityRole, SymbolSpec


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Fee schedule in basis points plus per-lot fees/rebates.

    Positive maker/taker values are charges; negative maker values represent
    rebates. Fees are rounded to the instrument's minor-unit precision only
    when requested by the reporting layer; the simulator retains full Decimal
    precision internally.
    """

    maker_bps: Decimal = Decimal("0")
    taker_bps: Decimal = Decimal("1.0")
    maker_per_lot: Decimal = Decimal("0")
    taker_per_lot: Decimal = Decimal("0")
    minimum_fee: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.minimum_fee < 0:
            raise ValueError("minimum_fee cannot be negative")

    def fee(self, role: LiquidityRole, spec: SymbolSpec, price_ticks: int, quantity: int) -> Decimal:
        if price_ticks <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        notional = spec.money(price_ticks, quantity)
        if role is LiquidityRole.MAKER:
            bps, per_lot = self.maker_bps, self.maker_per_lot
        else:
            bps, per_lot = self.taker_bps, self.taker_per_lot
        charge = notional * bps / Decimal("10000")
        charge += per_lot * Decimal(quantity)
        if charge > 0:
            charge = max(charge, self.minimum_fee)
        return charge

    @staticmethod
    def round_currency(value: Decimal, places: int = 2) -> Decimal:
        quantum = Decimal(1).scaleb(-places)
        return value.quantize(quantum, rounding=ROUND_HALF_EVEN)
