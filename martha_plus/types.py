"""Typed domain objects shared across strategies, risk, and execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Mapping
from uuid import uuid4

from .errors import DataValidationError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError(f"{label} must be timezone-aware")


def _positive(value: float, label: str, *, allow_zero: bool = False) -> None:
    if not isfinite(value) or (value < 0 if allow_zero else value <= 0):
        op = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be finite and {op}")

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in {self.FILLED, self.CANCELLED, self.REJECTED, self.EXPIRED}

@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime = field(default_factory=utc_now)
    bid_size: float = 0.0
    ask_size: float = 0.0
    venue: str = "SIMULATED"

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol or len(symbol) > 32:
            raise DataValidationError("symbol must be non-empty and <= 32 characters")
        object.__setattr__(self, "symbol", symbol)
        _positive(self.bid, "bid")
        _positive(self.ask, "ask")
        _positive(self.bid_size, "bid_size", allow_zero=True)
        _positive(self.ask_size, "ask_size", allow_zero=True)
        _aware(self.timestamp, "quote timestamp")
        if self.ask < self.bid:
            raise DataValidationError("ask cannot be below bid")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return 10_000.0 * self.spread / self.midpoint

@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not self.symbol:
            raise DataValidationError("bar symbol is required")
        _aware(self.timestamp, "bar timestamp")
        for name in ("open", "high", "low", "close"):
            _positive(getattr(self, name), name)
        _positive(self.volume, "volume", allow_zero=True)
        if self.high < max(self.open, self.close, self.low):
            raise DataValidationError("high must be >= open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise DataValidationError("low must be <= open, close, and high")

@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    target_weight: float
    strength: float
    reason: str
    timestamp: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not self.symbol:
            raise DataValidationError("signal symbol is required")
        _aware(self.timestamp, "signal timestamp")
        if not isfinite(self.target_weight):
            raise DataValidationError("target_weight must be finite")
        if not isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise DataValidationError("strength must be between zero and one")
        if not self.reason.strip():
            raise DataValidationError("signal reason is required")

@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    client_order_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utc_now)
    strategy_id: str = "default"
    metadata: Mapping[str, str | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not self.symbol:
            raise DataValidationError("order symbol is required")
        _positive(self.quantity, "quantity")
        _aware(self.created_at, "order created_at")
        if not self.client_order_id.strip():
            raise DataValidationError("client_order_id is required")
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise DataValidationError("limit orders require limit_price")
            _positive(self.limit_price, "limit_price")
        elif self.limit_price is not None:
            raise DataValidationError("market order cannot have limit_price")

@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _positive(self.quantity, "fill quantity")
        _positive(self.price, "fill price")
        _positive(self.commission, "commission", allow_zero=True)
        _aware(self.timestamp, "fill timestamp")

@dataclass(slots=True)
class Order:
    order_id: str
    request: OrderRequest
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    reason: str = ""
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.request.quantity - self.filled_quantity)

@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: float
    average_cost: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.average_cost)

@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    daily_pnl: float
    high_water_mark: float
    positions: Mapping[str, Position]

@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str
    requested_notional: float = 0.0
    allowed_quantity: float = 0.0
    checks: tuple[str, ...] = ()
