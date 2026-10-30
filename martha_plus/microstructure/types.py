"""Strongly typed primitives shared by the microstructure simulator.

All prices are integer ticks and quantities are integer lots.  Keeping exchange
units integral avoids binary floating-point drift in matching and queue logic.
The package is intentionally independent of brokers and network clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    GTC = "gtc"          # Good until cancelled; residual rests on the book.
    IOC = "ioc"          # Immediate or cancel; residual is cancelled.
    FOK = "fok"          # Fill all immediately or do not execute.
    POST_ONLY = "post_only"  # Reject rather than remove displayed liquidity.


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


class LiquidityRole(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


class EventKind(str, Enum):
    ADD = "add"
    CANCEL = "cancel"
    REDUCE = "reduce"
    TRADE = "trade"
    REPLACE = "replace"
    HALT = "halt"
    RESUME = "resume"


class RejectCode(str, Enum):
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_PRICE = "invalid_price"
    DUPLICATE_ORDER_ID = "duplicate_order_id"
    MARKET_ORDER_WITHOUT_PROTECTION = "market_order_without_protection"
    POST_ONLY_WOULD_TAKE = "post_only_would_take"
    FOK_NOT_FILLABLE = "fok_not_fillable"
    HALTED = "halted"
    RISK_LIMIT = "risk_limit"
    UNKNOWN_ORDER = "unknown_order"
    ORDER_NOT_OPEN = "order_not_open"
    INVALID_AMENDMENT = "invalid_amendment"


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Instrument convention used by the simulator.

    ``tick_size`` converts integer price ticks into currency. ``lot_size`` is
    retained as metadata; quantities passed to the engine are integer lots.
    """

    symbol: str
    tick_size: Decimal = Decimal("0.01")
    lot_size: int = 1
    currency: str = "USD"

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol or any(ch.isspace() for ch in symbol):
            raise ValueError("symbol must be a non-empty token")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        object.__setattr__(self, "symbol", symbol)

    def money(self, price_ticks: int, quantity: int) -> Decimal:
        return self.tick_size * Decimal(price_ticks) * Decimal(quantity * self.lot_size)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    limit_price_ticks: Optional[int] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    max_price_ticks: Optional[int] = None
    min_price_ticks: Optional[int] = None
    strategy_id: str = "manual"
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.client_order_id.strip():
            raise ValueError("client_order_id must not be blank")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.limit_price_ticks is None:
            raise ValueError("limit orders require limit_price_ticks")
        if self.order_type is OrderType.MARKET and self.limit_price_ticks is not None:
            raise ValueError("market orders must not set limit_price_ticks")
        if self.max_price_ticks is not None and self.min_price_ticks is not None:
            if self.min_price_ticks > self.max_price_ticks:
                raise ValueError("min_price_ticks must not exceed max_price_ticks")


@dataclass(slots=True)
class OrderRecord:
    """Mutable lifecycle record for an accepted client order."""

    request: OrderRequest
    submitted_at_ns: int
    accepted_at_ns: int
    status: OrderStatus = OrderStatus.ACCEPTED
    leaves_quantity: int = 0
    filled_quantity: int = 0
    cancelled_quantity: int = 0
    average_price_ticks: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    last_update_ns: int = 0
    reject_code: Optional[RejectCode] = None
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if self.leaves_quantity == 0:
            self.leaves_quantity = self.request.quantity
        if self.last_update_ns == 0:
            self.last_update_ns = self.accepted_at_ns

    @property
    def symbol(self) -> str:
        return self.request.symbol

    @property
    def side(self) -> Side:
        return self.request.side

    @property
    def total_quantity(self) -> int:
        return self.request.quantity

    @property
    def is_open(self) -> bool:
        return not self.status.terminal and self.leaves_quantity > 0

    def apply_fill(self, quantity: int, price_ticks: int, fee: Decimal, ts_ns: int) -> None:
        if quantity <= 0 or quantity > self.leaves_quantity:
            raise ValueError("fill quantity must be positive and no greater than leaves")
        old_notional = self.average_price_ticks * Decimal(self.filled_quantity)
        new_filled = self.filled_quantity + quantity
        self.average_price_ticks = (old_notional + Decimal(price_ticks * quantity)) / Decimal(new_filled)
        self.filled_quantity = new_filled
        self.leaves_quantity -= quantity
        self.fees_paid += fee
        self.last_update_ns = ts_ns
        self.status = OrderStatus.FILLED if self.leaves_quantity == 0 else OrderStatus.PARTIALLY_FILLED

    def cancel_leaves(self, ts_ns: int) -> int:
        quantity = self.leaves_quantity
        self.cancelled_quantity += quantity
        self.leaves_quantity = 0
        self.last_update_ns = ts_ns
        if not self.status.terminal:
            self.status = OrderStatus.CANCELLED
        return quantity


@dataclass(frozen=True, slots=True)
class Execution:
    execution_id: str
    ts_ns: int
    symbol: str
    price_ticks: int
    quantity: int
    maker_order_id: str
    taker_order_id: str
    maker_owner: str
    taker_owner: str
    maker_side: Side
    taker_side: Side
    maker_fee: Decimal = Decimal("0")
    taker_fee: Decimal = Decimal("0")

    @property
    def notional_ticks(self) -> int:
        return self.price_ticks * self.quantity


@dataclass(frozen=True, slots=True)
class OrderAck:
    client_order_id: str
    accepted: bool
    ts_ns: int
    status: OrderStatus
    reason: str = ""
    reject_code: Optional[RejectCode] = None


@dataclass(frozen=True, slots=True)
class BookLevelView:
    price_ticks: int
    quantity: int
    order_count: int


@dataclass(frozen=True, slots=True)
class BookTop:
    symbol: str
    ts_ns: int
    bid: Optional[BookLevelView]
    ask: Optional[BookLevelView]

    @property
    def spread_ticks(self) -> Optional[int]:
        if self.bid is None or self.ask is None:
            return None
        return self.ask.price_ticks - self.bid.price_ticks

    @property
    def mid_ticks(self) -> Optional[Decimal]:
        if self.bid is None or self.ask is None:
            return None
        return (Decimal(self.bid.price_ticks) + Decimal(self.ask.price_ticks)) / 2

    @property
    def is_locked_or_crossed(self) -> bool:
        return bool(self.bid and self.ask and self.bid.price_ticks >= self.ask.price_ticks)


@dataclass(frozen=True, slots=True)
class MarketEvent:
    ts_ns: int
    sequence: int
    symbol: str
    kind: EventKind
    side: Optional[Side] = None
    price_ticks: Optional[int] = None
    quantity: Optional[int] = None
    order_id: str = ""
    reference_price_ticks: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ts_ns < 0 or self.sequence < 0:
            raise ValueError("event time and sequence must be non-negative")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not self.symbol:
            raise ValueError("event symbol must not be blank")
        if self.kind in {EventKind.ADD, EventKind.REDUCE, EventKind.TRADE}:
            if self.side is None:
                raise ValueError(f"{self.kind.value} event requires side")
            if self.quantity is None or self.quantity <= 0:
                raise ValueError(f"{self.kind.value} event requires positive quantity")
        if self.kind in {EventKind.ADD, EventKind.TRADE}:
            if self.price_ticks is None or self.price_ticks <= 0:
                raise ValueError(f"{self.kind.value} event requires positive price")
        if self.kind in {EventKind.ADD, EventKind.CANCEL, EventKind.REDUCE, EventKind.REPLACE}:
            if not self.order_id:
                raise ValueError(f"{self.kind.value} event requires order_id")
