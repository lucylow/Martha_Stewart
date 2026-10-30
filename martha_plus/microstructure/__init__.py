"""Paper-only, deterministic market-microstructure research toolkit.

This package is deliberately broker-free.  It provides exchange-style queue
simulation, event replay, calibration, diagnostics, and reproducible scenarios.
It must not be treated as a live execution adapter or as investment advice.
"""
from .types import (
    BookLevelView, BookTop, EventKind, Execution, LiquidityRole, MarketEvent,
    OrderAck, OrderRecord, OrderRequest, OrderStatus, OrderType, RejectCode,
    Side, SymbolSpec, TimeInForce,
)

__all__ = [
    "BookLevelView", "BookTop", "EventKind", "Execution", "LiquidityRole",
    "MarketEvent", "OrderAck", "OrderRecord", "OrderRequest", "OrderStatus",
    "OrderType", "RejectCode", "Side", "SymbolSpec", "TimeInForce",
]
