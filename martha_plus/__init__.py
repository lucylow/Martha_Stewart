"""Martha Stewart Plus: a safe, testable research and paper-trading toolkit.

This package is additive to the original Martha_Stewart repository. It does not
connect to a live broker and will not place real-money orders.
"""
from .config import Settings
from .types import Bar, Quote, Side, OrderType, OrderStatus, Signal, OrderRequest

__all__ = [
    "Settings", "Bar", "Quote", "Side", "OrderType", "OrderStatus",
    "Signal", "OrderRequest",
]
__version__ = "0.3.0"
