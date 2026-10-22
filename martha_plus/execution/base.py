"""Execution adapter protocol. Only the paper implementation ships here."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from ..types import Fill, Order, OrderRequest, Quote

@dataclass(frozen=True, slots=True)
class ExecutionResult:
    order: Order
    fills: tuple[Fill, ...] = ()

class ExecutionAdapter(Protocol):
    def execute(self, request: OrderRequest, quote: Quote) -> ExecutionResult: ...
