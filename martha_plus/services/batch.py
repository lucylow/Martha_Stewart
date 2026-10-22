"""Batch processing with per-symbol isolation and structured error summaries."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .engine import TickResult, TradingResearchEngine

@dataclass(frozen=True, slots=True)
class SymbolFailure:
    symbol: str
    error_type: str
    message: str

@dataclass(frozen=True, slots=True)
class BatchResult:
    results: tuple[TickResult, ...]
    failures: tuple[SymbolFailure, ...]
    requested_count: int

    @property
    def succeeded_count(self) -> int:
        return len(self.results)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    def to_dict(self) -> dict:
        return {
            "requested_count": self.requested_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "results": [{"symbol": r.symbol, "decision": r.decision, "reason": r.reason,
                         "order_id": r.order_id, "filled_quantity": r.filled_quantity,
                         "fill_price": r.fill_price, "signal_reason": r.signal.reason}
                        for r in self.results],
            "failures": [{"symbol": f.symbol, "error_type": f.error_type, "message": f.message}
                         for f in self.failures],
        }

class BatchProcessor:
    def __init__(self, engine: TradingResearchEngine):
        self.engine = engine

    def run(self, symbols: Sequence[str]) -> BatchResult:
        normalized = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate symbols are not allowed in one batch")
        successes: list[TickResult] = []
        failures: list[SymbolFailure] = []
        for symbol in normalized:
            try:
                successes.append(self.engine.process(symbol))
            except Exception as exc:
                # A single broken provider/symbol should not prevent unrelated symbols
                # from receiving a result. Details are returned to the caller for audit.
                failures.append(SymbolFailure(symbol, type(exc).__name__, str(exc)[:500]))
        return BatchResult(tuple(successes), tuple(failures), len(normalized))
