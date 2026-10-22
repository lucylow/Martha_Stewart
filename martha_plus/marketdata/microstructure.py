"""Level-one order-book features computed from top-of-book quotes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from ..types import Quote

@dataclass(frozen=True, slots=True)
class MicrostructureFeatures:
    symbol: str
    midpoint: float
    microprice: float
    spread: float
    spread_bps: float
    depth_imbalance: float
    total_displayed_size: float


def compute_features(quote: Quote) -> MicrostructureFeatures:
    total = quote.bid_size + quote.ask_size
    imbalance = (quote.bid_size - quote.ask_size) / total if total > 0 else 0.0
    if total > 0:
        microprice = (quote.ask * quote.bid_size + quote.bid * quote.ask_size) / total
    else:
        microprice = quote.midpoint
    return MicrostructureFeatures(quote.symbol, quote.midpoint, microprice,
                                  quote.spread, quote.spread_bps, imbalance, total)


def order_flow_imbalance(previous: Quote, current: Quote) -> float:
    """Approximate one-level order-flow imbalance between consecutive observations.

    Quote snapshots must be for one symbol and strictly increase in timestamp.
    This statistic is descriptive and should not be interpreted as an executable
    prediction without appropriate sampling, latency, and venue validation.
    """
    if previous.symbol != current.symbol:
        raise ValueError("quotes must have the same symbol")
    if current.timestamp <= previous.timestamp:
        raise ValueError("quote timestamps must increase")
    if current.bid >= previous.bid:
        bid_term = current.bid_size if current.bid > previous.bid else current.bid_size - previous.bid_size
    else:
        bid_term = -previous.bid_size
    if current.ask <= previous.ask:
        ask_term = -current.ask_size if current.ask < previous.ask else previous.ask_size - current.ask_size
    else:
        ask_term = previous.ask_size
    return bid_term + ask_term


def summarize_quote_sequence(quotes: Sequence[Quote]) -> dict[str, float | int]:
    if not quotes:
        return {"count": 0, "mean_spread_bps": 0.0, "mean_imbalance": 0.0,
                "mean_microprice_edge_bps": 0.0, "cumulative_ofi": 0.0}
    ordered = list(quotes)
    if any(a.timestamp >= b.timestamp for a, b in zip(ordered, ordered[1:])):
        raise ValueError("quotes must be strictly time-ordered")
    if len({quote.symbol for quote in ordered}) != 1:
        raise ValueError("quotes must represent one symbol")
    features = [compute_features(quote) for quote in ordered]
    ofi = sum(order_flow_imbalance(a, b) for a, b in zip(ordered, ordered[1:]))
    return {
        "count": len(ordered),
        "mean_spread_bps": sum(item.spread_bps for item in features) / len(features),
        "mean_imbalance": sum(item.depth_imbalance for item in features) / len(features),
        "mean_microprice_edge_bps": sum((item.microprice / item.midpoint - 1) * 10_000
                                         for item in features) / len(features),
        "cumulative_ofi": ofi,
    }
