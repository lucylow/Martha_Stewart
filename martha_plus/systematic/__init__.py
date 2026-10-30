"""Systematic research orchestration and point-in-time safe primitives.

The package composes established implementations in ``martha_plus.quant_research``
with strict data-integrity controls. It never connects to brokers or submits orders.
"""
from .adaptive import PageHinkley, conformal_interval
from .backtest import BacktestResult, lagged_signal_backtest
from .integrity import (
    DataIntegrityError,
    PointInTimeMatch,
    UniverseInterval,
    assert_available_by,
    delisting_return,
    members_at,
    point_in_time_join,
)

__all__ = [
    "BacktestResult",
    "DataIntegrityError",
    "PointInTimeMatch",
    "PageHinkley",
    "UniverseInterval",
    "assert_available_by",
    "conformal_interval",
    "delisting_return",
    "lagged_signal_backtest",
    "members_at",
    "point_in_time_join",
]

__version__ = "0.1.0"
