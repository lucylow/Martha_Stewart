"""Event-ordered backtesting and performance reporting."""
from .engine import BacktestEngine, BacktestResult
from .metrics import calculate_metrics, trade_statistics
from .validation import parameter_sweep, transaction_cost_sensitivity, block_bootstrap_returns
from .multi_asset import MultiAssetBacktestEngine, MultiAssetBacktestResult
__all__ = ["BacktestEngine", "BacktestResult", "calculate_metrics", "trade_statistics", "parameter_sweep", "transaction_cost_sensitivity", "block_bootstrap_returns", "MultiAssetBacktestEngine", "MultiAssetBacktestResult"]
