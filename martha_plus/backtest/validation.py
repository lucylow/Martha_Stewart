"""Research validation helpers to expose parameter and transaction-cost fragility."""
from __future__ import annotations
from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Mapping, Sequence, TypeVar
from ..config import Settings
from ..types import Bar
from .engine import BacktestEngine, BacktestResult

P = TypeVar("P")

@dataclass(frozen=True, slots=True)
class SweepResult:
    parameters: object
    score: float
    metrics: Mapping[str, float | int]
    total_trades: int

@dataclass(frozen=True, slots=True)
class CostScenarioResult:
    commission_bps: float
    slippage_bps: float
    total_return: float
    sharpe: float
    max_drawdown: float
    ending_equity: float


def parameter_sweep(bars: Sequence[Bar], parameter_sets: Sequence[P],
                    strategy_factory: Callable[[P], object],
                    metric: str = "sharpe", settings: Settings | None = None,
                    minimum_history: int = 35) -> tuple[SweepResult, ...]:
    """Run fixed parameter choices on one sample; use walk-forward for selection."""
    if not parameter_sets:
        raise ValueError("parameter_sets cannot be empty")
    results: list[SweepResult] = []
    for parameters in parameter_sets:
        strategy = strategy_factory(parameters)
        result = BacktestEngine(settings or Settings(mode="research")).run(
            bars, strategy, minimum_history=minimum_history)
        if metric not in result.metrics:
            raise ValueError(f"unknown score metric {metric!r}")
        score = float(result.metrics[metric])
        results.append(SweepResult(parameters, score, dict(result.metrics),
                                   int(result.trades.get("trade_count", 0))))
    return tuple(sorted(results, key=lambda item: item.score, reverse=True))


def transaction_cost_sensitivity(bars: Sequence[Bar], strategy_factory: Callable[[], object],
                                 commission_levels: Sequence[float],
                                 slippage_levels: Sequence[float],
                                 base_settings: Settings | None = None,
                                 minimum_history: int = 35) -> tuple[CostScenarioResult, ...]:
    """Report the same strategy under explicitly chosen transaction-cost scenarios."""
    base = base_settings or Settings(mode="research")
    results: list[CostScenarioResult] = []
    for commission in commission_levels:
        for slippage in slippage_levels:
            if commission < 0 or slippage < 0:
                raise ValueError("cost levels must be non-negative")
            config = replace(base, commission_bps=float(commission), slippage_bps=float(slippage))
            config.validate()
            backtest = BacktestEngine(config).run(bars, strategy_factory(),
                                                  minimum_history=minimum_history)
            results.append(CostScenarioResult(float(commission), float(slippage),
                         float(backtest.metrics["total_return"]), float(backtest.metrics["sharpe"]),
                         float(backtest.metrics["max_drawdown"]), backtest.equity_curve[-1].equity))
    return tuple(results)


def block_bootstrap_returns(period_returns: Sequence[float], simulations: int = 1000,
                            sample_size: int | None = None, block_size: int = 1,
                            seed: int = 13) -> dict[str, float | int]:
    """Circular block-bootstrap distribution of compounded returns.

    Blocks preserve some short-range sequencing. The method does not create new
    market regimes and should not be read as a probability of future profitability.
    """
    if not period_returns:
        raise ValueError("period_returns cannot be empty")
    if simulations < 1 or block_size < 1:
        raise ValueError("simulations and block_size must be positive")
    size = sample_size or len(period_returns)
    if size < 1:
        raise ValueError("sample_size must be positive")
    source = [float(value) for value in period_returns]
    if any(value <= -1 for value in source):
        raise ValueError("period returns cannot be <= -100%")
    rng = Random(seed)
    totals: list[float] = []
    for _ in range(simulations):
        sample: list[float] = []
        while len(sample) < size:
            start = rng.randrange(len(source))
            for offset in range(block_size):
                sample.append(source[(start + offset) % len(source)])
                if len(sample) >= size:
                    break
        wealth = 1.0
        for value in sample:
            wealth *= 1.0 + value
        totals.append(wealth - 1.0)
    ordered = sorted(totals)
    return {"simulations": simulations, "sample_size": size,
            "mean_return": sum(ordered) / len(ordered),
            "p05_return": ordered[int((len(ordered)-1) * 0.05)],
            "median_return": ordered[int((len(ordered)-1) * 0.50)],
            "p95_return": ordered[int((len(ordered)-1) * 0.95)],
            "probability_loss": sum(value < 0 for value in ordered) / len(ordered)}
