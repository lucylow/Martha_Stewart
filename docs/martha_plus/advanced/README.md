# Advanced Research and Operations Extension (v0.2.0)

This extension adds a second layer of research and operational tooling under `martha_plus.advanced`. It is implemented with the Python standard library and can be used independently of optional Flask APIs. The addition contains more than 5,100 lines of implementation code, separate from its tests. See [`../ADVANCED_EXTENSION.md`](../ADVANCED_EXTENSION.md) for usage and boundaries.

## Main modules

| Module | Purpose |
|---|---|
| `data_contracts` | Strict reusable schemas, OHLCV validation, timestamp and numeric-field rules |
| `features`, `feature_store`, `feature_diagnostics` | Causal features, immutable JSONL partitions, missingness/drift/correlation diagnostics |
| `portfolio`, `constraints`, `risk_budget`, `portfolio_diagnostics`, `rebalancing` | Mean-variance weights, bounded allocations, risk-parity estimation, drift/turnover, trade deltas |
| `attribution`, `statistical_tests`, `sensitivity`, `walkforward` | Benchmark/factor analysis, bootstrap and multiple-testing diagnostics, parameter-grid checks, purged folds |
| `regimes`, `universe`, `corporate_actions`, `session_calendar` | Interpretable regimes, point-in-time symbol filters, split/dividend helpers, configurable sessions |
| `data_sources`, `data_quality_monitor`, `liquidity` | Provider failover, freshness enforcement, anomaly findings, stylized impact/capacity estimates |
| `order_slicing`, `execution_quality` | TWAP/VWAP/participation schedules and fill-quality reports (schedule only; no venue connection) |
| `eventbus`, `replay`, `scheduler` | In-process events with bounded retry/dead letters, deterministic replay, manual-tick scheduling |
| `alerts`, `incidents`, `operations` | Alert cooldowns, incident lifecycle, readiness probes, error-budget and latency summaries |
| `experiments`, `lineage`, `snapshots`, `config_diff`, `reporting` | Run registry, SHA-256 provenance, atomic checkpoints, redacted drift reports, escaped exports |

## Example: causal feature creation

```python
from datetime import datetime, timedelta, timezone
from martha_plus.types import Bar
from martha_plus.advanced.features import FeatureConfig, FeaturePipeline

start = datetime(2025, 1, 1, tzinfo=timezone.utc)
bars = [
    Bar("DEMO", start + timedelta(days=i), close - 0.5, close + 1,
        close - 1, close, 1000 + i)
    for i, close in enumerate([100, 101, 99, 102, 103, 101, 104, 105])
]
features = FeaturePipeline(FeatureConfig(return_windows=(1, 3))).transform(bars)
labels = FeaturePipeline.forward_returns(bars, horizon=1)
# Features at t are available at t; labels are explicitly forward-looking.
for row, target in zip(features, labels):
    print(row.timestamp, row.get("return_1"), target)
```

## Example: bounded portfolio weights

```python
from martha_plus.advanced.portfolio import PortfolioOptimizer

covariance = {
    "AAA": {"AAA": 0.04, "BBB": 0.01},
    "BBB": {"AAA": 0.01, "BBB": 0.09},
}
result = PortfolioOptimizer(risk_aversion=3.0, max_weight=0.70).optimize(
    {"AAA": 0.10, "BBB": 0.06}, covariance,
    previous_weights={"AAA": 0.50, "BBB": 0.50},
)
print(result.weights, result.converged, result.constraint_violation)
```

Optimization results are diagnostics. Check `converged` and `constraint_violation`, validate covariance assumptions, and independently verify portfolios before use.

## Example: experiment registry

```python
from martha_plus.advanced.experiments import ExperimentRegistry

registry = ExperimentRegistry("data/experiments.sqlite3")
run = registry.create("daily-momentum", {"fast": 8, "slow": 32},
                      code_version="local-git-sha", data_fingerprint="bars-sha256")
registry.start(run.run_id)
registry.complete(run.run_id, {"total_return": 0.03, "max_drawdown": 0.07})
print(registry.get(run.run_id))
registry.close()
```

Use a source-data fingerprint that reflects the actual data vendor, adjustment settings, symbol universe, and sample interval. Never compare runs as though data were identical when their provenance differs.

## Run checks

```bash
python -m compileall -q martha_plus tests
sh scripts/martha_plus/check.sh
python scripts/count_lines.py
```

The optional Flask tests run when the `api` extra is installed. CI installs that extra. All tests use synthetic/local fixtures; there is no required network access.

## Important implementation boundaries

- The optimizer, risk parity routine, statistical diagnostics, and impact model are small-matrix/research implementations, not certified financial models. Review their assumptions and test against an independent reference before use.
- The configured `SessionCalendar` is a weekday calendar with supplied holidays/early closes, **not** an official exchange calendar or holiday service.
- Corporate-action helpers require a complete and correctly sourced action history. Applying a split adjustment to already-adjusted prices can double-adjust them.
- The provider cache's stale-on-error behavior is opt-in. Failover validates timestamps, but callers must still validate adjustment policy, symbol mapping, timezone, and vendor licensing.
- Feature engineering prevents obvious future-index use in built-in rolling features, but the toolkit cannot prove that an externally supplied feature was available at decision time. Audit feature lineage and label windows.
- Order slicing creates schedules only. It does not send orders or connect to an exchange or brokerage.
- `EventBus`, `EventJournal`, `IncidentManager`, and `ManualScheduler` are local/in-process utilities; they are not distributed queues, durable incident databases, or background job daemons.
- Nothing in the package enables live trading. Paper simulation and successful tests do not imply a strategy is profitable or operationally ready.
