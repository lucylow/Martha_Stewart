# Martha Stewart Plus — Research, Risk, and Paper-Trading Enhancements

This is an **additive enhancement pack** for [`lucylow/Martha_Stewart`](https://github.com/lucylow/Martha_Stewart), whose public repository describes a Python-based algorithmic-trading application with app, migration, Nginx, and Docker components.

The enhancement pack focuses on engineering fundamentals needed for a more testable trading research system: typed market data, data-quality checks, independent signal strategies, portfolio accounting, pre-trade risk gates, a paper broker, SQLite persistence, hash-chained audit logs, multi-asset backtesting, walk-forward evaluation, operational monitoring, and an optional Flask operations API.

> **Integration note:** `martha_plus/` is now integrated at the root of this repository. It remains isolated from live order routing; review the integration and advanced extension guides before wiring any service into the Flask app.

## What is included

- `martha_plus/`: implementation package, with no required third-party runtime dependencies.
- `tests/`: regression tests for data validation, strategies, accounting, risk, execution, persistence, backtesting, logging, APIs, and operational safeguards.
- `examples/sample_bars.csv`: 180 seeded synthetic OHLCV bars; **not real market data**.
- `docs/`: architecture, safe integration steps, risk controls, backtesting caveats, configuration, and API documentation.
- Local checks and line-count utility are under `scripts/martha_plus/`; the legacy application and deployment files remain in place.

The integrated project contains **16,538 Python lines across implementation and tests** (331 page-equivalents at 50 lines per page). The v0.2.0 advanced layer contributes 5,165 implementation lines and the v0.3.0 enterprise layer adds 5,121 more. The goal is functional coverage, not repeated boilerplate. Run `python scripts/martha_plus/count_lines.py` after edits to regenerate exact figures.

## Quick start

Python 3.10+ is supported. The base package and test suite use Python's standard library; Flask is an optional extra for API integration.

```bash
# From this folder
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
# The martha_plus core uses the standard library; install requirements.txt only for the legacy Flask app.

# Validate configuration defaults
python -m martha_plus config-check --json

# Run a seeded synthetic backtest
python -m martha_plus demo --bars 500 --seed 7 --strategy momentum

# Backtest the bundled synthetic CSV
python -m martha_plus backtest --csv examples/sample_bars.csv --symbol DEMO \
  --strategy momentum --output reports/demo-backtest.json

# Run automated regression tests
sh scripts/martha_plus/check.sh
```

The CLI emits JSON including equity observations, order decisions, risk/performance statistics, and rejections. All examples are research/paper-only; a backtest result does not imply a strategy is profitable or ready to trade live.

## Minimal Python example

```python
from martha_plus.backtest import BacktestEngine
from martha_plus.config import Settings
from martha_plus.marketdata import CSVBarProvider
from martha_plus.strategies import MomentumStrategy

provider = CSVBarProvider("examples/sample_bars.csv")
settings = Settings(mode="research", starting_cash=100_000)
result = BacktestEngine(settings).run(
    provider.bars("DEMO"), MomentumStrategy(fast_window=8, slow_window=32)
)
print(result.metrics)
```

For several assets sharing one ledger, use `MultiAssetBacktestEngine`; see `docs/BACKTESTING.md` for timing conventions and limitations.

## Package map

| Area | Functionality |
|---|---|
| `types.py`, `config.py`, `errors.py`, `clock.py` | Validated domain objects, safe settings, explicit errors, deterministic time |
| `marketdata/` | CSV and synthetic providers, quote cache, quality reports, resampling, top-of-book features |
| `indicators.py`, `math_utils.py` | Rolling technical indicators and numerical/statistical helpers |
| `strategies/` | Momentum, mean reversion, pairs spread, ensemble, breakout, volatility targeting |
| `risk/` | Order limits, sizing, emergency halt, circuit breaker, exposure analysis, stress tests, tail metrics |
| `portfolio/` | Cash/position ledger, allocation weights, snapshot statistics, drawdown episodes |
| `execution/` | Paper fills, costs, token-bucket rate limit, order lifecycle, reconciliation |
| `persistence/` | SQLite order/fill/event/snapshot storage and tamper-evident hash-chain audit file |
| `backtest/` | Single- and multi-asset backtests, metrics, walk-forward folds, parameter/cost/bootstrap validation |
| `services/` | Research engine, health checks, batch isolation |
| `api/` | Optional Flask blueprint for health/status/orders and authenticated kill-switch control |
| `observability/` | Structured JSON logs with redaction and in-process counters/gauges/samples |
| `advanced/` | Causal features, portfolio diagnostics, validation, data lineage, workflows and operations controls |
| `enterprise/` | Authorization, request signatures, outbox, governance, cash/tax-lot tooling, risk limits, replay and safe exports |

## Safety model

- `Settings.mode` permits only `research` and `paper`; **live mode is explicitly rejected**.
- No real broker adapter ships in this pack. `PaperBroker` is a local simulator, not a market connection.
- Risk checks cover symbol freshness, order and position notional, gross exposure, available cash, open-order limits, daily loss, drawdown, fractional quantity policy, short-sale policy, and a kill switch.
- The HTTP API exposes no order-submission endpoint. `/status` and `/orders` require a bearer token, and the API refuses short/weak configured tokens.
- SQLite persistence uses transactions, uniqueness constraints, and a foreign key from fills to orders. The separate audit file hashes each event and links it to the prior event.
- Synthetic data, fixed spread/slippage, and simple fill simulation are for engineering tests—not realistic market microstructure or financial forecasts.

See `docs/RISK_AND_OPERATIONS.md` before connecting the service to any external application.

## Test status

The regression suite is run using:

```bash
sh scripts/martha_plus/check.sh
```

Flask API tests are skipped when Flask is not installed. Install the `api` extra to run them. Continuous integration installs the API and developer extras and runs all tests, plus bytecode compilation.

## Original-project integration

Start at `docs/martha_plus/INTEGRATION.md`, `docs/martha_plus/ADVANCED_EXTENSION.md`, and `docs/martha_plus/ENTERPRISE_EXTENSION.md`. The package is already integrated, but API/service wiring remains opt-in. The legacy app can run independently; its Python 3.6 image cannot import the Python 3.10+ research package.

## Disclaimer

This is software engineering and research tooling, not investment advice. Market data can be delayed, incomplete, inaccurate, or improperly adjusted; simulated execution differs from live execution; estimated costs can be wrong; and all strategies can lose money. Validate the entire stack with independent data, operational controls, legal/compliance review, and a controlled paper-trading period before considering any real-money system.
