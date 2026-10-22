# File Layout

```text
Martha_Stewart_Enhancement_Pack/
├── martha_plus/
│   ├── api/                 # Optional Flask operations blueprint
│   ├── backtest/            # Single/multi-asset engine, validation and metrics
│   ├── execution/           # Paper broker, costs, throttling, reconciliation
│   ├── marketdata/          # CSV/synthetic sources, cache, quality, microstructure
│   ├── observability/       # Structured logs and in-process metrics
│   ├── persistence/         # SQLite store and hash-chain audit log
│   ├── portfolio/           # Cash/position ledger, allocation and analytics
│   ├── risk/                # Limits, sizing, exposure, stress and guard monitors
│   ├── services/            # Research engine, health checks and batch processing
│   ├── strategies/          # Momentum, mean reversion, pairs, breakout and ensemble
│   ├── config.py            # Validated environment/mapping settings
│   ├── indicators.py        # Technical indicators
│   ├── math_utils.py        # Numerical helper functions
│   ├── types.py             # Validated domain objects
│   └── cli.py               # Local research CLI
├── tests/                   # Standard-library unittest regression suite
├── examples/sample_bars.csv # Seeded synthetic OHLCV history
├── docs/                    # Architecture, integration, risk, API and backtest docs
├── .github/workflows/ci.yml # Matrix CI for Python 3.10–3.13
├── scripts/check.sh         # Local compile/test/CLI smoke checks
├── scripts/count_lines.py  # LOC counter
├── pyproject.toml           # Package metadata and optional extras
└── Dockerfile               # Research CLI container example
```
