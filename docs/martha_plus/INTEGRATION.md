# Integration into the Existing Martha_Stewart Repository

## Integration boundary

The `martha_plus/` package is integrated at the repository root beside the existing Flask app. This is intentionally additive: the legacy web app, database configuration, migrations, requirements, Docker/Gunicorn setup, and Nginx configuration have not been replaced. The package does not connect to a live broker, start a trading loop, or enable live mode. The v0.3.0 enterprise helpers are also present under `martha_plus/enterprise/`, but they are not wired into the Flask app or any order path.

## Runtime compatibility

The research package requires Python 3.10 or newer and uses only the standard library for its core. Flask is needed only for the optional operations blueprint. The legacy Dockerfile currently specifies Python 3.6, so it cannot run the new package. Use a compatible Python 3.10+ environment from the repository root, or review the full legacy application's dependencies and compatibility before making a separate container upgrade. No deployment upgrade is implied by this integration.

## Quick verification

From the repository root:

```bash
python -c "from martha_plus.config import Settings; print(Settings().mode)"
python -m martha_plus config-check --json
python -m martha_plus demo --bars 500 --seed 7 --strategy momentum
python -m martha_plus backtest --csv examples/sample_bars.csv --symbol DEMO --strategy momentum
python -m pip install -r requirements-dev.txt
sh scripts/martha_plus/check.sh
```

The bundled CSV contains seeded synthetic OHLCV data, not real market data. The test suite is isolated under `tests/`; the legacy test folder named `testing/` is left untouched.

## Optional service wiring

Keep the package disabled in the web app unless there is a specific, reviewed need. To expose the optional API, install Flask in a Python 3.10+ environment, construct a configured research engine in the existing application factory, and register `martha_plus.api.create_blueprint(engine, api_token)`. Supply the token through a secret manager or environment injection; never commit it. See [`API.md`](API.md) and [`ENTERPRISE_EXTENSION.md`](ENTERPRISE_EXTENSION.md) and review the host app's authentication, TLS, reverse-proxy, network-access, and logging controls before exposure. The API intentionally has no order-submission endpoint.

## Operational review before broader use

- Validate market-data provenance, timestamp/time-zone handling, corporate actions, gaps, and symbol identity.
- Reconcile order and fill state with any authoritative source before sending new requests.
- Review all risk limits and persistence/recovery behavior against your deployment needs.
- Run paper-only through normal, volatile, missing-data, and restart scenarios.
- Keep kill-switch and audit-file procedures documented and tested.
- Obtain independent engineering, financial-risk, and legal/compliance review before considering any real-money system.

Backtest output and unit-test success are not evidence of profitability or production readiness. See [`BACKTESTING.md`](BACKTESTING.md) and [`RISK_AND_OPERATIONS.md`](RISK_AND_OPERATIONS.md) for the limitations and controls.
