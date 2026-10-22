# Risk and Operational Controls

## Pre-trade risk checks

`RiskLimits.evaluate(request, context)` returns a `RiskDecision`. On approval, `reason` is `approved` and `allowed_quantity` repeats the request's quantity. On rejection, `reason` contains one or more machine-readable reason codes separated by commas, including:

- `symbol_mismatch`
- `stale_quote` or `future_quote`
- `trading_halted`
- `open_order_limit`
- `daily_loss_limit`
- `drawdown_limit`
- `short_selling_disabled`
- `max_order_notional`
- `fractional_quantity_disabled`
- `max_position_notional`
- `gross_exposure_limit`
- `nonpositive_equity`
- `insufficient_cash`

The current implementation is a conservative base layer, not a substitute for broker-side controls. In particular, commissions, fees, borrow costs, margin rules, exchange price bands, trading halts, corporate actions, partial fills, and instrument-specific limits can require extra checks.

## Configuration

All environment variables are prefixed with `MARTHA_`. Important settings:

| Variable | Default | Purpose |
|---|---:|---|
| `MARTHA_MODE` | `research` | Only `research` and `paper` are supported |
| `MARTHA_STARTING_CASH` | `100000` | Paper ledger starting cash |
| `MARTHA_MAX_ORDER_NOTIONAL` | `5000` | Maximum order dollar value |
| `MARTHA_MAX_POSITION_NOTIONAL` | `15000` | Per-symbol position value limit |
| `MARTHA_MAX_GROSS_EXPOSURE` | `0.75` | Gross exposure relative to equity |
| `MARTHA_MAX_DAILY_LOSS` | `2000` | Daily loss halt threshold in account currency |
| `MARTHA_MAX_DRAWDOWN_FRACTION` | `0.12` | High-water-mark drawdown limit |
| `MARTHA_MAX_OPEN_ORDERS` | `20` | Open-order count limit |
| `MARTHA_MAX_QUOTE_AGE_SECONDS` | `5` | Quote freshness bound |
| `MARTHA_COMMISSION_BPS` | `1` | Simulated commission rate |
| `MARTHA_SLIPPAGE_BPS` | `2` | Simulated slippage rate |
| `MARTHA_PAPER_FILL_PROBABILITY` | `1` | Probability the simulation fills an order |
| `MARTHA_ALLOW_SHORT` | `false` | Whether risk checks can permit short positions |
| `MARTHA_ALLOW_FRACTIONAL_SHARES` | `false` | Whether fractional order quantities are permitted |
| `MARTHA_KILL_SWITCH_FILE` | `data/KILL_SWITCH` | Halt file path |
| `MARTHA_DATABASE_PATH` | `data/martha_plus.sqlite3` | SQLite file path |
| `MARTHA_API_TOKEN` | empty | Optional API bearer token; never commit this |

Check the active parse with `python -m martha_plus config-check --json`. Secrets are masked in output.

## Emergency halt

The presence of the configured kill-switch file means the strategy service rejects new order requests. `KillSwitch.activate(reason)` writes a timestamp and reason through an atomic rename. `deactivate()` removes the file. The monitor doesn't create threads or poll in the background; the caller has to provide fresh equity values to `PortfolioRiskMonitor.update()`.

A kill switch is not magic: if you connect a real broker, document whether an emergency stop cancels open orders, flattens positions, blocks only risk-increasing orders, or also blocks risk-reducing orders. This pack does not connect to a real broker and does not cancel any external order.

## Persistence and reconciliation

Use `SQLiteStore` for local records, and run `reconcile(orders, fills, positions)` to find obvious orphans, overfills, filled-quantity mismatches, and invalid positions. `AuditLog.verify()` checks whether the JSONL hash chain is intact. Back up both the database and audit output. Filesystem permissions, access control, retention, key management, monitoring, and external hash anchoring remain deployment responsibilities.

The included order manager's idempotency map is in memory. In a multi-worker system, it is not enough by itself to guarantee idempotency across process restarts. The persistent database constraint helps catch conflicts, but a robust distributed system needs a durable order-state machine and defined recovery/replay semantics.

## Observability

`configure_logging()` emits JSON logs and redacts common token/password fields plus Bearer credentials. Redaction is defense in depth, not a guarantee; do not log secrets or full authorization headers in the first place. `MetricsRegistry` is process-local and resets on restart; export counters to your established metrics backend if long-lived monitoring is needed.

## API hardening

- Use TLS and the existing authentication/session framework where possible.
- Generate a high-entropy API token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- Keep the token in a secret manager or environment injection path; never log it.
- Restrict the operations endpoints at the network layer and monitor changes to kill-switch state.
- Add rate limiting at the API gateway. `TokenBucket` is a local helper for internal call throttling, not public endpoint protection.
- Never rely on CORS or a hard-to-guess URL as authentication.

## Financial limitations

The included risk and simulation tools are illustrative engineering components. Historical VaR, expected shortfall, standard deviation, z-scores, stress shocks, and backtest ratios are sample-dependent. The square-root impact estimator is a rough sensitivity curve. None predicts next-day loss, fills, or profits. Validate formulas and conventions with a qualified risk reviewer before any use in decisions.
