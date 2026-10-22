# Architecture and Design Decisions

## 1. Intent

`martha_plus` is an additive domain package intended to sit beside the existing Martha_Stewart application. It separates the core trading-research mechanics from web presentation, deployment, and any real-broker integration.

The main data flow is:

```text
MarketDataProvider
      |
      v
Quote validation + historical bars
      |
      v
Strategy.generate() ---> Signal
      |                       |
      |                       v
      |                 Target portfolio weight
      |                       |
      v                       v
Portfolio snapshot <--- Position sizing
      |                       |
      +------> RiskLimits <---+
                    |
                approved?
                /       \
              no         yes
              |           |
           audit       OrderManager
                          |
                      PaperBroker
                          |
                 Fill -> PortfolioLedger
                          |
                      SQLiteStore
                          |
                   metrics / API
```

A strategy returns a `Signal`, not an order. This keeps signal generation separate from risk approval and execution. Strategy output should never bypass `RiskLimits` in a production integration.

## 2. Domain types

`types.py` defines immutable quotes, OHLCV bars, signals, order requests, fills, positions, snapshots, and risk decisions. Values are validated at object construction time: symbols are normalized, timestamps must be timezone-aware, prices and quantities must be finite and positive, and OHLC relationships must be coherent.

`Settings` is immutable and has conservative defaults. Unknown settings are rejected by `Settings.from_mapping`; environment parsing is explicit and does not load `.env` files behind the caller's back. `as_safe_dict()` masks the API token.

## 3. Clock and deterministic behavior

`SystemClock` is the normal clock. `FrozenClock` makes service tests reproducible. Synthetic prices use a fixed seed. Backtests use execution-bar timestamps for their simulated fills and do not submit the current bar to the signal before that bar's open fill.

A replay or integration loop should inject one clock source where possible. Avoid using wall-clock `datetime.now()` deep in strategy code; use event timestamps so ordering and tests remain deterministic.

## 4. Market data boundary

The `MarketDataProvider` protocol requires a latest quote and historical bars. `CSVBarProvider` treats a final close as a zero-spread quote proxy and clearly labels its venue `CSV_CLOSE_PROXY`; this is not an executable market quote. `SyntheticMarketData` is for demos only.

`QuoteCache` rejects out-of-order, stale, future-dated, or excessively wide quotes. `inspect_bars()` makes data-quality problems visible rather than trying to repair data silently. `resample_bars()` aggregates OHLCV, but partial buckets are retained; callers should track bucket completeness if their strategy requires full intervals.

`microstructure.py` derives spread, depth imbalance, microprice, and a one-level order-flow-imbalance statistic from quote snapshots. These features are descriptive; latency, sampling, order-book depth, and venue rules can materially change their interpretation.

## 5. Strategy interface

Strategies accept one symbol's strictly ordered bars and return `Signal(symbol, target_weight, strength, reason, timestamp, metadata)`. Weights are portfolio weights, not share quantities. Signal strength is constrained to the interval `[0, 1]` and is descriptive—not calibrated probability.

Included strategies:

- Momentum: fast/slow EMA spread.
- Mean reversion: rolling-window z-score, long-only by default.
- Pairs z-score: synchronized spread research signal; it intentionally does not execute both legs.
- Weighted ensemble: deterministic blend of strategy weights.
- Breakout: Donchian-style high/low breakout with ATR fraction filter.
- Volatility targeting: lagged momentum with EWMA volatility scaling.

Parameter search must not be performed on the same sample used for the final performance claim. Use walk-forward folds and out-of-sample evaluation.

## 6. Portfolio accounting

`PortfolioLedger` keeps cash, net quantity, average cost, last marks, fills, fill-ID de-duplication, and daily realized P&L. Reapplying the same `fill_id` is a no-op. Equity is cash plus marked market value. The default average-cost accounting is useful for engineering and portfolio-level P&L checks; it is not a tax-lot accounting system.

`RiskLimits` computes order notional using ask for buys, bid for sells, or limit price for limit orders. It returns an approval/rejection plus explicit reason codes. The single-asset and multi-asset backtest engines cap orders by configured notional/position limits and use a fixed synthetic spread.

## 7. Persistence and auditability

`SQLiteStore` creates a versioned base schema for orders, fills, events, and equity snapshots. Writes are transactional. Fills reference an existing order, and repeated fill identifiers are ignored.

`AuditLog` writes append-only JSON lines. Every entry has a sequence, UTC timestamp, event type, payload, previous hash, and SHA-256 hash of canonical content. Verification detects payload edits, broken links, sequence changes, and malformed lines. A hash chain does not defend against an attacker who can rewrite all data and the verifier; export or anchor the final hash in separately controlled storage if tamper-evidence is part of your control framework.

## 8. Execution layer

The only included broker adapter is `PaperBroker`. It applies a configurable, deterministic random fill probability, spread side, limit marketability checks, fixed slippage, and commission. The API doesn't provide an order-entry endpoint. If adding a real adapter later, require a separate security/risk review, a broker sandbox, idempotency recovery, server-side order status reconciliation, and explicit environment isolation.

`OrderManager` checks risk before calling the adapter, tracks client order IDs in-process, applies fills idempotently to the ledger, and can call persistence hooks. For multi-process deployments, replace its in-memory idempotency map with a durable shared order service and process-safe uniqueness handling.

## 9. API boundary

`martha_plus.api.create_blueprint()` is optional and requires Flask. Health is minimal and public; status and recent in-memory order details require a bearer token. The token is compared in constant time and must be at least 24 characters. Kill-switch POST/DELETE routes are authenticated. Deploy behind TLS, ensure secrets are not logged, and configure proxy headers deliberately.

## 10. Concurrency and deployment

The package does not start background threads or a market-data loop automatically. The caller owns scheduling, graceful shutdown, worker topology, and health checks. SQLite is intended as a modest single-node audit store; a multi-worker system should review its transaction and connection model and likely use a server database. `TokenBucket` is a local client-side throttle, not a substitute for exchange or broker rate limits.
