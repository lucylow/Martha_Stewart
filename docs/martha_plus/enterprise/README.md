# Enterprise module reference

All modules are in `martha_plus.enterprise` and use the Python standard library.

| Module | Purpose | Important boundary |
|---|---|---|
| `authz.py` | RBAC decisions, explicit-deny precedence, decision audit | Assumes caller already authenticated the principal; live trades denied by default |
| `api_guard.py` | Request metadata validation, token bucket, circuit guard, log sanitization | Rate limiting is process-local; reject results at your HTTP boundary |
| `signed_requests.py` | Canonical request HMAC and nonce replay defense | Keep secrets in a manager; use shared nonce storage across workers |
| `order_lifecycle.py` | Order state machine and transition journal | Persist journal for restart recovery |
| `idempotency.py` | Canonical request digests and duplicate response reuse | Process-local memory; use a shared unique-key store in deployments |
| `outbox.py` | SQLite outbox, leases, retry delay, dead-letter state | Consumer delivery can be at-least-once; consumers should deduplicate |
| `job_runner.py` | Dependency-aware jobs, retries, cooperative cancellation | Sequential runner; timeout is detected after a task returns, not a hard kill |
| `governance.py` | Immutable strategy manifest versions, approval workflow, shadow comparison | Paper approval is not investment validation or authorization for live execution |
| `streaming.py` | Event-time ordering/watermark, bounded lateness, OHLCV bars, topic router | No network transport, persistence, or distributed watermark coordination |
| `point_in_time.py` | Versioned values, backward as-of joins, future-leakage checks | Correctness still depends on trustworthy publication timestamps |
| `quality_rules.py` | Reusable required-field/range/unique/time/freshness checks | Add domain-specific rules and decide which severities block downstream use |
| `fx.py` | FX quote freshness and deterministic cross-rate conversion | Rates must be supplied by caller; currency calendars and settlement are out of scope |
| `taxlots.py` | FIFO/LIFO/HIFO/LOFO lot disposition, basis adjustment, split helpers | No short lots, jurisdiction-specific tax calculations, or wash-sale handling |
| `cash_management.py` | Cash ledger, reservations, pending flows and forecasts | Caller must reconcile balances and settlement sources |
| `cashflows.py` | XIRR/XNPV, time-weighted return, annualization, drawdown | External cash-flow classification must be correct |
| `benchmark.py` | Tracking error, information ratio, beta, capture, rolling diagnostics | Input series must be aligned and use consistent periodicity/currencies |
| `allocation.py` | Box-constrained simplex projection, inverse-vol and risk-budget helpers | Simplex projection does not enforce sector/cardinality constraints by itself |
| `scenario.py` | Deterministic symbol/sector shocks and historical scenario replay | Shock mappings are stylized, not calibrated forecasts |
| `risk_limits.py` | Configurable limit evaluation, block/halt severity, acknowledgements | Integrating service must enforce returned decisions |
| `liquidity.py` | Spread/ADV/participation checks and capped VWAP schedule | Indicative inputs; schedules are not broker orders or fill guarantees |
| `portfolio_replay.py` | Sequence-checked event store, holding/cash projection and digest | Caller supplies cash effects and price marks; review short-position accounting carefully before use |
| `ledger_reconciliation.py` | Internal-vs-external fills and cash control differences | Matching keys and tolerances must reflect the broker statement format |
| `surveillance.py` | Explainable cancellation/order-to-fill/notional triage rules | Flags are review leads, not proof of intent or a compliance determination |
| `signal_calibration.py` | Classification scores, calibration bins, threshold search | Suitable for probabilistic classifiers; thresholds should be selected out-of-sample |
| `reproducibility.py` | Dataset/code/output hashes and manifest generation/verification | Digests prove file identity, not the truth or provenance of source data |
| `exports.py` | Safe JSON/CSV/HTML exports, atomic replacement, secret redaction | Export authorization and encryption at rest remain application responsibilities |

## Cross-cutting implementation conventions

- Times entering risk and market-data APIs should be timezone-aware.
- Non-finite financial values should be rejected before downstream calculations.
- Decisions that block or halt must be enforced by the calling application; a record in this library does not stop an unrelated order path.
- Prefer immutable records for events/audit entries; collections returned from the public API should not be mutated to change internal state.
- Write integration tests against the actual legacy app before enabling any route or scheduler.
