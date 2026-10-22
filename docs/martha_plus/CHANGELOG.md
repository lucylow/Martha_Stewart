# Changelog

## 0.3.0 — Enterprise Controls and Portfolio Operations
- Added 5,121 lines under `martha_plus/enterprise/` and enterprise regression coverage.
- Added authorization, signed requests, order lifecycle, idempotency, transactional outbox, event-time streaming, point-in-time joins, FX, tax lots, cash management, governance, risk limits, scenario and benchmark analytics, portfolio replay, reconciliation, surveillance, and safe exports.
- Added development pytest requirements and a unified test runner so pytest function tests and unittest suites are both exercised.
- Kept all new controls opt-in and paper/research-only; no live order path was added.


## 0.2.1 — Compatibility and startup fixes
- Corrected legacy test-helper imports that collided with the newly added `tests/` package.
- Replaced removed Flask and SQLAlchemy APIs and modernized WTForms field names.
- Added missing login, registration, base-layout, and error templates; added a Flask-enabled app smoke test.
- Added the referenced Python 3.6 SQLite dependency lock file and removed secret-file copying from the Docker image.
- Read the Flask secret key from the environment and added a non-secret `.env.example`.

## 0.2.0 — Advanced research and operations extension
- Added causal features, data contracts, feature diagnostics and a local feature store.
- Added portfolio optimization/constraints, risk budgets, attribution, rebalancing and portfolio diagnostics.
- Added regimes, point-in-time universe filters, corporate-action helpers, session calendars, data-source failover and quality monitoring.
- Added purged/expanding walk-forward validation, multiple-testing and sensitivity diagnostics.
- Added liquidity/capacity estimation, order-slicing schedules and execution-quality analysis (no order submission).
- Added local event/replay/scheduler utilities, alerts, incident management, readiness probes, experiments, lineage manifests, snapshots, config drift checks, and report exports.
- Added 7 advanced test modules and documentation; updated package version.

## 0.1.0 — Research and paper-trading enhancement
- Added validated domain objects and safe settings, market-data providers and quality checks, strategies, risk controls, accounting, paper execution, persistence/audit, backtesting, services, optional API, and regression tests.
