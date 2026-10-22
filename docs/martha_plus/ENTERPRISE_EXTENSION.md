# Enterprise Extension (v0.3.0)

This extension adds dependency-free controls and stateful helpers under `martha_plus/enterprise/`, integrated additively in this repository. These helpers are not automatically wired into the Flask application or any live execution path; integrate one component at a time through a reviewed service layer.

## Quick examples

### Role-based authorization

```python
from martha_plus.enterprise.authz import Action, AccessController, Principal, standard_paper_policies

controller = AccessController(standard_paper_policies())
operator = Principal("operator-1", frozenset({"operator"}))
decision = controller.evaluate(operator, Action.TRADE_PAPER, "orders/paper/ABC")
if not decision.allowed:
    raise PermissionError(decision.reason)
```

Authorization is only one layer. Authenticate the principal at the API boundary, protect policy administration, and put shared audit data in a durable store for multi-process applications. `TRADE_LIVE` is rejected by default regardless of user policy.

### HMAC signed request verification

```python
from martha_plus.enterprise.signed_requests import HMACRequestVerifier, sign_request

secret = secret_manager.get_bytes("paper-api-hmac")  # never hard-code secrets
signed = sign_request(secret, "POST", "/paper/orders", now_epoch, request_nonce, {"symbol": "ABC"})
verifier = HMACRequestVerifier({"paper-key": secret})
result = verifier.verify("paper-key", signed, {"symbol": "ABC"})
assert result.accepted
```

Use a shared durable nonce store and a key rotation process in a distributed service. The included nonce store is process-local.

### Order lifecycle

```python
from martha_plus.enterprise.order_lifecycle import OrderEvent, OrderStateMachine

orders = OrderStateMachine()
orders.create("paper-001", requested_quantity=10)
orders.transition("paper-001", OrderEvent.VALIDATE, actor="risk-gate")
orders.transition("paper-001", OrderEvent.ACCEPT, actor="paper-broker")
orders.transition("paper-001", OrderEvent.PARTIAL_FILL, actor="simulator", fill_quantity=4)
```

Transitions reject illegal states, stale version numbers, negative quantities and oversize fills. Keep the event history in durable storage when recovery after process restart is required.

### SQLite outbox

```python
import sqlite3
from time import monotonic
from martha_plus.enterprise.outbox import SqliteOutbox

outbox = SqliteOutbox(sqlite3.connect("outbox.db"), clock=monotonic)
event_id = outbox.enqueue("audit.paper_order", {"order_id": "paper-001"})
for event in outbox.lease("worker-1", limit=25):
    try:
        dispatch_to_internal_consumer(event.topic, event.payload)
        outbox.acknowledge(event.event_id, "worker-1")
    except Exception as exc:
        outbox.fail(event.event_id, "worker-1", str(exc), retry_delay=5)
```

The outbox guarantees local leasing semantics only. External delivery is typically at-least-once, so consumers should be idempotent. The caller must coordinate domain changes and outbox inserts in the same database transaction where atomicity is required.

### Point-in-time data joins

```python
from datetime import timedelta
from martha_plus.enterprise.point_in_time import PointInTimeIndex, VersionedValue

index = PointInTimeIndex()
index.add(VersionedValue("earnings:ABC", 1.23, valid_from=known_date,
                         published_at=release_timestamp, source="filing-feed"))
result = index.lookup("earnings:ABC", decision_timestamp, max_age=timedelta(days=120))
if not result.matched:
    # Fail closed or skip the feature rather than use future data.
    skip_trade_decision()
```

### Portfolio replay and risk limits

`PortfolioEventStore` enforces per-account sequence numbers and stable event identifiers. `PortfolioProjector` rebuilds current holdings and cash from stored events and externally supplied marks. `RiskLimitEngine` evaluates explicitly named metrics against configured thresholds; the integrating service is responsible for preventing new orders when the result is blocked or halted.

### Safe exports

`write_csv` prefixes values that could be interpreted as spreadsheet formulas, `html_table` escapes cell/title values, `redact_mapping` strips keys that resemble secrets, and `write_json` replaces output files atomically. Validate report access control before exporting account or position data.

## Module map

See `enterprise/README.md` for the complete module-by-module reference, assumptions, and non-goals.

## Production limitations

- **Paper/research only.** No external broker integration or live order submission is included.
- **No turnkey identity provider.** The RBAC module assumes a trusted principal was authenticated elsewhere.
- **No distributed guarantees from in-memory helpers.** Use database-backed idempotency, nonce, rate-limit, and job state for multiple processes.
- **Not tax or legal advice.** Tax-lot and surveillance tools provide accounting primitives and heuristic triage only; they do not decide legal compliance.
- **Data integrity depends on the source.** Point-in-time joins guard temporal availability but cannot prove vendor revisions are complete or correct.
- **Not a financial forecast.** Simulation and risk estimates do not guarantee profitability or future performance.

## Validation

From the repository root, install the test runner with `python -m pip install -r requirements-dev.txt`, then run `sh scripts/martha_plus/check.sh`. Flask API checks use the dependencies already listed in `requirements.txt`.
