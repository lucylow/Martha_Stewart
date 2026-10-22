# Optional Flask Operations API

The `martha_plus` core has no third-party dependencies. Flask is included in the legacy app requirements. Import `create_blueprint` from `martha_plus.api` and register it with the host Flask app; the returned blueprint is prefixed by `/martha-plus` by default. The token must be nonempty and at least 24 characters. Clients send `Authorization: Bearer <token>`.

| Method | Route | Auth | Behavior |
|---|---|---|---|
| `GET` | `/martha-plus/health` | No | Minimal health response; no account or path details |
| `GET` | `/martha-plus/status` | Yes | Paper/research mode, equity, cash, exposure, positions, open-order count |
| `GET` | `/martha-plus/orders` | Yes | Last 100 in-memory orders |
| `POST` | `/martha-plus/kill-switch` | Yes | Activate a local halt; accepts optional JSON `reason` |
| `DELETE` | `/martha-plus/kill-switch` | Yes | Deactivate halt file; ordinary risk checks remain in force |

There is intentionally no order-submission endpoint. This API is an operational view and halt control; it is not a portfolio management UI or broker gateway. `/health` only means that the blueprint responded, not that the market data, database, or portfolio source is healthy. For comprehensive readiness, wire `HealthMonitor` probes into the host application's own health checks.

Deploy behind TLS, set strong tokens using a secrets manager, constrain network access, avoid proxy header trust without explicit configuration, audit kill-switch changes, and apply request-rate limits at the gateway.
