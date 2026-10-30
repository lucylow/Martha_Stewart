# Martha Plus — Market Microstructure Research Extension

**Status:** local simulation and research only. This extension contains no broker
adapter, exchange connection, credential handling, live-order endpoint, or live
trading switch. Synthetic examples use generated data and must not be mistaken
for market observations.

## Why this extension exists

The upstream project describes a research and paper-trading layer for validated
OHLCV research, portfolio controls, event replay, auditability, and execution
analysis. The `microstructure` namespace adds a deterministic limit-order-book
and event-time simulation layer below those higher-level research workflows.
Its purpose is to make assumptions that are often hidden in candle-based
backtests explicit: integer-tick price priority, FIFO order queues, partial
fills, protected market orders, arrival timestamps, data-feed delay, queue
position, and transaction costs.

The code is deliberately additive. It does not modify the legacy Flask app,
Docker configuration, Gunicorn entry point, database migrations, or existing
`martha_plus` research interfaces.

## Requirements and installation

- Python 3.10 or newer.
- Python standard library only for the extension itself and its tests.
- No broker credentials or network access are required.

Copy the `martha_plus/microstructure/` directory and `tests/microstructure/`
folder from this extension into the corresponding paths in the repository.
Copy `examples/microstructure/`, `scripts/microstructure/`, and this document
as desired. Run from the repository root:

```sh
python -m compileall -q martha_plus/microstructure
python -m unittest discover -s tests/microstructure -p 'test_*.py' -v
sh scripts/microstructure/check.sh
```

The legacy app's historical Python 3.6 environment is not supported by these
modules. Keep this package on the separately documented Python 3.10+ research
runtime until the app runtime is independently reviewed and upgraded.

## Quick start

Run a deterministic synthetic replay:

```sh
python examples/microstructure/run_demo.py --events 3000 --seed 7
```

Write an event archive and JSON summary:

```sh
python examples/microstructure/run_demo.py \
  --events 3000 --seed 7 \
  --archive artifacts/sim-events.jsonl \
  --report artifacts/replay-summary.json
```

The seed controls synthetic event generation. The strategy is a deliberately
simple displayed-imbalance taker and is intended to exercise the plumbing, not
to demonstrate a profitable signal. The output is a simulation diagnostic.

## Package map

| Namespace | Responsibility |
|---|---|
| `types.py` | Integer-tick instrument types, side/order enums, market events and lifecycle records. |
| `book/` | L2 depth snapshots and order-by-order FIFO L3 price levels. |
| `matching/` | Protected market orders, limit orders, IOC/FOK/post-only handling, maker/taker fees, cancellations and partial fills. |
| `clock.py` | Monotonic virtual time and seeded latency distributions. |
| `calibration/` | Basic calibration of latency and event-intensity assumptions. |
| `replay/` | Stable event scheduler, delayed feed observations, delayed order entry, risk checks and replay result. |
| `simulation/` | Seeded synthetic L3 feed, event transformations and paper portfolio ledger. |
| `strategies/` | Small protocol and example strategies for controlled experiments. |
| `metrics/` | Order-flow, volatility and execution-quality summaries. |
| `analytics/` | Bounded online statistics, volume bars, microprice, imbalance, toxicity proxy, Kyle-lambda estimate and post-fill markouts. |
| `data/` | Feed contracts, timestamp and price normalization, sequence monitoring, book snapshot validation and checksummed JSONL archive. |
| `execution/` | Queue position estimates, square-root/linear impact sensitivity, parent-order TWAP/VWAP/POV slicing and slippage accumulation. |
| `risk/` | Portfolio position book, exposure and drawdown analysis, limit evaluation and deterministic stress scenarios. |
| `observability/` | Hash-chained local journal, feed-health evaluation, metric alerts and a simulation-only circuit breaker. |
| `reporting/` | Run-summary creation and escaped JSON/CSV/HTML export. |

## Data and unit conventions

- Prices are positive integer tick counts in matching, book, and execution
  primitives. Convert currency prices through `PriceConverter` and an explicit
  `SymbolSpec.tick_size`.
- Quantities are positive integer lots. `SymbolSpec.lot_size` converts lots to
  instrument units for notional calculations.
- Internal timestamps are non-negative integer nanoseconds.
- Feed normalizers require explicit timestamp units when the input is not
  already in `ts_ns`; there is no guesswork based on magnitude.
- Decimal arithmetic is used for fees, notional, P&L, and configured model
  parameters. Statistical accumulators may use floats and are labelled as
  estimates rather than exchange accounting values.

## Matching and replay semantics

The limit order book maintains a price index and FIFO order queue at each price.
The matching engine accepts one instrument specification per instance. A
market order must include a protection bound (`max_price_ticks` for buys or
`min_price_ticks` for sells). This prevents unbounded sweeps when a book is
empty or malformed. IOC residuals are cancelled; FOK checks displayed contra
quantity before executing; post-only orders that would immediately remove
liquidity are rejected.

Replay applies canonical market events to the simulated venue at source event
time. The strategy sees a captured point-in-time book snapshot after a sampled
market-data delay. Its child order reaches the exchange after a separate
order-entry delay. Order acknowledgements have their own delivery delay.
Scheduler tie-breaking uses stable insertion sequence, not wall-clock time.

The simulator uses displayed liquidity only. It does not model hidden/iceberg
orders, venue-specific auction allocation, self-trade prevention rules,
regulatory throttles, order protection collars, exchange outages beyond a
simple halt/resume state, or a full multi-venue consolidated tape. Partial
fills are deterministic given the event sequence and latency seed.

## Data quality and archives

`MarketEventValidator` checks canonical event shape and instrument identity.
`FeedSequenceMonitor` detects provider sequence gaps, duplicates, and ordering
regressions. `MarketDataNormalizer` converts explicit prices and timestamp
units, sorts by `(timestamp, sequence)`, drops duplicate canonical identities,
and records an audit summary. In strict mode, any malformed row rejects the
batch instead of returning partially normalized data.

`EventArchiveWriter` writes a versioned JSONL archive through a temporary file
and atomically renames on successful completion. Each event includes a SHA-256
record checksum; `EventArchiveReader` verifies it while reading. The returned
whole-file digest can be retained separately for later comparison. This is a
local integrity feature, not a distributed database or signed provenance
system.

## Execution model assumptions

`QueuePosition` models one hypothetical order within an individual price
queue. Aggregate cancellations are ambiguous without order-level feed data,
so the default policy does not improve queue priority from an unclassified
cancel. `ProbabilisticQueueModel` exposes a caller-configured cancellation
split; its output is a model estimate, not a fill guarantee.

`SquareRootImpactModel` and `LinearImpactModel` are transparent sensitivity
models. They are not trained or calibrated by default. `ExecutionCostEstimator`
combines spread, commission, impact, and delay assumptions; users should
calibrate the parameters to the venue, instrument, and period they study.

`ScheduleBuilder` creates deterministic child plans for TWAP, VWAP, POV,
front-loaded, and back-loaded schedules. A POV schedule may leave a parent
residual unexecuted when forecast volume is insufficient. The parent tracker
keeps fills, cancellations, fees, and child identifiers separate from the
matching engine's order lifecycle.

## Risk, audit, and reporting boundaries

`PositionBook` records weighted-average inventory, realized/unrealized P&L,
fees, and marked exposure. `PortfolioRiskMonitor` checks caller-supplied
thresholds; `StressEngine` applies explicit first-order price shocks. Stress
shocks are scenario arithmetic, not a VaR model and not a prediction.

`HashChainJournal` gives local event evidence a deterministic hash chain. It
is tamper-evident only when the expected final hash is retained separately; it
is not digitally signed. `SimulationCircuitBreaker` controls a local research
loop and must not be described as a live broker kill switch.

`RunSummaryBuilder` summarizes replay output. `ReportExporter.to_html` escapes
all dynamic values, but exported reports should still be treated as research
artifacts and reviewed before publication.

## Test coverage

The bundled `unittest` suite covers:

- FIFO matching and protected order semantics;
- replay timing and strategy-order fills from later external flow;
- online statistics, bar aggregation and microstructure indicators;
- event normalization, price/time conversion, sequence gaps and archive checksums;
- queue position accounting, impact monotonicity, and order-slice conservation;
- long/short position accounting, drawdown, concentration and stress arithmetic;
- hash-chain verification, health states, alerts, and safe report exports.

Run the suite after changing any primitive. The engine and risk code are small
reference models and should be extended with venue-specific conformance tests
before being used for any decision beyond offline research.
