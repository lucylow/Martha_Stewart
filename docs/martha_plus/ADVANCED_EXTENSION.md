# Advanced Extension — Integration and Review Guide

The v0.2.0 extension sits under `martha_plus.advanced` and intentionally leaves the earlier research/backtesting package intact. It is dependency-free and opt-in. Integrate one concern at a time rather than coupling all modules into the original web app in a single deployment.

## Suggested integration sequence

1. **Start with contracts and data quality.** Run `ohlcv_contract()` against normalized input rows. Use `FailoverMarketDataProvider` around providers you already trust; its source interface expects `get_bars(symbol, start, end)` and `get_quote(symbol)`. Set age limits appropriate to the frequency. Do not turn on stale fallback unless the consuming strategy handles stale data explicitly.
2. **Create point-in-time features.** Use `FeaturePipeline.transform()` on sorted single-symbol bars. Forward labels are deliberately separate through `FeaturePipeline.forward_returns()`. Do not merge future labels into input features. Persist only reviewed partitions with `LocalFeatureStore` and record a source-data fingerprint.
3. **Validate the research method.** Use `expanding_window_folds()` for chronological validation, or `purged_kfold_indices()` only when the experiment design justifies training on observations both before and after the test block. Use `alignment_diagnostics()` as one smoke check, not as proof that information leakage is absent. Parameter sweeps need holdout evaluation and correction for selection bias.
4. **Inspect portfolio construction.** `PortfolioOptimizer` is a dependency-free projected-gradient mean-variance routine. Validate covariance symmetry/quality, inspect convergence, then pass weights through your portfolio constraint checks. Risk-parity estimation may fail to converge for difficult covariance matrices; never ignore its status. Weight/turnover reports are diagnostics, not regulatory compliance.
5. **Estimate costs before creating paper schedules.** `liquidity` models are stylized estimates. `order_slicing` returns a schedule; it does not submit orders. Any eventual integration should pass every order delta through the existing risk gate and paper execution manager.
6. **Add operational controls.** Register readiness probes in `HealthRegistry`; add bounded alerts and incident lifecycle; persist experiment metadata and checkpoints. Local event/incident state is lost on process restart unless backed by a durable service.
7. **Export reports.** `reporting.render_html()` escapes user-controlled text and provides a self-contained summary. Don't embed secrets or customer data into reports. Keep any exports under access-controlled paths.

## Service lifecycle

`ManualScheduler` runs callbacks only when the host invokes `tick(now)`; it starts no background thread. Keep job functions short, idempotent where possible, and responsible for their own transaction boundaries. Retry policies are bounded and synchronous; configured exponential backoff blocks the calling tick. For true deadlines, use a hosting platform/process supervisor capable of terminating a stuck worker rather than relying on the scheduler's recorded runtime threshold.

`EventBus` is synchronous and process-local. Failed handler deliveries go to an in-memory dead-letter list; replay targets the handler that failed so already-successful subscribers don't repeat side effects. That dead-letter list is not durable. For multi-process or at-least-once delivery, use a separately operated durable message broker and maintain idempotent consumers.

## Production-readiness gaps that remain

- No live broker, exchange adapter, authentication session, or order submission endpoint is provided.
- No exchange-official holiday calendar or automatically refreshed corporate-action data source is bundled.
- No distributed lock, database migration system for the new extension, durable event queue, or multi-process scheduler is included.
- No guarantee is made about numerical stability for large optimizations, inferred strategy edge, model correctness, or real-market execution quality.
- The earlier public repository's source wasn't available in the build environment. This ZIP combines the previous enhancement pack with an additive extension; it is not a verified in-place patch to the original app. Follow the original project's app factory, migrations, logging, and container conventions when integrating.

## Change safety checklist

Before merging into the original repo, verify dependencies in its target runtime, review package naming conflicts, set explicit storage paths and retention, check timezone/market-calendar assumptions, test rollback and corrupted-state recovery, run paper simulation on representative historical and current data, verify access control on reports and internal API routes, and have risk/compliance owners review any transition from research to money movement.
