# Backtesting and Research Validation

## Single-asset engine

`BacktestEngine.run(bars, strategy, minimum_history=35)` iterates through an ordered set of OHLCV bars. The strategy sees only bars strictly before the current execution bar. Target-weight changes are sized on the current bar's open and filled through a synthetic quote derived from the open and the configured spread. Holdings are marked at that bar's close.

The engine charges the configured commission and slippage bps and applies order and position limits. Signals are generated once per current event. The output includes an equity curve, order log, risk rejection count, performance metrics, and basic realized-trade summary.

## Multi-asset engine

`MultiAssetBacktestEngine.run(bars_by_symbol, strategies)` uses a shared cash and position ledger. Timestamps are unioned across symbol histories. At a timestamp, signals use that symbol's strictly prior history; fills use its current bar's open. Symbols at the same timestamp are processed lexicographically to provide deterministic results. This is an explicit ordering convention, not an exchange matching model. Different timestamp grids and market hours deserve additional calendar/session rules in a deployment-specific simulator.

## Metric definitions

- Total return: ending equity / starting equity minus one.
- Annualized return: geometric rescaling across the number of observations, using `periods_per_year` (default 252).
- Annualized volatility: sample standard deviation of period returns times square root of periods per year.
- Sharpe: mean excess period return divided by period-return standard deviation, annualized using square root of periods per year.
- Maximum drawdown: largest peak-to-subsequent-trough fractional equity decline.
- Calmar: annualized return / maximum drawdown, or zero when there is no drawdown.
- Win rate/profit factor: computed over the supplied P&L samples; these samples reflect the included simulator/ledger behavior, not independent broker statements.

Period settings must match the bar frequency. The default 252 annualization factor assumes daily trading periods; intraday bars require an appropriate factor, and missing sessions can bias interpretation.

## Walk-forward evaluation

`walk_forward_folds()` builds chronological train/test slices. `evaluate_walk_forward()` chooses a parameter setting on each training slice and evaluates it on a later test slice. User-supplied scoring callbacks must confine fitting, standardization, feature selection and model selection to the training slice; the helper cannot detect data leakage within an arbitrary callback.

## Parameter sweeps and sensitivity

`parameter_sweep()` reports a sorted set of in-sample results for a fixed grid; it is exploratory only. `transaction_cost_sensitivity()` evaluates fixed commission/slippage combinations to show whether gross performance evaporates when trading friction increases. `block_bootstrap_returns()` estimates a resampled compound-return distribution using circular blocks to retain limited local dependence. That distribution does not generate new regimes and is not a forecast.

Before using any output as evidence, include:

1. Time-ordered train, validation, and final untouched out-of-sample windows.
2. Data-source and adjustment policy, including delistings and survivorship bias handling.
3. Corporate actions, dividends, splits, symbol changes, market sessions, and missing-bar handling.
4. Realistic bid/ask spread, commission, exchange/regulatory fees, slippage, latency, queue priority, partial fills, rejects, and borrow costs where relevant.
5. Sensitivity across regimes and parameter neighborhoods, not just the highest metric in one sweep.
6. Comparison with simple baselines and a no-trade cash baseline.
7. Reproducibility metadata: source file hash, seed, configuration, package version, dates, strategy parameters, and output hash.

## Limitations not modeled

The engines are not HFT venue simulators. They do not model nanosecond/microsecond latency, order-book queues, hidden liquidity, routing decisions, market impact calibrated by instrument, halts, intraday auction mechanics, margin calls, corporate actions, asynchronous feeds, or exchange-specific order rules. The included pairs strategy emits a leg signal only; it does not guarantee a dollar-, beta-, or factor-neutral hedge.

Never infer expected future profitability from a successful unit test or synthetic-price backtest.
