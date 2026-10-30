# Systematic Research Reference

## Design boundary

This namespace is an orchestration and data-integrity layer. It intentionally reuses mature modules already in this checkout rather than duplicating their algorithms. The package stays dependency-free on the Python standard library plus in-repository APIs. It is not a broker adapter, signal service, execution gateway, or investment recommendation.

## Point-in-time joins

`point_in_time_join` groups feature rows by explicit key fields and performs a backward as-of lookup using the row's `available_ts_ns`, which should represent publication/availability time rather than the date the underlying economic event occurred. A candidate is eligible only when availability is no later than the decision timestamp. The last input row wins for duplicate same-key, same-time observations, allowing a deterministic correction convention. `max_age_ns` is an explicit freshness ceiling. By default a missing or stale match raises `DataIntegrityError`; `require_match=False` returns `observation=None` and `age_ns=None` so downstream code can withhold a result. No future fill or implicit stale carry is performed.

Times are non-negative integer nanoseconds. The caller must normalize source calendars, time zones, releases and corporate-action bases before joining. The join does not infer when a source became public; incorrect `available_ts_ns` still creates leakage.

`assert_available_by` is a small guard for a feature/decision pair. Use it at data boundaries or in tests when feature availability is assembled outside the join function.

## Historical constituents and delistings

`UniverseInterval` separates `valid_from_ns` / `valid_to_ns` from `known_from_ns`. Validity bounds are start-inclusive and end-exclusive. `members_at` includes an instrument only when it is valid and that membership fact was known by the decision time. This prevents a present-day constituent list from silently being projected into history. The caller remains responsible for complete membership histories, ticker changes, mergers, suspensions and identifier resolution.

`delisting_return(last_mark, recovery_value)` requires an explicit final recovery value in the same units and computes recovery / last mark - 1. Missing recovery is an error, not an assumed zero, carried mark, or guessed terminal return. This helper does not adjust for splits, distributions or currency changes.

## Signal backtest

`lagged_signal_backtest` is a single-asset timing/accounting primitive. Signal at index t is applied to the return at t+1; the first period is flat. Exposure is clipped to caller-specified symmetric bounds. Per-period cost is absolute position change times basis points / 10,000. It returns aligned positions, gross and net simple returns, turnover, costs, Sharpe and drawdown summaries. Inputs must already be aligned total returns. Returns below -100%, nonfinite observations, length mismatch, invalid periods and a net period loss at or below -100% are rejected. It does not model margin, borrow availability, financing, spreads, market impact, partial fills, capacity, or trading halts; use the microstructure execution models for those studies.

## Adaptive diagnostics

`conformal_interval` uses the finite-sample corrected absolute-residual order statistic for a split-conformal interval. When the requested rank exceeds the available calibration sample, it returns an unbounded interval instead of overstating coverage. The guarantee depends on exchangeability between calibration and future residuals, which market drift can violate. `PageHinkley` is an online, two-sided mean-shift alarm with caller-configured threshold, tolerance and forgetting. It should be calibrated to the sampling cadence; a flag is a prompt for review, not proof of a regime change. Recursive least squares and quantile regression already live in `quant_research` and are not duplicated here.

## Reused research modules

| Capability | Existing in-repository module families |
| --- | --- |
| Factor research, labels, signal diagnostics | `quant_research.signals`, `quant_research.portfolio.factor_model` |
| Cointegration, Kalman hedge, regimes, stationarity | `quant_research.timeseries` |
| Portfolio constraints, rebalance costs, risk budget | `quant_research.portfolio`, `quant_research.risk` |
| Futures, options, credit and rates | `quant_research.derivatives`, `technical_finance.options`, `technical_finance.fixed_income` |
| RLS, quantile regression and walk-forward | `quant_research.regression`, `quant_research.econometrics`, `quant_research.validation` |
| Split-conformal intervals and mean-shift alarm | `systematic.adaptive` |
| Intraday profiles, impact and execution schedule | `quant_research.timeseries.seasonality`, `microstructure.execution` |
| Event simulation, block bootstrap, stress and reporting | `microstructure.replay`, `quant_research.validation.stability`, `quant_research.risk.stress` |

## Verification

`sh scripts/systematic/check.sh` checks Python 3.10+, compiles the new package, runs deterministic unit tests, and executes the synthetic demo. Inputs and demo values are synthetic; passing tests verify software behavior, not model validity, data quality, capacity, alpha or profitability.
