# Quantitative Research API Reference

## Import layout

All functionality is grouped by research domain beneath `martha_plus.quant_research`.
Public functions are exported from each domain's `__init__.py` where practical.

```python
from martha_plus.quant_research import TimeSeries
from martha_plus.quant_research import risk, portfolio, timeseries, signals
from martha_plus.quant_research import validation, volatility, fixed_income
```

The core package uses only Python's standard library. Matrices are nested sequences in
row-major form. Unless a function says otherwise, return panels use **time × asset**
orientation; factor model estimators explicitly accept **asset × time** panels for assets
and factors. Carefully read each function's docstring before combining orientations.

## 1. Time-series modelling

`timeseries` includes simple/log returns, ordinary and fractional differencing, rolling
and exponentially weighted transforms, ADF/KPSS-style stationarity diagnostics,
variance-ratio tests, Engle–Granger cointegration, hedge-ratio estimation and half-life
summaries. Kalman-filter utilities support dynamic linear regression. AR forecasting,
VAR forecasting, Granger tests, autocorrelation diagnostics, periodograms, partial
autocorrelation, spectral entropy, Hurst/DFA summaries, bootstrap methods, seasonality,
CUSUM/Page–Hinkley and binary segmentation are also provided.

The univariate Gaussian HMM estimates a small finite-state model with EM. It can converge
to local optima; compare multiple initializations and state counts. Filtered probabilities
are appropriate for online-style state estimates; smoothed probabilities use future data
and must not be used as real-time features.

## 2. Signal construction and financial labels

`signals` provides information-coefficient summaries, cross-sectional quantile portfolios,
alpha decay and breadth, feature screening, abnormal-return event studies, CUSUM event
sampling, fixed-horizon/triple-barrier labels and average-uniqueness weights. Event intervals
use inclusive end indices. Triple-barrier labels are research labels, not simulated fills;
when both horizontal barriers occur in one bar, the default tie rule chooses the stop.

## 3. Validation and research integrity

`validation` contains purged K-fold, walk-forward, combinatorial purged CV, embargo and
interval-overlap checks. Backtest-statistics utilities include probabilistic/deflated Sharpe,
probability of backtest overfitting, White-style Reality Check bootstrap and
Benjamini–Hochberg correction. Bootstrap stability and block-permutation importance are
available for time-dependent data.

Validation inputs should be aligned to the actual label interval. Do not use smoothed HMM
probabilities, centered filters, full-sample normalization, or labels whose end dates overlap
the test interval in a training fold. PBO/DSR outputs depend on assumptions about the trial
universe and sampling process; report those assumptions with results.

## 4. Volatility and risk

`volatility` supplies realized variance, bipower variation, quarticity, jump variation,
realized kernels and an asynchronous covariance estimator, together with GARCH(1,1),
GJR-GARCH, HAR-RV and two-stage scalar DCC multivariate volatility. DCC can use GARCH,
EWMA or sample marginal standardization and exposes rolling one-step covariance forecasts.
Forecast evaluation includes QLIKE, quantile loss, Kupiec coverage and Christoffersen
independence/conditional-coverage diagnostics.

`risk` supplies sample/EWMA/robust/shrinkage covariance estimators; PCA and eigenvalue
cleaning; Gaussian-copula scenarios; Euler and factor risk contributions; historical VaR/ES
allocation; liquidity-adjusted risk; regime-conditioned covariance; drawdown measures;
and stress/reverse-stress helpers. The automatic shrinkage intensity is deliberately a
simple heuristic. It is not the analytical Ledoit–Wolf estimator. Gaussian copulas assume
a Gaussian dependence structure even when empirical marginal ranks are used.

## 5. Portfolio construction

`portfolio` includes Black–Litterman priors/views, long-only constrained mean-variance
optimization, turnover-aware rebalance, equal/custom risk budgets, linear factor risk
models, robust return estimators and uncertainty-penalized expected returns. Numerical
solvers are transparent iterative methods; inspect convergence flags and constraint
residuals. A valid optimizer output is not a guarantee that expected returns or covariance
estimates are accurate.

## 6. Fixed income and derivatives

`fixed_income` includes Nelson–Siegel fitting, discount-factor interpolation and bootstrap,
bond cash-flow valuation, Macaulay/modified duration, convexity, key-rate duration, curve
shape factors, survival/hazard transformations, default probabilities, risky annuity, CDS
par spread and CS01 estimates. Curve bootstrap conventions are simplified and should be
reconciled with the conventions of the relevant market (day count, calendars, stubs,
compounding, collateral and payment lags).

`derivatives` includes cost-of-carry futures values, implied carry/roll, continuous-contract
adjustments, par swaps and swap DV01, bond-futures invoice/net basis and CTD selection.
Monte Carlo tools price GBM European/Asian/discretely monitored barrier claims and Heston
European claims with full-truncation Euler; the latter has time-step bias and the barrier
engine monitors only simulation dates. Variance-swap utilities estimate discrete option-strip
variance, forward variance, realized variance and variance risk premia. These are analytical
templates: production valuation requires instrument-specific terms, calendars, delivery
options, funding curves, settlement conventions, robust calibration and independent checks.

## 7. Econometrics, Bayesian estimation and performance

`econometrics` adds panel fixed effects, Fama–MacBeth, VAR/Granger causality and residual
diagnostics, as well as weighted least squares, Huber and quantile regression. `bayesian`
contains conjugate return/linear-regression updates and model averaging. `performance`
contains path-aware risk-adjusted ratios and benchmark/capture decomposition.

Approximate p-values are labelled as approximations where relevant. Classical test
assumptions (stationarity, independence, homoskedasticity, model specification and sample
size) may not hold for financial data. Robust alternatives help diagnose sensitivity but
do not remove model risk.

## Synthetic demo

Run `python examples/quant_research/run_demo.py`. The demo generates synthetic factor and
asset returns using a fixed seed, estimates covariance/factor exposures, constructs a
risk-budget portfolio, fits a regime model, and reports validation/performance diagnostics.
It proves that the APIs can be composed on toy data; it does **not** prove alpha, capacity,
market realism or future performance.
