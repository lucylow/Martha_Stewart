# Martha_Stewart — Quantitative Research Extension

A dependency-free Python research package that adds advanced quantitative methods under
`martha_plus.quant_research`. It is designed to complement the separate microstructure
and technical-finance extensions without changing the legacy Flask application.

## Scope and safety

This is a **research, analysis and simulation toolkit**. It does not connect to a broker,
submit live orders, provide investment advice, or establish that a strategy is profitable.
All examples use deterministic synthetic data. Distributional tests, shrinkage heuristics,
small-sample approximations, numerical optimizers and regime models should be validated
against independent implementations before consequential use.

## Requirements

- Python 3.10 or newer
- Python standard library only

## Installation / overlay

Extract this archive at the root of the Martha_Stewart checkout. The archive paths are
repository-relative and add these paths without replacing the application's existing files:

- `martha_plus/quant_research/`
- `tests/quant_research/`
- `examples/quant_research/`
- `docs/martha_plus/quant_research/`
- `scripts/quant_research/`

Run the checks from the repository root:

```bash
sh scripts/quant_research/check.sh
```

Run the synthetic workflow directly:

```bash
python examples/quant_research/run_demo.py
```

## Feature map

| Domain | Capabilities |
|---|---|
| Time series | stationarity diagnostics, fractional differences, cointegration, Kalman hedge ratios, AR/VAR, spectra, long memory, change points, seasonality and Gaussian HMMs |
| Signal research | information coefficients, factor quantiles, decay, event studies, CUSUM samples, triple-barrier labels, uniqueness weights and feature screening |
| Validation | purged/embargoed folds, walk-forward and combinatorial CV, PSR/DSR, PBO, Reality Check bootstrap, multiple-testing correction and time-series permutation importance |
| Volatility | realized variance, bipower variation, jump variation, realized kernels, asynchronous covariance, GARCH/GJR-GARCH, HAR-RV, DCC multivariate volatility and forecast backtests |
| Risk | covariance estimation, shrinkage, eigenvalue cleaning, PCA, copula scenarios, stress testing, liquidity risk, regime-conditioned risk, factor VaR/ES attribution and drawdowns |
| Portfolio | Black–Litterman, constrained mean-variance, equal-risk contribution, factor models, robust means, rebalance costs and turnover controls |
| Fixed income | Nelson–Siegel, discount-curve bootstrapping, bond cash flows, duration/convexity, key-rate risk and credit hazard/CDS measures |
| Derivatives | futures fair value/carry, continuous-contract rolls, swaps, bond-futures basis/CTD, GBM/Heston Monte Carlo, Asian/barrier options and variance swaps |
| Econometrics / Bayesian | panel fixed effects, Fama–MacBeth, VAR/Granger, robust and quantile regression, conjugate updates and Bayesian model averaging |
| Performance | Sharpe/Sortino/Calmar, Omega, ulcer index, capture ratios, active-return decomposition and benchmark summaries |

See [`docs/martha_plus/quant_research/REFERENCE.md`](docs/martha_plus/quant_research/REFERENCE.md)
for the module map, API patterns and model limitations.

## Design principles

1. Inputs are explicit sequences/matrices; callers own data cleaning, timestamps and corporate-action adjustments.
2. Time-series functions preserve causal ordering where applicable and expose event-end indices for leakage-aware validation.
3. Statistical approximations identify themselves in result fields or docstrings; automatic covariance shrinkage is a heuristic, not a claim of exact Ledoit–Wolf calibration.
4. The package avoids silent external data downloads and optional numerical dependencies.
5. Outputs are research diagnostics—not regulatory, accounting, or execution guarantees.

## Build size and verification

The delivered source package contains 78 Python modules and 5,091 nonblank Python
source lines (5,961 total source lines). At a conservative 50 nonblank code lines per
page, that is approximately 101.8 pages of implementation. The included regression
suite contains 66 tests. The package check compiles the modules, runs the tests, and
executes the synthetic demonstration.
