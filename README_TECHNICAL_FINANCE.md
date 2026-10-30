# Martha_Stewart — Technical Finance Extension

**Version:** 0.1.0  
**Runtime:** Python 3.10+  
**Dependencies:** Python standard library only  
**Purpose:** quantitative-finance research, valuation, risk measurement and paper-portfolio analytics.

This ZIP is a merge-ready overlay. Extract it at the root of the existing Martha_Stewart repository. It adds `martha_plus/technical_finance/`, `tests/technical_finance/`, `examples/technical_finance/`, documentation and a local test script. It does not overwrite the earlier `martha_plus/microstructure/` extension.

## Scope

The package adds approximately **5,061 nonblank Python source lines** (5,983 physical lines including blank lines) across **34 Python modules** under `martha_plus/technical_finance/`. At a convention of 50 nonblank source lines per page, that is approximately **101 pages of Python source** (actual printed pagination depends on font, margins and formatting). The archive also contains tests and usage material, which are additional to that source count.

### Quantitative research and statistical tools

- Simple and log returns, compounding, wealth paths, drawdown episodes and rolling beta.
- Sharpe, Sortino, Calmar, Omega, information ratio, tracking error, Sterling, Martin, payoff and gain-to-pain measures.
- Distribution diagnostics, robust statistics, serial correlation, Ljung-Box approximation and rolling quantiles.
- OLS with classic standard errors, Newey-West HAC covariance, rolling regression and VIF.
- Cross-sectional factor z-scores, composite ranking, IC/rank-IC, factor attribution, group neutralization, quantile spreads and decay summaries.
- OU process calibration, AR(1) mean-reversion diagnostics, Engle-Granger residual diagnostics, ADF regression, rolling pair hedge ratios and Kalman hedge estimation.

### Portfolio and market risk

- Sample covariance/correlation and EWMA covariance.
- Diagonal and constant-correlation covariance shrinkage.
- Long-only minimum-variance, risk-parity, hierarchical risk-parity and mean-variance portfolio weights.
- Tangency portfolio, approximate efficient frontier, Black-Litterman posterior estimates and turnover-penalized rebalancing.
- Historical, Gaussian, Cornish-Fisher, Monte Carlo and peaks-over-threshold/GPD VaR/expected shortfall.
- Component VaR, drawdown risk, delta-normal factor risk, stress covariance, reverse stress and liquidity-adjusted VaR screens.
- Realized volatility, bipower/jump variation, Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang estimators; EWMA and fixed-parameter GARCH(1,1); empirical tail-dependence and drawdown-at-risk diagnostics.

### Derivatives and fixed income

- Black-Scholes-Merton European option prices, analytic Greeks, implied volatility and put-call parity.
- CRR binomial pricing for European and American options, IV surface interpolation/consistency screens and option-book Greek aggregation.
- Equity, commodity and FX forwards; futures basis and implied repo; FRA and single-curve interest-rate swaps.
- Bond cash flows, clean/dirty price utilities, yield-to-maturity, duration, convexity, DV01 and z-spread.
- Log-linear discount curves, deposit/swap bootstrap utilities, par swap rates, Nelson-Siegel/Svensson evaluation and Nelson-Siegel profile-grid fitting.
- Vasicek and CIR short-rate pricing; simplified CDS risky annuity, protection leg, hazard-rate and structural-credit measures.

### Valuation, portfolio accounting and capital

- DCF, terminal-value, Gordon growth and dividend-discount models; EV/equity bridges and multiples.
- CAPM and multi-factor cost of equity, after-tax debt cost, WACC and leverage/beta conversions.
- DuPont analysis, Altman Z screen, Piotroski-style nine-check score, cash conversion, accruals and FCF yield.
- A long-only, multi-currency paper ledger with FIFO lots, trades, splits, dividends, cash interest, marked NAV and TWR/IRR utilities.

## Installation

1. Extract the ZIP into the root of the Martha_Stewart checkout.
2. Ensure the repository root is on `PYTHONPATH` (or run commands from that root).
3. Run the extension checks:

```bash
sh scripts/technical_finance/check.sh
```

No third-party pip packages are needed for this extension. If you already use a virtual environment for the repository, it can be used unchanged.

## Golden-path demo

```bash
python examples/technical_finance/run_demo.py
```

The demo uses deterministic synthetic returns, builds a covariance matrix, estimates a long-only allocation, prints risk and performance measures, prices an option and solves implied volatility, prices a bond, and runs a small paper-ledger workflow. Its values are illustrative only; no market data are fetched.

## Minimal usage

```python
from martha_plus.technical_finance.risk import historical_var, historical_expected_shortfall
from martha_plus.technical_finance.portfolio import covariance_matrix, risk_parity_weights
from martha_plus.technical_finance.options.black_scholes import price, implied_volatility

asset_returns = [
    [0.010, -0.004, 0.006, 0.002, -0.003, 0.008],
    [0.006, -0.002, 0.004, 0.001, -0.001, 0.005],
]
sigma = covariance_matrix(asset_returns)
weights = risk_parity_weights(sigma)
portfolio_returns = [
    weights[0] * a + weights[1] * b
    for a, b in zip(*asset_returns)
]
print(historical_var(portfolio_returns, confidence=0.95))
print(historical_expected_shortfall(portfolio_returns, confidence=0.95))

call = price(100, 100, 1.0, 0.04, 0.22, option_type="call")
vol = implied_volatility(call, 100, 100, 1.0, 0.04, option_type="call")
print(call, vol)
```

## Tests

The extension includes a `unittest` suite covering numerical primitives, regression, factor construction, portfolio optimizers, risk measures, option prices/Greeks/IV, fixed income, credit, econometrics, valuation, ledger accounting, volatility estimators and Monte Carlo simulation.

```bash
PYTHONPATH=. python -m unittest discover -s tests/technical_finance -v
```

## Important modeling limitations

This is a research package, not an investment recommendation, live execution adapter, risk certification or regulatory capital calculator. In particular:

- Statistical estimators assume correctly aligned, point-in-time inputs. The package cannot determine whether upstream data contains look-ahead bias, survivorship bias, stale marks or corporate-action errors.
- The plain OLS implementation uses normal equations; ill-conditioned regressions should be checked with QR/SVD tools.
- The Black-Scholes models assume European exercise, constant parameters and frictionless markets. The CRR and stochastic processes are numerical research implementations.
- The option-surface convexity check is a screening diagnostic; it is not a full static-arbitrage proof.
- The bond and swap examples use regular schedules and simplified single-curve assumptions. They do not model calendars, stubs, collateral discounting, payment lags, settlement conventions or optionality in full detail.
- The CDS formulas use simplified flat-hazard and midpoint-default approximations. The Merton probability is risk-neutral within a stylized structural model, not a real-world default forecast.
- GARCH parameters are supplied by the caller; this extension does not claim an optimizer-estimated or validated GARCH fit.
- Monte Carlo uncertainty estimates depend on the chosen model, discretization and sampling scheme.
- The paper ledger is long-only, does not implement short borrow, tax lots with jurisdictional rules, settlement fails, margin, financing schedules or official books and records.
- Financial-statement ratios depend on consistent definitions and accounting periods. Piotroski-style checks are a screening implementation, not an audited factor replication.

Treat all outputs as diagnostics that require independent validation before use in financial decisions.
