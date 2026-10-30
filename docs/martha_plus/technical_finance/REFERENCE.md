# Technical Finance API Reference and Conventions

## Numerical conventions

All rates and returns are decimals (`0.05` means five percent). Time is measured in years unless a function explicitly accepts days or periods. Volatility is annualized only when an annualization argument is supplied or documented. Currency amounts are expressed in the same currency unless a function explicitly converts them.

The core validators reject NaN and infinities. Monetary arithmetic in the analytical layer uses `float`; a settlement or general ledger boundary should use fixed-precision `Decimal` and an authoritative rounding convention.

## Returns and performance

`simple_returns(prices)` and `log_returns(prices)` require positive prices. `compound` accepts simple returns and rejects values below -100%. Annualized geometric return compounds the observed total return over the supplied annual observation count. Annualized volatility uses sample standard deviation and the square root of the annualization factor.

The Sharpe estimator uses arithmetic mean excess returns and sample volatility. `risk_free` can be a constant period return or aligned series. Sortino uses target shortfall root-mean-square. Maximum drawdown is expressed as a negative decimal. The report returns `NaN` for Sortino when all periods are non-negative and downside deviation is zero.

## Regression and factors

`ols(y, x)` accepts a sequence of feature columns, not a row-major design matrix. An intercept is included by default. Normal equations are convenient for small, well-conditioned examples but can be numerically unstable under multicollinearity. `newey_west_covariance` returns a HAC coefficient covariance matrix using the Bartlett kernel.

Cross-sectional factor functions accept mappings keyed by asset identifier. Composite factor scores standardize each factor cross-section and normalize weights by their absolute sum. The IC functions operate on a single cross-section; a time series of ICs should be evaluated separately with `ic_summary`.

## Portfolio math

Covariance inputs are in squared return units. `portfolio_variance(w, Sigma)` computes `w' Sigma w`. Weights may be signed for variance calculations, while the default optimization routines are long-only and fully invested. Covariance shrinkage intensity is an explicit user parameter; it is not an estimated Ledoit-Wolf coefficient.

`minimum_variance_weights` uses projected gradient descent. `risk_parity_weights` iteratively targets equal component variance contribution. `mean_variance_weights` maximizes a quadratic objective with box bounds and a fully-invested constraint. These routines are intended for small research allocations; validate convergence and compare against a dedicated optimizer when constraints are more complex.

Black-Litterman inputs use `P` as views-by-assets, `Q` as view returns and `Omega` as view uncertainty covariance. `tau` scales the prior covariance. All covariance and view-uncertainty matrices must be invertible and appropriately scaled. `hierarchical_risk_parity` uses single-linkage correlation clustering and recursive bisection; its linkage tree and weights should be checked for sensitivity to the sample window.

## Risk measures

Historical VaR is the positive loss at the empirical lower-return quantile. Historical expected shortfall averages observations at or below that quantile, so finite-sample ES can be sensitive to ties and sample size. Gaussian and Cornish-Fisher VaR are parametric approximations. Monte Carlo VaR draws multivariate normal returns using Cholesky factorization.

VaR should not be compared across horizons without a coherent scaling or full revaluation. `tailrisk.py` includes Hill and generalized-Pareto method-of-moments approximations; threshold selection and sample size dominate their stability, and the estimates are not production EVT calibration. The extension does not implement a complete Basel traffic-light backtest, filtered historical simulation, liquidity horizon aggregation or regulatory capital rules.

## Options

All Black-Scholes Greeks are per underlying share and use the annual rate, annual volatility and years to expiry. Vega is change in option value for a 1.00 absolute change in volatility; to express one volatility point multiply by 0.01. Theta is per year, not per calendar day. Rho is per 1.00 absolute change in annual interest rate.

`implied_volatility` uses a bracketed bisection and returns volatility in decimal units. `crr_price` supports American early exercise using a Cox-Ross-Rubinstein tree. The implied-volatility surface interpolates on absolute strike and time to expiry and requires complete four-corner cells for bilinear interpolation.

The surface calendar test checks non-decreasing total variance at each strike. The butterfly test is a discrete curvature screen on total variance, not a full option-price convexity test.

## Fixed income

Bond yield is nominal annual yield compounded at coupon frequency. Bond maturity must align exactly with coupon periods. Duration and convexity are against annual yield changes under the same periodic compounding convention. DV01 is a one-basis-point estimate using modified duration.

`DiscountCurve` interpolates linearly in log discount factors. Its flat-forward extrapolation extends the final log-DF slope. Deposit and swap bootstrap helpers assume regular dates and simplified deposit/fixed-leg conventions. For multi-curve discounting, collateral, business-day adjustments and instrument-specific day counts, build a richer curve layer around the returned discount factors.

Nelson-Siegel fitting profiles over a supplied grid of `tau` values and solves beta parameters by OLS. The selected tau is the grid point with minimum in-sample squared error; it is not a nonlinear confidence interval or a guaranteed global optimum.

## Credit

The flat-hazard approximation is `hazard = spread / (1 - recovery)`. CDS protection-leg and premium-leg PVs use midpoint survival and a regular premium schedule. The transition-matrix helper validates row-stochastic transition probabilities. The Merton structural model is a simplified equity-as-call calculation and should not be read as a physical default probability.

## Paper accounting

`PortfolioLedger` holds long-only positions and per-currency cash. A buy reduces cash and creates a FIFO lot including fees in unit cost. A sale consumes FIFO lots and records realized P&L net of fees. Splits adjust quantity and unit cost without changing position value; dividends credit net cash using current inventory. FX marks are reporting-currency units per one unit of local currency.

The time-weighted-return helper assumes external flows occur immediately after the preceding NAV observation. `money_weighted_return` solves an IRR from annualized time coordinates and may fail if a unique root is not bracketed.

## Reproducibility

Stochastic functions accept explicit seeds. Use deterministic seeds in tests and notebooks, record model assumptions and input data hashes, and never treat a seeded synthetic example as evidence of expected live performance.
