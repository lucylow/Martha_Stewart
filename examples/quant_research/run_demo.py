#!/usr/bin/env python3
"""Deterministic synthetic workflow for the quant_research extension."""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from martha_plus.quant_research import bayesian, performance, portfolio, risk, signals, timeseries, validation
from martha_plus.quant_research import volatility


def main() -> None:
    rng = random.Random(20261030)
    n = 520
    market, value, momentum = [], [], []
    for i in range(n):
        # Two synthetic volatility states: quieter early sample, noisier later sample.
        regime_scale = 0.008 if i < 340 else 0.018
        market.append(rng.gauss(0.00025, regime_scale))
        value.append(rng.gauss(0.00005, regime_scale * 0.65))
        momentum.append(rng.gauss(0.00010, regime_scale * 0.8))

    assets = [
        [0.0001 + 1.05*m + 0.20*v + rng.gauss(0, 0.004) for m, v in zip(market, value)],
        [0.0002 + 0.75*m - 0.15*v + 0.25*mo + rng.gauss(0, 0.005) for m, v, mo in zip(market, value, momentum)],
        [-0.0001 + 0.45*m + 0.65*v - 0.10*mo + rng.gauss(0, 0.006) for m, v, mo in zip(market, value, momentum)],
        [0.0001 + 0.30*m - 0.25*v + 0.70*mo + rng.gauss(0, 0.007) for m, v, mo in zip(market, value, momentum)],
    ]
    panel = [list(row) for row in zip(*assets)]  # time x asset
    cov_result = risk.shrink_covariance(panel, target="diagonal")
    covariance = cov_result["covariance"]
    mean_returns = portfolio.robust_expected_returns(panel, method="huber")
    risk_budget = portfolio.equal_risk_contribution(covariance)
    factor_model = portfolio.factor_risk_model(assets, [market, value, momentum])

    market_states = timeseries.fit_gaussian_hmm(market, states=2, max_iter=80)
    realized_variance = [max(1e-8, x*x) for x in market]
    har_model = volatility.fit_har(realized_variance)
    har_next = volatility.forecast_har(har_model, realized_variance, horizon=5)

    strat = [sum(w * row[j] for j, w in enumerate(risk_budget["weights"])) for row in panel]
    # PBO expects strategies x contiguous time-block performance.
    candidates = [strat]
    for asset in assets[:3]:
        candidates.append([0.5*a + 0.5*b for a, b in zip(strat, asset)])
    score_matrix = [[sum(candidate[k:k+65]) / 65 for k in range(0, n, 65)]
                    for candidate in candidates]
    pbo = validation.probability_of_backtest_overfitting(score_matrix)
    dsr = validation.deflated_sharpe(strat, trials=25)
    perf = performance.summary(strat)
    labels = signals.triple_barrier_labels(
        [100.0 * math.exp(sum(market[:i+1])) for i in range(n)],
        [max(0.001, math.sqrt(realized_variance[i])) for i in range(n)],
        profit_take=1.5, stop_loss=1.5, max_holding=10,
    )
    posterior = bayesian.bayesian_shrinkage_mean(market, prior_mean=0.0, prior_strength=20.0)

    print("Martha_Stewart quant_research — synthetic demonstration")
    print("------------------------------------------------------")
    print(f"observations/assets/factors : {n} / {len(assets)} / 3")
    print(f"shrunk covariance intensity : {cov_result['intensity']:.4f} (heuristic)")
    print("robust daily mean estimates : " + ", ".join(f"{x:.6f}" for x in mean_returns))
    print("equal-risk weights          : " + ", ".join(f"{x:.3f}" for x in risk_budget["weights"]))
    print(f"risk-budget solver converged: {risk_budget['converged']}")
    print(f"factor-model residual risk  : " + ", ".join(f"{x:.6g}" for x in factor_model["specific_variance"]))
    print(f"HMM estimated states        : {market_states.means}")
    print("HAR 5-step variance forecast: " + ", ".join(f"{x:.6g}" for x in har_next))
    print(f"PBO diagnostic              : {pbo['pbo']:.4f} (synthetic block scores)")
    print(f"deflated Sharpe probability : {dsr['deflated_sharpe_probability']:.4f} (25 trials assumed)")
    print(f"annualized Sharpe estimate  : {perf['sharpe']:.4f}")
    print(f"triple-barrier event labels : {len(labels['labels'])}")
    print(f"Bayesian market mean        : {posterior:.6f}")
    print("\nLIMITATION: deterministic synthetic data only; no alpha, profitability, capacity,")
    print("execution realism, or production valuation is established by this demonstration.")


if __name__ == "__main__":
    main()
