"""Deterministic synthetic demo; it does not establish investment profitability."""
from __future__ import annotations

import json
from dataclasses import asdict

from martha_plus.systematic import lagged_signal_backtest, point_in_time_join


def main() -> None:
    decisions = [
        {"symbol": "DEMO", "decision_ts_ns": t, "signal": s}
        for t, s in zip((10, 20, 30, 40, 50), (0.2, 0.8, 1.2, -0.4, 0.0))
    ]
    features = [
        {"symbol": "DEMO", "available_ts_ns": t, "factor": value}
        for t, value in ((8, 0.1), (18, 0.3), (35, 0.4), (42, -0.1))
    ]
    matches = point_in_time_join(decisions, features, require_match=False)
    signals = [row["signal"] for row in decisions]
    synthetic_returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    result = lagged_signal_backtest(signals, synthetic_returns,
                                    cost_bps=5.0, exposure_limit=1.0)
    summary = {
        "mode": "synthetic_demo_only",
        "pit_matches": [
            {"decision_ts_ns": item.decision["decision_ts_ns"],
             "feature_available_ts_ns": item.observation["available_ts_ns"] if item.observation else None,
             "age_ns": item.age_ns}
            for item in matches
        ],
        "positions": result.positions,
        "net_returns": result.net_returns,
        "total_turnover": result.total_turnover,
        "total_cost": result.total_cost,
        "sharpe": result.sharpe,
        "drawdown": result.drawdown,
        "notice": "Synthetic fixtures only; no alpha or profitability is implied.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
