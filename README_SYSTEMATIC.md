# Systematic Research Extension

`martha_plus.systematic` composes the existing quantitative, finance, and microstructure libraries and adds fail-closed point-in-time data controls plus a lagged, turnover-cost-aware backtest primitive. Existing tested implementations are reused rather than copied into a parallel 5,000-line tree.

## Run

From the repository root, on Python 3.10+:

```sh
sh scripts/systematic/check.sh
python examples/systematic/run_demo.py
```

## New safety APIs

- `point_in_time_join`: backward as-of join on explicit availability timestamps; never selects future rows, detects stale/missing matches and preserves match age.
- `UniverseInterval` / `members_at`: uses valid time and known-at time to reduce survivorship leakage.
- `delisting_return`: requires explicit recovery proceeds.
- `lagged_signal_backtest`: applies close-t signal to t+1, bounds exposure and charges turnover costs.

For factor models, cointegration, adaptive statistics, portfolio optimizers, cross-asset models, execution costs, and validation, see `martha_plus.quant_research`, `martha_plus.technical_finance`, and `martha_plus.microstructure`. This is research-only; there is no broker connection or order submission. Synthetic outputs do not establish profitability.
