# Technical Finance Extension — Merge Notes

This overlay adds a namespace-package subtree at `martha_plus/technical_finance/` and isolated test/example paths. It intentionally does not create `martha_plus/__init__.py`, so it can coexist with namespace-style extensions such as `martha_plus/microstructure/`.

## Suggested merge sequence

1. Extract at repository root.
2. Run `sh scripts/technical_finance/check.sh`.
3. Run the golden-path script and inspect its synthetic-data warnings.
4. Review modeling assumptions in `docs/martha_plus/technical_finance/REFERENCE.md` before wiring functions into upstream backtest or risk paths.
5. Integrate through explicit adapters rather than coupling the analytical layer to web routes or any live order-submission path.

## Integration suggestions

- Use `returns.py` and `performance.py` to standardize performance report conventions.
- Pass point-in-time factor panels to `factors.py`; keep data alignment and survivorship policy in the upstream data-contract layer.
- Feed `portfolio.py` covariance estimates into optimizer routines only after validating return frequency, missing-data rules and annualization.
- Route model outputs through existing risk controls; do not treat standalone VaR or stress functions as order-blocking gates.
- Keep options, rates and credit curves in independently versioned modules so model calibration assumptions remain auditable.
- Treat `ledger.py` as a paper/research ledger only. It should not replace official cash, position, margin or settlement records.

No vendor adapter, brokerage adapter, authentication flow, order submission endpoint or network data client is added by this extension.
