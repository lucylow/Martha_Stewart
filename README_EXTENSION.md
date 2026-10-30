# Martha Stewart — Microstructure Extension Bundle

This ZIP is an **additive patch bundle**, not a full copy of the upstream GitHub
repository. Merge its `martha_plus/microstructure/`, `tests/microstructure/`,
`examples/microstructure/`, `scripts/microstructure/`, and
`docs/martha_plus/microstructure/` paths into your checkout of
<https://github.com/lucylow/Martha_Stewart>.

It adds a paper-only event-driven order-book simulator, queue and impact
research models, data-quality/archive utilities, online analytics, portfolio
stress tools, audit/health utilities, and report exporters. It does not connect
to a live broker or submit live orders.

- Python implementation: 5,600+ lines (112+ page-equivalents at 50 lines/page).
- Included regression tests: 550+ lines across 45+ standard-library tests.
- Runtime: Python 3.10+; standard library only.
- Test command: `sh scripts/microstructure/check.sh`.
- Demo: `python examples/microstructure/run_demo.py --events 3000 --seed 7`.

Detailed architecture and unit conventions are in
`docs/martha_plus/microstructure/README.md`.
