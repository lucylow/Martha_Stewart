# Codebase Size and Validation Notes — v0.3.0

Recalculate with `python scripts/martha_plus/count_lines.py`. The script counts non-empty Python files under `martha_plus/` and `tests/`; page equivalents use 50 physical source lines.

- Base v0.1.0 implementation: **3,772 lines**.
- v0.2.0 advanced implementation: **5,165 lines**.
- v0.3.0 enterprise implementation: **5,121 lines** (103 page-equivalents).
- Total implementation: **14,058 lines**.
- Base and advanced tests: **1,754 lines**.
- Enterprise regression tests: **646 lines**.
- Integrated legacy-compatibility tests: **80 lines**.
- Total tests: **2,480 lines**.
- Combined total: **16,538 Python lines**, equivalent to **331 pages** at 50 lines/page.

Validation in the merged repository is run with `sh scripts/martha_plus/check.sh`. The v3 pack itself passed **241 pytest tests** with the optional Flask stack installed. The merged count also includes the v2 compatibility regressions; the merged tree adds the two v2 compatibility regression tests, and the check script exercises the combined tree. The checks do not establish production readiness, regulatory compliance, financial correctness, or profitability.

The package remains research/paper-only. It contains no live broker adapter or live order-submission endpoint. In-memory authorization/rate/idempotency controls are not distributed guarantees; caller services must enforce block/halt results and durable persistence where required.
