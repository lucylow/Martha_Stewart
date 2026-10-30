#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT"
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("technical_finance requires Python 3.10+")
print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
PY
python -m compileall -q martha_plus/technical_finance
PYTHONPATH=. python -m unittest discover -s tests/technical_finance -v
python examples/technical_finance/run_demo.py >/dev/null
printf '%s\n' "technical_finance checks passed"
