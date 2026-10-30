#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("systematic research requires Python 3.10+")
print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
PY
python3 -m compileall -q martha_plus/systematic
PYTHONPATH=. python3 -m unittest discover -s tests/systematic -p 'test_*.py' -v
PYTHONPATH=. python3 examples/systematic/run_demo.py >/dev/null
printf '%s\n' "systematic research checks passed"
