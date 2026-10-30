#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q "$ROOT/martha_plus/quant_research"
python -m unittest discover -s "$ROOT/tests/quant_research" -p 'test_*.py' -v
python "$ROOT/examples/quant_research/run_demo.py"
