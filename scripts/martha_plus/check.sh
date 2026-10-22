#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"
python -m compileall -q martha_plus tests
python -m pytest -q tests
python -m martha_plus demo --bars 120 --seed 7 >/tmp/martha-plus-demo.json
python - <<'PYCODE'
import json
with open('/tmp/martha-plus-demo.json', encoding='utf-8') as handle:
    result = json.load(handle)
assert result['metrics']['observations'] > 0
assert result['equity_curve']
print('CLI smoke test passed')
PYCODE
