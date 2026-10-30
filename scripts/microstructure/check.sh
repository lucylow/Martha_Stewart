#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT"
python -m compileall -q martha_plus/microstructure tests/microstructure examples/microstructure
python -m unittest discover -s tests/microstructure -p 'test_*.py' -v
