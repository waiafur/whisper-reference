#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT="${REPO}/reference_outputs.json"
RUN1="${REPO}/reference_outputs.run1.json"

cp "$CURRENT" "$RUN1"
bash "${REPO}/scripts/run_on_device.sh"

python3 - "$RUN1" "$CURRENT" <<'PY'
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
if a != b:
    sys.exit("FAIL: reference outputs differ between repeated runs")
print("OK: repeated runs produced identical text and timestamps")
PY
