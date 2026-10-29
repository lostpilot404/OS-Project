#!/bin/bash
# Manual evidence capture: drive the live dashboard API with curl (browser unavailable).
# Usage: ./run_scenario.sh <scenario-id> [options-json-file]
set -u
S=http://127.0.0.1:8000
OUT=audit/manual-iceberg/evidence/api
SCEN=$1
OPT_FILE=${2:-}
if [ -n "$OPT_FILE" ] && [ -f "$OPT_FILE" ]; then
  BODY=$(python -c "import json,sys; print(json.dumps({'scenario':'$SCEN','options':json.load(open('$OPT_FILE'))}))")
else
  BODY=$(python -c "import json; print(json.dumps({'scenario':'$SCEN','options':{}}))")
fi
echo "POST $BODY"
curl -s -m 60 -X POST $S/api/simulations \
  -H 'Content-Type: application/json' \
  -d "$BODY" -o "$OUT/${SCEN}.json"
python -m json.tool "$OUT/${SCEN}.json" > "$OUT/${SCEN}.pretty.json" 2>&1 || true
echo "--- saved $OUT/${SCEN}.json ---"
