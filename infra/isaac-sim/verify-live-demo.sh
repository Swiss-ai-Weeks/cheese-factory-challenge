#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$SCRIPT_DIR/docker-compose.yml"
MODE=${1:-showcase}
SCENARIO=${2:-judge-showcase}
STATUS="$PROJECT_ROOT/outputs/factory/runtime-status.json"
EXPECTED_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)
VIEWER_URL=${CHEESE_VIEWER_URL:-http://127.0.0.1:6080/vnc.html}

for service in isaac-sim web-viewer remote-desktop; do
  state=missing
  for _ in $(seq 1 60); do
    container=$(docker compose -p isim -f "$COMPOSE" ps -q "$service")
    if [[ -n "$container" ]]; then
      state=$(docker inspect --format '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container")
      [[ "$state" == "running/healthy" ]] && break
      [[ "$state" == exited/* || "$state" == dead/* ]] && break
    fi
    sleep 2
  done
  [[ "$state" == "running/healthy" ]] || { printf 'FAIL  %s is %s after health wait\n' "$service" "$state" >&2; exit 1; }
  printf 'PASS  %s: %s\n' "$service" "$state"
done

curl -fsS --max-time 10 "$VIEWER_URL" >/dev/null
printf 'PASS  browser viewer: %s\n' "$VIEWER_URL"

python3 - "$STATUS" "$EXPECTED_COMMIT" "$MODE" "$SCENARIO" <<'PY'
import json
import sys
from pathlib import Path

path, commit, mode, scenario = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding="utf-8"))
expected = {
    "application": "cheese_factory",
    "commit": commit,
    "classifier_mode": mode,
    "scenario": scenario,
}
errors = [f"{key}={payload.get(key)!r}, expected {value!r}" for key, value in expected.items() if payload.get(key) != value]
if payload.get("phase") not in {"loaded", "reset", "running", "complete"}:
    errors.append(f"phase={payload.get('phase')!r} is not a healthy runtime phase")
if not payload.get("runtime_id"):
    errors.append("runtime_id is empty")
if errors:
    raise SystemExit("FAIL  runtime identity: " + "; ".join(errors))
print(f"PASS  runtime identity: {payload['runtime_id']}")
print(f"PASS  commit/mode/scenario: {commit} / {mode} / {scenario}")
PY

if [[ "$MODE" == "model" ]]; then
  health=$(curl -fsS --max-time 10 http://127.0.0.1:8765/health)
  [[ "$health" == *'"routing": "direct_bin"'* &&
     "$health" == *'"contract_version": 2'* &&
     "$health" == *'"timing_contract_version": 1'* ]] || {
    printf 'FAIL  production sorter contract mismatch\n' >&2
    exit 1
  }
  printf 'PASS  production sorter contract v2 / timing v1\n'
fi

printf 'READY %s\n' "$VIEWER_URL"
