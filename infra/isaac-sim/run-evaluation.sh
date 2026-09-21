#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$PROJECT_ROOT/infra/isaac-sim/docker-compose.yml"
CLASSIFIER=${1:-model}
MAX_OBJECTS=${2:-11}
EVIDENCE_PATH=${3:-}

if [[ -n "$EVIDENCE_PATH" && ("$EVIDENCE_PATH" = /* || "/$EVIDENCE_PATH/" == *"/../"*) ]]; then
  echo "evidence path must be relative to the repository and cannot contain '..'" >&2
  exit 2
fi

if [[ "$CLASSIFIER" != "model" && "$CLASSIFIER" != "development" && "$CLASSIFIER" != "showcase" ]]; then
  echo "classifier must be model, development, or showcase" >&2
  exit 2
fi
if [[ ! "$MAX_OBJECTS" =~ ^[0-9]+$ ]] || ((MAX_OBJECTS < 1 || MAX_OBJECTS > 11)); then
  echo "max_objects must be an integer from 1 through 11" >&2
  exit 2
fi

restore_service() {
  "$SCRIPT_DIR/run-gui.sh" "$CLASSIFIER" >/dev/null
}
trap restore_service EXIT
trap 'exit 130' INT TERM

docker compose -p isim -f "$COMPOSE" stop isaac-sim >/dev/null
EVALUATION_STATUS=0
CHEESE_MAX_OBJECTS="$MAX_OBJECTS" "$SCRIPT_DIR/run-headless.sh" "$CLASSIFIER" || EVALUATION_STATUS=$?
if [[ -n "$EVIDENCE_PATH" && "$EVALUATION_STATUS" -eq 0 ]]; then
  install -D -m 0644 "$PROJECT_ROOT/outputs/factory/results.json" "$PROJECT_ROOT/$EVIDENCE_PATH"
  printf 'EVALUATION_EVIDENCE=%s\n' "$EVIDENCE_PATH"
fi
restore_service
trap - EXIT
exit "$EVALUATION_STATUS"
