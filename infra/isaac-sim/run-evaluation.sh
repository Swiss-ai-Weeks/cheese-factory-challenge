#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$PROJECT_ROOT/infra/isaac-sim/docker-compose.yml"
CLASSIFIER=${1:-model}
MAX_OBJECTS=${2:-11}

if [[ "$CLASSIFIER" != "model" && "$CLASSIFIER" != "development" ]]; then
  echo "classifier must be model or development" >&2
  exit 2
fi
if [[ ! "$MAX_OBJECTS" =~ ^[0-9]+$ ]] || ((MAX_OBJECTS < 1 || MAX_OBJECTS > 11)); then
  echo "max_objects must be an integer from 1 through 11" >&2
  exit 2
fi

restore_service() {
  docker compose -p isim -f "$COMPOSE" up -d isaac-sim >/dev/null
}
trap restore_service EXIT
trap 'exit 130' INT TERM

docker compose -p isim -f "$COMPOSE" stop isaac-sim >/dev/null
CHEESE_MAX_OBJECTS="$MAX_OBJECTS" "$SCRIPT_DIR/run-headless.sh" "$CLASSIFIER"
