#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$PROJECT_ROOT/infra/isaac-sim/docker-compose.yml"
CHUNK_SIZE=${CHEESE_CAPTURE_CHUNK_SIZE:-100}
VIEWS=${CHEESE_CAPTURE_VIEWS:-2}
START_OFFSET=${CHEESE_CAPTURE_START_OFFSET:-0}
VIEW_START=${CHEESE_CAPTURE_VIEW_START:-0}

restore_service() {
  docker compose -p isim -f "$COMPOSE" up -d isaac-sim >/dev/null
}
trap restore_service EXIT
trap 'exit 130' INT TERM

TOTAL=$(
  "$PROJECT_ROOT/.venv/bin/python" - "$PROJECT_ROOT/data/processed" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
groups = {
    row["group"]
    for row in csv.DictReader((root / "manifest_sim.csv").open())
}
print(sum(
    1 for row in csv.DictReader((root / "cutouts" / "manifest.csv").open())
    if row["uid"] in groups
))
PY
)

docker compose -p isim -f "$COMPOSE" stop isaac-sim >/dev/null
for ((offset = START_OFFSET; offset < TOTAL; offset += CHUNK_SIZE)); do
  echo "FACTORY_CAPTURE_BATCH offset=$offset total=$TOTAL"
  CHEESE_CAPTURE_VIEWS="$VIEWS" \
  CHEESE_CAPTURE_OFFSET="$offset" \
  CHEESE_CAPTURE_LIMIT="$CHUNK_SIZE" \
  CHEESE_CAPTURE_VIEW_START="$VIEW_START" \
    "$SCRIPT_DIR/capture-domain.sh"
done
