#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$PROJECT_ROOT/infra/isaac-sim/docker-compose.yml"
CHUNK_SIZE=${CHEESE_CAPTURE_CHUNK_SIZE:-100}
VIEWS=${CHEESE_CAPTURE_VIEWS:-2}
START_OFFSET=${CHEESE_CAPTURE_START_OFFSET:-0}
VIEW_START=${CHEESE_CAPTURE_VIEW_START:-0}
SPLITS=${CHEESE_CAPTURE_SPLITS:-}
BINS=${CHEESE_CAPTURE_BINS:-}
RESTORE_REQUIRED=0
RESTORE_CLASSIFIER=${CHEESE_RESTORE_CLASSIFIER:-model}

restore_service() {
  if ((RESTORE_REQUIRED)); then
    "$SCRIPT_DIR/run-gui.sh" "$RESTORE_CLASSIFIER" >/dev/null
  fi
}
trap restore_service EXIT
trap 'exit 130' INT TERM

TOTAL=$(
  "$PROJECT_ROOT/.venv/bin/python" - "$PROJECT_ROOT/data/processed" "$SPLITS" "$BINS" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = {value.strip() for value in sys.argv[2].split(",") if value.strip()}
bins = {value.strip() for value in sys.argv[3].split(",") if value.strip()}
valid_splits = {"train", "val", "test"}
valid_bins = {"bin_blue", "bin_fresh", "bin_hard", "bin_semi_hard", "bin_soft", "not_cheese", "empty"}
if splits - valid_splits or bins - valid_bins:
    raise SystemExit(f"invalid capture filter: splits={sorted(splits)} bins={sorted(bins)}")
groups = {
    row["group"]: row["split"]
    for row in csv.DictReader((root / "manifest_sim.csv").open())
}
print(sum(
    1 for row in csv.DictReader((root / "cutouts" / "manifest.csv").open())
    if row["uid"] in groups
    and (not splits or groups[row["uid"]] in splits)
    and (not bins or row["bin"] in bins)
))
PY
)

if ((TOTAL == 0)); then
  echo "FACTORY_CAPTURE_DONE no eligible sources"
  exit 0
fi

RESTORE_REQUIRED=1
docker compose -p isim -f "$COMPOSE" stop isaac-sim >/dev/null
for ((offset = START_OFFSET; offset < TOTAL; offset += CHUNK_SIZE)); do
  echo "FACTORY_CAPTURE_BATCH offset=$offset total=$TOTAL"
  CHEESE_CAPTURE_VIEWS="$VIEWS" \
  CHEESE_CAPTURE_OFFSET="$offset" \
  CHEESE_CAPTURE_LIMIT="$CHUNK_SIZE" \
  CHEESE_CAPTURE_VIEW_START="$VIEW_START" \
  CHEESE_CAPTURE_SPLITS="$SPLITS" \
  CHEESE_CAPTURE_BINS="$BINS" \
    "$SCRIPT_DIR/capture-domain.sh"
done
restore_service
RESTORE_REQUIRED=0
