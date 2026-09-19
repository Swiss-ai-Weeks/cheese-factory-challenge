#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$PROJECT_ROOT/infra/isaac-sim/docker-compose.yml"
SEED=${CHEESE_REPLICATOR_SEED:-2026}
VIEWS=${CHEESE_REPLICATOR_VIEWS:-3}
LIMIT=${CHEESE_REPLICATOR_LIMIT:-6}
RESTORE_CLASSIFIER=${CHEESE_RESTORE_CLASSIFIER:-model}
DATASET_A=replicator_smoke_a
DATASET_B=replicator_smoke_b
RESTORE_REQUIRED=0

restore_service() {
  if ((RESTORE_REQUIRED)); then
    "$SCRIPT_DIR/run-gui.sh" "$RESTORE_CLASSIFIER" >/dev/null
  fi
}
trap restore_service EXIT
trap 'exit 130' INT TERM

RESTORE_REQUIRED=1
docker compose -p isim -f "$COMPOSE" stop isaac-sim >/dev/null

for dataset in "$DATASET_A" "$DATASET_B"; do
  echo "REPLICATOR_SMOKE dataset=$dataset seed=$SEED limit=$LIMIT views=$VIEWS"
  CHEESE_REPLICATOR_DATASET="$dataset" \
  CHEESE_REPLICATOR_SEED="$SEED" \
  CHEESE_REPLICATOR_VIEWS="$VIEWS" \
  CHEESE_REPLICATOR_LIMIT="$LIMIT" \
    "$SCRIPT_DIR/capture-replicator.sh"
done

"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/audit_replicator_dataset.py" \
  --manifest "$PROJECT_ROOT/data/processed/manifest_${DATASET_A}.csv" \
  --reference "$PROJECT_ROOT/data/processed/manifest_${DATASET_B}.csv" \
  --views "$VIEWS" --sources "$LIMIT" \
  --report "$PROJECT_ROOT/outputs/replicator-smoke-audit.json"

restore_service
RESTORE_REQUIRED=0
