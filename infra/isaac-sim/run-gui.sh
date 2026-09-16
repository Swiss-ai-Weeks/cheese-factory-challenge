#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLASSIFIER=${1:-model}
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

install -d -m 0777 "$PROJECT_ROOT/outputs/factory"

cd "$SCRIPT_DIR"
CHEESE_CLASSIFIER="$CLASSIFIER" \
ISAAC_SIM_ARGS="--exec /workspace/sim/factory/kit_entry.py" \
docker compose -p isim up -d --build --force-recreate isaac-sim web-viewer remote-desktop
docker compose -p isim ps
