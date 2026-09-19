#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}
ISAAC_SIM_DATA_DIR=${ISAAC_SIM_DATA:-"$HOME/docker/isaac-sim"}
DATASET=${CHEESE_REPLICATOR_DATASET:-factory_replicator_v1}
SEED=${CHEESE_REPLICATOR_SEED:-2026}
VIEWS=${CHEESE_REPLICATOR_VIEWS:-3}
LIMIT=${CHEESE_REPLICATOR_LIMIT:-6}
SUBFRAMES=${CHEESE_REPLICATOR_SUBFRAMES:-8}
SOURCE_MANIFEST=${CHEESE_REPLICATOR_SOURCE_MANIFEST:-data/processed/manifest_factory_balanced_v2.csv}

if [[ ! "$DATASET" =~ ^[a-z0-9_]+$ ]]; then
  echo "CHEESE_REPLICATOR_DATASET must match [a-z0-9_]+" >&2
  exit 2
fi
if ((VIEWS < 1 || LIMIT < 1 || SUBFRAMES < 1)); then
  echo "views, limit and subframes must be positive" >&2
  exit 2
fi

CAPTURE_DIR="$PROJECT_ROOT/data/processed/images/$DATASET"
MANIFEST="$PROJECT_ROOT/data/processed/manifest_${DATASET}.csv"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$DATASET"
FULL_DIR="$OUTPUT_DIR/full"
install -d -m 0777 "$CAPTURE_DIR" "$OUTPUT_DIR" "$FULL_DIR"
chmod 0777 "$OUTPUT_DIR"
touch "$MANIFEST"
chmod 0666 "$MANIFEST"

exec docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace \
  -e CHEESE_REPLICATOR_DATASET="$DATASET" \
  -e CHEESE_REPLICATOR_SEED="$SEED" \
  -e CHEESE_REPLICATOR_VIEWS="$VIEWS" \
  -e CHEESE_REPLICATOR_LIMIT="$LIMIT" \
  -e CHEESE_REPLICATOR_SUBFRAMES="$SUBFRAMES" \
  -e CHEESE_REPLICATOR_SOURCE_MANIFEST="$SOURCE_MANIFEST" \
  -v "$PROJECT_ROOT:/workspace:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/main:/isaac-sim/.cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$ISAAC_SIM_DATA_DIR/data:/isaac-sim/.local/share/ov/data:rw" \
  --tmpfs /var/cache/hub:rw,uid=1234,gid=1234 \
  --entrypoint /isaac-sim/runheadless.sh "$IMAGE" \
  --exec /workspace/sim/factory/capture_replicator.py
