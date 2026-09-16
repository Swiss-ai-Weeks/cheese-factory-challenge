#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CLASSIFIER=${1:-model}
MAX_OBJECTS=${CHEESE_MAX_OBJECTS:-11}
IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}
ISAAC_SIM_DATA_DIR=${ISAAC_SIM_DATA:-"$HOME/docker/isaac-sim"}

# Isaac's container runs as UID 1234; expose only this ignored output directory
# for writes instead of making the source tree broadly writable.
install -d -m 0777 "$PROJECT_ROOT/outputs/factory"

exec docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace \
  -e CHEESE_CLASSIFIER="$CLASSIFIER" \
  -e CHEESE_MAX_OBJECTS="$MAX_OBJECTS" \
  -e CHEESE_EXIT_ON_COMPLETE=1 \
  -v "$PROJECT_ROOT:/workspace:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/main:/isaac-sim/.cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$ISAAC_SIM_DATA_DIR/data:/isaac-sim/.local/share/ov/data:rw" \
  --tmpfs /var/cache/hub:rw,uid=1234,gid=1234 \
  --entrypoint /isaac-sim/runheadless.sh \
  "$IMAGE" \
  --exec /workspace/sim/factory/kit_entry.py
