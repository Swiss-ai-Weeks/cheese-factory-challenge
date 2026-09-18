#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}
ISAAC_SIM_DATA_DIR=${ISAAC_SIM_DATA:-"$HOME/docker/isaac-sim"}
VIEWS=${CHEESE_CAPTURE_VIEWS:-2}
OFFSET=${CHEESE_CAPTURE_OFFSET:-0}
LIMIT=${CHEESE_CAPTURE_LIMIT:-100}
VIEW_START=${CHEESE_CAPTURE_VIEW_START:-0}

# Isaac Sim runs as UID 1234 in the NVIDIA image, while the repository mount is
# normally owned by the host user. Limit shared write access to generated data.
CAPTURE_DIR="$PROJECT_ROOT/data/processed/images/factory_adapt"
CAPTURE_MANIFEST="$PROJECT_ROOT/data/processed/manifest_factory_adapt.csv"
FACTORY_MANIFEST="$PROJECT_ROOT/data/processed/manifest_factory_only.csv"
install -d -m 0777 "$CAPTURE_DIR"
touch "$CAPTURE_MANIFEST" "$FACTORY_MANIFEST"
chmod 0666 "$CAPTURE_MANIFEST" "$FACTORY_MANIFEST"

exec docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace -e CHEESE_CAPTURE_VIEWS="$VIEWS" \
  -e CHEESE_CAPTURE_OFFSET="$OFFSET" -e CHEESE_CAPTURE_LIMIT="$LIMIT" \
  -e CHEESE_CAPTURE_VIEW_START="$VIEW_START" \
  -v "$PROJECT_ROOT:/workspace:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/main:/isaac-sim/.cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$ISAAC_SIM_DATA_DIR/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$ISAAC_SIM_DATA_DIR/data:/isaac-sim/.local/share/ov/data:rw" \
  --tmpfs /var/cache/hub:rw,uid=1234,gid=1234 \
  --entrypoint /isaac-sim/runheadless.sh "$IMAGE" \
  --exec /workspace/sim/factory/capture_domain.py
