#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CLASSIFIER=${1:-model}
MAX_OBJECTS=${CHEESE_MAX_OBJECTS:-11}
IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20}
ISAAC_SIM_DATA_DIR=${ISAAC_SIM_DATA:-"$HOME/docker/isaac-sim"}
SORTER_HOST_URL=${CHEESE_SORTER_HOST_URL:-http://127.0.0.1:8765}
SORTER_CONTAINER_URL=${CHEESE_SORTER_URL:-http://host.docker.internal:8765}
ROUTING_CHECKPOINT=${CHEESE_ROUTING_CHECKPOINT:-"$PROJECT_ROOT/runs/sim_bin_adapt_v2/best.pt"}
SORTER_PID=""
GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)
RUNTIME_ID="headless-${GIT_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
SCENARIO=${CHEESE_SCENARIO:-default-evaluation}

# Isaac's container runs as UID 1234; expose only this ignored output directory
# for writes instead of making the source tree broadly writable.
install -d -m 0777 "$PROJECT_ROOT/outputs/factory"

if [[ "$CLASSIFIER" != "model" && "$CLASSIFIER" != "development" && "$CLASSIFIER" != "showcase" ]]; then
  echo "classifier must be model, development, or showcase" >&2
  exit 2
fi

cleanup() {
  if [[ -n "$SORTER_PID" ]]; then
    kill "$SORTER_PID" >/dev/null 2>&1 || true
    wait "$SORTER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$CLASSIFIER" == "model" ]]; then
  CUDNN_LIB="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" || ! -d "$CUDNN_LIB" ]]; then
    echo "trained-model runtime is missing (.venv or cuDNN); see docs/stages/04-real-perception.md" >&2
    exit 1
  fi
  if [[ ! -f "$ROUTING_CHECKPOINT" ]]; then
    echo "target-domain routing checkpoint is missing: $ROUTING_CHECKPOINT" >&2
    echo "capture and train Stage 5 before running the production classifier" >&2
    exit 1
  fi
  SORTER_HEALTH=$(curl -fsS --max-time 3 "$SORTER_HOST_URL/health" 2>/dev/null || true)
  if [[ -z "$SORTER_HEALTH" ]]; then
    env LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}" \
      "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/sim/sort_server.py" \
        --checkpoint "$PROJECT_ROOT/runs/sim_type13/best.pt" \
        --routing-checkpoint "$ROUTING_CHECKPOINT" \
        --host 0.0.0.0 --port 8765 \
        >"$PROJECT_ROOT/outputs/factory/sort-server.log" 2>&1 &
    SORTER_PID=$!
    for _ in $(seq 1 90); do
      SORTER_HEALTH=$(curl -fsS --max-time 2 "$SORTER_HOST_URL/health" 2>/dev/null || true)
      [[ -n "$SORTER_HEALTH" ]] && break
      kill -0 "$SORTER_PID" 2>/dev/null || {
        cat "$PROJECT_ROOT/outputs/factory/sort-server.log" >&2
        exit 1
      }
      sleep 1
    done
    if [[ -z "$SORTER_HEALTH" ]]; then
      echo "trained perception service did not become healthy within 90 seconds" >&2
      exit 1
    fi
  fi
  if [[ "$SORTER_HEALTH" != *'"routing": "direct_bin"'* ||
        "$SORTER_HEALTH" != *'"contract_version": 2'* ||
        "$SORTER_HEALTH" != *'"timing_contract_version": 1'* ]]; then
    echo "perception service on $SORTER_HOST_URL lacks the required routing/timing contract" >&2
    echo "stop the stale service before starting the factory" >&2
    exit 1
  fi
fi

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace \
  -e CHEESE_CLASSIFIER="$CLASSIFIER" \
  -e CHEESE_MAX_OBJECTS="$MAX_OBJECTS" \
  -e CHEESE_EXIT_ON_COMPLETE=1 \
  -e CHEESE_SORTER_URL="$SORTER_CONTAINER_URL" \
  -e CHEESE_RUNTIME_ID="$RUNTIME_ID" \
  -e CHEESE_GIT_COMMIT="$GIT_COMMIT" \
  -e CHEESE_SCENARIO="$SCENARIO" \
  -e CHEESE_ISAAC_IMAGE="$IMAGE" \
  --add-host host.docker.internal:host-gateway \
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
