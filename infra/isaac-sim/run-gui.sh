#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLASSIFIER=${1:-model}
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SORTER_URL=${CHEESE_SORTER_HOST_URL:-http://127.0.0.1:8765}
SORTER_PID_FILE="$PROJECT_ROOT/outputs/factory/sort-server.pid"
ROUTING_CHECKPOINT=${CHEESE_ROUTING_CHECKPOINT:-"$PROJECT_ROOT/runs/sim_bin_adapt_v2/best.pt"}
SORTER_PID=""
GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)
RUNTIME_ID="${GIT_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
SCENARIO=${CHEESE_SCENARIO:-default-evaluation}

install -d -m 0777 "$PROJECT_ROOT/outputs/factory"

if [[ "$CLASSIFIER" != "model" && "$CLASSIFIER" != "development" && "$CLASSIFIER" != "showcase" ]]; then
  echo "classifier must be model, development, or showcase" >&2
  exit 2
fi

if [[ "$CLASSIFIER" == "model" ]]; then
  CUDNN_LIB="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" || ! -d "$CUDNN_LIB" ]]; then
    echo "trained-model runtime is missing (.venv or cuDNN)" >&2
    exit 1
  fi
  if [[ ! -f "$PROJECT_ROOT/runs/sim_type13/best.pt" || ! -f "$ROUTING_CHECKPOINT" ]]; then
    echo "production checkpoints are missing; see docs/stages/05-production-perception.md" >&2
    exit 1
  fi
  SORTER_HEALTH=$(curl -fsS --max-time 3 "$SORTER_URL/health" 2>/dev/null || true)
  if [[ -n "$SORTER_HEALTH" && (
        "$SORTER_HEALTH" != *'"routing": "direct_bin"'* ||
        "$SORTER_HEALTH" != *'"contract_version": 2'* ||
        "$SORTER_HEALTH" != *'"decision_policy": "route_authoritative_fail_closed_v1"'*
      ) ]]; then
    if [[ -f "$SORTER_PID_FILE" ]]; then
      STALE_SORTER_PID=$(cat "$SORTER_PID_FILE")
      STALE_SORTER_COMMAND=$(tr '\0' ' ' <"/proc/$STALE_SORTER_PID/cmdline" 2>/dev/null || true)
      if kill -0 "$STALE_SORTER_PID" 2>/dev/null &&
         [[ "$STALE_SORTER_COMMAND" == *"$PROJECT_ROOT/sim/sort_server.py"* ]]; then
        kill "$STALE_SORTER_PID"
        wait "$STALE_SORTER_PID" 2>/dev/null || true
        rm -f "$SORTER_PID_FILE"
        SORTER_HEALTH=""
      fi
    fi
  fi
  if [[ -z "$SORTER_HEALTH" ]]; then
    nohup env LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}" \
      "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/sim/sort_server.py" \
        --checkpoint "$PROJECT_ROOT/runs/sim_type13/best.pt" \
        --routing-checkpoint "$ROUTING_CHECKPOINT" \
        --host 0.0.0.0 --port 8765 \
        >"$PROJECT_ROOT/outputs/factory/sort-server.log" 2>&1 </dev/null &
    SORTER_PID=$!
    echo "$SORTER_PID" >"$SORTER_PID_FILE"
    for _ in $(seq 1 90); do
      SORTER_HEALTH=$(curl -fsS --max-time 2 "$SORTER_URL/health" 2>/dev/null || true)
      [[ -n "$SORTER_HEALTH" ]] && break
      kill -0 "$SORTER_PID" 2>/dev/null || {
        cat "$PROJECT_ROOT/outputs/factory/sort-server.log" >&2
        rm -f "$SORTER_PID_FILE"
        exit 1
      }
      sleep 1
    done
  fi
  if [[ "$SORTER_HEALTH" != *'"routing": "direct_bin"'* ||
        "$SORTER_HEALTH" != *'"contract_version": 2'* ||
        "$SORTER_HEALTH" != *'"decision_policy": "route_authoritative_fail_closed_v1"'* ]]; then
    if [[ -n "$SORTER_PID" ]]; then
      kill "$SORTER_PID" >/dev/null 2>&1 || true
      wait "$SORTER_PID" 2>/dev/null || true
      rm -f "$SORTER_PID_FILE"
    fi
    echo "service on $SORTER_URL does not implement the safe perception contract v2" >&2
    exit 1
  fi
fi

cd "$SCRIPT_DIR"
CHEESE_CLASSIFIER="$CLASSIFIER" \
CHEESE_SORTER_URL="$SORTER_URL" \
CHEESE_RUNTIME_ID="$RUNTIME_ID" \
CHEESE_GIT_COMMIT="$GIT_COMMIT" \
CHEESE_SCENARIO="$SCENARIO" \
ISAAC_SIM_ARGS="--exec /workspace/sim/factory/kit_entry.py" \
docker compose -p isim up -d --build --force-recreate isaac-sim web-viewer remote-desktop
docker compose -p isim ps
echo "FACTORY_RUNTIME_ID=$RUNTIME_ID"
