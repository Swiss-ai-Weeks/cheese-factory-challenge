#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PID_FILE="$PROJECT_ROOT/outputs/factory/sort-server.pid"

docker compose -p isim -f "$SCRIPT_DIR/docker-compose.yml" stop \
  remote-desktop web-viewer isaac-sim

if [[ -f "$PID_FILE" ]]; then
  PID=$(<"$PID_FILE")
  if [[ "$PID" =~ ^[0-9]+$ ]] && [[ -r "/proc/$PID/cmdline" ]]; then
    COMMAND=$(tr '\0' ' ' <"/proc/$PID/cmdline")
    if [[ "$COMMAND" == *"$PROJECT_ROOT/sim/sort_server.py"* ]]; then
      kill "$PID"
    else
      echo "refusing to stop PID $PID because it is not this repository's sorter" >&2
      exit 1
    fi
  fi
  rm -f "$PID_FILE"
fi
