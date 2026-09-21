#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=${ISAAC_SIM_DATA:-"${HOME}/docker/isaac-sim"}
HUB_ROOT=${ISAACSIM_HUB_CACHE_PATH:-"${HOME}/.cache/ov/hub"}

# The Isaac container runs as uid/gid 1234. Newly created cache paths therefore
# need container-write permission without relying on an interactive sudo repair.
paths=(
  "$DATA_ROOT/cache/main"
  "$DATA_ROOT/cache/computecache"
  "$DATA_ROOT/cache/kit"
  "$DATA_ROOT/config"
  "$DATA_ROOT/data"
  "$DATA_ROOT/logs"
  "$DATA_ROOT/pkg"
  "$HUB_ROOT"
)

for path in "${paths[@]}"; do
  if [[ ! -d "$path" ]]; then
    install -d -m 0777 "$path"
  fi
done

printf 'Prepared persistent Isaac Sim paths under %s\n' "$DATA_ROOT"
printf 'Hub cache: %s\n' "$HUB_ROOT"
