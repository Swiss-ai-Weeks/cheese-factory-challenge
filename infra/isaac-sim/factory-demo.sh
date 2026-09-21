#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE="$SCRIPT_DIR/docker-compose.yml"

usage() {
  cat <<'EOF'
Cheese Factory operator command

Usage: infra/isaac-sim/factory-demo.sh ACTION [MODE]

Actions:
  status              Show commit, services, canonical runtime and viewing URL.
  launch [MODE]       Preflight, prepare, start and verify one healthy live demo.
  showcase            Alias for: launch showcase (recommended for judges).
  replay              Reset and replay the showcase from object 1.
  model               Start honest trained-perception mode.
  development         Start the lighting-sensitive pixel proxy diagnostic.
  recover [MODE]      Recreate the full stream in showcase, model, or development mode.
  preflight [MODE]    Fast static check; does not require running services.
  stop                Stop the streamed factory and repository-owned sorter.
  help                 Show this help.

The showcase uses camera detection/localization but scripted ground-truth routing.
It is physical-integration evidence, not classifier accuracy.
EOF
}

valid_mode() {
  [[ "$1" == "showcase" || "$1" == "model" || "$1" == "development" ]]
}

start_mode() {
  local mode=$1
  local scenario=$2
  "$SCRIPT_DIR/check-environment.sh" "$mode"
  "$SCRIPT_DIR/prepare-environment.sh"
  CHEESE_SCENARIO="$scenario" "$SCRIPT_DIR/run-gui.sh" "$mode"
  "$SCRIPT_DIR/verify-live-demo.sh" "$mode" "$scenario"
}

action=${1:-status}
case "$action" in
  help|-h|--help)
    usage
    ;;
  status)
    printf 'Cheese Factory status\n'
    printf 'branch=%s\n' "$(git -C "$PROJECT_ROOT" branch --show-current)"
    printf 'commit=%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    docker compose -p isim -f "$COMPOSE" ps
    if [[ -f "$PROJECT_ROOT/outputs/factory/runtime-status.json" ]]; then
      printf '\nCanonical runtime\n'
      python3 -m json.tool "$PROJECT_ROOT/outputs/factory/runtime-status.json"
      runtime_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$PROJECT_ROOT/outputs/factory/runtime-status.json")
      repository_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
      if [[ "$runtime_commit" != "$repository_commit" ]]; then
        printf '\nWARN runtime commit does not match repository HEAD; run factory-demo.sh recover showcase\n'
      fi
    else
      printf '\nCanonical runtime: no status has been published\n'
    fi
    printf '\nViewer (host-local): http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale\n'
    ;;
  launch)
    mode=${2:-showcase}
    if ! valid_mode "$mode"; then
      printf 'launch mode must be showcase, model, or development\n' >&2
      exit 2
    fi
    start_mode "$mode" "canonical-$mode"
    ;;
  showcase)
    start_mode showcase canonical-showcase
    ;;
  replay)
    start_mode showcase judge-showcase-replay
    ;;
  model)
    start_mode model trained-perception
    ;;
  development)
    start_mode development pixel-proxy-diagnostic
    ;;
  recover)
    mode=${2:-showcase}
    if ! valid_mode "$mode"; then
      printf 'recover mode must be showcase, model, or development\n' >&2
      exit 2
    fi
    start_mode "$mode" "recovery-$mode"
    ;;
  preflight)
    mode=${2:-showcase}
    if ! valid_mode "$mode"; then
      printf 'preflight mode must be showcase, model, or development\n' >&2
      exit 2
    fi
    exec "$SCRIPT_DIR/check-environment.sh" "$mode"
    ;;
  stop)
    exec "$SCRIPT_DIR/stop-gui.sh"
    ;;
  *)
    printf 'unknown action: %s\n\n' "$action" >&2
    usage >&2
    exit 2
    ;;
esac
