#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CRITICAL_FAILURES=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1)); }

printf 'Cheese Factory environment check\n'
printf 'project=%s\n' "$PROJECT_ROOT"
printf 'isaac_image=%s\n' "${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  if GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>&1); then
    pass "GPU: $GPU_INFO"
  else
    fail "nvidia-smi query failed: $GPU_INFO"
  fi
else
  fail "nvidia-smi is not installed"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  pass "Docker daemon is reachable"
else
  fail "Docker daemon is not reachable"
fi

for container in isim-isaac-sim-1 isim-web-viewer-1 isim-remote-desktop-1 isaacsim-mcp; do
  state=$(docker inspect --format '{{.State.Status}}{{if .State.Health}}/{{.State.Health.Status}}{{end}}' "$container" 2>/dev/null || true)
  if [[ "$state" == "running/healthy" || "$state" == "running" ]]; then
    pass "$container: $state"
  else
    fail "$container: ${state:-missing}"
  fi
done

if curl -fsS --max-time 5 http://127.0.0.1:9904/health >/dev/null 2>&1; then
  pass "Isaac Sim documentation MCP: healthy"
else
  fail "Isaac Sim documentation MCP: unavailable at 127.0.0.1:9904"
fi

if curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null 2>&1; then
  pass "optional H200 Qwen endpoint: healthy"
else
  warn "optional H200 Qwen endpoint is offline (not required by the factory)"
fi

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  pass "Python: $("$PROJECT_ROOT/.venv/bin/python" --version 2>&1)"
else
  fail "Project Python environment is missing at .venv/bin/python"
fi

df -h "$PROJECT_ROOT" | awk 'NR == 2 {print "INFO  disk: " $4 " free of " $2 " (" $5 " used)"}'

if [[ -f "$PROJECT_ROOT/runs/sim_type13/best.pt" ]]; then
  pass "trained checkpoint present: runs/sim_type13/best.pt"
else
  warn "trained checkpoint missing: runs/sim_type13/best.pt"
fi

if [[ -f "$PROJECT_ROOT/runs/sim_bin_adapt_v2/best.pt" ]]; then
  pass "routing checkpoint present: runs/sim_bin_adapt_v2/best.pt"
else
  warn "routing checkpoint missing: runs/sim_bin_adapt_v2/best.pt"
fi

SORTER_HEALTH=$(curl -fsS --max-time 5 http://127.0.0.1:8765/health 2>/dev/null || true)
if [[ "$SORTER_HEALTH" == *'"routing": "direct_bin"'* ]]; then
  pass "production perception service: healthy"
else
  warn "production perception service is stopped (run-gui.sh model starts it)"
fi

if curl -fsS --max-time 5 http://127.0.0.1:8766/health >/dev/null 2>&1; then
  pass "optional factory control gateway: healthy"
else
  warn "optional factory control gateway is stopped"
fi

if [[ -d "$PROJECT_ROOT/data" ]]; then
  pass "dataset directory present: data/"
else
  warn "dataset directory missing: data/"
fi

if [[ -d "$PROJECT_ROOT/sim/out" ]]; then
  pass "raw render directory present: sim/out/"
else
  warn "raw render directory missing: sim/out/"
fi

if (( CRITICAL_FAILURES > 0 )); then
  printf 'RESULT critical_failures=%d\n' "$CRITICAL_FAILURES"
  exit 1
fi

printf 'RESULT critical_failures=0\n'
