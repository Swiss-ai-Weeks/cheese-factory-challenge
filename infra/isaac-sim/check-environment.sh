#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
MODE=${1:-showcase}
PROVENANCE="$PROJECT_ROOT/config/runtime-provenance.json"
CRITICAL_FAILURES=0

pass() { printf 'PASS  %s\n' "$*"; }
info() { printf 'INFO  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1)); }

if [[ "$MODE" != "showcase" && "$MODE" != "model" && "$MODE" != "development" ]]; then
  printf 'mode must be showcase, model, or development\n' >&2
  exit 2
fi

printf 'Cheese Factory fast preflight\n'
printf 'project=%s\nmode=%s\n' "$PROJECT_ROOT" "$MODE"

for command in git docker curl python3 nvidia-smi sha256sum; do
  if command -v "$command" >/dev/null 2>&1; then
    pass "command: $command"
  else
    fail "required command is missing: $command"
  fi
done

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then pass "Docker daemon is reachable"; else fail "Docker daemon is not reachable by this user"; fi
  if docker compose version >/dev/null 2>&1; then pass "Docker Compose plugin is available"; else fail "Docker Compose plugin is unavailable"; fi
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    pass "NVIDIA Container Runtime is registered"
  else
    fail "Docker has no NVIDIA runtime; install NVIDIA Container Toolkit"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>&1); then pass "GPU: $gpu_info"; else fail "nvidia-smi query failed: $gpu_info"; fi
fi

if [[ -f "$PROVENANCE" ]]; then
  image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["isaac_sim"]["image"])' "$PROVENANCE" 2>/dev/null || true)
  if [[ -n "$image" ]]; then
    printf 'isaac_image=%s\n' "${ISAAC_SIM_IMAGE:-$image}"
    if docker image inspect "${ISAAC_SIM_IMAGE:-$image}" >/dev/null 2>&1; then pass "pinned Isaac Sim image is cached"; else warn "pinned Isaac Sim image is not cached; the first launch will pull it"; fi
  else
    fail "cannot read Isaac Sim provenance from $PROVENANCE"
  fi
else
  fail "runtime provenance lock is missing: $PROVENANCE"
fi

if git -C "$PROJECT_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then pass "repository commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)"; else fail "project is not a readable Git checkout"; fi

data_root=${ISAAC_SIM_DATA:-"${HOME}/docker/isaac-sim"}
existing_parent=$data_root
while [[ ! -e "$existing_parent" && "$existing_parent" != "/" ]]; do
  existing_parent=$(dirname "$existing_parent")
done
if [[ -d "$data_root" ]]; then
  pass "persistent cache location exists: $data_root"
elif [[ -w "$existing_parent" ]]; then
  pass "persistent cache location can be prepared: $data_root"
else
  fail "cannot create persistent cache location: $data_root"
fi

if [[ "$MODE" == "model" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    python_version=$($PROJECT_ROOT/.venv/bin/python --version 2>&1)
    [[ "$python_version" == "Python 3.12.14" ]] && pass "$python_version" || warn "$python_version; reference lock is Python 3.12.14"
  else
    fail "model mode needs .venv/bin/python; install requirements-runtime.txt with Python 3.12"
  fi

  python3 - "$PROVENANCE" "$PROJECT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
failed = False
for name, entry in lock["models"].items():
    path = root / entry["path"]
    if not path.is_file():
        print(f"FAIL  {name}: missing {entry['path']}")
        failed = True
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    if digest != entry["sha256"] or size != entry["bytes"]:
        print(f"FAIL  {name}: provenance mismatch sha256={digest} bytes={size}")
        failed = True
    else:
        print(f"PASS  {name}: sha256={digest} bytes={size}")
raise SystemExit(1 if failed else 0)
PY
  [[ $? -eq 0 ]] || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
fi

if curl -fsS --max-time 2 http://127.0.0.1:9904/health >/dev/null 2>&1; then info "optional Isaac Sim documentation MCP is healthy"; else info "documentation MCP is offline; it is not part of the factory runtime"; fi
if curl -fsS --max-time 2 http://127.0.0.1:18000/health >/dev/null 2>&1; then info "optional H200 Qwen endpoint is healthy"; else info "optional H200 Qwen endpoint is offline; it is not used by the factory"; fi

df -h "$PROJECT_ROOT" | awk 'NR == 2 {print "INFO  disk: " $4 " free of " $2 " (" $5 " used)"}'

if (( CRITICAL_FAILURES > 0 )); then
  printf 'RESULT critical_failures=%d\n' "$CRITICAL_FAILURES"
  exit 1
fi
printf 'RESULT critical_failures=0\n'
