# Stage 14 — safe operator controls

## Problem

The streamed factory could be started through several low-level scripts, but a demo
operator had to remember environment variables and recovery details. A stale Isaac
process could also look healthy while serving an older repository commit.

## Resolution

`infra/isaac-sim/factory-demo.sh` is now the narrow operator entry point. It exposes an
allowlisted set of actions: `status`, `showcase`, `replay`, `model`, `development`,
`recover`, `preflight`, and `stop`. The actions delegate to the canonical launcher and
never evaluate arbitrary input. `replay` recreates the showcase with a new runtime ID,
while `recover` accepts only the three documented modes.

`status` reports the branch, repository commit, Compose health, factory-owned runtime
identity, mode, scenario, phase, metrics, and viewer URL. It warns when the live runtime
commit differs from repository `HEAD`. The demo runbook now starts with these commands
and states the evidence boundary between scripted showcase routing and trained-model
behavior.

## Verification

- `bash -n infra/isaac-sim/factory-demo.sh`
- `PYTHONPATH=. .venv/bin/pytest -q` — 46 tests passed
- `git diff --check`
- `CHEESE_MAX_OBJECTS=1 infra/isaac-sim/factory-demo.sh replay`
- all three stream services became healthy
- runtime `074ff798181b-20260919T085303Z-2587232-22961` reported scenario
  `judge-showcase-replay`, phase `complete`, one object, one successful physical pick,
  and one correct placement
- the runtime commit matched repository `HEAD`, so `status` emitted no stale-runtime
  warning

The one-object replay verifies reset/recovery and the physical control path. Its
scripted routing result is not classifier-accuracy evidence.
