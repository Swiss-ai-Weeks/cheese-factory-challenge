# Stage 10 — Canonical runtime identity and recovery

## Problem

The old Docker health check only searched the latest Kit log for `AppReady`. Plain Isaac
Sim could therefore be reported healthy even when `sim/factory/kit_entry.py` had never
loaded. Evaluation and domain-capture cleanup also restarted the compose service without
the canonical `--exec` argument, which is how WebRTC ended up showing the wrong app.

## Resolution

- `run-gui.sh` creates a unique runtime ID and supplies the exact Git commit, classifier
  mode and scenario to the container.
- The factory atomically publishes `outputs/factory/runtime-status.json` through loading,
  reset, running, completion and fatal phases.
- Docker health now requires a factory-owned status whose runtime ID matches the current
  container and whose phase proves that the scene loaded. `AppReady` alone is no longer
  sufficient.
- Evaluation and capture flows explicitly restore the canonical launcher on normal
  completion, while retaining failure/signal traps.
- Headless evaluations also record a unique identity instead of writing ambiguous
  `unknown`/`unmanaged` status.

## Verification

- Full repository tests: **34 passed**.
- Python compilation, Bash syntax, Compose configuration and `git diff --check`: passed.
- Live streamed development smoke: one object detected, picked and placed; all measured
  smoke metrics were 1.0 for that single object.
- Recovery smoke: `run-evaluation.sh development 1` completed and restored all three
  streamed services. The restored Isaac process contained
  `--exec /workspace/sim/factory/kit_entry.py` and reached Docker `healthy`.
- A deliberately mismatched runtime ID did not match the status identity.

The one-object development runs verify runtime identity, physical execution and recovery;
they are not classifier-performance evidence and are not reported as such.
