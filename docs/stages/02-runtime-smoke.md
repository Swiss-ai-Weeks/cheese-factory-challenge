# Stage 2 — Isaac Sim 6.1 runtime smoke

## Scope

This stage aligns active launch examples and operator documentation with the
installed Isaac Sim 6.1 container while retaining an explicit
`ISAAC_SIM_IMAGE` override for historical reproduction. It also adds a
non-destructive environment check.

No sorting-line geometry, perception model, robot policy, reward, or controller
logic changed in this stage.

## Environment check

Command:

```bash
infra/isaac-sim/check-environment.sh
```

Observed on the RTX PRO 6000 workstation:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- NVIDIA driver: 595.91.07.
- Docker daemon reachable.
- Isaac Sim, web viewer, remote desktop and Isaac documentation MCP healthy.
- H200 Qwen endpoint healthy through the private tunnel.
- Project Python: 3.12.14.
- Approximately 388 GB filesystem space free.
- `runs/sim_type13/best.pt`, `data/` and `sim/out/` remain absent and are
  reported as warnings, not silently substituted.

## One-object smoke trial

Exact command:

```bash
CHEESE_MAX_OBJECTS=1 infra/isaac-sim/run-headless.sh development
```

The streaming stack was stopped for the isolated run and restored afterward.
Isaac reported `Isaac Sim Full Streaming Version: 6.1.0-rc.26`. The command
exited successfully.

Observed machine-readable result:

| Metric | Result |
|---|---:|
| objects | 1 |
| empty intervals | 1 |
| detection success | 1.0 |
| classification accuracy | 1.0 |
| physical pick success | 1.0 |
| correct-bin rate | 1.0 |
| end-to-end success | 1.0 |
| mean cycle time | 16.1788 s |

The hard-cheese proxy was localized from camera RGB/depth, classified to
`bin_hard`, picked, and placed. The empty interval caused no pick.

## Evidence boundary

This run used `classifier_mode=development` and
`development_classifier=true`. It validates the Isaac camera, localization,
state machine, robot controller, placement and metrics pipeline. It is **not**
evidence of trained cheese-recognition accuracy.

Generated evidence remains ignored under:

- `outputs/factory/results.json`
- `outputs/factory/frames/object-000.png`
- `outputs/factory/frames/empty-interval-000.png`

The trained `sim_type13` checkpoint is recovered or reproduced in a later
stage before the final real-perception demo.
