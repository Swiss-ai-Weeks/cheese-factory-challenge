# Autonomous factory demo runbook

## Prerequisites

1. NVIDIA driver, Docker, NVIDIA Container Toolkit, and access to
   `nvcr.io/nvidia/isaac-sim:6.1.0`.
2. Start the MCP server and restart the coding-agent session after first adding
   it. See [Isaac MCP research](isaac_mcp_research.md).
3. For production, restore or rebuild both ignored checkpoints:
   `runs/sim_type13/best.pt` and `runs/sim_bin_adapt_v2/best.pt`. Both are
   present on the evaluation workstation.

The `development` classifier is an explicit integration aid: it classifies
rendered crop colors and never reads spawn labels, but it is not a substitute
for the trained 13-class model.

## Commands

Headless one-object smoke test:

```bash
infra/isaac-sim/run-evaluation.sh development 1
```

Deterministic 11-object evaluation (ten cheeses, two per bin, one foreign
object, plus an empty-belt interval):

```bash
infra/isaac-sim/run-evaluation.sh development 11
```

Streamed GUI on the existing SSH/Brev noVNC stack:

```bash
infra/isaac-sim/run-gui.sh development
```

Open the authenticated noVNC endpoint already provisioned for port 6080, or
forward it locally with `ssh -N -L 6080:127.0.0.1:6080 <host>` and open
`http://localhost:6080`.

Production commands use the trained hybrid service:

```bash
infra/isaac-sim/run-evaluation.sh model 11
infra/isaac-sim/run-gui.sh model
```

`run-evaluation.sh` stops and restores the persistent WebRTC service around a
disposable run. `run-gui.sh model` starts the host perception service if needed.
Both fail closed if either checkpoint is absent or the service contract is
incompatible. Stop the streamed demo and its repository-owned sorter with:

```bash
infra/isaac-sim/stop-gui.sh
```

## Expected outputs

Generated artifacts are ignored by Git and written under `outputs/factory/`:

- `results.json`: per-object lifecycle and separate detection, classification,
  pick, correct-bin, and end-to-end metrics.
- `frames/*.png`: camera frames with crop, state, prediction, bin, confidence,
  status, and inference latency.

The terminal emits one `FACTORY_RESULTS` JSON line on success. Any object whose
status is `empty`, `not_cheese`, or `uncertain` has `pick_attempted=false`.

## Operator checklist

- Confirm all three `isim` services are healthy and the camera view is visible.
- Confirm the terminal says which classifier is active; do not present the
  development classifier as model accuracy.
- Verify belt, pick line, camera framing, five colored bins, reject chute, and
  Franka reach before starting the full run.
- Watch the displayed state, object ID, predicted type/bin, confidence, and
  cumulative result.
- Keep the emergency action simple: stop the Isaac container. The loop has
  motion timeouts and fail-closed rejection, but a hackathon operator should
  still monitor it.
- After the run, inspect at least the first, one middle, foreign-object, and last
  annotated frame plus `results.json`.

## Troubleshooting

- `ModuleNotFoundError: isaacsim.robot_motion.examples`: the runner enables the
  extension after `SimulationApp` starts. Do not import it earlier.
- No camera frame: allow the configured 40 warm-up frames and verify the RTX
  extension started.
- Robot stage load appears frozen on the first run: the Franka USD is fetched
  from NVIDIA's Isaac 6.1 asset service. Verify outbound access and reuse the
  mounted Hub/cache paths.
- Production command says checkpoint missing: restore or rebuild both Stage 5
  checkpoints; do not silently switch a public demo to the development classifier.
- Stream busy or blank: restart the three compose services as documented in
  `infra/isaac-sim/README.md`.
