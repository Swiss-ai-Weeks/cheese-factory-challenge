# Autonomous factory demo runbook

## Prerequisites

1. NVIDIA driver, Docker, NVIDIA Container Toolkit, and access to
   `nvcr.io/nvidia/isaac-sim:6.1.0`.
2. Start the MCP server and restart the coding-agent session after first adding
   it. See [Isaac MCP research](isaac_mcp_research.md).
3. For production, restore or rebuild both ignored checkpoints:
   `runs/sim_type13/best.pt` and `runs/sim_bin_adapt_v2/best.pt`. Both are
   present on the evaluation workstation.

The live runtime has three deliberately separate modes:

| Mode | What it proves | What it does not prove |
|---|---|---|
| `showcase` | camera detection/localization and repeatable physical manipulation | classifier accuracy; routing is scripted from the known scenario |
| `model` | honest trained-perception behavior, including fail-closed rejects | reliable motion on items the model rejects |
| `development` | pixel proxy diagnostics | trained perception or lighting invariance |

## Commands

Use one allowlisted operator command for the live demo:

```bash
infra/isaac-sim/factory-demo.sh preflight
infra/isaac-sim/factory-demo.sh showcase
infra/isaac-sim/factory-demo.sh status
```

`showcase` is the recommended judge-facing experience. The on-screen HUD labels
it `SCRIPTED ROUTING (NOT MODEL ACCURACY)`. Reset it to object 1 at any time:

```bash
infra/isaac-sim/factory-demo.sh replay
```

Start honest trained perception separately:

```bash
infra/isaac-sim/factory-demo.sh model
```

Open the authenticated noVNC endpoint already provisioned for port 6080, or
forward it locally with `ssh -N -L 6080:127.0.0.1:6080 <host>` and open
`http://localhost:6080`.

If the viewer is blank, stale, or showing the wrong app, recreate the complete
stack with its canonical entry point:

```bash
infra/isaac-sim/factory-demo.sh recover showcase
```

For reproducible headless evidence, the lower-level bounded command remains:

```bash
infra/isaac-sim/run-evaluation.sh showcase 11
infra/isaac-sim/run-evaluation.sh model 11
```

The model commands fail closed if either checkpoint is absent or the service
contract is incompatible. Stop the stream and repository-owned sorter with:

```bash
infra/isaac-sim/factory-demo.sh stop
```

For a local agent or operator UI, start the read-only-by-default control gateway:

```bash
.venv/bin/python sim/factory/control_server.py
curl -fsS http://127.0.0.1:8766/status
curl -fsS http://127.0.0.1:8766/capabilities
```

Authenticated reset, mode/scenario selection, bounded evaluation and emergency-stop
examples are in [Safe factory control](stages/21-safe-factory-control.md). The gateway
is a convenience control plane for this simulated cell, not a safety-rated industrial
emergency-stop system.

## Expected outputs

Generated artifacts are ignored by Git and written under `outputs/factory/`:

- `results.json`: per-object lifecycle and separate detection, classification,
  pick, correct-bin, and end-to-end metrics. Successful model records also carry the
  observation sequence, request UUID, frame SHA-256, observation/decision/authorization/
  action timestamps, decision age, authorization age and whether arm action was
  authorized. Perception-fault records keep `arm_action_authorized=false` and identify
  the failed timing check.
- `frames/*.png`: camera frames with crop, state, prediction, bin, confidence,
  status, inference latency and timing/fault context.

The terminal emits one `FACTORY_RESULTS` JSON line on success. Any object whose
status is `empty`, `not_cheese`, or `uncertain` has `pick_attempted=false`.

## Operator checklist

- Run `factory-demo.sh status`; confirm all three services are healthy and the
  runtime commit matches the branch HEAD.
- Confirm the HUD mode before speaking about evidence. Never call showcase or
  development output trained-model accuracy.
- Verify the industrial belt, inspection portal, five labelled cheese receivers,
  red reject receiver and Franka are visible without overlap.
- Watch the HUD state, item, decision, destination, confidence/evidence type and totals.
- During observation/classification, confirm the HUD shows the timing deadline and
  `ARM INHIBITED`. A validated response changes the safety line to authorized; a
  timeout, missing field or correlation mismatch must show a perception fault and must
  not start robot motion.
- Keep the emergency action simple: use the authenticated `/emergency-stop` endpoint
  when the control gateway is enabled, or run `factory-demo.sh stop` directly. The loop
  has motion timeouts and fail-closed rejection, but a hackathon operator should still
  monitor it.
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
- Stream busy, blank, stale, or wrong app: run
  `infra/isaac-sim/factory-demo.sh recover showcase`.
