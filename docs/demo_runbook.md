# Autonomous factory demo runbook

## Prerequisites

1. NVIDIA driver, Docker, NVIDIA Container Toolkit, and access to
   `nvcr.io/nvidia/isaac-sim:6.1.0`.
2. Start the MCP server and restart the coding-agent session after first adding
   it. See [Isaac MCP research](isaac_mcp_research.md).
3. For production, put the trained checkpoint at
   `runs/sim_type13/best.pt`. It is intentionally Git-ignored and is not present
   on this workstation.

The `development` classifier is an explicit integration aid: it classifies
rendered crop colors and never reads spawn labels, but it is not a substitute
for the trained 13-class model.

## Commands

Headless one-object smoke test:

```bash
CHEESE_MAX_OBJECTS=1 infra/isaac-sim/run-headless.sh development
```

Deterministic 11-object evaluation (ten cheeses, two per bin, one foreign
object, plus an empty-belt interval):

```bash
infra/isaac-sim/run-headless.sh development
```

Streamed GUI on the existing SSH/Brev noVNC stack:

```bash
infra/isaac-sim/run-gui.sh development
```

Open the authenticated noVNC endpoint already provisioned for port 6080, or
forward it locally with `ssh -N -L 6080:127.0.0.1:6080 <host>` and open
`http://localhost:6080`.

Production commands omit `development`:

```bash
infra/isaac-sim/run-headless.sh
infra/isaac-sim/run-gui.sh
```

They fail closed with the exact missing checkpoint path if the model is absent.

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
- Production command says checkpoint missing: restore or rebuild
  `runs/sim_type13/best.pt`; do not silently switch a public demo to the
  development classifier.
- Stream busy or blank: restart the three compose services as documented in
  `infra/isaac-sim/README.md`.
