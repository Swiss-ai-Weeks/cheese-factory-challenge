# Stage 6 — agent-ready factory control

The workstation exposes two deliberately separate services:

1. NVIDIA's official `isaacsim_mcp` at `http://127.0.0.1:9904/mcp` searches
   Isaac Sim instructions, extensions, examples, and settings. It is a
   documentation service and does not control the live simulator.
2. This repository's localhost gateway at `http://127.0.0.1:8766` reports
   factory health, current run state, and the latest machine-readable result.
   A bounded evaluation start is available only when an operator supplies a
   token through the process environment.

This boundary prevents an agent from turning documentation-search output into
arbitrary code execution while still making the WebRTC factory observable and
automatable.

## Start in read-only mode

```bash
.venv/bin/python sim/factory/control_server.py
curl -fsS http://127.0.0.1:8766/capabilities
curl -fsS http://127.0.0.1:8766/status
```

## Enable the bounded start action

Generate a token in the shell that launches the gateway. Do not write it to the
repository or command-line arguments:

```bash
export CHEESE_FACTORY_TOKEN="$(openssl rand -hex 32)"
.venv/bin/python sim/factory/control_server.py
```

From another local shell:

```bash
curl -fsS -X POST http://127.0.0.1:8766/run \
  -H "Authorization: Bearer $CHEESE_FACTORY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"classifier":"model","max_objects":11}'
```

The gateway accepts only `model` or `development` and an integer from 1 to 11.
It executes a fixed argument vector without a shell, allows one run at a time,
stops the persistent Isaac service for the disposable evaluation, and restores
the WebRTC service afterward. It intentionally provides no arbitrary Python,
USD mutation, filesystem, process-stop, or shell endpoint.
