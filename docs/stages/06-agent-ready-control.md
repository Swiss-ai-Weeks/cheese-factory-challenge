# Stage 6 — agent-ready factory control

The workstation exposes two deliberately separate services:

1. NVIDIA's official `isaacsim_mcp` at `http://127.0.0.1:9904/mcp` searches
   Isaac Sim instructions, extensions, examples and settings. It is a
   documentation service and does not control the live simulator.
2. This repository's gateway at `http://127.0.0.1:8766` reports factory health,
   runtime status, latest results and the mutation audit. Its write surface is
   disabled unless the operator supplies a strong token.

This boundary prevents documentation-search output from becoming arbitrary simulator
or shell execution. The gateway itself refuses non-loopback binds and never accepts a
command, executable, filesystem path, Python fragment or USD mutation.

## Read-only mode

```bash
.venv/bin/python sim/factory/control_server.py
curl -fsS http://127.0.0.1:8766/health
curl -fsS http://127.0.0.1:8766/capabilities
curl -fsS http://127.0.0.1:8766/status
curl -fsS 'http://127.0.0.1:8766/audit?limit=20'
```

Without `CHEESE_FACTORY_TOKEN`, every mutation returns HTTP 403. Detailed authenticated
examples and emergency-stop/reset semantics are documented in
[Stage 21](21-safe-factory-control.md).
