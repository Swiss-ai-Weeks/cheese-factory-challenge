# Stage 21 — safe factory control

P1.6 upgrades the local gateway from a single run trigger into a narrow factory control
plane. Reads remain open on loopback; writes require all three identity elements:

- `Authorization: Bearer <CHEESE_FACTORY_TOKEN>` with at least 32 characters;
- `X-Cheese-Actor`, a validated human/agent label;
- `X-Request-Id`, a canonical UUID that has never been accepted before.

The token is supplied only through the gateway process environment. It is never put in
the repository, command line, status response or audit. The actor label makes a request
attributable within this demo, but a shared token does not prove a person's identity.

The server enforces loopback-only binding even when `--host` is supplied. It accepts no
command, path or shell field. Approved subprocesses use fixed repository scripts,
validated argument vectors and `shell=False` semantics.

## Start the gateway

Read-only is the default:

```bash
.venv/bin/python sim/factory/control_server.py
```

Enable the three approved mutations from a private operator shell:

```bash
export CHEESE_FACTORY_TOKEN="$(openssl rand -hex 32)"
.venv/bin/python sim/factory/control_server.py
```

Use a fresh UUID for every mutation:

```bash
export ACTOR="judge-console"
export REQUEST_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
```

## API

The read endpoints require no token:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | gateway liveness and API version |
| `GET` | `/capabilities` | exact read/write surface and MCP boundary |
| `GET` | `/status` | control latch, active operation, Isaac runtime and latest result |
| `GET` | `/audit?limit=20` | last 1–100 mutation events |

All mutation requests need the token, actor and request-ID headers.

Start a bounded evaluation (one concurrent operation maximum):

```bash
curl -fsS -X POST http://127.0.0.1:8766/evaluations \
  -H "Authorization: Bearer $CHEESE_FACTORY_TOKEN" \
  -H "X-Cheese-Actor: $ACTOR" \
  -H "X-Request-Id: $REQUEST_ID" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"model","scenario":"agent-evaluation","max_objects":1}'
```

Reset/recreate the canonical stream in an allowlisted mode and safe scenario label:

```bash
export REQUEST_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
curl -fsS -X POST http://127.0.0.1:8766/reset \
  -H "Authorization: Bearer $CHEESE_FACTORY_TOKEN" \
  -H "X-Cheese-Actor: $ACTOR" \
  -H "X-Request-Id: $REQUEST_ID" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"showcase","scenario":"judge-reset"}'
```

Stop the stream, active factory operation and repository-owned sorter:

```bash
export REQUEST_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
curl -fsS -X POST http://127.0.0.1:8766/emergency-stop \
  -H "Authorization: Bearer $CHEESE_FACTORY_TOKEN" \
  -H "X-Cheese-Actor: $ACTOR" \
  -H "X-Request-Id: $REQUEST_ID" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Emergency stop persists a latch in ignored runtime state. New evaluations return HTTP
409 until an authenticated `/reset` launches an allowlisted mode/scenario and clears the
latch. A repeated request UUID also returns 409. Unknown routes, methods, extra fields,
unsafe scenario strings and out-of-range object counts are rejected.

## Audit and failure behavior

`outputs/factory/control-audit.jsonl` is append-only from the gateway's perspective.
Every accepted mutation records epoch time, request UUID, actor, action and sanitized
parameters. Asynchronous actions add `completed`, `failed` or
`cancelled` records. Emergency stop records both acceptance and confirmed completion.
The token is never logged. `outputs/factory/control-state.json` persists the stop latch
and selected mode/scenario across gateway restarts.

An audit/state disk failure prevents ordinary mutations. Emergency stop is the exception:
it still attempts to stop the physical simulation, then reports that persistence failed.
This prioritizes motion safety over telemetry availability.

The files are local operational evidence, not a tamper-proof security ledger. A user
with workstation filesystem access can edit them. Likewise, this simulated emergency
stop is not a certified hardware E-stop or PLC safety function.

## Acceptance evidence

The isolated feature was validated on the RTX PRO 6000 workstation:

- 100 tests passed, including 26 focused control-server tests;
- weak/missing tokens, malformed identities, reused UUIDs, unsafe modes/scenarios,
  arbitrary fields, non-loopback binds and wrong HTTP methods fail closed;
- an authenticated live stop returned HTTP 202 and latched the factory;
- an evaluation attempted while latched returned HTTP 409;
- an authenticated reset relaunched `showcase` with scenario `p16-live-reset`; runtime
  status reported the feature commit and Docker reported Isaac Sim healthy;
- a second authenticated stop succeeded, and the audit preserved the actor, request UUID,
  accepted/completed outcomes and sanitized parameters;
- Python compilation and diff checks passed.

The live gate exercised the simulated workstation only. It does not establish network
multi-user authorization, cryptographic per-person identity, tamper-evident logging or
real industrial safety certification.
