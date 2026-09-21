# P2.1 — One-command reproducibility

## Outcome

The supported judge launch is now a single command from the repository root:

```bash
infra/isaac-sim/factory-demo.sh launch showcase
```

The command has four explicit gates: static preflight, idempotent cache preparation,
canonical stack launch, and post-launch verification. It succeeds only when all three
containers are healthy, the browser endpoint responds, and the live runtime reports the
exact checked-out commit, requested mode and scenario.

## Reproducibility locks

- `requirements-runtime.txt` pins the host perception/evaluation environment captured
  from the accepted RTX workstation. Isaac Sim continues to use its own container
  Python rather than mixing Kit packages into that environment.
- `config/runtime-provenance.json` records the reference OS, Python, GPU/driver,
  digest-pinned Isaac Sim image, checkpoint paths, sizes and SHA-256 hashes.
- `.env.example` and Compose use the same immutable Isaac image digest by default.
- Model mode fails before launch when its Python environment, type checkpoint or
  routing checkpoint is missing or differs from the accepted artifact.

The two checkpoints remain intentionally outside Git: each is approximately 350 MB and
exceeds GitHub's normal file limit. Their hashes prove identity without pretending that
the repository contains them.

## Operational boundaries

The fast preflight does not require an already-running demo. The Isaac documentation MCP
and optional H200/Qwen endpoint are reported only as information because neither executes
the factory. Showcase mode does not require trained checkpoints; model mode does.

`prepare-environment.sh` only creates missing persistent cache directories and never
removes cache or user data. The launch remains safe to repeat because Compose recreates
the named stack while preserving mounted caches.

## Acceptance evidence

The stage was accepted only after shell syntax and whitespace validation, the full Python
test suite, static showcase/model preflight, and a live isolated-checkout showcase launch.
The live gate checked container health, HTTP viewer availability and exact runtime
identity. Final evidence is recorded against the integration commit after squash.
