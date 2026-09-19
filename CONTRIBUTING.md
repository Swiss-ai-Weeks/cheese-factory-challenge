# Contributing

Read these documents before changing the project:

1. [`docs/DEVELOPER_HANDOFF.md`](docs/DEVELOPER_HANDOFF.md)
2. [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)
3. [`docs/OPERATIONS_AND_GIT.md`](docs/OPERATIONS_AND_GIT.md)

Work on a feature branch created from `codex/hackathon-integration`. Preserve
the separation between showcase, development and model evidence. Never present
scripted routing as model accuracy.

Every change requires focused tests, `git diff --check`, and review of the full
diff. Visual or physics changes additionally require a complete runtime run and
human inspection of both the viewport and generated evidence. Dataset and model
changes require group-integrity audits and the frozen promotion gate.

Do not commit datasets, generated captures, output logs or large checkpoints
without an explicit artifact/LFS and licensing policy. Do not force-push shared
branches or discard workstation changes you did not create.
