"""Factory-owned runtime identity and health status.

The file written here is mounted into the Isaac container and is deliberately tied to
the unique runtime ID supplied by ``run-gui.sh``.  A stale status from an earlier
container therefore cannot make a plain Isaac process look like the factory.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class RuntimeStatus:
    """Atomically publish the identity and current phase of one factory run."""

    def __init__(
        self,
        path: Path,
        *,
        runtime_id: str,
        commit: str,
        classifier_mode: str,
        scenario: str,
    ) -> None:
        self.path = path
        self.started_at_epoch = time.time()
        self.identity = {
            "application": "cheese_factory",
            "runtime_id": runtime_id,
            "commit": commit,
            "classifier_mode": classifier_mode,
            "scenario": scenario,
            "pid": os.getpid(),
        }

    @classmethod
    def from_environment(cls, output_root: Path, classifier_mode: str) -> "RuntimeStatus":
        return cls(
            output_root / "runtime-status.json",
            runtime_id=os.environ.get("CHEESE_RUNTIME_ID", "unmanaged"),
            commit=os.environ.get("CHEESE_GIT_COMMIT", "unknown"),
            classifier_mode=classifier_mode,
            scenario=os.environ.get("CHEESE_SCENARIO", "default-evaluation"),
        )

    def update(self, phase: str, **details: Any) -> None:
        now = time.time()
        payload = {
            **self.identity,
            "phase": phase,
            "started_at_epoch": self.started_at_epoch,
            "updated_at_epoch": now,
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)
