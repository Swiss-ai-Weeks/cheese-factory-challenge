"""Fail closed unless a classifier candidate passes every frozen validation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(document: dict, dotted_path: str):
    value = document
    for part in dotted_path.split("."):
        value = value[part]
    return value


def check_candidate(spec: dict, reports: dict[str, dict], manifest_hashes: dict[str, str]) -> dict:
    checks = []
    checkpoints = set()
    for suite_name, suite in spec["suites"].items():
        report = reports[suite_name]
        checkpoints.add(report["checkpoint"])
        identity = {
            "manifest": report["manifest"] == suite["manifest"],
            "manifest_sha256": manifest_hashes[suite_name] == suite["manifest_sha256"],
            "split": report["split"] == suite["split"],
            "samples": int(report["samples"]) == int(suite["samples"]),
            "threshold": float(report["threshold"]) == float(spec["threshold"]),
        }
        for name, passed in identity.items():
            checks.append({
                "suite": suite_name,
                "check": name,
                "passed": passed,
            })
        for metric, minimum in suite["minimums"].items():
            actual = float(nested_value(report, metric))
            checks.append({
                "suite": suite_name,
                "check": metric,
                "actual": actual,
                "minimum": float(minimum),
                "passed": actual >= float(minimum),
            })
    checks.append({
        "suite": "cross_suite",
        "check": "same_candidate_checkpoint",
        "actual": sorted(checkpoints),
        "passed": len(checkpoints) == 1,
    })
    return {
        "gate_version": spec["version"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "locked_evaluations": spec["locked_until_validation_passes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "p14_candidate_gate.json")
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--operational", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    reports = {
        "balanced": json.loads(args.balanced.read_text()),
        "operational": json.loads(args.operational.read_text()),
    }
    manifest_hashes = {}
    for suite_name, suite in spec["suites"].items():
        manifest = ROOT / suite["manifest"]
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifest_hashes[suite_name] = sha256(manifest)
    result = check_candidate(spec, reports, manifest_hashes)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
