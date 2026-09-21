"""Run the final branch, repository, test, evidence and live-service release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARENTS = {
    "work/perception": "fe70eab58185881190fc3ddc257f0933159384ed",
    "work/isaac-factory": "0f144ae65c23b13e3b92e0cf29d61ee71680122b",
}
EXPECTED_EVIDENCE = {
    "docs/evidence/p23-live-showcase.mp4": "ddf4239bdd50985f1c5ddb7ae2dfd75001033f56d0895261d720b062b95922a4",
    "docs/evidence/p23-live-showcase-preview.jpg": "52aa3e7194d542d864bf7bd9346d4e655fa189412dd9f77876ee76ba4bf659bd",
}
FORBIDDEN_TRACKED = re.compile(
    r"^(?:data/(?:raw|processed)/|sim/out/|outputs/|\.venv/)|(?:.*\.(?:pt|onnx)(?:\.data)?)$"
)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=check)


def git(*args: str) -> str:
    return run("git", *args).stdout.strip()


def record(checks: list[dict], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise RuntimeError(f"{name}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final-audit.json")
    args = parser.parse_args()
    checks: list[dict] = []

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    record(checks, "integration_branch", branch == "codex/hackathon-integration", branch)
    remote_head = git("ls-remote", "origin", "refs/heads/codex/hackathon-integration").split()[0]
    record(checks, "remote_sha", remote_head == head, {"local": head, "remote": remote_head})

    remote_lines = git("ls-remote", "--heads", "origin").splitlines()
    remote_heads = {line.split()[1].removeprefix("refs/heads/"): line.split()[0] for line in remote_lines}
    for name, expected in EXPECTED_PARENTS.items():
        record(checks, f"source_unchanged:{name}", remote_heads.get(name) == expected, remote_heads.get(name))
        contained = run("git", "merge-base", "--is-ancestor", expected, head, check=False).returncode == 0
        record(checks, f"source_contained:{name}", contained, expected)

    tracked = git("ls-files").splitlines()
    leaked = sorted(path for path in tracked if FORBIDDEN_TRACKED.fullmatch(path))
    record(checks, "no_dataset_or_checkpoint_leak", not leaked, leaked)

    tree = git("ls-tree", "-r", "-l", "HEAD").splitlines()
    blobs = []
    for line in tree:
        metadata, path = line.split("\t", 1)
        size = int(metadata.split()[-1])
        blobs.append((size, path))
    largest = max(blobs)
    record(checks, "github_blob_limit", largest[0] < 100_000_000, {"bytes": largest[0], "path": largest[1]})

    required_docs = ["docs/JUDGE_GUIDE.md", "docs/ARCHITECTURE.md", "docs/MODEL_CARD.md", "docs/evaluation_results.md"]
    missing_docs = [path for path in required_docs if not (ROOT / path).is_file()]
    record(checks, "judge_package", not missing_docs, missing_docs or required_docs)
    for path, expected in EXPECTED_EVIDENCE.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        record(checks, f"evidence_hash:{path}", actual == expected, actual)

    status_path = ROOT / "outputs/factory/runtime-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    identity_ok = (
        status.get("application") == "cheese_factory"
        and status.get("commit") == head
        and status.get("classifier_mode") == "showcase"
        and status.get("phase") in {"loaded", "reset", "running", "complete"}
    )
    record(checks, "live_runtime_identity", identity_ok, status)

    compose = str(ROOT / "infra/isaac-sim/docker-compose.yml")
    service_states = {}
    for service in ("isaac-sim", "web-viewer", "remote-desktop"):
        container = run("docker", "compose", "-p", "isim", "-f", compose, "ps", "-q", service).stdout.strip()
        state = run("docker", "inspect", "--format", "{{.State.Status}}/{{.State.Health.Status}}", container).stdout.strip() if container else "missing"
        service_states[service] = state
    record(checks, "live_service_health", all(value == "running/healthy" for value in service_states.values()), service_states)
    with urllib.request.urlopen("http://127.0.0.1:6080/vnc.html", timeout=10) as response:
        viewer_status = response.status
    record(checks, "viewer_http", viewer_status == 200, viewer_status)

    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "sim", "tests"],
        [sys.executable, "-m", "pytest", "-q"],
        ["bash", "-n", *sorted(str(path.relative_to(ROOT)) for path in (ROOT / "infra/isaac-sim").glob("*.sh"))],
        ["git", "diff", "--check"],
    ]
    for command in commands:
        result = run(*command, check=False)
        record(checks, f"command:{' '.join(command)}", result.returncode == 0, (result.stdout + result.stderr).strip()[-2000:])

    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    record(checks, "clean_worktree", not dirty, dirty)

    report = {
        "schema_version": 1,
        "generated_at_epoch": time.time(),
        "branch": branch,
        "commit": head,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "evidence_boundary": "Simulation and scripted-showcase evidence; not real-world safety or classifier accuracy.",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "commit": head, "checks": len(checks), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
