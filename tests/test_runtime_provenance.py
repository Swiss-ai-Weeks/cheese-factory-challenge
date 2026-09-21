import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "runtime-provenance.json"


def test_runtime_lock_is_well_formed_and_immutable_image_is_used():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    image = payload["isaac_sim"]["image"]
    assert image.startswith("nvcr.io/nvidia/isaac-sim:6.1.0@sha256:")
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", image)

    compose = (ROOT / "infra" / "isaac-sim" / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (ROOT / "infra" / "isaac-sim" / ".env.example").read_text(encoding="utf-8")
    assert image in compose
    assert image in env_example


def test_model_provenance_paths_and_digests_are_safe():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    assert set(payload["models"]) == {"type_classifier", "routing_classifier"}
    for entry in payload["models"].values():
        path = Path(entry["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert entry["bytes"] > 100_000_000


def test_runtime_requirements_are_exactly_pinned():
    lines = [
        line.strip()
        for line in (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(lines) == len(set(line.lower() for line in lines))
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", line) for line in lines)
