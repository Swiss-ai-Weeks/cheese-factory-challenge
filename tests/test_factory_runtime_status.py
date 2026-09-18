import json

from sim.factory.runtime_status import RuntimeStatus


def test_runtime_status_is_atomic_and_preserves_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEESE_RUNTIME_ID", "abc-123")
    monkeypatch.setenv("CHEESE_GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("CHEESE_SCENARIO", "judge-showcase")

    status = RuntimeStatus.from_environment(tmp_path, "development")
    status.update("running", completed_objects=3, total_objects=11)

    payload = json.loads((tmp_path / "runtime-status.json").read_text())
    assert payload["application"] == "cheese_factory"
    assert payload["runtime_id"] == "abc-123"
    assert payload["commit"] == "deadbeef"
    assert payload["classifier_mode"] == "development"
    assert payload["scenario"] == "judge-showcase"
    assert payload["phase"] == "running"
    assert payload["completed_objects"] == 3
    assert payload["updated_at_epoch"] >= payload["started_at_epoch"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_runtime_status_uses_explicit_unmanaged_defaults(tmp_path, monkeypatch):
    for name in ("CHEESE_RUNTIME_ID", "CHEESE_GIT_COMMIT", "CHEESE_SCENARIO"):
        monkeypatch.delenv(name, raising=False)

    RuntimeStatus.from_environment(tmp_path, "model").update("loading")
    payload = json.loads((tmp_path / "runtime-status.json").read_text())
    assert payload["runtime_id"] == "unmanaged"
    assert payload["commit"] == "unknown"
    assert payload["scenario"] == "default-evaluation"
