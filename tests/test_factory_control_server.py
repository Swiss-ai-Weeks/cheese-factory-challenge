import http.client
import json
import subprocess
import threading
import uuid
from http.server import ThreadingHTTPServer

import pytest

from sim.factory.control_server import (
    RESET_SCRIPT,
    RUN_SCRIPT,
    STOP_SCRIPT,
    AuditLog,
    RequestConflictError,
    RunManager,
    _parse_exact_payload,
    _require_loopback,
    handler_for,
)


TOKEN = "a" * 64


class FakeProcess:
    pid = 1234

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


def make_manager(
    tmp_path, *, token=TOKEN, popen=None, run_command=None, terminate_group=None, audit=None,
):
    return RunManager(
        token,
        popen=popen or (lambda *args, **kwargs: FakeProcess()),
        run_command=run_command or (
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "stopped")
        ),
        terminate_group=terminate_group or (lambda process: None),
        audit=audit or AuditLog(tmp_path / "audit.jsonl"),
        state_path=tmp_path / "state.json",
        clock=lambda: 123.5,
    )


def request_id():
    return str(uuid.uuid4())


def test_controls_are_read_only_without_token(tmp_path):
    manager = make_manager(tmp_path, token=None)

    with pytest.raises(PermissionError):
        manager.start_evaluation("", "operator", request_id(), "model", "smoke", 1)
    assert manager.status()["control"]["controls_enabled"] is False


def test_weak_token_is_rejected_at_startup(tmp_path):
    with pytest.raises(ValueError, match="at least 32"):
        make_manager(tmp_path, token="weak")


def test_evaluation_uses_fixed_script_allowlisted_arguments_and_audit(tmp_path, monkeypatch):
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = make_manager(tmp_path, popen=fake_popen)
    mutation_id = request_id()
    result = manager.start_evaluation(
        TOKEN, "judge-console", mutation_id, "showcase", "judge-showcase", 3,
    )

    assert result == {"accepted": True, "request_id": mutation_id, "pid": 1234}
    assert calls[0][0] == [str(RUN_SCRIPT), "showcase", "3"]
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"]["CHEESE_SCENARIO"] == "judge-showcase"
    assert calls[0][1].get("shell") is None
    event = manager.audit_tail(1)[0]
    assert event == {
        "timestamp_epoch": 123.5,
        "request_id": mutation_id,
        "actor": "judge-console",
        "action": "start_bounded_evaluation",
        "parameters": {
            "mode": "showcase",
            "scenario": "judge-showcase",
            "max_objects": 3,
        },
        "outcome": "accepted",
    }


@pytest.mark.parametrize(
    "mode,scenario,max_objects",
    [
        ("shell", "safe", 1),
        ("model", "../../escape", 1),
        ("model", "UPPER", 1),
        ("model", "safe", 0),
        ("model", "safe", 12),
        ("model", "safe", "1"),
        ("model", "safe", True),
        (["model"], "safe", 1),
    ],
)
def test_evaluation_arguments_are_strictly_bounded(tmp_path, mode, scenario, max_objects):
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError):
        manager.start_evaluation(TOKEN, "operator", request_id(), mode, scenario, max_objects)


@pytest.mark.parametrize("actor", ["", "two words", "a/../../b", "x" * 65])
def test_actor_identity_is_strict(tmp_path, actor):
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError, match="X-Cheese-Actor"):
        manager.reset(TOKEN, actor, request_id(), "showcase", "reset-test")


def test_request_id_is_required_and_cannot_be_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    process = FakeProcess()
    manager = make_manager(tmp_path, popen=lambda *args, **kwargs: process)
    mutation_id = request_id()
    manager.reset(TOKEN, "operator", mutation_id, "model", "first-reset")
    process.returncode = 0

    with pytest.raises(RequestConflictError, match="already accepted"):
        manager.reset(TOKEN, "operator", mutation_id, "model", "second-reset")

    reloaded = make_manager(tmp_path)
    with pytest.raises(RequestConflictError, match="already accepted"):
        reloaded.reset(TOKEN, "operator", mutation_id, "model", "third-reset")


def test_reset_uses_fixed_launcher_and_clears_persisted_stop_latch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = make_manager(tmp_path, popen=lambda command, **kwargs: calls.append(command) or FakeProcess())
    manager._control_state["emergency_stopped"] = True
    manager._write_control_state()

    manager.reset(TOKEN, "operator", request_id(), "development", "lighting-check")

    assert calls == [[str(RESET_SCRIPT), "development"]]
    assert manager.status()["control"] == {
        "emergency_stopped": False,
        "selected_mode": "development",
        "selected_scenario": "lighting-check",
        "updated_at_epoch": 123.5,
        "controls_enabled": True,
    }


def test_emergency_stop_terminates_owned_operation_latches_and_audits(tmp_path, monkeypatch):
    launched = FakeProcess()
    terminated = []
    stop_calls = []
    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")

    def fake_stop(command, **kwargs):
        stop_calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "stopped")

    manager = make_manager(
        tmp_path,
        popen=lambda *args, **kwargs: launched,
        run_command=fake_stop,
        terminate_group=lambda process: terminated.append(process.pid),
    )
    manager.start_evaluation(TOKEN, "operator", request_id(), "model", "active-run", 1)
    stop_request_id = request_id()
    result = manager.emergency_stop(TOKEN, "safety-panel", stop_request_id)

    assert result == {
        "accepted": True,
        "request_id": stop_request_id,
        "emergency_stopped": True,
    }
    assert terminated == [1234]
    assert stop_calls[0][0] == [str(STOP_SCRIPT)]
    assert "shell" not in stop_calls[0][1]
    assert manager.status()["control"]["emergency_stopped"] is True
    stop_events = [
        event for event in manager.audit_tail(5) if event["request_id"] == stop_request_id
    ]
    assert [event["outcome"] for event in stop_events] == ["accepted", "completed"]
    assert manager.audit_tail(2)[0]["outcome"] == "cancelled"


def test_completed_async_operation_is_added_to_audit(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = make_manager(tmp_path, popen=lambda *args, **kwargs: process)
    mutation_id = request_id()
    manager.reset(TOKEN, "operator", mutation_id, "showcase", "completion-check")
    process.returncode = 0

    operation = manager.status()["operation"]

    assert operation["running"] is False
    assert operation["completion_audit_error"] is None
    assert [event["outcome"] for event in manager.audit_tail(2)] == ["accepted", "completed"]


def test_emergency_latch_blocks_evaluation_until_reset(tmp_path, monkeypatch):
    processes = []

    def fake_popen(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = make_manager(tmp_path, popen=fake_popen)
    manager.emergency_stop(TOKEN, "operator", request_id())

    with pytest.raises(RequestConflictError, match="reset"):
        manager.start_evaluation(TOKEN, "operator", request_id(), "model", "blocked", 1)

    manager.reset(TOKEN, "operator", request_id(), "showcase", "safe-reset")
    processes[-1].returncode = 0
    manager.start_evaluation(TOKEN, "operator", request_id(), "showcase", "allowed", 1)


def test_failed_emergency_stop_stays_latched_and_is_audited(tmp_path):
    manager = make_manager(
        tmp_path,
        run_command=lambda command, **kwargs: subprocess.CompletedProcess(command, 7, "failed"),
    )

    with pytest.raises(RuntimeError, match="could not confirm"):
        manager.emergency_stop(TOKEN, "operator", request_id())

    assert manager.status()["control"]["emergency_stopped"] is True
    assert [event["outcome"] for event in manager.audit_tail(2)] == ["accepted", "failed"]


def test_emergency_stop_runs_even_when_audit_storage_fails(tmp_path):
    stop_calls = []

    class FailedAudit:
        def has_accepted(self, _request_id):
            return False

        def append(self, _event):
            raise OSError("disk unavailable")

        def tail(self, _limit):
            return []

    manager = make_manager(
        tmp_path,
        audit=FailedAudit(),
        run_command=lambda command, **kwargs: (
            stop_calls.append(command) or subprocess.CompletedProcess(command, 0, "stopped")
        ),
    )

    with pytest.raises(RuntimeError, match="factory stopped, but control persistence failed"):
        manager.emergency_stop(TOKEN, "operator", request_id())

    assert stop_calls == [[str(STOP_SCRIPT)]]
    assert manager.status()["control"]["emergency_stopped"] is True


def test_launch_is_cancelled_when_control_state_cannot_be_persisted(tmp_path, monkeypatch):
    terminated = []
    stop_calls = []
    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = make_manager(
        tmp_path,
        terminate_group=lambda process: terminated.append(process.pid),
        run_command=lambda command, **kwargs: (
            stop_calls.append(command) or subprocess.CompletedProcess(command, 0, "stopped")
        ),
    )
    monkeypatch.setattr(
        manager,
        "_write_control_state",
        lambda: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        manager.reset(TOKEN, "operator", request_id(), "showcase", "safe-reset")

    assert terminated == [1234]
    assert stop_calls == [[str(STOP_SCRIPT)]]
    assert [event["outcome"] for event in manager.audit_tail(2)] == ["accepted", "failed"]


def test_exact_payload_rejects_unknown_command_fields():
    assert _parse_exact_payload({}, set()) == {}
    with pytest.raises(ValueError):
        _parse_exact_payload({"command": "rm -rf /"}, set())
    with pytest.raises(ValueError):
        _parse_exact_payload({"mode": "model", "path": "/tmp/x"}, {"mode"})


def test_control_host_is_loopback_only():
    _require_loopback("127.0.0.1")
    _require_loopback("localhost")
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback("0.0.0.0")


def test_http_api_is_read_only_by_default_and_rejects_unknown_mutations(tmp_path):
    manager = make_manager(tmp_path, token=None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(manager))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", "/capabilities")
        response = connection.getresponse()
        capabilities = json.loads(response.read())
        assert response.status == 200
        assert capabilities["mutations_with_token"] == []
        assert capabilities["arbitrary_commands"] is False

        body = json.dumps({"mode": "model", "scenario": "smoke", "max_objects": 1})
        connection.request(
            "POST",
            "/evaluations",
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        assert response.status == 403
        response.read()

        malicious = json.dumps({"command": "python evil.py"})
        connection.request(
            "POST",
            "/emergency-stop",
            malicious,
            {"Content-Type": "application/json", "Content-Length": str(len(malicious))},
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
