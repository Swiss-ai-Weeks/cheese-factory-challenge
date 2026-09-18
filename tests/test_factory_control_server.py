import pytest

from sim.factory.control_server import RunManager, _parse_run_request


class FakeProcess:
    pid = 1234

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


def test_controls_are_disabled_without_token():
    manager = RunManager(None, popen=lambda *args, **kwargs: FakeProcess())

    with pytest.raises(PermissionError):
        manager.start("", "model", 1)
    assert manager.status()["run"]["controls_enabled"] is False


def test_start_uses_fixed_script_and_allowlisted_arguments(tmp_path, monkeypatch):
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("sim.factory.control_server.RUN_LOG", tmp_path / "run.log")
    manager = RunManager("secret", popen=fake_popen)
    result = manager.start("secret", "development", 3)

    assert result == {"accepted": True, "pid": 1234}
    assert calls[0][0][-2:] == ["development", "3"]
    assert calls[0][1]["start_new_session"] is True
    with pytest.raises(RuntimeError):
        manager.start("secret", "model", 1)


@pytest.mark.parametrize(
    "classifier,max_objects",
    [("shell", 1), ("model", 0), ("model", 12), ("model", "1"), ("model", True)],
)
def test_control_arguments_are_strictly_bounded(classifier, max_objects):
    manager = RunManager("secret", popen=lambda *args, **kwargs: FakeProcess())

    with pytest.raises(ValueError):
        manager.start("secret", classifier, max_objects)


def test_run_request_requires_both_fields_and_rejects_extras():
    assert _parse_run_request({"classifier": "model", "max_objects": 1}) == ("model", 1)
    with pytest.raises(ValueError):
        _parse_run_request({})
    with pytest.raises(ValueError):
        _parse_run_request({"classifier": "model", "max_objects": 1, "command": "oops"})
