"""Local, fail-closed monitoring and control gateway for the cheese factory.

Reads are enabled by default. Mutations require a strong operator token, an actor
label, and a unique UUID request ID. The gateway launches only repository-owned
scripts with validated argument vectors; it never accepts commands or paths.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "factory"
RESULTS_PATH = OUTPUT_DIR / "results.json"
RUNTIME_STATUS_PATH = OUTPUT_DIR / "runtime-status.json"
RUN_LOG = OUTPUT_DIR / "control-run.log"
AUDIT_PATH = OUTPUT_DIR / "control-audit.jsonl"
CONTROL_STATE_PATH = OUTPUT_DIR / "control-state.json"
RUN_SCRIPT = PROJECT_ROOT / "infra" / "isaac-sim" / "run-evaluation.sh"
RESET_SCRIPT = PROJECT_ROOT / "infra" / "isaac-sim" / "run-gui.sh"
STOP_SCRIPT = PROJECT_ROOT / "infra" / "isaac-sim" / "stop-gui.sh"
MODES = {"model", "development", "showcase"}
ACTOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}$")
SCENARIO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
TOKEN_MIN_LENGTH = 32


class RequestConflictError(RuntimeError):
    """The mutation conflicts with current state or reuses an old request ID."""


class AuditLog:
    def __init__(self, path: Path = AUDIT_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._accepted_request_ids: set[str] = set()
        self._load_request_ids()

    def _load_request_ids(self) -> None:
        if not self.path.is_file():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("outcome") == "accepted" and isinstance(event.get("request_id"), str):
                    self._accepted_request_ids.add(event["request_id"])
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"control audit is unreadable: {self.path}") from exc

    def has_accepted(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._accepted_request_ids

    def append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            if event.get("outcome") == "accepted":
                self._accepted_request_ids.add(event["request_id"])

    def tail(self, limit: int) -> list[dict]:
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"outcome": "unreadable_audit_record"})
        return events


def _read_json(path: Path, unreadable_message: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": unreadable_message}
    return value if isinstance(value, dict) else {"error": unreadable_message}


def _default_control_state() -> dict:
    return {
        "emergency_stopped": False,
        "selected_mode": None,
        "selected_scenario": None,
        "updated_at_epoch": None,
    }


def _terminate_process_group(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGKILL)


class RunManager:
    def __init__(
        self,
        token: str | None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        terminate_group: Callable[[subprocess.Popen], None] = _terminate_process_group,
        audit: AuditLog | None = None,
        state_path: Path = CONTROL_STATE_PATH,
        clock: Callable[[], float] = time.time,
    ):
        self.token = token or ""
        if self.token and len(self.token) < TOKEN_MIN_LENGTH:
            raise ValueError(f"CHEESE_FACTORY_TOKEN must contain at least {TOKEN_MIN_LENGTH} characters")
        self._popen = popen
        self._run_command = run_command
        self._terminate_group = terminate_group
        self._audit = audit or AuditLog()
        self._state_path = state_path
        self._clock = clock
        self._process: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._operation: str | None = None
        self._mode: str | None = None
        self._scenario: str | None = None
        self._max_objects: int | None = None
        self._active_actor: str | None = None
        self._active_request_id: str | None = None
        self._active_parameters: dict = {}
        self._completion_recorded = True
        self._completion_audit_error: str | None = None
        self._lock = threading.Lock()
        loaded_state = _read_json(state_path, "control state is unreadable")
        if loaded_state and "error" in loaded_state:
            raise RuntimeError(loaded_state["error"])
        self._control_state = _default_control_state()
        if loaded_state:
            self._control_state.update({
                key: loaded_state.get(key)
                for key in self._control_state
                if key in loaded_state
            })

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _refresh_completion_locked(self) -> None:
        if self._process is None or self._completion_recorded:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        outcome = "completed" if return_code == 0 else "failed"
        detail = None if return_code == 0 else f"approved process exited {return_code}"
        try:
            self._audit_event(
                actor=self._active_actor or "unknown",
                request_id=self._active_request_id or "unknown",
                action=self._operation or "unknown",
                parameters=self._active_parameters,
                outcome=outcome,
                detail=detail,
            )
        except RuntimeError as exc:
            self._completion_audit_error = str(exc)
            return
        self._completion_recorded = True
        self._completion_audit_error = None

    def _write_control_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(f".{self._state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self._control_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self._state_path)

    def _validate_identity(self, token: str, actor: str, request_id: str) -> tuple[str, str]:
        if not self.token or not hmac.compare_digest(token.encode(), self.token.encode()):
            raise PermissionError("factory controls are disabled or the token is invalid")
        if not isinstance(actor, str) or not ACTOR_PATTERN.fullmatch(actor):
            raise ValueError("X-Cheese-Actor must be 1-64 safe identifier characters")
        try:
            normalized_request_id = str(uuid.UUID(request_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("X-Request-Id must be a UUID") from exc
        if normalized_request_id != request_id.lower():
            raise ValueError("X-Request-Id must use canonical UUID form")
        if self._audit.has_accepted(normalized_request_id):
            raise RequestConflictError("request ID was already accepted")
        return actor, normalized_request_id

    @staticmethod
    def _validate_mode_scenario(mode: str, scenario: str) -> tuple[str, str]:
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError("mode must be model, development, or showcase")
        if not isinstance(scenario, str) or not SCENARIO_PATTERN.fullmatch(scenario):
            raise ValueError("scenario must be a lowercase safe identifier of at most 48 characters")
        return mode, scenario

    def _audit_event(
        self,
        *,
        actor: str,
        request_id: str,
        action: str,
        parameters: dict,
        outcome: str,
        detail: str | None = None,
    ) -> None:
        event = {
            "timestamp_epoch": self._clock(),
            "request_id": request_id,
            "actor": actor,
            "action": action,
            "parameters": parameters,
            "outcome": outcome,
        }
        if detail:
            event["detail"] = detail
        try:
            self._audit.append(event)
        except OSError as exc:
            raise RuntimeError("control audit is unavailable") from exc

    def status(self) -> dict:
        with self._lock:
            self._refresh_completion_locked()
            running = self._running()
            exit_code = None if self._process is None or running else self._process.poll()
            operation = {
                "running": running,
                "pid": self._process.pid if running else None,
                "last_exit_code": exit_code,
                "operation": self._operation,
                "mode": self._mode,
                "scenario": self._scenario,
                "max_objects": self._max_objects,
                "started_at_epoch": self._started_at,
                "completion_audit_error": self._completion_audit_error,
            }
            control = dict(self._control_state)
            control["controls_enabled"] = bool(self.token)
        return {
            "control": control,
            "operation": operation,
            "runtime": _read_json(RUNTIME_STATUS_PATH, "runtime status is unreadable"),
            "latest_result": _read_json(RESULTS_PATH, "latest result is unreadable"),
        }

    def audit_tail(self, limit: int) -> list[dict]:
        return self._audit.tail(limit)

    def _launch(
        self,
        *,
        action: str,
        actor: str,
        request_id: str,
        mode: str,
        scenario: str,
        command: list[str],
        max_objects: int | None,
    ) -> dict:
        parameters = {"mode": mode, "scenario": scenario}
        if max_objects is not None:
            parameters["max_objects"] = max_objects
        self._audit_event(
            actor=actor,
            request_id=request_id,
            action=action,
            parameters=parameters,
            outcome="accepted",
        )
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CHEESE_SCENARIO"] = scenario
        log = RUN_LOG.open("ab")
        try:
            process = self._popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._audit_event(
                actor=actor,
                request_id=request_id,
                action=action,
                parameters=parameters,
                outcome="failed",
                detail=f"process launch failed: {type(exc).__name__}",
            )
            raise RuntimeError("failed to launch approved factory operation") from exc
        finally:
            log.close()
        self._process = process
        self._started_at = self._clock()
        self._operation = action
        self._mode = mode
        self._scenario = scenario
        self._max_objects = max_objects
        self._active_actor = actor
        self._active_request_id = request_id
        self._active_parameters = parameters
        self._completion_recorded = False
        self._completion_audit_error = None
        self._control_state.update({
            "emergency_stopped": False,
            "selected_mode": mode,
            "selected_scenario": scenario,
            "updated_at_epoch": self._clock(),
        })
        try:
            self._write_control_state()
        except OSError as exc:
            try:
                self._terminate_group(process)
            except (OSError, ProcessLookupError):
                pass
            try:
                self._run_command(
                    [str(STOP_SCRIPT)],
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._process = None
            self._audit_event(
                actor=actor,
                request_id=request_id,
                action=action,
                parameters=parameters,
                outcome="failed",
                detail="control state could not be persisted",
            )
            raise RuntimeError("approved operation was cancelled because control state is unavailable") from exc
        return {"accepted": True, "request_id": request_id, "pid": process.pid}

    def start_evaluation(
        self,
        token: str,
        actor: str,
        request_id: str,
        mode: str,
        scenario: str,
        max_objects: int,
    ) -> dict:
        with self._lock:
            self._refresh_completion_locked()
            actor, request_id = self._validate_identity(token, actor, request_id)
            mode, scenario = self._validate_mode_scenario(mode, scenario)
            if not isinstance(max_objects, int) or isinstance(max_objects, bool) or not 1 <= max_objects <= 11:
                raise ValueError("max_objects must be an integer from 1 through 11")
            if self._control_state["emergency_stopped"]:
                raise RequestConflictError("factory is emergency-stopped; reset it before evaluation")
            if self._running():
                raise RequestConflictError("another factory operation is already running")
            return self._launch(
                action="start_bounded_evaluation",
                actor=actor,
                request_id=request_id,
                mode=mode,
                scenario=scenario,
                command=[str(RUN_SCRIPT), mode, str(max_objects)],
                max_objects=max_objects,
            )

    def reset(
        self,
        token: str,
        actor: str,
        request_id: str,
        mode: str,
        scenario: str,
    ) -> dict:
        with self._lock:
            self._refresh_completion_locked()
            actor, request_id = self._validate_identity(token, actor, request_id)
            mode, scenario = self._validate_mode_scenario(mode, scenario)
            if self._running():
                raise RequestConflictError("another factory operation is already running")
            return self._launch(
                action="reset_factory",
                actor=actor,
                request_id=request_id,
                mode=mode,
                scenario=scenario,
                command=[str(RESET_SCRIPT), mode],
                max_objects=None,
            )

    def emergency_stop(self, token: str, actor: str, request_id: str) -> dict:
        with self._lock:
            self._refresh_completion_locked()
            actor, request_id = self._validate_identity(token, actor, request_id)
            parameters: dict = {}
            persistence_errors = []
            try:
                self._audit_event(
                    actor=actor,
                    request_id=request_id,
                    action="emergency_stop",
                    parameters=parameters,
                    outcome="accepted",
                )
            except RuntimeError as exc:
                # Physical safety takes precedence over availability of the audit disk.
                persistence_errors.append(str(exc))
            self._control_state.update({
                "emergency_stopped": True,
                "updated_at_epoch": self._clock(),
            })
            try:
                self._write_control_state()
            except OSError:
                # Continue to stop the real runtime even if the latch cannot be persisted.
                persistence_errors.append("control state is unavailable")
            if self._running():
                try:
                    self._terminate_group(self._process)
                except ProcessLookupError:
                    pass
                try:
                    self._audit_event(
                        actor=self._active_actor or "unknown",
                        request_id=self._active_request_id or "unknown",
                        action=self._operation or "unknown",
                        parameters=self._active_parameters,
                        outcome="cancelled",
                        detail=f"cancelled by emergency stop {request_id}",
                    )
                    self._completion_recorded = True
                except RuntimeError as exc:
                    persistence_errors.append(str(exc))
            try:
                completed = self._run_command(
                    [str(STOP_SCRIPT)],
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._audit_event(
                    actor=actor,
                    request_id=request_id,
                    action="emergency_stop",
                    parameters=parameters,
                    outcome="failed",
                    detail=f"stop command failed: {type(exc).__name__}",
                )
                raise RuntimeError("emergency stop could not confirm factory shutdown") from exc
            if completed.returncode != 0:
                self._audit_event(
                    actor=actor,
                    request_id=request_id,
                    action="emergency_stop",
                    parameters=parameters,
                    outcome="failed",
                    detail=f"stop command exited {completed.returncode}",
                )
                raise RuntimeError("emergency stop could not confirm factory shutdown")
            self._process = None
            self._operation = "emergency_stop"
            self._mode = None
            self._scenario = None
            self._max_objects = None
            self._started_at = self._clock()
            self._active_actor = actor
            self._active_request_id = request_id
            self._active_parameters = parameters
            self._completion_recorded = True
            self._completion_audit_error = None
            try:
                self._audit_event(
                    actor=actor,
                    request_id=request_id,
                    action="emergency_stop",
                    parameters=parameters,
                    outcome="completed",
                )
            except RuntimeError as exc:
                persistence_errors.append(str(exc))
            if persistence_errors:
                raise RuntimeError(
                    "factory stopped, but control persistence failed: "
                    + "; ".join(sorted(set(persistence_errors)))
                )
            return {"accepted": True, "request_id": request_id, "emergency_stopped": True}


def _parse_exact_payload(payload, required: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != required:
        names = ", ".join(sorted(required)) or "no fields"
        raise ValueError(f"request must contain exactly: {names}")
    return payload


def _bearer_token(value: str) -> str:
    prefix = "Bearer "
    if not value.startswith(prefix) or not value[len(prefix):] or " " in value[len(prefix):]:
        return ""
    return value[len(prefix):]


def handler_for(manager: RunManager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "CheeseFactoryControl/2"
        sys_version = ""

        def _json(self, status: int, payload: dict | list, *, allow: str | None = None) -> None:
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if allow:
                self.send_header("Allow", allow)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/health" and not parsed.query:
                self._json(200, {"ok": True, "service": "cheese-factory-control", "version": 2})
            elif parsed.path == "/status" and not parsed.query:
                self._json(200, manager.status())
            elif parsed.path == "/audit":
                try:
                    query = parse_qs(parsed.query, strict_parsing=bool(parsed.query))
                    if set(query) - {"limit"}:
                        raise ValueError("audit accepts only the limit query parameter")
                    limit = int(query.get("limit", ["20"])[0])
                    if not 1 <= limit <= 100:
                        raise ValueError("audit limit must be from 1 through 100")
                    self._json(200, {"events": manager.audit_tail(limit)})
                except (ValueError, TypeError):
                    self._json(400, {"error": "invalid audit query"})
            elif parsed.path == "/capabilities" and not parsed.query:
                self._json(200, {
                    "read_without_token": ["health", "status", "audit"],
                    "mutations_with_token": [
                        "start_bounded_evaluation",
                        "reset_factory",
                        "emergency_stop",
                    ] if manager.token else [],
                    "mutation_identity": ["X-Cheese-Actor", "X-Request-Id"],
                    "arbitrary_commands": False,
                    "bind_scope": "loopback_only",
                    "official_isaacsim_mcp": {
                        "endpoint": "http://127.0.0.1:9904/mcp",
                        "role": "documentation_and_extension_search_only",
                    },
                })
            else:
                self._json(404, {"error": "unknown route"})

        def _read_payload(self) -> dict:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if size < 2 or size > 4096:
                raise ValueError("request body must be 2 to 4096 bytes")
            try:
                payload = json.loads(self.rfile.read(size))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON request") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_POST(self) -> None:
            try:
                payload = self._read_payload()
                token = _bearer_token(self.headers.get("Authorization", ""))
                actor = self.headers.get("X-Cheese-Actor", "")
                request_id = self.headers.get("X-Request-Id", "")
                if self.path == "/evaluations":
                    _parse_exact_payload(payload, {"mode", "scenario", "max_objects"})
                    result = manager.start_evaluation(
                        token, actor, request_id,
                        payload["mode"], payload["scenario"], payload["max_objects"],
                    )
                elif self.path == "/reset":
                    _parse_exact_payload(payload, {"mode", "scenario"})
                    result = manager.reset(
                        token, actor, request_id, payload["mode"], payload["scenario"],
                    )
                elif self.path == "/emergency-stop":
                    _parse_exact_payload(payload, set())
                    result = manager.emergency_stop(token, actor, request_id)
                else:
                    return self._json(404, {"error": "unknown route"})
                self._json(202, result)
            except PermissionError as exc:
                self._json(403, {"error": str(exc)})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except RequestConflictError as exc:
                self._json(409, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(503, {"error": str(exc)})

        def _method_not_allowed(self) -> None:
            self._json(405, {"error": "method not allowed"}, allow="GET, POST")

        do_DELETE = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_PUT = _method_not_allowed

        def log_message(self, *_args) -> None:
            pass

    return Handler


def _require_loopback(host: str) -> None:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve control host: {host}") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses):
        raise ValueError("factory control must bind only to a loopback address")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    _require_loopback(args.host)
    manager = RunManager(os.environ.get("CHEESE_FACTORY_TOKEN"))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(manager))
    server.daemon_threads = True
    print(
        f"factory control listening on http://{args.host}:{args.port} "
        f"controls_enabled={bool(manager.token)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
