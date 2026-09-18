"""Local, fail-closed monitoring and evaluation gateway for the factory.

Read endpoints are enabled by default. Starting a bounded evaluation requires
``CHEESE_FACTORY_TOKEN`` and never accepts a command, path, or shell fragment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "outputs" / "factory" / "results.json"
RUN_LOG = PROJECT_ROOT / "outputs" / "factory" / "control-run.log"
RUN_SCRIPT = PROJECT_ROOT / "infra" / "isaac-sim" / "run-evaluation.sh"
CLASSIFIERS = {"model", "development"}


class RunManager:
    def __init__(
        self,
        token: str | None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.token = token or ""
        self._popen = popen
        self._process: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._classifier: str | None = None
        self._max_objects: int | None = None
        self._lock = threading.Lock()

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> dict:
        with self._lock:
            running = self._running()
            exit_code = None if self._process is None or running else self._process.poll()
            run = {
                "running": running,
                "pid": self._process.pid if running else None,
                "last_exit_code": exit_code,
                "classifier": self._classifier if running else None,
                "max_objects": self._max_objects if running else None,
                "started_at_unix": self._started_at if running else None,
                "controls_enabled": bool(self.token),
            }
        latest = None
        if RESULTS_PATH.is_file():
            try:
                latest = json.loads(RESULTS_PATH.read_text())
            except (OSError, json.JSONDecodeError):
                latest = {"error": "latest result is unreadable"}
        return {"run": run, "latest_result": latest}

    def start(self, token: str, classifier: str, max_objects: int) -> dict:
        if not self.token or not _constant_time_equal(token, self.token):
            raise PermissionError("factory controls are disabled or the token is invalid")
        if classifier not in CLASSIFIERS:
            raise ValueError("classifier must be model or development")
        if not isinstance(max_objects, int) or isinstance(max_objects, bool) or not 1 <= max_objects <= 11:
            raise ValueError("max_objects must be an integer from 1 through 11")
        with self._lock:
            if self._running():
                raise RuntimeError("an evaluation is already running")
            RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
            log = RUN_LOG.open("wb")
            try:
                try:
                    process = self._popen(
                        [str(RUN_SCRIPT), classifier, str(max_objects)],
                        cwd=PROJECT_ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as exc:
                    raise RuntimeError(f"failed to start evaluation: {exc}") from exc
            finally:
                log.close()
            self._process = process
            self._started_at = time.time()
            self._classifier = classifier
            self._max_objects = max_objects
            return {"accepted": True, "pid": process.pid}


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.encode(), right.encode())


def _parse_run_request(payload) -> tuple[str, int]:
    required = {"classifier", "max_objects"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("request must contain exactly classifier and max_objects")
    return payload["classifier"], payload["max_objects"]


def handler_for(manager: RunManager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True, "service": "cheese-factory-control", "version": 1})
            elif self.path == "/status":
                self._json(200, manager.status())
            elif self.path == "/capabilities":
                self._json(200, {
                    "read": ["health", "status", "latest_result"],
                    "control": ["start_bounded_evaluation"] if manager.token else [],
                    "official_isaacsim_mcp": {
                        "endpoint": "http://127.0.0.1:9904/mcp",
                        "role": "documentation_and_extension_search_only",
                    },
                })
            else:
                self._json(404, {"error": "unknown route"})

        def do_POST(self) -> None:
            if self.path != "/run":
                return self._json(404, {"error": "unknown route"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 2 or size > 4096:
                    raise ValueError("request body must be 2 to 4096 bytes")
                body = json.loads(self.rfile.read(size))
                classifier, max_objects = _parse_run_request(body)
                result = manager.start(
                    self.headers.get("Authorization", "").removeprefix("Bearer "),
                    classifier,
                    max_objects,
                )
                self._json(202, result)
            except PermissionError as exc:
                self._json(403, {"error": str(exc)})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except (json.JSONDecodeError, TypeError):
                self._json(400, {"error": "invalid JSON request"})

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    manager = RunManager(os.environ.get("CHEESE_FACTORY_TOKEN"))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(manager))
    print(
        f"factory control listening on http://{args.host}:{args.port} "
        f"controls_enabled={bool(manager.token)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
