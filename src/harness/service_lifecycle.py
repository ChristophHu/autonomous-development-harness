"""Fail-closed local service PID identity, readiness, and shutdown helpers."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path


class ServiceLifecycle:
    def __init__(self, pidfile: Path, *, process_runner=None, sleep=time.sleep):
        self.pidfile = Path(pidfile)
        self.process_runner = process_runner or subprocess.run
        self.sleep = sleep

    def process_fingerprint(self, pid: int) -> str | None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        result = self.process_runner(
            ["ps", "-p", str(pid), "-o", "lstart=,command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise RuntimeError("process identity could not be verified")
        return hashlib.sha256(result.stdout.strip().encode()).hexdigest()

    def read_record(self):
        try:
            value = json.loads(self.pidfile.read_text())
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"state": "invalid_record"}
        if (
            not isinstance(value, dict)
            or set(value)
            != {"schema", "pid", "process_fingerprint", "started_at", "host", "port"}
            or value.get("schema") != 1
            or not isinstance(value.get("pid"), int)
            or isinstance(value.get("pid"), bool)
            or value["pid"] < 1
            or not isinstance(value.get("process_fingerprint"), str)
            or len(value["process_fingerprint"]) != 64
            or not isinstance(value.get("started_at"), str)
            or not value["started_at"]
            or value.get("host") not in {"127.0.0.1", "localhost"}
            or not isinstance(value.get("port"), int)
            or isinstance(value.get("port"), bool)
            or not 1 <= value["port"] <= 65535
        ):
            return {"state": "invalid_record"}
        return value

    def inspect(self, *, probe_health=True):
        record = self.read_record()
        if record is None:
            return {"state": "stopped", "record": None, "ready": False}
        if record.get("state") == "invalid_record":
            return {"state": "invalid_record", "record": None, "ready": False}
        fingerprint = self.process_fingerprint(record["pid"])
        if fingerprint is None:
            return {"state": "stale", "record": record, "ready": False}
        if fingerprint != record["process_fingerprint"]:
            return {"state": "pid_reused", "record": record, "ready": False}
        ready = self.health(record["host"], record["port"]) if probe_health else False
        return {
            "state": "running",
            "record": record,
            "ready": ready,
        }

    @staticmethod
    def health(host, port, timeout=0.5):
        request = urllib.request.Request(
            f"http://{host}:{port}/health", headers={"Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(4097)
                if len(body) > 4096:
                    return False
                payload = json.loads(body)
            return (
                response.status == 200
                and isinstance(payload, dict)
                and payload.get("status") == "ok"
                and payload.get("service") == "harness"
            )
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return False

    def register_current(self, host, port, pid=None):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("API host must remain local")
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise ValueError("API port must be between 1 and 65535")
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        lock = self.pidfile.with_suffix(self.pidfile.suffix + ".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("Harness start is already in progress") from exc
        try:
            os.close(descriptor)
            state = self.inspect(probe_health=False)
            if state["state"] == "running":
                raise RuntimeError("Harness already appears to be running")
            if state["state"] == "pid_reused":
                raise RuntimeError("PID record belongs to a different process")
            if state["state"] == "invalid_record":
                raise RuntimeError("Harness PID record is invalid; inspect it manually")
            if state["state"] == "stale":
                self._unlink_if_unchanged(state["record"])
            pid = os.getpid() if pid is None else pid
            fingerprint = self.process_fingerprint(pid)
            if fingerprint is None:
                raise RuntimeError("Harness process is not alive")
            record = {
                "schema": 1,
                "pid": pid,
                "process_fingerprint": fingerprint,
                "started_at": datetime.now(UTC).isoformat(),
                "host": host,
                "port": port,
            }
            temporary = self.pidfile.with_name(
                f".{self.pidfile.name}.{uuid.uuid4().hex}.tmp"
            )
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(record, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.pidfile)
            return record
        finally:
            lock.unlink(missing_ok=True)

    def _unlink_if_unchanged(self, record):
        if self.read_record() == record:
            self.pidfile.unlink(missing_ok=True)

    def remove_record(self, record):
        self._unlink_if_unchanged(record)

    def wait_until_ready(self, host, port, timeout=10.0, *, monotonic=time.monotonic):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self.health(host, port):
                return True
            self.sleep(min(0.1, max(0, deadline - monotonic())))
        return False

    def stop(self, timeout=5.0, *, terminate=None, monotonic=time.monotonic):
        state = self.inspect(probe_health=False)
        if state["state"] in {"stopped", "stale"}:
            if state["state"] == "stale":
                self._unlink_if_unchanged(state["record"])
                return "stale"
            return "not_running"
        if state["state"] != "running":
            raise RuntimeError(f"cannot safely stop Harness: {state['state']}")
        record = state["record"]
        if terminate is None:
            os.kill(record["pid"], signal.SIGTERM)
        else:
            terminate(record["pid"], signal.SIGTERM)
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            fingerprint = self.process_fingerprint(record["pid"])
            if fingerprint is None or fingerprint != record["process_fingerprint"]:
                self._unlink_if_unchanged(record)
                return "stopped"
            self.sleep(min(0.1, max(0, deadline - monotonic())))
        raise TimeoutError("Harness did not stop before the shutdown timeout")
