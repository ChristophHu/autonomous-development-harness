"""PID identity and lifecycle control must fail closed and avoid stale-PID kills."""

import json
from types import SimpleNamespace

import pytest

from harness.service_lifecycle import ServiceLifecycle


def _record(pid=123, fingerprint="a" * 64):
    return {
        "schema": 1,
        "pid": pid,
        "process_fingerprint": fingerprint,
        "started_at": "2026-09-29T10:00:00+00:00",
        "host": "127.0.0.1",
        "port": 8080,
    }


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_fingerprint_binds_start_time_and_command_without_exposing_them(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr("harness.service_lifecycle.os.kill", lambda *_: None)
    lifecycle = ServiceLifecycle(
        tmp_path / "harness.pid",
        process_runner=lambda args, **kwargs: (
            commands.append((args, kwargs))
            or SimpleNamespace(
                returncode=0, stdout="Mon Sep 29 10:00:00 2026 python app"
            )
        ),
    )

    fingerprint = lifecycle.process_fingerprint(123)

    assert len(fingerprint) == 64
    assert commands[0][0] == ["ps", "-p", "123", "-o", "lstart=,command="]
    assert "python app" not in fingerprint


def test_process_fingerprint_distinguishes_dead_and_uninspectable_process(
    tmp_path, monkeypatch
):
    lifecycle = ServiceLifecycle(tmp_path / "harness.pid")
    monkeypatch.setattr(
        "harness.service_lifecycle.os.kill",
        lambda *_: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert lifecycle.process_fingerprint(123) is None
    monkeypatch.setattr(
        "harness.service_lifecycle.os.kill",
        lambda *_: (_ for _ in ()).throw(PermissionError()),
    )
    with pytest.raises(PermissionError):
        lifecycle.process_fingerprint(123)


def test_process_fingerprint_fails_closed_when_ps_cannot_verify(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.service_lifecycle.os.kill", lambda *_: None)
    lifecycle = ServiceLifecycle(
        tmp_path / "harness.pid",
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout=""
        ),
    )
    with pytest.raises(RuntimeError, match="identity"):
        lifecycle.process_fingerprint(123)


def test_read_record_rejects_malformed_json_and_invalid_fields(tmp_path):
    pidfile = tmp_path / "data" / "harness.pid"
    lifecycle = ServiceLifecycle(pidfile)
    assert lifecycle.read_record() is None
    pidfile.parent.mkdir()
    pidfile.write_text("{")
    assert lifecycle.read_record() == {"state": "invalid_record"}
    for value in (
        [],
        _record(pid=True),
        _record(fingerprint="short"),
        _record() | {"host": "0.0.0.0"},
        _record() | {"port": 0},
        _record() | {"unexpected": True},
    ):
        _write(pidfile, value)
        assert lifecycle.read_record() == {"state": "invalid_record"}


def test_inspect_distinguishes_stopped_stale_reused_running_and_invalid(
    tmp_path, monkeypatch
):
    pidfile = tmp_path / "harness.pid"
    lifecycle = ServiceLifecycle(pidfile)
    assert lifecycle.inspect() == {
        "state": "stopped",
        "record": None,
        "ready": False,
    }
    _write(pidfile, _record())
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: None)
    assert lifecycle.inspect()["state"] == "stale"
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "b" * 64)
    assert lifecycle.inspect()["state"] == "pid_reused"
    monkeypatch.setattr(
        lifecycle,
        "process_fingerprint",
        lambda _pid: "a" * 64,
    )
    monkeypatch.setattr(lifecycle, "health", lambda *_args: True)
    assert lifecycle.inspect() == {
        "state": "running",
        "record": _record(),
        "ready": True,
    }
    pidfile.write_text("bad")
    assert lifecycle.inspect()["state"] == "invalid_record"


def test_register_current_is_atomic_and_refuses_running_or_unsafe_records(
    tmp_path, monkeypatch
):
    pidfile = tmp_path / "data" / "harness.pid"
    lifecycle = ServiceLifecycle(pidfile)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "a" * 64)
    record = lifecycle.register_current("localhost", 8090, pid=123)
    assert record["pid"] == 123 and record["port"] == 8090
    assert pidfile.stat().st_mode & 0o777 == 0o600
    with pytest.raises(RuntimeError, match="already appears"):
        lifecycle.register_current("localhost", 8090, pid=123)
    assert not pidfile.with_suffix(".pid.lock").exists()
    with pytest.raises(ValueError, match="host"):
        lifecycle.register_current("0.0.0.0", 8090)
    with pytest.raises(ValueError, match="port"):
        lifecycle.register_current("localhost", 0)


def test_register_current_cleans_stale_record_but_preserves_pid_reuse(
    tmp_path, monkeypatch
):
    pidfile = tmp_path / "harness.pid"
    _write(pidfile, _record())
    lifecycle = ServiceLifecycle(pidfile)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: None)
    lifecycle.process_fingerprint = lambda pid: None if pid == 123 else "c" * 64
    lifecycle.register_current("127.0.0.1", 8080, pid=456)
    assert lifecycle.read_record()["pid"] == 456
    _write(pidfile, _record())
    lifecycle.process_fingerprint = lambda _pid: "b" * 64
    with pytest.raises(RuntimeError, match="different process"):
        lifecycle.register_current("127.0.0.1", 8080, pid=456)
    _write(pidfile, {"schema": 99})
    with pytest.raises(RuntimeError, match="invalid"):
        lifecycle.register_current("127.0.0.1", 8080, pid=456)


def test_register_current_refuses_concurrent_start_lock(tmp_path):
    lifecycle = ServiceLifecycle(tmp_path / "harness.pid")
    lifecycle.pidfile.with_suffix(".pid.lock").touch()
    with pytest.raises(RuntimeError, match="already in progress"):
        lifecycle.register_current("localhost", 8080, pid=123)


def test_register_current_requires_live_pid(tmp_path, monkeypatch):
    lifecycle = ServiceLifecycle(tmp_path / "harness.pid")
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: None)
    with pytest.raises(RuntimeError, match="not alive"):
        lifecycle.register_current("localhost", 8080, pid=123)
    assert not lifecycle.pidfile.exists()


def test_remove_record_preserves_replaced_record_and_ready_wait(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    lifecycle = ServiceLifecycle(pidfile, sleep=lambda _seconds: None)
    original = _record()
    _write(pidfile, original)
    lifecycle.remove_record(original | {"pid": 456})
    assert lifecycle.read_record() == original

    probes = iter([False, True])
    monkeypatch.setattr(lifecycle, "health", lambda *_args: next(probes))
    ticks = iter([0, 0, 0.05, 0.1])
    assert lifecycle.wait_until_ready("localhost", 8080, monotonic=lambda: next(ticks))


def test_wait_until_ready_times_out(tmp_path, monkeypatch):
    lifecycle = ServiceLifecycle(tmp_path / "harness.pid", sleep=lambda _seconds: None)
    monkeypatch.setattr(lifecycle, "health", lambda *_args: False)
    ticks = iter([0, 0, 0.05, 0.1])
    assert not lifecycle.wait_until_ready(
        "localhost", 8080, timeout=0.1, monotonic=lambda: next(ticks)
    )


def test_stop_default_terminator_signals_pid(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    _write(pidfile, _record())
    lifecycle = ServiceLifecycle(pidfile)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "a" * 64)
    signals = []
    monkeypatch.setattr(
        "harness.service_lifecycle.os.kill",
        lambda pid, sig: signals.append((pid, sig)),
    )
    with pytest.raises(TimeoutError):
        lifecycle.stop(timeout=0)
    assert signals == [(123, 15)]


def test_stop_waits_until_shutdown_deadline(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    _write(pidfile, _record())
    lifecycle = ServiceLifecycle(pidfile, sleep=lambda _seconds: None)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "a" * 64)
    ticks = iter([0, 0, 0.05, 0.1])
    with pytest.raises(TimeoutError):
        lifecycle.stop(
            timeout=0.1,
            terminate=lambda *_args: None,
            monotonic=lambda: next(ticks),
        )


def test_stop_only_signals_matching_process_and_waits_for_exit(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    _write(pidfile, _record())
    lifecycle = ServiceLifecycle(pidfile, sleep=lambda _seconds: None)
    fingerprints = iter(["a" * 64, None])
    monkeypatch.setattr(
        lifecycle,
        "process_fingerprint",
        lambda _pid: next(fingerprints),
    )
    signals = []

    assert (
        lifecycle.stop(terminate=lambda pid, sig: signals.append((pid, sig)))
        == "stopped"
    )
    assert signals == [(123, 15)]
    assert not pidfile.exists()


def test_stop_refuses_reused_pid_and_keeps_record(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    _write(pidfile, _record())
    lifecycle = ServiceLifecycle(pidfile)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "b" * 64)
    with pytest.raises(RuntimeError, match="pid_reused"):
        lifecycle.stop()
    assert pidfile.exists()


def test_stop_cleans_dead_pid_and_keeps_timed_out_process(tmp_path, monkeypatch):
    pidfile = tmp_path / "harness.pid"
    lifecycle = ServiceLifecycle(pidfile)
    assert lifecycle.stop() == "not_running"
    _write(pidfile, _record())
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: None)
    assert lifecycle.stop() == "stale"
    assert not pidfile.exists()
    _write(pidfile, _record())
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: "a" * 64)
    with pytest.raises(TimeoutError, match="shutdown timeout"):
        lifecycle.stop(timeout=0, terminate=lambda *_: None)
    assert pidfile.exists()


def test_health_probe_accepts_only_harness_health_payload(tmp_path, monkeypatch):
    lifecycle = ServiceLifecycle(tmp_path / "harness.pid")

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(self.payload).encode()[:_size]

    monkeypatch.setattr(
        "harness.service_lifecycle.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response({"status": "ok", "service": "harness"}),
    )
    assert lifecycle.health("localhost", 8080)
    monkeypatch.setattr(
        "harness.service_lifecycle.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response({"status": "ok", "service": "other"}),
    )
    assert not lifecycle.health("localhost", 8080)
    monkeypatch.setattr(
        "harness.service_lifecycle.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response({"payload": "x" * 5000}),
    )
    assert not lifecycle.health("localhost", 8080)
    monkeypatch.setattr(
        "harness.service_lifecycle.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    assert not lifecycle.health("localhost", 8080)
