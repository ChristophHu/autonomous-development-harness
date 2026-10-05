import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import isolation_diagnostics as diagnostics
from harness.process_failures import process_failure_category


def supported_socket_probe(*_args, **_kwargs):
    return {"status": "supported", "reason": None}


def test_probe_reports_non_macos_without_running_child():
    report = diagnostics.probe_sandbox_capability(
        platform_name="linux", runner=lambda *_a, **_k: pytest.fail("must not run")
    )
    assert report == {
        "status": "unsupported_platform",
        "supported": False,
        "reason": "macos_required",
        "checks": {},
    }


def test_probe_reports_missing_sandbox_binary(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/missing/sandbox-exec"))
    report = diagnostics.probe_sandbox_capability(
        platform_name="darwin", runner=lambda *_a, **_k: pytest.fail("must not run")
    )
    assert report["status"] == "sandbox_unavailable"
    assert report["reason"] == "sandbox_exec_missing"


def test_probe_reports_profile_creation_failure(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/usr/bin/true"))
    monkeypatch.setattr(
        diagnostics,
        "isolated_command",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("invalid")),
    )
    assert (
        diagnostics.probe_sandbox_capability(
            platform_name="darwin", socket_probe=supported_socket_probe
        )["status"]
        == "profile_error"
    )


def test_probe_reports_timeout(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/usr/bin/true"))

    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sandbox", 10)

    assert (
        diagnostics.probe_sandbox_capability(
            platform_name="darwin", runner=run, socket_probe=supported_socket_probe
        )["status"]
        == "probe_timeout"
    )


def test_probe_reports_launch_error(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/usr/bin/true"))

    def run(*_args, **_kwargs):
        raise OSError("internal detail")

    report = diagnostics.probe_sandbox_capability(
        platform_name="darwin", runner=run, socket_probe=supported_socket_probe
    )
    assert report["status"] == "sandbox_unavailable"
    assert report["checks"]["process_exec"] == {
        "status": "sandbox_unavailable",
        "reason": "OSError",
    }


def test_probe_classifies_host_policy_denial_without_leaking_stderr(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/usr/bin/true"))
    report = diagnostics.probe_sandbox_capability(
        platform_name="darwin",
        runner=lambda *_a, **_k: SimpleNamespace(
            returncode=71, stderr="sandbox_apply: Operation not permitted SECRET"
        ),
        socket_probe=supported_socket_probe,
    )
    assert report["status"] == "blocked_by_host"
    assert report["reason"] == "sandbox_apply_denied"
    assert report["checks"]["workspace_write"] == {
        "status": "not_tested",
        "reason": "process_probe_did_not_complete",
    }
    assert "SECRET" not in str(report)


def test_probe_classifies_child_socket_denial_without_raw_stderr_leak():
    report = diagnostics._status_for_result(
        SimpleNamespace(
            returncode=1,
            stderr="PermissionError: [Errno 1] Operation not permitted SECRET",
        )
    )
    assert report == {
        "status": "probe_failed",
        "reason": "child_socket_denied",
        "returncode": 1,
    }
    assert "SECRET" not in str(report)


@pytest.mark.parametrize(
    ("error_type", "errno", "reason"),
    [
        ("PermissionError", 1, "child_socket_denied"),
        ("ConnectionRefusedError", 61, "child_connection_failed"),
        ("TimeoutError", None, "child_timeout"),
        ("RuntimeError", None, "child_runtime_error"),
    ],
)
def test_probe_reports_safe_child_exception_details(error_type, errno, reason):
    suffix = f":{errno}" if errno is not None else ":None"
    report = diagnostics._status_for_result(
        SimpleNamespace(
            returncode=1,
            stderr=f"HARNESS_SOCKET_PROBE_ERROR={error_type}{suffix} SECRET",
        )
    )
    assert report["reason"] == reason
    assert report["child_error_type"] == error_type
    if errno is not None:
        assert report["child_errno"] == errno
    assert "SECRET" not in str(report)


@pytest.mark.parametrize(
    ("returncode", "stderr", "reason"),
    [
        (65, "", "sandbox_profile_rejected"),
        (1, "", "child_not_started"),
        (2, "HARNESS_SOCKET_PROBE_STARTED", "child_failed_after_start"),
    ],
)
def test_probe_distinguishes_profile_rejection_and_child_start(
    returncode, stderr, reason
):
    report = diagnostics._status_for_result(
        SimpleNamespace(returncode=returncode, stderr=stderr)
    )
    assert report == {
        "status": "probe_failed",
        "reason": reason,
        "returncode": returncode,
    }


@pytest.mark.parametrize(
    ("stderr", "reason"),
    [
        ("connection refused", "child_connection_failed"),
        ("sandbox-exec profile syntax error", "sandbox_profile_rejected"),
    ],
)
def test_probe_classifies_socket_and_profile_errors(stderr, reason):
    report = diagnostics._status_for_result(
        SimpleNamespace(returncode=1, stderr=stderr)
    )
    assert report["reason"] == reason


def test_probe_requires_child_success_and_workspace_write(monkeypatch):
    monkeypatch.setattr(diagnostics, "_SANDBOX_EXEC", Path("/usr/bin/true"))

    def successful(command, *, cwd, **_kwargs):
        assert command[0] == "/usr/bin/sandbox-exec" or command[0] == "/usr/bin/true"
        (Path(cwd) / ".harness-isolation-probe").write_text("ok")
        return SimpleNamespace(returncode=0, stderr="")

    report = diagnostics.probe_sandbox_capability(
        platform_name="darwin",
        runner=successful,
        socket_probe=supported_socket_probe,
    )
    assert report == {
        "status": "supported",
        "supported": True,
        "reason": None,
        "checks": {
            "process_exec": {"status": "supported", "reason": None},
            "workspace_write": {"status": "supported", "reason": None},
            "tcp_loopback": {"status": "supported", "reason": None},
            "unix_socket": {"status": "supported", "reason": None},
        },
    }

    failed = diagnostics.probe_sandbox_capability(
        platform_name="darwin",
        runner=lambda *_a, **_k: SimpleNamespace(returncode=1, stderr=b"denied"),
        socket_probe=supported_socket_probe,
    )
    assert failed["status"] == "probe_failed"


@pytest.mark.parametrize("kind", ["tcp_loopback", "unix_socket"])
def test_socket_probe_reports_supported_when_local_accept_is_observed(
    tmp_path, monkeypatch, kind
):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def recv(self, _size):
            return b"harness-probe"

    class Listener:
        def __init__(self):
            self.address = None

        def bind(self, address):
            self.address = (
                ("127.0.0.1", 45678) if isinstance(address, tuple) else address
            )

        def getsockname(self):
            return self.address

        def listen(self, _backlog):
            pass

        def settimeout(self, _timeout):
            pass

        def accept(self):
            return Connection(), None

        def close(self):
            pass

    monkeypatch.setattr(diagnostics.socket, "socket", lambda *_args: Listener())
    captured = {}

    def passthrough_command(command, _workspace, **kwargs):
        captured.update(kwargs)
        return command

    monkeypatch.setattr(diagnostics, "isolated_command", passthrough_command)

    def successful(command, **_kwargs):
        compile(command[2], "<socket-probe>", "exec")
        time.sleep(0.01)
        return SimpleNamespace(returncode=0, stderr="")

    report = diagnostics._socket_probe(kind, tmp_path, successful)
    assert report == {"status": "supported", "reason": None}
    assert captured["access_profile"].executable_paths == ()


def test_socket_probe_classifies_host_denial(tmp_path, monkeypatch):
    def denied(*_args):
        raise OSError(1, "denied")

    monkeypatch.setattr(diagnostics.socket, "socket", denied)
    assert diagnostics._socket_probe("tcp_loopback", tmp_path, subprocess.run) == {
        "status": "blocked_by_host",
        "reason": "local_socket_denied",
    }


def test_socket_probe_classifies_timeout_and_missing_accept(tmp_path, monkeypatch):
    class Listener:
        def bind(self, _address):
            pass

        def getsockname(self):
            return ("127.0.0.1", 45678)

        def listen(self, _backlog):
            pass

        def settimeout(self, _timeout):
            pass

        def accept(self):
            raise OSError("closed")

        def close(self):
            pass

    monkeypatch.setattr(diagnostics.socket, "socket", lambda *_args: Listener())
    monkeypatch.setattr(
        diagnostics, "isolated_command", lambda command, *_a, **_k: command
    )
    timed_out = diagnostics._socket_probe(
        "tcp_loopback",
        tmp_path,
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)),
    )
    assert timed_out == {"status": "probe_timeout", "reason": "child_timeout"}

    monkeypatch.setattr(
        diagnostics,
        "_status_for_result",
        lambda _result, **_kwargs: {"status": "supported", "reason": None},
    )
    no_accept = diagnostics._socket_probe(
        "tcp_loopback",
        tmp_path,
        lambda *_a, **_k: SimpleNamespace(returncode=0, stderr=""),
    )
    assert no_accept == {
        "status": "probe_failed",
        "reason": "local_connection_not_observed",
    }


def test_socket_probe_classifies_profile_and_non_host_socket_errors(
    tmp_path, monkeypatch
):
    class Listener:
        def bind(self, _address):
            pass

        def getsockname(self):
            return ("127.0.0.1", 45678)

        def listen(self, _backlog):
            pass

        def settimeout(self, _timeout):
            pass

        def accept(self):
            raise OSError("closed")

        def close(self):
            pass

    monkeypatch.setattr(diagnostics.socket, "socket", lambda *_a: Listener())
    monkeypatch.setattr(
        diagnostics,
        "isolated_command",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("invalid profile")),
    )
    assert diagnostics._socket_probe("tcp_loopback", tmp_path, subprocess.run) == {
        "status": "profile_error",
        "reason": "profile_invalid",
    }

    def unavailable(*_args):
        raise OSError(2, "missing")

    monkeypatch.setattr(diagnostics.socket, "socket", unavailable)
    assert diagnostics._socket_probe("tcp_loopback", tmp_path, subprocess.run) == {
        "status": "probe_failed",
        "reason": "local_socket_unavailable",
    }


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            SimpleNamespace(stderr="sandbox_apply: Operation not permitted"),
            "host_sandbox_blocked",
        ),
        (SimpleNamespace(stderr=b"ordinary child error"), "child_process_failed"),
        (SimpleNamespace(stderr=None), "child_process_failed"),
        (SimpleNamespace(), "child_process_failed"),
    ],
)
def test_process_failure_category_is_bounded_and_safe(result, expected):
    assert process_failure_category(result) == expected


def test_process_failure_category_does_not_include_child_output():
    secret = "sandbox_apply: Operation not permitted private-path-and-secret"
    assert (
        process_failure_category(SimpleNamespace(stderr=secret))
        == "host_sandbox_blocked"
    )
