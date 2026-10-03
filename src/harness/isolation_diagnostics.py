"""Safe capability probes for nested macOS process isolation."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .isolation import ProcessAccessProfile, isolated_command

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_HOST_DENIALS = {1, 13}  # EPERM and EACCES, intentionally serialized as codes.


def _status_for_result(result, *, marker=None):
    stderr = result.stderr or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    lowered = stderr.casefold()
    if "sandbox_apply" in lowered and "operation not permitted" in lowered:
        return {"status": "blocked_by_host", "reason": "sandbox_apply_denied"}
    if result.returncode == 0 and (marker is None or marker()):
        return {"status": "supported", "reason": None}
    child_error = re.search(
        r"HARNESS_SOCKET_PROBE_ERROR=([A-Za-z][A-Za-z0-9]{0,39})(?::(-?\d+))?",
        stderr,
    )
    if child_error is not None:
        error_type = child_error.group(1)
        error_number = int(child_error.group(2)) if child_error.group(2) else None
        if error_number in {1, 13} or error_type == "PermissionError":
            reason = "child_socket_denied"
        elif error_type in {"TimeoutError", "socket.timeout"}:
            reason = "child_timeout"
        elif error_type in {"ConnectionRefusedError", "OSError"}:
            reason = "child_connection_failed"
        else:
            reason = "child_runtime_error"
        report = {
            "status": "probe_failed",
            "reason": reason,
            "returncode": result.returncode,
            "child_error_type": error_type,
        }
        if error_number is not None:
            report["child_errno"] = error_number
        return report
    started = "HARNESS_SOCKET_PROBE_STARTED" in stderr
    if result.returncode == 65:
        return {
            "status": "probe_failed",
            "reason": "sandbox_profile_rejected",
            "returncode": result.returncode,
        }
    if "operation not permitted" in lowered or "permission denied" in lowered:
        reason = "child_socket_denied"
    elif "connection refused" in lowered or "network is unreachable" in lowered:
        reason = "child_connection_failed"
    elif "sandbox-exec" in lowered and any(
        word in lowered for word in ("syntax", "parse", "profile")
    ):
        reason = "sandbox_profile_rejected"
    elif not started:
        reason = "child_not_started"
    else:
        reason = "child_failed_after_start"
    return {
        "status": "probe_failed",
        "reason": reason,
        "returncode": result.returncode,
    }


def _socket_probe(kind, workspace, runner):
    address = None
    listener = None
    server_thread = None
    accepted = threading.Event()
    try:
        listener = socket.socket(
            socket.AF_INET if kind == "tcp_loopback" else socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        if kind == "tcp_loopback":
            listener.bind(("127.0.0.1", 0))
            address = listener.getsockname()
            source = (
                "import socket, sys\n"
                "print('HARNESS_SOCKET_PROBE_STARTED', file=sys.stderr, flush=True)\n"
                "try:\n"
                " s=socket.create_connection(('127.0.0.1', "
                f"{address[1]}), timeout=2); s.sendall(b'harness-probe'); s.close()\n"
                "except OSError as error:\n"
                " print(f'HARNESS_SOCKET_PROBE_ERROR={type(error).__name__}:{error.errno}', file=sys.stderr)\n"
                " sys.exit(1)"
            )
        else:
            address = workspace / "probe.sock"
            listener.bind(str(address))
            source = (
                "import socket, sys\n"
                "print('HARNESS_SOCKET_PROBE_STARTED', file=sys.stderr, flush=True)\n"
                "try:\n"
                " s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
                f"s.connect({str(address)!r}); s.sendall(b'harness-probe'); s.close()\n"
                "except OSError as error:\n"
                " print(f'HARNESS_SOCKET_PROBE_ERROR={type(error).__name__}:{error.errno}', file=sys.stderr)\n"
                " sys.exit(1)"
            )
        listener.listen(1)
        listener.settimeout(3)

        def accept_one():
            try:
                connection, _peer = listener.accept()
                with connection:
                    accepted.set()
                    connection.recv(32)
            except OSError:
                return

        server_thread = threading.Thread(target=accept_one, daemon=True)
        server_thread.start()
        profile = ProcessAccessProfile.broker(
            "capability_probe",
            read_roots=(Path(sys.prefix),),
            write_roots=(workspace,),
            unix_sockets=(address,) if kind == "unix_socket" else (),
        )
        command = isolated_command(
            [sys.executable, "-c", source],
            workspace,
            access_profile=profile,
            network_proxy=address[1] if kind == "tcp_loopback" else None,
            unix_sockets=(address,) if kind == "unix_socket" else (),
        )
        result = runner(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        report = _status_for_result(result, marker=accepted.is_set)
        if report["status"] == "supported" and not accepted.is_set():
            return {"status": "probe_failed", "reason": "local_connection_not_observed"}
        return report
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout", "reason": "child_timeout"}
    except OSError as error:
        if error.errno in _HOST_DENIALS:
            return {"status": "blocked_by_host", "reason": "local_socket_denied"}
        return {"status": "probe_failed", "reason": "local_socket_unavailable"}
    except (RuntimeError, TypeError, ValueError):
        return {"status": "profile_error", "reason": "profile_invalid"}
    finally:
        if listener is not None:
            listener.close()
        if server_thread is not None:
            server_thread.join(timeout=0.1)
        if kind == "unix_socket" and address is not None:
            Path(address).unlink(missing_ok=True)


def probe_sandbox_capability(
    *, platform_name=None, runner=subprocess.run, socket_probe=None
):
    """Probe process, workspace write, loopback TCP, and Unix-socket capabilities.

    All sockets are bound to loopback or a temporary Unix-socket path and are
    closed before return. No external network endpoint is contacted.
    """
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "darwin":
        return {
            "status": "unsupported_platform",
            "supported": False,
            "reason": "macos_required",
            "checks": {},
        }
    if not _SANDBOX_EXEC.is_file():
        return {
            "status": "sandbox_unavailable",
            "supported": False,
            "reason": "sandbox_exec_missing",
            "checks": {},
        }

    with tempfile.TemporaryDirectory(prefix="harness-cap-") as name:
        workspace = Path(name).resolve(strict=True)
        marker = workspace / ".harness-isolation-probe"
        source = "from pathlib import Path; Path('.harness-isolation-probe').write_text('ok')"
        checks = {}
        profile = ProcessAccessProfile.workspace(
            read_roots=(Path(sys.prefix),), write_roots=(workspace,)
        )
        try:
            command = isolated_command(
                [sys.executable, "-c", source], workspace, access_profile=profile
            )
            result = runner(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            checks["process_exec"] = _status_for_result(
                result, marker=lambda: result.returncode == 0
            )
            checks["workspace_write"] = (
                {"status": "supported", "reason": None}
                if marker.is_file() and marker.read_text(encoding="utf-8") == "ok"
                else {
                    "status": "not_tested",
                    "reason": "process_probe_did_not_complete",
                }
                if checks["process_exec"]["status"]
                in {
                    "blocked_by_host",
                    "probe_timeout",
                    "sandbox_unavailable",
                    "profile_error",
                }
                else {
                    "status": "probe_failed",
                    "reason": "workspace_write_not_observed",
                }
            )
        except subprocess.TimeoutExpired:
            checks["process_exec"] = {
                "status": "probe_timeout",
                "reason": "child_timeout",
            }
            checks["workspace_write"] = {
                "status": "not_tested",
                "reason": "process_probe_did_not_complete",
            }
        except OSError as error:
            checks["process_exec"] = {
                "status": "sandbox_unavailable",
                "reason": type(error).__name__,
            }
            checks["workspace_write"] = {
                "status": "not_tested",
                "reason": "process_probe_did_not_complete",
            }
        except (RuntimeError, TypeError, ValueError):
            checks["process_exec"] = {
                "status": "profile_error",
                "reason": "profile_invalid",
            }
            checks["workspace_write"] = {
                "status": "not_tested",
                "reason": "profile_invalid",
            }

        socket_probe = socket_probe or _socket_probe
        checks["tcp_loopback"] = socket_probe("tcp_loopback", workspace, runner)
        checks["unix_socket"] = socket_probe("unix_socket", workspace, runner)

    blocked = any(item["status"] == "blocked_by_host" for item in checks.values())
    supported = all(item["status"] == "supported" for item in checks.values())
    if supported:
        status, reason = "supported", None
    elif blocked:
        status = "blocked_by_host"
        reason = next(
            (
                item["reason"]
                for item in checks.values()
                if item["status"] == "blocked_by_host"
            ),
            "one_or_more_capabilities_blocked",
        )
    elif checks["process_exec"]["status"] in {
        "profile_error",
        "probe_timeout",
        "sandbox_unavailable",
    }:
        status = checks["process_exec"]["status"]
        reason = checks["process_exec"]["reason"]
    else:
        status, reason = "probe_failed", "one_or_more_capabilities_unavailable"
    return {
        "status": status,
        "supported": supported,
        "reason": reason,
        "checks": checks,
    }
