from pathlib import Path
from unittest.mock import Mock

import pytest
from jsonschema import ValidationError

from harness.mcp_servers.apple_shell import server


@pytest.fixture
def workspace(tmp_path):
    return server.AppleShellServer(tmp_path)


def test_fixed_diagnostics_use_exact_argv_and_return_bounded_text(
    workspace, monkeypatch
):
    called = []

    def run(command, **kwargs):
        called.append((command, kwargs))
        return Mock(returncode=0, stdout=b"macOS\n")

    monkeypatch.setattr(server.subprocess, "run", run)
    assert workspace.call("diagnostic", {"operation": "software_version"}) == {
        "operation": "software_version",
        "output": "macOS\n",
    }
    assert called[0][0] == ["/usr/bin/sw_vers"]
    assert "shell" not in called[0][1]


def test_all_allowlisted_operations_are_fixed(workspace, monkeypatch):
    commands = []
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command) or Mock(returncode=0, stdout=b"ok")
        ),
    )
    for operation in server.OPERATIONS:
        workspace.call("diagnostic", {"operation": operation})
    assert commands == [
        ["/usr/bin/sw_vers"],
        ["/usr/bin/uname", "-a"],
        ["/bin/date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
    ]


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("exec", {"command": "id"}),
        ("diagnostic", {"operation": "id"}),
        ("diagnostic", {"operation": "utc_time", "argv": ["x"]}),
    ],
)
def test_rejects_arbitrary_commands_and_extra_arguments(
    workspace, name, arguments, monkeypatch
):
    monkeypatch.setattr(server.subprocess, "run", Mock(side_effect=AssertionError))
    with pytest.raises((ValueError, ValidationError)):
        workspace.call(name, arguments)


def test_nonzero_command_is_failure(workspace, monkeypatch):
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: Mock(returncode=1, stdout=b"")
    )
    with pytest.raises(ValueError, match="failed"):
        workspace.call("diagnostic", {"operation": "kernel_info"})


def test_output_is_capped_and_invalid_utf8_replaced(workspace, monkeypatch):
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *a, **k: Mock(
            returncode=0, stdout=b"\xff" + b"x" * (server.MAX_OUTPUT + 10)
        ),
    )
    output = workspace.call("diagnostic", {"operation": "utc_time"})["output"]
    assert len(output) == server.MAX_OUTPUT
    assert output.startswith("\ufffd")


def test_timeout_propagates(workspace, monkeypatch):
    monkeypatch.setattr(
        server.subprocess,
        "run",
        Mock(side_effect=server.subprocess.TimeoutExpired("date", 5)),
    )
    with pytest.raises(ValueError, match="timed out"):
        workspace.call("diagnostic", {"operation": "utc_time"})


def test_response_uses_shared_mcp_protocol(workspace):
    response = server._response(
        workspace, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert response["result"]["tools"][0]["name"] == "diagnostic"


def test_main_serves_selected_workspace(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        server,
        "serve_stdio",
        lambda instance, responder: captured.append((instance, responder)),
    )
    monkeypatch.setattr("sys.argv", ["apple-shell", str(tmp_path)])
    server.main()
    assert captured[0][0].workspace == Path(tmp_path)


def test_rejects_file_as_workspace(tmp_path):
    file_path = tmp_path / "file"
    file_path.write_text("x")
    with pytest.raises(ValueError, match="directory"):
        server.AppleShellServer(file_path)


def test_module_entry_point_executes_main(monkeypatch, tmp_path):
    import runpy

    captured = []
    monkeypatch.setattr(
        "harness.mcp_servers.filesystem.serve_stdio",
        lambda instance, responder: captured.append(instance),
    )
    monkeypatch.setattr("sys.argv", ["apple-shell", str(tmp_path)])
    with pytest.warns(RuntimeWarning):
        runpy.run_module("harness.mcp_servers.apple_shell.server", run_name="__main__")
    assert captured[0].workspace == Path(tmp_path)
