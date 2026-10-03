"""MCP stdio and filesystem security contracts."""

import errno
import json
import os
import runpy
import subprocess
import sys
import time
import tomllib
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from harness.configuration import HarnessConfig
from harness.core import Config, Permissions
from harness.mcp import MCPClient, MCPError, builtin_filesystem_command
from harness.mcp_servers.filesystem import FilesystemServer, _response
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.tools import ToolExecutionError, ToolRegistry


def test_builtin_server_lives_in_dedicated_installable_package(tmp_path):
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    assert (project / "src" / "harness" / "mcp_servers" / "__init__.py").is_file()
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert metadata["project"]["scripts"]["harness-filesystem-mcp"] == (
        "harness.mcp_servers.filesystem:main"
    )
    assert builtin_filesystem_command(tmp_path)[2] == ("harness.mcp_servers.filesystem")


def test_external_server_requires_explicit_trusted_local_acknowledgement():
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {
                "tools": {
                    "mcp": {
                        "servers": {
                            "external": {
                                "command": ["/bin/echo"],
                                "allow_tools": ["echo"],
                            }
                        }
                    }
                }
            }
        )
    settings = HarnessConfig.model_validate(
        {
            "tools": {
                "mcp": {
                    "servers": {
                        "external": {
                            "command": ["/bin/echo"],
                            "allow_tools": ["echo"],
                            "trusted_local": True,
                            "read_roots": ["."],
                        }
                    }
                }
            }
        }
    )
    assert settings.tools.mcp.servers["external"].trusted_local
    with pytest.raises(ValidationError, match="requires explicit read_roots"):
        HarnessConfig.model_validate(
            {
                "tools": {
                    "mcp": {
                        "servers": {
                            "external": {
                                "command": ["/bin/echo"],
                                "allow_tools": ["echo"],
                                "trusted_local": True,
                            }
                        }
                    }
                }
            }
        )


def test_stdio_mcp_process_tree_is_registered_and_killed_on_task_abort(
    tmp_path, monkeypatch
):
    from harness import mcp

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_started = tmp_path / "child-started"
    child_survived = tmp_path / "child-survived"
    child_code = (
        "import time; from pathlib import Path; time.sleep(0.6); "
        f"Path({str(child_survived)!r}).touch()"
    )
    server_code = (
        "import signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"Path({str(child_started)!r}).touch(); time.sleep(30)"
    )
    real_popen = subprocess.Popen
    control = RunControl()

    def start_server(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 2
        while not child_started.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert child_started.exists()
        return process

    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    monkeypatch.setattr(mcp.subprocess, "Popen", start_server)
    client = MCPClient([sys.executable, "-c", server_code], workspace)

    def abort_before_first_rpc(process):
        control.stop_event.set()
        RunControl.register(control, process)

    monkeypatch.setattr(control, "register", abort_before_first_rpc)
    with use_run_control(control), pytest.raises(TaskCancelled):
        client.discover()

    time.sleep(0.7)
    assert not control._processes
    assert not child_survived.exists()


def test_obsidian_server_reads_only_visible_markdown_in_vault(tmp_path):
    from harness.mcp_servers.obsidian import ObsidianServer

    vault = tmp_path / "vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "a.md").write_text(
        "---\ntype: decision\n---\n# Alpha\nalpha decision [[Beta]]"
    )
    (vault / "notes" / "b.md").write_text("beta decision")
    (vault / "notes" / "raw.txt").write_text("private")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "hidden.md").write_text("private")
    (vault / "notes" / "link.md").symlink_to(tmp_path / "outside.md")
    server = ObsidianServer(vault)
    assert server.call("list_notes", {"limit": 1}) == {"notes": ["notes/a.md"]}
    assert server.call("read_note", {"path": "notes/a.md"}) == {
        "content": "---\ntype: decision\n---\n# Alpha\nalpha decision [[Beta]]"
    }
    assert server.call("search_notes", {"query": "decision"}) == {
        "matches": ["notes/a.md", "notes/b.md"]
    }
    assert server.call("knowledge_note", {"path": "notes/a.md"})["metadata"] == {
        "type": "decision"
    }
    assert server.call("knowledge_note", {"path": "notes/a.md"})["links"] == ["Beta"]
    assert server.call("backlinks", {"path": "Beta.md"}) == {"matches": ["notes/a.md"]}
    assert (
        server.call("knowledge_search", {"query": "decision"})["matches"][0]["path"]
        == "notes/a.md"
    )
    for path in (
        "../outside.md",
        ".obsidian/hidden.md",
        "notes/raw.txt",
        "notes/link.md",
    ):
        with pytest.raises((ValueError, PermissionError, OSError)):
            server.call("read_note", {"path": path})
    with pytest.raises(ValueError):
        server.call("read_note", {"path": "."})
    with pytest.raises(ValueError):
        server.call("write_note", {"path": "notes/a.md", "content": "bad"})


def test_obsidian_knowledge_note_with_review_date_is_json_rpc_serializable(tmp_path):
    from harness.mcp_servers import obsidian

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "review.md").write_text(
        "---\nlast_reviewed: 2026-10-03\n---\n# Review\nVault evidence"
    )
    response = obsidian._response(
        obsidian.ObsidianServer(vault),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "knowledge_note", "arguments": {"path": "review.md"}},
        },
    )
    assert response["result"]["structuredContent"]["metadata"]["last_reviewed"] == (
        "2026-10-03"
    )
    assert json.loads(json.dumps(response))["id"] == 1


def test_obsidian_server_protocol_limits_and_binary_skip(tmp_path, monkeypatch):
    from harness.mcp_servers import obsidian

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("needle")
    (vault / "b.md").write_bytes(b"\xff")
    (vault / "hidden").mkdir()
    (vault / "hidden" / "c.md").write_text("needle")
    (vault / "alias").symlink_to(vault / "hidden", target_is_directory=True)
    server = obsidian.ObsidianServer(vault)
    assert server.call("search_notes", {"query": "needle", "limit": 1}) == {
        "matches": ["a.md"]
    }
    assert server.call("search_notes", {"query": "other"}) == {"matches": []}
    assert server.call("list_notes", {}) == {"notes": ["a.md", "b.md", "hidden/c.md"]}
    with pytest.raises(JsonSchemaValidationError):
        server.call("search_notes", {"query": "x", "limit": 101})
    with pytest.raises(ValueError):
        obsidian.ObsidianServer(tmp_path.anchor)
    initialized = obsidian._response(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        },
    )
    assert initialized["result"]["serverInfo"]["name"] == "harness-obsidian"
    tools = obsidian._response(
        server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    assert {item["name"] for item in tools["result"]["tools"]} == obsidian.READ_TOOLS
    failed = obsidian._response(
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "read_note", "arguments": {"path": "b.md"}},
        },
    )
    assert failed["result"]["isError"]
    assert "needle" not in str(failed)
    monkeypatch.setattr(obsidian, "MAX_SCAN", 1)
    with pytest.raises(ValueError, match="scan limit"):
        server.call("list_notes", {})


def test_obsidian_stdio_entrypoint(tmp_path, monkeypatch):
    from harness.mcp_servers import obsidian

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("hello")
    message = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}).encode()
    output = StringIO()
    monkeypatch.setattr(sys, "argv", ["harness-obsidian-mcp", str(vault)])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=BytesIO(message + b"\n")))
    monkeypatch.setattr(sys, "stdout", output)
    obsidian.main()
    assert len(json.loads(output.getvalue())["result"]["tools"]) == 6
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=BytesIO(message + b"\n")))
    second = StringIO()
    monkeypatch.setattr(sys, "stdout", second)
    monkeypatch.delitem(sys.modules, "harness.mcp_servers.obsidian")
    runpy.run_module("harness.mcp_servers.obsidian", run_name="__main__")
    assert len(json.loads(second.getvalue())["result"]["tools"]) == 6


def test_obsidian_registry_rejects_unexpected_builtin_tool(tmp_path, monkeypatch):
    from harness import mcp

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config()
    config.data["paths"]["obsidian_vault"] = str(vault)
    config.data["tools"] = {"mcp": {"servers": {"vault": {"builtin": "obsidian"}}}}
    monkeypatch.setattr(
        mcp,
        "MCPClient",
        lambda *_args, **_kwargs: SimpleNamespace(
            discover=lambda: [{"name": "rogue", "inputSchema": {}}]
        ),
    )
    with pytest.raises(ValueError, match="unknown tool"):
        ToolRegistry(Permissions(config), workspace=workspace)


def test_builtin_mcp_isolation_includes_python_base_runtime(tmp_path, monkeypatch):
    from harness import mcp

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config()
    config.data["paths"]["obsidian_vault"] = str(vault)
    config.data["tools"] = {"mcp": {"servers": {"vault": {"builtin": "obsidian"}}}}
    captured = {}

    class Client:
        def __init__(
            self, _command, _workspace, *, read_roots, python_import_roots, **_kwargs
        ):
            captured["read_roots"] = read_roots
            captured["python_import_roots"] = python_import_roots

        def discover(self):
            return []

    monkeypatch.setattr(mcp, "MCPClient", Client)
    ToolRegistry(Permissions(config), workspace=workspace)
    assert Path(sys.base_prefix) in captured["read_roots"]
    assert (
        Path(__file__).resolve().parents[1] / "src" in captured["python_import_roots"]
    )


def test_builtin_python_import_roots_are_scoped_to_builtin_mcp(tmp_path, monkeypatch):
    from harness import mcp

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.data["tools"] = {
        "mcp": {
            "servers": {
                "external": {
                    "command": ["/bin/echo"],
                    "trusted_local": True,
                    "read_roots": ["."],
                    "allow_tools": ["echo"],
                }
            }
        }
    }
    captured = {}

    class Client:
        def __init__(self, _command, _workspace, *, python_import_roots, **_kwargs):
            captured["python_import_roots"] = python_import_roots

        def discover(self):
            return [{"name": "echo", "inputSchema": {"type": "object"}}]

    monkeypatch.setattr(mcp, "MCPClient", Client)
    ToolRegistry(Permissions(config), workspace=workspace)
    assert captured["python_import_roots"] == ()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec acceptance")
def test_native_obsidian_builtin_uses_read_only_vault_profile(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("hello vault")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.data["paths"]["obsidian_vault"] = str(vault)
    config.data["tools"] = {
        "permissions": {"obsidian": "read"},
        "mcp": {"servers": {"vault": {"builtin": "obsidian"}}},
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert registry.execute("mcp.vault.read_note", {"path": "note.md"}) == {
        "content": "hello vault"
    }
    assert registry.execute("mcp.vault.search_notes", {"query": "hello"}) == {
        "matches": ["note.md"]
    }


def test_obsidian_builtin_is_read_only_and_profile_gated(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("memory")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.data["paths"]["obsidian_vault"] = str(vault)
    config.data["tools"] = {
        "permissions": {"obsidian": "read"},
        "mcp": {"servers": {"vault": {"builtin": "obsidian"}}},
    }
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert registry.execute("mcp.vault.read_note", {"path": "note.md"}) == {
        "content": "memory"
    }
    assert registry.execute("mcp.vault.list_notes", {}) == {"notes": ["note.md"]}
    profile = SimpleNamespace(tools=[], permissions=[])
    with pytest.raises(PermissionError):
        registry.execute("mcp.vault.read_note", {"path": "note.md"}, profile=profile)
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {
                "tools": {
                    "mcp": {
                        "servers": {
                            "vault": {"builtin": "obsidian", "read_only": False}
                        }
                    }
                }
            }
        )
    for unsafe in ({"allow_delete": True}, {"trusted_local": True}):
        with pytest.raises(ValidationError):
            HarnessConfig.model_validate(
                {
                    "tools": {
                        "mcp": {"servers": {"vault": {"builtin": "obsidian", **unsafe}}}
                    }
                }
            )


def test_apple_shell_builtin_registers_only_fixed_read_tool(tmp_path, monkeypatch):
    from harness import mcp, tools
    from harness.mcp_servers.apple_shell import EXECUTABLES, TOOLS
    from harness.mcp_servers.apple_shell.server import OUTPUTS

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    class Client:
        def __init__(self, command, _workspace, **kwargs):
            captured["command"] = command
            captured.update(kwargs)

        def discover(self):
            return [
                {
                    "name": name,
                    "description": "fixed diagnostic",
                    "inputSchema": schema,
                    "outputSchema": OUTPUTS[name],
                }
                for name, schema in TOOLS.items()
            ]

    monkeypatch.setattr(mcp, "MCPClient", Client)
    monkeypatch.setattr(tools.sys, "platform", "darwin")
    config = Config()
    config.data["tools"] = {
        "permissions": {"apple_shell": "read"},
        "mcp": {"servers": {"apple_diagnostics": {"builtin": "apple_shell"}}},
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert captured["command"] == mcp.builtin_apple_shell_command(workspace)
    assert captured["executable_paths"] == tuple(EXECUTABLES.values())
    assert (
        registry.specs["mcp.apple_diagnostics.diagnostic"].permission == "apple_shell"
    )


def test_apple_shell_builtin_rejects_non_macos(tmp_path, monkeypatch):
    from harness import tools

    monkeypatch.setattr(tools.sys, "platform", "linux")
    config = Config()
    config.data["tools"] = {
        "mcp": {"servers": {"apple_diagnostics": {"builtin": "apple_shell"}}}
    }
    with pytest.raises(ValueError, match="requires macOS"):
        ToolRegistry(Permissions(config), workspace=tmp_path)


def test_filesystem_server_confines_paths_and_separates_permissions(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    server = FilesystemServer(root, read_only=False, allow_delete=False)
    assert server.call(
        "write_file", {"path": "nested/file.txt", "content": "hello"}
    ) == {"path": "nested/file.txt"}
    assert server.call("read_file", {"path": "nested/file.txt"}) == {"content": "hello"}
    assert server.call("list_directory", {"path": "nested"}) == {
        "entries": ["file.txt"]
    }
    assert server.call("search", {"query": "hello"})["matches"] == ["nested/file.txt"]
    for path in ("../outside.txt", str(outside), ".git/config"):
        with pytest.raises((PermissionError, ValueError)):
            server.call("read_file", {"path": path})
    (root / "link").symlink_to(outside)
    with pytest.raises((PermissionError, OSError)):
        server.call("read_file", {"path": "link"})
    with pytest.raises(PermissionError):
        server.call("delete_file", {"path": "nested/file.txt"})
    with pytest.raises(PermissionError):
        server.call("move_file", {"path": "nested/file.txt", "destination": "new.txt"})
    readonly = FilesystemServer(root, read_only=True, allow_delete=False)
    with pytest.raises(PermissionError):
        readonly.call("write_file", {"path": "nested/file.txt", "content": "changed"})


def test_filesystem_copy_exists_glob_and_permission_split(tmp_path):
    root = tmp_path / "workspace"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text("alpha")
    (root / "docs" / "b.md").write_text("beta")
    (root / "docs" / "note.txt").write_text("note")
    (root / "docs" / "link.md").symlink_to(root / "docs" / "a.md")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private")
    readonly = FilesystemServer(root, read_only=True)
    assert readonly.call("exists", {"path": "."}) == {"exists": True}
    assert readonly.call("exists", {"path": "docs/a.md"}) == {"exists": True}
    assert readonly.call("exists", {"path": "docs/missing.md"}) == {"exists": False}
    assert readonly.call("exists", {"path": "docs/link.md"}) == {"exists": False}
    assert readonly.call("glob", {"pattern": "docs/*.md", "limit": 1}) == {
        "matches": ["docs/a.md"]
    }
    assert readonly.call("glob", {"pattern": "*.txt"}) == {"matches": ["docs/note.txt"]}
    assert readonly.call("glob", {"pattern": "*.md"}) == {
        "matches": ["docs/a.md", "docs/b.md"]
    }
    assert readonly.call("exists", {"path": "docs"}) == {"exists": True}
    assert readonly.call("exists", {"path": "docs/missing/child"}) == {"exists": False}
    with pytest.raises(PermissionError):
        readonly.call("copy_file", {"path": "docs/a.md", "destination": "copy.md"})
    server = FilesystemServer(root, read_only=False, allow_delete=False)
    assert server.call(
        "copy_file", {"path": "docs/a.md", "destination": "copies/a.md"}
    ) == {"path": "copies/a.md"}
    assert (root / "docs" / "a.md").read_text() == "alpha"
    assert (root / "copies" / "a.md").read_text() == "alpha"
    with pytest.raises(FileExistsError):
        server.call("copy_file", {"path": "docs/a.md", "destination": "copies/a.md"})
    with pytest.raises(ValueError):
        server.call("copy_file", {"path": "docs/a.md", "destination": "docs/a.md"})
    for bad in ("../outside.md", ".git/config", str(tmp_path / "outside")):
        with pytest.raises((ValueError, PermissionError)):
            readonly.call("glob", {"pattern": bad})
        with pytest.raises((ValueError, PermissionError)):
            readonly.call("exists", {"path": bad})
    with pytest.raises((OSError, PermissionError)):
        server.call(
            "copy_file", {"path": "docs/link.md", "destination": "copies/link.md"}
        )
    with pytest.raises((OSError, PermissionError)):
        server.call("copy_file", {"path": "docs/a.md", "destination": "docs/link.md"})
    assert (root / "docs" / "a.md").read_text() == "alpha"
    with pytest.raises(JsonSchemaValidationError):
        server.call("glob", {"pattern": "*.md", "limit": 101})
    with pytest.raises(ValueError):
        server.call("glob", {"pattern": "."})


def test_filesystem_exists_handles_loop_and_propagates_other_io_errors(
    tmp_path, monkeypatch
):
    root = tmp_path / "workspace"
    root.mkdir()
    server = FilesystemServer(root)
    original_stat = os.stat

    def fail_stat(path, *args, **kwargs):
        if path == "loop":
            raise OSError(errno.ELOOP, "loop")
        if path == "denied":
            raise OSError(errno.EACCES, "denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_stat)
    assert server.call("exists", {"path": "loop"}) == {"exists": False}
    with pytest.raises(OSError) as error:
        server.call("exists", {"path": "denied"})
    assert error.value.errno == errno.EACCES


def test_stdio_server_protocol(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "harness.mcp_servers.filesystem",
            str(root),
            "--read-only",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    try:

        def request(message):
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            return json.loads(process.stdout.readline())

        assert (
            request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
            )["result"]["serverInfo"]["name"]
            == "harness-filesystem"
        )
        assert (
            "tools"
            in request(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            )["result"]
        )
        result = request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "write_file",
                    "arguments": {"path": "x", "content": "x"},
                },
            }
        )["result"]
        assert result["isError"]
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_registry_discovers_builtin_and_enforces_permissions(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "read"},
        "mcp": {"servers": {"files": {"builtin": "filesystem", "read_only": True}}},
    }
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert "mcp.files.read_file" in registry.list()
    (workspace / "sample.txt").write_text("hello")
    assert registry.execute("mcp.files.read_file", {"path": "sample.txt"}) == {
        "content": "hello"
    }
    with pytest.raises(PermissionError):
        registry.execute(
            "mcp.files.write_file", {"path": "sample.txt", "content": "bad"}
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec acceptance")
def test_native_builtin_mcp_runs_under_process_isolation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("native")
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "read"},
        "mcp": {"servers": {"files": {"builtin": "filesystem"}}},
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert registry.execute("mcp.files.read_file", {"path": "sample.txt"}) == {
        "content": "native"
    }
    config.data["tools"]["permissions"] = {
        "filesystem": "write",
        "filesystem.delete": "write",
    }
    config.data["tools"]["mcp"]["servers"]["files"] = {
        "builtin": "filesystem",
        "read_only": False,
        "allow_delete": True,
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert registry.execute(
        "mcp.files.create_file", {"path": "new.txt", "content": "created"}
    ) == {"path": "new.txt"}
    assert registry.execute("mcp.files.exists", {"path": "new.txt"}) == {"exists": True}
    assert registry.execute("mcp.files.glob", {"pattern": "*.txt"}) == {
        "matches": ["new.txt", "sample.txt"]
    }
    assert registry.execute(
        "mcp.files.copy_file", {"path": "new.txt", "destination": "copy.txt"}
    ) == {"path": "copy.txt"}
    assert registry.execute(
        "mcp.files.move_file", {"path": "new.txt", "destination": "moved.txt"}
    ) == {"path": "moved.txt"}
    assert registry.execute("mcp.files.delete_file", {"path": "moved.txt"}) == {
        "deleted": True
    }
    assert not (workspace / "moved.txt").exists()


def test_client_rejects_untrusted_executable_and_protocol_failure(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError):
        MCPClient(["relative-command"], workspace)
    client = MCPClient(
        [
            sys.executable,
            "-m",
            "harness.mcp_servers.filesystem",
            str(workspace),
            "--read-only",
        ],
        workspace,
    )
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    assert any(item["name"] == "read_file" for item in client.discover())


def test_filesystem_mutations_and_limits(tmp_path, monkeypatch):
    from harness.mcp_servers import filesystem as fs

    root = tmp_path / "workspace"
    root.mkdir()
    server = FilesystemServer(root, read_only=False, allow_delete=True)
    assert server.call("create_directory", {"path": "nested"}) == {"path": "nested"}
    assert server.call("create_directory", {"path": "nested"}) == {"path": "nested"}
    assert server.call("create_file", {"path": "nested/a.txt", "content": "alpha"}) == {
        "path": "nested/a.txt"
    }
    with pytest.raises(FileExistsError):
        server.call("create_file", {"path": "nested/a.txt", "content": "duplicate"})
    assert server.call("write_file", {"path": "nested/a.txt", "content": "beta"}) == {
        "path": "nested/a.txt"
    }
    assert server.call(
        "move_file", {"path": "nested/a.txt", "destination": "nested/b.txt"}
    ) == {"path": "nested/b.txt"}
    with pytest.raises(ValueError):
        server.call(
            "move_file", {"path": "nested/b.txt", "destination": "nested/b.txt"}
        )
    with pytest.raises(FileExistsError):
        server.call("move_file", {"path": "nested/b.txt", "destination": "nested"})
    assert not (root / "nested/a.txt").exists()
    assert (root / "nested/b.txt").read_text() == "beta"
    assert server.call("search", {"query": "beta", "limit": 1}) == {
        "matches": ["nested/b.txt"]
    }
    assert server.call("list_directory", {"path": "."}) == {"entries": ["nested"]}
    assert server.call("delete_file", {"path": "nested/b.txt"}) == {"deleted": True}
    with pytest.raises(FileNotFoundError):
        server.call("read_file", {"path": "nested/a.txt"})
    with pytest.raises(ValueError):
        server.call("delete_file", {"path": "nested"})
    with pytest.raises(ValueError):
        server.call("read_file", {"path": "."})
    with pytest.raises(ValueError):
        server.call("unknown", {})
    with pytest.raises(JsonSchemaValidationError):
        server.call("read_file", {})
    monkeypatch.setattr(fs, "MAX_FILE", 2)
    with pytest.raises(ValueError):
        server.call("write_file", {"path": "a", "content": "long"})
    (root / "large").write_text("long")
    with pytest.raises(ValueError):
        server.call("read_file", {"path": "large"})


def test_filesystem_search_skips_links_git_and_binary(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("needle")
    (root / "good.txt").write_text("needle")
    (root / "bad.bin").write_bytes(b"\xffneedle")
    (root / "link").symlink_to(root / "good.txt")
    server = FilesystemServer(root)
    assert server.call("search", {"query": "needle"}) == {"matches": ["good.txt"]}
    assert server.call("list_directory", {"path": "."}) == {
        "entries": ["bad.bin", "good.txt", "link"]
    }


def test_server_rpc_contract(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    server = FilesystemServer(root)
    assert _response(server, ["not", "an", "object"]) is None
    assert _response(server, {"method": "notifications/initialized"}) is None
    assert _response(server, {"jsonrpc": "2.0", "method": "tools/list"}) is None
    assert (
        _response(server, {"jsonrpc": "1.0", "id": 1, "method": "tools/list"}) is None
    )
    assert (
        _response(server, {"jsonrpc": "2.0", "id": 1, "method": "unknown"})["error"][
            "code"
        ]
        == -32601
    )
    assert (
        _response(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "old"},
            },
        )["error"]["code"]
        == -32602
    )
    result = _response(
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_directory", "arguments": {"path": "."}},
        },
    )["result"]
    assert result["structuredContent"] == {"entries": []}
    assert _response(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call"})[
        "result"
    ]["isError"]
    assert _response(
        server, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": []}
    )["result"]["isError"]
    assert _response(
        server,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": 12}},
        },
    )["result"]["isError"]


FAKE_SERVER = r"""
import json, signal, sys, time
mode = sys.argv[1]
if mode == 'hang_ignores_term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
for line in sys.stdin:
    message = json.loads(line)
    if 'id' not in message: continue
    method = message['method']
    if mode in {'hang', 'hang_ignores_term'}: time.sleep(2); continue
    if mode == 'close': break
    if mode == 'partial': sys.stdout.write('{'); sys.stdout.flush(); continue
    if mode == 'malformed': sys.stdout.write('bad\n'); sys.stdout.flush(); continue
    if mode == 'oversize': sys.stdout.write('x' * 2100000); sys.stdout.flush(); continue
    if method == 'initialize':
        result = {'protocolVersion': 'bad' if mode == 'version' else '2025-11-25', 'capabilities': {} if mode == 'capability' else {'tools': {}}}
    elif method == 'tools/list':
        tool = {'name': 'echo', 'inputSchema': {'type': 'object'}}
        if mode == 'bad_schema': tool['inputSchema'] = {'type': 'not-a-type'}
        if mode == 'bad_name': tool['name'] = 'bad name'
        if mode == 'no_schema': tool.pop('inputSchema')
        if mode == 'bad_output_type': tool['outputSchema'] = 'not a schema'
        if mode == 'bad_output_schema': tool['outputSchema'] = {'type': 'not-a-type'}
        if mode == 'output_mismatch': tool['outputSchema'] = {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}
        result = {'tools': 'bad' if mode == 'bad_list' else [tool, tool] if mode == 'duplicate' else [tool]}
    else:
        result = {'structuredContent': {'value': 'ok'}, 'isError': mode == 'tool_error'}
        if mode == 'no_structured': result.pop('structuredContent')
        if mode == 'text_only': result = {'content': [{'type': 'text', 'text': 'plain text'}]}
    reply = {'jsonrpc': '2.0', 'id': message['id'], 'result': result}
    if mode == 'bad_id': reply['id'] += 1
    if mode == 'rpc_error': reply = {'jsonrpc': '2.0', 'id': message['id'], 'error': {'code': -1, 'message': 'private'}}
    sys.stdout.write(json.dumps(reply) + '\n'); sys.stdout.flush()
"""


@pytest.mark.parametrize(
    "mode",
    [
        "version",
        "capability",
        "bad_schema",
        "bad_name",
        "no_schema",
        "bad_output_type",
        "bad_output_schema",
        "bad_list",
        "duplicate",
        "bad_id",
        "rpc_error",
        "tool_error",
        "no_structured",
        "hang",
        "close",
        "partial",
        "malformed",
        "oversize",
        "hang_ignores_term",
    ],
)
def test_client_rejects_bad_servers(tmp_path, monkeypatch, mode):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    client = MCPClient([sys.executable, "-c", FAKE_SERVER, mode], root, timeout=0.3)
    with pytest.raises(MCPError):
        if mode in {"tool_error", "no_structured"}:
            client.call("echo", {}, {"type": "object"})
        else:
            client.discover()


def test_client_call_and_schema_pin(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    client = MCPClient([sys.executable, "-c", FAKE_SERVER, "ok"], root)
    assert client.call("echo", {}, {"type": "object"}) == {"value": "ok"}
    with pytest.raises(MCPError, match="schema changed"):
        client.call("echo", {}, {"type": "array"})
    with pytest.raises(MCPError, match="schema changed"):
        client.call("missing", {}, {})
    assert MCPClient([sys.executable, "-c", FAKE_SERVER, "text_only"], root).call(
        "echo", {}, {"type": "object"}
    ) == {"content": "plain text"}
    assert (
        builtin_filesystem_command(root, read_only=False, allow_delete=True)[-1]
        == "--allow-delete"
    )


def test_filesystem_server_entrypoint_and_boundaries(tmp_path, monkeypatch):
    from harness.mcp_servers import filesystem as fs

    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ValueError):
        FilesystemServer(tmp_path.anchor)
    server = FilesystemServer(root, read_only=False, allow_delete=True)
    assert (
        _response(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            },
        )["result"]["protocolVersion"]
        == "2025-11-25"
    )
    assert (
        len(
            _response(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})[
                "result"
            ]["tools"]
        )
        >= 7
    )
    (root / "miss.txt").write_text("other")
    assert server.call("search", {"query": "absent"}) == {"matches": []}
    (root / "outside-link").symlink_to(tmp_path)
    with pytest.raises(OSError):
        server.call("create_file", {"path": "outside-link/x", "content": "bad"})
    (root / "destination").symlink_to(tmp_path / "private")
    server.call("write_file", {"path": "destination", "content": "safe"})
    assert (root / "destination").read_text() == "safe"
    assert not (tmp_path / "private").exists()
    (root / "tiny").write_text("a")
    with monkeypatch.context() as patch:
        patch.setattr(fs.os, "read", lambda *_args: b"long")
        patch.setattr(fs, "MAX_FILE", 2)
        with pytest.raises(ValueError):
            server.call("read_file", {"path": "tiny"})

    inputs = BytesIO(
        b"not-json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).encode()
        + b"\n"
    )
    output = StringIO()
    monkeypatch.setattr(sys, "argv", ["mcp_filesystem", str(root), "--read-only"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=inputs))
    monkeypatch.setattr(sys, "stdout", output)
    runpy.run_path(fs.__file__, run_name="__main__")
    assert json.loads(output.getvalue())["id"] == 3
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(buffer=BytesIO(b"x" * 2_000_001 + b"\n"))
    )
    monkeypatch.setattr(sys, "stdout", StringIO())
    runpy.run_path(fs.__file__, run_name="__main__")
    assert sys.stdout.getvalue() == ""


def test_filesystem_temporary_write_cleanup_on_error(tmp_path, monkeypatch):
    from harness.mcp_servers import filesystem as fs

    root = tmp_path / "workspace"
    root.mkdir()
    server = FilesystemServer(root, read_only=False)
    original_write = fs.os.write

    def fail_write(*_args):
        raise OSError("injected")

    monkeypatch.setattr(fs.os, "write", fail_write)
    with pytest.raises(OSError):
        server.call("write_file", {"path": "x", "content": "value"})
    assert list(root.iterdir()) == []
    with pytest.raises(OSError):
        server.call("create_file", {"path": "x", "content": "value"})
    assert list(root.iterdir()) == []
    calls = []

    def partial_write(fd, raw):
        calls.append(True)
        return original_write(fd, raw[:1])

    monkeypatch.setattr(fs.os, "write", partial_write)
    server.call("create_file", {"path": "x", "content": "value"})
    assert (root / "x").read_text() == "value"
    assert len(calls) == 5
    monkeypatch.setattr(fs.os, "write", lambda *_args: 0)
    with pytest.raises(OSError):
        server.call("write_file", {"path": "x", "content": "changed"})
    assert (root / "x").read_text() == "value"


def test_client_configuration_and_transport_limits(tmp_path, monkeypatch):
    from harness import mcp

    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ValueError):
        MCPClient([sys.executable], root, timeout=31)
    with pytest.raises(ValueError):
        MCPClient([str(root / "owned")], root)
    with pytest.raises(ValueError):
        MCPClient([], root)
    with pytest.raises(ValueError):
        MCPClient([sys.executable, 3], root)
    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    client = MCPClient([sys.executable, "-c", FAKE_SERVER, "ok"], root)
    monkeypatch.delenv("TMPDIR", raising=False)
    assert client.discover()[0]["name"] == "echo"
    monkeypatch.setattr(mcp, "MAX_MESSAGE", 1)
    with pytest.raises(MCPError, match="request is too large"):
        client.discover()


def test_client_uses_only_explicit_python_import_roots(tmp_path, monkeypatch):
    from harness import mcp

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/untrusted/ambient")
    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    environments = []

    def capture_spawn(*_args, **kwargs):
        environments.append(kwargs["env"])
        raise OSError("injected")

    monkeypatch.setattr(mcp.subprocess, "Popen", capture_spawn)
    for roots in ((), (trusted,)):
        client = MCPClient([sys.executable], workspace, python_import_roots=roots)
        with pytest.raises(MCPError, match="could not start"):
            client.discover()
    assert "PYTHONPATH" not in environments[0]
    assert environments[1]["PYTHONPATH"] == str(trusted.resolve())
    for invalid in (Path("relative"), tmp_path / "missing", workspace / "file"):
        with pytest.raises((ValueError, FileNotFoundError)):
            MCPClient([sys.executable], workspace, python_import_roots=(invalid,))


def test_client_redacts_spawn_and_read_errors(tmp_path, monkeypatch):
    from harness import mcp

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    client = MCPClient([sys.executable, "-c", FAKE_SERVER, "ok"], root)
    with monkeypatch.context() as patch:
        patch.setattr(
            mcp.subprocess,
            "Popen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private")),
        )
        with pytest.raises(MCPError, match="could not start") as error:
            client.discover()
        assert "private" not in str(error.value)
    with monkeypatch.context() as patch:
        patch.setattr(
            mcp.select,
            "select",
            lambda *_args: (_ for _ in ()).throw(OSError("private")),
        )
        with pytest.raises(MCPError, match="transport failed"):
            client.discover()


def test_external_mcp_requires_explicit_tool_and_write_grant(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    config = Config()
    config.data["tools"] = {
        "permissions": {"mcp.external": "read"},
        "mcp": {
            "servers": {
                "external": {
                    "command": [sys.executable, "-c", FAKE_SERVER, "ok"],
                    "allow_tools": ["echo"],
                    "trusted_local": True,
                    "read_roots": ["."],
                }
            }
        },
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    with pytest.raises(PermissionError):
        registry.execute("mcp.external.echo", {})
    config.data["tools"]["permissions"]["mcp.external"] = "write"
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert registry.execute("mcp.external.echo", {}) == {"value": "ok"}
    config.data["tools"]["mcp"]["servers"]["external"]["allow_tools"] = ["other"]
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    assert "mcp.external.echo" not in registry.list()


def test_mcp_profile_and_recovery_boundaries(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "write"},
        "mcp": {"servers": {"files": {"builtin": "filesystem", "read_only": False}}},
    }
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    registry = ToolRegistry(Permissions(config), workspace=root)
    profile = SimpleNamespace(tools=["mcp.files.read_file"], permissions=["filesystem"])
    with pytest.raises(PermissionError, match="profile"):
        registry.execute(
            "mcp.files.write_file", {"path": "x", "content": "x"}, profile=profile
        )
    with (
        registry.recovery_writes({root / "x"}),
        pytest.raises(PermissionError, match="recovery"),
    ):
        registry.execute("mcp.files.write_file", {"path": "x", "content": "x"})
    assert not (root / "x").exists()


def test_mcp_registry_rejects_bad_server_names_and_unknown_builtin_tools(
    tmp_path, monkeypatch
):
    from harness import mcp

    root = tmp_path / "workspace"
    root.mkdir()
    config = Config()
    config.data["tools"] = {"mcp": {"servers": {"bad.name": {"builtin": "filesystem"}}}}
    with pytest.raises(ValueError, match="server name"):
        ToolRegistry(Permissions(config), workspace=root)
    config.data["tools"]["mcp"]["servers"] = {"files": {"builtin": "filesystem"}}
    with monkeypatch.context() as patch:
        patch.setattr(
            mcp,
            "MCPClient",
            lambda *_args, **_kwargs: SimpleNamespace(
                discover=lambda: [{"name": "rogue", "inputSchema": {}}]
            ),
        )
        with pytest.raises(ValueError, match="unknown tool"):
            ToolRegistry(Permissions(config), workspace=root)
    with monkeypatch.context() as patch:

        def unavailable(*_args, **_kwargs):
            raise MCPError("private")

        patch.setattr(
            mcp,
            "MCPClient",
            lambda *_args, **_kwargs: SimpleNamespace(discover=unavailable),
        )
        with pytest.raises(ValueError, match="unavailable") as error:
            ToolRegistry(Permissions(config), workspace=root)
        assert "private" not in str(error.value)


def test_mcp_audit_never_persists_arguments_or_results(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "private.txt").write_text("UNLISTED_PRIVATE_VALUE")
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "read"},
        "mcp": {"servers": {"files": {"builtin": "filesystem"}}},
    }
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    events = []
    registry = ToolRegistry(
        Permissions(config),
        lambda kind, payload: events.append((kind, payload)),
        workspace=root,
    )
    assert registry.execute("mcp.files.read_file", {"path": "private.txt"}) == {
        "content": "UNLISTED_PRIVATE_VALUE"
    }
    with pytest.raises(ValueError, match="invalid tool arguments") as error:
        registry.execute("mcp.files.read_file", {"path": ["UNLISTED_PRIVATE_VALUE"]})
    assert "UNLISTED_PRIVATE_VALUE" not in str(error.value)
    with pytest.raises(MCPError):
        registry.execute("mcp.files.read_file", {"path": "missing.txt"})
    assert "UNLISTED_PRIVATE_VALUE" not in json.dumps(events)
    assert events[0][1]["input"] == {"argument_names": ["path"]}
    assert events[1][1]["output"] == {"type": "dict"}
    assert events[-1][1]["error"] == "MCPError"


def test_mcp_output_schema_is_enforced_without_exposing_result(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    config = Config()
    config.data["tools"] = {
        "permissions": {"mcp.external": "write"},
        "mcp": {
            "servers": {
                "external": {
                    "command": [sys.executable, "-c", FAKE_SERVER, "output_mismatch"],
                    "allow_tools": ["echo"],
                    "trusted_local": True,
                    "read_roots": ["."],
                }
            }
        },
    }
    monkeypatch.setattr(
        "harness.mcp.isolated_command", lambda command, *_args, **_kwargs: command
    )
    events = []
    registry = ToolRegistry(
        Permissions(config),
        lambda kind, payload: events.append((kind, payload)),
        workspace=root,
    )
    with pytest.raises(ToolExecutionError, match="MCP output schema violation"):
        registry.execute("mcp.external.echo", {})
    assert events[-1][1]["error"] == "ToolExecutionError"
    assert "ok" not in json.dumps(events)


def test_client_checks_run_control_and_write_deadline(tmp_path, monkeypatch):
    from harness import mcp

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    checks = []
    registered = set()
    control = SimpleNamespace(
        check=lambda: checks.append(True),
        register=registered.add,
        unregister=registered.remove,
    )
    monkeypatch.setattr(mcp, "current_run_control", lambda: control)
    client = MCPClient([sys.executable, "-c", FAKE_SERVER, "ok"], root)
    assert client.discover()[0]["name"] == "echo"
    assert len(checks) >= 2
    assert not registered
    original_select = mcp.select.select
    monkeypatch.setattr(
        mcp.select,
        "select",
        lambda reads, writes, errors, timeout: (
            ([], [], []) if writes else original_select(reads, writes, errors, timeout)
        ),
    )
    with pytest.raises(MCPError, match="timed out"):
        MCPClient(
            [sys.executable, "-c", FAKE_SERVER, "ok"], root, timeout=0.1
        ).discover()


def test_client_notices_stopped_server_without_readability(tmp_path, monkeypatch):
    from harness import mcp

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        mcp, "isolated_command", lambda command, *_args, **_kwargs: command
    )
    original_select = mcp.select.select

    def no_read(reads, writes, errors, timeout):
        if reads:
            time.sleep(0.05)
            return [], [], []
        return original_select(reads, writes, errors, timeout)

    monkeypatch.setattr(mcp.select, "select", no_read)
    with pytest.raises(MCPError, match="server stopped"):
        MCPClient(
            [sys.executable, "-c", FAKE_SERVER, "close"], root, timeout=1
        ).discover()


@pytest.mark.parametrize(
    "server",
    [
        {},
        {"builtin": "filesystem", "command": ["/bin/echo"]},
        {"command": ["/bin/echo"]},
        {"builtin": "filesystem", "allow_tools": ["read_file"]},
        {"builtin": "filesystem", "read_only": True, "allow_delete": True},
        {"builtin": "filesystem", "timeout": 50},
    ],
)
def test_mcp_configuration_rejects_ambiguous_or_unsafe_server(server):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"tools": {"mcp": {"servers": {"example": server}}}}
        )


def test_disabled_mcp_server_does_not_start(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    config = Config()
    config.data["tools"] = {
        "mcp": {"servers": {"files": {"builtin": "filesystem", "enabled": False}}}
    }
    monkeypatch.setattr(
        "harness.mcp.MCPClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("started")),
    )
    assert not any(
        name.startswith("mcp.")
        for name in ToolRegistry(Permissions(config), workspace=root).list()
    )


def test_registry_rejects_invalid_mcp_section_without_leaking_values(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    config = Config()
    config.data["tools"] = {
        "mcp": {
            "servers": {"files": {"builtin": "filesystem", "timeout": "PRIVATE_VALUE"}}
        }
    }
    with pytest.raises(
        ValueError, match="tools.mcp has invalid configuration"
    ) as error:
        ToolRegistry(Permissions(config), workspace=root)
    assert "PRIVATE_VALUE" not in str(error.value)
