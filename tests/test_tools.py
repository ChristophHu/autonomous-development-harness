import subprocess

import pytest

from harness.core import Config, Permissions
from harness.tools import ToolExecutor, ToolRegistry, ToolSpec


def executor(*rules):
    c = Config()
    c.data["tools"] = {"permissions": dict(rules)}
    return ToolExecutor(Permissions(c))


def test_filesystem_full(tmp_path):
    tool = executor(("filesystem", "write"), ("filesystem.delete", "write"))
    root = tmp_path / "ws"
    root.mkdir()
    tool.filesystem("create", "x/a.txt", workspace=str(root), content="needle")
    assert tool.filesystem("read", "x/a.txt", workspace=str(root)) == "needle"
    assert tool.filesystem("exists", "x/a.txt", workspace=str(root))
    assert tool.filesystem("list", "x", workspace=str(root)) == ["a.txt"]
    tool.filesystem("copy", "x/a.txt", workspace=str(root), destination="x/b.txt")
    assert tool.filesystem("glob", "*.txt", workspace=str(root))
    assert tool.filesystem("search", ".", workspace=str(root), query="needle")
    tool.filesystem("move", "x/b.txt", workspace=str(root), destination="x/c.txt")
    assert tool.filesystem("mkdir", "empty", workspace=str(root))
    assert tool.filesystem("delete", "x/c.txt", workspace=str(root))
    with pytest.raises(FileExistsError):
        tool.filesystem("create", "x/a.txt", workspace=str(root))
    with pytest.raises(ValueError):
        tool.filesystem("bogus", "x", workspace=str(root))
    with pytest.raises(PermissionError):
        tool.filesystem("read", "../outside", workspace=str(root))


def test_filesystem_permissions(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    tool = executor(("filesystem", "read"))
    with pytest.raises(PermissionError):
        tool.filesystem("write", "x", workspace=str(root), content="x")
    with pytest.raises(PermissionError):
        tool.filesystem("delete", "x", workspace=str(root))
    writable = executor(("filesystem", "write"))
    (root / "x").write_text("x")
    with pytest.raises(PermissionError):
        writable.filesystem("copy", "x", workspace=str(root), destination="../escape")


def test_process_git_docker_http(monkeypatch, tmp_path):
    calls = []
    cfg = Config()
    cfg.data["tools"] = {
        "permissions": {
            "shell": "write",
            "git": "write",
            "docker": "write",
            "http": "write",
        },
        "http": {"allowed_hosts": ["localhost"]},
    }
    tool = ToolExecutor(Permissions(cfg), http_allow_hosts=["localhost"])

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "out", "")

    monkeypatch.setattr("harness.tools.subprocess.run", run)
    monkeypatch.setattr(
        "harness.docker_broker.DockerComposeBroker.run",
        lambda _broker, action, tail=100: subprocess.CompletedProcess(
            ["docker", action, str(tail)], 0, "out", ""
        ),
    )
    assert tool.shell(["echo", "hi"], tmp_path).stdout == "out"
    assert tool.git(["status"], tmp_path).returncode == 0
    assert tool.docker("status").returncode == 0
    monkeypatch.setattr("httpx.request", lambda *a, **k: "response")
    assert tool.http("GET", "http://localhost") == "response"
    with pytest.raises(PermissionError):
        tool.http("GET", "https://outside.test")
    assert len(calls) == 2
    assert "core.hooksPath=/dev/null" in calls[1]


def test_docker_actions_require_correct_permissions_and_use_broker(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "harness.docker_broker.DockerComposeBroker.run",
        lambda _broker, action, tail=100: (
            calls.append((action, tail))
            or subprocess.CompletedProcess(["docker", action], 0, "", "")
        ),
    )
    writable = Config()
    writable.data["tools"] = {"permissions": {"docker": "write"}}
    executor = ToolExecutor(Permissions(writable))
    for action in ("status", "logs", "start", "stop"):
        assert executor.docker(action, tail=25).returncode == 0
    assert calls == [(action, 25) for action in ("status", "logs", "start", "stop")]
    with pytest.raises(PermissionError, match="not allowlisted"):
        executor.docker("exec")

    readonly = Config()
    readonly.data["tools"] = {"permissions": {"docker": "read"}}
    with pytest.raises(PermissionError, match="not permitted"):
        ToolExecutor(Permissions(readonly)).docker("start")
    assert ToolExecutor(Permissions(readonly)).docker("status").returncode == 0


def test_registry_schema_events_and_failure(tmp_path):
    events = []
    tool = ToolRegistry(
        Permissions(Config()), lambda kind, payload: events.append(kind)
    )
    tool.register(
        ToolSpec(
            "value",
            "return value",
            {"type": "object", "required": ["value"]},
            "filesystem",
            "READ",
            lambda value: value,
        )
    )
    assert tool.list()[-1] == "value" and tool.execute("value", {"value": 3}) == 3
    with pytest.raises(ValueError):
        tool.execute("value", {})
    with pytest.raises(KeyError):
        tool.execute("unknown", {})
    tool.register(
        ToolSpec(
            "failure",
            "failure",
            {"type": "object"},
            "filesystem",
            "READ",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )
    with pytest.raises(RuntimeError):
        tool.execute("failure", {})
    assert events == [
        "TOOL_CALL_STARTED",
        "TOOL_CALL_COMPLETED",
        "TOOL_CALL_STARTED",
        "TOOL_CALL_FAILED",
    ]


def test_registry_denies_write_without_permission():
    config = Config()
    config.data["tools"] = {"permissions": {"filesystem": "read"}}
    registry = ToolRegistry(Permissions(config))
    registry.register(
        ToolSpec(
            "write", "write", {"type": "object"}, "filesystem", "WRITE", lambda: None
        )
    )
    with pytest.raises(PermissionError):
        registry.execute("write", {})


def test_registry_confines_process_tools_to_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config = Config()
    config.data["tools"] = {
        "permissions": {"shell": "write", "git": "write", "docker": "write"}
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    observed = []
    docker_actions = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda *args, **kwargs: (
            observed.append(kwargs.get("cwd"))
            or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )
    registry.execute("shell.execute", {"command": ["pwd"]})
    registry.execute("git.execute", {"args": ["status"]})
    monkeypatch.setattr(
        "harness.docker_broker.DockerComposeBroker.run",
        lambda _broker, action, tail=100: (
            docker_actions.append((action, tail))
            or subprocess.CompletedProcess(["docker", action], 0, "", "")
        ),
    )
    registry.execute("docker.execute", {"action": "status"})
    assert observed == [workspace.resolve()] * 2
    assert docker_actions == [("status", 100)]
    with pytest.raises(PermissionError, match="escapes"):
        registry.execute("shell.execute", {"command": ["pwd"], "cwd": str(outside)})
    with pytest.raises(PermissionError, match="escapes"):
        registry.execute("git.execute", {"args": ["status"], "cwd": "../outside"})
    (workspace / "file.txt").write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        registry.execute("shell.execute", {"command": ["pwd"], "cwd": "file.txt"})


@pytest.mark.parametrize(
    "tool,args",
    [
        ("filesystem.write", {"path": ".git/config", "content": "unsafe"}),
        ("filesystem.create", {"path": ".git/hooks/post-commit", "content": "unsafe"}),
        ("filesystem.delete", {"path": ".git/refs/heads/main"}),
    ],
)
def test_registry_forbids_direct_mutation_inside_git_metadata(tmp_path, tool, args):
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "write", "filesystem.delete": "write"}
    }
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    with pytest.raises(PermissionError, match="Git metadata"):
        registry.execute(tool, args)


def test_filesystem_copy_cannot_target_git_metadata(tmp_path):
    config = Config()
    config.data["tools"] = {"permissions": {"filesystem": "write"}}
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    (tmp_path / "source.txt").write_text("content")
    with pytest.raises(PermissionError, match="Git metadata"):
        registry.execute(
            "filesystem.copy",
            {"path": "source.txt", "destination": ".git/config"},
        )


def test_shell_blocks_direct_git_but_allows_other_commands(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data["tools"] = {"permissions": {"shell": "write"}}
    tool = ToolExecutor(Permissions(cfg))
    calls = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: (
            calls.append(args) or subprocess.CompletedProcess(args, 0, "ok", "")
        ),
    )
    assert tool.shell(["echo", "safe"], tmp_path).stdout == "ok"
    for command in (
        ["git", "branch", "-D", "topic"],
        ["/opt/homebrew/bin/git", "branch", "-d", "topic"],
        "echo start && git branch -D topic",
    ):
        with pytest.raises(PermissionError, match="policy-controlled Git tool"):
            tool.shell(command, tmp_path)
    assert calls[0][3:] == ["echo", "safe"]


def test_http_secret_and_git_approval(monkeypatch):
    from harness.approvals import ApprovalGrant, GitApprovalTarget

    cfg = Config()
    cfg.data["tools"] = {"permissions": {"http": "write", "git": "write"}}
    tool = ToolExecutor(Permissions(cfg), http_allow_hosts=["localhost"])
    monkeypatch.setattr(
        "harness.security.SecretResolver",
        lambda: type("S", (), {"get": lambda self, name: "token"})(),
    )
    seen = {}
    monkeypatch.setattr(
        "httpx.request", lambda method, url, **kwargs: seen.update(kwargs) or "ok"
    )
    assert tool.http("GET", "http://localhost/path", secret_name="API_KEY") == "ok"
    assert seen["headers"]["Authorization"] == "Bearer token"
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    with pytest.raises(PermissionError):
        tool.git(["branch", "-d", "topic"], ".")
    with pytest.raises(PermissionError, match="recorded human approval"):
        tool.git(
            ["branch", "-d", "topic"],
            ".",
            {"approved": True, "action": "branch.delete"},
        )
    from harness import approvals

    target = GitApprovalTarget(1, ".", ("branch", "-d", "topic"))
    grant = ApprovalGrant._issue(1, "branch.delete", 7, approvals._SEAL, target)
    assert tool.git(["branch", "-d", "topic"], ".", grant, task_id=1).returncode == 0
    with pytest.raises(PermissionError, match="recorded human approval"):
        tool.git(["branch", "-d", "topic"], ".", grant, task_id=1)


def test_http_denies_missing_secret_and_bad_scheme(monkeypatch):
    cfg = Config()
    cfg.data["tools"] = {"permissions": {"http": "write"}}
    tool = ToolExecutor(Permissions(cfg), http_allow_hosts=["localhost"])
    monkeypatch.setattr(
        "harness.security.SecretResolver",
        lambda: type("S", (), {"get": lambda self, name: None})(),
    )
    with pytest.raises(PermissionError, match="secret"):
        tool.http("GET", "http://localhost", secret_name="MISSING")
    with pytest.raises(PermissionError, match="allowlisted"):
        tool.http("GET", "file://localhost/etc/passwd")


def test_shell_wrappers_and_registry_without_sink(monkeypatch, tmp_path):
    tool = executor(("shell", "write"), ("git", "read"), ("filesystem", "read"))
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "ok", ""),
    )
    assert tool.test(["pytest"], tmp_path).stdout == "ok"
    assert tool.lint(["ruff"], tmp_path).stdout == "ok"
    registry = ToolRegistry(Permissions(Config()), workspace=tmp_path)
    assert registry.list()
    assert registry.git_status(str(tmp_path)) == "ok"
    registry.register(
        ToolSpec(
            "no-sink", "test", {"type": "object"}, "filesystem", "READ", lambda: "ok"
        )
    )
    assert registry.execute("no-sink", {}) == "ok"
    registry.register(
        ToolSpec(
            "no-sink-error",
            "test",
            {"type": "object"},
            "filesystem",
            "READ",
            lambda: (_ for _ in ()).throw(RuntimeError("no sink")),
        )
    )
    with pytest.raises(RuntimeError):
        registry.execute("no-sink-error", {})
