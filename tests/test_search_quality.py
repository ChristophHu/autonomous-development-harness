import subprocess
import sys

import pytest

from harness.core import Config, Permissions
from harness.search import RepositorySearch
from harness.tools import ToolExecutionError, ToolRegistry


def test_repository_search_text_files_and_symbols(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text(
        "class Widget:\n    def render(self):\n        return 'needle'\n"
    )
    (root / "src" / "view.ts").write_text(
        "export function renderView() { return 'needle'; }\n"
    )
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("needle")
    (root / "outside.py").symlink_to(tmp_path / "secret.py")
    (tmp_path / "secret.py").write_text("needle")
    search = RepositorySearch(root)
    assert search.file_search("*.py") == ["src/module.py"]
    assert [(item["path"], item["line"]) for item in search.text_search("needle")] == [
        ("src/module.py", 3),
        ("src/view.ts", 1),
    ]
    assert [
        (item["name"], item["kind"]) for item in search.symbol_search("render")
    ] == [
        ("render", "function"),
        ("renderView", "function"),
    ]
    assert search.symbol_search("Widget")[0]["kind"] == "class"


def test_repository_search_bounds_and_workspace_scope(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("hit\nhit\n")
    (root / "b.bin").write_bytes(b"hit\x00hit")
    (root / "large.txt").write_text("hit" * 400_000)
    (root / "subdir").mkdir()
    (root / "subdir" / "c.txt").write_text("hit")
    search = RepositorySearch(root)
    assert len(search.text_search("hit", limit=1)) == 1
    assert search.text_search("missing") == []
    assert search.file_search("*.bin") == ["b.bin"]
    assert search.file_search("*.txt", limit=1) == ["a.txt"]
    assert search.text_search("hit", directory="subdir") == [
        {"path": "subdir/c.txt", "line": 1, "text": "hit"}
    ]
    assert search.symbol_search("hit") == []
    with pytest.raises(ValueError, match="limit"):
        search.text_search("hit", limit=0)
    with pytest.raises(ValueError, match="limit"):
        search.file_search("*.txt", limit=True)
    with pytest.raises(ValueError, match="pattern"):
        search.file_search("")
    with pytest.raises(ValueError, match="query"):
        search.text_search("")
    with pytest.raises(ValueError, match="query"):
        search.symbol_search("")
    with pytest.raises(PermissionError, match="escapes"):
        search.file_search("*", directory="../")
    with pytest.raises(NotADirectoryError):
        search.file_search("*", directory="a.txt")


def test_symbol_search_limit_and_source_filter(tmp_path):
    (tmp_path / "a.py").write_text("class Alpha:\n    pass\nclass Alpine:\n    pass\n")
    (tmp_path / "b.txt").write_text("class Alarming:\n")
    search = RepositorySearch(tmp_path)
    assert [item["name"] for item in search.symbol_search("Al", limit=1)] == ["Alpha"]


def test_registered_test_and_quality_actions_use_scoped_processes(
    tmp_path, monkeypatch
):
    config = Config()
    config.data["tools"] = {"permissions": {"shell": "write", "filesystem": "read"}}
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs["cwd"]))
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr("harness.tools.subprocess.run", run)
    names = (
        "test.run_tests",
        "test.run_file",
        "test.run_coverage",
        "quality.lint",
        "quality.format",
        "quality.typecheck",
    )
    for name in names:
        arguments = {"command": ["echo", name]}
        if name == "test.run_file":
            (tmp_path / "test_one.py").write_text("pass\n")
            arguments["path"] = "test_one.py"
        assert registry.execute(name, arguments).stdout == "ok"
    assert len(calls) == 6
    assert all(cwd == tmp_path.resolve() for _args, cwd in calls)
    with pytest.raises(PermissionError, match="escapes"):
        registry.execute("test.run_file", {"command": ["pytest"], "path": "../x.py"})
    with pytest.raises(FileNotFoundError):
        registry.execute("test.run_file", {"command": ["pytest"], "path": "missing.py"})
    with pytest.raises(ValueError, match="invalid tool arguments"):
        registry.execute("quality.format", {"command": []})


def test_registered_search_actions_and_permissions(tmp_path):
    (tmp_path / "x.py").write_text("def example():\n    return 'hello'\n")
    config = Config()
    config.data["tools"] = {"permissions": {"filesystem": "read", "shell": "denied"}}
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    assert registry.execute("search.files", {"pattern": "*.py"}) == ["x.py"]
    assert registry.execute("search.text", {"query": "hello"})[0]["line"] == 2
    assert (
        registry.execute("search.symbols", {"query": "example"})[0]["name"] == "example"
    )
    with pytest.raises(PermissionError):
        registry.execute("quality.lint", {"command": ["echo", "x"]})


def test_real_test_and_quality_process_actions(tmp_path):
    config = Config()
    config.data["tools"] = {"permissions": {"shell": "write"}}
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    (tmp_path / "one.py").write_text("print('file-ok')\n")
    script = [sys.executable, "-I", "-c"]
    assert (
        "file-ok"
        in registry.execute(
            "test.run_file", {"command": [sys.executable, "-I"], "path": "one.py"}
        ).stdout
    )
    assert (
        "suite-ok"
        in registry.execute(
            "test.run_tests", {"command": [*script, "print('suite-ok')"]}
        ).stdout
    )
    assert (
        registry.execute(
            "test.run_coverage",
            {"command": [*script, "open('coverage-artifact', 'w').write('100')"]},
        ).returncode
        == 0
    )
    assert (tmp_path / "coverage-artifact").read_text() == "100"
    assert (
        registry.execute(
            "quality.format",
            {"command": [*script, "open('formatted', 'w').write('ok')"]},
        ).returncode
        == 0
    )
    assert (tmp_path / "formatted").read_text() == "ok"
    for action in ("quality.lint", "quality.typecheck"):
        assert (
            registry.execute(
                action, {"command": [*script, "print('checked')"]}
            ).stdout.strip()
            == "checked"
        )
    with pytest.raises(ToolExecutionError, match="exited with code 3"):
        registry.execute(
            "quality.typecheck", {"command": [*script, "raise SystemExit(3)"]}
        )
