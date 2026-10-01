import os
import shutil
import socket
import subprocess
import sys
import tempfile

import pytest

from harness import isolation
from harness.isolation import _stat_if_present, git_metadata, isolated_command


def test_profile_confines_descendants_and_metadata(tmp_path):
    command = isolated_command([sys.executable, "-c", "pass"], tmp_path)
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "deny file-write*" in command[2]
    assert "deny process-exec" in command[2]
    assert command[3:] == [sys.executable, "-c", "pass"]


def test_stat_of_transient_workspace_file_can_race_with_sqlite_journal_cleanup(
    tmp_path,
):
    assert _stat_if_present(tmp_path / "db.sqlite-journal") is None


def test_git_metadata_skips_files_removed_during_both_snapshot_passes(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    metadata = workspace / ".git"
    metadata.mkdir(parents=True)
    protected_transient = metadata / "protected-transient"
    protected_transient.write_text("short-lived")
    workspace_transient = workspace / "db.sqlite-journal"
    workspace_transient.write_text("short-lived")
    original = isolation._stat_if_present
    calls = {protected_transient: 0}

    def racing_stat(path):
        if path == protected_transient and calls[path] == 0:
            calls[path] += 1
            return None
        if path == workspace_transient:
            return None
        return original(path)

    monkeypatch.setattr(isolation, "_stat_if_present", racing_stat)
    assert git_metadata(workspace) == {metadata}


def test_restricted_mcp_profile_lists_read_roots_without_global_read(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = isolated_command(
        [sys.executable, "-c", "pass"],
        workspace,
        read_only=True,
        read_roots=(workspace,),
    )
    assert '(deny file-read-data (subpath "/Users"))' in command[2]
    assert '(deny file-read-data (subpath "/private/var/folders"))' in command[2]
    assert "(allow file-read-data (subpath " in command[2]
    assert str(workspace) in command[2]
    with pytest.raises(PermissionError):
        isolated_command(
            [sys.executable], workspace, read_roots=(tmp_path / "missing",)
        )
    with pytest.raises(PermissionError):
        isolated_command([sys.executable], workspace, read_roots=("relative",))


def test_native_restricted_mcp_profile_denies_outside_read(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "allowed.txt").write_text("allowed")
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    code = (
        "from pathlib import Path; "
        "print(Path('allowed.txt').read_text()); "
        f"Path({str(outside)!r}).read_text()"
    )
    result = subprocess.run(
        isolated_command(
            [sys.executable, "-c", code],
            workspace,
            read_only=True,
            read_roots=(workspace, sys.prefix, sys.base_prefix),
        ),
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout.strip() == "allowed"
    assert "secret" not in result.stdout


def test_unsupported_host_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.isolation.sys.platform", "linux")
    with pytest.raises(PermissionError, match="macOS"):
        isolated_command(["true"], tmp_path)


def test_missing_git_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.isolation.shutil.which", lambda *a, **k: None)
    with pytest.raises(PermissionError, match="unavailable"):
        isolated_command(["git", "status"], tmp_path, git=True)


def test_broad_workspace_and_mutable_git_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match="filesystem root"):
        isolated_command(["true"], "/")
    executable = tmp_path / "git"
    executable.touch()
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *a, **k: str(executable)
    )
    with pytest.raises(PermissionError, match="outside"):
        isolated_command(["git", "status"], tmp_path, git=True)


def test_profile_path_cannot_replace_trusted_git(tmp_path, monkeypatch):
    malicious = tmp_path / "git"
    malicious.write_text("#!/bin/sh\necho bypass\n")
    malicious.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    command = isolated_command(["git", "status"], tmp_path, git=True)
    assert command[3] != str(malicious)


def test_worktree_and_bare_metadata_are_protected(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    metadata = workspace / "metadata"
    metadata.mkdir()
    common = workspace / "common"
    common.mkdir()
    (metadata / "commondir").write_text("../common")
    (workspace / ".git").write_text("gitdir: metadata")
    bare = workspace / "bare"
    bare.mkdir()
    (bare / "HEAD").write_text("ref: refs/heads/main")
    (bare / "objects").mkdir()
    (bare / "refs").mkdir()
    (workspace / "HEAD").write_text("not a repo")
    incomplete = workspace / "incomplete"
    incomplete.mkdir()
    (incomplete / "HEAD").write_text("not a repo")
    (incomplete / "objects").mkdir()
    assert {metadata, common, bare} <= git_metadata(workspace)
    profile = isolated_command(["true"], workspace)[2]
    assert str(common) in profile
    (metadata / "commondir").unlink()
    assert metadata in git_metadata(workspace)
    (workspace / ".git").write_text("malformed")
    with pytest.raises(PermissionError, match="pointer"):
        isolated_command(["true"], workspace)


def test_default_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert str(tmp_path) in isolated_command(["true"], None)[2]


@pytest.mark.parametrize(
    "attack",
    [
        "from pathlib import Path; Path('.git/config').write_text('corrupt')",
        "import subprocess; subprocess.run(['/opt/homebrew/bin/git', 'branch', '-D', 'topic'], check=True)",
        "from pathlib import Path; Path('../outside').write_text('escape')",
        "import subprocess,sys; subprocess.run([sys.executable,'-c',\"from pathlib import Path; Path('.git/config').write_text('child')\"],check=True)",
    ],
)
def test_real_sandbox_blocks_attacks(tmp_path, attack):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git/config").write_text("original")
    result = subprocess.run(
        isolated_command([sys.executable, "-c", attack], root),
        cwd=root,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert (root / ".git/config").read_text() == "original"
    assert not (tmp_path / "outside").exists()


def test_real_sandbox_allows_workspace_work(tmp_path):
    result = subprocess.run(
        isolated_command(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('output').write_text('ok')",
            ],
            tmp_path,
        ),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text() == "ok"


def test_copied_git_cannot_delete_bare_branch(tmp_path):
    git = shutil.which("git", path="/opt/homebrew/bin:/usr/bin")
    repo = tmp_path / "bare"
    subprocess.run([git, "init", "--bare", str(repo)], check=True, capture_output=True)
    tree = subprocess.run(
        [git, "--git-dir", str(repo), "mktree"],
        input="",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    commit = subprocess.run(
        [
            git,
            "--git-dir",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit-tree",
            tree,
            "-m",
            "seed",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    ref = repo / "refs/heads/topic"
    ref.write_text(commit + "\n")
    copied = tmp_path / "innocent"
    shutil.copy2(git, copied)
    result = subprocess.run(
        isolated_command(
            [str(copied), "--git-dir", str(repo), "branch", "-D", "topic"], tmp_path
        ),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert ref.exists()
    assert b"Operation not permitted" in result.stderr


@pytest.mark.parametrize(
    "target",
    [
        "alias/config",
        "metadata/config",
        "common/refs/heads/topic",
        ".git",
        "../outside",
    ],
)
def test_real_pointer_symlink_and_rename_protection(tmp_path, target):
    root = tmp_path / "project"
    root.mkdir()
    metadata = root / "metadata"
    metadata.mkdir()
    common = root / "common"
    (common / "refs/heads").mkdir(parents=True)
    (metadata / "commondir").write_text("../common")
    (root / ".git").write_text("gitdir: metadata")
    (root / "alias").symlink_to(metadata, target_is_directory=True)
    (metadata / "config").write_text("original")
    (common / "refs/heads/topic").write_text("original")
    code = f"from pathlib import Path; Path({target!r}).write_text('corrupt')"
    result = subprocess.run(
        isolated_command([sys.executable, "-c", code], root),
        cwd=root,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert (metadata / "config").read_text() == "original"
    assert (common / "refs/heads/topic").read_text() == "original"


def test_git_filter_child_is_denied(tmp_path):
    git = shutil.which("git", path="/opt/homebrew/bin:/usr/bin")
    for args in (
        ["init"],
        [
            "config",
            "filter.attack.clean",
            f"{sys.executable} -c \"from pathlib import Path; Path('.git/HEAD').write_text('corrupt')\"",
        ],
        ["config", "filter.attack.required", "true"],
    ):
        subprocess.run([git, *args], cwd=tmp_path, check=True, capture_output=True)
    original = (tmp_path / ".git/HEAD").read_text()
    (tmp_path / ".gitattributes").write_text("*.txt filter=attack\n")
    (tmp_path / "file.txt").write_text("input")
    result = subprocess.run(
        isolated_command(
            [git, "-c", "core.hooksPath=/dev/null", "add", "file.txt"],
            tmp_path,
            git=True,
        ),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert (tmp_path / ".git/HEAD").read_text() == original


def test_network_and_daemon_access_are_denied(tmp_path):
    code = "import socket; socket.socket().connect(('127.0.0.1',9))"
    result = subprocess.run(
        isolated_command([sys.executable, "-c", code], tmp_path),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"Operation not permitted" in result.stderr


def test_sandbox_activation_failure_is_not_retried_unsafely(tmp_path, monkeypatch):
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    config = Config()
    config.data["tools"] = {"permissions": {"shell": "write"}}
    calls = []

    def denied(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 71, "", "sandbox_apply: Operation not permitted"
        )

    monkeypatch.setattr("harness.tools.subprocess.run", denied)
    result = ToolExecutor(Permissions(config)).shell(["true"], tmp_path)
    assert result.returncode == 71
    assert len(calls) == 1 and calls[0][0] == "/usr/bin/sandbox-exec"


def test_profiles_cannot_enable_environment_injection(monkeypatch):
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    names = (
        "DYLD_INSERT_LIBRARIES",
        "LD_PRELOAD",
        "GIT_CONFIG_COUNT",
        "SAFE_TEST_VALUE",
    )
    for name in names:
        monkeypatch.setenv(name, "injected")
    env = ToolExecutor(Permissions(Config()), env_allowlist=names)._environment()
    assert env["SAFE_TEST_VALUE"] == "injected"
    assert all(name not in env for name in names[:-1])


def test_filesystem_guards_real_metadata_aliases_and_parent(tmp_path):
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    config = Config()
    config.data["tools"] = {"permissions": {"filesystem": "write"}}
    tool = ToolExecutor(Permissions(config))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (tmp_path / ".git").symlink_to(metadata, target_is_directory=True)
    (metadata / "config").write_text("original")
    (tmp_path / "source").write_text("safe")
    for path in ("metadata/config", "metadata", "."):
        with pytest.raises(PermissionError, match="Git metadata"):
            tool.filesystem("write", path, workspace=tmp_path, content="corrupt")
    with pytest.raises(PermissionError, match="Git metadata"):
        tool.filesystem(
            "copy", "source", workspace=tmp_path, destination="metadata/config"
        )
    assert (metadata / "config").read_text() == "original"


def test_real_git_hooks_are_disabled(tmp_path):
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    git = shutil.which("git", path="/opt/homebrew/bin:/usr/bin")
    for args in (
        ["init"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.invalid"],
    ):
        subprocess.run([git, *args], cwd=tmp_path, check=True, capture_output=True)
    hook = tmp_path / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\nprintf corrupt > .git/HEAD\n")
    hook.chmod(0o755)
    (tmp_path / "file").write_text("content")
    subprocess.run([git, "add", "file"], cwd=tmp_path, check=True, capture_output=True)
    original = (tmp_path / ".git/HEAD").read_text()
    config = Config()
    config.data["tools"] = {"permissions": {"git": "write"}}
    result = ToolExecutor(Permissions(config)).git(["commit", "-m", "safe"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".git/HEAD").read_text() == original


def test_real_daemon_socket_access_is_denied(tmp_path):
    with (
        tempfile.TemporaryDirectory(prefix="h1c-", dir="/tmp") as socket_dir,
        socket.socket(socket.AF_UNIX) as server,
    ):
        endpoint = f"{socket_dir}/daemon.sock"
        server.bind(str(endpoint))
        server.listen()
        code = (
            f"import socket; socket.socket(socket.AF_UNIX).connect({str(endpoint)!r})"
        )
        result = subprocess.run(
            isolated_command([sys.executable, "-c", code], tmp_path),
            cwd=tmp_path,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert b"Operation not permitted" in result.stderr


def test_real_metadata_rename_is_denied(tmp_path):
    (tmp_path / ".git").mkdir()
    result = subprocess.run(
        isolated_command(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('.git').rename('disguised')",
            ],
            tmp_path,
        ),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert (tmp_path / ".git").is_dir()


def test_real_metadata_hardlink_creation_is_denied(tmp_path):
    (tmp_path / ".git").mkdir()
    target = tmp_path / ".git/config"
    target.write_text("original")
    code = "import os; from pathlib import Path; os.link('.git/config','alias'); Path('alias').write_text('corrupt')"
    result = subprocess.run(
        isolated_command([sys.executable, "-c", code], tmp_path),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert target.read_text() == "original"


def test_existing_metadata_hardlink_is_protected(tmp_path):
    (tmp_path / ".git").mkdir()
    target = tmp_path / ".git/config"
    target.write_text("original")
    os.link(target, tmp_path / "alias")
    code = "from pathlib import Path; Path('alias').write_text('corrupt')"
    result = subprocess.run(
        isolated_command([sys.executable, "-c", code], tmp_path),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert target.read_text() == "original"
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    config = Config()
    config.data["tools"] = {"permissions": {"filesystem": "write"}}
    with pytest.raises(PermissionError, match="Git metadata"):
        ToolExecutor(Permissions(config)).filesystem(
            "write", "alias", workspace=tmp_path, content="corrupt"
        )
