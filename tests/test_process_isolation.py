import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from harness import isolation
from harness.isolation import (
    ProcessAccessProfile,
    _stat_if_present,
    git_metadata,
    isolated_command,
    trusted_git_metadata,
)


def test_profile_confines_descendants_and_metadata(tmp_path):
    command = isolated_command([sys.executable, "-c", "pass"], tmp_path)
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "deny file-write*" in command[2]
    assert "deny process-exec" in command[2]
    assert '(allow file-read-data (literal "/dev/null"))' in command[2]
    assert command[3:] == [str(Path(sys.executable).resolve()), "-c", "pass"]


def test_profile_allows_executable_mapping_only_for_runtime_paths(tmp_path):
    command = isolated_command([sys.executable, "-c", "pass"], tmp_path)
    profile = command[2]

    assert '(allow file-map-executable (subpath "/System"))' in profile
    assert (
        f'(allow file-map-executable (subpath "{Path(sys.base_prefix).resolve()}"))'
        in profile
    )
    assert f'(allow file-map-executable (subpath "{tmp_path}"))' not in profile
    assert "(allow file-map-executable)\n" not in profile


def test_git_profile_grants_only_transitive_runtime_dylibs(tmp_path, monkeypatch):
    if not Path("/opt/homebrew/bin/git").exists():
        pytest.skip("Homebrew Git is not installed")
    monkeypatch.setattr(isolation.sys, "platform", "darwin")

    profile = isolated_command(["git", "--version"], tmp_path, git=True)[2]

    pcre = "/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib"
    gettext = "/opt/homebrew/opt/gettext/lib/libintl.8.dylib"
    for dylib in (pcre, gettext):
        assert f'(allow file-read* (literal "{dylib}"))' in profile
        assert f'(allow file-map-executable (literal "{dylib}"))' in profile
    assert '(allow file-read* (subpath "/opt/homebrew"))' not in profile
    assert '(allow file-map-executable (subpath "/opt/homebrew"))' not in profile


def test_git_runtime_dylibs_walks_transitive_dependencies_and_cycles(
    tmp_path, monkeypatch
):
    binary = tmp_path / "git"
    first = tmp_path / "libfirst.dylib"
    second = tmp_path / "libsecond.dylib"
    for path in (binary, first, second):
        path.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(16))
    alias = tmp_path / "libfirst-alias.dylib"
    alias.symlink_to(first)
    dependencies = {
        binary.resolve(): [alias, Path("/usr/lib/libSystem.B.dylib")],
        first.resolve(): [second],
        second.resolve(): [first],
    }

    def run(command, **_kwargs):
        executable = Path(command[-1])
        lines = [f"{executable}:", ""]
        lines.extend(
            f"\t{path} (compatibility version 1.0.0, current version 1.0.0)"
            for path in dependencies[executable]
        )
        lines.append("")
        return subprocess.CompletedProcess(command, 0, "\n".join(lines), "")

    monkeypatch.setattr(isolation, "_OTOOL_RUN", run)

    result = isolation._git_runtime_dylibs((binary,))

    assert set(result) == {alias, first.resolve(), second.resolve()}


def test_git_runtime_dylibs_skips_non_macho_helpers(tmp_path):
    helper = tmp_path / "git-helper"
    helper.write_text("#!/bin/sh\nexit 0\n")

    assert isolation._git_runtime_dylibs((helper,)) == ()


def test_git_runtime_dylibs_rejects_unreadable_executable(tmp_path, monkeypatch):
    binary = tmp_path / "git"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(16))
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        if path == binary:
            raise OSError("inaccessible")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)

    with pytest.raises(PermissionError, match="runtime executable is unavailable"):
        isolation._git_runtime_dylibs((binary,))


@pytest.mark.parametrize(
    ("dependency", "message"),
    [
        ("@rpath/libmissing.dylib", "path is unresolved"),
        ("/no/such/libmissing.dylib", "dependency is unavailable"),
    ],
)
def test_git_runtime_dylibs_rejects_unresolved_dependencies(
    tmp_path, monkeypatch, dependency, message
):
    binary = tmp_path / "git"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(16))
    output = (
        f"{binary}:\n\t{dependency} "
        "(compatibility version 1.0.0, current version 1.0.0)\n"
    )
    monkeypatch.setattr(
        isolation,
        "_OTOOL_RUN",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    with pytest.raises(PermissionError, match=message):
        isolation._git_runtime_dylibs((binary,))


def test_git_runtime_dylibs_rejects_dependency_that_is_not_a_file(
    tmp_path, monkeypatch
):
    binary = tmp_path / "git"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(16))
    dependency = tmp_path / "not-a-dylib"
    dependency.mkdir()
    output = (
        f"{binary}:\n\t{dependency} "
        "(compatibility version 1.0.0, current version 1.0.0)\n"
    )
    monkeypatch.setattr(
        isolation,
        "_OTOOL_RUN",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    with pytest.raises(PermissionError, match="dependency is unavailable"):
        isolation._git_runtime_dylibs((binary,))


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "otool"),
        FileNotFoundError("otool is missing"),
    ],
)
def test_git_runtime_dylibs_fails_closed_if_otool_fails(tmp_path, monkeypatch, failure):
    binary = tmp_path / "git"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(16))
    monkeypatch.setattr(
        isolation,
        "_OTOOL_RUN",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(PermissionError, match="cannot be inspected"):
        isolation._git_runtime_dylibs((binary,))


def test_run_otool_raises_with_captured_output_on_nonzero_exit(monkeypatch):
    class FakeProcess:
        returncode = 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def communicate(self):
            return "", "inspection failed"

    monkeypatch.setattr(isolation, "_OTOOL_POPEN", lambda *_a, **_k: FakeProcess())
    with pytest.raises(subprocess.CalledProcessError) as error:
        isolation._run_otool(("/usr/bin/otool", "-L", "/usr/bin/git"))
    assert error.value.stderr == "inspection failed"


def test_git_profile_rejects_runtime_root_that_disappears_during_build(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(isolation.sys, "platform", "darwin")
    original_resolve = Path.resolve
    system_root_calls = 0

    def resolve(path, *args, **kwargs):
        nonlocal system_root_calls
        if path == Path("/System"):
            system_root_calls += 1
            if system_root_calls == 2:
                raise OSError("runtime path disappeared")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(PermissionError, match="runtime root is unavailable"):
        isolated_command([sys.executable, "-c", "pass"], tmp_path)


def test_profile_allows_reading_root_directory_entry_for_runtime_startup(tmp_path):
    profile = isolated_command([sys.executable, "-c", "pass"], tmp_path)[2]

    assert '(allow file-read-data (literal "/"))' in profile


def test_profile_does_not_map_a_venv_inside_the_writable_workspace(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    virtualenv = workspace / ".venv"
    virtualenv.mkdir(parents=True)
    monkeypatch.setattr(isolation.sys, "prefix", str(virtualenv))

    profile = isolated_command([sys.executable, "-c", "pass"], workspace)[2]

    assert f'(allow file-read* (subpath "{virtualenv}"))' in profile
    assert f'(allow file-map-executable (subpath "{virtualenv}"))' not in profile


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
    assert "(allow file-read*)\n" not in command[2]
    assert "(allow file-read* (subpath " in command[2]
    assert str(workspace) in command[2]
    with pytest.raises(PermissionError):
        isolated_command(
            [sys.executable], workspace, read_roots=(tmp_path / "missing",)
        )
    with pytest.raises(PermissionError):
        isolated_command([sys.executable], workspace, read_roots=("relative",))


def test_process_profile_adds_validated_explicit_write_roots(tmp_path):
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()

    command = isolated_command(
        [sys.executable, "-c", "pass"],
        workspace,
        read_roots=(workspace,),
        write_roots=(temporary,),
    )

    assert f'(allow file-write* (subpath "{temporary}"))' in command[2]
    with pytest.raises(PermissionError, match="absolute"):
        isolated_command([sys.executable], workspace, write_roots=("relative",))
    with pytest.raises(PermissionError, match="unavailable"):
        isolated_command(
            [sys.executable], workspace, write_roots=(tmp_path / "missing",)
        )


def test_process_access_profile_scopes_read_write_exec_and_socket_roots(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()
    socket_path = temporary / "broker.sock"
    socket_path.touch()
    monkeypatch.setattr(
        Path, "is_socket", lambda path: path.resolve() == socket_path.resolve()
    )
    executable = Path(sys.executable).resolve()
    profile = ProcessAccessProfile.broker(
        "unit_broker",
        read_roots=(workspace,),
        write_roots=(temporary,),
        executable_paths=(executable,),
        unix_sockets=(socket_path,),
    )

    command = isolated_command(
        [str(executable), "-c", "pass"], workspace, access_profile=profile
    )
    assert "(deny process-exec)" in command[2]
    assert f'(allow process-exec (literal "{executable}"))' in command[2]
    assert f'(allow file-read* (subpath "{workspace}"))' in command[2]
    assert f'(allow file-write* (subpath "{temporary}"))' in command[2]
    assert f'(literal "{socket_path}")' in command[2]
    assert "(allow file-read*)\n" not in command[2]


@pytest.mark.parametrize("socket_path_kind", ["missing", "regular"])
def test_access_profile_rejects_missing_or_non_socket_unix_paths(
    tmp_path, socket_path_kind
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    socket_path = tmp_path / "broker.sock"
    if socket_path_kind == "regular":
        socket_path.touch()
    with pytest.raises(PermissionError, match="Unix socket is unavailable"):
        isolated_command(
            [sys.executable],
            workspace,
            access_profile=ProcessAccessProfile.broker(
                "broker", unix_sockets=(socket_path,)
            ),
        )


def test_access_profile_cannot_be_combined_with_legacy_roots(tmp_path):
    profile = ProcessAccessProfile.workspace()
    with pytest.raises(ValueError, match="either an access profile"):
        isolated_command(
            [sys.executable],
            tmp_path,
            access_profile=profile,
            read_roots=(tmp_path,),
        )


def test_access_profile_rejects_unknown_object(tmp_path):
    with pytest.raises(TypeError, match="ProcessAccessProfile"):
        isolated_command([sys.executable], tmp_path, access_profile=object())


def test_mcp_stdio_profile_normalizes_read_roots(tmp_path):
    profile = ProcessAccessProfile.mcp_stdio(read_roots=(tmp_path,))
    assert profile.name == "mcp_stdio"
    assert profile.read_roots == (tmp_path,)


def test_absolute_executable_must_resolve(tmp_path):
    with pytest.raises(PermissionError, match="executable is unavailable"):
        isolated_command([str(tmp_path / "missing"), "--version"], tmp_path)


def test_trusted_git_metadata_accepts_bare_and_regular_repositories(tmp_path):
    bare = tmp_path / "bare.git"
    (bare / "objects").mkdir(parents=True)
    (bare / "refs").mkdir()
    (bare / "HEAD").write_text("ref: refs/heads/main\n")
    assert trusted_git_metadata(bare) == (bare.resolve(),)

    workspace = tmp_path / "workspace"
    metadata = workspace / ".git"
    metadata.mkdir(parents=True)
    assert trusted_git_metadata(workspace) == (metadata.resolve(),)


def test_trusted_git_metadata_validates_worktree_pointers_and_symlinks(tmp_path):
    common = tmp_path / "repository.git"
    worktree = tmp_path / "worktree"
    metadata = worktree / ".git"
    gitdir = common / "worktrees" / "feature"
    for path in (common / "objects", common / "refs", gitdir):
        path.mkdir(parents=True, exist_ok=True)
    worktree.mkdir()
    (common / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "commondir").write_text("../..\n")
    (gitdir / "gitdir").write_text(f"{metadata}\n")
    metadata.write_text("gitdir: ../repository.git/worktrees/feature\n")
    assert trusted_git_metadata(worktree) == (gitdir.resolve(), common.resolve())

    metadata.unlink()
    metadata.symlink_to(gitdir)
    with pytest.raises(PermissionError, match="symlinks"):
        trusted_git_metadata(worktree)


def test_trusted_git_metadata_rejects_invalid_pointer_and_unverified_worktree(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = workspace / ".git"
    metadata.write_text("not a gitdir pointer")
    with pytest.raises(PermissionError, match="invalid Git worktree pointer"):
        trusted_git_metadata(workspace)

    metadata.write_text("gitdir: missing\n")
    with pytest.raises(OSError):
        trusted_git_metadata(workspace)

    common = tmp_path / "common"
    worktree = tmp_path / "verified-pointer"
    metadata = worktree / ".git"
    gitdir = common / "worktrees" / "entry"
    (common / "refs").mkdir(parents=True)
    gitdir.mkdir(parents=True)
    worktree.mkdir()
    (common / "HEAD").touch()
    (gitdir / "commondir").write_text("../..\n")
    (gitdir / "gitdir").write_text(f"{metadata}\n")
    metadata.write_text("gitdir: ../common/worktrees/entry\n")
    with pytest.raises(PermissionError, match="unverified Git worktree"):
        trusted_git_metadata(worktree)


def test_git_access_profile_contains_fixed_helper_and_network_grants(
    tmp_path, monkeypatch
):
    git = Path("/usr/bin/git")
    helper = Path(sys.executable).resolve()
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *_args, **_kwargs: str(git)
    )
    command = isolated_command(
        ["git", "fetch"],
        tmp_path,
        git=True,
        git_helpers=(helper,),
        git_shell=True,
        network_proxy=3128,
        network_remotes=(("example.invalid", 443), ("::1", 443)),
    )
    profile = command[2]
    assert f'(allow process-exec (literal "{helper}"))' in profile
    assert '(allow process-exec (literal "/bin/sh"))' in profile
    assert (
        '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))'
        in profile
    )
    assert '(allow network-outbound (remote ip "localhost:3128"))' in profile
    assert '(allow network-outbound (remote ip "example.invalid:443"))' in profile
    assert '(allow network-outbound (remote ip "[::1]:443"))' in profile
    read_only = isolated_command(["git", "fetch"], tmp_path, git=True, read_only=True)[
        2
    ]
    assert '(allow process-exec (literal "/usr/bin/git"))' in read_only


def test_git_profile_rejects_relative_helper(tmp_path):
    with pytest.raises(PermissionError, match="must be absolute"):
        isolated_command(
            ["git", "fetch"], tmp_path, git=True, git_helpers=(Path("git-index-pack"),)
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"network_proxy": True}, "proxy port"),
        ({"network_remotes": (("", 443),)}, "network remote"),
        ({"network_remotes": (("example.invalid", 0),)}, "network remote"),
        ({"network_remotes": (("example.invalid", True),)}, "network remote"),
    ],
)
def test_git_profile_rejects_invalid_network_grants(
    tmp_path, monkeypatch, kwargs, message
):
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/git"
    )
    with pytest.raises(ValueError, match=message):
        isolated_command(["git", "fetch"], tmp_path, git=True, **kwargs)


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


def test_shell_tool_denies_reads_outside_workspace_but_allows_workspace(tmp_path):
    from harness.core import Config, Permissions
    from harness.tools import ToolRegistry

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "allowed.txt").write_text("workspace-readable")
    config = Config()
    config.data["tools"] = {
        "permissions": {"shell": "write"},
        "mcp": {"servers": {}},
    }
    registry = ToolRegistry(Permissions(config), workspace=workspace)
    private_path = Path.home()
    code = (
        "from pathlib import Path\n"
        "print(Path('allowed.txt').read_text())\n"
        f"p=Path({str(private_path)!r})\n"
        "try:\n    next(p.iterdir())\nexcept PermissionError:\n    print('outside-denied')\n"
        "except Exception as error:\n    print(type(error).__name__)\n"
        "else:\n    print('outside-readable')"
    )

    result = registry.shell([sys.executable, "-c", code])

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["workspace-readable", "outside-denied"]


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
    process = subprocess.Popen(
        isolated_command(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('output').write_text('ok')",
            ],
            tmp_path,
        ),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0, (
        f"sandbox child pid={process.pid}; stdout={stdout!r}; stderr={stderr!r}"
    )
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
