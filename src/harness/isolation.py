"""Inherited macOS kernel restrictions for untrusted tool processes.

No fallback to unrestricted execution: unavailable or rejected sandbox activation
is a failed tool call. Profiles are passed directly as argv, never via a writable
profile file. The kernel applies restrictions to descendants and resolved paths.
"""

import json
import shutil
import sys
from pathlib import Path


def git_metadata(root):
    """Include linked worktrees, symlinks, nested and bare repositories."""
    root = Path(root).resolve()
    markers = set(root.rglob(".git")) | {
        parent / ".git" for parent in (root, *root.parents)
    }
    protected = set()
    for marker in markers:
        if marker.exists():
            protected.add(marker.resolve())
            if marker.is_file():
                text = marker.read_text().strip()
                if not text.startswith("gitdir: "):
                    raise PermissionError("invalid Git metadata pointer")
                directory = (marker.parent / text[8:]).resolve()
                protected.add(directory)
                common = directory / "commondir"
                if common.is_file():
                    protected.add((directory / common.read_text().strip()).resolve())
    for head in root.rglob("HEAD"):
        if (head.parent / "objects").is_dir() and (head.parent / "refs").is_dir():
            protected.add(head.parent.resolve())
    shared = set()
    for metadata in protected:
        files = metadata.rglob("*") if metadata.is_dir() else (metadata,)
        for file in files:
            if file.is_file():
                info = file.stat()
                if info.st_nlink > 1:
                    shared.add((info.st_dev, info.st_ino))
    for file in root.rglob("*"):
        if file.is_file():
            info = file.stat()
            if (info.st_dev, info.st_ino) in shared:
                protected.add(file.resolve())
    return protected


def trusted_git_metadata(cwd):
    """Expand native Git writes only for a verified repository/worktree link."""
    root = Path(cwd).resolve()
    if (
        (root / "HEAD").is_file()
        and (root / "objects").is_dir()
        and (root / "refs").is_dir()
    ):
        return (root,)
    for parent in (root, *root.parents):
        marker = parent / ".git"
        if marker.is_symlink():
            raise PermissionError("native Git metadata symlinks are prohibited")
        if marker.is_dir():
            return (marker,)
        if marker.is_file():
            pointer = marker.read_text().strip()
            if not pointer.startswith("gitdir: "):
                raise PermissionError("invalid Git worktree pointer")
            directory = (parent / pointer[8:]).resolve(strict=True)
            common = (
                directory / (directory / "commondir").read_text().strip()
            ).resolve(strict=True)
            backlink = Path((directory / "gitdir").read_text().strip()).resolve()
            if (
                directory.parent != common / "worktrees"
                or backlink != marker
                or not all(
                    (common / name).exists() for name in ("HEAD", "objects", "refs")
                )
            ):
                raise PermissionError("unverified Git worktree metadata")
            return (directory, common)
    return ()


def isolated_command(
    command,
    cwd,
    *,
    git=False,
    read_only=False,
    network_proxy=None,
    git_helpers=(),
    network_remotes=(),
    unix_sockets=(),
    git_shell=False,
):
    if sys.platform != "darwin":
        raise PermissionError("process isolation requires macOS sandbox-exec")
    root = Path(cwd or Path.cwd()).resolve(strict=True)
    if root == Path(root.anchor):
        raise PermissionError("filesystem root cannot be an isolated workspace")
    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow file-read*)",
        '(allow file-write* (literal "/dev/null"))',
    ]
    if not read_only:
        rules.append(f"(allow file-write* (subpath {json.dumps(str(root))}))")
    if git:
        # Never trust task/profile PATH or an executable written by a task.
        executable = shutil.which("git", path="/opt/homebrew/bin:/usr/bin:/bin")
        if not executable:
            raise PermissionError("Git executable is unavailable")
        command = [str(Path(executable).resolve()), *command[1:]]
        if Path(command[0]).is_relative_to(root):
            raise PermissionError(
                "Git executable must be outside the writable workspace"
            )
        rules.extend(
            [
                "(deny process-exec)",
                f"(allow process-exec (literal {json.dumps(command[0])}))",
            ]
        )
        for helper in git_helpers:
            helper = Path(helper).resolve(strict=True)
            rules.append(f"(allow process-exec (literal {json.dumps(str(helper))}))")
        if git_shell:
            # Git runs the fixed, quoted SSH command through /bin/sh (bash on
            # macOS). OpenDirectory is needed for getpwuid inside that sandbox.
            rules.extend(
                [
                    '(allow process-exec (literal "/bin/sh"))',
                    '(allow process-exec (literal "/bin/bash"))',
                    '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))',
                ]
            )
        if network_proxy is not None:
            if not isinstance(network_proxy, int) or not 1 <= network_proxy <= 65535:
                raise ValueError("network proxy port is invalid")
            rules.append("(allow system-socket (socket-domain AF_INET))")
            rules.append(
                f'(allow network-outbound (remote ip "localhost:{network_proxy}"))'
            )
        for host, port in network_remotes:
            if (
                not isinstance(host, str)
                or not host
                or not isinstance(port, int)
                or not 1 <= port <= 65535
            ):
                raise ValueError("network remote is invalid")
            target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            rules.append(f'(allow network-outbound (remote ip "{target}"))')
        if unix_sockets:
            rules.append("(allow system-socket (socket-domain AF_UNIX))")
            for socket_path in unix_sockets:
                try:
                    socket_path = Path(socket_path).resolve(strict=True)
                except OSError as error:
                    raise PermissionError(
                        "allowed Unix socket is unavailable"
                    ) from error
                if not socket_path.is_socket():
                    raise PermissionError("allowed Unix socket is unavailable")
                rules.append(
                    "(allow network-outbound (remote unix-socket "
                    f"(literal {json.dumps(str(socket_path))})))"
                )
        if not read_only:
            for path in trusted_git_metadata(root):
                rules.append(f"(allow file-write* (subpath {json.dumps(str(path))}))")
    else:
        rules.extend(
            [
                '(deny file-write* (regex #"(^|/)\\.git(/|$)"))',
                '(deny process-exec (regex #"(^|/)git$"))',
            ]
        )
        for path in sorted(git_metadata(root)):
            rules.append(f"(deny file-write* (subpath {json.dumps(str(path))}))")
    profile = "\n".join(rules)
    return ["/usr/bin/sandbox-exec", "-p", profile, *command]
