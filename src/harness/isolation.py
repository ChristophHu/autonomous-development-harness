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


def isolated_command(command, cwd, *, git=False):
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
        f"(allow file-write* (subpath {json.dumps(str(root))}))",
        '(allow file-write* (literal "/dev/null"))',
    ]
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
