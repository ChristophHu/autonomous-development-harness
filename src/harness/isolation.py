"""Inherited macOS kernel restrictions for untrusted tool processes.

No fallback to unrestricted execution: unavailable or rejected sandbox activation
is a failed tool call. Profiles are passed directly as argv, never via a writable
profile file. The kernel applies restrictions to descendants and resolved paths.
"""

import json
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

_MACHO_MAGICS = {
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
}
_SYSTEM_LIBRARY_PREFIXES = ("/System/", "/usr/lib/")
_OTOOL_POPEN = subprocess.Popen


def _run_otool(command, **_kwargs):
    with _OTOOL_POPEN(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        stdout, stderr = process.communicate()
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


_OTOOL_RUN = _run_otool


def _homebrew_cellar_version(path):
    """Locate the installed formula version containing a trusted runtime path."""
    resolved = Path(path).resolve(strict=True)
    for cellar in (Path("/opt/homebrew/Cellar"), Path("/usr/local/Cellar")):
        if resolved.is_relative_to(cellar):
            parts = resolved.relative_to(cellar).parts
            if len(parts) >= 2:
                return cellar / parts[0] / parts[1]
    return None


def _metadata_path_entries(path):
    """Return exact path entries traversed, following symlink targets."""
    path = Path(path)
    if not path.is_absolute():
        raise PermissionError("process read root must be absolute")
    pending = [path]
    visited = set()
    entries = set()
    while pending:
        current = pending.pop()
        prefix = Path(current.anchor)
        entries.add(prefix)
        for index, part in enumerate(current.parts[1:], start=1):
            prefix = prefix / part
            if prefix in visited:
                continue
            visited.add(prefix)
            entries.add(prefix)
            try:
                if not prefix.is_symlink():
                    continue
                target = prefix.readlink()
            except OSError as error:
                raise PermissionError(
                    "process read path cannot be inspected"
                ) from error
            resolved_target = target if target.is_absolute() else prefix.parent / target
            pending.append(resolved_target.joinpath(*current.parts[index + 1 :]))
    return tuple(sorted(entries, key=str))


def _runtime_dylibs(executables):
    """Resolve non-system dylibs loaded by explicitly launched executables."""
    pending = [Path(path).resolve(strict=True) for path in executables]
    inspected = set()
    allowed = set()
    while pending:
        executable = pending.pop()
        if executable in inspected:
            continue
        inspected.add(executable)
        try:
            with executable.open("rb") as stream:
                magic = stream.read(4)
        except OSError as error:
            raise PermissionError("runtime executable is unavailable") from error
        if magic not in _MACHO_MAGICS:
            continue
        try:
            result = _OTOOL_RUN(
                ("/usr/bin/otool", "-L", str(executable)),
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise PermissionError("runtime dependencies cannot be inspected") from error
        for line in result.stdout.splitlines()[1:]:
            entry = line.strip()
            if not entry:
                continue
            install_name = entry.split(" (compatibility ", 1)[0]
            if install_name.startswith(_SYSTEM_LIBRARY_PREFIXES):
                continue
            dependency = Path(install_name)
            if not dependency.is_absolute():
                raise PermissionError("runtime dependency path is unresolved")
            try:
                resolved = dependency.resolve(strict=True)
            except OSError as error:
                raise PermissionError("runtime dependency is unavailable") from error
            if not resolved.is_file():
                raise PermissionError("runtime dependency is unavailable")
            allowed.update((dependency, resolved))
            pending.append(resolved)
    return tuple(sorted(allowed, key=str))


def _stat_if_present(path):
    try:
        return path.stat()
    except FileNotFoundError:
        # SQLite journals and other transient workspace files may disappear
        # between rglob/is_file and stat while independent tasks run in parallel.
        return None


def _sqlite_extension_for_interpreter(executable):
    """Return CPython's SQLite extension when sandboxing this interpreter."""
    try:
        same_interpreter = Path(executable).resolve(strict=True) == Path(
            sys.executable
        ).resolve(strict=True)
    except OSError:
        return None
    if not same_interpreter:
        return None
    extension_dir = sysconfig.get_config_var("DESTSHARED")
    if not extension_dir:
        return None
    matches = sorted(Path(extension_dir).glob("_sqlite3*.so"))
    return matches[0] if matches else None


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
                info = _stat_if_present(file)
                if info is None:
                    continue
                if info.st_nlink > 1:
                    shared.add((info.st_dev, info.st_ino))
    for file in root.rglob("*"):
        if file.is_file():
            info = _stat_if_present(file)
            if info is None:
                continue
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


@dataclass(frozen=True)
class ProcessAccessProfile:
    """Explicit filesystem and socket permissions for one child-process class."""

    name: str
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    executable_paths: tuple[Path, ...] = ()
    unix_sockets: tuple[Path, ...] = ()
    workspace_writable: bool = True
    write_paths: tuple[Path, ...] = ()

    @classmethod
    def workspace(
        cls, *, read_roots=(), write_roots=(), workspace_writable=True, write_paths=()
    ):
        return cls(
            "workspace",
            tuple(map(Path, read_roots)),
            tuple(map(Path, write_roots)),
            workspace_writable=workspace_writable,
            write_paths=tuple(map(Path, write_paths)),
        )

    @classmethod
    def mcp_stdio(cls, *, read_roots=(), executable_paths=()):
        return cls(
            "mcp_stdio",
            tuple(map(Path, read_roots)),
            executable_paths=tuple(map(Path, executable_paths)),
        )

    @classmethod
    def broker(
        cls,
        name,
        *,
        read_roots=(),
        write_roots=(),
        executable_paths=(),
        unix_sockets=(),
    ):
        return cls(
            name,
            tuple(map(Path, read_roots)),
            tuple(map(Path, write_roots)),
            tuple(map(Path, executable_paths)),
            tuple(map(Path, unix_sockets)),
        )


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
    read_roots=None,
    write_roots=None,
    access_profile=None,
):
    if sys.platform != "darwin":
        raise PermissionError("process isolation requires macOS sandbox-exec")
    root = Path(cwd or Path.cwd()).resolve(strict=True)
    if root == Path(root.anchor):
        raise PermissionError("filesystem root cannot be an isolated workspace")
    if access_profile is not None and (
        read_roots is not None or write_roots is not None
    ):
        raise ValueError("use either an access profile or legacy root arguments")
    if access_profile is None:
        access_profile = (
            ProcessAccessProfile.workspace(
                read_roots=read_roots or (), write_roots=write_roots or ()
            )
            if not git
            else ProcessAccessProfile.broker(
                "git", read_roots=read_roots or (), write_roots=write_roots or ()
            )
        )
    if not isinstance(access_profile, ProcessAccessProfile):
        raise TypeError("access_profile must be a ProcessAccessProfile")
    command = list(command)
    if command and Path(command[0]).is_absolute():
        try:
            Path(command[0]).resolve(strict=True)
        except OSError as error:
            raise PermissionError("process executable is unavailable") from error
    git_executable = None
    runtime_executables = []
    sqlite_extension = None
    if command and Path(command[0]).is_absolute():
        runtime_executables.append(Path(command[0]).resolve(strict=True))
        sqlite_extension = _sqlite_extension_for_interpreter(command[0])
        if sqlite_extension is not None:
            runtime_executables.append(sqlite_extension)
    if git:
        selected_git = shutil.which("git", path="/opt/homebrew/bin:/usr/bin:/bin")
        if not selected_git:
            raise PermissionError("Git executable is unavailable")
        git_executable = Path(selected_git).resolve(strict=True)
        if git_executable.is_relative_to(root):
            raise PermissionError(
                "Git executable must be outside the writable workspace"
            )
        command = [str(git_executable), *command[1:]]
        if any(not Path(helper).is_absolute() for helper in git_helpers):
            raise PermissionError("Git helper path must be absolute")
        runtime_executables.append(git_executable)
        runtime_executables.extend(Path(path) for path in git_helpers)
        if git_shell:
            runtime_executables.extend((Path("/bin/sh"), Path("/bin/bash")))
    runtime_dylibs = _runtime_dylibs(runtime_executables)
    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        '(allow file-read-data (literal "/dev/null"))',
        '(allow file-write* (literal "/dev/null"))',
    ]
    if git:
        # Git may invoke itself through the fixed PATH entry. Permit stat() on
        # that symlink so its child lookup does not fall back to /usr/bin/git.
        rules.append(f"(allow file-read-metadata (literal {json.dumps(selected_git)}))")
    if sqlite_extension is not None:
        python_version = _homebrew_cellar_version(sys.base_prefix)
        if python_version is not None:
            rules.append(
                f"(allow file-read-metadata (subpath {json.dumps(str(python_version))}))"
            )
        sqlite_libs = {
            dependency.resolve(strict=True).parent
            for dependency in runtime_dylibs
            if dependency.name.startswith("libsqlite")
            and _homebrew_cellar_version(dependency) is not None
        }
        for directory in sorted(sqlite_libs, key=str):
            rules.append(
                f"(allow file-read-metadata (subpath {json.dumps(str(directory))}))"
            )
    system_roots = (
        Path("/System"),
        Path("/usr/lib"),
        Path("/usr/bin"),
        Path("/bin"),
        Path("/private/etc"),
    )
    read_candidates = [
        *system_roots,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.base_prefix).parent.parent,
        root,
        *access_profile.read_roots,
        *access_profile.executable_paths,
    ]
    if git:
        read_candidates.append(git_executable)
        read_candidates.extend(trusted_git_metadata(root))
    executable = Path(command[0]) if command else None
    if executable is not None and executable.is_absolute():
        read_candidates.append(executable)
    for path in dict.fromkeys(read_candidates):
        candidate = Path(path)
        if not candidate.is_absolute():
            raise PermissionError("process read root must be absolute")
        for entry in _metadata_path_entries(candidate):
            rules.append(
                f"(allow file-read-metadata (literal {json.dumps(str(entry))}))"
            )
        # A sandboxed execvp()/dyld traversal can inspect the entry itself,
        # including a venv symlink, in addition to its parent directories.
        rules.append(
            f"(allow file-read-metadata (literal {json.dumps(str(candidate))}))"
        )
        # sandbox-exec must be able to traverse every lexical component (in
        # particular venv/Homebrew symlink paths), not only the resolved root.
        for parent in candidate.parents:
            rules.append(
                f"(allow file-read-metadata (literal {json.dumps(str(parent))}))"
            )
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as error:
            raise PermissionError("process read root is unavailable") from error
        for entry in _metadata_path_entries(candidate):
            rules.append(
                f"(allow file-read-metadata (literal {json.dumps(str(entry))}))"
            )
        scope = "literal" if candidate.is_file() else "subpath"
        rules.append(f"(allow file-read* ({scope} {json.dumps(str(candidate))}))")
        for parent in candidate.parents:
            rules.append(
                f"(allow file-read-metadata (literal {json.dumps(str(parent))}))"
            )
    for dependency in runtime_dylibs:
        rules.append(f"(allow file-read* (literal {json.dumps(str(dependency))}))")
        for parent in dependency.parents:
            rules.append(
                f"(allow file-read-metadata (literal {json.dumps(str(parent))}))"
            )
    # Runtime startup may inspect the root directory entry itself. Permit
    # reading that entry without granting access to the rest of the filesystem.
    rules.append('(allow file-read-data (literal "/"))')
    executable_mapping_roots = (
        *system_roots,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.base_prefix).parent.parent,
    )
    for path in dict.fromkeys(executable_mapping_roots):
        try:
            candidate = Path(path).resolve(strict=True)
        except OSError as error:
            raise PermissionError("process runtime root is unavailable") from error
        if candidate.is_relative_to(root):
            continue
        rules.append(
            f"(allow file-map-executable (subpath {json.dumps(str(candidate))}))"
        )
    for dependency in runtime_dylibs:
        rules.append(
            f"(allow file-map-executable (literal {json.dumps(str(dependency))}))"
        )
    if not read_only:
        if access_profile.workspace_writable:
            rules.append(f"(allow file-write* (subpath {json.dumps(str(root))}))")
        for path in access_profile.write_paths:
            candidate = Path(path)
            if not candidate.is_absolute():
                raise PermissionError("process write path must be absolute")
            candidate = candidate.resolve()
            if not candidate.is_relative_to(root):
                raise PermissionError("process write path escapes workspace")
            scope = "subpath" if candidate.is_dir() else "literal"
            rules.append(f"(allow file-write* ({scope} {json.dumps(str(candidate))}))")
        for path in access_profile.write_roots:
            candidate = Path(path)
            if not candidate.is_absolute():
                raise PermissionError("process write root must be absolute")
            try:
                candidate = candidate.resolve(strict=True)
            except OSError as error:
                raise PermissionError("process write root is unavailable") from error
            rules.append(f"(allow file-write* (subpath {json.dumps(str(candidate))}))")
    if git:
        # Never trust task/profile PATH or an executable written by a task.
        rules.extend(
            [
                "(deny process-exec)",
                f"(allow process-exec (literal {json.dumps(command[0])}))",
            ]
        )
        for helper in git_helpers:
            helper_path = Path(helper)
            helper = helper_path.resolve(strict=True)
            for parent in {helper_path.parent, helper.parent}:
                rules.append(f"(allow file-read* (subpath {json.dumps(str(parent))}))")
            rules.append(f"(allow process-exec (literal {json.dumps(str(helper))}))")
        if git_shell:
            # Git runs the fixed, quoted SSH command through /bin/sh (bash on
            # macOS). OpenDirectory is needed for getpwuid inside that sandbox.
            rules.extend(
                [
                    '(allow process-exec (literal "/bin/sh"))',
                    '(allow process-exec (literal "/bin/bash"))',
                    # OpenSSH resolves its default shell through this macOS
                    # selector symlink; grant only the selector itself.
                    '(allow file-read* (literal "/private/var/select/sh"))',
                    '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))',
                ]
            )
        for host, port in network_remotes:
            if (
                not isinstance(host, str)
                or not host
                or isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= 65535
            ):
                raise ValueError("network remote is invalid")
            target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            rules.append(f'(allow network-outbound (remote ip "{target}"))')
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
    if network_proxy is not None:
        if (
            isinstance(network_proxy, bool)
            or not isinstance(network_proxy, int)
            or not 1 <= network_proxy <= 65535
        ):
            raise ValueError("network proxy port is invalid")
        rules.append("(allow system-socket (socket-domain AF_INET))")
        rules.append(
            f'(allow network-outbound (remote ip "localhost:{network_proxy}"))'
        )
    executable_files = set(access_profile.executable_paths)
    executable_files.update(git_helpers)
    if command and Path(command[0]).is_absolute():
        executable_files.add(Path(command[0]))
    for item in sorted(executable_files, key=str):
        executable_path = Path(item)
        try:
            resolved_path = executable_path.resolve(strict=True)
        except OSError as error:
            raise PermissionError("process executable is unavailable") from error
        rules.append(
            f"(allow file-map-executable (literal {json.dumps(str(resolved_path))}))"
        )
    if access_profile.executable_paths:
        rules.append("(deny process-exec)")
        allowed_executables = {
            Path(command[0]).resolve(strict=True),
            *(
                Path(item).resolve(strict=True)
                for item in access_profile.executable_paths
            ),
        }
        for executable_path in sorted(allowed_executables, key=str):
            rules.append(
                f"(allow process-exec (literal {json.dumps(str(executable_path))}))"
            )
    socket_paths = (*unix_sockets, *access_profile.unix_sockets)
    if socket_paths:
        rules.append("(allow system-socket (socket-domain AF_UNIX))")
        for socket_path in socket_paths:
            lexical_socket = Path(socket_path)
            try:
                socket_path = lexical_socket.resolve(strict=True)
            except OSError as error:
                raise PermissionError("allowed Unix socket is unavailable") from error
            if not socket_path.is_socket():
                raise PermissionError("allowed Unix socket is unavailable")
            for entry in _metadata_path_entries(lexical_socket):
                rules.append(
                    f"(allow file-read-metadata (literal {json.dumps(str(entry))}))"
                )
            rules.append(
                "(allow network-outbound (remote unix-socket "
                f"(literal {json.dumps(str(socket_path))})))"
            )
    profile = "\n".join(rules)
    return ["/usr/bin/sandbox-exec", "-p", profile, *command]
