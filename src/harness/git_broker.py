"""Local Git smart transport using inherited pipes, never a general shell.

The receiver has its own kernel profile. Fetch/clone servers cannot write;
push servers can write only the approved bare repository. No network access
or additional executable is granted. Git's remote-fd helper must be the same
trusted executable image as Git itself.
"""

import base64
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .git_http import https_proxy, validate_https_url
from .git_policy import validate_git_destination
from .git_ssh import (
    agent_identities,
    key_fingerprint,
    parse_ssh_url,
    resolve_ssh_addresses,
    ssh_proxy,
    validate_host_keys,
)
from .isolation import isolated_command
from .process_control import run_cancellable, supervised_popen
from .process_failures import process_failure_category
from .workflows import GitWorkflow

_LOCAL_GIT_HELPER_NAMES = (
    "git-index-pack",
    "git-maintenance",
    "git-pack-objects",
    "git-remote-fd",
    "git-unpack-objects",
)


def local_path(url, cwd):
    parsed = urlsplit(url)
    if (
        "::" in url
        or parsed.scheme not in {"", "file"}
        or (
            parsed.scheme == "file"
            and (
                parsed.netloc not in {"", "localhost"}
                or parsed.query
                or parsed.fragment
            )
        )
    ):
        raise PermissionError("only local Git transports are enabled")
    path = Path(unquote(parsed.path) if parsed.scheme else url)
    return (path if path.is_absolute() else Path(cwd) / path).resolve()


def transport_index(args):
    index = 1
    while index < len(args):
        token = args[index]
        if args[0] == "clone" and token in {"--depth", "--branch", "-b"}:
            index += 2
            continue
        if not token.startswith("-"):
            return index
        allowed_options = {
            "--ff-only",
            "--quiet",
            "--tags",
            "--no-tags",
            "--prune",
            "--delete",
            "-d",
            "--force",
            "-f",
            "--atomic",
            "--dry-run",
            "--porcelain",
            "--force-with-lease",
            "--set-upstream",
            "-u",
        }
        if args[0] != "push":
            allowed_options -= {"--set-upstream", "-u"}
        if args[0] != "fetch":
            allowed_options.discard("--prune")
        if (
            args[0] != "clone"
            and token not in allowed_options
            and not token.startswith("--force-with-lease=")
        ):
            raise PermissionError("unsupported broker transport option")
        index += 1
    raise PermissionError("transport requires an explicit remote")


def fetch_plan(args, index):
    """Return safe fetch flags and explicit branch/tag refspecs."""
    command = args[0]
    if command not in {"fetch", "pull"}:
        raise PermissionError("fetch plan requires fetch or pull")
    remote = args[index]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", remote):
        raise PermissionError("fetch requires a named remote")
    prefix = args[1:index]
    allowed = {"--quiet", "--tags", "--no-tags"}
    if command == "fetch":
        allowed.add("--prune")
    else:
        allowed.add("--ff-only")
    if any(option not in allowed for option in prefix):
        raise PermissionError("fetch option is not supported")
    if len(prefix) != len(set(prefix)):
        raise PermissionError("fetch options cannot be repeated")
    if "--tags" in prefix and "--no-tags" in prefix:
        raise PermissionError("fetch cannot combine --tags and --no-tags")
    if command == "pull" and "--ff-only" not in prefix:
        raise PermissionError("pull requires --ff-only")
    options = tuple(option for option in prefix if option != "--ff-only")
    refs = args[index + 1 :] or ["*"]
    if any(not isinstance(ref, str) or not ref or ref.startswith("-") for ref in refs):
        raise PermissionError("fetch ref is invalid")
    refspecs = []
    has_tag = False
    for ref in refs:
        if ref == "*" and len(args[index + 1 :]) == 0:
            name, is_tag = "*", False
        elif ref.startswith("refs/heads/"):
            name, is_tag = ref.removeprefix("refs/heads/"), False
        elif ref.startswith("refs/tags/"):
            name, is_tag = ref.removeprefix("refs/tags/"), True
        elif ref.startswith("refs/"):
            raise PermissionError("fetch only supports branch and tag refs")
        else:
            name, is_tag = ref, False
        if name != "*":
            try:
                GitWorkflow._validate_ref(name)
            except ValueError as error:
                raise PermissionError("fetch ref is invalid") from error
        has_tag = has_tag or is_tag
        if is_tag:
            refspecs.append(f"refs/tags/{name}:refs/tags/{name}")
        else:
            refspecs.append(f"+refs/heads/{name}:refs/remotes/{remote}/{name}")
    if len(set(refspecs)) != len(refspecs):
        raise PermissionError("fetch refs cannot be repeated")
    if "--prune" in options and (has_tag or "--tags" in options):
        raise PermissionError("prune is limited to remote-tracking branches")
    if command == "pull" and has_tag:
        raise PermissionError("pull requires a branch ref")
    return options, tuple(refspecs)


def push_branch(args, index):
    """Resolve the one branch affected by the bounded local push contract."""
    tail = args[index + 1 :]
    options = {
        "--quiet",
        "--force",
        "-f",
        "--atomic",
        "--porcelain",
        "--delete",
        "-d",
        "--set-upstream",
        "-u",
    }
    if any(arg.startswith("-") and arg not in options for arg in tail):
        raise PermissionError("push branch option is not supported")
    refs = [arg for arg in tail if not arg.startswith("-")]
    if len(refs) != 1:
        raise PermissionError("push branch requires exactly one refspec")
    deleting = "--delete" in tail or "-d" in tail
    if deleting and any(arg in {"-u", "--set-upstream"} for arg in tail):
        raise PermissionError("push upstream cannot be configured for a deletion")
    ref = refs[0].removeprefix("+")
    if ref.startswith(":"):
        deleting = True
        ref = ref[1:]
    elif deleting:
        if ":" in ref:
            raise PermissionError("push branch deletion requires a destination")
    else:
        source, sep, destination = ref.partition(":")
        if not source or not sep or not destination:
            raise PermissionError(
                "push branch requires an explicit source and destination"
            )
        if source != "HEAD":
            try:
                GitWorkflow._validate_ref(source.removeprefix("refs/heads/"))
            except ValueError as error:
                raise PermissionError("push branch source is invalid") from error
        ref = destination
    if not ref or ref == "HEAD" or ref.startswith("refs/tags/"):
        raise PermissionError("push branch destination is not a branch")
    if not deleting and not ref.startswith("refs/heads/"):
        raise PermissionError("push branch destination must be fully qualified")
    if ref.startswith("refs/") and not ref.startswith("refs/heads/"):
        raise PermissionError("push branch destination is not under refs/heads")
    branch = ref.removeprefix("refs/heads/")
    try:
        GitWorkflow._validate_ref(branch)
    except ValueError as error:
        raise PermissionError("push branch destination is invalid") from error
    return branch, deleting


def push_refs(args, index):
    """Normalize one or more explicit, non-forced branch update refspecs."""
    tail = args[index + 1 :]
    safe_options = {"--quiet", "--atomic", "--porcelain", "-u", "--set-upstream"}
    if any(arg.startswith("-") and arg not in safe_options for arg in tail):
        raise PermissionError("push branch option is not supported")
    raw_refs = [arg for arg in tail if not arg.startswith("-")]
    if not raw_refs:
        raise PermissionError("push branch requires explicit refspecs")
    if any(arg in {"-u", "--set-upstream"} for arg in args[1:]) and len(raw_refs) != 1:
        raise PermissionError("push branch upstream requires exactly one refspec")
    pairs = []
    for refspec in raw_refs:
        if refspec.startswith(("+", ":")) or "*" in refspec:
            raise PermissionError("push branch refspec is not a bounded update")
        source, separator, destination = refspec.partition(":")
        if not separator or not source or not destination or ":" in destination:
            raise PermissionError(
                "push branch requires explicit source and destination refs"
            )
        if source == "HEAD":
            raise PermissionError("push branch source must be a named branch")
        source = source.removeprefix("refs/heads/")
        try:
            GitWorkflow._validate_ref(source)
        except ValueError as error:
            raise PermissionError("push branch source is invalid") from error
        if not destination.startswith("refs/heads/"):
            raise PermissionError("push branch destination must be fully qualified")
        destination = destination.removeprefix("refs/heads/")
        try:
            GitWorkflow._validate_ref(destination)
        except ValueError as error:
            raise PermissionError("push branch destination is invalid") from error
        pairs.append((source, destination))
    sources = [source for source, _ in pairs]
    destinations = [destination for _, destination in pairs]
    if len(sources) != len(set(sources)) or len(destinations) != len(set(destinations)):
        raise PermissionError(
            "push branch refspecs contain duplicate sources or destinations"
        )
    return tuple(pairs)


def push_targets(args, index):
    """Return normalized (source, destination) pairs; None source means delete."""
    tail = args[index + 1 :]
    if any(arg in {"--force", "-f", "--force-with-lease"} for arg in tail):
        raise PermissionError("push branch force updates are prohibited")
    if (
        "--delete" in tail
        or "-d" in tail
        or any(arg.startswith((":", "+:")) for arg in tail)
    ):
        destination, deleting = push_branch(args, index)
        assert deleting
        return ((None, destination),)
    return push_refs(args, index)


def push_execution_args(args, index, targets, approved_updates=()):
    """Build a push using approved object IDs and atomic multi-ref semantics."""
    if not targets or targets[0][0] is None:
        return list(args), index
    approved = {
        (source, destination): oid for source, destination, oid in approved_updates
    }
    if approved_updates and set(approved) != set(targets):
        raise PermissionError(
            "push branch refs differ from the approved source commits"
        )
    options = [arg for arg in args[1:] if arg.startswith("-")]
    refspecs = [
        f"{approved.get((source, destination), source)}:refs/heads/{destination}"
        for source, destination in targets
    ]
    remote = args[index]
    if len(targets) > 1 and "--atomic" not in options:
        options.append("--atomic")
    built = [args[0], *options, remote, *refspecs]
    return built, transport_index(built)


def verify_push_tracking(push_refs, remote, cwd, env, approved_updates=()):
    """Check every fetched remote-tracking ref equals its pushed source branch."""
    approved = {
        (source, destination): oid for source, destination, oid in approved_updates
    }
    for source, destination in push_refs:
        if source is None:
            continue
        source_oid = run_cancellable(
            subprocess.run,
            isolated_command(
                ["git", "rev-parse", "--verify", f"refs/heads/{source}^{{commit}}"],
                cwd,
                git=True,
            ),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        target_oid = run_cancellable(
            subprocess.run,
            isolated_command(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"refs/remotes/{remote}/{destination}^{{commit}}",
                ],
                cwd,
                git=True,
            ),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        expected = approved.get((source, destination), source_oid.stdout.strip())
        if (
            source_oid.returncode
            or target_oid.returncode
            or expected != target_oid.stdout.strip()
        ):
            raise RuntimeError("pushed remote ref does not match its local source")


def record_tracking_verification(result, push_refs, remote, cwd, env, approved_updates):
    try:
        verify_push_tracking(push_refs, remote, cwd, env, approved_updates)
    except RuntimeError as error:
        result.returncode = 1
        result.stderr += f"\npush succeeded but tracking verification failed: {error}"
    return result


def push_upstream_branch(args, index):
    """Resolve an explicit source branch for a bounded `push -u` request."""
    tail = args[index + 1 :]
    if not any(arg in {"-u", "--set-upstream"} for arg in args[1:]):
        return None
    refs = [arg for arg in tail if not arg.startswith("-")]
    if len(refs) != 1:
        raise PermissionError("push upstream requires exactly one explicit branch")
    source = refs[0].removeprefix("+").partition(":")[0]
    if source == "HEAD":
        raise PermissionError("push upstream requires a named source branch")
    source = source.removeprefix("refs/heads/")
    try:
        GitWorkflow._validate_ref(source)
    except ValueError as error:
        raise PermissionError("push upstream source branch is invalid") from error
    return source


def configure_push_upstream(args, index, push_ref, cwd, env):
    """Bind the explicit local branch to the reconciled named-remote ref."""
    source = push_upstream_branch(args, index)
    if source is None:
        return None
    destination, deleting = push_ref
    if deleting:
        raise PermissionError("push upstream cannot be configured for a deletion")
    remote_ref = f"{args[index]}/{destination}"
    command = isolated_command(
        ["git", "branch", f"--set-upstream-to={remote_ref}", source], cwd, git=True
    )
    return run_cancellable(
        subprocess.run,
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def validate_push_upstream(args, index, push_ref):
    source = push_upstream_branch(args, index)
    if source is not None and (push_ref is None or push_ref[1]):
        raise PermissionError("push upstream requires a non-deleting branch push")
    return source


def strip_push_upstream(args, index):
    """Keep Git's FD/URL transport name out of persistent branch config."""
    if not args or args[0] != "push":
        return list(args), index
    removed_before_remote = sum(
        arg in {"-u", "--set-upstream"} for arg in args[1:index]
    )
    return (
        [arg for arg in args if arg not in {"-u", "--set-upstream"}],
        index - removed_before_remote,
    )


def reconcile_push_upstream(result, args, index, push_ref, cwd, env):
    if result.returncode:
        return result
    try:
        followup = configure_push_upstream(args, index, push_ref, cwd, env)
    except (OSError, PermissionError, subprocess.TimeoutExpired, ValueError) as error:
        result.returncode = 1
        result.stderr += f"\npush succeeded but upstream setup failed: {error}"
        return result
    if followup is None:
        return result
    result.stdout += followup.stdout
    result.stderr += followup.stderr
    if followup.returncode:
        result.returncode = followup.returncode
        result.stderr += "\npush succeeded but upstream setup failed"
    return result


def probe(command, cwd, env):
    result = run_cancellable(
        subprocess.run,
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise PermissionError("Git sandbox preflight failed: " + result.stderr)


def stop_group(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The process may exit between poll and killpg.
        except PermissionError:
            # Some macOS sandbox profiles deny signaling the process group while
            # still allowing the parent to terminate its direct child.
            process.kill()
        process.wait()


@contextmanager
def pinned_directory(path, identity):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != identity:
            raise PermissionError("local remote identity changed after preflight")
        yield descriptor
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _LocalTransportCore:
    args: tuple[str, ...]
    cwd: Path
    remote: Path
    index: int
    server_command: tuple[str, ...]
    env: dict
    identity: tuple[int, int]
    push_ref: tuple[str, bool] | None = None
    push_refs: tuple[tuple[str | None, str], ...] | None = None
    approved_updates: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def prepare(cls, args, cwd, env, remote_url=None, approved_updates=()):
        cwd = Path(cwd).resolve()
        index = transport_index(args)
        if args[0] != "clone" and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*", args[index]
        ):
            raise PermissionError("transport requires a named remote")
        if args[0] == "pull" and (
            "--ff-only" not in args or len(args[index + 1 :]) != 1
        ):
            raise PermissionError("broker pull requires ff-only and an explicit branch")
        if args[0] in {"fetch", "pull"}:
            fetch_plan(args, index)
        push_refs = push_targets(args, index) if args[0] == "push" else None
        push_ref = None
        if push_refs and len(push_refs) == 1:
            push_ref = (
                (push_refs[0][1], True)
                if push_refs[0][0] is None
                else (push_refs[0][1], False)
            )
        validate_push_upstream(args, index, push_ref)
        remote = local_path(remote_url or args[index], cwd)
        if not remote.is_dir():
            raise PermissionError("local Git remote is unavailable")
        git_executable = Path(
            shutil.which("git", path="/opt/homebrew/bin:/usr/bin:/bin") or ""
        ).resolve(strict=True)
        helper_dir = git_executable.parent.parent / "libexec/git-core"
        helpers = tuple(helper_dir / name for name in _LOCAL_GIT_HELPER_NAMES)
        for helper in helpers:
            if not helper.is_file() or helper.resolve(strict=True) != git_executable:
                label = "remote-fd" if helper.name == "git-remote-fd" else "Git"
                raise PermissionError(f"trusted {label} helper is unavailable")
        writing = args[0] == "push"
        server = isolated_command(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "receive-pack" if writing else "upload-pack",
                ".",
            ],
            remote,
            git=True,
            git_helpers=helpers,
            read_only=not writing,
        )
        check = isolated_command(
            ["git", "rev-parse", "--is-bare-repository"],
            remote,
            git=True,
            read_only=True,
        )
        result = run_cancellable(
            subprocess.run,
            check,
            cwd=remote,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode:
            category = process_failure_category(result)
            if category == "host_sandbox_blocked":
                detail = "bare remote status" if writing else "remote Git status"
                raise PermissionError(
                    f"host sandbox blocked local Git verification; {detail} could not be verified"
                )
            raise PermissionError("local Git repository could not be verified")
        if writing and result.stdout.strip() != "true":
            raise PermissionError("push requires a verified bare local remote")
        helper_dir = Path(server[3]).parent.parent / "libexec/git-core"
        helper = helper_dir / "git-remote-fd"
        if not helper.is_file() or helper.resolve() != Path(server[3]):
            raise PermissionError("trusted Git remote-fd helper is unavailable")
        env = {
            **env,
            "GIT_EXEC_PATH": str(helper_dir),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        }
        probe(isolated_command(["git", "--version"], cwd, git=True), cwd, env)
        info = remote.stat()
        return cls(
            tuple(args),
            cwd,
            remote,
            index,
            tuple(server),
            env,
            (info.st_dev, info.st_ino),
            push_ref,
            push_refs,
            tuple(approved_updates),
        )

    def _reconcile_push(self, result):
        targets = self.push_refs or ()
        try:
            if targets and targets[0][0] is None:
                branch = targets[0][1]
                tracking = f"refs/remotes/{self.args[self.index]}/{branch}"
                command = isolated_command(
                    ["git", "update-ref", "-d", tracking], self.cwd, git=True
                )
                followup = run_cancellable(
                    subprocess.run,
                    command,
                    cwd=self.cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            else:
                branches = [destination for _source, destination in targets]
                followup_plan = LocalTransport.prepare(
                    ["fetch", "--no-tags", self.args[self.index], *branches],
                    self.cwd,
                    self.env,
                    str(self.remote),
                )
                if followup_plan.identity != self.identity:
                    raise PermissionError(
                        "remote identity changed before tracking refresh"
                    )
                followup = followup_plan.run()
                if followup.returncode == 0:
                    verify_push_tracking(
                        targets,
                        self.args[self.index],
                        self.cwd,
                        self.env,
                        self.approved_updates,
                    )
        except Exception as error:
            raise RuntimeError(
                f"push succeeded but local tracking reconciliation failed: {error}"
            ) from error
        result.stdout += followup.stdout
        result.stderr += followup.stderr
        if followup.returncode:
            result.returncode = followup.returncode
            result.stderr += "\npush succeeded but local tracking reconciliation failed"
        result = reconcile_push_upstream(
            result, self.args, self.index, self.push_ref, self.cwd, self.env
        )
        return result


def _ssh_config_quote(value):
    if not isinstance(value, str) or any(char in value for char in "\r\n\x00"):
        raise PermissionError("SSH configuration path is invalid")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class SshTransport:
    args: tuple[str, ...]
    cwd: Path
    url: str
    index: int
    env: dict
    host: str
    port: int
    address: str
    host_keys: tuple[tuple[str, str], ...]
    identity: tuple[str, str]
    agent_socket: str
    agent_file_identity: tuple[int, int]
    ssh_executable: Path
    push_ref: tuple[str, bool] | None = None
    push_refs: tuple[tuple[str | None, str], ...] | None = None
    approved_updates: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def prepare(
        cls,
        args,
        cwd,
        env,
        remote_url=None,
        allowed_hosts=(),
        allowed_ports=(22,),
        host_keys=None,
        credentials=None,
        agent_socket=None,
        approved_updates=(),
    ):
        cwd = Path(cwd).resolve()
        index = transport_index(args)
        if args[0] != "clone" and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*", args[index]
        ):
            raise PermissionError("SSH transport requires a named remote")
        if args[0] == "pull" and (
            "--ff-only" not in args or len(args[index + 1 :]) != 1
        ):
            raise PermissionError("SSH pull requires ff-only and an explicit branch")
        if args[0] in {"fetch", "pull"}:
            fetch_plan(args, index)
        push_refs = push_targets(args, index) if args[0] == "push" else None
        push_ref = None
        if push_refs and len(push_refs) == 1:
            push_ref = (
                (push_refs[0][1], True)
                if push_refs[0][0] is None
                else (push_refs[0][1], False)
            )
        validate_push_upstream(args, index, push_ref)
        source = remote_url or args[index]
        _normalized, host, url_username, port, path = parse_ssh_url(
            source, allowed_hosts, allowed_ports
        )
        credential = (credentials or {}).get(host)
        if (
            not isinstance(credential, dict)
            or set(credential) != {"username", "fingerprint"}
            or not isinstance(credential["username"], str)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", credential["username"])
            or not isinstance(credential["fingerprint"], str)
            or not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", credential["fingerprint"])
        ):
            raise PermissionError("SSH Git credential identity is not configured")
        if url_username and url_username != credential["username"]:
            raise PermissionError("SSH URL username differs from configured identity")
        key_pins = validate_host_keys(host, (host_keys or {}).get(host))
        agent_socket = agent_socket or os.environ.get("SSH_AUTH_SOCK")
        identities = agent_identities(agent_socket)
        selected = None
        for key_type, encoded in identities:
            candidate = f"{key_type} {encoded}"
            if key_fingerprint(candidate) == credential["fingerprint"]:
                selected = (key_type, encoded)
                break
        if selected is None:
            raise PermissionError("configured SSH identity is not loaded in the agent")
        try:
            agent_info = os.lstat(agent_socket)
        except OSError as error:
            raise PermissionError(
                "SSH agent socket changed during preflight"
            ) from error
        addresses = resolve_ssh_addresses(host, port)
        ssh_executable = Path("/usr/bin/ssh").resolve(strict=True)
        protected_env = {
            **env,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        }
        for key in (r"^core\.sshcommand$", r"^url\..*\.insteadof$"):
            check = run_cancellable(
                subprocess.run,
                isolated_command(["git", "config", "--get-regexp", key], cwd, git=True),
                cwd=cwd,
                env=protected_env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if check.returncode == 0:
                raise PermissionError("local SSH command or URL rewrite is prohibited")
            if check.returncode != 1:
                raise PermissionError("local SSH configuration could not be verified")
        url_user = credential["username"]
        authority = f"[{host}]" if ":" in host else host
        normalized = f"ssh://{url_user}@{authority}"
        if port != 22:
            normalized += f":{port}"
        normalized += "/" + path
        return cls(
            tuple(args),
            cwd,
            normalized,
            index,
            protected_env,
            host,
            port,
            addresses[0],
            tuple(tuple(key) for key in key_pins),
            selected,
            agent_socket,
            (agent_info.st_dev, agent_info.st_ino),
            ssh_executable,
            push_ref,
            push_refs,
            tuple(approved_updates),
        )

    def _config_files(self, directory, proxy_port):
        ssh_dir = Path(directory) / ".ssh"
        ssh_dir.mkdir(mode=0o700)
        known_hosts = ssh_dir / "known_hosts"
        identity = ssh_dir / "identity.pub"
        config = ssh_dir / "config"
        alias = (
            "harness-"
            + hashlib.sha256(f"{self.host}:{self.port}".encode()).hexdigest()[:24]
        )
        known_hosts.write_text(
            "".join(
                f"{alias} {key_type} {encoded}\n"
                for key_type, encoded in self.host_keys
            )
        )
        identity.write_text(f"{self.identity[0]} {self.identity[1]}\n")
        config.write_text(
            "\n".join(
                (
                    f"Host {self.host}",
                    f"  HostName {self.address}",
                    "  AddressFamily inet",
                    f"  Port {proxy_port}",
                    f"  ProxyCommand /usr/bin/nc -w 15 127.0.0.1 {proxy_port}",
                    "  ConnectTimeout 5",
                    "  ConnectionAttempts 1",
                    f"  User {self.url.split('@', 1)[0].removeprefix('ssh://')}",
                    f"  HostKeyAlias {alias}",
                    f"  UserKnownHostsFile {_ssh_config_quote(str(known_hosts))}",
                    "  GlobalKnownHostsFile /dev/null",
                    "  StrictHostKeyChecking yes",
                    "  UpdateHostKeys no",
                    "  CheckHostIP no",
                    "  VerifyHostKeyDNS no",
                    "  IdentityFile " + _ssh_config_quote(str(identity)),
                    "  IdentityAgent " + _ssh_config_quote(self.agent_socket),
                    "  IdentitiesOnly yes",
                    "  PreferredAuthentications publickey",
                    "  PubkeyAuthentication yes",
                    "  PasswordAuthentication no",
                    "  KbdInteractiveAuthentication no",
                    "  BatchMode yes",
                    "  ForwardAgent no",
                    "  ClearAllForwardings yes",
                    "  ControlMaster no",
                    "  ControlPath none",
                    "  ProxyCommand none",
                    "  ProxyJump none",
                    "  PermitLocalCommand no",
                    "  HostbasedAuthentication no",
                    "  GSSAPIAuthentication no",
                    "  CanonicalizeHostname no",
                    "",
                )
            )
        )
        for path in (known_hosts, identity, config):
            path.chmod(0o600)
        return config

    def run(self):
        try:
            agent_info = os.lstat(self.agent_socket)
        except OSError as error:
            raise PermissionError(
                "SSH agent socket changed before process start"
            ) from error
        if (agent_info.st_dev, agent_info.st_ino) != self.agent_file_identity:
            raise PermissionError("SSH agent socket changed before process start")
        with (
            ssh_proxy(self.address, self.port) as proxy_port,
            tempfile.TemporaryDirectory(
                prefix="harness-ssh-", dir="/private/tmp"
            ) as temporary,
        ):
            config = self._config_files(temporary, proxy_port)
            args, index = strip_push_upstream(self.args, self.index)
            if self.push_refs is not None:
                args, index = push_execution_args(
                    args, index, self.push_refs, self.approved_updates
                )
            args[index] = self.url
            if args[0] in {"fetch", "pull"}:
                options, refspecs = fetch_plan(self.args, self.index)
                git_args = [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "fetch",
                    *options,
                    self.url,
                    *refspecs,
                ]
            else:
                git_args = ["git", "-c", "core.hooksPath=/dev/null", *args]
            command = isolated_command(
                git_args,
                self.cwd,
                git=True,
                git_helpers=(self.ssh_executable, "/usr/bin/nc"),
                network_proxy=proxy_port,
                unix_sockets=(self.agent_socket,),
                git_shell=True,
                read_roots=(config.parent,),
            )
            env = {
                **self.env,
                "GIT_SSH_COMMAND": f"/usr/bin/ssh -F {shlex.quote(str(config))}",
                "GIT_SSH_VARIANT": "ssh",
                "SSH_AUTH_SOCK": self.agent_socket,
                "HOME": temporary,
            }
            with supervised_popen(
                subprocess.Popen,
                command,
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            ) as client:
                try:
                    stdout, stderr = client.communicate(timeout=120)
                finally:
                    stop_group(client)
            result = subprocess.CompletedProcess(
                list(self.args),
                client.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
        if result.returncode == 0 and self.args[0] == "pull":
            final = run_cancellable(
                subprocess.run,
                isolated_command(
                    [
                        "git",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "merge",
                        "--ff-only",
                        "FETCH_HEAD",
                    ],
                    self.cwd,
                    git=True,
                ),
                cwd=self.cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            result.returncode = final.returncode
            result.stdout += final.stdout
            result.stderr += final.stderr
        if result.returncode == 0 and self.push_refs is not None:
            targets = self.push_refs
            if targets and targets[0][0] is None:
                branch = targets[0][1]
                tracking = f"refs/remotes/{self.args[self.index]}/{branch}"
                followup = run_cancellable(
                    subprocess.run,
                    isolated_command(
                        ["git", "update-ref", "-d", tracking], self.cwd, git=True
                    ),
                    cwd=self.cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            else:
                followup = replace(
                    self,
                    args=(
                        "fetch",
                        "--no-tags",
                        self.args[self.index],
                        *[dest for _src, dest in targets],
                    ),
                    index=2,
                    push_ref=None,
                    push_refs=None,
                    approved_updates=(),
                ).run()
            result.stdout += followup.stdout
            result.stderr += followup.stderr
            if followup.returncode:
                result.returncode = followup.returncode
                result.stderr += (
                    "\npush succeeded but local tracking reconciliation failed"
                )
            elif targets and targets[0][0] is not None and self.approved_updates:
                result = record_tracking_verification(
                    result,
                    targets,
                    self.args[self.index],
                    self.cwd,
                    self.env,
                    self.approved_updates,
                )
            result = reconcile_push_upstream(
                result, self.args, self.index, self.push_ref, self.cwd, self.env
            )
        return result


@dataclass(frozen=True)
class HttpsTransport:
    args: tuple[str, ...]
    cwd: Path
    url: str
    index: int
    helper: Path
    env: dict
    ca_bundle: str | None = None
    auth_username: str | None = None
    auth_secret: str | None = field(default=None, repr=False)
    push_ref: tuple[str, bool] | None = None
    push_refs: tuple[tuple[str | None, str], ...] | None = None
    approved_updates: tuple[tuple[str, str, str], ...] = ()
    auth_mode: str | None = None

    @classmethod
    def prepare(
        cls,
        args,
        cwd,
        env,
        remote_url=None,
        allowed_hosts=(),
        ca_bundle=None,
        credentials=None,
        approved_updates=(),
    ):
        cwd = Path(cwd).resolve()
        index = transport_index(args)
        if args[0] != "clone" and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*", args[index]
        ):
            raise PermissionError("HTTPS transport requires a named remote")
        source = remote_url or args[index]
        url = validate_https_url(source, allowed_hosts)
        push_refs = push_targets(args, index) if args[0] == "push" else None
        push_ref = None
        if push_refs and len(push_refs) == 1:
            push_ref = (
                (push_refs[0][1], True)
                if push_refs[0][0] is None
                else (push_refs[0][1], False)
            )
        validate_push_upstream(args, index, push_ref)
        credential = (credentials or {}).get(urlsplit(url).hostname)
        auth_username = auth_secret = None
        auth_mode = None
        if credential is not None:
            legacy_basic = isinstance(credential, dict) and set(credential) == {
                "username",
                "secret_name",
            }
            if not isinstance(credential, dict):
                raise PermissionError("HTTPS Git credential configuration is invalid")
            auth_mode = "basic" if legacy_basic else credential.get("mode")
            if auth_mode == "basic":
                valid_shape = set(credential) in (
                    {"username", "secret_name"},
                    {"mode", "username", "secret_name"},
                )
                username = credential.get("username")
                if (
                    not valid_shape
                    or not isinstance(username, str)
                    or not username
                    or ":" in username
                    or any(char in username for char in "\r\n")
                ):
                    raise PermissionError(
                        "HTTPS Git credential configuration is invalid"
                    )
                auth_username = username
            elif auth_mode == "bearer":
                if set(credential) != {"mode", "secret_name"}:
                    raise PermissionError(
                        "HTTPS Git credential configuration is invalid"
                    )
            else:
                raise PermissionError("HTTPS Git credential configuration is invalid")
            secret_name = credential.get("secret_name")
            if not isinstance(secret_name, str) or not re.fullmatch(
                r"[A-Z0-9_]{1,128}", secret_name
            ):
                raise PermissionError("HTTPS Git credential configuration is invalid")
            from .security import SecretResolver

            auth_secret = SecretResolver().get(secret_name)
            if not auth_secret or any(char in auth_secret for char in "\r\n"):
                raise PermissionError("configured HTTPS Git credential is unavailable")
        if args[0] == "push" and auth_secret is None:
            raise PermissionError("HTTPS Git push requires a configured credential")
        if args[0] == "pull" and (
            "--ff-only" not in args or len(args[index + 1 :]) != 1
        ):
            raise PermissionError("HTTPS pull requires ff-only and an explicit branch")
        if args[0] in {"fetch", "pull"}:
            fetch_plan(args, index)
        if ca_bundle is not None:
            bundle = Path(ca_bundle).expanduser().resolve(strict=True)
            if not bundle.is_file():
                raise PermissionError("configured Git CA bundle is not a file")
            ca_bundle = str(bundle)
        command = isolated_command(["git", "--exec-path"], cwd, git=True)
        result = run_cancellable(
            subprocess.run,
            command,
            cwd=cwd,
            env={**env, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode:
            if process_failure_category(result) == "host_sandbox_blocked":
                raise PermissionError("host sandbox blocked HTTPS Git helper lookup")
            raise PermissionError("trusted HTTPS Git helper lookup failed")
        helper_dir = Path(result.stdout.strip()).resolve(strict=True)
        helper = helper_dir / "git-remote-https"
        if not helper.is_file() or helper.resolve().parent != helper_dir:
            raise PermissionError("trusted Git HTTPS helper is unavailable")
        protected_env = {
            **env,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_EXEC_PATH": str(helper_dir),
        }
        for key in (
            r"^http(\..*)?\.(extraheader|cookiefile|sslcert|sslkey|sslcainfo|sslcapath)$",
            r"^credential(\..*)?\.(helper|username|password)$",
            r"^url\..*\.insteadof$",
        ):
            check = run_cancellable(
                subprocess.run,
                isolated_command(["git", "config", "--get-regexp", key], cwd, git=True),
                cwd=cwd,
                env=protected_env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if check.returncode == 0:
                raise PermissionError(
                    "local Git HTTPS credential or URL override is prohibited"
                )
            if check.returncode not in {1}:
                if process_failure_category(check) == "host_sandbox_blocked":
                    raise PermissionError(
                        "host sandbox blocked HTTPS Git configuration verification"
                    )
                raise PermissionError(
                    "local Git HTTPS configuration could not be verified"
                )
        return cls(
            tuple(args),
            cwd,
            url,
            index,
            helper,
            protected_env,
            ca_bundle,
            auth_username,
            auth_secret,
            push_ref,
            push_refs,
            tuple(approved_updates),
            auth_mode,
        )

    def _authorization_header(self):
        if self.auth_secret is None:
            return None
        if self.auth_mode == "bearer":
            return f"Authorization: Bearer {self.auth_secret}"
        if self.auth_mode in {None, "basic"} and self.auth_username is not None:
            encoded = base64.b64encode(
                f"{self.auth_username}:{self.auth_secret}".encode()
            ).decode("ascii")
            return f"Authorization: Basic {encoded}"
        raise PermissionError("HTTPS Git credential configuration is invalid")

    def run(self):
        with https_proxy([urlsplit(self.url).hostname]) as proxy_port:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            git_args = [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                f"http.proxy={proxy_url}",
                "-c",
                f"http.https://{urlsplit(self.url).hostname}/.proxy={proxy_url}",
                "-c",
                "http.followRedirects=false",
                "-c",
                f"http.https://{urlsplit(self.url).hostname}/.followRedirects=false",
                "-c",
                "http.sslVerify=true",
                "-c",
                f"http.https://{urlsplit(self.url).hostname}/.sslVerify=true",
                "-c",
                "http.netrc=false",
                "-c",
                f"http.https://{urlsplit(self.url).hostname}/.netrc=false",
                "-c",
                "http.saveCookies=false",
                "-c",
                f"http.https://{urlsplit(self.url).hostname}/.saveCookies=false",
                "-c",
                "http.delegation=none",
            ]
            if self.ca_bundle:
                git_args.extend(
                    [
                        "-c",
                        f"http.sslCAInfo={self.ca_bundle}",
                        "-c",
                        f"http.https://{urlsplit(self.url).hostname}/.sslCAInfo={self.ca_bundle}",
                    ]
                )
            args, index = strip_push_upstream(self.args, self.index)
            if self.push_refs is not None:
                args, index = push_execution_args(
                    args, index, self.push_refs, self.approved_updates
                )
            args[index] = self.url
            if args[0] in {"fetch", "pull"}:
                options, refspecs = fetch_plan(self.args, self.index)
                git_args.extend(
                    [
                        "fetch",
                        *options,
                        self.url,
                        *refspecs,
                    ]
                )
            elif args[0] == "push":
                git_args.extend(args)
            else:
                git_args.extend(args)
            command = isolated_command(
                git_args,
                self.cwd,
                git=True,
                network_proxy=proxy_port,
                git_helpers=(self.helper,),
                read_roots=(Path(self.ca_bundle),) if self.ca_bundle else (),
            )
            env = {
                **self.env,
                "HTTPS_PROXY": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "ALL_PROXY": proxy_url,
                "all_proxy": proxy_url,
                "NO_PROXY": "",
                "no_proxy": "",
            }
            header = self._authorization_header()
            if header is not None:
                env.update(
                    {
                        "GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.extraHeader",
                        "GIT_CONFIG_VALUE_0": header,
                    }
                )
            with supervised_popen(
                subprocess.Popen,
                command,
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            ) as client:
                try:
                    stdout, stderr = client.communicate(timeout=120)
                finally:
                    stop_group(client)
            result = subprocess.CompletedProcess(
                list(self.args),
                client.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
            auth_failure = re.search(
                r"(?:HTTP\s+|error:\s*)(401|403)\b", result.stderr, re.IGNORECASE
            )
            credential_rejected = re.search(
                r"could not read Username|authentication failed",
                result.stderr,
                re.IGNORECASE,
            )
            if result.returncode and (auth_failure or credential_rejected):
                status = f" (HTTP {auth_failure.group(1)})" if auth_failure else ""
                result.stderr = f"HTTPS Git authentication failed{status}."
            if self.auth_secret is not None:
                from .security import SecretResolver

                encoded = (
                    base64.b64encode(
                        f"{self.auth_username}:{self.auth_secret}".encode()
                    ).decode("ascii")
                    if self.auth_mode in {None, "basic"}
                    and self.auth_username is not None
                    else None
                )
                secrets = (self.auth_secret, encoded, header)
                result.stdout = SecretResolver().redact(result.stdout, secrets)
                result.stderr = SecretResolver().redact(result.stderr, secrets)
        if result.returncode == 0 and self.args[0] == "pull":
            final = run_cancellable(
                subprocess.run,
                isolated_command(
                    [
                        "git",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "merge",
                        "--ff-only",
                        "FETCH_HEAD",
                    ],
                    self.cwd,
                    git=True,
                ),
                cwd=self.cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            result.returncode = final.returncode
            result.stdout += final.stdout
            result.stderr += final.stderr
        if result.returncode == 0 and (
            self.push_refs is not None or (self.push_ref and self.push_ref[1])
        ):
            targets = self.push_refs or ((None, self.push_ref[0]),)
            if targets and targets[0][0] is None:
                branch = targets[0][1]
                tracking = f"refs/remotes/{self.args[self.index]}/{branch}"
                followup = run_cancellable(
                    subprocess.run,
                    isolated_command(
                        ["git", "update-ref", "-d", tracking], self.cwd, git=True
                    ),
                    cwd=self.cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            else:
                followup = replace(
                    self,
                    args=(
                        "fetch",
                        "--no-tags",
                        self.args[self.index],
                        *[dest for _src, dest in targets],
                    ),
                    index=2,
                    push_ref=None,
                    push_refs=None,
                    approved_updates=(),
                ).run()
            result.stdout += followup.stdout
            result.stderr += followup.stderr
            if followup.returncode:
                result.returncode = followup.returncode
                result.stderr += (
                    "\npush succeeded but local tracking reconciliation failed"
                )
            elif targets and targets[0][0] is not None and self.approved_updates:
                result = record_tracking_verification(
                    result,
                    targets,
                    self.args[self.index],
                    self.cwd,
                    self.env,
                    self.approved_updates,
                )
            result = reconcile_push_upstream(
                result, self.args, self.index, self.push_ref, self.cwd, self.env
            )
        return result


@dataclass(frozen=True)
class LocalTransport(_LocalTransportCore):
    def run(self):
        with (
            pinned_directory(self.remote, self.identity) as remote_fd,
            supervised_popen(
                subprocess.Popen,
                [
                    str(Path(sys.executable).resolve()),
                    "-I",
                    "-S",
                    "-c",
                    "import os,sys; os.fchdir(int(sys.argv[1])); os.execv(sys.argv[2],sys.argv[2:])",
                    str(remote_fd),
                    *self.server_command,
                ],
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(remote_fd,),
            ) as server,
            ThreadPoolExecutor(max_workers=1) as readers,
        ):
            server_errors = readers.submit(server.stderr.read)
            try:
                url = f"fd::{server.stdout.fileno()},{server.stdin.fileno()}"
                args, index = strip_push_upstream(self.args, self.index)
                if self.push_refs is not None:
                    args, index = push_execution_args(
                        args, index, self.push_refs, self.approved_updates
                    )
                args[index] = url
                if args[0] == "clone" and len(args) == self.index + 1:
                    args.append(
                        str(validate_git_destination(list(self.args), self.cwd))
                    )
                if args[0] in {"fetch", "pull"}:
                    options, refspecs = fetch_plan(self.args, self.index)
                    args = [
                        "fetch",
                        *options,
                        url,
                        *refspecs,
                    ]
                command = isolated_command(
                    [
                        "git",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "-c",
                        "protocol.fd.allow=always",
                        *args,
                    ],
                    self.cwd,
                    git=True,
                    git_helpers=tuple(
                        Path(self.server_command[3]).parent.parent
                        / "libexec/git-core"
                        / name
                        for name in _LOCAL_GIT_HELPER_NAMES
                    ),
                )
                with supervised_popen(
                    subprocess.Popen,
                    command,
                    cwd=self.cwd,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(server.stdout.fileno(), server.stdin.fileno()),
                    start_new_session=True,
                ) as client:
                    # Only the client keeps these ends; early client exit must
                    # deliver EOF to the receiver instead of waiting on us.
                    server.stdin.close()
                    server.stdout.close()
                    try:
                        stdout, stderr = client.communicate(timeout=120)
                    finally:
                        stop_group(client)
                    result = subprocess.CompletedProcess(
                        list(self.args),
                        client.returncode,
                        stdout.decode(errors="replace"),
                        stderr.decode(errors="replace"),
                    )
                server.wait(timeout=15)
            finally:
                stop_group(server)
            result.stderr += server_errors.result().decode(errors="replace")
            if server.returncode and not result.returncode:
                result.returncode = server.returncode
        if result.returncode == 0 and self.args[0] in {"clone", "pull"}:
            target = (
                validate_git_destination(list(self.args), self.cwd)
                if self.args[0] == "clone"
                else self.cwd
            )
            tail = (
                ["config", "remote.origin.url", str(self.remote)]
                if self.args[0] == "clone"
                else ["merge", "--ff-only", "FETCH_HEAD"]
            )
            final = run_cancellable(
                subprocess.run,
                isolated_command(
                    ["git", "-c", "core.hooksPath=/dev/null", *tail], target, git=True
                ),
                cwd=target,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            result.returncode = final.returncode
            result.stdout += final.stdout
            result.stderr += final.stderr
        if result.returncode == 0 and self.push_refs is not None:
            return self._reconcile_push(result)
        return result
