"""Fail-closed Git permissions; exact destructive/external actions need HITL."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit


class GitRisk(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


@dataclass(frozen=True)
class GitOperation:
    command: str
    risk: GitRisk
    approval_action: str | None = None
    remote: str | None = None


def git_operation(arguments):
    if not arguments or not all(isinstance(arg, str) and arg for arg in arguments):
        raise ValueError("Git requires nonempty arguments")
    command = arguments[0]
    if command not in {
        "status",
        "init",
        "clone",
        "remote",
        "fetch",
        "pull",
        "branch",
        "checkout",
        "switch",
        "add",
        "commit",
        "merge",
        "push",
        "diff",
        "log",
        "tag",
        "rev-parse",
        "ls-files",
        "show",
    }:
        raise PermissionError("Git command/global options are not allowlisted")
    if any(arg == "--output" or arg.startswith("--output=") for arg in arguments[1:]):
        raise PermissionError("Git read commands cannot write output files")
    action = None
    destination = None
    if command == "branch" and any(
        arg == "--delete"
        or arg.startswith("--delete=")
        or (
            arg.startswith("-")
            and not arg.startswith("--")
            and any(char in arg[1:] for char in "dD")
        )
        for arg in arguments[1:]
    ):
        action = "branch.delete"
    if command == "push":
        if any(
            arg in {"--mirror", "--prune", "--all"}
            or arg.startswith(("--repo", "--receive-pack", "--exec"))
            for arg in arguments[1:]
        ):
            raise PermissionError(
                "push requires an explicit, bounded remote/ref action"
            )
        positional = [arg for arg in arguments[1:] if not arg.startswith("-")]
        if (
            len(positional) < 2
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", positional[0])
            or any("*" in arg for arg in positional[1:])
        ):
            raise PermissionError("push requires a configured remote and explicit refs")
        destination = positional[0]
        deleting = any(
            arg in {"--delete", "-d"} or arg.startswith(("--delete=", ":", "+:"))
            for arg in arguments[1:]
        )
        action = "branch.delete" if deleting else "git.push"
    read = command in {"status", "diff", "log", "rev-parse", "ls-files", "show"}
    if command == "branch":
        read = len(arguments) == 1 or any(
            arg
            in {
                "--list",
                "--show-current",
                "--contains",
                "--merged",
                "--no-merged",
                "-r",
                "--remotes",
                "-a",
                "--all",
            }
            for arg in arguments[1:]
        )
    if command == "remote":
        read = len(arguments) == 1 or arguments[1] in {
            "get-url",
            "show",
            "-v",
            "--verbose",
        }
    if command == "tag":
        read = len(arguments) == 1 or "--list" in arguments or "-l" in arguments
    return (not read or action is not None), action, destination


def describe_git_operation(arguments):
    write, action, remote = git_operation(arguments)
    risk = (
        GitRisk.DESTRUCTIVE
        if action == "branch.delete"
        else GitRisk.WRITE
        if write
        else GitRisk.READ
    )
    return GitOperation(arguments[0], risk, action, remote)


def validate_git_destination(arguments, workspace):
    """Restrict init/clone writes to the configured workspace and safe flags."""
    command = arguments[0]
    if command not in {"init", "clone"}:
        return None
    values = arguments[1:]
    positional = []
    index = 0
    while index < len(values):
        token = values[index]
        if command == "init" and token == "--bare":
            index += 1
            continue
        if command == "init" and token.startswith("--initial-branch="):
            if not token.partition("=")[2]:
                raise ValueError("init branch name must not be empty")
            index += 1
            continue
        if command == "init" and token in {"-b", "--initial-branch"}:
            if index + 1 >= len(values) or not values[index + 1]:
                raise ValueError("init branch name must not be empty")
            index += 2
            continue
        if command == "clone" and token == "--single-branch":
            index += 1
            continue
        if command == "clone" and token in {"--quiet", "--no-checkout"}:
            index += 1
            continue
        if command == "clone" and token in {"--depth", "--branch", "-b"}:
            if index + 1 >= len(values) or not values[index + 1]:
                raise ValueError(f"clone option {token} requires a value")
            index += 2
            continue
        if command == "clone" and token.startswith(("--depth=", "--filter=")):
            if not token.partition("=")[2]:
                raise ValueError(f"invalid clone option: {token}")
            index += 1
            continue
        if token.startswith("-"):
            raise PermissionError(f"unsupported Git {command} option: {token}")
        positional.append(token)
        index += 1
    if command == "init":
        if len(positional) > 1:
            raise ValueError("git init accepts at most one destination")
        destination = positional[0] if positional else "."
    else:
        if not 1 <= len(positional) <= 2:
            raise ValueError("git clone requires a source and optional destination")
        source = positional[0]
        if not source or source.startswith("-"):
            raise ValueError("git clone source is invalid")
        parsed_source = urlsplit(source)
        if "::" in source or (
            parsed_source.scheme
            and parsed_source.scheme not in {"file", "git", "http", "https", "ssh"}
        ):
            raise PermissionError("git clone remote helper protocols are prohibited")
        if parsed_source.password or (
            parsed_source.scheme in {"http", "https"} and parsed_source.username
        ):
            raise PermissionError("git clone URLs must not contain credentials")
        destination = (
            positional[1]
            if len(positional) == 2
            else source.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        )
        if not destination:
            raise ValueError("git clone requires a destination name")
    root = Path(workspace).resolve()
    candidate = Path(destination)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PermissionError("Git destination escapes configured workspace") from exc
    return target
