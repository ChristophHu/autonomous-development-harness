from __future__ import annotations

import fnmatch
import ipaddress
import os
import re
import shlex
import socket
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, ValidationError

from .isolation import ProcessAccessProfile, git_metadata, isolated_command
from .process_control import current_run_control, run_cancellable
from .process_failures import process_failure_category


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: str
    risk_level: str
    handler: Callable[..., Any]
    output_schema: dict[str, Any] | None = None


class ToolExecutionError(RuntimeError):
    pass


def resolve_http_addresses(host, port):
    """Resolve a destination once for policy inspection; transports may resolve again."""
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        try:
            records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise PermissionError("HTTP destination could not be resolved") from exc
    return list({ipaddress.ip_address(record[4][0]) for record in records})


class ToolExecutor:
    def __init__(
        self,
        permissions,
        *,
        http_allow_hosts=(),
        http_private_hosts=(),
        http_resolver=resolve_http_addresses,
        http_timeout=30,
        git_allow_hosts=(),
        git_ca_bundle=None,
        git_credentials=None,
        git_ssh_allowed_hosts=(),
        git_ssh_allowed_ports=(22,),
        git_ssh_host_keys=None,
        git_ssh_credentials=None,
        docker_compose_file=None,
        docker_socket_path=None,
        env_allowlist=(),
    ):
        self.permissions = permissions
        self.http_allow_hosts = set(http_allow_hosts)
        self.http_private_hosts = {host.casefold() for host in http_private_hosts}
        self.http_resolver = http_resolver
        self.http_timeout = http_timeout
        self.git_allow_hosts = set(git_allow_hosts)
        self.git_ca_bundle = git_ca_bundle
        self.git_credentials = dict(git_credentials or {})
        self.git_ssh_allowed_hosts = set(git_ssh_allowed_hosts)
        self.git_ssh_allowed_ports = tuple(git_ssh_allowed_ports)
        self.git_ssh_host_keys = dict(git_ssh_host_keys or {})
        self.git_ssh_credentials = dict(git_ssh_credentials or {})
        self.docker_compose_file = docker_compose_file
        self.docker_socket_path = docker_socket_path
        self.env_allowlist = set(env_allowlist)

    def _environment(self):
        allowed = {"PATH", "TMPDIR", "LANG", "LC_ALL", *self.env_allowlist}
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in allowed and not key.startswith(("DYLD_", "LD_", "GIT_"))
        }
        # Never let host-wide or per-user Git settings weaken broker policy.
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        environment["GIT_CONFIG_GLOBAL"] = "/dev/null"
        return environment

    def shell(self, command, cwd=None, *, read_roots=None, write_roots=None):
        self.permissions.require("shell", write=True)
        args = ["/bin/zsh", "-lc", command] if isinstance(command, str) else command
        if not args or not all(isinstance(arg, str) and arg for arg in args):
            raise ValueError("command requires non-empty arguments")
        shell_text = command if isinstance(command, str) else " ".join(command)
        if any(Path(token).name == "git" for token in shlex.split(shell_text)):
            raise PermissionError(
                "Git commands must use the policy-controlled Git tool"
            )
        return run_cancellable(
            subprocess.run,
            isolated_command(
                args,
                cwd,
                access_profile=ProcessAccessProfile.workspace(
                    read_roots=read_roots or (), write_roots=write_roots or ()
                ),
            ),
            shell=False,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=self._environment(),
        )

    def _remote_url(self, destination, cwd, *, push=False):
        result = run_cancellable(
            subprocess.run,
            isolated_command(
                [
                    "git",
                    "remote",
                    "get-url",
                    *(["--push"] if push else []),
                    "--all",
                    destination,
                ],
                cwd,
                git=True,
            ),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=self._environment(),
        )
        if result.returncode or not result.stdout.strip():
            raise PermissionError("push remote cannot be resolved")
        destinations = result.stdout.strip().splitlines()
        if len(destinations) != 1:
            raise PermissionError("push requires a single actual destination")
        remote_url = destinations[0]
        parsed = urlsplit(remote_url)
        if parsed.password or (parsed.scheme in {"http", "https"} and parsed.username):
            raise PermissionError("remote URL must not contain credentials")
        return remote_url

    def git_target(self, task_id, args, cwd):
        from .approvals import GitApprovalTarget
        from .git_broker import local_path, push_targets, transport_index
        from .git_policy import git_operation

        _write, action, destination = git_operation(args)
        remote_url = ""
        identity = None
        ref_updates = ()
        if destination:
            targets = push_targets(args, transport_index(args))
            updates = []
            for source, target_branch in targets:
                if source is None:
                    continue
                resolved = run_cancellable(
                    subprocess.run,
                    isolated_command(
                        [
                            "git",
                            "rev-parse",
                            "--verify",
                            f"refs/heads/{source}^{{commit}}",
                        ],
                        cwd,
                        git=True,
                    ),
                    cwd=cwd,
                    env=self._environment(),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if resolved.returncode:
                    category = process_failure_category(resolved)
                    if category == "host_sandbox_blocked":
                        raise PermissionError(
                            "host sandbox blocked Git source verification"
                        )
                    raise PermissionError("Git source verification failed")
                if not re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved.stdout.strip()):
                    raise PermissionError(
                        f"push source branch is unavailable: {source}"
                    )
                updates.append((source, target_branch, resolved.stdout.strip()))
            ref_updates = tuple(updates)
            remote_url = self._remote_url(destination, cwd, push=True)
            parsed = urlsplit(remote_url)
            if parsed.scheme == "file" or (not parsed.scheme and ":" not in remote_url):
                path = local_path(remote_url, cwd)
                remote_url = str(path)
                if path.is_dir():
                    info = path.stat()
                    identity = (info.st_dev, info.st_ino)
        return GitApprovalTarget(
            task_id,
            str(cwd),
            tuple(args),
            action or "git.push",
            remote_url,
            identity,
            ref_updates,
        )

    def git(self, args, cwd, approval=None, task_id=None):
        from .approvals import ApprovalGrant
        from .audit import CURRENT_RUN
        from .git_broker import (
            HttpsTransport,
            LocalTransport,
            SshTransport,
            probe,
            transport_index,
        )
        from .git_policy import git_operation, validate_git_destination

        write, action, _destination = git_operation(args)
        validate_git_destination(args, cwd)
        self.permissions.require("git", write=write)
        if action:
            task_id = task_id or (CURRENT_RUN.get() or {}).get("task_id")
            if not task_id or not isinstance(approval, ApprovalGrant):
                raise PermissionError(
                    "Git action requires a recorded human approval and task"
                )
            target = self.git_target(task_id, args, cwd)
            if not approval.permits(action, target):
                raise PermissionError(
                    "Git action requires a matching recorded human approval"
                )
        transport = None
        if args[0] in {"clone", "fetch", "pull", "push"}:
            remote_url = target.remote_url if action else None
            if args[0] in {"fetch", "pull"}:
                remote_url = self._remote_url(args[transport_index(args)], cwd)
            if urlsplit(remote_url or args[transport_index(args)]).scheme == "https":
                transport = HttpsTransport.prepare(
                    args,
                    cwd,
                    self._environment(),
                    remote_url,
                    self.git_allow_hosts,
                    self.git_ca_bundle,
                    self.git_credentials,
                    target.ref_updates if action else (),
                )
            elif urlsplit(
                remote_url or args[transport_index(args)]
            ).scheme == "ssh" or (
                urlsplit(remote_url or args[transport_index(args)]).scheme == ""
                and ":" in (remote_url or args[transport_index(args)])
                and not (remote_url or args[transport_index(args)]).startswith(
                    ("/", "./", "../")
                )
            ):
                transport = SshTransport.prepare(
                    args,
                    cwd,
                    self._environment(),
                    remote_url,
                    self.git_ssh_allowed_hosts,
                    self.git_ssh_allowed_ports,
                    self.git_ssh_host_keys,
                    self.git_ssh_credentials,
                    os.environ.get("SSH_AUTH_SOCK"),
                    target.ref_updates if action else (),
                )
            else:
                transport = LocalTransport.prepare(
                    args,
                    cwd,
                    self._environment(),
                    remote_url,
                    target.ref_updates if action else (),
                )
            if (
                action
                and isinstance(transport, LocalTransport)
                and transport.identity != target.remote_identity
            ):
                raise PermissionError("local remote identity changed during preflight")
        command = isolated_command(
            ["git", "-c", "core.hooksPath=/dev/null", *args], cwd, git=True
        )
        if action:
            if transport is None:
                probe(
                    isolated_command(["git", "--version"], cwd, git=True),
                    cwd,
                    self._environment(),
                )
            if not approval.consume(action, target):
                raise PermissionError(
                    "Git action requires a matching recorded human approval"
                )
        if transport is not None:
            return transport.run()
        return run_cancellable(
            subprocess.run,
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=self._environment(),
        )

    def docker(self, action, tail=100):
        from .docker_broker import DockerComposeBroker

        if action not in {"status", "logs", "start", "stop"}:
            raise PermissionError("Docker broker action is not allowlisted")
        self.permissions.require("docker", write=action in {"start", "stop"})
        return DockerComposeBroker(
            self.docker_compose_file, self.docker_socket_path
        ).run(action, tail=tail)

    def test(self, command, cwd):
        return self.shell(command, cwd)

    def lint(self, command, cwd):
        return self.shell(command, cwd)

    def http(self, method, url, **kwargs):
        self.permissions.require("http")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or parsed.hostname not in self.http_allow_hosts
        ):
            raise PermissionError("HTTP destination is not allowlisted")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise PermissionError("HTTP destination has an invalid port") from exc
        secret_name = kwargs.pop("secret_name", None)
        if secret_name:
            from .security import SecretResolver

            secret = SecretResolver().get(secret_name)
            if not secret:
                raise PermissionError(f"secret '{secret_name}' is unavailable")
            kwargs.setdefault("headers", {})["Authorization"] = f"Bearer {secret}"
        try:
            addresses = self.http_resolver(parsed.hostname, port)
        except (OSError, ValueError) as exc:
            raise PermissionError("HTTP destination could not be resolved") from exc
        if not addresses:
            raise PermissionError("HTTP destination resolved to no addresses")
        if parsed.hostname.casefold() not in self.http_private_hosts and any(
            not address.is_global for address in addresses
        ):
            raise PermissionError("HTTP destination resolves to a non-public address")
        from .http_control import request

        return request(
            method,
            url,
            timeout=self.http_timeout,
            pinned_addresses=[(parsed.hostname, port, addresses)],
            **kwargs,
        )

    def filesystem(
        self, action, path, *, workspace, content=None, destination=None, query=None
    ):
        root = Path(workspace).resolve()
        candidate = Path(path)
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                "filesystem path escapes configured workspace"
            ) from exc
        if action in {"write", "create", "mkdir", "move", "copy", "delete"}:
            if ".git" in target.relative_to(root).parts or any(
                target == metadata
                or target.is_relative_to(metadata)
                or metadata.is_relative_to(target)
                for metadata in git_metadata(root)
            ):
                raise PermissionError(
                    "filesystem mutation inside Git metadata is prohibited"
                )
            self.permissions.require("filesystem", write=True)
        else:
            self.permissions.require("filesystem")
        if action == "read":
            return target.read_text()
        if action in {"write", "create"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            if action == "create":
                with target.open("x") as handle:
                    handle.write(content or "")
            else:
                target.write_text(content or "")
            return str(target)
        if action == "delete":
            self.permissions.require("filesystem.delete", write=True)
            target.unlink()
            return True
        if action == "list":
            return sorted(p.name for p in target.iterdir())
        if action == "mkdir":
            target.mkdir(parents=True, exist_ok=True)
            return str(target)
        if action in {"move", "copy"}:
            dest_candidate = Path(destination)
            dest = (
                dest_candidate
                if dest_candidate.is_absolute()
                else root / dest_candidate
            ).resolve()
            try:
                dest.relative_to(root)
            except ValueError as exc:
                raise PermissionError(
                    "destination escapes configured workspace"
                ) from exc
            if ".git" in dest.relative_to(root).parts or any(
                dest == metadata
                or dest.is_relative_to(metadata)
                or metadata.is_relative_to(dest)
                for metadata in git_metadata(root)
            ):
                raise PermissionError(
                    "filesystem mutation inside Git metadata is prohibited"
                )
            if action == "move":
                target.rename(dest)
            else:
                import shutil

                shutil.copy2(target, dest)
            return str(dest)
        if action == "exists":
            return target.exists()
        if action == "glob":
            return [str(p) for p in root.rglob("*") if fnmatch.fnmatch(p.name, path)]
        if action == "search":
            return [
                str(p)
                for p in root.rglob("*")
                if p.is_file() and query and query in p.read_text(errors="ignore")
            ]
        raise ValueError(f"unsupported filesystem action: {action}")


class ToolRegistry:
    def __init__(self, permissions, event_sink=None, workspace=None):
        self.permissions = permissions
        tool_config = permissions.config.data.get("tools", {})
        http = tool_config.get("http", {})
        git = tool_config.get("git", {})
        docker = tool_config.get("docker", {})
        self.executor = ToolExecutor(
            permissions,
            http_allow_hosts=http.get("allowed_hosts", []),
            http_private_hosts=http.get("private_hosts", []),
            http_timeout=http.get("timeout", 30),
            git_allow_hosts=git.get("allowed_hosts", []),
            git_ca_bundle=git.get("ca_bundle"),
            git_credentials=git.get("credentials", {}),
            git_ssh_allowed_hosts=git.get("ssh", {}).get("allowed_hosts", []),
            git_ssh_allowed_ports=git.get("ssh", {}).get("allowed_ports", [22]),
            git_ssh_host_keys=git.get("ssh", {}).get("host_keys", {}),
            git_ssh_credentials=git.get("ssh", {}).get("credentials", {}),
            docker_compose_file=docker.get("compose_file"),
            docker_socket_path=docker.get("socket_path"),
            env_allowlist=tool_config.get("env_allowlist", []),
        )
        self.event_sink = event_sink
        self.approvals = None
        self.workspace = Path(
            workspace or permissions.config.path("workspace")
        ).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.specs = {}
        self.recovery_paths = ContextVar("recovery_paths", default=None)
        self.register(
            ToolSpec(
                "filesystem.read",
                "Read a workspace file",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "filesystem",
                "READ",
                self.read_file,
            )
        )
        self.register(
            ToolSpec(
                "filesystem.search",
                "Search workspace text",
                {
                    "type": "object",
                    "properties": {
                        "root": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["root", "query"],
                    "additionalProperties": False,
                },
                "filesystem",
                "READ",
                self.search,
            )
        )
        from .search import RepositorySearch

        search = RepositorySearch(self.workspace)
        for name, description, key, handler in (
            ("search.text", "Search workspace text", "query", search.text_search),
            (
                "search.symbols",
                "Search source declarations",
                "query",
                search.symbol_search,
            ),
            (
                "search.files",
                "Search workspace filenames",
                "pattern",
                search.file_search,
            ),
        ):
            self.register(
                ToolSpec(
                    name,
                    description,
                    {
                        "type": "object",
                        "properties": {
                            key: {"type": "string", "minLength": 1},
                            "directory": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                        },
                        "required": [key],
                        "additionalProperties": False,
                    },
                    "filesystem",
                    "READ",
                    handler,
                )
            )
        process_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "cwd": {"type": "string"},
            },
            "required": ["command"],
            "additionalProperties": False,
        }
        for name, description in (
            ("test.run_tests", "Run workspace tests"),
            ("test.run_coverage", "Run workspace coverage"),
            ("quality.lint", "Run workspace lint"),
            ("quality.format", "Format workspace code"),
            ("quality.typecheck", "Run workspace typecheck"),
        ):
            self.register(
                ToolSpec(
                    name,
                    description,
                    process_schema,
                    "shell",
                    "WRITE",
                    lambda command, cwd=None: self.shell(command, cwd),
                )
            )
        self.register(
            ToolSpec(
                "test.run_file",
                "Run a selected workspace test file",
                {
                    "type": "object",
                    "properties": {
                        **process_schema["properties"],
                        "path": {"type": "string", "minLength": 1},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
                "shell",
                "WRITE",
                self.run_test_file,
            )
        )
        self.register(
            ToolSpec(
                "shell.execute",
                "Execute an approved local command",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                "shell",
                "WRITE",
                self.shell,
            )
        )
        for action, description, risk in (
            ("write", "Write a workspace file", "WRITE"),
            ("create", "Create a new workspace file", "WRITE"),
            ("delete", "Delete a workspace file", "DESTRUCTIVE"),
            ("list", "List workspace directory entries", "READ"),
            ("mkdir", "Create a workspace directory", "WRITE"),
            ("move", "Move a workspace file", "WRITE"),
            ("copy", "Copy a workspace file", "WRITE"),
            ("exists", "Check whether a workspace path exists", "READ"),
            ("glob", "Find workspace files by name pattern", "READ"),
        ):
            required = ["path"]
            if action in {"write", "create"}:
                required.append("content")
            if action in {"move", "copy"}:
                required.append("destination")
            self.register(
                ToolSpec(
                    f"filesystem.{action}",
                    description,
                    {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": required,
                        "additionalProperties": False,
                    },
                    "filesystem.delete" if action == "delete" else "filesystem",
                    risk,
                    lambda path, _action=action, **kwargs: self.executor.filesystem(
                        _action, path, workspace=self.workspace, **kwargs
                    ),
                )
            )
        self.register(
            ToolSpec(
                "git.execute",
                "Execute a Git command",
                {
                    "type": "object",
                    "properties": {
                        "args": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                    "required": ["args"],
                    "additionalProperties": False,
                },
                "git",
                "WRITE",
                lambda args, cwd=None: self.git(args, cwd, _audit=False),
            )
        )
        self.register(
            ToolSpec(
                "docker.execute",
                "Manage the Harness Qdrant Compose service",
                {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "logs", "start", "stop"],
                        },
                        "tail": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "docker",
                "WRITE",
                self.docker,
            )
        )
        self.register(
            ToolSpec(
                "http.request",
                "Call an HTTP service",
                {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                        },
                        "url": {"type": "string"},
                        "secret_name": {"type": "string"},
                        "json": {},
                        "headers": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["method", "url"],
                    "additionalProperties": False,
                },
                "http",
                "EXTERNAL",
                self.executor.http,
            )
        )
        self._configure_mcp()

    def _configure_mcp(self):
        from .mcp_manager import MCPServerManager

        raw = self.permissions.config.data.get("tools", {}).get("mcp", {})
        self.mcp_manager = MCPServerManager(raw, loader=self._discover_mcp_server)

    def _discover_mcp_server(self, server_name):
        from .mcp import (
            MCPClient,
            MCPHTTPClient,
            builtin_apple_shell_command,
            builtin_filesystem_command,
            builtin_obsidian_command,
        )
        from .mcp_servers.apple_shell import EXECUTABLES as APPLE_SHELL_EXECUTABLES
        from .mcp_servers.apple_shell import TOOLS as APPLE_SHELL_TOOLS
        from .mcp_servers.filesystem import _DELETE, _READ, _WRITE
        from .mcp_servers.obsidian import READ_TOOLS as OBSIDIAN_READ

        settings = self.mcp_manager.settings(server_name)
        if settings is None or not settings.enabled:
            return
        vault = (
            self.permissions.config.path("obsidian_vault")
            if settings.builtin == "obsidian"
            else None
        )
        if settings.transport == "streamable_http":
            token = None
            if settings.auth_secret:
                from .security import SecretResolver

                token = SecretResolver().get(settings.auth_secret)
                if not token:
                    raise ValueError("MCP server authentication is unavailable")
            client = MCPHTTPClient(
                settings.url,
                settings.allowed_hosts,
                timeout=settings.timeout,
                bearer_token=token,
            )
        else:
            if settings.builtin == "filesystem":
                command = builtin_filesystem_command(
                    self.workspace,
                    read_only=settings.read_only,
                    allow_delete=settings.allow_delete,
                )
            elif settings.builtin == "obsidian":
                command = builtin_obsidian_command(vault, self.workspace)
            elif settings.builtin == "apple_shell":
                if sys.platform != "darwin":
                    raise ValueError("Apple Shell MCP requires macOS")
                command = builtin_apple_shell_command(self.workspace)
            else:
                command = settings.command
            read_roots = (
                (
                    self.workspace,
                    Path(__file__).resolve().parents[1],
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                )
                + ((vault,) if vault is not None else ())
                if settings.builtin
                else tuple(
                    (self.workspace / path).resolve()
                    if not Path(path).is_absolute()
                    else Path(path).resolve()
                    for path in settings.read_roots
                )
            )
            client = MCPClient(
                command,
                self.workspace,
                timeout=settings.timeout,
                read_roots=read_roots,
                python_import_roots=(
                    Path(__file__).resolve().parents[1],
                    Path(sysconfig.get_paths()["purelib"]),
                )
                if settings.builtin
                else (),
                executable_paths=(
                    tuple(APPLE_SHELL_EXECUTABLES.values())
                    if settings.builtin == "apple_shell"
                    else ()
                ),
            )
        discovered = client.discover()
        server_specs = []
        for remote in discovered:
            name = remote["name"]
            if settings.builtin == "filesystem":
                if name in _READ:
                    permission, risk = "filesystem", "READ"
                elif name in _WRITE:
                    permission, risk = "filesystem", "WRITE"
                elif name in _DELETE:
                    permission, risk = "filesystem.delete", "DESTRUCTIVE"
                else:
                    raise ValueError("builtin MCP server advertised unknown tool")
            elif settings.builtin == "obsidian":
                if name not in OBSIDIAN_READ:
                    raise ValueError("builtin MCP server advertised unknown tool")
                permission, risk = "obsidian", "READ"
            elif settings.builtin == "apple_shell":
                if name not in APPLE_SHELL_TOOLS:
                    raise ValueError("Apple Shell MCP advertised unknown tool")
                permission, risk = "apple_shell", "READ"
            elif name in settings.allow_tools:
                permission, risk = f"mcp.{server_name}", "DESTRUCTIVE"
            else:
                continue
            schema = remote["inputSchema"]
            spec_name = f"mcp.{server_name}.{name}"
            server_specs.append(
                ToolSpec(
                    spec_name,
                    str(remote.get("description", "MCP tool"))[:500],
                    schema,
                    permission,
                    risk,
                    lambda _client=client, _name=name, _schema=schema, **kwargs: (
                        _client.call(_name, kwargs, _schema)
                    ),
                    output_schema=remote.get("outputSchema"),
                )
            )
        for spec in server_specs:
            self.register(spec)

    def register(self, spec: ToolSpec):
        Draft202012Validator.check_schema(spec.input_schema)
        self.specs[spec.name] = spec

    def list(self):
        return list(self.specs)

    def schemas(self, allowed_names):
        requested = set(allowed_names)
        for server_name in self.mcp_manager.names():
            prefix = f"mcp.{server_name}."
            if any(name.startswith(prefix) for name in requested):
                self.mcp_manager.start(server_name)
        unavailable = {
            item["name"]
            for item in self.mcp_manager.report()
            if item["state"] in {"disabled", "unavailable", "invalid_config"}
        }
        allowed_names = [
            name
            for name in allowed_names
            if not any(
                name.startswith(f"mcp.{server_name}.") for server_name in unavailable
            )
        ]
        if not set(allowed_names) <= set(self.specs):
            raise ValueError("profile references unknown tools")
        return [
            {
                "name": self.specs[name].name,
                "description": self.specs[name].description,
                "parameters": self.specs[name].input_schema,
            }
            for name in allowed_names
        ]

    def mcp_status(self, *, probe=False):
        if not probe:
            return self.mcp_manager.report()
        if probe:
            for server_name in self.mcp_manager.names():
                self.mcp_manager.start(server_name, force=True)
        return self.mcp_manager.report()

    def execute(
        self,
        name,
        arguments,
        profile=None,
        *,
        allow_nonzero=False,
        approval=None,
        task_id=None,
    ):
        control = current_run_control()
        if control is not None:
            control.check()
        spec = self.specs.get(name)
        if spec is None and name.startswith("mcp."):
            parts = name.split(".", 2)
            if len(parts) == 3:
                self.mcp_manager.start(parts[1])
                spec = self.specs.get(name)
        if not spec:
            if name.startswith("mcp."):
                parts = name.split(".", 2)
                if len(parts) == 3:
                    status = next(
                        (
                            item
                            for item in self.mcp_status()
                            if item["name"] == parts[1]
                        ),
                        None,
                    )
                    if status and status["state"] in {"unavailable", "invalid_config"}:
                        raise RuntimeError("MCP server is unavailable")
            raise KeyError(f"unknown tool: {name}")
        mcp_tool = name.startswith("mcp.")
        try:
            Draft202012Validator(spec.input_schema).validate(arguments)
        except ValidationError as exc:
            if mcp_tool:
                raise ValueError("invalid tool arguments") from None
            raise ValueError("invalid tool arguments: " + exc.message) from exc
        if profile and (
            name not in profile.tools
            or not ({name, spec.permission} & set(profile.permissions))
        ):
            raise PermissionError("tool is not permitted by profile")
        risk = spec.risk_level
        if name == "git.execute":
            from .git_policy import describe_git_operation

            risk = describe_git_operation(arguments["args"]).risk.value
        self.permissions.require(
            spec.permission, write=risk in {"WRITE", "DESTRUCTIVE"}
        )
        paths = self.recovery_paths.get()
        if paths is not None and risk != "READ":
            if name not in {"filesystem.write", "filesystem.create"}:
                raise PermissionError(
                    "unrestricted mutation prohibited during recovery"
                )
            target = (self.workspace / arguments["path"]).resolve()
            if target not in paths:
                raise PermissionError("write is outside recovery step scope")
        if name.startswith("filesystem.") and risk in {"WRITE", "DESTRUCTIVE"}:
            from .isolation import git_metadata

            path_values = [arguments.get("path"), arguments.get("destination")]
            for value in path_values:
                if isinstance(value, str):
                    target = (self.workspace / value).resolve()
                    relative = target.relative_to(self.workspace)
                    if ".git" in relative.parts or any(
                        target == metadata
                        or target.is_relative_to(metadata)
                        or metadata.is_relative_to(target)
                        for metadata in git_metadata(self.workspace)
                    ):
                        raise PermissionError(
                            "filesystem mutation inside Git metadata is prohibited"
                        )
        # Git mutations retain their stronger GitApprovalTarget contract.
        if risk in {"DESTRUCTIVE", "EXTERNAL"} and name != "git.execute":
            from .approvals import ApprovalGrant, ToolApprovalTarget
            from .audit import CURRENT_RUN

            task_id = task_id or (CURRENT_RUN.get() or {}).get("task_id")
            if task_id is not None and self.approvals is None:
                raise PermissionError(
                    "high-impact tool action requires task-bound human approval"
                )
            affected = [
                str((self.workspace / value).resolve())
                for key in ("path", "destination")
                if isinstance((value := arguments.get(key)), str)
            ]
            if task_id is not None:
                target = ToolApprovalTarget.create(task_id, name, arguments, affected)
                approval = approval or self.approvals.grant_tool_for_target(target)
                if not isinstance(approval, ApprovalGrant) or not approval.consume(
                    target.action, target
                ):
                    raise PermissionError(
                        "high-impact tool action requires a matching single-use approval"
                    )
        call_id = str(uuid.uuid4())
        if self.event_sink:
            self.event_sink(
                "TOOL_CALL_STARTED",
                {
                    "tool": name,
                    "risk": risk,
                    "call_id": call_id,
                    "input": {"argument_names": sorted(arguments)}
                    if mcp_tool
                    else arguments,
                },
            )
        try:
            result = spec.handler(**arguments)
            if control is not None:
                control.check()
            if hasattr(result, "raise_for_status"):
                result.raise_for_status()
            if getattr(result, "returncode", 0) and not allow_nonzero:
                raise ToolExecutionError(f"{name} exited with code {result.returncode}")
            if spec.output_schema:
                try:
                    Draft202012Validator(spec.output_schema).validate(result)
                except ValidationError:
                    if mcp_tool:
                        raise ToolExecutionError(
                            "MCP output schema violation"
                        ) from None
                    raise
            if self.event_sink:
                self.event_sink(
                    "TOOL_CALL_COMPLETED",
                    {
                        "tool": name,
                        "call_id": call_id,
                        "output": {"type": type(result).__name__}
                        if mcp_tool
                        else str(result),
                    },
                )
            return result
        except Exception as exc:
            if self.event_sink:
                self.event_sink(
                    "TOOL_CALL_FAILED",
                    {
                        "tool": name,
                        "call_id": call_id,
                        "error": type(exc).__name__ if mcp_tool else str(exc),
                    },
                )
            raise

    @contextmanager
    def recovery_writes(self, paths):
        token = self.recovery_paths.set(
            {(self.workspace / name).resolve() for name in paths}
        )
        try:
            yield
        finally:
            self.recovery_paths.reset(token)

    def read_file(self, path):
        return self.executor.filesystem("read", path, workspace=self.workspace)

    def _workspace_cwd(self, cwd=None):
        candidate = (
            (self.workspace / cwd).resolve()
            if cwd and not Path(cwd).is_absolute()
            else Path(cwd or self.workspace).resolve()
        )
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError("process cwd escapes configured workspace") from exc
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        return candidate

    def shell(self, command, cwd=None):
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        runtime_roots = tuple(
            dict.fromkeys(
                (
                    self.workspace,
                    Path(sys.prefix).resolve(strict=True),
                    Path(sys.base_prefix).resolve(strict=True),
                    temporary_root,
                )
            )
        )
        return self.executor.shell(
            command,
            cwd=self._workspace_cwd(cwd),
            read_roots=runtime_roots,
            write_roots=(temporary_root,),
        )

    def run_test_file(self, command, path, cwd=None):
        selected = (self.workspace / path).resolve()
        if not selected.is_relative_to(self.workspace):
            raise PermissionError("test file escapes configured workspace")
        if not selected.is_file():
            raise FileNotFoundError(selected)
        return self.shell([*command, str(selected)], cwd)

    def git(self, args, cwd=None, approval=None, task_id=None, _audit=True):
        from .git_policy import describe_git_operation, validate_git_destination

        directory = self._workspace_cwd(cwd)
        operation = describe_git_operation(args)
        destination = validate_git_destination(args, directory)
        if destination is not None:
            self.permissions.require("git", write=True)
        if not _audit or not self.event_sink:
            return self.executor.git(
                args, cwd=directory, approval=approval, task_id=task_id
            )
        call_id = str(uuid.uuid4())
        self.event_sink(
            "TOOL_CALL_STARTED",
            {
                "tool": "git.execute",
                "risk": operation.risk.value,
                "call_id": call_id,
                "input": {"args": args, "cwd": str(directory)},
            },
        )
        try:
            result = self.executor.git(
                args, cwd=directory, approval=approval, task_id=task_id
            )
        except Exception as error:
            self.event_sink(
                "TOOL_CALL_FAILED",
                {"tool": "git.execute", "call_id": call_id, "error": str(error)},
            )
            raise
        if result.returncode:
            self.event_sink(
                "TOOL_CALL_FAILED",
                {"tool": "git.execute", "call_id": call_id, "error": result.stderr},
            )
        else:
            self.event_sink(
                "TOOL_CALL_COMPLETED",
                {"tool": "git.execute", "call_id": call_id, "output": result.stdout},
            )
        return result

    def git_target(self, task_id, args, cwd=None):
        self.permissions.require("git")
        return self.executor.git_target(task_id, args, self._workspace_cwd(cwd))

    def docker(self, action, tail=100):
        return self.executor.docker(action, tail=tail)

    def search(self, root, query):
        selected = (self.workspace / root).resolve()
        try:
            selected.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError("search root escapes configured workspace") from exc
        return self.executor.filesystem("search", ".", workspace=selected, query=query)

    def git_status(self, root):
        self.permissions.require("git")
        return self.executor.git(
            ["status", "--short"], self._workspace_cwd(root)
        ).stdout
