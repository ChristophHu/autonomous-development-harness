from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, ValidationError

from .isolation import git_metadata, isolated_command


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


class ToolExecutor:
    def __init__(self, permissions, *, http_allow_hosts=(), env_allowlist=()):
        self.permissions = permissions
        self.http_allow_hosts = set(http_allow_hosts)
        self.env_allowlist = set(env_allowlist)

    def _environment(self):
        allowed = {"PATH", "TMPDIR", "LANG", "LC_ALL", *self.env_allowlist}
        return {
            key: value
            for key, value in os.environ.items()
            if key in allowed and not key.startswith(("DYLD_", "LD_", "GIT_"))
        }

    def shell(self, command, cwd=None):
        self.permissions.require("shell", write=True)
        args = ["/bin/zsh", "-lc", command] if isinstance(command, str) else command
        if not args or not all(isinstance(arg, str) and arg for arg in args):
            raise ValueError("command requires non-empty arguments")
        shell_text = command if isinstance(command, str) else " ".join(command)
        if any(Path(token).name == "git" for token in shlex.split(shell_text)):
            raise PermissionError(
                "Git commands must use the policy-controlled Git tool"
            )
        return subprocess.run(
            isolated_command(args, cwd),
            shell=False,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=self._environment(),
        )

    def git_target(self, task_id, args, cwd):
        from urllib.parse import urlsplit

        from .approvals import GitApprovalTarget
        from .git_policy import git_operation

        _write, action, destination = git_operation(args)
        remote_url = ""
        if destination:
            result = subprocess.run(
                isolated_command(
                    ["git", "remote", "get-url", "--push", "--all", destination],
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
            if parsed.password or (
                parsed.scheme in {"http", "https"} and parsed.username
            ):
                raise PermissionError("remote URL must not contain credentials")
        return GitApprovalTarget(
            task_id, str(cwd), tuple(args), action or "git.push", remote_url
        )

    def git(self, args, cwd, approval=None, task_id=None):
        from .approvals import ApprovalGrant
        from .audit import CURRENT_RUN
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
            if not approval.consume(action, target):
                raise PermissionError(
                    "Git action requires a matching recorded human approval"
                )
        return subprocess.run(
            isolated_command(
                ["git", "-c", "core.hooksPath=/dev/null", *args], cwd, git=True
            ),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=self._environment(),
        )

    def docker(self, args, cwd=None):
        self.permissions.require("docker")
        return subprocess.run(
            isolated_command(["docker", *args], cwd),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )

    def test(self, command, cwd):
        return self.shell(command, cwd)

    def lint(self, command, cwd):
        return self.shell(command, cwd)

    def http(self, method, url, **kwargs):
        import httpx

        self.permissions.require("http")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or parsed.hostname not in self.http_allow_hosts
        ):
            raise PermissionError("HTTP destination is not allowlisted")
        secret_name = kwargs.pop("secret_name", None)
        if secret_name:
            from .security import SecretResolver

            secret = SecretResolver().get(secret_name)
            if not secret:
                raise PermissionError(f"secret '{secret_name}' is unavailable")
            kwargs.setdefault("headers", {})["Authorization"] = f"Bearer {secret}"
        return httpx.request(method, url, timeout=30, **kwargs)

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
        self.executor = ToolExecutor(
            permissions,
            http_allow_hosts=http.get("allowed_hosts", []),
            env_allowlist=tool_config.get("env_allowlist", []),
        )
        self.event_sink = event_sink
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
                "Execute Docker CLI command",
                {
                    "type": "object",
                    "properties": {
                        "args": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                    "required": ["args"],
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

    def register(self, spec: ToolSpec):
        Draft202012Validator.check_schema(spec.input_schema)
        self.specs[spec.name] = spec

    def list(self):
        return list(self.specs)

    def schemas(self, allowed_names):
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

    def execute(self, name, arguments, profile=None):
        spec = self.specs.get(name)
        if not spec:
            raise KeyError(f"unknown tool: {name}")
        try:
            Draft202012Validator(spec.input_schema).validate(arguments)
        except ValidationError as exc:
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
        call_id = str(uuid.uuid4())
        if self.event_sink:
            self.event_sink(
                "TOOL_CALL_STARTED",
                {
                    "tool": name,
                    "risk": risk,
                    "call_id": call_id,
                    "input": arguments,
                },
            )
        try:
            result = spec.handler(**arguments)
            if hasattr(result, "raise_for_status"):
                result.raise_for_status()
            if getattr(result, "returncode", 0):
                raise ToolExecutionError(f"{name} exited with code {result.returncode}")
            if spec.output_schema:
                Draft202012Validator(spec.output_schema).validate(result)
            if self.event_sink:
                self.event_sink(
                    "TOOL_CALL_COMPLETED",
                    {"tool": name, "call_id": call_id, "output": str(result)},
                )
            return result
        except Exception as exc:
            if self.event_sink:
                self.event_sink(
                    "TOOL_CALL_FAILED",
                    {"tool": name, "call_id": call_id, "error": str(exc)},
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
        return self.executor.shell(command, cwd=self._workspace_cwd(cwd))

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

    def docker(self, args, cwd=None):
        return self.executor.docker(args, cwd=self._workspace_cwd(cwd))

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
