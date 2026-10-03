"""Bounded, explicitly configured MCP stdio client for the tool registry."""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .http_control import request as http_request
from .isolation import ProcessAccessProfile, isolated_command
from .process_control import _terminate, current_run_control

PROTOCOL_VERSION = "2025-11-25"
MAX_MESSAGE = 2_000_000
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


class MCPError(RuntimeError):
    """A redacted transport or remote-tool failure."""


class MCPClient:
    def __init__(
        self, command, workspace, *, timeout=10, read_roots=None, executable_paths=()
    ):
        self.workspace = Path(workspace).resolve(strict=True)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
            or not Path(command[0]).is_absolute()
            or Path(command[0]).resolve().is_relative_to(self.workspace)
        ):
            raise ValueError(
                "MCP command requires an absolute executable outside workspace"
            )
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 0 < timeout <= 30
        ):
            raise ValueError("MCP timeout must be between 0 and 30 seconds")
        self.command = command
        self.timeout = timeout
        self.read_roots = read_roots
        self.executable_paths = tuple(Path(path) for path in executable_paths)

    def _exchange(self, method, params):
        control = current_run_control()
        if control is not None:
            control.check()
        command = isolated_command(
            self.command,
            self.workspace,
            access_profile=ProcessAccessProfile.mcp_stdio(
                read_roots=self.read_roots or (),
                executable_paths=self.executable_paths,
            ),
        )
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        if "TMPDIR" in os.environ:
            env["TMPDIR"] = os.environ["TMPDIR"]
        try:
            process = subprocess.Popen(
                command,
                cwd=self.workspace,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise MCPError("MCP server could not start") from exc
        if control is not None:
            control.register(process)
        os.set_blocking(process.stdin.fileno(), False)
        deadline = time.monotonic() + self.timeout
        pending = bytearray()
        next_id = 0

        def request(name, payload):
            nonlocal next_id
            next_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": next_id,
                "method": name,
                "params": payload,
            }
            encoded = json.dumps(message, separators=(",", ":")).encode() + b"\n"
            if len(encoded) > MAX_MESSAGE:
                raise MCPError("MCP request is too large")
            offset = 0
            while offset < len(encoded):
                control = current_run_control()
                if control is not None:
                    control.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError("MCP request timed out")
                _, writable, _ = select.select(
                    [], [process.stdin], [], min(0.1, remaining)
                )
                if writable:
                    offset += os.write(process.stdin.fileno(), encoded[offset:])
            while True:
                if b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending[:] = rest
                    try:
                        reply = json.loads(line)
                    except (UnicodeError, ValueError) as exc:
                        raise MCPError("invalid MCP response") from exc
                    if (
                        not isinstance(reply, dict)
                        or reply.get("jsonrpc") != "2.0"
                        or reply.get("id") != next_id
                    ):
                        raise MCPError("invalid MCP response identity")
                    if "error" in reply or not isinstance(reply.get("result"), dict):
                        raise MCPError("MCP request failed")
                    return reply["result"]
                control = current_run_control()
                if control is not None:
                    control.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError("MCP request timed out")
                readable, _, _ = select.select(
                    [process.stdout], [], [], min(0.1, remaining)
                )
                if not readable:
                    if process.poll() is not None:
                        raise MCPError("MCP server stopped")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise MCPError("MCP server closed the connection")
                pending.extend(chunk)
                if len(pending) > MAX_MESSAGE:
                    raise MCPError("MCP response is too large")

        try:
            initialized = request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "autonomous-development-harness",
                        "version": "0.1.0",
                    },
                },
            )
            if initialized.get(
                "protocolVersion"
            ) != PROTOCOL_VERSION or "tools" not in initialized.get("capabilities", {}):
                raise MCPError("incompatible MCP server")
            process.stdin.write(
                b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            )
            process.stdin.flush()
            discovered = request("tools/list", {})
            tools = discovered.get("tools")
            if not isinstance(tools, list) or len(tools) > 100:
                raise MCPError("invalid MCP tool list")
            names = set()
            for tool in tools:
                if (
                    not isinstance(tool, dict)
                    or not isinstance(tool.get("name"), str)
                    or not _NAME.fullmatch(tool["name"])
                    or tool["name"] in names
                ):
                    raise MCPError("invalid MCP tool name")
                names.add(tool["name"])
                schema = tool.get("inputSchema")
                if not isinstance(schema, dict):
                    raise MCPError("invalid MCP input schema")
                try:
                    Draft202012Validator.check_schema(schema)
                except Exception as exc:
                    raise MCPError("invalid MCP input schema") from exc
                output_schema = tool.get("outputSchema")
                if output_schema is not None:
                    if not isinstance(output_schema, dict):
                        raise MCPError("invalid MCP output schema")
                    try:
                        Draft202012Validator.check_schema(output_schema)
                    except Exception as exc:
                        raise MCPError("invalid MCP output schema") from exc
            if method == "tools/list":
                return tools
            name = params["name"]
            expected = params["expected"]
            matching = next((tool for tool in tools if tool["name"] == name), None)
            if matching is None or matching["inputSchema"] != expected:
                raise MCPError("MCP tool schema changed")
            result = request(
                "tools/call", {"name": name, "arguments": params["arguments"]}
            )
            if result.get("isError") is True:
                raise MCPError("MCP tool reported failure")
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                return structured
            blocks = result.get("content")
            if (
                not isinstance(blocks, list)
                or not blocks
                or not all(
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    for block in blocks
                )
            ):
                raise MCPError("MCP tool returned no usable result")
            return {"content": "\n".join(block["text"] for block in blocks)}
        except (OSError, BrokenPipeError) as exc:
            raise MCPError("MCP transport failed") from exc
        finally:
            try:
                if process.poll() is None:
                    _terminate(process, 0.2)
            finally:
                if control is not None:
                    control.unregister(process)
                process.stdin.close()
                process.stdout.close()

    def discover(self):
        return self._exchange("tools/list", {})

    def call(self, name, arguments, expected_schema):
        return self._exchange(
            "tools/call",
            {"name": name, "arguments": arguments, "expected": expected_schema},
        )


class MCPHTTPClient:
    """Pinned HTTPS Streamable-HTTP MCP client with bounded responses."""

    def __init__(self, url, allowed_hosts, *, timeout=10, bearer_token=None):
        parsed = urlsplit(url) if isinstance(url, str) else None
        try:
            port = parsed.port if parsed is not None else None
        except ValueError:
            raise ValueError(
                "remote MCP endpoint must be HTTPS and host-allowlisted"
            ) from None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
            or not isinstance(allowed_hosts, (list, tuple, set))
            or parsed.hostname.casefold()
            not in {host.casefold() for host in allowed_hosts if isinstance(host, str)}
        ):
            raise ValueError("remote MCP endpoint must be HTTPS and host-allowlisted")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 0 < timeout <= 30
        ):
            raise ValueError("MCP timeout must be between 0 and 30 seconds")
        if bearer_token is not None and (
            not isinstance(bearer_token, str) or not bearer_token
        ):
            raise ValueError("remote MCP bearer token is invalid")
        self.url = url
        self.timeout = timeout
        self.headers = {"Accept": "application/json, text/event-stream"}
        if bearer_token is not None:
            self.headers["Authorization"] = f"Bearer {bearer_token}"
        self.session_id = None
        self._next_id = 0
        self._initialized = False
        self._lock = threading.RLock()

    def _post(self, body, *, notification=False):
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        if self.session_id:
            headers["MCP-Session-Id"] = self.session_id
        try:
            response = http_request(
                "POST",
                self.url,
                timeout=self.timeout,
                headers=headers,
                json=body,
            )
            if response.is_redirect:
                raise MCPError("remote MCP redirect refused")
            if notification and response.status_code == 202:
                return None
            response.raise_for_status()
            if len(response.content) > MAX_MESSAGE:
                raise MCPError("MCP response is too large")
            session_id = response.headers.get("MCP-Session-Id")
            if session_id:
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", session_id):
                    raise MCPError("invalid MCP session identity")
                self.session_id = session_id
            content_type = response.headers.get("content-type", "").casefold()
            if "text/event-stream" in content_type:
                data = [
                    line[5:].strip()
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                ]
                if not data:
                    raise MCPError("invalid MCP event stream")
                payload = json.loads(data[-1])
            elif "application/json" in content_type:
                payload = response.json()
            else:
                raise MCPError("remote MCP content type is invalid")
        except MCPError:
            raise
        except (httpx.HTTPError, UnicodeError, ValueError, TypeError):
            raise MCPError("remote MCP transport failed") from None
        if notification:
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("jsonrpc") != "2.0"
            or payload.get("id") != body.get("id")
            or "error" in payload
            or not isinstance(payload.get("result"), dict)
        ):
            raise MCPError("invalid remote MCP response")
        return payload["result"]

    def _rpc(self, method, params):
        self._next_id += 1
        return self._post(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        )

    def _initialize(self):
        if self._initialized:
            return
        initialized = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "autonomous-development-harness",
                    "version": "0.1.0",
                },
            },
        )
        if initialized.get(
            "protocolVersion"
        ) != PROTOCOL_VERSION or "tools" not in initialized.get("capabilities", {}):
            raise MCPError("incompatible remote MCP server")
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            notification=True,
        )
        self._initialized = True

    def discover(self):
        with self._lock:
            self._initialize()
            result = self._rpc("tools/list", {})
            tools = result.get("tools")
            if not isinstance(tools, list) or len(tools) > 100:
                raise MCPError("invalid remote MCP tool list")
            names = set()
            for tool in tools:
                if (
                    not isinstance(tool, dict)
                    or not isinstance(tool.get("name"), str)
                    or not _NAME.fullmatch(tool["name"])
                    or tool["name"] in names
                    or not isinstance(tool.get("inputSchema"), dict)
                ):
                    raise MCPError("invalid remote MCP tool schema")
                names.add(tool["name"])
                try:
                    Draft202012Validator.check_schema(tool["inputSchema"])
                    if tool.get("outputSchema") is not None:
                        Draft202012Validator.check_schema(tool["outputSchema"])
                except (SchemaError, ValueError, TypeError):
                    raise MCPError("invalid remote MCP tool schema") from None
            return tools

    def call(self, name, arguments, expected_schema):
        with self._lock:
            tools = self.discover()
            matching = next((tool for tool in tools if tool["name"] == name), None)
            if matching is None or matching["inputSchema"] != expected_schema:
                raise MCPError("remote MCP tool schema changed")
            result = self._rpc("tools/call", {"name": name, "arguments": arguments})
            if result.get("isError") is True:
                raise MCPError("remote MCP tool reported failure")
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                output_schema = matching.get("outputSchema")
                if output_schema is not None:
                    try:
                        Draft202012Validator(output_schema).validate(structured)
                    except (ValidationError, SchemaError, ValueError, TypeError):
                        raise MCPError(
                            "remote MCP tool returned invalid output"
                        ) from None
                return structured
            blocks = result.get("content")
            if (
                not isinstance(blocks, list)
                or not blocks
                or not all(
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    for block in blocks
                )
            ):
                raise MCPError("remote MCP tool returned no usable result")
            return {"content": "\n".join(block["text"] for block in blocks)}


def builtin_filesystem_command(workspace, *, read_only=True, allow_delete=False):
    command = [sys.executable, "-m", "harness.mcp_servers.filesystem", str(workspace)]
    if read_only:
        command.append("--read-only")
    if allow_delete:
        command.append("--allow-delete")
    return command


def builtin_obsidian_command(vault):
    return [sys.executable, "-m", "harness.mcp_servers.obsidian", str(vault)]


def builtin_apple_shell_command(workspace):
    return [sys.executable, "-m", "harness.mcp_servers.apple_shell", str(workspace)]
