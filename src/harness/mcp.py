"""Bounded, explicitly configured MCP stdio client for the tool registry."""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from .isolation import isolated_command
from .process_control import current_run_control

PROTOCOL_VERSION = "2025-11-25"
MAX_MESSAGE = 2_000_000
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


class MCPError(RuntimeError):
    """A redacted transport or remote-tool failure."""


class MCPClient:
    def __init__(self, command, workspace, *, timeout=10):
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

    def _exchange(self, method, params):
        command = isolated_command(self.command, self.workspace)
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
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=1)
            process.stdin.close()
            process.stdout.close()

    def discover(self):
        return self._exchange("tools/list", {})

    def call(self, name, arguments, expected_schema):
        return self._exchange(
            "tools/call",
            {"name": name, "arguments": arguments, "expected": expected_schema},
        )


def builtin_filesystem_command(workspace, *, read_only=True, allow_delete=False):
    command = [sys.executable, "-m", "harness.mcp_servers.filesystem", str(workspace)]
    if read_only:
        command.append("--read-only")
    if allow_delete:
        command.append("--allow-delete")
    return command
