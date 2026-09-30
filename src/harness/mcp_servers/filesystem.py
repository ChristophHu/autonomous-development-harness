"""Workspace-scoped MCP filesystem reference server (stdio transport)."""

from __future__ import annotations

import argparse
import errno
import fnmatch
import json
import os
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

PROTOCOL_VERSION = "2025-11-25"
MAX_FILE = 1_000_000
MAX_MESSAGE = 2_000_000
_READ = {"read_file", "list_directory", "search", "exists", "glob"}
_WRITE = {"write_file", "create_file", "create_directory", "copy_file"}
_DELETE = {"delete_file", "move_file"}


def _schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_PATH = {"path": {"type": "string", "minLength": 1}}
TOOLS = {
    "read_file": _schema(_PATH, ["path"]),
    "list_directory": _schema(_PATH, ["path"]),
    "search": _schema(
        {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        ["query"],
    ),
    "exists": _schema(_PATH, ["path"]),
    "glob": _schema(
        {
            "pattern": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        ["pattern"],
    ),
    "write_file": _schema(
        {**_PATH, "content": {"type": "string"}}, ["path", "content"]
    ),
    "create_file": _schema(
        {**_PATH, "content": {"type": "string"}}, ["path", "content"]
    ),
    "create_directory": _schema(_PATH, ["path"]),
    "delete_file": _schema(_PATH, ["path"]),
    "move_file": _schema(
        {**_PATH, "destination": {"type": "string", "minLength": 1}},
        ["path", "destination"],
    ),
    "copy_file": _schema(
        {**_PATH, "destination": {"type": "string", "minLength": 1}},
        ["path", "destination"],
    ),
}
OUTPUTS = {
    "read_file": _schema({"content": {"type": "string"}}, ["content"]),
    "list_directory": _schema(
        {"entries": {"type": "array", "items": {"type": "string"}}}, ["entries"]
    ),
    "search": _schema(
        {"matches": {"type": "array", "items": {"type": "string"}}}, ["matches"]
    ),
    "exists": _schema({"exists": {"type": "boolean"}}, ["exists"]),
    "glob": _schema(
        {"matches": {"type": "array", "items": {"type": "string"}}}, ["matches"]
    ),
    "write_file": _schema(_PATH, ["path"]),
    "create_file": _schema(_PATH, ["path"]),
    "create_directory": _schema(_PATH, ["path"]),
    "delete_file": _schema({"deleted": {"type": "boolean"}}, ["deleted"]),
    "move_file": _schema(_PATH, ["path"]),
    "copy_file": _schema(_PATH, ["path"]),
}


class FilesystemServer:
    def __init__(self, workspace, *, read_only=True, allow_delete=False):
        self.root = Path(workspace).resolve(strict=True)
        if not self.root.is_dir() or self.root == Path(self.root.anchor):
            raise ValueError("invalid workspace root")
        self.read_only = read_only
        self.allow_delete = allow_delete

    @staticmethod
    def _parts(path):
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise ValueError("path must be relative")
        parts = path.split("/")
        if any(part in {"", "..", ".git"} for part in parts):
            raise PermissionError("path is outside allowed workspace")
        return [part for part in parts if part != "."]

    @contextmanager
    def _directory(self, parts, *, create=False):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = next_fd
            yield fd
        finally:
            os.close(fd)

    def _read(self, parts):
        with self._directory(parts[:-1]) as parent:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE:
                    raise ValueError("file is unsupported or too large")
                raw = os.read(fd, MAX_FILE + 1)
                if len(raw) > MAX_FILE:
                    raise ValueError("file is too large")
                return raw.decode("utf-8")
            finally:
                os.close(fd)

    def _write(self, parts, content, *, create):
        raw = content.encode("utf-8")
        if len(raw) > MAX_FILE:
            raise ValueError("file is too large")
        with self._directory(parts[:-1], create=True) as parent:
            if create:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                fd = os.open(parts[-1], flags, 0o600, dir_fd=parent)
                completed = False
                try:
                    self._write_all(fd, raw)
                    completed = True
                finally:
                    os.close(fd)
                    if not completed:
                        os.unlink(parts[-1], dir_fd=parent)
            else:
                temporary = ".harness-mcp-" + uuid.uuid4().hex
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
                try:
                    self._write_all(fd, raw)
                    os.close(fd)
                    fd = -1
                    os.rename(
                        temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent
                    )
                finally:
                    if fd != -1:
                        os.close(fd)
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass
        return {"path": "/".join(parts)}

    @staticmethod
    def _write_all(fd, raw):
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("filesystem write failed")
            view = view[written:]

    def _search(self, query, limit):
        found = []

        def walk(parts):
            with self._directory(parts) as directory:
                entries = sorted(os.listdir(directory))
                for name in entries:
                    if name == ".git":
                        continue
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    candidate = [*parts, name]
                    if stat.S_ISDIR(info.st_mode):
                        walk(candidate)
                    elif stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE:
                        try:
                            if query in self._read(candidate):
                                found.append("/".join(candidate))
                        except (UnicodeError, OSError, ValueError):
                            pass
                    if len(found) >= limit:
                        return

        walk([])
        return {"matches": found}

    def _exists(self, parts):
        if not parts:
            return {"exists": True}
        try:
            with self._directory(parts[:-1]) as parent:
                info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                return {"exists": not stat.S_ISLNK(info.st_mode)}
        except (FileNotFoundError, NotADirectoryError):
            return {"exists": False}
        except OSError as error:
            if error.errno == errno.ELOOP:
                return {"exists": False}
            raise

    def _glob(self, pattern, limit):
        found = []

        def walk(parts):
            with self._directory(parts) as directory:
                for name in sorted(os.listdir(directory)):
                    if name == ".git":
                        continue
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    candidate = [*parts, name]
                    if stat.S_ISDIR(info.st_mode):
                        walk(candidate)
                    elif stat.S_ISREG(info.st_mode):
                        path = "/".join(candidate)
                        if fnmatch.fnmatchcase(
                            path if "/" in pattern else name, pattern
                        ):
                            found.append(path)
                    if len(found) >= limit:
                        return

        walk([])
        return {"matches": found[:limit]}

    def call(self, name, arguments):
        if name not in TOOLS:
            raise ValueError("unknown filesystem tool")
        Draft202012Validator(TOOLS[name]).validate(arguments)
        if name in _WRITE and self.read_only:
            raise PermissionError("filesystem is read-only")
        if name in _DELETE and (self.read_only or not self.allow_delete):
            raise PermissionError("filesystem deletion is disabled")
        if name == "search":
            return self._search(arguments["query"], arguments.get("limit", 100))
        if name == "glob":
            pattern = "/".join(self._parts(arguments["pattern"]))
            if not pattern:
                raise ValueError("glob pattern must name files")
            return self._glob(pattern, arguments.get("limit", 100))
        parts = self._parts(arguments["path"])
        if not parts and name not in {"list_directory", "exists"}:
            raise ValueError("path must name an entry")
        if name == "exists":
            return self._exists(parts)
        if name == "copy_file":
            destination = self._parts(arguments["destination"])
            if not destination or destination == parts:
                raise ValueError("copy destination must differ from source")
            return self._write(destination, self._read(parts), create=True)
        if name == "move_file":
            destination = self._parts(arguments["destination"])
            if not destination or destination == parts:
                raise ValueError("move destination must differ from source")
            content = self._read(parts)
            self._write(destination, content, create=True)
            self.call("delete_file", {"path": arguments["path"]})
            return {"path": "/".join(destination)}
        if name == "read_file":
            return {"content": self._read(parts)}
        if name == "list_directory":
            with self._directory(parts) as directory:
                return {
                    "entries": sorted(
                        name for name in os.listdir(directory) if name != ".git"
                    )
                }
        if name in {"write_file", "create_file"}:
            return self._write(
                parts, arguments["content"], create=name == "create_file"
            )
        if name == "create_directory":
            with self._directory(parts, create=True):
                return {"path": "/".join(parts)}
        with self._directory(parts[:-1]) as parent:
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("only regular files may be deleted")
            os.unlink(parts[-1], dir_fd=parent)
        return {"deleted": True}


def _response(server, message):
    if not isinstance(message, dict):
        return None
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    identifier = message.get("id")
    if identifier is None or message.get("jsonrpc") != "2.0":
        return None
    if method == "initialize":
        params = message.get("params")
        if (
            not isinstance(params, dict)
            or params.get("protocolVersion") != PROTOCOL_VERSION
        ):
            return {
                "jsonrpc": "2.0",
                "id": identifier,
                "error": {"code": -32602, "message": "unsupported protocol version"},
            }
        return {
            "jsonrpc": "2.0",
            "id": identifier,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "harness-filesystem", "version": "0.1.0"},
            },
        }
    if method == "tools/list":
        tools = [
            {
                "name": name,
                "description": f"Workspace {name}",
                "inputSchema": schema,
                "outputSchema": OUTPUTS[name],
            }
            for name, schema in TOOLS.items()
        ]
        return {"jsonrpc": "2.0", "id": identifier, "result": {"tools": tools}}
    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        try:
            result = server.call(params.get("name"), params.get("arguments") or {})
        except (OSError, ValueError, PermissionError, UnicodeError, ValidationError):
            return {
                "jsonrpc": "2.0",
                "id": identifier,
                "result": {
                    "content": [
                        {"type": "text", "text": "filesystem operation failed"}
                    ],
                    "isError": True,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": identifier,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result)}],
                "structuredContent": result,
                "isError": False,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": identifier,
        "error": {"code": -32601, "message": "method not found"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--allow-delete", action="store_true")
    args = parser.parse_args()
    server = FilesystemServer(
        args.workspace, read_only=args.read_only, allow_delete=args.allow_delete
    )
    for line in sys.stdin.buffer:
        if len(line) > MAX_MESSAGE:
            break
        try:
            response = _response(server, json.loads(line))
        except (ValueError, TypeError):
            response = None
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
