"""Read-only Obsidian vault MCP server with workspace-style path confinement."""

from __future__ import annotations

import argparse
import os
import stat

from jsonschema import Draft202012Validator

from .filesystem import FilesystemServer, _schema, serve_stdio
from .filesystem import _response as base_response

MAX_SCAN = 1000
READ_TOOLS = {"read_note", "list_notes", "search_notes"}
_LIMIT = {"limit": {"type": "integer", "minimum": 1, "maximum": 100}}
TOOLS = {
    "read_note": _schema({"path": {"type": "string", "minLength": 1}}, ["path"]),
    "list_notes": _schema(_LIMIT, []),
    "search_notes": _schema(
        {"query": {"type": "string", "minLength": 1}, **_LIMIT}, ["query"]
    ),
}
OUTPUTS = {
    "read_note": _schema({"content": {"type": "string"}}, ["content"]),
    "list_notes": _schema(
        {"notes": {"type": "array", "items": {"type": "string"}}}, ["notes"]
    ),
    "search_notes": _schema(
        {"matches": {"type": "array", "items": {"type": "string"}}}, ["matches"]
    ),
}


class ObsidianServer:
    def __init__(self, vault):
        self.files = FilesystemServer(vault, read_only=True)

    def _parts(self, path):
        parts = self.files._parts(path)
        if (
            not parts
            or any(part.startswith(".") for part in parts)
            or not parts[-1].endswith(".md")
        ):
            raise ValueError("path must name a visible Markdown note")
        return parts

    def _notes(self):
        notes = []
        scanned = 0

        def walk(parts):
            nonlocal scanned
            with self.files._directory(parts) as directory:
                for name in sorted(os.listdir(directory)):
                    if name.startswith("."):
                        continue
                    scanned += 1
                    if scanned > MAX_SCAN:
                        raise ValueError("vault scan limit exceeded")
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    candidate = [*parts, name]
                    if stat.S_ISDIR(info.st_mode):
                        walk(candidate)
                    elif stat.S_ISREG(info.st_mode) and name.endswith(".md"):
                        notes.append("/".join(candidate))

        walk([])
        return sorted(notes)

    def call(self, name, arguments):
        if name not in TOOLS:
            raise ValueError("unknown Obsidian tool")
        Draft202012Validator(TOOLS[name]).validate(arguments)
        if name == "read_note":
            return {"content": self.files._read(self._parts(arguments["path"]))}
        notes = self._notes()
        limit = arguments.get("limit", 100)
        if name == "list_notes":
            return {"notes": notes[:limit]}
        matches = []
        for path in notes:
            try:
                if arguments["query"] in self.files._read(path.split("/")):
                    matches.append(path)
            except (OSError, UnicodeError, ValueError):
                continue
            if len(matches) >= limit:
                break
        return {"matches": matches}


def _response(server, message):
    return base_response(
        server,
        message,
        tools=TOOLS,
        outputs=OUTPUTS,
        server_name="harness-obsidian",
        description_prefix="Vault",
        error_text="Obsidian operation failed",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vault")
    args = parser.parse_args()
    serve_stdio(ObsidianServer(args.vault), _response)


if __name__ == "__main__":
    main()
