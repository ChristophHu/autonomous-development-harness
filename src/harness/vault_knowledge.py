"""Read-only structured knowledge queries over a confined Obsidian vault."""

from __future__ import annotations

import re

from .mcp_servers.obsidian import ObsidianServer

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$")
_LINK = re.compile(r"!?\[\[([^\]|#]+)")


class VaultKnowledgeService:
    """Provide bounded note, section, link and backlink retrieval without writes."""

    def __init__(self, vault, *, _server=None):
        self._vault = _server or ObsidianServer(vault)

    def _read(self, path):
        parts = self._vault._parts(path)
        return self._vault.files._read(parts)

    @staticmethod
    def _metadata(text):
        match = _FRONTMATTER.match(text)
        if match is None:
            return {}
        import yaml

        try:
            value = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise ValueError("note frontmatter is invalid") from exc
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("note frontmatter must be a mapping")
        return value

    def get_note(self, path):
        text = self._read(path)
        match = _FRONTMATTER.match(text)
        body = text[match.end() :] if match else text
        return {
            "path": path,
            "metadata": self._metadata(text),
            "headings": _HEADING.findall(body),
            "links": sorted({link.strip() for link in _LINK.findall(body)}),
            "content": body,
        }

    def backlinks(self, path, *, limit=100):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be between 1 and 100")
        target = path.rsplit("/", 1)[-1].removesuffix(".md").casefold()
        matches = []
        for candidate in self._vault._notes():
            try:
                text = self._read(candidate)
            except (OSError, UnicodeError, ValueError):
                continue
            if any(
                link.rsplit("/", 1)[-1].casefold() == target
                for link in _LINK.findall(text)
            ):
                matches.append(candidate)
                if len(matches) >= limit:
                    break
        return matches

    def search(self, query, *, limit=20, context_chars=160):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be between 1 and 100")
        if (
            isinstance(context_chars, bool)
            or not isinstance(context_chars, int)
            or not 20 <= context_chars <= 1000
        ):
            raise ValueError("context_chars must be between 20 and 1000")
        matches = []
        needle = query.casefold()
        for path in self._vault._notes():
            try:
                text = self._read(path)
            except (OSError, UnicodeError, ValueError):
                continue
            offset = text.casefold().find(needle)
            if offset < 0:
                continue
            start = max(0, offset - context_chars // 2)
            end = min(len(text), offset + len(query) + context_chars // 2)
            matches.append({"path": path, "excerpt": text[start:end]})
            if len(matches) >= limit:
                break
        return matches
