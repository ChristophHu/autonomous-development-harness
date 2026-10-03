"""Read-only structured knowledge queries over a confined Obsidian vault."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import UTC, date, datetime
from pathlib import Path

from .mcp_servers.obsidian import ObsidianServer
from .vault_review import classify_review

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$")
_LINK = re.compile(r"!?\[\[([^\]|#]+)")


class VaultKnowledgeService:
    """Provide bounded note, section, link and backlink retrieval without writes."""

    def __init__(
        self,
        vault,
        *,
        source_root=None,
        _server=None,
        today=None,
        max_review_age_days=180,
    ):
        if (
            isinstance(max_review_age_days, bool)
            or not isinstance(max_review_age_days, int)
            or max_review_age_days < 0
        ):
            raise ValueError("max_review_age_days must be a non-negative integer")
        if today is not None and not isinstance(today, date):
            raise ValueError("today must be a date")
        self._vault = _server or ObsidianServer(vault)
        self._source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._today = today or datetime.now(UTC).date()
        self._max_review_age_days = max_review_age_days

    def _review_status(self, metadata):
        return classify_review(metadata, self._today, self._max_review_age_days)

    def _source_checks(self, metadata):
        declarations = metadata.get("sources", [])
        if declarations is None:
            declarations = []
        if not isinstance(declarations, list):
            return "invalid", [{"path": "", "status": "invalid"}]
        hashes = metadata.get("source_hashes", {})
        if isinstance(hashes, dict):
            declarations = [
                *declarations,
                *({"path": path, "sha256": digest} for path, digest in hashes.items()),
            ]
        if not declarations:
            return "unverified", []
        if len(declarations) > 32:
            return "invalid", [{"path": "", "status": "invalid"}]
        checks = []
        for declaration in declarations:
            if not isinstance(declaration, dict) or set(declaration) != {
                "path",
                "sha256",
            }:
                checks.append({"path": "", "status": "invalid"})
                continue
            relative, digest = declaration["path"], declaration["sha256"]
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or "\\" in relative
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                checks.append({"path": str(relative), "status": "invalid"})
                continue
            candidate = self._source_root / relative
            try:
                current = self._source_root
                unsafe = False
                for part in Path(relative).parts:
                    current = current / part
                    if current.is_symlink():
                        unsafe = True
                        break
                resolved = candidate.resolve(strict=True)
                if (
                    unsafe
                    or not resolved.is_relative_to(self._source_root)
                    or not stat.S_ISREG(resolved.stat().st_mode)
                ):
                    checks.append({"path": relative, "status": "invalid"})
                    continue
                with resolved.open("rb") as source_file:
                    actual = hashlib.file_digest(source_file, "sha256").hexdigest()
            except (OSError, ValueError):
                checks.append({"path": relative, "status": "unavailable"})
                continue
            checks.append(
                {
                    "path": relative,
                    "status": "current" if actual == digest else "stale",
                }
            )
        states = {item["status"] for item in checks}
        state = next(
            (
                candidate
                for candidate in ("invalid", "unavailable", "stale")
                if candidate in states
            ),
            "current",
        )
        return state, checks

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

        def encode_date(item):
            if isinstance(item, date):
                return item.isoformat()
            raise TypeError("note frontmatter contains a non-JSON value")

        return json.loads(
            json.dumps(value, default=encode_date, ensure_ascii=False, allow_nan=False)
        )

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
        terms = tuple(dict.fromkeys(re.findall(r"\w+", query.casefold())))
        if not terms:
            raise ValueError("query must contain searchable characters")
        for path in self._vault._notes():
            try:
                text = self._read(path)
            except (OSError, UnicodeError, ValueError):
                continue
            frontmatter = _FRONTMATTER.match(text)
            body_start = frontmatter.end() if frontmatter else 0
            body = text[body_start:]
            folded = body.casefold()
            offsets = [folded.find(term) for term in terms]
            offsets = [offset for offset in offsets if offset >= 0]
            if not offsets:
                continue
            offset = offsets[0]
            headings = list(_HEADING.finditer(body))
            section_match = next(
                (
                    heading
                    for index, heading in enumerate(headings)
                    if heading.start() <= offset
                    and (
                        index + 1 == len(headings)
                        or offset < headings[index + 1].start()
                    )
                ),
                None,
            )
            section = section_match.group(1) if section_match else None
            try:
                metadata = self._metadata(text)
            except (TypeError, ValueError):
                # Malformed metadata cannot safely contribute claims or provenance.
                continue
            source_hashes = metadata.get("source_hashes", {})
            if not isinstance(source_hashes, dict):
                source_hashes = {}
            source_state, source_checks = self._source_checks(metadata)
            review_state, reviewed_on = self._review_status(metadata)
            source_ref = f"context:vault/{path}"
            if section:
                source_ref += f"#{section}"
            score = sum(folded.count(term) for term in terms)
            stem = path.rsplit("/", 1)[-1].removesuffix(".md").casefold()
            if any(term in stem for term in terms):
                score += 5
            start = max(0, offset - context_chars // 2)
            end = min(len(body), offset + len(terms[0]) + context_chars // 2)
            matches.append(
                {
                    "path": path,
                    "source_ref": source_ref,
                    "section": section,
                    "excerpt": body[start:end],
                    "review_state": review_state,
                    "provenance": {
                        "reviewed_on": reviewed_on,
                        "review_state": review_state,
                        "source_hashes": source_hashes,
                        "source_state": source_state,
                        "source_checks": source_checks,
                    },
                    "claims": (
                        metadata.get("claims", {})
                        if review_state == "current"
                        and isinstance(metadata.get("claims", {}), dict)
                        else {}
                    ),
                    "_score": score,
                }
            )
        matches.sort(key=lambda item: (-item["_score"], item["path"].casefold()))
        return [
            {key: value for key, value in item.items() if key != "_score"}
            for item in matches[:limit]
        ]
