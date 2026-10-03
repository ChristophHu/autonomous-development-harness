"""Human-approved promotion of verified task knowledge into the curated Vault."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .observability import redact_log_message

_CATEGORIES = {
    "project-knowledge",
    "lessons-learned",
    "known-problems",
    "conventions",
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[0-9a-f]{24}\Z")
_REVIEWED = re.compile(r"(?m)^last_reviewed:\s*[^\r\n]+$")


class VaultKnowledgeSteward:
    """Store immutable proposals and apply them only after digest-bound approval."""

    def __init__(self, vault, source_root, *, redactor=None, create_inbox=True):
        try:
            self.vault = Path(vault).resolve(strict=True)
            self.source_root = Path(source_root).resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("vault and source root must exist") from error
        if not self.vault.is_dir() or not self.source_root.is_dir():
            raise ValueError("vault and source root must be directories")
        self.inbox = self.vault / "_harness" / "knowledge-inbox"
        self.redactor = redactor or redact_log_message
        if create_inbox:
            self._ensure_directory(self.vault / "_harness")
            self._ensure_directory(self.inbox)
        else:
            harness = self.vault / "_harness"
            if (
                self.inbox.is_symlink()
                or harness.is_symlink()
                or (harness.exists() and not harness.is_dir())
                or (self.inbox.exists() and not self.inbox.is_dir())
            ):
                raise ValueError("knowledge inbox path is unsafe")

    @staticmethod
    def _ensure_directory(path):
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError("knowledge inbox path is unsafe")
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("knowledge inbox path is unsafe")

    def _source(self, relative, expected):
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or (
                expected is not None
                and (
                    not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None
                )
            )
        ):
            raise ValueError("knowledge source reference is invalid")
        candidate = self.source_root / relative
        current = self.source_root
        try:
            for part in Path(relative).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("knowledge source reference is unsafe")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.source_root) or not stat.S_ISREG(
                resolved.stat().st_mode
            ):
                raise ValueError("knowledge source reference is unsafe")
            with resolved.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        except OSError as error:
            raise ValueError("knowledge source is unavailable") from error
        if expected is not None and digest != expected:
            raise ValueError("knowledge source has changed")
        return relative, digest

    @staticmethod
    def _canonical(value):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def propose(
        self,
        *,
        task_id,
        title,
        body,
        category,
        source_path,
        source_sha256=None,
    ):
        if (
            not isinstance(task_id, (str, int))
            or isinstance(task_id, bool)
            or not str(task_id).strip()
        ):
            raise ValueError("task id is invalid")
        if (
            not isinstance(title, str)
            or not title.strip()
            or len(title) > 200
            or "\n" in title
            or "\r" in title
        ):
            raise ValueError("knowledge title is invalid")
        if (
            not isinstance(body, str)
            or not body.strip()
            or len(body.encode("utf-8")) > 20_000
        ):
            raise ValueError("knowledge body is invalid")
        if not isinstance(category, str) or category not in _CATEGORIES:
            raise ValueError("knowledge category is invalid")
        source_path, source_sha256 = self._source(source_path, source_sha256)
        title = self.redactor(title.strip())
        body = self.redactor(body.strip())
        if not isinstance(title, str) or not isinstance(body, str):
            raise TypeError("redacted knowledge content must be text")
        payload = {
            "task_id": str(task_id),
            "title": title.strip(),
            "body": body.strip(),
            "category": category,
            "source_path": source_path,
            "source_sha256": source_sha256,
        }
        identifier = hashlib.sha256(self._canonical(payload)).hexdigest()[:24]
        document = {"id": identifier, **payload}
        serialized = self._canonical(document) + b"\n"
        path = self.inbox / f"{identifier}.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != serialized:
                raise ValueError("knowledge proposal id collision")
        else:
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(serialized)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                path.unlink(missing_ok=True)
                raise
        return {
            "id": identifier,
            "digest": hashlib.sha256(serialized).hexdigest(),
            "status": "pending",
            "target": self._target(document),
        }

    @staticmethod
    def _target(proposal):
        slug = re.sub(r"[^a-z0-9]+", "-", proposal["title"].casefold()).strip("-")
        slug = (slug or "knowledge")[:64].strip("-")
        return f"knowledge/{slug}-{proposal['id'][:8]}.md"

    def _load(self, identifier, expected_digest):
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError("knowledge proposal id is invalid")
        path = self.inbox / f"{identifier}.json"
        if path.is_symlink():
            raise ValueError("knowledge proposal is unsafe")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError("knowledge proposal is unavailable") from error
        digest = hashlib.sha256(raw).hexdigest()
        if not isinstance(expected_digest, str) or digest != expected_digest:
            raise ValueError("knowledge proposal changed; review it again")
        try:
            proposal = json.loads(raw)
        except (UnicodeError, ValueError) as error:
            raise ValueError("knowledge proposal is invalid") from error
        if not isinstance(proposal, dict) or set(proposal) != {
            "id",
            "task_id",
            "title",
            "body",
            "category",
            "source_path",
            "source_sha256",
        }:
            raise ValueError("knowledge proposal is invalid")
        payload = {key: value for key, value in proposal.items() if key != "id"}
        if (
            proposal.get("id") != identifier
            or hashlib.sha256(self._canonical(payload)).hexdigest()[:24] != identifier
            or not all(
                isinstance(payload.get(key), str)
                for key in (
                    "task_id",
                    "title",
                    "body",
                    "category",
                    "source_path",
                    "source_sha256",
                )
            )
            or not payload["title"].strip()
            or not payload["body"].strip()
            or payload["category"] not in _CATEGORIES
        ):
            raise ValueError("knowledge proposal is invalid")
        self._source(proposal.get("source_path"), proposal.get("source_sha256"))
        return proposal, digest

    def pending(self):
        if self.inbox.is_symlink():
            raise ValueError("knowledge inbox path is unsafe")
        items = []
        for path in sorted(self.inbox.glob("*.json")):
            if path.is_symlink() or not _IDENTIFIER.fullmatch(path.stem):
                continue
            try:
                raw = path.read_bytes()
                proposal = json.loads(raw)
                payload = (
                    {key: value for key, value in proposal.items() if key != "id"}
                    if isinstance(proposal, dict)
                    else {}
                )
                if (
                    not isinstance(proposal, dict)
                    or proposal.get("id") != path.stem
                    or hashlib.sha256(self._canonical(payload)).hexdigest()[:24]
                    != path.stem
                ):
                    continue
                state = "pending"
                source_state = "current"
                try:
                    self._source(
                        proposal.get("source_path"), proposal.get("source_sha256")
                    )
                except ValueError:
                    source_state = "stale_or_unavailable"
                for decision in ("approved", "rejected"):
                    marker = self.inbox / f"{path.stem}.{decision}.json"
                    if marker.is_file() and not marker.is_symlink():
                        state = decision
                items.append(
                    {
                        "id": path.stem,
                        "title": proposal.get("title", ""),
                        "body": proposal.get("body", ""),
                        "category": proposal.get("category", ""),
                        "task_id": proposal.get("task_id", ""),
                        "source_path": proposal.get("source_path", ""),
                        "source_sha256": proposal.get("source_sha256", ""),
                        "source_state": source_state,
                        "status": state,
                        "digest": hashlib.sha256(raw).hexdigest(),
                        "target": self._target(proposal),
                    }
                )
            except (OSError, UnicodeError, ValueError):
                continue
        return items

    def _write_decision(self, identifier, decision, digest, target):
        record = (
            self._canonical(
                {
                    "id": identifier,
                    "decision": decision,
                    "proposal_sha256": digest,
                    "target": target,
                    "decided_at": datetime.now(UTC).isoformat(),
                }
            )
            + b"\n"
        )
        path = self.inbox / f"{identifier}.{decision}.json"
        other = "rejected" if decision == "approved" else "approved"
        if (self.inbox / f"{identifier}.{other}.json").exists():
            raise ValueError("knowledge proposal already has a different decision")
        if path.is_symlink():
            raise ValueError("knowledge decision record is unsafe")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("proposal_sha256") != digest:
                raise ValueError("knowledge proposal already has a different decision")
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(record)

    def _register_index_link(self, target, title):
        index = self.vault / "knowledge" / "Projektwissen.md"
        if index.is_symlink() or not index.is_file():
            raise ValueError("curated project knowledge index is unavailable")
        if not index.resolve(strict=True).is_relative_to(self.vault):
            raise ValueError("curated project knowledge index is unsafe")
        raw = index.read_text(encoding="utf-8")
        link = f"[[{target.removesuffix('.md')}]]"
        if link in raw:
            return
        refreshed, count = _REVIEWED.subn(
            f"last_reviewed: {datetime.now(UTC).date().isoformat()}", raw, count=1
        )
        if count != 1:
            raise ValueError("curated project knowledge index has no review date")
        updated = refreshed.rstrip() + f"\n\n- {link} — {title}\n"
        mode = stat.S_IMODE(index.stat().st_mode)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".Projektwissen-", suffix=".tmp", dir=index.parent
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            if index.is_symlink() or index.read_text(encoding="utf-8") != raw:
                raise ValueError(
                    "curated project knowledge index changed during approval"
                )
            os.replace(temporary, index)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def reject(self, identifier, expected_digest):
        proposal, digest = self._load(identifier, expected_digest)
        target = self._target(proposal)
        self._write_decision(identifier, "rejected", digest, target)
        return {"id": identifier, "status": "rejected", "target": target}

    def approve(self, identifier, expected_digest):
        proposal, digest = self._load(identifier, expected_digest)
        target = self._target(proposal)
        path = self.vault / target
        if (self.inbox / f"{identifier}.rejected.json").exists():
            raise ValueError("knowledge proposal already has a different decision")
        approved_marker = self.inbox / f"{identifier}.approved.json"
        if approved_marker.is_file() and not approved_marker.is_symlink():
            existing = json.loads(approved_marker.read_text(encoding="utf-8"))
            if (
                existing.get("proposal_sha256") != digest
                or not path.is_file()
                or path.is_symlink()
            ):
                raise ValueError("approved knowledge record is inconsistent")
            return {"id": identifier, "status": "approved", "target": target}
        directory = path.parent
        self._ensure_directory(directory)
        if not directory.resolve(strict=True).is_relative_to(self.vault):
            raise ValueError("knowledge target is unsafe")
        frontmatter = (
            "---\n"
            f"type: {proposal['category']}\n"
            f"last_reviewed: {datetime.now(UTC).date().isoformat()}\n"
            f"task_id: {json.dumps(proposal['task_id'], ensure_ascii=False)}\n"
            "sources:\n"
            f"  - path: {json.dumps(proposal['source_path'], ensure_ascii=False)}\n"
            f"    sha256: {proposal['source_sha256']}\n"
            "---\n\n"
        )
        content = f"{frontmatter}# {proposal['title']}\n\n{proposal['body']}\n"
        if path.is_symlink():
            raise ValueError("knowledge target is unsafe")
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    "refusing to overwrite an existing knowledge note"
                )
        else:
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                raise FileExistsError(
                    "refusing to overwrite an existing knowledge note"
                ) from None
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
        self._register_index_link(target, proposal["title"])
        self._write_decision(identifier, "approved", digest, target)
        return {"id": identifier, "status": "approved", "target": target}
