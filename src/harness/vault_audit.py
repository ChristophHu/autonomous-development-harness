"""Read-only structural and freshness audit for the curated Obsidian vault."""

from __future__ import annotations

import hashlib
import re
import stat
from datetime import UTC, date, datetime
from pathlib import Path
from posixpath import normpath

import yaml

from .vault_review import classify_review

DEFAULT_REQUIRED_NOTES = (
    "Willkommen.md",
    "Vault-Übersicht.md",
    "rules/Harness-Prinzipien.md",
    "rules/Coding-Standards.md",
    "rules/Konventionen.md",
    "architecture/Systemarchitektur.md",
    "architecture/Vault und Memory.md",
    "architecture/Modelle und Routing.md",
    "architecture/Git und Isolation.md",
    "operations/Lokaler Betrieb.md",
    "decisions/Entscheidungsregister.md",
    "decisions/Architekturentscheidungen.md",
    "decisions/Technische Entscheidungen.md",
    "agents/Agentenprofile.md",
    "tasks/Task-Register.md",
    "knowledge/Projektwissen.md",
    "knowledge/Lessons Learned.md",
    "knowledge/Bekannte Probleme.md",
    "docs/Dokumentation.md",
)
CURATED_NOTE_TYPES = {
    "Willkommen.md": "vault-home",
    "Vault-Übersicht.md": "vault-index",
    "rules/Harness-Prinzipien.md": "project-rules",
    "rules/Coding-Standards.md": "coding-standards",
    "rules/Konventionen.md": "conventions",
    "architecture/Systemarchitektur.md": "architecture",
    "architecture/Vault und Memory.md": "architecture",
    "architecture/Modelle und Routing.md": "model-routing",
    "architecture/Git und Isolation.md": "git-isolation-architecture",
    "operations/Lokaler Betrieb.md": "operations-guide",
    "decisions/Entscheidungsregister.md": "decision-index",
    "decisions/Architekturentscheidungen.md": "architecture-decisions",
    "decisions/Technische Entscheidungen.md": "technical-decisions",
    "agents/Agentenprofile.md": "agent-index",
    "tasks/Task-Register.md": "task-index",
    "knowledge/Projektwissen.md": "project-knowledge",
    "knowledge/Lessons Learned.md": "lessons-learned",
    "knowledge/Bekannte Probleme.md": "known-problems",
    "docs/Dokumentation.md": "documentation-index",
}
WIKILINK = re.compile(r"!?\[\[([^\]|#]+)")
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", re.DOTALL)


def _excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if path.is_symlink() or any(
        part.startswith(".") or part == "_harness" for part in relative.parts
    ):
        return True
    if (
        len(relative.parts) == 3
        and relative.parts[0] == "tasks"
        and relative.parts[1].isdigit()
        and relative.name == "plan.md"
    ):
        return True
    try:
        return not path.resolve(strict=True).is_relative_to(root)
    except OSError:
        return True


def _review_date(text: str, reference_date: date, max_age_days: int):
    frontmatter = FRONTMATTER.match(text)
    if frontmatter is None:
        return "missing", None
    try:
        metadata = yaml.safe_load(frontmatter.group(1))
    except yaml.YAMLError:
        return "invalid", None
    if not isinstance(metadata, dict):
        return "invalid", None
    state, reviewed = classify_review(metadata, reference_date, max_age_days)
    if state == "unreviewed":
        return "missing", None
    return state, reviewed


def _finding(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _curated_type_state(text, expected):
    frontmatter = FRONTMATTER.match(text)
    if frontmatter is None:
        return "missing"
    try:
        metadata = yaml.safe_load(frontmatter.group(1))
    except yaml.YAMLError:
        return "invalid"
    if not isinstance(metadata, dict):
        return "invalid"
    actual = metadata.get("type")
    if actual is None or actual == "":
        return "missing"
    return "valid" if actual == expected else "invalid"


def _source_findings(text, note_path, source_root):
    frontmatter = FRONTMATTER.match(text)
    if frontmatter is None:
        return [
            _finding(
                "content_sources_missing",
                note_path,
                "Curated note has no source references",
            )
        ]
    try:
        metadata = yaml.safe_load(frontmatter.group(1))
    except yaml.YAMLError:
        return [
            _finding(
                "content_sources_invalid",
                note_path,
                "Curated note source references are invalid",
            )
        ]
    references = metadata.get("sources") if isinstance(metadata, dict) else None
    if not isinstance(references, list) or not references:
        return [
            _finding(
                "content_sources_missing",
                note_path,
                "Curated note has no source references",
            )
        ]
    try:
        root = Path(source_root).resolve(strict=True)
    except (OSError, TypeError):
        return [
            _finding(
                "content_source_unavailable", note_path, "Source root is unavailable"
            )
        ]
    findings = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            findings.append(
                _finding(
                    "content_sources_invalid",
                    note_path,
                    "Source reference must contain path and sha256",
                )
            )
            continue
        relative, digest = reference["path"], reference["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            findings.append(
                _finding(
                    "content_sources_invalid",
                    note_path,
                    "Source reference path or SHA-256 is invalid",
                )
            )
            continue
        candidate = root / relative
        try:
            current = root
            unsafe = False
            for component in Path(relative).parts:
                current = current / component
                if current.is_symlink():
                    unsafe = True
                    break
            resolved = candidate.resolve(strict=True)
            if (
                unsafe
                or not resolved.is_relative_to(root)
                or not stat.S_ISREG(resolved.stat().st_mode)
            ):
                findings.append(
                    _finding(
                        "content_source_invalid",
                        note_path,
                        "Source path is unsafe or not a regular file",
                    )
                )
                continue
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError:
            findings.append(
                _finding(
                    "content_source_unavailable",
                    note_path,
                    "Referenced source is unavailable",
                )
            )
            continue
        if actual != digest:
            findings.append(
                _finding(
                    "content_source_stale",
                    note_path,
                    "Referenced source SHA-256 has changed",
                )
            )
    return findings


def _decision_projection_findings(root, decision_rows):
    from .memory_projection import DecisionProjection

    projection = DecisionProjection(root)
    findings = []
    try:
        registered = set(projection._read_manifest())
        expected = {
            f"decisions/{row['id']}.md": projection._render(row)
            for row in decision_rows
        }
        for name in sorted(set(expected) - registered):
            findings.append(
                _finding(
                    "decision_projection_missing",
                    f"_harness/{name}",
                    "SQLite decision is not registered in the projection manifest",
                )
            )
        for name in sorted(registered):
            path = projection._inside_vault(projection.root / name)
            if name not in expected:
                if not path.exists():
                    findings.append(
                        _finding(
                            "decision_projection_stale",
                            f"_harness/{name}",
                            "Manifest entry has no authoritative SQLite decision",
                        )
                    )
                    continue
                content = path.read_text(encoding="utf-8")
                if not DecisionProjection._is_owned(content):
                    findings.append(
                        _finding(
                            "decision_projection_unowned",
                            f"_harness/{name}",
                            "Stale manifest entry points to a note without the Harness marker",
                        )
                    )
                    continue
                findings.append(
                    _finding(
                        "decision_projection_stale",
                        f"_harness/{name}",
                        "Projection has no authoritative SQLite decision",
                    )
                )
                continue
            if not path.is_file():
                findings.append(
                    _finding(
                        "decision_projection_missing",
                        f"_harness/{name}",
                        "SQLite decision projection file is missing",
                    )
                )
                continue
            content = path.read_text(encoding="utf-8")
            if not DecisionProjection._is_owned(content) or content != expected[name]:
                findings.append(
                    _finding(
                        "decision_projection_conflict",
                        f"_harness/{name}",
                        "Projection differs from authoritative SQLite decision or is not Harness-owned",
                    )
                )
    except (OSError, UnicodeError, ValueError, PermissionError):
        findings.append(
            _finding(
                "decision_projection_unreadable",
                "_harness",
                "Decision projection cannot be safely audited",
            )
        )
    return findings


def audit_vault(
    vault: str | Path,
    *,
    required_notes=DEFAULT_REQUIRED_NOTES,
    max_review_age_days: int = 180,
    today: date | None = None,
    decision_rows=None,
    content_governance: bool = False,
    source_root: str | Path | None = None,
) -> dict:
    """Audit curated Markdown without modifying files or following symlinks."""
    if (
        isinstance(max_review_age_days, bool)
        or not isinstance(max_review_age_days, int)
        or max_review_age_days < 1
    ):
        raise ValueError("max_review_age_days must be positive")
    root = Path(vault).resolve()
    exists = root.is_dir()
    candidates = sorted(root.rglob("*.md")) if exists else []
    notes = [
        path for path in candidates if path.is_file() and not _excluded(path, root)
    ]
    excluded = len(candidates) - len(notes)
    findings: list[dict[str, str]] = []
    note_hashes: dict[str, str] = {}
    if not exists:
        findings.append(
            _finding("vault_missing", ".", "Vault directory does not exist")
        )

    available = {path.relative_to(root).as_posix(): path for path in notes}
    for required in required_notes:
        normalized = Path(required).as_posix()
        if normalized not in available:
            findings.append(
                _finding(
                    "required_note_missing", normalized, "Required note is missing"
                )
            )

    by_stem: dict[str, list[str]] = {}
    for relative in available:
        by_stem.setdefault(Path(relative).with_suffix("").as_posix(), []).append(
            relative
        )
        by_stem.setdefault(Path(relative).stem, []).append(relative)

    reference_date = today or datetime.now(UTC).date()
    inbound_links: set[str] = set()
    for path in notes:
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(
                _finding("note_unreadable", relative, "Note cannot be read as UTF-8")
            )
            continue
        note_hashes[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_governance and relative in CURATED_NOTE_TYPES:
            type_state = _curated_type_state(text, CURATED_NOTE_TYPES[relative])
            if type_state != "valid":
                code = (
                    "content_type_missing"
                    if type_state == "missing"
                    else "content_type_invalid"
                )
                findings.append(
                    _finding(code, relative, "Curated note has an invalid type field")
                )
            findings.extend(
                _source_findings(text, relative, source_root or root.parent)
            )
        review_state, _reviewed = _review_date(
            text, reference_date, max_review_age_days
        )
        if review_state == "missing":
            findings.append(
                _finding("review_missing", relative, "last_reviewed is missing")
            )
        elif review_state == "invalid":
            findings.append(
                _finding("review_invalid", relative, "last_reviewed must be YYYY-MM-DD")
            )
        elif review_state == "future":
            findings.append(
                _finding("review_future", relative, "last_reviewed is in the future")
            )
        elif review_state == "stale":
            findings.append(
                _finding(
                    "review_stale", relative, "last_reviewed exceeds the age limit"
                )
            )

        for raw_target in WIKILINK.findall(text):
            target = raw_target.strip().removesuffix(".md")
            matches = []
            if "/" in target:
                exact = normpath((Path(relative).parent / target).as_posix())
                rooted = target
                matches = [
                    candidate
                    for item in (exact, rooted)
                    for candidate in (item, f"{item}.md")
                    if candidate in available
                ]
            else:
                matches = by_stem.get(target, [])
            matches = sorted(set(matches))
            if not matches:
                findings.append(
                    _finding(
                        "broken_wikilink",
                        relative,
                        f"Wiki link target is missing: {target}",
                    )
                )
            elif len(matches) > 1:
                findings.append(
                    _finding(
                        "ambiguous_wikilink",
                        relative,
                        f"Wiki link target is ambiguous: {target}",
                    )
                )
            else:
                inbound_links.update(matches)

    if content_governance:
        required_paths = set(CURATED_NOTE_TYPES)
        root_entries = {"Index.md", "Welcome.md", "Willkommen.md", "Vault-Übersicht.md"}
        for relative in sorted(
            set(available) - required_paths - root_entries - inbound_links
        ):
            findings.append(
                _finding(
                    "orphan_note", relative, "Note has no incoming local wiki link"
                )
            )

    if decision_rows is not None:
        findings.extend(_decision_projection_findings(root, decision_rows))

    findings.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    return {
        "vault": str(root),
        "exists": exists,
        "markdown_notes": len(candidates),
        "audited_notes": len(notes),
        "note_hashes": note_hashes,
        "excluded_notes": excluded,
        "max_review_age_days": max_review_age_days,
        "findings": findings,
        "healthy": not findings,
    }
