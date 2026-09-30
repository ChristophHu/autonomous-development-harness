"""Read-only structural and freshness audit for the curated Obsidian vault."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from posixpath import normpath

DEFAULT_REQUIRED_NOTES = (
    "Willkommen.md",
    "Vault-Übersicht.md",
    "rules/Harness-Prinzipien.md",
    "architecture/Systemarchitektur.md",
    "architecture/Vault und Memory.md",
    "decisions/Entscheidungsregister.md",
    "agents/Agentenprofile.md",
    "tasks/Task-Register.md",
)
WIKILINK = re.compile(r"!?\[\[([^\]|#]+)")
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", re.DOTALL)
REVIEWED = re.compile(r"^last_reviewed\s*:\s*(.*?)\s*$", re.MULTILINE)


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


def _review_date(text: str):
    frontmatter = FRONTMATTER.match(text)
    if frontmatter is None:
        return "missing", None
    match = REVIEWED.search(frontmatter.group(1))
    if match is None:
        return "missing", None
    raw = match.group(1).strip().strip("\"'")
    try:
        return "valid", date.fromisoformat(raw)
    except ValueError:
        return "invalid", None


def _finding(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def audit_vault(
    vault: str | Path,
    *,
    required_notes=DEFAULT_REQUIRED_NOTES,
    max_review_age_days: int = 180,
    today: date | None = None,
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
    for path in notes:
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(
                _finding("note_unreadable", relative, "Note cannot be read as UTF-8")
            )
            continue
        review_state, reviewed = _review_date(text)
        if review_state == "missing":
            findings.append(
                _finding("review_missing", relative, "last_reviewed is missing")
            )
        elif review_state == "invalid":
            findings.append(
                _finding("review_invalid", relative, "last_reviewed must be YYYY-MM-DD")
            )
        elif reviewed > reference_date:
            findings.append(
                _finding("review_future", relative, "last_reviewed is in the future")
            )
        elif (reference_date - reviewed).days > max_review_age_days:
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

    findings.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    return {
        "vault": str(root),
        "exists": exists,
        "markdown_notes": len(candidates),
        "audited_notes": len(notes),
        "excluded_notes": excluded,
        "max_review_age_days": max_review_age_days,
        "findings": findings,
        "healthy": not findings,
    }
