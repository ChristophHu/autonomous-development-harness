"""Deterministic, SQLite-authoritative decision notes for an Obsidian vault."""

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml


class DecisionProjection:
    VERSION = 1
    MARKER = "projection: harness-decisions-v1"

    def __init__(self, vault):
        self.vault = Path(vault).resolve()
        self.root = self.vault / "_harness"
        self.notes = self.root / "decisions"
        self.manifest_path = self.root / "manifest.json"

    def _inside_vault(self, path):
        candidate = Path(path).absolute()
        try:
            relative = candidate.relative_to(self.vault)
        except ValueError as exc:
            raise PermissionError("projection path escapes Obsidian vault") from exc
        current = self.vault
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("projection path cannot traverse symlinks")
        try:
            candidate.resolve().relative_to(self.vault)
        except ValueError as exc:
            raise PermissionError("projection path escapes Obsidian vault") from exc
        return candidate

    def _read_manifest(self):
        for path in (self.root, self.manifest_path):
            if path.is_symlink():
                raise PermissionError("projection manifest path cannot be a symlink")
        self._inside_vault(self.manifest_path)
        if not self.manifest_path.exists():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid projection manifest") from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != self.VERSION
            or not isinstance(data.get("files"), list)
        ):
            raise ValueError("invalid projection manifest")
        files = data["files"]
        if any(
            not isinstance(name, str)
            or not name.startswith("decisions/")
            or not name.endswith(".md")
            or Path(name).name != name.removeprefix("decisions/")
            or not name.removeprefix("decisions/").removesuffix(".md").isdecimal()
            for name in files
        ) or len(files) != len(set(files)):
            raise ValueError("invalid projection manifest")
        for name in files:
            path = self.root / name
            if path.is_symlink():
                raise PermissionError("managed projection note cannot be a symlink")
            self._inside_vault(path)
        return files

    @classmethod
    def _render(cls, row):
        metadata = {
            "id": row["id"],
            "task_id": row["task_id"],
            "question_id": row.get("question_id"),
            "category": row["category"],
            "source": row["source"],
            "field_names": row.get("field_names", []),
            "evidence": row.get("evidence", []),
            "alternatives": row.get("alternatives", []),
            "outcome": row.get("outcome"),
            "tags": row.get("tags", []),
            "supersedes_id": row.get("supersedes_id"),
            "created_at": row["created_at"],
        }
        alternatives = row.get("alternatives", [])
        alternatives_text = ""
        if alternatives:
            alternatives_text = (
                "\n## Alternatives\n\n"
                + "\n".join(
                    "- **"
                    + item["option"]
                    + (" (selected)" if item.get("selected") else "")
                    + (
                        ": " + "; ".join(item.get("consequences", []))
                        if item.get("consequences")
                        else ""
                    )
                    for item in alternatives
                )
                + "\n"
            )
        outcome = row.get("outcome")
        outcome_text = "\n## Outcome\n\n" + outcome + "\n" if outcome else ""
        return (
            "---\n"
            + cls.MARKER
            + "\n"
            + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True).rstrip()
            + "\n---\n\n## Decision\n\n"
            + str(row["decision"])
            + "\n\n## Rationale\n\n"
            + str(row["rationale"])
            + "\n"
            + alternatives_text
            + outcome_text
        )

    @classmethod
    def _is_owned(cls, content):
        return content.startswith(f"---\n{cls.MARKER}\n")

    @staticmethod
    def _atomic_write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = stream.name
        try:
            os.replace(temporary_path, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)

    def sync(self, rows):
        old_files = self._read_manifest()
        self._inside_vault(self.notes)
        desired = {}
        for row in rows:
            if (
                not isinstance(row.get("id"), int)
                or isinstance(row["id"], bool)
                or row["id"] < 1
            ):
                raise ValueError("decision projection requires positive integer IDs")
            name = f"decisions/{row['id']}.md"
            if name in desired:
                raise ValueError("duplicate decision ID in projection input")
            desired[name] = self._render(row)

        counts = {"created": 0, "updated": 0, "removed": 0, "unchanged": 0}
        existing_content = {}
        for name in desired:
            path = self._inside_vault(self.root / name)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if not self._is_owned(content):
                    raise FileExistsError(f"refusing to replace unowned note: {path}")
                existing_content[name] = content
        for name in old_files:
            if name in desired:
                continue
            path = self._inside_vault(self.root / name)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if not self._is_owned(content):
                    raise FileExistsError(f"refusing to remove unowned note: {path}")
                existing_content[name] = content

        for name, content in desired.items():
            path = self._inside_vault(self.root / name)
            if name in existing_content:
                existing = existing_content[name]
                if existing == content:
                    counts["unchanged"] += 1
                    continue
                counts["updated"] += 1
            else:
                counts["created"] += 1
            self._atomic_write(path, content)

        for name in old_files:
            if name in desired:
                continue
            path = self._inside_vault(self.root / name)
            if path.exists():
                path.unlink()
                counts["removed"] += 1

        self._atomic_write(
            self.manifest_path,
            json.dumps({"version": self.VERSION, "files": sorted(desired)}, indent=2)
            + "\n",
        )
        return counts

    def status(self, obsidian_enabled=True, mcp_enabled=False):
        """Describe the configured vault without changing its contents."""
        markdown = sorted(
            path.relative_to(self.vault).as_posix()
            for path in self.vault.rglob("*.md")
            if not any(
                part.startswith(".") for part in path.relative_to(self.vault).parts
            )
            and path.is_file()
            and not path.is_symlink()
        )
        manifest = self._read_manifest()
        return {
            "path": str(self.vault),
            "exists": self.vault.is_dir(),
            "obsidian_enabled": bool(obsidian_enabled),
            "mcp_enabled": bool(mcp_enabled),
            "markdown_notes": len(markdown),
            "managed_decisions": len(manifest),
            "nested_vaults": sorted(
                path.relative_to(self.vault).as_posix()
                for path in self.vault.rglob(".obsidian")
                if path.is_dir() and path.parent != self.vault
            ),
        }
