import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from harness.memory_projection import DecisionProjection
from harness.vault_audit import audit_vault


def note(path, body, *, reviewed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    current_day = datetime.now(UTC).date()
    frontmatter = f"---\nlast_reviewed: {reviewed or current_day.isoformat()}\n---\n"
    path.write_text(frontmatter + body, encoding="utf-8")


def test_audit_accepts_required_notes_and_resolves_wikilinks(tmp_path):
    vault = tmp_path / "vault"
    required = (
        "Willkommen.md",
        "Vault-Übersicht.md",
        "rules/Harness-Prinzipien.md",
        "architecture/Systemarchitektur.md",
        "architecture/Vault und Memory.md",
        "architecture/Modelle und Routing.md",
        "architecture/Git und Isolation.md",
        "operations/Lokaler Betrieb.md",
        "decisions/Entscheidungsregister.md",
        "agents/Agentenprofile.md",
        "tasks/Task-Register.md",
    )
    for path in required:
        note(vault / path, "# Root knowledge\n\nSiehe [[Harness-Prinzipien]].")
    report = audit_vault(vault)
    assert report == {
        "vault": str(vault.resolve()),
        "exists": True,
        "markdown_notes": 11,
        "audited_notes": 11,
        "note_hashes": {
            path: hashlib.sha256(
                (vault / path).read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for path in required
        },
        "excluded_notes": 0,
        "max_review_age_days": 180,
        "findings": [],
        "healthy": True,
    }


def test_audit_reports_missing_required_notes(tmp_path):
    report = audit_vault(tmp_path / "vault")
    assert report["exists"] is False
    assert report["markdown_notes"] == 0
    assert report["healthy"] is False
    assert {finding["code"] for finding in report["findings"]} == {
        "vault_missing",
        "required_note_missing",
    }


def test_audit_reports_broken_links_missing_and_stale_review_dates(tmp_path):
    vault = tmp_path / "vault"
    old = (datetime.now(UTC).date() - timedelta(days=181)).isoformat()
    note(vault / "Willkommen.md", "# Start\n\n[[DoesNotExist]]", reviewed=old)
    (vault / "No-Review.md").write_text("# No review field", encoding="utf-8")
    report = audit_vault(vault, required_notes=())
    codes = [finding["code"] for finding in report["findings"]]
    assert "broken_wikilink" in codes
    assert "review_stale" in codes
    assert "review_missing" in codes
    assert "review_invalid" not in codes
    assert report["healthy"] is False


def test_audit_reports_invalid_review_date_and_ambiguous_links(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "one/Topic.md", "# One")
    note(vault / "two/Topic.md", "# Two")
    note(vault / "Index.md", "[[Topic]]", reviewed="yesterday")
    report = audit_vault(vault, required_notes=())
    codes = {finding["code"] for finding in report["findings"]}
    assert "ambiguous_wikilink" in codes
    assert "review_invalid" in codes


def test_audit_resolves_rooted_and_note_relative_path_links(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Index.md", "[[Target/Note]]")
    note(vault / "section/Index.md", "[[../Target/Note]]")
    note(vault / "Target/Note.md", "# Target")
    report = audit_vault(vault, required_notes=())
    assert report["findings"] == []


def test_audit_reports_future_review_dates(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Future.md", "# Future", reviewed="2026-01-02")
    report = audit_vault(vault, required_notes=(), today=date(2026, 1, 1))
    assert [item["code"] for item in report["findings"]] == ["review_future"]


def test_audit_reports_frontmatter_without_review_date(tmp_path):
    vault = tmp_path / "vault"
    path = vault / "No-Review.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ntype: note\n---\n# Note", encoding="utf-8")
    report = audit_vault(vault, required_notes=())
    assert report["findings"][0]["code"] == "review_missing"


def test_audit_compares_decision_projection_to_sqlite_rows_read_only(tmp_path):
    vault = tmp_path / "vault"
    projection = DecisionProjection(vault)
    row = {
        "id": 4,
        "task_id": 2,
        "question_id": None,
        "category": "architecture",
        "source": "human",
        "field_names": [],
        "evidence": [],
        "alternatives": [],
        "outcome": None,
        "tags": [],
        "supersedes_id": None,
        "created_at": "2026-09-30T00:00:00+00:00",
        "decision": "Use SQLite",
        "rationale": "Local authority",
    }
    projection.sync([row])
    healthy = audit_vault(vault, required_notes=(), decision_rows=[row])
    assert healthy["healthy"] is True

    note_path = vault / "_harness/decisions/4.md"
    note_path.write_text("tampered", encoding="utf-8")
    drifted = audit_vault(vault, required_notes=(), decision_rows=[row])
    assert any(
        item["code"] == "decision_projection_conflict" for item in drifted["findings"]
    )
    assert note_path.read_text(encoding="utf-8") == "tampered"


def test_audit_reports_missing_and_stale_decision_projection(tmp_path):
    vault = tmp_path / "vault"
    projection = DecisionProjection(vault)
    stale = {
        "id": 9,
        "task_id": None,
        "question_id": None,
        "category": "architecture",
        "source": "human",
        "field_names": [],
        "evidence": [],
        "alternatives": [],
        "outcome": None,
        "tags": [],
        "supersedes_id": None,
        "created_at": "2026-09-30T00:00:00+00:00",
        "decision": "old",
        "rationale": "old",
    }
    projection.sync([stale])
    report = audit_vault(vault, required_notes=(), decision_rows=[])
    assert any(
        item["code"] == "decision_projection_stale" for item in report["findings"]
    )
    report = audit_vault(
        tmp_path / "empty-vault", required_notes=(), decision_rows=[stale]
    )
    assert any(
        item["code"] == "decision_projection_missing" for item in report["findings"]
    )


def test_audit_distinguishes_unowned_stale_projection_without_modifying_it(tmp_path):
    vault = tmp_path / "vault"
    projection_root = vault / "_harness"
    note_path = projection_root / "decisions" / "1.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("User-authored content", encoding="utf-8")
    manifest = projection_root / "manifest.json"
    manifest.write_text('{"version":1,"files":["decisions/1.md"]}', encoding="utf-8")

    report = audit_vault(vault, required_notes=(), decision_rows=[])

    assert report["findings"] == [
        {
            "code": "decision_projection_unowned",
            "path": "_harness/decisions/1.md",
            "message": "Stale manifest entry points to a note without the Harness marker",
        }
    ]
    assert note_path.read_text(encoding="utf-8") == "User-authored content"
    assert manifest.read_text(encoding="utf-8") == (
        '{"version":1,"files":["decisions/1.md"]}'
    )


def test_audit_reports_registered_but_missing_authoritative_projection(tmp_path):
    from harness.memory_projection import DecisionProjection

    vault = tmp_path / "vault"
    projection = DecisionProjection(vault)
    projection.root.mkdir(parents=True)
    projection.manifest_path.write_text(
        '{"version":1,"files":["decisions/4.md"]}', encoding="utf-8"
    )
    row = {
        "id": 4,
        "task_id": None,
        "question_id": None,
        "category": "architecture",
        "source": "human",
        "field_names": [],
        "evidence": [],
        "alternatives": [],
        "outcome": None,
        "tags": [],
        "supersedes_id": None,
        "created_at": "2026-10-02T00:00:00+00:00",
        "decision": "Use SQLite",
        "rationale": "Local authority",
    }

    report = audit_vault(vault, required_notes=(), decision_rows=[row])

    assert report["findings"] == [
        {
            "code": "decision_projection_missing",
            "path": "_harness/decisions/4.md",
            "message": "SQLite decision projection file is missing",
        }
    ]


def test_audit_reports_manifest_entry_without_file_or_authoritative_decision(
    tmp_path,
):
    from harness.memory_projection import DecisionProjection

    vault = tmp_path / "vault"
    projection = DecisionProjection(vault)
    projection.root.mkdir(parents=True)
    projection.manifest_path.write_text(
        '{"version":1,"files":["decisions/9.md"]}', encoding="utf-8"
    )

    report = audit_vault(vault, required_notes=(), decision_rows=[])

    assert report["findings"] == [
        {
            "code": "decision_projection_stale",
            "path": "_harness/decisions/9.md",
            "message": "Manifest entry has no authoritative SQLite decision",
        }
    ]


def test_audit_fails_closed_when_projection_manifest_is_unreadable(
    tmp_path, monkeypatch
):
    from harness.memory_projection import DecisionProjection

    def unreadable(_self):
        raise ValueError("private manifest detail")

    monkeypatch.setattr(DecisionProjection, "_read_manifest", unreadable)
    report = audit_vault(tmp_path / "vault", required_notes=(), decision_rows=[])
    assert report["findings"] == [
        {
            "code": "vault_missing",
            "path": ".",
            "message": "Vault directory does not exist",
        },
        {
            "code": "decision_projection_unreadable",
            "path": "_harness",
            "message": "Decision projection cannot be safely audited",
        },
    ]


def test_audit_skips_paths_whose_containment_cannot_be_verified(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    note(vault / "Unresolvable.md", "# Note")
    original_resolve = type(vault).resolve

    def resolve(path, *args, **kwargs):
        if path.name == "Unresolvable.md":
            raise OSError("unavailable")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(vault), "resolve", resolve)
    report = audit_vault(vault, required_notes=())
    assert report["audited_notes"] == 0
    assert report["excluded_notes"] == 1
    assert report["findings"] == []


def test_audit_reports_unreadable_notes_and_continues(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    note(vault / "Unreadable.md", "# Note")
    note(vault / "Readable.md", "# Note")
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.name == "Unreadable.md":
            raise UnicodeError("invalid bytes")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    report = audit_vault(vault, required_notes=())
    assert [item["code"] for item in report["findings"]] == ["note_unreadable"]
    assert report["audited_notes"] == 2


def test_audit_ignores_hidden_symlink_and_generated_harness_notes(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Visible.md", "# visible")
    note(vault / ".hidden/Hidden.md", "[[missing]]")
    note(vault / "_harness/decisions/generated.md", "[[missing]]")
    note(vault / "tasks/42/plan.md", "[[missing]]")
    outside = tmp_path / "Outside.md"
    note(outside, "[[missing]]")
    (vault / "linked.md").symlink_to(outside)
    report = audit_vault(vault, required_notes=())
    assert report["markdown_notes"] == 5
    assert report["audited_notes"] == 1
    assert report["excluded_notes"] == 4
    assert report["findings"] == []


def test_audit_rejects_nonpositive_age_limit(tmp_path):
    for invalid in (0, True, 1.5):
        try:
            audit_vault(tmp_path, max_review_age_days=invalid)
        except ValueError as exc:
            assert str(exc) == "max_review_age_days must be positive"
        else:
            raise AssertionError("invalid age limit was accepted")


def test_content_governance_accepts_typed_curated_taxonomy(tmp_path):
    from harness.vault_audit import CURATED_NOTE_TYPES

    vault = tmp_path / "vault"
    source = tmp_path / "reference.md"
    source.write_text("canonical source", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for relative, note_type in CURATED_NOTE_TYPES.items():
        path = vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntype: {note_type}\nlast_reviewed: {datetime.now(UTC).date()}\nsources:\n  - path: reference.md\n    sha256: {digest}\n---\n# Curated",
            encoding="utf-8",
        )
    report = audit_vault(
        vault, required_notes=(), content_governance=True, source_root=tmp_path
    )
    assert report["findings"] == []


def test_new_curated_knowledge_notes_are_linked_and_source_verified(tmp_path):
    from harness.vault_audit import CURATED_NOTE_TYPES

    vault = tmp_path / "vault"
    source_root = tmp_path / "source"
    source_root.mkdir()
    targets = (
        "architecture/Modelle und Routing.md",
        "architecture/Git und Isolation.md",
        "operations/Lokaler Betrieb.md",
    )
    links = []
    for index, relative in enumerate(targets):
        source = source_root / f"source-{index}.md"
        source.write_text(f"canonical source {index}", encoding="utf-8")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        path = vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            + yaml.safe_dump(
                {
                    "type": CURATED_NOTE_TYPES[relative],
                    "last_reviewed": datetime.now(UTC).date().isoformat(),
                    "sources": [{"path": source.name, "sha256": source_hash}],
                },
                sort_keys=False,
            )
            + f"---\n# Note {index}\n",
            encoding="utf-8",
        )
        links.append(f"[[{Path(relative).stem}]]")
    note(vault / "Index.md", "\n".join(links))

    report = audit_vault(
        vault,
        required_notes=(),
        content_governance=True,
        source_root=source_root,
    )
    assert report["findings"] == []

    (source_root / "source-1.md").write_text("changed source", encoding="utf-8")
    stale = audit_vault(
        vault,
        required_notes=(),
        content_governance=True,
        source_root=source_root,
    )
    assert stale["findings"] == [
        {
            "code": "content_source_stale",
            "path": targets[1],
            "message": "Referenced source SHA-256 has changed",
        }
    ]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ({"path": "../outside.md", "sha256": "0" * 64}, "content_sources_invalid"),
        ({"path": "absent.md", "sha256": "0" * 64}, "content_source_unavailable"),
        ({"path": "reference.md", "sha256": "0" * 64}, "content_source_stale"),
        ({"path": "reference.md", "sha256": "invalid"}, "content_sources_invalid"),
    ],
)
def test_content_governance_audits_source_provenance(tmp_path, reference, expected):
    vault = tmp_path / "vault"
    (tmp_path / "reference.md").write_text("current", encoding="utf-8")
    note_path = vault / "Willkommen.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        "---\n"
        + yaml.safe_dump(
            {
                "type": "vault-home",
                "last_reviewed": datetime.now(UTC).date().isoformat(),
                "sources": [reference],
            },
            sort_keys=False,
        )
        + "---\n# Home",
        encoding="utf-8",
    )
    report = audit_vault(
        vault, required_notes=(), content_governance=True, source_root=tmp_path
    )
    assert expected in {finding["code"] for finding in report["findings"]}


def test_content_governance_requires_source_references(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Willkommen.md", "# Home")
    report = audit_vault(vault, required_notes=(), content_governance=True)
    assert "content_sources_missing" in {item["code"] for item in report["findings"]}


@pytest.mark.parametrize(
    "source_root,reference,expected",
    [
        (
            "missing-root",
            {"path": "ref.md", "sha256": "0" * 64},
            "content_source_unavailable",
        ),
        ("root", "not-a-mapping", "content_sources_invalid"),
        ("root", {"path": "folder", "sha256": "0" * 64}, "content_source_invalid"),
    ],
)
def test_content_governance_rejects_unavailable_or_invalid_sources(
    tmp_path, source_root, reference, expected
):
    import yaml

    vault = tmp_path / "vault"
    note_path = vault / "Willkommen.md"
    note_path.parent.mkdir(parents=True)
    text = (
        "---\n"
        + yaml.safe_dump(
            {
                "type": "vault-home",
                "last_reviewed": datetime.now(UTC).date().isoformat(),
                "sources": [reference],
            },
            sort_keys=False,
        )
        + "---\n# Home"
    )
    note_path.write_text(text, encoding="utf-8")
    root = tmp_path / source_root
    root.mkdir(exist_ok=True)
    (root / "folder").mkdir(exist_ok=True)
    report = audit_vault(
        vault,
        required_notes=(),
        content_governance=True,
        source_root=root if source_root != "missing-root" else root / "absent",
    )
    assert expected in {item["code"] for item in report["findings"]}


def test_content_governance_rejects_symlinked_source(tmp_path):
    import yaml

    root = tmp_path / "sources"
    root.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("outside", encoding="utf-8")
    (root / "alias.md").symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    vault = tmp_path / "vault"
    path = vault / "Willkommen.md"
    path.parent.mkdir()
    path.write_text(
        "---\n"
        + yaml.safe_dump(
            {
                "type": "vault-home",
                "last_reviewed": datetime.now(UTC).date().isoformat(),
                "sources": [{"path": "alias.md", "sha256": digest}],
            },
            sort_keys=False,
        )
        + "---\n# Home",
        encoding="utf-8",
    )
    report = audit_vault(
        vault, required_notes=(), content_governance=True, source_root=root
    )
    assert "content_source_invalid" in {item["code"] for item in report["findings"]}


@pytest.mark.parametrize(
    "frontmatter,expected",
    [
        ("---\nlast_reviewed: 2026-09-30\n---\n", "content_type_missing"),
        ("# Welcome without frontmatter\n", "content_type_missing"),
        ("---\ntype: other\nlast_reviewed: 2026-09-30\n---\n", "content_type_invalid"),
        ("---\ntype: [\nlast_reviewed: 2026-09-30\n---\n", "content_type_invalid"),
        ("---\n\n---\n", "content_type_invalid"),
    ],
)
def test_content_governance_rejects_bad_curated_frontmatter(
    tmp_path, frontmatter, expected
):
    vault = tmp_path / "vault"
    path = vault / "Willkommen.md"
    path.parent.mkdir(parents=True)
    path.write_text(frontmatter + "# Welcome\n", encoding="utf-8")
    codes = {
        finding["code"]
        for finding in audit_vault(vault, required_notes=(), content_governance=True)[
            "findings"
        ]
    }
    assert expected in codes


def test_content_governance_reports_unlinked_additional_notes(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Additional.md", "# Not linked")
    codes = {
        finding["code"]
        for finding in audit_vault(vault, required_notes=(), content_governance=True)[
            "findings"
        ]
    }
    assert "orphan_note" in codes


def test_content_governance_accepts_additional_note_with_incoming_link(tmp_path):
    vault = tmp_path / "vault"
    note(vault / "Index.md", "See [[Additional]].")
    note(vault / "Additional.md", "# Linked note")
    report = audit_vault(vault, required_notes=(), content_governance=True)
    assert not any(item["code"] == "orphan_note" for item in report["findings"])
