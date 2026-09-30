from datetime import UTC, date, datetime, timedelta
from pathlib import Path

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
        "markdown_notes": 8,
        "audited_notes": 8,
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
