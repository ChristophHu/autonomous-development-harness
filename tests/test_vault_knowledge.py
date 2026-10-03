import hashlib
from datetime import date

import pytest

from harness.vault_knowledge import VaultKnowledgeService


@pytest.fixture
def knowledge(tmp_path):
    (tmp_path / "index.md").write_text(
        "---\ntype: architecture\ntags: [vault]\n---\n# Index\nSee [[Architecture]].\n"
    )
    (tmp_path / "Architecture.md").write_text(
        "---\ntype: design\n---\n# Overview\n## Storage\nQdrant stores vectors.\n"
    )
    return VaultKnowledgeService(tmp_path, today=date(2026, 10, 3))


def test_get_note_extracts_frontmatter_headings_and_links(knowledge):
    result = knowledge.get_note("index.md")
    assert result["metadata"] == {"type": "architecture", "tags": ["vault"]}
    assert result["headings"] == ["Index"]
    assert result["links"] == ["Architecture"]
    assert "See" in result["content"]


def test_get_note_returns_json_serializable_review_dates(knowledge, tmp_path):
    import json

    (tmp_path / "review.md").write_text(
        "---\nlast_reviewed: 2026-10-03\n---\n# Review\nVault"
    )
    note = knowledge.get_note("review.md")
    assert note["metadata"]["last_reviewed"] == "2026-10-03"
    assert json.loads(json.dumps(note))["path"] == "review.md"


def test_get_note_rejects_non_json_frontmatter_values(knowledge, tmp_path):
    (tmp_path / "bad.md").write_text("---\nvalue: !!set {one: null}\n---\n# Bad")
    with pytest.raises(TypeError, match="non-JSON"):
        knowledge.get_note("bad.md")


def test_get_note_without_frontmatter(knowledge, tmp_path):
    (tmp_path / "plain.md").write_text("# Plain\nbody")
    result = knowledge.get_note("plain.md")
    assert result["metadata"] == {}
    assert result["headings"] == ["Plain"]


def test_empty_frontmatter_is_empty_mapping(knowledge, tmp_path):
    (tmp_path / "empty.md").write_text("---\nnull\n---\n# Empty")
    assert knowledge.get_note("empty.md")["metadata"] == {}


def test_get_note_rejects_invalid_yaml(knowledge, tmp_path):
    (tmp_path / "bad.md").write_text("---\n: broken: [\n---\n")
    with pytest.raises(ValueError, match="frontmatter"):
        knowledge.get_note("bad.md")


def test_get_note_rejects_non_mapping_frontmatter(knowledge, tmp_path):
    (tmp_path / "bad.md").write_text("---\n- item\n---\n")
    with pytest.raises(TypeError, match="mapping"):
        knowledge.get_note("bad.md")


def test_backlinks_resolve_vault_link_by_note_stem(knowledge):
    assert knowledge.backlinks("Architecture.md") == ["index.md"]


def test_backlinks_honor_limit_and_skip_unreadable(knowledge, monkeypatch):
    original = knowledge._read

    def read(path):
        if path == "Architecture.md":
            raise OSError("unreadable")
        return original(path)

    monkeypatch.setattr(knowledge, "_read", read)
    assert knowledge.backlinks("Architecture.md") == ["index.md"]
    assert knowledge.backlinks("Architecture.md", limit=1) == ["index.md"]


@pytest.mark.parametrize("limit", [0, 101, True, "1"])
def test_backlinks_reject_bad_limit(knowledge, limit):
    with pytest.raises(ValueError, match="limit"):
        knowledge.backlinks("Architecture.md", limit=limit)


def test_search_is_case_insensitive_and_bounded(knowledge):
    results = knowledge.search("qdrant", context_chars=20)
    assert results[0]["path"] == "Architecture.md"
    assert "Qdrant" in results[0]["excerpt"]
    assert results[0]["source_ref"] == "context:vault/Architecture.md#Storage"
    assert results[0]["section"] == "Storage"
    assert results[0]["review_state"] == "unreviewed"


def test_search_returns_review_provenance_and_deterministic_relevance(
    knowledge, tmp_path
):
    (tmp_path / "Architecture.md").write_text(
        "---\nreviewed_on: '2026-10-01'\nsource_hashes: {src/a.py: abc}\n---\n"
        "# Overview\nQdrant stores vectors."
    )
    (tmp_path / "qdrant.md").write_text("# Qdrant\nQdrant configuration.")
    matches = knowledge.search("qdrant", limit=10)
    assert [item["path"] for item in matches] == ["qdrant.md", "Architecture.md"]
    assert matches[1]["review_state"] == "current"
    assert matches[1]["provenance"] == {
        "reviewed_on": "2026-10-01",
        "review_state": "current",
        "source_hashes": {"src/a.py": "abc"},
        "source_state": "invalid",
        "source_checks": [{"path": "src/a.py", "status": "invalid"}],
    }


def test_search_treats_malformed_provenance_shape_as_unverified(knowledge, tmp_path):
    (tmp_path / "provenance.md").write_text(
        "---\nreviewed_on: today\nsource_hashes: invalid\n---\nQdrant evidence"
    )
    result = next(
        item for item in knowledge.search("evidence") if item["path"] == "provenance.md"
    )
    assert result["review_state"] == "invalid"
    assert result["provenance"]["source_hashes"] == {}


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ("last_reviewed: 2026-10-02", "current"),
        ("last_reviewed: 2026-03-01", "stale"),
        ("last_reviewed: 2026-10-04", "future"),
        ("last_reviewed: not-a-date", "invalid"),
        ("", "unreviewed"),
        ("reviewed_on: 2026-10-02", "current"),
        ("last_reviewed: 2026-10-02T12:30:00Z", "current"),
        ("last_reviewed: [not, a, date]", "invalid"),
    ],
)
def test_review_freshness_states_are_explicit_and_deterministic(
    knowledge, tmp_path, frontmatter, expected
):
    (tmp_path / "review.md").write_text(
        f"---\n{frontmatter}\nclaims: {{goal: untrusted-unless-current}}\n---\nQdrant review"
    )
    result = next(
        item for item in knowledge.search("review") if item["path"] == "review.md"
    )
    assert result["review_state"] == expected
    assert bool(result["claims"]) is (expected == "current")


def test_review_freshness_rejects_invalid_age_configuration(tmp_path):
    with pytest.raises(ValueError, match="max_review_age_days"):
        VaultKnowledgeService(tmp_path, max_review_age_days=True)
    with pytest.raises(ValueError, match="today"):
        VaultKnowledgeService(tmp_path, today="2026-10-03")


def test_search_reports_current_source_hashes(knowledge, tmp_path):
    source = tmp_path / "src" / "design.py"
    source.parent.mkdir()
    source.write_text("authoritative source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    note = tmp_path / "current.md"
    note.write_text(
        f"---\ntype: architecture\nsources: [{{path: src/design.py, sha256: {digest}}}]\n---\nQdrant"
    )
    service = VaultKnowledgeService(tmp_path, source_root=tmp_path)
    result = next(
        item for item in service.search("qdrant") if item["path"] == "current.md"
    )
    assert result["provenance"]["source_state"] == "current"
    assert result["provenance"]["source_checks"] == [
        {"path": "src/design.py", "status": "current"}
    ]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ('{path: src/design.py, sha256: "' + "0" * 64 + '"}', "stale"),
        ('{path: ../outside.py, sha256: "' + "0" * 64 + '"}', "invalid"),
        ('{path: src/missing.py, sha256: "' + "0" * 64 + '"}', "unavailable"),
    ],
)
def test_search_marks_stale_invalid_and_unavailable_sources(
    knowledge, tmp_path, reference, expected
):
    source = tmp_path / "src" / "design.py"
    source.parent.mkdir(exist_ok=True)
    source.write_text("changed source")
    (tmp_path / "checked.md").write_text(
        f"---\nsources: [{reference}]\n---\nQdrant checked"
    )
    service = VaultKnowledgeService(tmp_path, source_root=tmp_path)
    result = next(
        item for item in service.search("checked") if item["path"] == "checked.md"
    )
    assert result["provenance"]["source_state"] == expected


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ("sources: null", "unverified"),
        ("sources: {path: src/design.py}", "invalid"),
        ("sources: [malformed]", "invalid"),
        (
            "sources: ["
            + ", ".join(
                "{path: src/design.py, sha256: '" + "a" * 64 + "'}" for _ in range(33)
            )
            + "]",
            "invalid",
        ),
    ],
)
def test_search_bounds_source_declaration_shapes(
    knowledge, tmp_path, frontmatter, expected
):
    (tmp_path / "bounded.md").write_text(f"---\n{frontmatter}\n---\nBounded source")
    result = next(
        item
        for item in VaultKnowledgeService(tmp_path, source_root=tmp_path).search(
            "bounded"
        )
        if item["path"] == "bounded.md"
    )
    assert result["provenance"]["source_state"] == expected


@pytest.mark.parametrize("target_kind", ["symlink", "directory"])
def test_search_rejects_source_symlinks_and_non_files(knowledge, tmp_path, target_kind):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "target"
    if target_kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("outside")
        source.symlink_to(outside)
    else:
        source.mkdir()
    digest = "a" * 64
    (tmp_path / "safe.md").write_text(
        f"---\nsources: [{{path: target, sha256: '{digest}'}}]\n---\nSafe query"
    )
    result = next(
        item
        for item in VaultKnowledgeService(tmp_path, source_root=source_root).search(
            "safe"
        )
        if item["path"] == "safe.md"
    )
    assert result["provenance"]["source_state"] == "invalid"


def test_search_skips_notes_with_invalid_frontmatter(knowledge, tmp_path):
    (tmp_path / "broken.md").write_text("---\nclaims: [unterminated\n---\nQdrant")
    assert all(item["path"] != "broken.md" for item in knowledge.search("qdrant"))


def test_search_skips_unreadable_and_stops_at_limit(knowledge, monkeypatch):
    original = knowledge._read

    def read(path):
        if path == "index.md":
            raise OSError("unreadable")
        if path == "Architecture.md":
            return "Qdrant vectors"
        return original(path)

    monkeypatch.setattr(knowledge, "_read", read)
    assert len(knowledge.search("qdrant", limit=1)) == 1
    assert knowledge.search("not present") == []


@pytest.mark.parametrize("query", ["", "  ", None])
def test_search_rejects_empty_query(knowledge, query):
    with pytest.raises(ValueError, match="query"):
        knowledge.search(query)


def test_search_rejects_query_without_search_terms(knowledge):
    with pytest.raises(ValueError, match="searchable characters"):
        knowledge.search("--- ...")


@pytest.mark.parametrize("limit", [0, 101, True, 1.2])
def test_search_rejects_bad_limit(knowledge, limit):
    with pytest.raises(ValueError, match="limit"):
        knowledge.search("qdrant", limit=limit)


@pytest.mark.parametrize("context_chars", [19, 1001, True, "100"])
def test_search_rejects_bad_context_size(knowledge, context_chars):
    with pytest.raises(ValueError, match="context_chars"):
        knowledge.search("qdrant", context_chars=context_chars)


def test_service_cannot_escape_vault(knowledge):
    with pytest.raises((ValueError, PermissionError)):
        knowledge.get_note("../outside.md")


def test_symlink_note_is_not_readable(tmp_path, knowledge):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("secret")
    (tmp_path / "linked.md").symlink_to(outside)
    with pytest.raises((OSError, ValueError, PermissionError)):
        knowledge.get_note("linked.md")
