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
    return VaultKnowledgeService(tmp_path)


def test_get_note_extracts_frontmatter_headings_and_links(knowledge):
    result = knowledge.get_note("index.md")
    assert result["metadata"] == {"type": "architecture", "tags": ["vault"]}
    assert result["headings"] == ["Index"]
    assert result["links"] == ["Architecture"]
    assert "See" in result["content"]


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
