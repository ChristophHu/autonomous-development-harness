from types import SimpleNamespace

import httpx
import pytest

from harness.memory import (
    ContextBuilder,
    EmbeddingProvider,
    ObsidianMemory,
    QdrantMemory,
    context_evidence_envelope,
    context_evidence_size,
)
from harness.memory_service import MemoryService


def response(ok=True, payload=None):
    return httpx.Response(
        200 if ok else 503,
        json=payload or {},
        request=httpx.Request("POST", "http://test"),
    )


def test_obsidian_read_write_append_search(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    assert memory.read("missing") is None
    memory.write("nested/note", "Alpha")
    memory.append("nested/note", "Beta")
    assert "Beta" in memory.read("nested/note") and len(memory.search("alpha")) == 1
    assert memory.search("not-found") == []


def test_obsidian_status_construction_does_not_create_missing_vault(tmp_path):
    vault = tmp_path / "missing-vault"

    ObsidianMemory(vault)

    assert not vault.exists()


def test_context_builder_memory_retrieval(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("decisions", "approved")
    memory.write("task", "task relevant")
    qdrant = SimpleNamespace(
        search=lambda *args, **kwargs: [
            {
                "id": "vector-task",
                "payload": {
                    "text": "task relevant",
                    "source": "task",
                    "source_hash": MemoryService.digest(memory.read("task")),
                },
            }
        ]
    )
    context = ContextBuilder(memory, None, qdrant).build(
        SimpleNamespace(title="task", description="details"), "workspace"
    )
    assert "task relevant" in context and "Obsidian match" in context
    offline = SimpleNamespace(
        search=lambda *args, **kwargs: (_ for _ in ()).throw(
            httpx.ConnectError("offline")
        )
    )
    assert "Qdrant unavailable" in ContextBuilder(memory, None, offline).build(
        SimpleNamespace(title="x", description=""), "."
    )


def test_context_builder_includes_source_hashed_symbols_and_related_tests(tmp_path):
    source = tmp_path / "src" / "memory.py"
    source.parent.mkdir()
    source.write_text(
        "class ContextBuilder:\n    def build(self):\n        return 'context'\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_memory.py").write_text("from harness.memory import ContextBuilder\n")
    memory = ObsidianMemory(tmp_path / "vault")
    result = ContextBuilder(memory, None).build_evidence(
        SimpleNamespace(
            title="ContextBuilder retrieval",
            description="Find the memory context builder",
        ),
        str(tmp_path),
    )
    match = next(
        item for item in result["fragments"] if item["kind"] == "repository_symbol"
    )
    assert match["ref"] == "context:repository/src/memory.py#ContextBuilder"
    assert "tests/test_memory.py" in match["text"]
    assert len(match["provenance"]["sha256"]) == 64
    assert "Qdrant unavailable" not in ContextBuilder(memory, None).build(
        SimpleNamespace(title="x", description=""), "."
    )


def test_context_builder_returns_cited_fragments_and_source_conflicts(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write(
        "knowledge/architecture",
        "---\nlast_reviewed: 2026-10-01\nclaims:\n  goal: alpha\n---\n# Design\nTask architecture alpha.",
    )
    qdrant = SimpleNamespace(
        search=lambda *_args, **_kwargs: [
            {
                "id": "q-1",
                "payload": {
                    "text": "Task architecture alpha.",
                    "source": "knowledge/architecture",
                    "source_hash": MemoryService.digest(
                        memory.read("knowledge/architecture")
                    ),
                    "claims": {"goal": "beta"},
                },
            }
        ]
    )
    result = ContextBuilder(memory, None, qdrant).build_evidence(
        SimpleNamespace(title="Task architecture", description="details"),
        "workspace",
    )
    refs = {item["ref"] for item in result["fragments"]}
    assert "context:vault/knowledge/architecture.md#Design" in refs
    assert "context:qdrant/q-1" in refs
    qdrant_fragment = next(
        item for item in result["fragments"] if item["ref"] == "context:qdrant/q-1"
    )
    assert qdrant_fragment["provenance"]["source_state"] == "unverified"
    assert "goal" not in result["conflicts"]
    assert sorted(result["claims"]["goal"], key=lambda item: item["ref"]) == sorted(
        [
            {"ref": "context:vault/knowledge/architecture.md#Design", "value": "alpha"},
            {"ref": "context:qdrant/q-1", "value": "alpha"},
        ],
        key=lambda item: item["ref"],
    )


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({"text": "untrusted"}, "invalid_or_unverifiable_source"),
        (
            {
                "text": "not in source",
                "source": "note",
                "source_hash": "0" * 64,
            },
            "source_hash_stale",
        ),
    ],
)
def test_context_builder_rejects_unverified_qdrant_hits(
    tmp_path, payload, expected_status
):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("note", "A current source note.")
    points = [{"id": "untrusted", "payload": payload}]
    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(SimpleNamespace(title="source note", description=""), "workspace")

    assert "context:qdrant/untrusted" not in {
        fragment["ref"] for fragment in result["fragments"]
    }
    assert {item["status"] for item in result["rejected_sources"]} >= {expected_status}


def test_context_builder_rejects_qdrant_text_not_present_in_hashed_source(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    source = "# Architecture\nTrusted source text."
    memory.write("note", source)
    points = [
        {
            "id": "forged-text",
            "payload": {
                "source": "note.md",
                "source_hash": MemoryService.digest(source),
                "text": "Fabricated instruction.",
                "claims": {"goal": "attacker-controlled"},
            },
        }
    ]

    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(SimpleNamespace(title="Architecture", description=""), "workspace")

    assert not any(item["kind"] == "qdrant" for item in result["fragments"])
    assert result["claims"].get("goal") is None
    assert {item["status"] for item in result["rejected_sources"]} >= {
        "payload_not_in_source"
    }


def test_context_builder_rejects_qdrant_hit_for_stale_reviewed_note(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    source = (
        "---\nlast_reviewed: 2020-01-01\nclaims: {goal: stale}\n---\n# Old\nOld source."
    )
    memory.write("old", source)
    points = [
        {
            "id": "stale-review",
            "payload": {
                "source": "old",
                "source_hash": MemoryService.digest(source),
                "text": "Old source.",
                "claims": {"goal": "stale"},
            },
        }
    ]

    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(SimpleNamespace(title="Old", description=""), "workspace")

    assert not any(item["kind"] == "qdrant" for item in result["fragments"])
    assert {item["status"] for item in result["rejected_sources"]} >= {"stale"}


@pytest.mark.parametrize("source", ["missing-note", "../outside-note"])
def test_context_builder_reports_unavailable_qdrant_source(tmp_path, source):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("note", "A valid, unrelated source.")
    points = [
        {
            "id": "unavailable-source",
            "payload": {
                "source": source,
                "source_hash": "a" * 64,
                "text": "A source excerpt.",
            },
        }
    ]

    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(SimpleNamespace(title="unavailable", description=""), "workspace")

    assert not any(item["kind"] == "qdrant" for item in result["fragments"])
    assert {item["status"] for item in result["rejected_sources"]} >= {
        "source_unavailable"
    }


def test_context_builder_rejects_qdrant_note_with_stale_repository_provenance(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "design.py").write_text("current repository content")
    memory = ObsidianMemory(tmp_path / "vault")
    source = (
        "---\nlast_reviewed: 2026-10-04\nsources: "
        '[{path: design.py, sha256: "' + "0" * 64 + '"}]\n'
        'claims: {goal: "stale source claim"}\n---\n'
        "# Architecture\nTask architecture context."
    )
    memory.write("architecture", source)
    points = [
        {
            "id": "stale-dependency",
            "payload": {
                "source": "architecture",
                "source_hash": MemoryService.digest(source),
                "text": "Task architecture context.",
            },
        }
    ]

    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(
        SimpleNamespace(title="Task architecture", description="context"),
        str(workspace),
    )

    assert not any(item["kind"] == "qdrant" for item in result["fragments"])
    assert result["claims"].get("goal") is None
    assert {item["status"] for item in result["rejected_sources"]} >= {"source_stale"}


@pytest.mark.parametrize(
    "last_reviewed,expected_tier",
    [
        ("2026-10-04", "qdrant_hit_current_source_and_review"),
        (None, "qdrant_hit_current_source_unreviewed"),
    ],
)
def test_context_builder_classifies_qdrant_with_current_repository_source(
    tmp_path, last_reviewed, expected_tier
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_path = workspace / "design.py"
    source_path.write_text("current repository design")
    digest = MemoryService.digest(source_path.read_text())
    review = f"last_reviewed: {last_reviewed}\n" if last_reviewed else ""
    source = (
        f'---\n{review}sources: [{{path: design.py, sha256: "{digest}"}}]\n'
        "---\n# Architecture\nTask architecture context."
    )
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("architecture", source)
    points = [
        {
            "id": "current-source",
            "payload": {
                "source": "architecture",
                "source_hash": MemoryService.digest(source),
                "text": "Task architecture context.",
            },
        }
    ]

    result = ContextBuilder(
        memory, None, SimpleNamespace(search=lambda *_args, **_kwargs: points)
    ).build_evidence(
        SimpleNamespace(title="Task architecture", description="context"),
        str(workspace),
    )
    fragment = next(item for item in result["fragments"] if item["kind"] == "qdrant")

    assert fragment["provenance"]["source_state"] == "current"
    assert ContextBuilder._trust_assessment(fragment)["tier"] == expected_tier


def test_context_builder_excludes_stale_vault_sources_with_diagnostic(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "design.py"
    source.write_text("changed")
    vault = tmp_path / "vault"
    memory = ObsidianMemory(vault)
    memory.write(
        "architecture",
        '---\nsources: [{path: design.py, sha256: "'
        + "0" * 64
        + '"}]\nclaims: {goal: stale}\n---\n# Architecture\nTask architecture',
    )
    result = ContextBuilder(memory, None).build_evidence(
        SimpleNamespace(title="Task architecture", description="details"),
        str(workspace),
    )
    assert not any(item["kind"] == "vault" for item in result["fragments"])
    assert result["rejected_sources"] == [
        {"ref": "context:vault/architecture.md#Architecture", "status": "stale"}
    ]


def test_context_builder_rejects_stale_review_and_keeps_unreviewed_as_nonclaim(
    tmp_path,
):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write(
        "stale",
        "---\nlast_reviewed: 2020-01-01\nclaims: {goal: stale}\n---\n# Stale\nTask freshness",
    )
    memory.write(
        "unreviewed",
        "---\nclaims: {goal: untrusted}\n---\n# Unreviewed\nTask freshness",
    )
    result = ContextBuilder(memory, None).build_evidence(
        SimpleNamespace(title="Task freshness", description="details"), "workspace"
    )
    assert "context:vault/stale.md#Stale" not in {
        fragment["ref"] for fragment in result["fragments"]
    }
    assert result["claims"].get("goal") is None
    assert {item["status"] for item in result["rejected_sources"]} >= {"stale"}
    unreviewed = next(
        fragment
        for fragment in result["fragments"]
        if fragment["ref"] == "context:vault/unreviewed.md#Unreviewed"
    )
    assert unreviewed["review_state"] == "unreviewed"


def test_context_evidence_budget_fails_closed_when_required_envelope_too_large(
    tmp_path,
):
    builder = ContextBuilder(ObsidianMemory(tmp_path / "vault"), None, max_bytes=300)
    with pytest.raises(ValueError, match="required context evidence"):
        builder.build_evidence(
            SimpleNamespace(title="short", description="short"), "workspace"
        )


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5])
def test_context_builder_rejects_invalid_byte_budgets(tmp_path, max_bytes):
    with pytest.raises(ValueError, match="positive integer"):
        ContextBuilder(ObsidianMemory(tmp_path / "vault"), None, max_bytes=max_bytes)


def test_context_builder_enforces_utf8_byte_budget_without_truncating_core(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("notes/task", "Task matching details " + "🙂" * 100)
    builder = ContextBuilder(memory, None, max_bytes=900)
    result = builder.build_evidence(
        SimpleNamespace(title="task", description="details"), "workspace"
    )
    assert result["used_bytes"] == context_evidence_size(result)
    assert result["used_bytes"] <= 900
    assert result["truncated_sources"] or result["omitted_sources"]
    assert "�" not in result["text"]
    with pytest.raises(ValueError, match="required context exceeds"):
        ContextBuilder(memory, None, max_bytes=20).build_evidence(
            SimpleNamespace(title="long mandatory title", description="details"),
            "workspace",
        )


def test_context_builder_skips_large_optional_note_and_keeps_smaller_evidence(
    tmp_path,
):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write(
        "large",
        "---\nlast_reviewed: 2026-10-03\n---\n# Large\n"
        + "X" * 1000
        + "Task knowledge " * 1000,
    )
    memory.write(
        "small",
        "---\nlast_reviewed: 2026-10-03\n---\n# Small\n"
        "Task knowledge has a concise source.",
    )
    result = ContextBuilder(memory, None, max_bytes=1450).build_evidence(
        SimpleNamespace(title="Task knowledge", description=""), str(tmp_path)
    )
    refs = {fragment["ref"] for fragment in result["fragments"]}
    assert "context:vault/large.md#Large" in result["omitted_sources"]
    assert "context:vault/small.md#Small" in refs
    assert result["truncated_sources"] == []
    assert result["used_bytes"] <= result["budget_bytes"]


@pytest.mark.parametrize(
    ("max_bytes", "ref", "text", "raises"),
    [
        (650, "x" * 500, "small", True),
        (1000, "optional", "x" * 2000, False),
    ],
)
def test_context_builder_omits_optional_fragment_when_envelope_or_content_exceeds(
    tmp_path, monkeypatch, max_bytes, ref, text, raises
):
    builder = ContextBuilder(
        ObsidianMemory(tmp_path / "vault"), None, max_bytes=max_bytes
    )
    ordered = builder._ordered_fragments

    def with_optional(fragments, task):
        return ordered(fragments, task) + [
            {"kind": "vault", "ref": ref, "text": text, "required": False}
        ]

    monkeypatch.setattr(builder, "_ordered_fragments", with_optional)
    if raises:
        with pytest.raises(ValueError, match="required context evidence"):
            builder.build_evidence(
                SimpleNamespace(title="t", description="d"), "workspace"
            )
    else:
        result = builder.build_evidence(
            SimpleNamespace(title="t", description="d"), "workspace"
        )
        assert ref in result["omitted_sources"]


def test_context_builder_rejects_required_fragment_content_over_budget(
    tmp_path, monkeypatch
):
    builder = ContextBuilder(ObsidianMemory(tmp_path / "vault"), None, max_bytes=1000)
    ordered = builder._ordered_fragments
    monkeypatch.setattr(
        builder,
        "_ordered_fragments",
        lambda fragments, task: (
            ordered(fragments, task)
            + [
                {
                    "kind": "task",
                    "ref": "required-extra",
                    "text": "x" * 2000,
                    "required": True,
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match="required context exceeds"):
        builder.build_evidence(SimpleNamespace(title="t", description="d"), "workspace")


def test_context_builder_prioritizes_reviewed_sources_over_unreviewed():
    task = SimpleNamespace(title="task", description="")
    fragments = [
        {
            "kind": "vault",
            "ref": "unreviewed",
            "text": "task",
            "required": False,
            "review_state": "unreviewed",
            "provenance": {"source_state": "unverified"},
        },
        {
            "kind": "vault",
            "ref": "reviewed",
            "text": "task",
            "required": False,
            "review_state": "current",
            "provenance": {"source_state": "current"},
        },
        {"kind": "task", "ref": "required", "text": "task", "required": True},
    ]
    ordered = ContextBuilder._ordered_fragments(fragments, task)
    assert [item["ref"] for item in ordered] == ["required", "reviewed", "unreviewed"]


@pytest.mark.parametrize(
    ("fragment", "tier", "basis"),
    [
        (
            {
                "kind": "vault",
                "review_state": "current",
                "provenance": {"source_state": "current"},
            },
            "reviewed_source_hash_current",
            ["review_current", "source_hash_current"],
        ),
        (
            {
                "kind": "vault",
                "review_state": "unreviewed",
                "provenance": {"source_state": "current"},
            },
            "source_hash_current_unreviewed",
            ["review_not_current", "source_hash_current"],
        ),
        (
            {
                "kind": "vault",
                "review_state": "current",
                "provenance": {"source_state": "unverified"},
            },
            "reviewed_source_unverified",
            ["review_current", "source_hash_not_current"],
        ),
        (
            {"kind": "vault", "provenance": None},
            "vault_unreviewed_unverified",
            ["review_not_current", "source_hash_not_current"],
        ),
        (
            {"kind": "decision_memory", "text": "Memory decisions: approved"},
            "curated_decision_memory",
            ["curated_decision_memory"],
        ),
        (
            {"kind": "decision_memory", "text": "Memory: none"},
            "empty_memory_context",
            ["no_decision_content"],
        ),
        (
            {
                "kind": "repository_symbol",
                "provenance": {"sha256": "a" * 64},
            },
            "static_symbol_hash_identified",
            ["static_symbol", "source_hash_present", "not_semantically_reviewed"],
        ),
        (
            {"kind": "repository_symbol", "provenance": {"sha256": "invalid"}},
            "static_symbol_hash_missing",
            ["static_symbol", "source_hash_missing", "not_semantically_reviewed"],
        ),
        (
            {"kind": "repository"},
            "repository_file_unreviewed",
            ["repository_file", "not_semantically_reviewed"],
        ),
        (
            {"kind": "qdrant"},
            "retrieval_without_current_source_proof",
            ["retrieval_only", "current_source_not_verified_here"],
        ),
        (
            {"kind": "qdrant_status"},
            "availability_notice",
            ["status_only"],
        ),
        (
            {"kind": "other"},
            "unknown_origin",
            ["origin_not_classified"],
        ),
    ],
)
def test_context_trust_assessment_is_explicit_and_not_a_confidence_probability(
    fragment, tier, basis
):
    result = ContextBuilder._trust_assessment(fragment)

    assert result["tier"] == tier
    assert result["basis"] == basis
    assert isinstance(result["priority"], int)


def test_context_envelope_explains_trust_tier_used_for_selection():
    fragments = ContextBuilder._ordered_fragments(
        [
            {
                "kind": "vault",
                "ref": "vault:weak",
                "text": "task evidence",
                "required": False,
                "review_state": "unreviewed",
                "provenance": {"source_state": "unverified"},
            },
            {
                "kind": "vault",
                "ref": "vault:strong",
                "text": "task evidence",
                "required": False,
                "review_state": "current",
                "provenance": {"source_state": "current"},
            },
        ],
        SimpleNamespace(title="task", description=""),
    )
    assert [item["ref"] for item in fragments] == ["vault:strong", "vault:weak"]
    assert fragments[0]["trust"] == {
        "tier": "reviewed_source_hash_current",
        "basis": ["review_current", "source_hash_current"],
    }
    envelope = context_evidence_envelope({"fragments": fragments})
    assert envelope["fragments"][0]["trust"] == fragments[0]["trust"]
    assert "priority" not in envelope["fragments"][0]["trust"]


def test_memory_search_accepts_only_current_vault_backed_hits(tmp_path):
    notes = ObsidianMemory(tmp_path / "vault")
    notes.write("current", "source text")
    service = MemoryService(notes, SimpleNamespace())
    points = [
        {"id": "stale", "payload": {"source": "current", "source_hash": "old"}},
        {"id": "escape", "payload": {"source": "../outside", "source_hash": "old"}},
        {"id": "missing", "payload": {"source": "missing", "source_hash": "old"}},
        {
            "id": "current",
            "payload": {
                "source": "current",
                "source_hash": service.digest("source text"),
            },
        },
        {
            "id": "duplicate",
            "payload": {
                "source": "current",
                "source_hash": service.digest("source text"),
            },
        },
    ]
    assert service.verified_search_points(points) == [
        {"id": "current", "payload": {"source": "current", "text": "source text"}}
    ]
    service.vectors.search = lambda *_args, **_kwargs: points
    assert service.search("query") == service.verified_search_points(points)


@pytest.mark.parametrize("points", [{"result": []}, ["bad"], [{"payload": "bad"}]])
def test_memory_search_rejects_invalid_vector_response(tmp_path, points):
    service = MemoryService(ObsidianMemory(tmp_path / "vault"), SimpleNamespace())
    with pytest.raises(TypeError):
        service.verified_search_points(points)


def test_context_builder_bounds_structured_claim_payloads(tmp_path):
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write(
        "notes/claim",
        "---\nlast_reviewed: 2026-10-01\nclaims:\n  goal: '"
        + "x" * 5000
        + "'\n  custom_field: ignored\n---\nTask claim",
    )
    result = ContextBuilder(memory, None).build_evidence(
        SimpleNamespace(title="Task claim", description=""), "workspace"
    )
    assert result["claims"] == {}
    assert result["claim_issues"]["goal"] == ["claim exceeds the evidence size limit"]


def test_context_builder_bounds_claim_count_bytes_and_json(tmp_path, monkeypatch):
    builder = ContextBuilder(ObsidianMemory(tmp_path / "vault"), None)
    required = {
        "kind": "task",
        "ref": "context:task/title",
        "text": "task",
        "required": True,
        "claims": {},
    }

    def render(fragments):
        monkeypatch.setattr(
            builder, "_ordered_fragments", lambda _items, _task: [required, *fragments]
        )
        return builder.build_evidence(
            SimpleNamespace(title="task", description=""), "workspace"
        )

    count_limited = render(
        [
            {
                "kind": "vault",
                "ref": f"context:vault/note-{index}.md",
                "text": "evidence",
                "required": False,
                "claims": {"goal": f"value-{index}"},
            }
            for index in range(33)
        ]
    )
    assert count_limited["claim_issues"]["goal"] == [
        "context claim count exceeds the evidence limit"
    ]

    byte_limited = render(
        [
            {
                "kind": "vault",
                "ref": f"context:vault/total-{index}.md",
                "text": "evidence",
                "required": False,
                "claims": {name: value},
            }
            for index, (name, value) in enumerate(
                (
                    ("goal", "x" * 3000),
                    ("requirements", "y" * 3000),
                    ("acceptance_criteria", "z" * 3000),
                )
            )
        ]
    )
    assert byte_limited["claim_issues"]["acceptance_criteria"] == [
        "total context claim payload exceeds the evidence limit"
    ]

    invalid = render(
        [
            {
                "kind": "vault",
                "ref": "context:vault/invalid.md",
                "text": "evidence",
                "required": False,
                "claims": {"goal": float("nan"), "custom": "ignored"},
            },
            {
                "kind": "vault",
                "ref": "context:vault/oversized.md",
                "text": "evidence",
                "required": False,
                "claims": {"requirements": "x" * 5000},
            },
            {
                "kind": "vault",
                "ref": "context:vault/non-dict-claims.md",
                "text": "evidence",
                "required": False,
                "claims": [],
            },
        ]
    )
    assert invalid["claim_issues"]["goal"] == ["claim is not valid JSON"]
    assert invalid["claim_issues"]["requirements"] == [
        "claim exceeds the evidence size limit"
    ]
    assert "custom" not in invalid["claim_issues"]


def test_qdrant_http_operations(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return response(
            payload={
                "result": {
                    "config": {"params": {"vectors": {"size": 4, "distance": "Cosine"}}}
                }
            }
        )

    def put(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        return response()

    def post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        return response(payload={"result": [{"id": "x"}]})

    monkeypatch.setattr("harness.memory.httpx.get", get)
    monkeypatch.setattr("harness.memory.httpx.put", put)
    monkeypatch.setattr("harness.memory.httpx.post", post)
    memory = QdrantMemory(
        "http://qdrant/",
        "collection",
        dimension=4,
        embedder=SimpleNamespace(embed=lambda text: [0.1] * 4),
    )
    assert memory.health() and memory.collection_exists() and memory.ensure_collection()
    assert memory.upsert("task-plan", "text")
    assert memory.search("text") == [{"id": "x"}] and memory.delete(["x"])
    assert calls


def test_qdrant_api_key_is_sent_as_header(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs.get("headers")))
        return response()

    monkeypatch.setattr("harness.memory.request", fake_request)
    memory = QdrantMemory("http://qdrant", "notes", api_key="private-test-key")

    assert memory.health()
    assert calls == [("GET", "http://qdrant/healthz", {"api-key": "private-test-key"})]


def test_qdrant_failed_responses_and_embedder(monkeypatch):
    monkeypatch.setattr("harness.memory.httpx.get", lambda *a, **k: response(False))
    monkeypatch.setattr("harness.memory.httpx.put", lambda *a, **k: response(False))
    monkeypatch.setattr("harness.memory.httpx.post", lambda *a, **k: response(False))
    assert QdrantMemory("http://q", "c").health() is False
    with pytest.raises(httpx.HTTPStatusError):
        QdrantMemory("http://q", "c").ensure_collection()
    with pytest.raises(ValueError, match="embedding provider"):
        QdrantMemory("http://q", "c")._vector("x")
    with pytest.raises(httpx.HTTPStatusError):
        QdrantMemory(
            "http://q",
            "c",
            dimension=2,
            embedder=SimpleNamespace(embed=lambda text: [0.1, 0.2]),
        ).search("x")
    embedder = type("E", (), {"embed": lambda self, text: [0.1, 0.2]})()
    assert QdrantMemory("http://q", "c", dimension=2, embedder=embedder)._vector(
        "x"
    ) == [0.1, 0.2]


def test_qdrant_health_report_includes_service_and_collection_evidence():
    def handler(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, request=request)
        return httpx.Response(
            200,
            json={
                "result": {
                    "status": "green",
                    "points_count": 12,
                    "indexed_vectors_count": 12,
                    "config": {
                        "params": {"vectors": {"size": 4, "distance": "Cosine"}}
                    },
                }
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = QdrantMemory("http://qdrant", "notes", 4, client=client).health_report()

    assert report == {
        "healthy": True,
        "service": "available",
        "collection": "notes",
        "collection_exists": True,
        "collection_status": "green",
        "points_count": 12,
        "indexed_vectors_count": 12,
        "dimension": 4,
        "distance": "Cosine",
        "errors": [],
    }


def test_qdrant_health_report_fails_closed_on_missing_or_mismatched_collection():
    def missing(request):
        return httpx.Response(
            200 if request.url.path == "/healthz" else 404, request=request
        )

    client = httpx.Client(transport=httpx.MockTransport(missing))
    absent = QdrantMemory("http://qdrant", "notes", 4, client=client).health_report()
    assert absent["service"] == "available"
    assert absent["healthy"] is False
    assert absent["errors"] == ["collection_missing"]

    def mismatch(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, request=request)
        return httpx.Response(
            200,
            json={
                "result": {
                    "status": "red",
                    "config": {"params": {"vectors": {"size": 3, "distance": "Dot"}}},
                }
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(mismatch))
    report = QdrantMemory("http://qdrant", "notes", 4, client=client).health_report()
    assert report["healthy"] is False
    assert report["errors"] == ["collection_contract_mismatch", "collection_not_ready"]


def test_qdrant_health_report_redacts_transport_and_shape_failures():
    def offline(_request):
        raise httpx.ConnectError("sensitive-host-token")

    client = httpx.Client(transport=httpx.MockTransport(offline))
    report = QdrantMemory(
        "http://private.invalid", "notes", client=client
    ).health_report()
    assert report["errors"] == ["qdrant_probe_failed"]
    assert "private.invalid" not in str(report)

    def malformed(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, request=request)
        return httpx.Response(200, json={"unexpected": True}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(malformed))
    report = QdrantMemory("http://qdrant", "notes", client=client).health_report()
    assert report["errors"] == ["collection_contract_invalid"]


def test_embedding_provider(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer token"
        return response(payload={"data": [{"index": 0, "embedding": [0.1]}]})

    monkeypatch.setattr("harness.memory.httpx.post", post)
    assert EmbeddingProvider("http://embed/v1", "model", "token", dimension=1).embed(
        "text"
    ) == [0.1]
