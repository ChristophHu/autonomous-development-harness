from types import SimpleNamespace

import httpx
import pytest

from harness.memory import (
    ContextBuilder,
    EmbeddingProvider,
    ObsidianMemory,
    QdrantMemory,
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
        search=lambda *args, **kwargs: [{"payload": {"text": "vector memory"}}]
    )
    context = ContextBuilder(memory, None, qdrant).build(
        SimpleNamespace(title="task", description="details"), "workspace"
    )
    assert "vector memory" in context and "Obsidian match" in context
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
                "payload": {"text": "retrieved", "claims": {"goal": "beta"}},
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
    assert result["conflicts"]["goal"]["values"] == ["alpha", "beta"]


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
