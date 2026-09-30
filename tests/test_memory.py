from types import SimpleNamespace

import httpx
import pytest

from harness.memory import (
    ContextBuilder,
    EmbeddingProvider,
    ObsidianMemory,
    QdrantMemory,
)


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
    assert "Qdrant unavailable" not in ContextBuilder(memory, None).build(
        SimpleNamespace(title="x", description=""), "."
    )


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
