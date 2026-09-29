from types import SimpleNamespace

import httpx
import pytest

from harness.core import Config, Orchestrator, Store
from harness.memory import EmbeddingProvider, QdrantMemory
from harness.memory_service import MemoryService


def _response(payload):
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "http://embedding/embeddings"),
    )


def test_embedding_batch_is_sorted_by_response_index_and_validated(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return _response(
            {
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )

    monkeypatch.setattr("harness.memory.httpx.post", post)
    provider = EmbeddingProvider("http://embedding", "model", dimension=2)
    assert provider.embed_many(["first", "second"]) == [
        [0.1, 0.2],
        [0.2, 0.3],
    ]
    assert calls == [{"model": "model", "input": ["first", "second"]}]


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [0.1, 0.2]}],
        [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 0, "embedding": [0.2, 0.3]},
        ],
        [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 2, "embedding": [0.2, 0.3]},
        ],
        [
            {"index": 0, "embedding": [0.1]},
            {"index": 1, "embedding": [0.2, 0.3]},
        ],
    ],
)
def test_embedding_batch_rejects_incomplete_duplicate_and_invalid_vectors(
    monkeypatch, data
):
    monkeypatch.setattr(
        "harness.memory.httpx.post", lambda *args, **kwargs: _response({"data": data})
    )
    with pytest.raises(ValueError):
        EmbeddingProvider("http://embedding", "model", dimension=2).embed_many(
            ["first", "second"]
        )


def test_embedding_batch_empty_and_size_limit(monkeypatch):
    monkeypatch.setattr(
        "harness.memory.httpx.post",
        lambda *args, **kwargs: pytest.fail("empty batches must not make requests"),
    )
    provider = EmbeddingProvider("http://embedding", "model", batch_size=1)
    assert provider.embed_many([]) == []
    with pytest.raises(ValueError, match="batch"):
        provider.embed_many(["first", "second"])
    with pytest.raises(ValueError, match="list of strings"):
        provider.embed_many("not a list")
    with pytest.raises(ValueError, match="list of strings"):
        provider.embed_many([1])
    with pytest.raises(ValueError, match="positive integer"):
        EmbeddingProvider("http://embedding", "model", dimension=True)
    with pytest.raises(ValueError, match="batch size"):
        EmbeddingProvider("http://embedding", "model", batch_size=257)


def test_embedding_batch_rejects_non_object_response_items(monkeypatch):
    monkeypatch.setattr(
        "harness.memory.httpx.post",
        lambda *args, **kwargs: _response({"data": [None]}),
    )
    with pytest.raises(TypeError, match="response item"):
        EmbeddingProvider("http://embedding", "model", dimension=2).embed_many(
            ["first"]
        )


def test_qdrant_upsert_many_embeds_and_writes_aligned_points(monkeypatch):
    seen = []

    class Embedder:
        def embed_many(self, texts):
            assert texts == ["one", "two"]
            return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(
        "harness.memory.httpx.put",
        lambda url, **kwargs: seen.append(kwargs["json"]) or _response({}),
    )
    qdrant = QdrantMemory("http://qdrant", "notes", 2, Embedder())
    assert qdrant.upsert_many(
        [("note:0", "one", {"chunk": 0}), ("note:1", "two", {"chunk": 1})]
    )
    assert [point["vector"] for point in seen[0]["points"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert [point["payload"]["chunk"] for point in seen[0]["points"]] == [0, 1]


def test_qdrant_upsert_many_rejects_bad_batches_without_upload(monkeypatch):
    monkeypatch.setattr(
        "harness.memory.httpx.put", lambda *args, **kwargs: pytest.fail("no upload")
    )
    qdrant = QdrantMemory("http://qdrant", "notes", 2)
    assert qdrant.upsert_many([])
    with pytest.raises(ValueError, match="embedding provider"):
        qdrant.upsert_many([("one", "text", {})])

    with pytest.raises(ValueError, match="each point"):
        QdrantMemory("http://qdrant", "notes", 2, SimpleNamespace()).upsert_many(
            [("one", "text")]
        )
    for points, error in (
        ([(1, "text", {})], TypeError),
        ([("", "text", {})], ValueError),
        ([("one", 1, {})], TypeError),
        (
            [("same", "one", {}), ("same", "two", {})],
            ValueError,
        ),
    ):
        with pytest.raises(error):
            QdrantMemory("http://qdrant", "notes", 2, SimpleNamespace()).upsert_many(
                points
            )


def test_qdrant_upsert_many_supports_single_embedder_and_validates_count(monkeypatch):
    embedder = SimpleNamespace(embed=lambda text: [0.1, 0.2])
    qdrant = QdrantMemory("http://qdrant", "notes", 2, embedder)
    # Exercise the legacy embed-only adapter without issuing network I/O.
    from harness import memory

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(memory, "request", lambda *args, **kwargs: Response())
    assert qdrant.upsert_many([("one", "text", {})])
    qdrant.embedder = SimpleNamespace(embed_many=lambda texts: [])
    with pytest.raises(ValueError, match="count"):
        qdrant.upsert_many([("one", "text", {})])


def test_memory_service_batches_and_does_not_prune_after_failed_batch(tmp_path):
    notes = SimpleNamespace(
        path=lambda name: tmp_path / f"{name}.md",
        read=lambda name: "abcdefghijk",
    )
    old = [{"id": "old", "payload": {"source": "knowledge", "source_type": "obsidian"}}]
    upserted = []

    class Vectors:
        def ensure_collection(self):
            return True

        def scroll_source(self, source=None):
            return old

        def upsert_many(self, batch):
            upserted.append(batch)
            if len(upserted) == 2:
                raise RuntimeError("batch upload failed")

        def delete(self, point_ids):
            pytest.fail(f"stale points must not be deleted: {point_ids}")

    service = MemoryService(notes, Vectors(), chunk_size=4, overlap=0, batch_size=2)
    with pytest.raises(RuntimeError, match="batch upload failed"):
        service.index("knowledge")
    assert [len(batch) for batch in upserted] == [2, 1]
    assert old[0]["id"] == "old"


@pytest.mark.parametrize("batch_size", [0, 257, True, 1.5])
def test_memory_service_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch"):
        MemoryService(SimpleNamespace(), SimpleNamespace(), batch_size=batch_size)


def test_orchestrator_uses_embedding_config_dimensions_and_batch_size(tmp_path):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "harness.db")
    config.data["paths"]["obsidian_vault"] = str(tmp_path / "vault")
    config.data["memory"]["embeddings"].pop("dimensions")
    orchestrator = Orchestrator(Store(config), config)
    assert orchestrator.qdrant.dimension == 1024
    assert orchestrator.qdrant.embedder.dimension == 1024
    assert orchestrator.memory_service.batch_size == 32


@pytest.mark.parametrize("batch_size", [0, 257, True, 1.5])
def test_config_rejects_invalid_embedding_batch_size(batch_size):
    config = Config()
    config.data["memory"]["embeddings"]["batch_size"] = batch_size
    with pytest.raises(ValueError, match="memory.embeddings.batch_size"):
        config.validate()
