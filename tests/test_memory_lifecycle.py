import json

import httpx
import pytest

from harness.memory import ObsidianMemory, QdrantMemory
from harness.memory_service import MemoryService


class MemoryVectors:
    def __init__(self):
        self.points = {}
        self.fail_on = None
        self.upserts = 0

    def ensure_collection(self):
        return True

    def upsert(self, point_id, text, payload):
        self.upserts += 1
        if self.fail_on == self.upserts:
            raise RuntimeError("injected upsert failure")
        self.points[point_id] = {"id": point_id, "payload": payload, "text": text}

    def scroll(self, source=None, offset=None, limit=100):
        values = [
            point
            for point in self.points.values()
            if source is None or point["payload"].get("source") == source
        ]
        start = offset or 0
        page = values[start : start + limit]
        next_offset = start + limit if start + limit < len(values) else None
        return page, next_offset

    def scroll_source(self, source=None):
        points, offset = [], None
        while True:
            page, offset = self.scroll(source, offset)
            points.extend(page)
            if offset is None:
                return points

    def delete(self, point_ids):
        for point_id in point_ids:
            self.points.pop(point_id, None)
        return True

    def search(self, *args, **kwargs):
        return list(self.points.values())


def lifecycle(tmp_path, chunk_size=6, overlap=2):
    notes = ObsidianMemory(tmp_path / "vault")
    vectors = MemoryVectors()
    return notes, vectors, MemoryService(notes, vectors, chunk_size, overlap)


def test_reindex_removes_surplus_chunks_only_after_successful_upserts(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "abcdefghijk")
    assert service.index("guide") == 3
    notes.write("guide", "short")

    assert service.index("guide") == 1

    assert len(vectors.points) == 1
    point = next(iter(vectors.points.values()))
    assert point["text"] == "short"
    assert point["payload"]["source_type"] == "obsidian"


def test_reindex_failure_keeps_old_chunks_and_does_not_prune(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "abcdefghijk")
    service.index("guide")
    old_ids = set(vectors.points)
    notes.write("guide", "0123456789")
    vectors.upserts = 0
    vectors.fail_on = 2

    with pytest.raises(RuntimeError, match="injected"):
        service.index("guide")

    assert old_ids <= set(vectors.points)


def test_unindex_removes_points_for_missing_note(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "abcdefghijk")
    service.index("guide")
    notes.path("guide").unlink()

    assert service.unindex("guide") == 3
    assert vectors.points == {}


def test_reconcile_indexes_vault_and_removes_only_marked_deleted_sources(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("a", "alpha")
    notes.write("b", "bravo")
    service.index("a")
    service.index("b")
    vectors.points["foreign"] = {
        "id": "foreign",
        "payload": {"source": "obsolete", "source_type": "other"},
    }
    notes.path("a").unlink()

    result = service.reconcile()

    assert result == {
        "notes": 1,
        "chunks": 1,
        "removed_sources": 1,
        "removed_points": 1,
    }
    assert {point["payload"]["source"] for point in vectors.points.values()} == {
        "b",
        "obsolete",
    }


def test_reconcile_skips_symlinks_and_is_idempotent(tmp_path):
    notes, _vectors, service = lifecycle(tmp_path)
    notes.write("nested/guide", "safe")
    outside = tmp_path / "outside.md"
    outside.write_text("not a vault source")
    (notes.vault / "linked.md").symlink_to(outside)

    assert notes.list_documents() == ["nested/guide"]
    assert service.reconcile()["notes"] == 1
    assert service.reconcile() == {
        "notes": 1,
        "chunks": 1,
        "removed_sources": 0,
        "removed_points": 0,
    }


def test_reconcile_plan_is_read_only_and_apply_is_selective(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("stable", "unchanged")
    notes.write("changed", "old")
    service.index("stable")
    service.index("changed")
    notes.write("changed", "new")
    notes.write("missing", "new note")
    vectors.points["orphan"] = {
        "id": "orphan",
        "payload": {
            "source": "removed",
            "source_type": "obsidian",
            "source_hash": "0" * 64,
            "chunk": 0,
            "text": "old",
        },
    }
    before = dict(vectors.points)
    upserts = vectors.upserts

    plan = service.plan_reconciliation()

    assert plan["needs_index"] == ["changed", "missing"]
    assert plan["orphan_sources"] == ["removed"]
    assert plan["orphan_points"] == {"removed": ["orphan"]}
    assert vectors.points == before
    assert vectors.upserts == upserts
    result = service.reconcile()
    assert result["removed_sources"] == 1
    assert result["removed_points"] == 1
    assert "orphan" not in vectors.points
    assert service.plan_reconciliation()["needs_index"] == []
    assert service.plan_reconciliation()["orphan_sources"] == []
    after_apply = vectors.upserts
    assert service.reconcile()["removed_points"] == 0
    assert vectors.upserts == after_apply


def test_reconcile_rechecks_orphan_before_delete(tmp_path, monkeypatch):
    notes, vectors, service = lifecycle(tmp_path)
    vectors.points["orphan"] = {
        "id": "orphan",
        "payload": {"source": "appeared", "source_type": "obsidian"},
    }
    preview = service.plan_reconciliation()
    assert preview["orphan_sources"] == ["appeared"]
    original = service.plan_reconciliation

    def raced_plan():
        plan = original()
        notes.write("appeared", "now present")
        return plan

    monkeypatch.setattr(service, "plan_reconciliation", raced_plan)
    assert service.reconcile()["removed_points"] == 0
    assert "orphan" in vectors.points


def test_reconcile_prunes_duplicate_owned_point_for_unchanged_note(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "short")
    service.index("guide")
    duplicate = dict(vectors.points["obsidian:guide:0"])
    duplicate["id"] = "duplicate"
    vectors.points["duplicate"] = duplicate

    assert service.plan_reconciliation()["needs_index"] == ["guide"]
    service.reconcile()

    assert set(vectors.points) == {"obsidian:guide:0"}
    assert service.plan_reconciliation()["needs_index"] == []


def test_memory_service_rejects_invalid_batch_size(tmp_path):
    notes, vectors, _service = lifecycle(tmp_path)
    with pytest.raises(ValueError, match="batch"):
        MemoryService(notes, vectors, batch_size=0)


def test_index_uses_batch_upsert_when_available(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "short")
    batches = []

    def upsert_many(batch):
        batches.append(batch)
        for point_id, text, payload in batch:
            vectors.upsert(point_id, text, payload)

    vectors.upsert_many = upsert_many
    assert service.index("guide") == 1
    assert len(batches) == 1


def test_index_removes_prior_points_when_upsert_does_not_materialize(tmp_path):
    notes, vectors, service = lifecycle(tmp_path)
    notes.write("guide", "old")
    service.index("guide")
    notes.write("guide", "new")
    vectors.upsert = lambda *_args: None

    assert service.index("guide") == 1
    assert vectors.points == {}
    assert service.plan_reconciliation()["needs_index"] == ["guide"]


def test_reconcile_skips_orphan_if_source_becomes_unsafe(tmp_path, monkeypatch):
    notes, vectors, service = lifecycle(tmp_path)
    vectors.points["orphan"] = {
        "id": "orphan",
        "payload": {"source": "appeared", "source_type": "obsidian"},
    }
    original = notes.read

    def changing_read(name):
        if name == "appeared":
            raise PermissionError("became unsafe")
        return original(name)

    monkeypatch.setattr(notes, "read", changing_read)
    assert service.reconcile()["removed_points"] == 0
    assert "orphan" in vectors.points


def test_reconcile_preview_tolerates_note_removed_after_listing(tmp_path, monkeypatch):
    notes, _vectors, service = lifecycle(tmp_path)
    monkeypatch.setattr(notes, "list_documents", lambda: ["missing"])
    assert service.plan_reconciliation()["needs_index"] == []


def test_qdrant_index_excludes_generated_and_hidden_vault_documents(tmp_path):
    notes, _vectors, service = lifecycle(tmp_path)
    notes.write("rules/Harness-Prinzipien", "Keep memory local.")
    notes.write("tasks/Task-Register", "Current tasks.")
    notes.write("tasks/42/plan", "Generated historical snapshot.")
    notes.write("_harness/decisions/42", "Generated projection.")
    (notes.vault / ".obsidian").mkdir()
    (notes.vault / ".obsidian/plugins.md").write_text("Private app metadata")

    assert notes.list_documents() == ["rules/Harness-Prinzipien", "tasks/Task-Register"]
    assert service.reconcile()["notes"] == 2


def test_memory_search_returns_current_original_note_with_source(tmp_path):
    notes, _vectors, service = lifecycle(tmp_path)
    notes.write("architecture/Systemarchitektur", "SQLite stays authoritative.")
    service.index("architecture/Systemarchitektur")

    assert service.search("SQLite", limit=3) == [
        {
            "id": "obsidian:architecture/Systemarchitektur:0",
            "payload": {
                "source": "architecture/Systemarchitektur",
                "text": "SQLite stays authoritative.",
            },
        }
    ]


def test_unindex_without_points_and_reconcile_ignores_foreign_payloads(tmp_path):
    _notes, vectors, service = lifecycle(tmp_path)
    assert service.unindex("missing") == 0
    vectors.points["empty"] = {"id": "empty", "payload": None}
    vectors.points["unsafe"] = {
        "id": "unsafe",
        "payload": {"source": "../outside", "source_type": "obsidian"},
    }
    vectors.points["no-id"] = {"payload": {"source": "gone", "source_type": "obsidian"}}

    result = service.reconcile()

    assert result["removed_sources"] == 0
    assert result["removed_points"] == 0
    assert {"empty", "unsafe", "no-id"} <= set(vectors.points)


def test_qdrant_scroll_paginates_with_exact_source_filter():
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if "offset" not in body:
            return httpx.Response(
                200,
                json={"result": {"points": [{"id": "a"}], "next_page_offset": "c1"}},
            )
        return httpx.Response(
            200,
            json={"result": {"points": [{"id": "b"}], "next_page_offset": None}},
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    qdrant = QdrantMemory("http://q", "memory", client=client)

    assert [point["id"] for point in qdrant.scroll_source("nested/guide")] == [
        "a",
        "b",
    ]
    assert calls[0]["filter"] == {
        "must": [{"key": "source", "match": {"value": "nested/guide"}}]
    }
    assert calls[1]["offset"] == "c1"


def test_qdrant_scroll_rejects_invalid_values_and_repeated_cursor():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"result": {"points": [], "next_page_offset": "repeat"}},
            )
        )
    )
    qdrant = QdrantMemory("http://q", "memory", client=client)
    with pytest.raises(ValueError, match="source"):
        qdrant.scroll_source("")
    with pytest.raises(ValueError, match="limit"):
        qdrant.scroll(source="x", limit=0)
    with pytest.raises(ValueError, match="cursor"):
        qdrant.scroll_source("x")


def test_qdrant_scroll_propagates_http_error_and_malformed_response():
    failed = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        QdrantMemory("http://q", "memory", client=failed).scroll_source("note")
    malformed = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": {"points": "bad"}})
        )
    )
    with pytest.raises(TypeError, match="scroll response"):
        QdrantMemory("http://q", "memory", client=malformed).scroll_source("note")
    missing_keys = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": {}})
        )
    )
    with pytest.raises(ValueError, match="scroll response"):
        QdrantMemory("http://q", "memory", client=missing_keys).scroll_source("note")


def test_qdrant_scroll_without_source_and_network_health_errors():
    seen = []

    def handler(request):
        if request.url.path.endswith("/scroll"):
            seen.append(json.loads(request.content))
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    qdrant = QdrantMemory("http://q", "memory", client=client)
    with pytest.raises(httpx.ConnectError):
        qdrant.scroll_source()
    assert "filter" not in seen[0]
    assert qdrant.health() is False
    assert qdrant.collection_exists() is False


def test_memory_service_rejects_unrecognized_point_payload():
    assert not MemoryService._owned_point({"payload": None})
