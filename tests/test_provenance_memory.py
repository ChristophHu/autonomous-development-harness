import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from test_evidence_workflow import ready_runtime, runtime

from harness.audit import CURRENT_RUN
from harness.domain import Task
from harness.memory import (
    ObsidianMemory,
    QdrantMemory,
    validate_vector,
)
from harness.memory_service import MemoryService
from harness.providers import ModelUsage, OpenAICompatibleProvider


def test_success_failure_fallback_and_provider_usage_are_correlated(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    task = store.create(task)
    config = orchestrator.config
    config.data["profiles"]["coding"]["model"] = {
        "primary": "bad",
        "fallback": ["good"],
    }
    config.data["models"] = {"rates": {"m": {"input": 1, "output": 2}}}
    orchestrator.models.register(
        "bad",
        SimpleNamespace(
            complete=lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("PRIVATE-PROMPT")
            )
        ),
    )
    usage = ModelUsage("good", "m", 10, 20, cached_tokens=3, reasoning_tokens=4)
    orchestrator.models.register(
        "good", SimpleNamespace(complete=lambda *a, **kw: ("PRIVATE-PROMPT", usage))
    )
    assert (
        orchestrator._invoke(
            task,
            "executor",
            "coding",
            orchestrator.router.complete,
            "coding",
            "FULL-SECRET-PROMPT",
        )
        == "PRIVATE-PROMPT"
    )
    assert CURRENT_RUN.get() is None
    with store.database.connect() as connection:
        rows = connection.execute("SELECT * FROM model_runs ORDER BY id").fetchall()
        agent = connection.execute("SELECT * FROM agent_runs").fetchone()
    assert [row["status"] for row in rows] == ["failed", "completed"]
    assert [row["fallback_index"] for row in rows] == [0, 1]
    assert rows[1]["task_id"] == task.id and rows[1]["agent_run_id"] == agent["id"]
    assert rows[1]["agent"] == "executor" and rows[1]["profile"] == "coding"
    assert rows[1]["cached_tokens"] == 3 and rows[1]["reasoning_tokens"] == 4
    assert rows[1]["cost"] == 0.00005 and rows[1]["latency_ms"] >= 0
    assert rows[1]["started_at"] <= rows[1]["finished_at"]
    assert "PRIVATE-PROMPT" not in str([dict(row) for row in rows])
    assert "FULL-SECRET-PROMPT" not in agent["output"]


def test_actual_provider_details_and_missing_usage():
    payload = {
        "choices": [{"message": {"content": "done"}}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    provider = OpenAICompatibleProvider(
        "fixture",
        "http://fixture",
        model="m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )
    _, usage = provider.complete("prompt")
    assert usage.cached_tokens == 2 and usage.reasoning_tokens == 3
    payload.pop("usage")
    _, usage = provider.complete("prompt")
    assert usage.prompt_tokens is None and usage.cached_tokens is None
    payload["usage"] = None
    assert provider.complete("prompt")[1].completion_tokens is None


def test_tool_failure_redaction_and_atomic_events(tmp_path):
    store, orchestrator = runtime(tmp_path)
    store.audit.secrets.update(
        {"API_KEY": "SENSITIVE-SECRET", "LMSTUDIO_MODEL": "model-name", "EMPTY": ""}
    )
    task = store.create(Task(title="audit"))

    def operation():
        orchestrator.tools.execute(
            "filesystem.write", {"path": "x", "content": "SENSITIVE-SECRET"}
        )
        with pytest.raises(FileNotFoundError):
            orchestrator.tools.execute("filesystem.read", {"path": "missing"})

    orchestrator._invoke(task, "executor", "coding", operation)
    with store.database.connect() as connection:
        tools = [
            dict(row)
            for row in connection.execute("SELECT * FROM tool_calls ORDER BY id")
        ]
    assert [row["status"] for row in tools] == ["completed", "failed"]
    assert "SENSITIVE-SECRET" not in str(tools)
    assert all(row["finished_at"] and row["agent_run_id"] for row in tools)
    assert len(store.events.list(task.id, "TOOL_CALL_FAILED")) == 1
    with pytest.raises(ValueError, match="active invocation"):
        orchestrator.audit.tool_event(
            "TOOL_CALL_COMPLETED", {"call_id": "unknown", "output": "none"}
        )
    assert store.audit.sanitize(
        {
            "Authorization": "unknown credential",
            "nested": ("SENSITIVE-SECRET", 3),
            "name": "model-name",
        }
    ) == {
        "Authorization": "[REDACTED]",
        "nested": ["[REDACTED]", 3],
        "name": "model-name",
    }
    store.event(
        task.id, "task.completed", {"token": "unknown", "body": "SENSITIVE-SECRET"}
    )
    assert "SENSITIVE-SECRET" not in store.events.list(task.id)[-1]["payload"]


def test_task_question_and_answer_persistence_redacts_secret_canary(tmp_path):
    store, _orchestrator = runtime(tmp_path)
    canary = "SS1-CANARY-DO-NOT-PERSIST"
    store.audit.secrets["SS1_TEST_TOKEN"] = canary
    task = store.create(
        Task(
            title=f"title {canary}",
            description=f"desc {canary}",
            context={"apiKey": canary, "ordinary": canary},
        )
    )
    assert canary not in task.title + task.description + str(task.context)

    task.title = f"updated {canary}"
    task.description = f"updated {canary}"
    task.result = f"failed {canary}"
    store.update(task)
    question_id = store.ask(
        task.id,
        f"question {canary}",
        f"reason {canary}",
        [f"option {canary}"],
    )
    store.answer(question_id, f"answer {canary}", task.id)
    store.event(task.id, "task.failed", {"detail": f"failure {canary}"})

    with store.database.connect() as connection:
        rows = [
            dict(row)
            for table in ("tasks", "questions", "events")
            for row in connection.execute(f"SELECT * FROM {table}")
        ]
    assert canary not in str(rows)
    assert "[REDACTED]" in str(rows)


def test_task_failure_result_and_event_redact_secret_canary(tmp_path, monkeypatch):
    store, orchestrator = runtime(tmp_path)
    canary = "SS1-FAILURE-CANARY"
    store.audit.secrets["SS1_TEST_TOKEN"] = canary
    task = store.create(Task(title="failure test"))

    def fail_before_planning(*_args, **_kwargs):
        raise RuntimeError(f"provider failed with {canary}")

    monkeypatch.setattr(orchestrator, "_invoke", fail_before_planning)
    with pytest.raises(RuntimeError, match="provider failed") as raised:
        asyncio.run(orchestrator.run(task.id))

    with store.database.connect() as connection:
        row = connection.execute(
            "SELECT result FROM tasks WHERE id=?", (task.id,)
        ).fetchone()
    events = store.events.list(task.id, "task.failed")
    assert canary not in str(row) + str(events)
    assert "[REDACTED]" in row["result"]
    assert canary not in str(raised.value)


def test_sensitive_field_names_are_redacted_recursively(tmp_path):
    store, _orchestrator = runtime(tmp_path)
    safe = store.audit.sanitize(
        {
            "apiKey": "unknown-key",
            "nested": [
                {"access_token": "unknown-token"},
                {"client-secret": "unknown-secret"},
                {"Authorization": "Bearer unknown-auth"},
                {"ordinary": "visible"},
            ],
        }
    )
    assert safe == {
        "apiKey": "[REDACTED]",
        "nested": [
            {"access_token": "[REDACTED]"},
            {"client-secret": "[REDACTED]"},
            {"Authorization": "[REDACTED]"},
            {"ordinary": "visible"},
        ],
    }


def test_plan_validation_subtask_and_correction_storage_redacts_canary(tmp_path):
    from harness.agents import Subtask

    store, _orchestrator = runtime(tmp_path)
    canary = "SS1-ARTIFACT-CANARY"
    store.audit.secrets["SS1_ARTIFACT_TOKEN"] = canary
    task = store.create(Task(title="artifact persistence"))
    plan_id = store.save_plan(
        task.id,
        f"summary {canary}",
        {"description": canary, "apiKey": "unknown-sensitive-value"},
    )
    store.save_subtasks(
        task.id,
        [Subtask(id="step", title=canary, description=f"desc {canary}")],
        plan_id,
    )
    store.update_subtask(
        task.id,
        "step",
        "completed",
        {"output": canary, "client_secret": "unknown-sensitive-value"},
        plan_id,
    )
    store.record_validation(
        task.id, False, {"errors": [canary], "access_token": "unknown-sensitive-value"}
    )
    store.record_correction(
        task.id,
        {
            "category": "validation",
            "source": "validator",
            "rule": "safe.output",
            "message": canary,
            "evidence": {"credential": "unknown-sensitive-value"},
        },
        plan_id,
    )

    with store.database.connect() as connection:
        rows = [
            dict(row)
            for table in ("plans", "subtasks", "validations", "correction_items")
            for row in connection.execute(f"SELECT * FROM {table}")
        ]
    assert canary not in str(rows)
    assert "unknown-sensitive-value" not in str(rows)


def test_context_is_isolated_between_parallel_tasks(tmp_path):
    store, orchestrator = runtime(tmp_path)
    first, second = (
        store.create(Task(title="first")),
        store.create(Task(title="second")),
    )

    def write(task):
        orchestrator.tools.execute(
            "filesystem.write", {"path": str(task.id), "content": task.title}
        )
        return CURRENT_RUN.get()["task_id"]

    async def parallel():
        return await asyncio.gather(
            asyncio.to_thread(
                orchestrator._invoke, first, "executor", "coding", write, first
            ),
            asyncio.to_thread(
                orchestrator._invoke, second, "executor", "coding", write, second
            ),
        )

    assert asyncio.run(parallel()) == [first.id, second.id]
    assert CURRENT_RUN.get() is None
    for task in (first, second):
        payload = json.loads(
            store.events.list(task.id, "TOOL_CALL_COMPLETED")[0]["payload"]
        )
        assert payload["task_id"] == task.id and payload["profile"] == "coding"


def test_vault_symlinks_absolute_paths_and_nested_append(tmp_path):
    notes = ObsidianMemory(tmp_path / "vault")
    outside = tmp_path / "outside.md"
    outside.write_text("PRIVATE")
    (notes.vault / "linked.md").symlink_to(outside)
    for name in ("", str(tmp_path / "absolute"), "linked", "../outside"):
        with pytest.raises(PermissionError):
            notes.read(name)
    assert notes.search("PRIVATE") == []
    notes.append("nested/new", "new content")
    assert notes.read("nested/new") == "new content"


@pytest.mark.parametrize(
    "vector", [None, [], [True], [float("nan")], [float("inf")], ["bad"]]
)
def test_invalid_embedding_vectors_are_rejected(vector):
    with pytest.raises(ValueError, match="embedding"):
        validate_vector(vector)


def test_collection_contract_creation_and_wrong_dimensions():
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json={"result": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    vectors = QdrantMemory(
        "http://q", "notes", 2, SimpleNamespace(embed=lambda text: [0.1, 0.2]), client
    )
    assert vectors.ensure_collection()
    assert json.loads(calls[1].content)["vectors"]["size"] == 2
    assert vectors.upsert(str(uuid.uuid4()), "text")
    assert vectors.delete([])
    vectors.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "result": {
                        "config": {
                            "params": {"vectors": {"size": 3, "distance": "Cosine"}}
                        }
                    }
                },
            )
        )
    )
    with pytest.raises(ValueError, match="contract"):
        vectors.ensure_collection()
    with pytest.raises(ValueError, match="dimension"):
        QdrantMemory(
            "http://q", "c", 3, SimpleNamespace(embed=lambda t: [0.1, 0.2])
        )._vector("x")
    for collection, dimension in (("../escape", 2), ("valid", 0)):
        with pytest.raises(ValueError):
            QdrantMemory("http://q", collection, dimension)


def test_source_truth_chunk_index_and_stale_retrieval(tmp_path):
    notes = ObsidianMemory(tmp_path / "vault")
    points = []

    def upsert(point_id, text, payload):
        point = {"id": point_id, "payload": payload, "text": text}
        for index, existing in enumerate(points):
            if existing.get("id") == point_id:
                points[index] = point
                return
        points.append(point)

    def scroll_source(source=None):
        return [
            point
            for point in points
            if source is None or point.get("payload", {}).get("source") == source
        ]

    def delete(point_ids):
        points[:] = [point for point in points if point.get("id") not in point_ids]
        return True

    vectors = SimpleNamespace(
        ensure_collection=lambda: True,
        upsert=upsert,
        scroll_source=scroll_source,
        delete=delete,
        search=lambda *a, **kw: points,
    )
    service = MemoryService(notes, vectors, chunk_size=6, overlap=2)
    notes.write("knowledge", "abcdefghijk")
    assert service.index("knowledge") == 3
    assert [p["payload"]["text"] for p in points] == ["abcdef", "efghij", "ijk"]
    points[0]["payload"]["text"] = "stale cached vector text"
    assert service.search("query")[0]["payload"]["text"] == "abcdefghijk"
    notes.write("knowledge", "changed original")
    assert service.search("query") == []
    points.extend(
        [{"payload": {}}, {"payload": {"source": "missing", "source_hash": "old"}}]
    )
    assert service.search("query") == []
    with pytest.raises(FileNotFoundError):
        service.index("missing")
    notes.write("empty", "")
    assert service.index("empty") == 0
    for chunk, overlap in ((0, 0), (2, 2), (2, -1)):
        with pytest.raises(ValueError):
            MemoryService(notes, vectors, chunk, overlap)


@pytest.mark.parametrize("status", [200, 503])
def test_http_tool_response_status_is_actually_checked(tmp_path, monkeypatch, status):
    store, orchestrator = runtime(tmp_path)
    orchestrator.tools.permissions.rules["http"] = "write"
    orchestrator.tools.executor.http_allow_hosts.add("fixture")
    response = httpx.Response(status, request=httpx.Request("GET", "http://fixture"))
    monkeypatch.setattr(httpx, "request", lambda *a, **kw: response)
    if status == 200:
        assert (
            orchestrator.tools.execute(
                "http.request", {"method": "GET", "url": "http://fixture"}
            ).status_code
            == 200
        )
    else:
        with pytest.raises(httpx.HTTPStatusError):
            orchestrator.tools.execute(
                "http.request", {"method": "GET", "url": "http://fixture"}
            )
    with store.database.connect() as connection:
        row = connection.execute("SELECT status FROM tool_calls").fetchone()
    assert row["status"] == ("completed" if status == 200 else "failed")
