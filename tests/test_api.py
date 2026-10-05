import asyncio
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from harness import api
from harness.core import Config, Orchestrator, Store, Task


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "api.db")
    config.data["paths"]["obsidian_vault"] = str(tmp_path / "vault")
    config.data["profiles"] = {
        k: {"model": {"primary": "local"}}
        for k in ["planner", "software-architect", "coding", "validator"]
    }
    config.data.setdefault("tools", {})["mcp"] = {
        "servers": {
            "files": {"enabled": False, "builtin": "filesystem"},
            "vault": {"enabled": False, "builtin": "obsidian"},
        }
    }
    store = Store(config)
    orchestrator = Orchestrator(store, config)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "orchestrator", orchestrator)
    monkeypatch.setattr(api, "config", config)
    return TestClient(api.app), store, orchestrator


def test_api_all_routes_and_errors(client):
    http, _, _ = client
    assert http.get("/api/health").status_code == 200
    assert http.get("/api/configuration/status").json() == {
        "valid": True,
        "environment": "development",
        "max_parallel_steps": 4,
    }
    assert http.patch("/api/tasks/999", json={"title": "x"}).status_code == 404
    assert http.patch("/api/tasks/999", json={"unknown": "x"}).status_code == 422
    task = http.post("/api/tasks", json={"title": "API test"}).json()
    task_id = task["id"]
    assert http.patch(f"/api/tasks/{task_id}", json={"unknown": "x"}).status_code == 422
    assert (
        http.patch(f"/api/tasks/{task_id}", json={"complexity": "urgent"}).status_code
        == 422
    )
    updated = http.patch(f"/api/tasks/{task_id}", json={"complexity": "CRITICAL"})
    assert updated.status_code == 200
    assert updated.json()["complexity"] == "CRITICAL"
    assert http.get("/api/tasks/999").status_code == 404
    assert http.get("/api/tasks/999/result").status_code == 404
    assert http.post("/api/tasks/999/start").status_code == 404
    assert http.post("/api/tasks/999/run").status_code == 404
    assert http.post("/api/tasks/999/abort").status_code == 404
    invalid_create = http.post("/api/tasks", json={"id": 1, "title": "bad"})
    assert invalid_create.status_code == 422
    assert http.get("/api/tasks/999/plan").status_code == 404
    assert http.get("/api/tasks/999/validation").status_code == 404
    question = http.post(
        f"/api/tasks/{task_id}/questions",
        json={"question": "choose", "reason": "test", "options": ["yes", "no"]},
    ).json()
    assert http.get(f"/api/tasks/{task_id}/questions").json()[0]["options"] == [
        "yes",
        "no",
    ]
    assert (
        http.post(
            f"/api/tasks/{task_id}/answers",
            json={"question_id": question["id"], "answer": "maybe"},
        ).status_code
        == 404
    )
    assert (
        http.post(
            f"/api/tasks/{task_id}/answers",
            json={"question_id": question["id"], "answer": "yes"},
        ).json()["status"]
        == "waiting_human"
    )
    assert (
        http.get("/api/events", params={"since": "9999-01-01T00:00:00Z"}).json() == []
    )
    assert http.delete("/api/tasks/999").status_code == 404


def test_configuration_status_fails_without_exposing_validation_details(client):
    http, _store, _orchestrator = client
    api.config.data["tools"]["http"]["timeout"] = -1
    response = http.get("/api/configuration/status")
    assert response.status_code == 503
    assert response.json() == {"detail": "configuration is invalid"}


def test_api_exposes_durable_metrics_and_event_catalogue(client):
    http, _store, _orchestrator = client
    created = http.post("/api/tasks", json={"title": "metrics task"}).json()
    report = http.get("/api/metrics").json()
    assert report["tasks"]["total"] == 1
    assert report["tasks"]["by_status"]["pending"] == 1
    catalogue = http.get("/api/event-kinds").json()["event_kinds"]
    assert "task.created" in catalogue
    assert report["events"]["catalogue"] == catalogue
    assert report["events"]["total"] == 0
    prometheus = http.get("/api/metrics/prometheus")
    assert prometheus.status_code == 200
    assert "harness_tasks_total 1" in prometheus.text
    assert "# TYPE harness_process_max_rss_bytes gauge" in prometheus.text
    assert "harness_sqlite_database_bytes " in prometheus.text
    assert "harness_process_uptime_seconds " in prometheus.text
    assert prometheus.headers["content-type"].startswith("text/plain")
    assert created["status"] == "pending"


def test_api_model_inventory_uses_shared_model_operations_service(client):
    from harness.providers import ProviderHealth

    http, _store, orchestrator = client
    orchestrator.models.models = {
        "local-model": {
            "provider": "fixture",
            "model": "loaded-model",
            "tier": "local",
        }
    }
    provider = type(
        "Provider",
        (),
        {
            "health_report": lambda _self: ProviderHealth(
                "openai_compatible", True, True, ("loaded-model",)
            )
        },
    )()
    orchestrator.models.providers = {"fixture": provider}
    response = http.get("/api/models/status")
    assert response.status_code == 200
    assert response.json()["providers"][0]["models"][0]["status"] == "available"


def test_api_model_test_uses_shared_redacted_service_contract(client):
    http, _store, orchestrator = client
    from types import SimpleNamespace

    orchestrator.models.resolve = lambda _name: (
        SimpleNamespace(complete=lambda *_args, **_kwargs: "OK"),
        "fixture-model",
    )
    success = http.post("/api/models/test", json={"model": "local"})
    assert success.status_code == 200
    assert success.json() == {"model": "local", "status": "successful"}

    orchestrator.models.resolve = lambda _name: (_ for _ in ()).throw(
        RuntimeError("secret response")
    )
    failure = http.post("/api/models/test", json={"model": "local"})
    assert failure.status_code == 200
    assert failure.json() == {
        "model": "local",
        "status": "failed",
        "error_type": "RuntimeError",
        "error_category": "execution",
    }

    from harness.providers import ProviderError

    orchestrator.models.resolve = lambda _name: (_ for _ in ()).throw(
        ProviderError("private provider response")
    )
    categorized = http.post("/api/models/test", json={"model": "local"})
    assert categorized.json()["error_category"] == "provider"
    assert "private provider response" not in categorized.text
    assert http.post("/api/models/test", json={"model": ""}).status_code == 422


def test_api_task_knowledge_search_is_read_only_and_returns_provenance(client):
    http, store, orchestrator = client
    task = http.post("/api/tasks", json={"title": "Architecture search"}).json()
    vault = orchestrator.context.memory.vault
    vault.mkdir(parents=True)
    (vault / "Architecture.md").write_text(
        "---\ntype: architecture\nreviewed_on: 2026-10-03\n---\n"
        "# Design\nArchitecture uses SQLite for task state.\n",
        encoding="utf-8",
    )
    store.audit.secrets["fixture"] = "PRIVATE-CANARY"

    response = http.get(
        f"/api/tasks/{task['id']}/knowledge", params={"query": "SQLite"}
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == task["id"]
    hit = response.json()["hits"][0]
    assert hit["source_ref"] == "context:vault/Architecture.md#Design"
    assert hit["provenance"]["review_state"] == "current"
    assert "PRIVATE-CANARY" not in response.text
    assert (vault / "Architecture.md").exists()
    assert (
        http.get("/api/tasks/999999/knowledge", params={"query": "SQLite"}).status_code
        == 404
    )
    assert (
        http.get(
            f"/api/tasks/{task['id']}/knowledge", params={"query": " "}
        ).status_code
        == 422
    )


def test_api_task_and_question_surfaces_do_not_expose_secret_canary(client):
    http, store, _orchestrator = client
    canary = "SS1-API-CANARY-NEVER-RETURN"
    store.audit.secrets["SS1_API_TOKEN"] = canary

    created = http.post(
        "/api/tasks",
        json={"title": f"task {canary}", "description": f"description {canary}"},
    )
    assert created.status_code == 200
    task_id = created.json()["id"]
    assert canary not in created.text
    assert canary not in http.get(f"/api/tasks/{task_id}").text

    patched = http.patch(
        f"/api/tasks/{task_id}", json={"description": f"patched {canary}"}
    )
    assert patched.status_code == 200
    assert canary not in patched.text

    question = http.post(
        f"/api/tasks/{task_id}/questions",
        json={
            "question": f"question {canary}",
            "reason": f"reason {canary}",
            "required": False,
        },
    )
    assert question.status_code == 200
    assert canary not in question.text
    listed = http.get(f"/api/tasks/{task_id}/questions")
    assert canary not in listed.text
    answered = http.post(
        f"/api/tasks/{task_id}/answers",
        json={"question_id": question.json()["id"], "answer": canary},
    )
    assert answered.status_code == 200
    assert canary not in answered.text
    assert canary not in http.get(f"/api/tasks/{task_id}/questions").text

    with store.database.connect() as connection:
        raw = [
            dict(row)
            for table in ("tasks", "questions", "events")
            for row in connection.execute(f"SELECT * FROM {table}")
        ]
    assert canary not in str(raw)


def test_api_readbacks_redact_legacy_unfiltered_rows(client):
    http, store, _orchestrator = client
    canary = "SS1-LEGACY-CANARY"
    store.audit.secrets["SS1_LEGACY_TOKEN"] = canary
    task_id = store.tasks.create(
        f"old {canary}",
        f"description {canary}",
        metadata={"context": {"api_key": canary}},
    )
    store.events.append(task_id, "task.failed", {"message": canary})
    store.questions.create(task_id, canary, canary, [canary])
    store.plans.save(task_id, canary, {"authorization": canary})
    store.validations.record(task_id, False, {"details": canary, "apiKey": canary})

    responses = (
        http.get(f"/api/tasks/{task_id}"),
        http.get(f"/api/tasks/{task_id}/events"),
        http.get("/api/events", params={"task_id": task_id}),
        http.get(f"/api/tasks/{task_id}/questions"),
        http.get(f"/api/tasks/{task_id}/plan"),
        http.get(f"/api/tasks/{task_id}/validation"),
    )
    assert all(response.status_code == 200 for response in responses)
    assert all(canary not in response.text for response in responses)


def test_api_typed_contracts_and_consistent_errors(client):
    http, store, _orchestrator = client
    created = http.post("/api/tasks", json={"title": "typed"})
    assert created.status_code == 200
    task_id = created.json()["id"]
    invalid = http.post("/api/tasks", json={"title": "bad", "unexpected": 1})
    assert invalid.status_code == 422
    assert isinstance(invalid.json()["detail"], str)
    assert http.post("/api/tasks/999/start").status_code == 404
    assert http.post(f"/api/tasks/{task_id}/answers", json={}).status_code == 422
    assert (
        http.post(f"/api/tasks/{task_id}/questions", json={"question": ""}).status_code
        == 422
    )
    assert http.get("/api/tasks", params={"status": "not-a-status"}).status_code == 422
    assert http.get("/api/tasks/999/events").status_code == 404
    assert http.get(f"/api/tasks/{task_id}/plan").json() is None
    assert http.get(f"/api/tasks/{task_id}/validation").json() is None
    with store.database.connect() as connection:
        connection.execute(
            "INSERT INTO plans(task_id,summary,payload,created_at) VALUES(?,?,?,?)",
            (task_id, "typed plan", '{"steps": []}', "2026-09-28T12:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO validations(task_id,valid,report,created_at) VALUES(?,?,?,?)",
            (task_id, 1, '{"coverage": 100}', "2026-09-28T12:00:00+00:00"),
        )
    assert http.get(f"/api/tasks/{task_id}/plan").json()["payload"] == {"steps": []}
    assert http.get(f"/api/tasks/{task_id}/validation").json()["valid"] is True
    store.tasks.update(task_id, status="cancelled")
    terminal_question = http.post(
        f"/api/tasks/{task_id}/questions",
        json={"question": "reopen?", "reason": "check"},
    )
    assert terminal_question.status_code == 422
    store.tasks.update(task_id, status="completed")
    assert http.post(f"/api/tasks/{task_id}/abort").status_code == 409
    schema = http.get("/openapi.json").json()
    assert schema["paths"]["/api/tasks"]["post"]["requestBody"]
    assert schema["paths"]["/api/tasks/{task_id}/answers"]["post"]["requestBody"]
    assert "Task" in schema["components"]["schemas"]
    assert "TaskResult" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]
    assert "QuestionResponse" in schema["components"]["schemas"]
    assert "ValidationResponse" in schema["components"]["schemas"]
    assert "examples" in schema["components"]["schemas"]["Task"]
    assert (
        "text/event-stream"
        in schema["paths"]["/api/events/stream"]["get"]["responses"]["200"]["content"]
    )
    assert http.get("/docs").status_code == 200


def test_artifact_api_versions_conflicts_redaction_and_task_scope(client):
    http, store, _ = client
    canary = "ARTIFACT-API-CANARY"
    store.audit.secrets["ARTIFACT_API_SECRET"] = canary
    task_id = http.post("/api/tasks", json={"title": "artifact API"}).json()["id"]
    created = http.post(
        f"/api/tasks/{task_id}/artifacts",
        json={"key": "report", "content": f"draft {canary}"},
    )
    assert created.status_code == 200
    first = created.json()
    assert first["version"] == 1 and canary not in created.text
    updated = http.post(
        f"/api/tasks/{task_id}/artifacts",
        json={
            "key": "report",
            "content": "final",
            "expected_version": 1,
        },
    )
    assert updated.status_code == 200 and updated.json()["version"] == 2
    assert http.get(f"/api/tasks/{task_id}/artifacts/report").json() == updated.json()
    assert len(http.get(f"/api/tasks/{task_id}/artifacts/report/history").json()) == 2
    assert http.get(f"/api/tasks/{task_id}/artifacts").json()[0]["version"] == 2
    conflict = http.post(
        f"/api/tasks/{task_id}/artifacts",
        json={"key": "report", "content": "stale", "expected_version": 1},
    )
    assert conflict.status_code == 409
    assert (
        http.post(
            f"/api/tasks/{task_id}/artifacts", json={"key": "bad/key", "content": "x"}
        ).status_code
        == 422
    )
    assert (
        http.post(
            "/api/tasks/999/artifacts", json={"key": "x", "content": "x"}
        ).status_code
        == 404
    )
    assert http.get("/api/tasks/999/artifacts").status_code == 404
    assert http.get("/api/tasks/999/artifacts/report").status_code == 404
    assert http.get("/api/tasks/999/artifacts/report/history").status_code == 404
    with store.database.connect() as connection:
        artifact_content = connection.execute(
            "SELECT content FROM artifacts"
        ).fetchall()
        artifact_events = connection.execute(
            "SELECT payload FROM events WHERE kind='artifact.recorded'"
        ).fetchall()
    assert canary not in str(artifact_content)
    assert canary not in str(artifact_events)


def test_decision_query_api_filters_and_reports_missing(client):
    http, store, _ = client
    task_id = http.post("/api/tasks", json={"title": "decision query"}).json()["id"]
    from harness.decisions import DecisionService

    decision = DecisionService(store).record(
        {
            "task_id": task_id,
            "category": "architecture",
            "source": "agent",
            "decision": "Choose local store",
            "rationale": "Keep data local",
            "evidence": [{"source": "task", "ref": f"task:{task_id}"}],
        }
    )
    rows = http.get("/api/decisions", params={"task_id": task_id}).json()
    assert rows[0]["decision"] == "Choose local store"
    assert rows[0]["id"] == decision.id
    assert http.get(f"/api/decisions/{rows[0]['id']}").json() == rows[0]
    assert http.get("/api/decisions", params={"category": "absent"}).json() == []
    assert http.get("/api/decisions/999").status_code == 404
    assert http.get("/api/decisions", params={"task_id": 0}).status_code == 422


def test_verification_evidence_api_import_list_audit_and_fail_closed(client):
    http, _store, _orchestrator = client
    body = {
        "kind": "ci",
        "source_id": "ci:api-run-1",
        "observed_at": datetime.now(UTC).isoformat(),
        "subject_sha256": "a" * 64,
        "passed": True,
        "checks": {"unit_tests": True},
    }
    created = http.post("/api/verification/evidence", json=body)
    assert created.status_code == 200
    assert len(created.json()["digest"]) == 64
    assert http.post("/api/verification/evidence", json=body).status_code == 409
    assert (
        http.get("/api/verification/evidence", params={"kind": "ci"}).json()[0][
            "source_id"
        ]
        == "ci:api-run-1"
    )
    audit = http.get(
        "/api/verification/evidence/audit", params={"subject_sha256": "a" * 64}
    )
    assert audit.status_code == 200
    assert audit.json()["healthy"] is True
    assert (
        http.get(
            "/api/verification/evidence/audit", params={"subject_sha256": "b" * 64}
        ).json()["items"][0]["reason"]
        == "subject_mismatch"
    )
    assert (
        http.post(
            "/api/verification/evidence", json=body | {"raw_log": "secret"}
        ).status_code
        == 422
    )


def test_correction_api_is_task_scoped_and_enforces_lifecycle(client):
    http, store, _ = client
    task_id = http.post("/api/tasks", json={"title": "correction API"}).json()["id"]
    other_id = http.post("/api/tasks", json={"title": "other"}).json()["id"]
    item = store.record_correction(
        task_id,
        {
            "category": "test",
            "source": "validator",
            "rule": "tests.failed",
            "message": "unit test failed",
            "affected_paths": ["src/a.py"],
        },
    )
    base = f"/api/tasks/{task_id}/corrections"
    assert http.get(base).json()[0]["id"] == item["id"]
    assert http.get(base, params={"status": "resolved"}).json() == []
    assert http.get(base, params={"status": "unknown"}).status_code == 422
    item_path = f"{base}/{item['id']}"
    assert (
        http.patch(
            f"/api/tasks/{other_id}/corrections/{item['id']}",
            json={"status": "in_progress"},
        ).status_code
        == 404
    )
    assert (
        http.patch(item_path, json={"status": "in_progress", "extra": 1}).status_code
        == 422
    )
    active = http.patch(item_path, json={"status": "in_progress"})
    assert active.status_code == 200 and active.json()["attempts"] == 1
    resolved = http.patch(item_path, json={"status": "resolved"})
    assert resolved.status_code == 200 and resolved.json()["resolved_at"]
    assert http.patch(item_path, json={"status": "in_progress"}).status_code == 409
    reopened = http.patch(item_path, json={"status": "open"})
    assert reopened.status_code == 200 and reopened.json()["resolved_at"] is None
    assert http.get("/api/tasks/999/corrections").status_code == 404


def test_event_actor_and_sse_cursor_validation(client):
    http, store, _ = client
    store.event(None, "agent.run", {"agent": "planner", "profile": "planner"})
    store.event(None, "agent.run", {"agent": "executor", "profile": "coding"})
    assert len(http.get("/api/events", params={"actor": "planner"}).json()) == 1
    assert len(http.get("/api/events", params={"actor": "coding"}).json()) == 1
    assert http.get("/api/events", params={"actor": "missing"}).json() == []
    assert http.get("/api/events", params={"since": "invalid"}).status_code == 422
    assert (
        http.get(
            "/api/events",
            params={"since": "2026-09-29T00:00:00Z", "until": "2026-09-28T00:00:00Z"},
        ).status_code
        == 422
    )

    invalid_cursor = http.get(
        "/api/events/stream", headers={"Last-Event-ID": "invalid"}
    )
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json() == {
        "detail": "Last-Event-ID must be a non-negative integer"
    }
    assert http.get("/api/events/stream", params={"task_id": 999}).status_code == 404
    assert (
        http.get("/api/events/stream", headers={"Last-Event-ID": "-1"}).status_code
        == 400
    )


def test_delete_running_and_questions_block_resume(client):
    http, store, _ = client
    item = store.create(Task(title="busy"))
    store.tasks.update(item.id, status="executing")
    assert store.tasks.claim(item.id, "running")
    assert http.delete(f"/tasks/{item.id}").status_code == 409
    store.tasks.release(item.id, "running")
    store.tasks.update(item.id, status="pending")
    q1 = store.ask(item.id, "one", "reason", required=True)
    q2 = store.ask(item.id, "two", "reason", required=True)
    assert (
        http.post(
            f"/tasks/{item.id}/answers", json={"question_id": q1, "answer": "yes"}
        ).json()["status"]
        == "waiting_human"
    )
    assert store.questions.has_open_required(item.id)
    assert (
        http.post(
            f"/tasks/{item.id}/answers", json={"question_id": q2, "answer": "yes"}
        ).json()["status"]
        == "waiting_human"
    )


def test_decision_question_purpose_is_exposed_and_persisted(client):
    http, store, _ = client
    item = store.create(Task(title="decision state"))
    response = http.post(
        f"/tasks/{item.id}/questions",
        json={
            "question": "Choose a migration strategy?",
            "reason": "Compatibility choice",
            "options": ["preserve", "reset"],
            "purpose": "decision",
        },
    )

    assert response.status_code == 200
    listed = http.get(f"/tasks/{item.id}/questions").json()
    assert listed[0]["purpose"] == "decision"
    assert store.get(item.id).status == "waiting_decision"
    assert (
        http.post(
            f"/tasks/{item.id}/questions",
            json={
                "question": "Approve?",
                "reason": "bad purpose",
                "purpose": "approval",
            },
        ).status_code
        == 422
    )


def test_sse_reads_persisted_events(client):
    _, store, _ = client
    store.event(None, "task.completed", {"value": 1})

    class Request:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    async def run_stream():
        response = await api.stream(Request())
        iterator = response.body_iterator
        message = await anext(iterator)
        assert "event: task.completed" in message
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

    asyncio.run(run_stream())


def test_sse_reconnect_cursor_and_independent_readers(client):
    _, store, _ = client
    store.event(None, "task.started", {"value": 1})
    store.event(None, "task.completed", {"value": 2})
    rows = store.events.list()

    class Request:
        def __init__(self, cursor):
            self.headers = {"last-event-id": str(cursor)}
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    async def read(cursor, expected_id, expected_event):
        response = await api.stream(Request(cursor))
        iterator = response.body_iterator
        message = await anext(iterator)
        assert f"id: {expected_id}" in message
        assert f"event: {expected_event}" in message
        if expected_event == "task.started":
            message = await anext(iterator)
            assert f"id: {rows[1]['id']}" in message
            assert "event: task.completed" in message
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

    async def run_readers():
        await asyncio.gather(
            read(rows[0]["id"], rows[1]["id"], "task.completed"),
            read(0, rows[0]["id"], "task.started"),
        )

    asyncio.run(run_readers())


def test_sse_keepalive_and_nonblocking_answer(client):
    _, store, _ = client
    item = store.create(Task(title="nonblocking"))
    qid = store.ask(item.id, "FYI", "nonblocking", required=False)
    response = api.answer(item.id, api.AnswerRequest(question_id=qid, answer="noted"))
    result = asyncio.run(response)
    assert result.status.value == "pending"

    class Request:
        def __init__(self):
            self.headers = {"last-event-id": "999999"}
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    async def read_keepalive():
        stream = await api.stream(Request())
        iterator = stream.body_iterator
        assert ": keep-alive" in await anext(iterator)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

    asyncio.run(read_keepalive())
