import asyncio

import pytest
from fastapi.testclient import TestClient

from harness import api
from harness.core import Config, Orchestrator, Store, Task


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "api.db")
    config.data["profiles"] = {
        k: {"model": {"primary": "local"}}
        for k in ["planner", "software-architect", "coding", "validator"]
    }
    store = Store(config)
    orchestrator = Orchestrator(store, config)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "orchestrator", orchestrator)
    return TestClient(api.app), store, orchestrator


def test_api_all_routes_and_errors(client):
    http, _, _ = client
    assert http.get("/api/health").status_code == 200
    assert http.patch("/api/tasks/999", json={"title": "x"}).status_code == 404
    assert http.patch("/api/tasks/999", json={"unknown": "x"}).status_code == 422
    task = http.post("/api/tasks", json={"title": "API test"}).json()
    task_id = task["id"]
    assert http.patch(f"/api/tasks/{task_id}", json={"unknown": "x"}).status_code == 422
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
