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
    assert http.patch("/api/tasks/999", json={"unknown": "x"}).status_code == 404
    task = http.post("/api/tasks", json={"title": "API test"}).json()
    task_id = task["id"]
    assert http.patch(f"/api/tasks/{task_id}", json={"unknown": "x"}).status_code == 422
    assert http.get("/api/tasks/999").status_code == 404
    assert http.get("/api/tasks/999/result").status_code == 404
    with pytest.raises(ValueError):
        http.post("/api/tasks/999/start")
    assert http.get("/api/tasks/999/plan").json() is None
    assert http.get("/api/tasks/999/validation").json() is None
    question = http.post(
        f"/api/tasks/{task_id}/questions",
        json={"question": "choose", "reason": "test", "options": ["yes", "no"]},
    ).json()
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
    assert http.get("/api/events", params={"since": "9999"}).json() == []
    assert http.delete("/api/tasks/999").status_code == 404


def test_delete_running_and_questions_block_resume(client):
    http, store, _ = client
    item = store.create(Task(title="busy"))
    store.tasks.update(item.id, status="executing")
    assert store.tasks.claim(item.id, "running")
    assert http.delete(f"/tasks/{item.id}").status_code == 404
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
    store.event(None, "persisted", {"value": 1})

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
        assert "event: persisted" in message
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

    asyncio.run(run_stream())


def test_sse_keepalive_and_nonblocking_answer(client):
    _, store, _ = client
    item = store.create(Task(title="nonblocking"))
    qid = store.ask(item.id, "FYI", "nonblocking", required=False)
    response = api.answer(item.id, {"question_id": qid, "answer": "noted"})
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
