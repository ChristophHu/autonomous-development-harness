import pytest
from test_evidence_workflow import ready_runtime


def test_task_service_shares_status_queries_and_sanitized_task_inspection(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    service = orchestrator.service

    counts = service.status_counts()
    assert counts["pending"] == 1
    assert counts["completed"] == 0
    store.event(created.id, "task.created", {"api_key": "must-not-leak"})
    events = service.events(created.id)
    assert "[REDACTED]" in events[0]["payload"]
    assert service.event_feed(created.id)[0]["task_id"] == created.id
    assert service.events_after(0, created.id)[0]["kind"] == "task.created"
    assert service.questions(created.id) == []
    assert service.plan(created.id) is None
    assert service.validation(created.id) is None
    assert service.model_usage()["total"] == 0


def test_task_service_routes_question_creation_and_unscoped_event_cursor(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    question_id = orchestrator.service.ask_question(
        created.id, "Review?", "Need owner decision", ["yes", "no"]
    )
    assert store.questions.get(question_id)["task_id"] == created.id
    store.event(created.id, "task.created", {})
    assert orchestrator.service.events_after(0) == store.events.after(0)


@pytest.mark.parametrize(
    "method,args",
    [
        ("events", ()),
        ("questions", ()),
        ("plan", ()),
        ("validation", ()),
        ("ask_question", ("Q", "R")),
    ],
)
def test_task_service_inspection_and_question_methods_require_existing_task(
    tmp_path, method, args
):
    _store, orchestrator, _task = ready_runtime(tmp_path)
    with pytest.raises(ValueError, match="task not found"):
        getattr(orchestrator.service, method)(987654, *args)
