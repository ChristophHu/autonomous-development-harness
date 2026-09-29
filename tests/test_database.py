import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness.database import (
    AgentRunRepository,
    CorrectionRepository,
    Database,
    DecisionRepository,
    EventRepository,
    ModelRunRepository,
    PlanRepository,
    QuestionRepository,
    SubtaskRepository,
    TaskRepository,
    ValidationRepository,
)
from harness.domain import EventKind
from harness.providers import ModelUsage


def test_legacy_migration(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE tasks(id INTEGER PRIMARY KEY,title TEXT NOT NULL,description TEXT NOT NULL,status TEXT NOT NULL,result TEXT)"
    )
    connection.execute(
        "INSERT INTO tasks VALUES(7,'preserved','legacy description','pending','old result')"
    )
    connection.execute(
        "CREATE TABLE decisions(id INTEGER PRIMARY KEY,task_id INTEGER,decision TEXT,rationale TEXT,created_at TEXT)"
    )
    connection.execute(
        "INSERT INTO decisions VALUES(3,7,'old decision','old rationale','old time')"
    )
    connection.commit()
    connection.close()
    db = Database(path)
    repo = TaskRepository(db)
    task_id = repo.create("legacy")
    assert repo.get(task_id)["created_at"] and db.now()
    assert repo.get(7)["title"] == "preserved" and repo.get(7)["result"] == "old result"
    db = Database(path)
    with db.connect() as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            )
        ] == [1, 2, 3, 4, 5]
    legacy_decision = DecisionRepository(db).get(3)
    assert legacy_decision["decision"] == "old decision"
    assert legacy_decision["category"] == legacy_decision["source"] == "legacy"
    assert legacy_decision["evidence"] == legacy_decision["field_names"] == []


def test_database_rollback_and_repository_lifecycle(tmp_path):
    db = Database(tmp_path / "state.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("state")
    with pytest.raises(sqlite3.IntegrityError), db.connect() as connection:
        connection.execute("INSERT INTO tasks(title) VALUES(NULL)")
    assert tasks.get(task_id) is not None
    tasks.update(task_id, title="updated")
    assert tasks.list("pending")[0]["title"] == "updated"
    assert tasks.list("missing") == []
    events = EventRepository(db)
    eid = events.append(task_id, EventKind.TASK_CREATED, {"actor": "test"})
    assert events.after(0, task_id)[0]["id"] == eid
    assert events.list(task_id, event_type=EventKind.TASK_CREATED)
    assert events.list(since="9999") == []
    assert PlanRepository(db).save(task_id, "summary", {})
    assert DecisionRepository(db).save(task_id, "choice", "reason")
    questions = QuestionRepository(db)
    qid = questions.create(task_id, "approve", "because")
    assert questions.has_open_required(task_id) and questions.answer(
        qid, "yes", task_id
    )
    assert not questions.has_open_required(task_id)
    assert questions.consume_answer(qid)
    assert not questions.consume_answer(qid)
    agent_runs = AgentRunRepository(db)
    run_id = agent_runs.start(task_id, "planner", "planner")
    assert agent_runs.finish(run_id, "completed", "done") and not agent_runs.finish(
        999, "failed", ""
    )
    usage = ModelUsage("local", "m", 2, 3, 0.1)
    assert ModelRunRepository(db).record(usage, run_id)


def test_event_catalog_rejects_unknown_writes_and_keeps_legacy_reads(tmp_path):
    db = Database(tmp_path / "events.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("event compatibility")
    events = EventRepository(db)

    with pytest.raises(ValueError, match="'not.registered' is not a valid EventKind"):
        events.append(task_id, "not.registered", {})
    assert events.list(task_id) == []

    with db.connect() as connection:
        connection.execute(
            "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
            (task_id, "historic.custom", "{}", db.now()),
        )
    assert events.list(task_id)[0]["kind"] == "historic.custom"


def test_event_kind_catalog_values_are_unique_and_string_backed():
    values = [kind.value for kind in EventKind]
    assert len(values) >= 25
    assert len(values) == len(set(values))
    assert EventKind.TASK_COMPLETED == "task.completed"


def test_status_transition_writes_catalogued_event_and_tool_events_reject_wrong_kind(
    tmp_path,
):
    db = Database(tmp_path / "typed-events.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("transition")
    tasks.transition(task_id, "analyzing")
    events = EventRepository(db)
    assert events.list(task_id)[0]["kind"] == EventKind.TASK_STATUS.value

    from harness.database import AuditRepository

    audit = AuditRepository(db)
    with pytest.raises(ValueError, match="unsupported tool event kind"):
        audit.tool_event(task_id, EventKind.TASK_COMPLETED, {}, {})


def test_subtask_checkpoints_survive_reopen_and_update(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "checkpoints.sqlite"
    db = Database(path)
    task_id = TaskRepository(db).create("checkpoint")
    repo = SubtaskRepository(db)
    repo.save_plan(
        task_id,
        [
            SimpleNamespace(
                id="step-1", title="Inspect", description="read", profile="planner"
            )
        ],
    )
    assert repo.list(task_id)[0]["status"] == "pending"
    assert repo.update(task_id, "step-1", "running")
    assert SubtaskRepository(Database(path)).list(task_id)[0]["status"] == "running"
    assert repo.update(task_id, "step-1", "completed", '{"success":true}')
    assert repo.list(task_id)[0]["output"] == '{"success":true}'
    assert not repo.update(task_id, "missing", "failed")


def test_validation_repository_round_trips_latest_and_empty(tmp_path):
    db = Database(tmp_path / "validation.sqlite")
    task_id = TaskRepository(db).create("validation")
    repo = ValidationRepository(db)
    assert repo.latest(task_id) is None
    first_id = repo.record(task_id, True, {"coverage": 100})
    second_id = repo.record(task_id, False, '{"reason":"tests failed"}')
    latest = repo.latest(task_id)
    assert latest == {
        "id": second_id,
        "task_id": task_id,
        "valid": False,
        "report": {"reason": "tests failed"},
        "created_at": latest["created_at"],
    }
    assert first_id < second_id


def _finding(**overrides):
    return {
        "category": "workspace",
        "source": "validator",
        "rule": "undeclared-change",
        "message": "A changed path was not reported.",
        "subtask_id": "implement",
        "affected_paths": ["src/a.py"],
        "evidence": {"observed": "src/a.py"},
        "expected": {"scope": "declared"},
    } | overrides


def test_correction_repository_persists_and_deduplicates_findings(tmp_path):
    db = Database(tmp_path / "corrections.sqlite")
    task_id = TaskRepository(db).create("correction")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    repo = CorrectionRepository(db)

    first = repo.record(task_id, _finding(), plan_id=plan_id)
    repeated = repo.record(task_id, _finding(), plan_id=plan_id)

    assert first == repeated
    assert first["status"] == "open"
    assert first["attempts"] == 0
    assert first["affected_paths"] == ["src/a.py"]
    assert first["evidence"] == {"observed": "src/a.py"}
    assert repo.get(first["id"]) == first
    assert repo.list_for_task(task_id, "open") == [first]
    assert repo.list_for_task(task_id) == [first]
    assert repo.list_for_task(task_id, "resolved") == []
    events = EventRepository(db).list(task_id)
    assert [event["kind"] for event in events] == [
        EventKind.CORRECTION_ITEM_RECORDED.value
    ]


def test_correction_finding_identity_is_stable_but_task_scoped(tmp_path):
    db = Database(tmp_path / "correction-identity.sqlite")
    tasks = TaskRepository(db)
    first_task = tasks.create("first")
    second_task = tasks.create("second")
    repo = CorrectionRepository(db)

    first = repo.record(first_task, _finding())
    updated = repo.record(
        first_task,
        _finding(
            message="Still not fixed.",
            affected_paths=["src/a.py"],
            evidence={"observed": "new evidence"},
        ),
    )
    other_task = repo.record(second_task, _finding())

    assert first["id"] == updated["id"]
    assert updated["message"] == "Still not fixed."
    assert updated["evidence"] == {"observed": "new evidence"}
    assert other_task["id"] != first["id"]
    assert len(EventRepository(db).list(first_task)) == 2

    reordered = repo.record(
        first_task,
        _finding(affected_paths=["src/b.py", "src/a.py"]),
    )
    same_paths_different_order = repo.record(
        first_task,
        _finding(affected_paths=["src/a.py", "src/b.py"]),
    )
    assert reordered["id"] == same_paths_different_order["id"]
    assert same_paths_different_order["affected_paths"] == ["src/a.py", "src/b.py"]


def test_correction_repository_applies_optional_field_defaults(tmp_path):
    db = Database(tmp_path / "correction-defaults.sqlite")
    task_id = TaskRepository(db).create("defaults")
    finding = _finding(subtask_id=None)
    del finding["affected_paths"]
    del finding["evidence"]
    del finding["expected"]

    item = CorrectionRepository(db).record(task_id, finding)

    assert item["subtask_id"] is None
    assert item["affected_paths"] == []
    assert item["evidence"] == {}
    assert item["expected"] == {}


def test_correction_repository_validates_inputs_and_plan_ownership(tmp_path):
    db = Database(tmp_path / "correction-validation.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("first")
    other_task = tasks.create("other")
    plan_id = PlanRepository(db).save(other_task, "other plan", {})
    repo = CorrectionRepository(db)

    with pytest.raises(ValueError, match="finding fields"):
        repo.record(task_id, None)
    with pytest.raises(ValueError, match="finding fields"):
        repo.record(task_id, {"message": "incomplete"})
    with pytest.raises(ValueError, match="finding fields"):
        repo.record(task_id, _finding(unexpected=True))
    with pytest.raises(ValueError, match="affected_paths"):
        repo.record(task_id, _finding(affected_paths=["src/a.py", "src/a.py"]))
    with pytest.raises(TypeError, match="evidence"):
        repo.record(task_id, _finding(evidence=["not a mapping"]))
    with pytest.raises(ValueError, match="plan does not belong"):
        repo.record(task_id, _finding(), plan_id=plan_id)
    with pytest.raises(ValueError, match="plan_id"):
        repo.record(task_id, _finding(), plan_id=True)
    with pytest.raises(ValueError, match="task_id"):
        repo.record(True, _finding())
    with pytest.raises(ValueError, match="task_id"):
        repo.record(0, _finding())
    with pytest.raises(ValueError, match="affected_paths"):
        repo.record(task_id, _finding(affected_paths=["src/a.py", " src/a.py "]))
    with pytest.raises(ValueError, match="affected_paths"):
        repo.record(task_id, _finding(affected_paths="src/a.py"))
    with pytest.raises(ValueError, match="affected_paths"):
        repo.record(task_id, _finding(affected_paths=[None]))
    with pytest.raises(ValueError, match="JSON serializable"):
        repo.record(task_id, _finding(expected={"not_finite": float("nan")}))
    with pytest.raises(ValueError, match="JSON serializable"):
        repo.record(task_id, _finding(evidence={"not_serializable": object()}))
    with pytest.raises(TypeError, match="expected must be a mapping"):
        repo.record(task_id, _finding(expected=[]))
    for key, value in (
        ("category", 3),
        ("source", " "),
        ("rule", None),
        ("message", []),
        ("subtask_id", 2),
        ("subtask_id", " "),
    ):
        with pytest.raises(ValueError):
            repo.record(task_id, _finding(**{key: value}))
    with pytest.raises(ValueError, match="plan does not belong"):
        repo.record(task_id, _finding(), plan_id=9999)
    with pytest.raises(ValueError, match="task not found"):
        repo.record(9999, _finding())
    with pytest.raises(ValueError, match="unsupported correction status"):
        repo.list_for_task(task_id, [])


def test_correction_repository_status_transitions_are_audited(tmp_path):
    db = Database(tmp_path / "correction-status.sqlite")
    task_id = TaskRepository(db).create("status")
    repo = CorrectionRepository(db)
    finding = repo.record(task_id, _finding())

    started = repo.set_status(finding["id"], "in_progress")
    resolved = repo.set_status(finding["id"], "resolved")
    repeated = repo.set_status(finding["id"], "resolved")

    assert started["status"] == "in_progress"
    assert started["attempts"] == 1
    assert resolved["resolved_at"] is not None
    assert repeated == resolved
    assert repo.list_for_task(task_id, "resolved") == [resolved]
    assert [event["kind"] for event in EventRepository(db).list(task_id)] == [
        EventKind.CORRECTION_ITEM_RECORDED.value,
        EventKind.CORRECTION_ITEM_STATUS.value,
        EventKind.CORRECTION_ITEM_STATUS.value,
    ]
    assert repo.set_status("missing", "open") is None
    with pytest.raises(ValueError, match="unsupported correction status"):
        repo.set_status(finding["id"], "invalid")
    with pytest.raises(ValueError, match="invalid correction status transition"):
        repo.set_status(finding["id"], "in_progress")
    reopened = repo.set_status(finding["id"], "open")
    assert reopened["status"] == "open"


def test_task_completion_is_atomically_blocked_by_open_correction(tmp_path):
    db = Database(tmp_path / "correction-completion.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("completion guard")
    with db.connect() as connection:
        connection.execute(
            "UPDATE tasks SET status='validating' WHERE id=?", (task_id,)
        )
    CorrectionRepository(db).record(task_id, _finding())

    with pytest.raises(ValueError, match="required correction"):
        tasks.transition(task_id, "completed")

    assert tasks.get(task_id)["status"] == "validating"


def test_task_completion_succeeds_after_correction_is_verified(tmp_path):
    db = Database(tmp_path / "correction-resolved-completion.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("resolved completion")
    with db.connect() as connection:
        connection.execute(
            "UPDATE tasks SET status='validating' WHERE id=?", (task_id,)
        )
    repo = CorrectionRepository(db)
    item = repo.record(task_id, _finding())
    repo.set_status(item["id"], "resolved")

    tasks.transition(task_id, "completed")

    assert tasks.get(task_id)["status"] == "completed"


def test_completed_task_rejects_late_correction_findings(tmp_path):
    db = Database(tmp_path / "completed-correction.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("completed task")
    with db.connect() as connection:
        connection.execute("UPDATE tasks SET status='completed' WHERE id=?", (task_id,))

    with pytest.raises(ValueError, match="completed task"):
        CorrectionRepository(db).record(task_id, _finding())

    assert CorrectionRepository(db).list_for_task(task_id) == []


def test_correction_repository_reopens_recurring_resolved_finding(tmp_path):
    db = Database(tmp_path / "correction-reopen.sqlite")
    task_id = TaskRepository(db).create("reopen")
    repo = CorrectionRepository(db)
    finding = repo.record(task_id, _finding())
    repo.set_status(finding["id"], "resolved")

    reopened = repo.record(task_id, _finding())

    assert reopened["id"] == finding["id"]
    assert reopened["status"] == "open"
    assert reopened["resolved_at"] is None


def test_correction_repository_rolls_back_finding_when_event_fails(tmp_path):
    db = Database(tmp_path / "correction-rollback.sqlite")
    task_id = TaskRepository(db).create("rollback")
    repo = CorrectionRepository(db)
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_correction_event BEFORE INSERT ON events "
            "WHEN NEW.kind='correction.item_recorded' "
            "BEGIN SELECT RAISE(ABORT, 'event rejected'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event rejected"):
        repo.record(task_id, _finding())
    with db.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM correction_items").fetchone()[0]
            == 0
        )


def test_correction_status_transition_rolls_back_when_event_fails(tmp_path):
    db = Database(tmp_path / "correction-status-rollback.sqlite")
    task_id = TaskRepository(db).create("status rollback")
    repo = CorrectionRepository(db)
    item = repo.record(task_id, _finding())
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_correction_status BEFORE INSERT ON events "
            "WHEN NEW.kind='correction.item_status' "
            "BEGIN SELECT RAISE(ABORT, 'status event rejected'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="status event rejected"):
        repo.set_status(item["id"], "in_progress")
    assert repo.get(item["id"])["status"] == "open"
    assert repo.get(item["id"])["attempts"] == 0


def test_concurrent_correction_finding_records_are_idempotent(tmp_path):
    db = Database(tmp_path / "correction-concurrent.sqlite")
    task_id = TaskRepository(db).create("concurrent")

    def record():
        return CorrectionRepository(db).record(task_id, _finding())

    with ThreadPoolExecutor(max_workers=6) as pool:
        items = list(pool.map(lambda _: record(), range(12)))
    assert len({item["id"] for item in items}) == 1
    with db.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM correction_items").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM events WHERE kind=?",
                (EventKind.CORRECTION_ITEM_RECORDED.value,),
            ).fetchone()[0]
            == 1
        )


def test_correction_items_cascade_with_task_deletion(tmp_path):
    db = Database(tmp_path / "correction-cascade.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("delete")
    repo = CorrectionRepository(db)
    item = repo.record(task_id, _finding())

    assert tasks.delete_if_idle(task_id)
    assert repo.get(item["id"]) is None
    assert EventRepository(db).list(task_id) == []


def test_task_repository_idle_mutations_guard_lease_and_missing_task(tmp_path):
    db = Database(tmp_path / "idle-mutations.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("before")
    assert tasks.update_if_idle(task_id, "after", "description", {"priority": 2})
    assert tasks.get(task_id)["title"] == "after"
    assert not tasks.update_if_idle(999, "missing", "", {})
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO task_leases(task_id,owner,expires_at) VALUES(?,?,?)",
            (task_id, "worker", time.time() + 60),
        )
    assert not tasks.update_if_idle(task_id, "blocked", "", {})
    assert not tasks.delete_if_idle(task_id)
    with db.connect() as connection:
        connection.execute("DELETE FROM task_leases WHERE task_id=?", (task_id,))
    assert tasks.delete_if_idle(task_id)
    assert not tasks.delete_if_idle(task_id)


def test_question_ask_is_idempotent_and_atomically_records_state_and_event(tmp_path):
    db = Database(tmp_path / "question-atomic.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("question")
    questions = QuestionRepository(db)
    event = {"question": "Need input", "required": True}
    question_id = questions.ask(task_id, "Need input", "unclear", ["yes"], True, event)
    assert (
        questions.ask(task_id, "Need input", "unclear", ["yes"], True, event)
        == question_id
    )
    assert tasks.get(task_id)["status"] == "waiting_human"
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT kind,payload FROM events WHERE task_id=?", (task_id,)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == EventKind.QUESTION_ASKED.value
    assert json.loads(rows[0]["payload"]) == {
        "question_id": question_id,
        "question": "Need input",
        "required": True,
    }


def test_concurrent_duplicate_question_asks_create_one_question_and_event(tmp_path):
    db = Database(tmp_path / "question-concurrent.sqlite")
    task_id = TaskRepository(db).create("concurrent question")

    def ask():
        return QuestionRepository(db).ask(
            task_id,
            "Same question",
            "same reason",
            event_payload={"question": "Same question", "required": True},
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(lambda _: ask(), range(12)))
    assert len(set(ids)) == 1
    with db.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM questions WHERE task_id=?", (task_id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM events WHERE task_id=?", (task_id,)
            ).fetchone()[0]
            == 1
        )


def test_question_ask_rolls_back_when_audit_insert_fails(tmp_path):
    db = Database(tmp_path / "question-rollback.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("question")
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_question_event BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        QuestionRepository(db).ask(
            task_id,
            "Need input",
            "unclear",
            required=True,
            event_payload={"question": "Need input", "required": True},
        )
    assert tasks.get(task_id)["status"] == "pending"
    assert QuestionRepository(db).list(task_id) == []


def test_question_ask_rejects_missing_or_terminal_task_and_supports_optional(tmp_path):
    db = Database(tmp_path / "question-guards.sqlite")
    tasks = TaskRepository(db)
    questions = QuestionRepository(db)
    with pytest.raises(ValueError, match="task not found"):
        questions.ask(999, "Question", "reason")
    task_id = tasks.create("terminal")
    tasks.update(task_id, status="completed")
    with pytest.raises(ValueError, match="terminal task"):
        questions.ask(task_id, "Question", "reason")
    question_id = questions.ask(
        task_id, "FYI", "optional", required=False, event_payload=None
    )
    assert question_id and tasks.get(task_id)["status"] == "completed"
    with db.connect() as connection:
        event = connection.execute(
            "SELECT payload FROM events WHERE task_id=?", (task_id,)
        ).fetchone()
    assert json.loads(event["payload"]) == {"question_id": question_id}
