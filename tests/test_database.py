import sqlite3

import pytest

from harness.database import (
    AgentRunRepository,
    Database,
    DecisionRepository,
    EventRepository,
    ModelRunRepository,
    PlanRepository,
    QuestionRepository,
    SubtaskRepository,
    TaskRepository,
)
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
        ] == [1, 2, 3]


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
    eid = events.append(task_id, "created", {"actor": "test"})
    assert events.after(0, task_id)[0]["id"] == eid
    assert events.list(task_id, event_type="created")
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
