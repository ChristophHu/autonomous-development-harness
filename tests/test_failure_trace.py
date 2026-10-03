import sqlite3

import pytest

from harness.failure_trace import read_git_workflow_trace


def test_git_workflow_trace_is_read_only_bounded_and_redacted(tmp_path):
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE events(id INTEGER, task_id INTEGER, kind TEXT, payload TEXT, created_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO events VALUES(?,?,?,?,?)",
            [
                (
                    1,
                    7,
                    "git.workflow",
                    '{"workflow":"feature","phase":"branch_created","branch":"secret-name"}',
                    "now",
                ),
                (
                    2,
                    7,
                    "QUESTION_ASKED",
                    '{"question_id":3,"purpose":"approval","question":"hidden prompt"}',
                    "now",
                ),
                (
                    3,
                    7,
                    "task.status",
                    '{"from":"executing","status":"waiting_approval"}',
                    "now",
                ),
                (4, 7, "model.call", '{"prompt":"not selected"}', "now"),
            ],
        )
        connection.execute("CREATE TABLE questions(id INTEGER, reason TEXT)")
        connection.execute("INSERT INTO questions VALUES(3, 'git:reconciliation')")

    trace = read_git_workflow_trace(database, limit=2)
    assert [event["kind"] for event in trace] == ["QUESTION_ASKED", "task.status"]
    assert trace[0]["state"] == {
        "purpose": "approval",
        "question_id": 3,
        "reason_category": "git:reconciliation",
    }
    assert trace[1]["state"] == {"from": "executing", "status": "waiting_approval"}
    assert "secret-name" not in repr(trace)
    assert "hidden prompt" not in repr(trace)
    with (
        sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro", uri=True
        ) as connection,
        pytest.raises(sqlite3.OperationalError),
    ):
        connection.execute("INSERT INTO events VALUES(5,7,'git.workflow','{}','now')")


@pytest.mark.parametrize("limit", [0, -1, 101, True, "bad"])
def test_git_workflow_trace_rejects_invalid_limit(tmp_path, limit):
    database = tmp_path / "empty.sqlite"
    sqlite3.connect(database).close()
    assert read_git_workflow_trace(database, limit=limit) == []


def test_git_workflow_trace_handles_missing_database_and_schema(tmp_path):
    missing = tmp_path / "missing.sqlite"
    assert read_git_workflow_trace(missing) == []
    assert not missing.exists()
    malformed = tmp_path / "malformed.sqlite"
    with sqlite3.connect(malformed) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    assert read_git_workflow_trace(malformed) == []


def test_git_workflow_trace_handles_invalid_payload_and_closed_files(tmp_path):
    database = tmp_path / "events.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE events(id, task_id, kind, payload)")
        connection.execute("INSERT INTO events VALUES(1,1,'git.workflow','not-json')")
    assert read_git_workflow_trace(database) == [
        {"event_id": 1, "task_id": 1, "kind": "git.workflow", "state": {}}
    ]
