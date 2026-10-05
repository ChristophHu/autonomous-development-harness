import sqlite3

import pytest

from harness.failure_trace import _safe_payload, read_git_workflow_trace


@pytest.mark.parametrize(
    "payload",
    [None, "not-json", "[]", '{"workflow":"private","phase":"Bad"}'],
)
def test_safe_payload_keeps_only_valid_allowlisted_fields(payload):
    assert _safe_payload(payload) == {}


def test_safe_payload_accepts_valid_states_but_rejects_malformed_optional_fields():
    assert _safe_payload(
        '{"status":"completed","from":"running","workflow":"feature",'
        '"phase":"git_push","purpose":"approval","question_id":7,'
        '"branch":"private","purpose2":"private"}'
    ) == {
        "status": "completed",
        "from": "running",
        "workflow": "feature",
        "phase": "git_push",
        "purpose": "approval",
        "question_id": 7,
    }


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
        connection.execute(
            "CREATE TABLE questions(id INTEGER, reason TEXT, question TEXT)"
        )
        connection.execute(
            "INSERT INTO questions VALUES(3, 'git:reconciliation', 'private question')"
        )

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
    assert "private question" not in repr(trace)
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            (
                "Git-Zustand manuell prüfen: git status failed: "
                "sandbox-exec: sandbox_apply: Operation not permitted"
            ),
            "host_sandbox_blocked",
        ),
        (
            "Git command failed: sandbox-exec: Operation not permitted",
            "sandbox_execution_denied",
        ),
        ("Git state needs review: private/path/value", None),
    ],
)
def test_git_workflow_trace_classifies_question_failure_without_text(
    tmp_path, question, expected
):
    database = tmp_path / "classified.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE events(id,task_id,kind,payload,created_at)")
        connection.execute(
            "INSERT INTO events VALUES(1,1,'QUESTION_ASKED',?, 'now')",
            ('{"question_id":2}',),
        )
        connection.execute("CREATE TABLE questions(id,reason,question)")
        connection.execute(
            "INSERT INTO questions VALUES(2,'git:reconciliation',?)", (question,)
        )

    trace = read_git_workflow_trace(database)
    state = trace[0]["state"]
    assert state.get("failure_class") == expected
    assert question not in repr(trace)
    assert "/private/path/value" not in repr(trace)


def test_git_workflow_trace_without_questions_table_omits_question_reason(tmp_path):
    database = tmp_path / "events-only.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE events(id, task_id, kind, payload)")
        connection.execute(
            "INSERT INTO events VALUES(1,1,'question.asked','{\"question_id\":4}')"
        )
    assert read_git_workflow_trace(database) == [
        {
            "event_id": 1,
            "task_id": 1,
            "kind": "question.asked",
            "state": {"question_id": 4},
        }
    ]


def test_git_workflow_trace_returns_empty_when_database_read_fails(
    tmp_path, monkeypatch
):
    database = tmp_path / "events.sqlite"
    database.touch()

    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("harness.failure_trace.sqlite3.connect", fail)
    assert read_git_workflow_trace(database) == []
