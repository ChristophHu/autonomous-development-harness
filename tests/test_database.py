import json
import multiprocessing
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from harness.database import (
    AgentRunRepository,
    ArtifactRepository,
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


def _initialize_database_process(path, barrier, queue):
    barrier.wait(timeout=10)
    Database(Path(path), timeout=10)
    queue.put("ok")


def _claim_task_process(path, task_id, owner, barrier, queue):
    db = Database(Path(path))
    barrier.wait(timeout=10)
    queue.put(TaskRepository(db).claim(task_id, owner))


def _crash_during_task_update(path, task_id):
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("UPDATE tasks SET title=? WHERE id=?", ("uncommitted", task_id))
    os._exit(73)


def _downgrade_schema(connection, target_version):
    """Build a historical fixture by reversing the checked-in migrations."""
    reverse_steps = {
        15: ("ALTER TABLE model_runs DROP COLUMN error_category",),
        14: ("ALTER TABLE questions DROP COLUMN purpose",),
        13: ("ALTER TABLE memory_health_snapshots DROP COLUMN source_sha256",),
        12: (
            "DROP TRIGGER memory_ops_evidence_no_update",
            "DROP TRIGGER memory_ops_evidence_no_delete",
            "DROP INDEX memory_ops_evidence_latest",
            "DROP TABLE memory_ops_evidence",
        ),
        11: (
            "DROP INDEX memory_health_latest",
            "DROP TABLE memory_health_snapshots",
        ),
        10: (
            "DROP INDEX model_discovery_latest",
            "DROP TABLE model_discovery_snapshots",
            "DROP TABLE qdrant_probe_snapshots",
        ),
        9: (
            "DROP TRIGGER verification_evidence_no_update",
            "DROP TRIGGER verification_evidence_no_delete",
            "DROP INDEX verification_evidence_kind_time",
            "DROP TABLE verification_evidence",
        ),
        8: (
            "DROP INDEX decisions_supersedes",
            "ALTER TABLE decisions DROP COLUMN supersedes_id",
        ),
        7: ("DROP INDEX artifacts_task_key", "DROP TABLE artifacts"),
        6: (
            "ALTER TABLE decisions DROP COLUMN tags",
            "ALTER TABLE decisions DROP COLUMN outcome",
            "ALTER TABLE decisions DROP COLUMN alternatives",
        ),
        5: ("DROP INDEX correction_items_task_status", "DROP TABLE correction_items"),
        4: (
            "ALTER TABLE decisions DROP COLUMN question_id",
            "ALTER TABLE decisions DROP COLUMN field_names",
            "ALTER TABLE decisions DROP COLUMN evidence",
            "ALTER TABLE decisions DROP COLUMN source",
            "ALTER TABLE decisions DROP COLUMN category",
        ),
        3: (
            "DROP INDEX tool_calls_identity",
            "ALTER TABLE tool_calls DROP COLUMN risk",
            "ALTER TABLE tool_calls DROP COLUMN profile",
            "ALTER TABLE tool_calls DROP COLUMN agent_run_id",
            "ALTER TABLE tool_calls DROP COLUMN call_id",
            "ALTER TABLE model_runs DROP COLUMN error_type",
            "ALTER TABLE model_runs DROP COLUMN fallback_index",
            "ALTER TABLE model_runs DROP COLUMN reasoning_tokens",
            "ALTER TABLE model_runs DROP COLUMN cached_tokens",
            "ALTER TABLE model_runs DROP COLUMN latency_ms",
            "ALTER TABLE model_runs DROP COLUMN finished_at",
            "ALTER TABLE model_runs DROP COLUMN started_at",
            "ALTER TABLE model_runs DROP COLUMN complexity",
            "ALTER TABLE model_runs DROP COLUMN status",
            "ALTER TABLE model_runs DROP COLUMN profile",
            "ALTER TABLE model_runs DROP COLUMN agent",
            "ALTER TABLE model_runs DROP COLUMN task_id",
        ),
        2: (
            "DROP TABLE task_leases",
            "ALTER TABLE subtasks DROP COLUMN plan_id",
            "ALTER TABLE tasks DROP COLUMN metadata",
        ),
    }
    for version in range(Database.CURRENT_SCHEMA_VERSION, target_version, -1):
        for statement in reverse_steps[version]:
            connection.execute(statement)
    connection.execute(
        "DELETE FROM schema_versions WHERE version > ?", (target_version,)
    )


def _seed_latest_schema(connection):
    connection.execute(
        "INSERT INTO tasks(id,title,description,status,result,created_at,updated_at,metadata) "
        "VALUES(1,'preserved task','description','pending','result','t0','t0','{\"k\":\"v\"}')"
    )
    connection.execute(
        "INSERT INTO plans(id,task_id,summary,payload,created_at) VALUES(1,1,'plan','{}','t1')"
    )
    connection.execute(
        "INSERT INTO questions(id,task_id,question,reason,created_at,purpose) "
        "VALUES(1,1,'preserved question','migration fixture','t2','approval')"
    )
    connection.execute(
        "INSERT INTO decisions(id,task_id,decision,rationale,created_at,category,source,evidence,field_names,question_id,alternatives,outcome,tags) "
        "VALUES(1,1,'preserved decision','rationale','t3','architecture','human','[]','[]',1,'[]','accepted','[]')"
    )
    connection.execute(
        "INSERT INTO subtasks(id,task_id,external_id,title,description,profile,status,plan_id) "
        "VALUES(1,1,'step-1','subtask','description','coding','completed',1)"
    )
    connection.execute(
        "INSERT INTO events(id,task_id,kind,payload,created_at) VALUES(1,1,'task.created','{}','t4')"
    )
    connection.execute(
        "INSERT INTO tool_calls(id,task_id,tool,status,input,started_at,call_id,profile,risk) "
        "VALUES(1,1,'shell','completed','{}','t5','call-1','coding','LOW')"
    )
    connection.execute(
        "INSERT INTO model_runs(id,provider,model,created_at,task_id,agent,profile,complexity,status) "
        "VALUES(1,'local','model','t6',1,'planner','planner','normal','completed')"
    )
    connection.execute(
        "INSERT INTO correction_items(id,task_id,category,source,rule,message,affected_paths,evidence,expected,created_at,updated_at) "
        "VALUES(?,1,'test','validator','rule','message','[]','[]','pass','t7','t7')",
        ("a" * 64,),
    )
    connection.execute(
        "INSERT INTO artifacts(task_id,artifact_key,version,content,sha256,created_at) "
        "VALUES(1,'agent/step-1',1,'output',?,'t8')",
        ("b" * 64,),
    )
    connection.execute(
        "INSERT INTO verification_evidence(kind,source_id,observed_at,subject_sha256,passed,checks_json,digest) "
        "VALUES('ci','ci:fixture','t9',?,1,'{\"tests\":true}',?)",
        ("c" * 64, "d" * 64),
    )
    connection.execute(
        "INSERT INTO model_discovery_snapshots(provider,observed_at,status,discovery_failed,models_json) "
        "VALUES('local','t10','available',0,'[]')"
    )
    connection.execute(
        "INSERT INTO qdrant_probe_snapshots(observed_at,healthy,status,latency_ms,collection_exists) "
        "VALUES('t11',1,'healthy',1.0,1)"
    )
    connection.execute(
        "INSERT INTO memory_health_snapshots(observed_at,healthy,checks_json,source_sha256) "
        "VALUES('t12',1,'{}',?)",
        ("e" * 64,),
    )
    connection.execute(
        "INSERT INTO memory_ops_evidence(observed_at,subject_sha256,passed,checks_json,digest) "
        "VALUES('t13',?,1,'{}',?)",
        ("f" * 64, "1" * 64),
    )


def test_failed_artifact_migration_does_not_record_version(tmp_path):
    path = tmp_path / "migration.sqlite"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP INDEX artifacts_task_key")
        connection.execute("DROP TABLE artifacts")
        connection.execute("DELETE FROM schema_versions WHERE version>=7")
        connection.execute(
            "CREATE TABLE artifacts(id INTEGER PRIMARY KEY, task_id INTEGER)"
        )

    with pytest.raises(sqlite3.OperationalError):
        Database(path)

    with closing(sqlite3.connect(path)) as connection, connection:
        assert (
            connection.execute(
                "SELECT 1 FROM schema_versions WHERE version=7"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='artifacts_task_key'"
            ).fetchone()
            is None
        )


def test_database_rejects_future_schema_version_without_mutating_it(tmp_path):
    path = tmp_path / "future-schema.sqlite"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO schema_versions(version, applied_at) VALUES(16, 'future')"
        )

    with pytest.raises(RuntimeError, match="newer than this Harness"):
        Database(path)

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA table_info(questions)").fetchall()
        assert (
            connection.execute(
                "SELECT version FROM schema_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            == 16
        )


def test_database_rejects_gapped_migration_history_without_mutating_it(tmp_path):
    path = tmp_path / "gapped-schema.sqlite"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM schema_versions WHERE version=7")

    with pytest.raises(RuntimeError, match="migration history is incomplete"):
        Database(path)

    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM schema_versions WHERE version=7"
            ).fetchone()
            is None
        )


def test_version_13_database_migrates_without_losing_question_data(tmp_path):
    path = tmp_path / "schema-13.sqlite"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO tasks(title,status,created_at,updated_at) VALUES('legacy','pending','t0','t0')"
        )
        task_id = connection.execute("SELECT id FROM tasks").fetchone()[0]
        connection.execute(
            "INSERT INTO questions(task_id,question,reason,created_at) VALUES(?,?,?,?)",
            (task_id, "Keep this question", "migration", "t1"),
        )
        connection.execute("ALTER TABLE questions DROP COLUMN purpose")
        connection.execute("ALTER TABLE model_runs DROP COLUMN error_category")
        connection.execute("DELETE FROM schema_versions WHERE version=15")
        connection.execute("DELETE FROM schema_versions WHERE version=14")

    Database(path)
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT question, reason, status, purpose FROM questions"
        ).fetchone()
        assert row == ("Keep this question", "migration", "open", "input")
        assert (
            connection.execute(
                "SELECT version FROM schema_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            == 15
        )
    Database(path)


def test_version_14_migration_preserves_model_error_and_adds_category(tmp_path):
    path = tmp_path / "schema-14-model-errors.sqlite"
    database = Database(path)
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO model_runs(provider,model,created_at,error_type) VALUES(?,?,?,?)",
            ("local", "fixture", "t0", "RuntimeError"),
        )
        connection.execute("ALTER TABLE model_runs DROP COLUMN error_category")
        connection.execute("DELETE FROM schema_versions WHERE version=15")

    Database(path)

    with database.connect() as connection:
        row = connection.execute(
            "SELECT error_type,error_category FROM model_runs"
        ).fetchone()
        assert tuple(row) == ("RuntimeError", None)
        assert (
            connection.execute(
                "SELECT version FROM schema_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            == Database.CURRENT_SCHEMA_VERSION
        )


@pytest.mark.parametrize(
    "starting_version", range(1, Database.CURRENT_SCHEMA_VERSION + 1)
)
def test_every_supported_historical_schema_migrates_without_data_loss(
    tmp_path, starting_version
):
    path = tmp_path / f"schema-{starting_version}.sqlite"
    Database(path)
    with Database(path).connect() as connection:
        _seed_latest_schema(connection)
    with closing(sqlite3.connect(path)) as connection, connection:
        _downgrade_schema(connection, starting_version)

    db = Database(path)
    Database(path)  # Reopening a migrated fixture must be idempotent.

    with db.connect() as connection:
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            )
        ]
        assert versions == list(range(1, Database.CURRENT_SCHEMA_VERSION + 1))
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert tuple(
            connection.execute(
                "SELECT title,result,metadata FROM tasks WHERE id=1"
            ).fetchone()
        ) == (
            "preserved task",
            "result",
            '{"k":"v"}' if starting_version >= 2 else "{}",
        )
        assert tuple(
            connection.execute(
                "SELECT question,answer,purpose FROM questions WHERE id=1"
            ).fetchone()
        ) == (
            "preserved question",
            None,
            "approval" if starting_version >= 14 else "input",
        )
        decision = connection.execute(
            "SELECT decision,category,source,alternatives,outcome,tags FROM decisions WHERE id=1"
        ).fetchone()
        assert tuple(decision) == (
            "preserved decision",
            "architecture" if starting_version >= 4 else "legacy",
            "human" if starting_version >= 4 else "legacy",
            "[]",
            "accepted" if starting_version >= 6 else None,
            "[]",
        )
        assert (
            connection.execute("SELECT kind FROM events WHERE id=1").fetchone()[0]
            == "task.created"
        )
        assert connection.execute("SELECT plan_id FROM subtasks WHERE id=1").fetchone()[
            0
        ] == (1 if starting_version >= 2 else None)
        assert connection.execute(
            "SELECT call_id FROM tool_calls WHERE id=1"
        ).fetchone()[0] == ("call-1" if starting_version >= 3 else None)
        feature_tables = {
            "correction_items": 5,
            "artifacts": 7,
            "verification_evidence": 9,
            "model_discovery_snapshots": 10,
            "qdrant_probe_snapshots": 10,
            "memory_health_snapshots": 11,
            "memory_ops_evidence": 12,
        }
        for table, introduced_version in feature_tables.items():
            count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert count == (1 if starting_version >= introduced_version else 0)
        source_hash_row = connection.execute(
            "SELECT source_sha256 FROM memory_health_snapshots WHERE id=1"
        ).fetchone()
        assert (source_hash_row[0] if source_hash_row else None) == (
            "e" * 64 if starting_version >= 13 else None
        )


def test_simultaneous_process_database_initialization_is_serialized(tmp_path):
    path = str(tmp_path / "multiprocess-init.sqlite")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_initialize_database_process, args=(path, barrier, queue)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    for process in processes:
        process.join(timeout=20)
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(queue.get(timeout=2) for _ in processes) == ["ok", "ok"]
    with closing(sqlite3.connect(path)) as connection:
        versions = connection.execute("SELECT version FROM schema_versions").fetchall()
        assert [row[0] for row in versions] == list(range(1, 16))


def test_independent_processes_cannot_claim_the_same_task(tmp_path):
    path = tmp_path / "multiprocess-lease.sqlite"
    db = Database(path)
    task_id = TaskRepository(db).create("single owner")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_claim_task_process,
            args=(str(path), task_id, f"owner-{index}", barrier, queue),
        )
        for index in range(2)
    ]

    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    results = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results) == [False, True]
    with db.connect() as connection:
        lease = connection.execute(
            "SELECT owner FROM task_leases WHERE task_id=?", (task_id,)
        ).fetchone()
    assert lease["owner"] in {"owner-0", "owner-1"}


def test_sqlite_recovers_atomically_after_process_crashes_mid_transaction(tmp_path):
    path = tmp_path / "crash-atomicity.sqlite"
    db = Database(path)
    tasks = TaskRepository(db)
    task_id = tasks.create("committed title")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_during_task_update, args=(str(path), task_id)
    )

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 73
    assert tasks.get(task_id)["title"] == "committed title"
    with db.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


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
        ] == list(range(1, 16))
    legacy_decision = DecisionRepository(db).get(3)
    assert legacy_decision["decision"] == "old decision"
    assert legacy_decision["category"] == legacy_decision["source"] == "legacy"
    assert legacy_decision["evidence"] == legacy_decision["field_names"] == []
    assert legacy_decision["alternatives"] == legacy_decision["tags"] == []
    assert legacy_decision["outcome"] is None


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


def test_recovery_state_transition_is_persisted_and_audited(tmp_path):
    db = Database(tmp_path / "recovery-state.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("recovery state")

    for status in ("analyzing", "planning", "ready", "executing", "recovering"):
        tasks.transition(task_id, status)

    assert tasks.get(task_id)["status"] == "recovering"
    statuses = [
        json.loads(event["payload"])["status"]
        for event in EventRepository(db).list(task_id, EventKind.TASK_STATUS)
    ]
    assert statuses[-1] == "recovering"
    tasks.transition(task_id, "analyzing")
    assert tasks.get(task_id)["status"] == "analyzing"

    with pytest.raises(ValueError, match="invalid task transition"):
        tasks.transition(task_id, "executing")


def test_database_busy_timeout_serializes_concurrent_writers(tmp_path):
    path = tmp_path / "busy.sqlite"
    db = Database(path, timeout=1)
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        started = time.monotonic()

        def contend():
            with db.connect() as contender:
                contender.execute("BEGIN IMMEDIATE")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(contend)
            time.sleep(0.05)
            connection.commit()
            future.result(timeout=2)
        assert time.monotonic() - started >= 0.04


def test_database_rejects_invalid_busy_timeout(tmp_path):
    for timeout in (-1, True, "1"):
        with pytest.raises(ValueError, match="non-negative number"):
            Database(tmp_path / f"invalid-{timeout}.sqlite", timeout=timeout)


def test_artifact_repository_versions_hashes_history_and_conflicts(tmp_path):
    db = Database(tmp_path / "artifacts.sqlite")
    task_id = TaskRepository(db).create("artifact")
    artifacts = ArtifactRepository(db)
    first = artifacts.save(task_id, "plan/step-1", "draft")
    second = artifacts.save(task_id, "plan/step-1", "final", expected_version=1)
    latest = artifacts.latest(task_id, "plan/step-1")
    assert latest["id"] == second["id"] and latest["version"] == 2
    assert latest["sha256"] == __import__("hashlib").sha256(b"final").hexdigest()
    assert [item["id"] for item in artifacts.history(task_id, "plan/step-1")] == [
        first["id"],
        second["id"],
    ]
    assert artifacts.latest(task_id, "missing") is None
    assert artifacts.history(task_id, "missing") == []
    with pytest.raises(ValueError, match="version conflict"):
        artifacts.save(task_id, "plan/step-1", "stale", expected_version=1)
    with pytest.raises(ValueError, match="task not found"):
        artifacts.save(task_id + 999, "missing-task", "content")


@pytest.mark.parametrize(
    "task_id,key,content,expected,message",
    [
        (0, "x", "y", None, "task_id"),
        (True, "x", "y", None, "task_id"),
        (1, "", "y", None, "artifact key"),
        (1, "x" * 201, "y", None, "artifact key"),
        (1, "x", 4, None, "content"),
        (1, "x", "y", -1, "expected_version"),
        (1, "x", "y", True, "expected_version"),
    ],
)
def test_artifact_repository_rejects_invalid_inputs(
    tmp_path, task_id, key, content, expected, message
):
    artifacts = ArtifactRepository(Database(tmp_path / "invalid.sqlite"))
    with pytest.raises((ValueError, TypeError), match=message):
        artifacts.save(task_id, key, content, expected)


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
    assert [row["kind"] for row in rows] == [
        EventKind.TASK_STATUS.value,
        EventKind.QUESTION_ASKED.value,
    ]
    assert json.loads(rows[1]["payload"]) == {
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
            == 2
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


@pytest.mark.parametrize(
    "purpose,status",
    [
        ("input", "waiting_human"),
        ("decision", "waiting_decision"),
        ("approval", "waiting_approval"),
    ],
)
def test_question_purpose_persists_a_distinct_task_lifecycle_state(
    tmp_path, purpose, status
):
    db = Database(tmp_path / f"question-{purpose}.sqlite")
    tasks = TaskRepository(db)
    questions = QuestionRepository(db)
    task_id = tasks.create("typed question")

    question_id = questions.ask(task_id, "Continue?", "needs input", purpose=purpose)

    assert tasks.get(task_id)["status"] == status
    assert questions.get(question_id)["purpose"] == purpose
    assert not questions.answer(question_id, "yes", task_id + 1)
    assert questions.answer(question_id, "yes", task_id)
    assert questions.get(question_id)["status"] == "answered"


def test_question_purpose_rejects_unknown_value_without_persisting(tmp_path):
    db = Database(tmp_path / "question-purpose-invalid.sqlite")
    tasks = TaskRepository(db)
    questions = QuestionRepository(db)
    task_id = tasks.create("invalid purpose")

    with pytest.raises(ValueError, match="purpose"):
        questions.ask(task_id, "Continue?", "reason", purpose="unknown")
    assert tasks.get(task_id)["status"] == "pending"
    assert questions.list(task_id) == []


def test_answering_one_of_multiple_purposes_keeps_remaining_state(tmp_path):
    db = Database(tmp_path / "question-purpose-multiple.sqlite")
    tasks = TaskRepository(db)
    questions = QuestionRepository(db)
    task_id = tasks.create("multiple purposes")
    question = questions.ask(task_id, "Choose?", "decision", purpose="decision")
    questions.ask(task_id, "Approve?", "approval", purpose="approval")

    assert questions.answer(question + 1, "yes", task_id)
    assert tasks.get(task_id)["status"] == "waiting_decision"
    assert questions.answer(question, "yes", task_id)
    assert tasks.get(task_id)["status"] == "waiting_decision"
    assert not questions.has_open_required(task_id)
