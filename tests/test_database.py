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
    AuditRepository,
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


def _crash_during_step_result(path, task_id, subtask_id):
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "UPDATE subtasks SET status='completed',output='partial' WHERE task_id=? AND external_id=?",
        (task_id, subtask_id),
    )
    connection.execute(
        "INSERT INTO artifacts(task_id,artifact_key,version,content,sha256,created_at) "
        "VALUES(?,?,?,?,?,?)",
        (task_id, "agent/step", 1, "partial", "0" * 64, "crash-test"),
    )
    os._exit(74)


def _crash_after_step_result_commit(path, task_id, plan_id, subtask_id):
    database = Database(Path(path))
    SubtaskRepository(database).persist_result(
        task_id,
        subtask_id,
        plan_id,
        "completed",
        '{"success":true}',
        f"agent/{subtask_id}",
        '{"success":true}',
    )
    os._exit(75)


def _crash_during_completion(path, task_id):
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "UPDATE tasks SET status='completed',result='uncommitted' WHERE id=?",
        (task_id,),
    )
    connection.execute(
        "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
        (task_id, EventKind.TASK_STATUS.value, "{}", "crash-test"),
    )
    os._exit(76)


def _crash_after_completion_commit(path, task_id):
    TaskRepository(Database(Path(path))).complete(task_id, "validated", 1)
    os._exit(77)


def _crash_after_validation_commit(path, task_id, plan_id, contract_digest):
    database = Database(Path(path))
    ValidationRepository(database).record(
        task_id,
        True,
        {
            "completion_evidence": {
                "plan_id": plan_id,
                "task_contract_sha256": contract_digest,
                "workspace_sha256": "a" * 64,
            }
        },
    )
    os._exit(78)


def _downgrade_schema(connection, target_version):
    """Build a historical fixture by reversing the checked-in migrations."""
    reverse_steps = {
        18: (
            "DROP INDEX model_runs_token_calibration",
            "ALTER TABLE model_runs DROP COLUMN calibration_key",
            "ALTER TABLE model_runs DROP COLUMN calibration_safety_margin",
            "ALTER TABLE model_runs DROP COLUMN calibration_raw_tokens",
        ),
        17: ("DROP TRIGGER mcp_status_no_update", "DROP TRIGGER mcp_status_no_delete"),
        16: ("DROP INDEX mcp_status_latest", "DROP TABLE mcp_status_snapshots"),
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
            "INSERT INTO schema_versions(version, applied_at) VALUES(?, 'future')",
            (Database.CURRENT_SCHEMA_VERSION + 1,),
        )

    with pytest.raises(RuntimeError, match="newer than this Harness"):
        Database(path)

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA table_info(questions)").fetchall()
        assert (
            connection.execute(
                "SELECT version FROM schema_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
            == Database.CURRENT_SCHEMA_VERSION + 1
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
        _downgrade_schema(connection, 13)

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
            == Database.CURRENT_SCHEMA_VERSION
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
        _downgrade_schema(connection, 14)

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


def test_model_run_rejects_unknown_failure_category(tmp_path):
    database = Database(tmp_path / "unknown-model-category.sqlite")
    repository = AuditRepository(database)
    run_id = repository.start_model("fixture", "model", {}, 0)

    with pytest.raises(ValueError, match="unsupported model failure category"):
        repository.finish_model(
            run_id,
            "failed",
            1,
            error_category="not-a-failure-category",
        )


def test_model_token_calibration_samples_are_numeric_and_scoped(tmp_path):
    db = Database(tmp_path / "model-calibration.sqlite")
    repository = AuditRepository(db)

    for model, key, raw, actual, status in (
        ("model-id", "characters_per_token:3", 100, 140, "completed"),
        ("model-id", "characters_per_token:3", 100, 150, "completed"),
        ("model-id", "characters_per_token:4", 100, 999, "completed"),
        ("other-model", "characters_per_token:3", 100, 999, "completed"),
        ("model-id", "characters_per_token:3", 100, 999, "failed"),
    ):
        run_id = repository.start_model("fixture", model, {}, 0)
        repository.finish_model(
            run_id,
            status,
            1,
            usage=ModelUsage(
                "fixture", model, prompt_tokens=actual, completion_tokens=0
            ),
            calibration_raw_tokens=raw,
            calibration_safety_margin=20,
            calibration_key=key,
        )

    assert repository.token_calibration_samples(
        "model-id", "characters_per_token:3"
    ) == [(100, 150, 20), (100, 140, 20)]
    assert repository.token_calibration_samples(
        "model-id", "characters_per_token:3", limit=1
    ) == [(100, 150, 20)]


@pytest.mark.parametrize(
    ("model", "calibration_key", "limit"),
    [("", "key", 10), ("model", "", 10), ("model", "key", 0), ("model", "key", True)],
)
def test_model_token_calibration_query_rejects_invalid_arguments(
    tmp_path, model, calibration_key, limit
):
    repository = AuditRepository(
        Database(tmp_path / "invalid-model-calibration.sqlite")
    )
    with pytest.raises(ValueError):
        repository.token_calibration_samples(model, calibration_key, limit=limit)


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
        assert [row[0] for row in versions] == list(
            range(1, Database.CURRENT_SCHEMA_VERSION + 1)
        )


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


def test_crash_during_step_result_leaves_no_partial_artifact_or_status(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "crash-step-result.sqlite"
    db = Database(path)
    task_id = TaskRepository(db).create("crash during result")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    SubtaskRepository(db).save_plan(
        task_id,
        [SimpleNamespace(id="step", title="Step", description="", profile="coding")],
        plan_id,
    )
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_during_step_result,
        args=(str(path), task_id, "step"),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 74
    assert SubtaskRepository(db).list(task_id)[0]["status"] == "pending"
    assert ArtifactRepository(db).latest(task_id, "agent/step") is None
    assert EventRepository(db).list(task_id) == []
    with db.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_retry_after_crash_post_commit_returns_original_artifact(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "crash-after-step-commit.sqlite"
    db = Database(path)
    task_id = TaskRepository(db).create("crash after result commit")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    steps = SubtaskRepository(db)
    steps.save_plan(
        task_id,
        [SimpleNamespace(id="step", title="Step", description="", profile="coding")],
        plan_id,
    )
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_step_result_commit,
        args=(str(path), task_id, plan_id, "step"),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 75
    original = ArtifactRepository(db).latest(task_id, "agent/step")
    retried = steps.persist_result(
        task_id,
        "step",
        plan_id,
        "completed",
        '{"success":true}',
        "agent/step",
        '{"success":true}',
    )
    assert retried["id"] == original["id"]
    assert len(ArtifactRepository(db).history(task_id, "agent/step")) == 1
    assert [event["kind"] for event in EventRepository(db).list(task_id)] == [
        EventKind.ARTIFACT_RECORDED.value
    ]


def test_repeated_step_result_returns_existing_artifact_without_duplicate_event(
    tmp_path,
):
    from types import SimpleNamespace

    db = Database(tmp_path / "repeat-step-result.sqlite")
    task_id = TaskRepository(db).create("repeat step result")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    steps = SubtaskRepository(db)
    steps.save_plan(
        task_id,
        [SimpleNamespace(id="step", title="Step", description="", profile="coding")],
        plan_id,
    )
    first = steps.persist_result(
        task_id, "step", plan_id, "completed", "{}", "agent/step", "{}"
    )
    repeated = steps.persist_result(
        task_id, "step", plan_id, "completed", "{}", "agent/step", "{}"
    )
    changed = steps.persist_result(
        task_id, "step", plan_id, "completed", "{}", "agent/step", "new content"
    )

    assert repeated["id"] == first["id"]
    assert changed["id"] != first["id"]
    assert len(ArtifactRepository(db).history(task_id, "agent/step")) == 2
    assert len(EventRepository(db).list(task_id)) == 2


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
        ] == list(range(1, Database.CURRENT_SCHEMA_VERSION + 1))
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


def test_concurrent_database_initialization_applies_migrations_once(tmp_path):
    from threading import Barrier

    path = tmp_path / "concurrent-initialization.sqlite"
    barrier = Barrier(6)

    def initialize():
        barrier.wait(timeout=5)
        return Database(path, timeout=5)

    with ThreadPoolExecutor(max_workers=6) as pool:
        databases = list(pool.map(lambda _index: initialize(), range(6)))

    assert len(databases) == 6
    with databases[0].connect() as connection:
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            )
        ]
        assert versions == list(range(1, Database.CURRENT_SCHEMA_VERSION + 1))
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


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


def test_subtask_result_and_artifact_are_committed_as_one_unit(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "subtask-result-atomic.sqlite")
    task_id = TaskRepository(db).create("atomic step")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    steps = SubtaskRepository(db)
    steps.save_plan(
        task_id,
        [
            SimpleNamespace(
                id="implement", title="Implement", description="", profile="coding"
            )
        ],
        plan_id,
    )

    artifact = steps.persist_result(
        task_id,
        "implement",
        plan_id,
        "completed",
        '{"success":true}',
        "agent/implement",
        '{"subtask_id":"implement","success":true}',
    )

    row = steps.list(task_id)[0]
    assert row["status"] == "completed"
    assert row["output"] == '{"success":true}'
    assert artifact["version"] == 1
    assert artifact["content"] == '{"subtask_id":"implement","success":true}'
    assert [event["kind"] for event in EventRepository(db).list(task_id)] == [
        EventKind.ARTIFACT_RECORDED.value
    ]


def test_subtask_result_rolls_back_when_artifact_write_fails(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "subtask-result-rollback.sqlite")
    task_id = TaskRepository(db).create("atomic step")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    steps = SubtaskRepository(db)
    steps.save_plan(
        task_id,
        [
            SimpleNamespace(
                id="implement", title="Implement", description="", profile="coding"
            )
        ],
        plan_id,
    )
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_step_artifact BEFORE INSERT ON artifacts "
            "BEGIN SELECT RAISE(ABORT,'artifact unavailable'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="artifact unavailable"):
        steps.persist_result(
            task_id,
            "implement",
            plan_id,
            "completed",
            '{"success":true}',
            "agent/implement",
            '{"subtask_id":"implement","success":true}',
        )

    assert steps.list(task_id)[0]["status"] == "pending"
    assert steps.list(task_id)[0]["output"] is None
    assert ArtifactRepository(db).latest(task_id, "agent/implement") is None
    assert EventRepository(db).list(task_id) == []


@pytest.mark.parametrize(
    "task_id,external_id,plan_id,status,output,key,content,error,message",
    [
        (0, "step", None, "completed", "{}", "artifact", "{}", ValueError, "task_id"),
        (1, " ", None, "completed", "{}", "artifact", "{}", ValueError, "external_id"),
        (1, "step", True, "completed", "{}", "artifact", "{}", ValueError, "plan_id"),
        (1, "step", None, "pending", "{}", "artifact", "{}", ValueError, "status"),
        (1, "step", None, "completed", None, "artifact", "{}", TypeError, "strings"),
        (1, "step", None, "completed", "{}", " ", "{}", ValueError, "artifact_key"),
    ],
)
def test_subtask_result_rejects_invalid_persistence_contract(
    tmp_path,
    task_id,
    external_id,
    plan_id,
    status,
    output,
    key,
    content,
    error,
    message,
):
    db = Database(tmp_path / f"invalid-subtask-result-{task_id}-{status}.sqlite")
    with pytest.raises(error, match=message):
        SubtaskRepository(db).persist_result(
            task_id, external_id, plan_id, status, output, key, content
        )


def test_subtask_result_requires_matching_plan_step(tmp_path):
    db = Database(tmp_path / "missing-subtask-result.sqlite")
    task_id = TaskRepository(db).create("missing step")
    with pytest.raises(ValueError, match="subtask not found"):
        SubtaskRepository(db).persist_result(
            task_id, "missing", None, "completed", "{}", "agent/missing", "{}"
        )


def test_subtask_result_detects_step_removed_before_update(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "removed-step-result.sqlite")
    task_id = TaskRepository(db).create("step removed during commit")
    plan_id = PlanRepository(db).save(task_id, "plan", {})
    steps = SubtaskRepository(db)
    steps.save_plan(
        task_id,
        [SimpleNamespace(id="step", title="Step", description="", profile="coding")],
        plan_id,
    )
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER ignore_step_update BEFORE UPDATE ON subtasks "
            "BEGIN SELECT RAISE(IGNORE); END"
        )

    with pytest.raises(ValueError, match="subtask not found"):
        steps.persist_result(
            task_id, "step", plan_id, "completed", "{}", "agent/step", "{}"
        )


def test_task_completion_commits_result_and_events_atomically(tmp_path):
    db = Database(tmp_path / "task-completion.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("complete atomically", status="validating")
    assert tasks.claim(task_id, "runner")

    tasks.complete(task_id, "validated", 2, "runner")

    row = tasks.get(task_id)
    assert (row["status"], row["result"]) == ("completed", "validated")
    events = EventRepository(db).list(task_id)
    assert [event["kind"] for event in events] == [
        EventKind.TASK_STATUS.value,
        EventKind.TASK_COMPLETED.value,
    ]
    assert json.loads(events[0]["payload"]) == {
        "from": "validating",
        "status": "completed",
    }
    assert json.loads(events[1]["payload"]) == {"attempt": 2}


def test_task_completion_rolls_back_when_completion_event_fails(tmp_path):
    db = Database(tmp_path / "task-completion-rollback.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("rollback completion", status="validating")
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_task_completed BEFORE INSERT ON events "
            "WHEN NEW.kind='task.completed' "
            "BEGIN SELECT RAISE(ABORT,'event unavailable'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event unavailable"):
        tasks.complete(task_id, "validated", 1)

    row = tasks.get(task_id)
    assert (row["status"], row["result"]) == ("validating", None)
    assert EventRepository(db).list(task_id) == []


def test_completion_crash_boundaries_never_expose_partial_success(tmp_path):
    path = tmp_path / "completion-crash.sqlite"
    db = Database(path)
    tasks = TaskRepository(db)
    task_id = tasks.create("completion crash", status="validating")
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_during_completion, args=(str(path), task_id)
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 76
    assert tasks.get(task_id)["status"] == "validating"
    assert tasks.get(task_id)["result"] is None
    assert EventRepository(db).list(task_id) == []

    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_completion_commit, args=(str(path), task_id)
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 77
    assert tasks.get(task_id)["status"] == "completed"
    assert tasks.get(task_id)["result"] == "validated"
    assert [event["kind"] for event in EventRepository(db).list(task_id)] == [
        EventKind.TASK_STATUS.value,
        EventKind.TASK_COMPLETED.value,
    ]
    with db.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_task_transition_rolls_back_status_when_event_insert_fails(tmp_path):
    db = Database(tmp_path / "transition-event-fault.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("atomic transition")
    original = tasks.get(task_id)["status"]
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_status_event BEFORE INSERT ON events "
            "WHEN NEW.kind='task.status' "
            "BEGIN SELECT RAISE(ABORT,'event unavailable'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event unavailable"):
        tasks.transition(task_id, "analyzing")

    assert tasks.get(task_id)["status"] == original
    assert EventRepository(db).list(task_id) == []


def test_task_claim_failure_rolls_back_expired_lease_cleanup(tmp_path):
    db = Database(tmp_path / "claim-lease-fault.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("atomic claim")
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO task_leases(task_id,owner,expires_at) VALUES(?,?,?)",
            (task_id, "expired", 0),
        )
        connection.execute(
            "CREATE TRIGGER reject_new_lease BEFORE INSERT ON task_leases "
            "BEGIN SELECT RAISE(ABORT,'lease unavailable'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="lease unavailable"):
        tasks.claim(task_id, "new-owner")

    with db.connect() as connection:
        leases = connection.execute(
            "SELECT owner,expires_at FROM task_leases WHERE task_id=?", (task_id,)
        ).fetchall()
    assert [(row["owner"], row["expires_at"]) for row in leases] == [("expired", 0)]


@pytest.mark.parametrize(
    "task_id,result,attempt,owner,message",
    [
        (0, "ok", 1, None, "contract"),
        (True, "ok", 1, None, "contract"),
        (1, " ", 1, None, "contract"),
        (1, "ok", True, None, "contract"),
        (1, "ok", -1, None, "contract"),
        (1, "ok", 1, "", "contract"),
    ],
)
def test_task_completion_rejects_invalid_contract(
    tmp_path, task_id, result, attempt, owner, message
):
    db = Database(tmp_path / f"task-completion-invalid-{task_id}-{attempt}.sqlite")
    with pytest.raises(ValueError, match=message):
        TaskRepository(db).complete(task_id, result, attempt, owner)


def test_task_completion_requires_existing_task_and_owned_lease(tmp_path):
    db = Database(tmp_path / "task-completion-lease.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("leased completion", status="validating")
    with pytest.raises(ValueError, match="task not found"):
        tasks.complete(task_id + 1, "ok", 1)
    with pytest.raises(ValueError, match="lease is no longer owned"):
        tasks.complete(task_id, "ok", 1, "wrong-owner")


def test_task_completion_requires_validating_state(tmp_path):
    db = Database(tmp_path / "task-completion-transition.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("not yet validating")
    with pytest.raises(ValueError, match="invalid task transition"):
        tasks.complete(task_id, "ok", 1)


def test_task_completion_blocks_open_required_question(tmp_path):
    db = Database(tmp_path / "task-completion-question.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("question blocks completion", status="validating")
    QuestionRepository(db).ask(task_id, "Need input?", "required")
    tasks.update(task_id, status="validating")
    with pytest.raises(ValueError, match="required question"):
        tasks.complete(task_id, "ok", 1)


def test_task_completion_blocks_open_correction(tmp_path):
    db = Database(tmp_path / "task-completion-correction.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("correction blocks completion", status="validating")
    CorrectionRepository(db).record(
        task_id,
        {
            "category": "test",
            "source": "unit-test",
            "rule": "must-fix",
            "message": "must be verified",
        },
    )
    with pytest.raises(ValueError, match="required correction"):
        tasks.complete(task_id, "ok", 1)


def test_planned_task_completion_requires_latest_bound_validation(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "bound-completion.sqlite")
    tasks = TaskRepository(db)
    metadata = {"title": "bound", "description": "", "goal": "finish"}
    task_id = tasks.create("bound", status="validating", metadata=metadata)
    plan_id = PlanRepository(db).save_execution_plan(
        task_id,
        "plan",
        {"subtasks": []},
        [SimpleNamespace(id="step", title="step", description="", profile="coding")],
        metadata,
    )
    digest = PlanRepository.contract_digest(metadata)
    binding = {
        "plan_id": plan_id,
        "task_contract_sha256": digest,
        "workspace_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(task_id, "done", 1)
    validation_id = ValidationRepository(db).record(
        task_id, True, {"completion_evidence": binding}
    )
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(
            task_id,
            "done",
            1,
            evidence={"validation_id": validation_id, "workspace_sha256": "b" * 64},
        )
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(
            task_id,
            "done",
            1,
            evidence={"validation_id": True, "workspace_sha256": "a" * 64},
        )
    assert tasks.get(task_id)["status"] == "validating"
    assert EventRepository(db).list(task_id) == []
    tasks.complete(
        task_id,
        "done",
        1,
        evidence={"validation_id": validation_id, "workspace_sha256": "a" * 64},
    )
    assert tasks.get(task_id)["status"] == "completed"


def test_planned_completion_rejects_newer_validation_and_changed_contract(tmp_path):
    db = Database(tmp_path / "stale-completion.sqlite")
    tasks = TaskRepository(db)
    metadata = {"title": "bound", "description": ""}
    task_id = tasks.create("bound", status="validating", metadata=metadata)
    plan_id = PlanRepository(db).save_execution_plan(task_id, "plan", {}, [], metadata)
    evidence = {"workspace_sha256": "a" * 64}
    evidence["validation_id"] = ValidationRepository(db).record(
        task_id,
        True,
        {
            "completion_evidence": {
                "plan_id": plan_id,
                "task_contract_sha256": PlanRepository.contract_digest(metadata),
                "workspace_sha256": evidence["workspace_sha256"],
            }
        },
    )
    ValidationRepository(db).record(task_id, False, {"errors": ["later failure"]})
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(task_id, "done", 1, evidence=evidence)
    tasks.update(task_id, title="changed")
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(task_id, "done", 1, evidence=evidence)
    assert tasks.get(task_id)["status"] == "validating"


def test_crash_after_validation_commit_does_not_complete_stale_plan(tmp_path):
    path = tmp_path / "validation-crash.sqlite"
    db = Database(path)
    tasks = TaskRepository(db)
    metadata = {"title": "recover", "description": ""}
    task_id = tasks.create("recover", status="validating", metadata=metadata)
    plans = PlanRepository(db)
    plan_id = plans.save_execution_plan(task_id, "first", {}, [], metadata)
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_validation_commit,
        args=(str(path), task_id, plan_id, plans.contract_digest(metadata)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 78
    assert tasks.get(task_id)["status"] == "validating"
    assert EventRepository(db).list(task_id, EventKind.TASK_COMPLETED.value) == []
    latest_validation = ValidationRepository(db).latest(task_id)
    assert latest_validation["valid"] is True
    evidence = {
        "validation_id": latest_validation["id"],
        "workspace_sha256": "a" * 64,
    }
    plans.save_execution_plan(task_id, "replacement", {}, [], metadata)
    with pytest.raises(ValueError, match="current validation"):
        tasks.complete(task_id, "done", 1, evidence=evidence)
    assert tasks.get(task_id)["status"] == "validating"


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


def test_execution_plan_revision_persists_plan_steps_and_task_snapshot_atomically(
    tmp_path,
):
    from types import SimpleNamespace

    db = Database(tmp_path / "atomic-plan.sqlite")
    task_id = TaskRepository(db).create("atomic plan", metadata={"revision": 0})
    plans = PlanRepository(db)
    steps = [
        SimpleNamespace(
            id="inspect", title="Inspect", description="read", profile="planner"
        ),
        SimpleNamespace(
            id="implement", title="Implement", description="write", profile="coding"
        ),
    ]

    plan_id = plans.save_execution_plan(
        task_id,
        "two-step plan",
        {"subtasks": ["inspect", "implement"]},
        steps,
        {"revision": 1, "plan": {"subtasks": ["inspect", "implement"]}},
    )

    assert plans.latest(task_id)["id"] == plan_id
    assert len(plans.latest(task_id)["payload"]["task_contract_sha256"]) == 64
    assert [row["external_id"] for row in SubtaskRepository(db).list(task_id)] == [
        "inspect",
        "implement",
    ]
    assert json.loads(TaskRepository(db).get(task_id)["metadata"]) == {
        "revision": 1,
        "plan": {"subtasks": ["inspect", "implement"]},
    }


def test_plan_revision_binds_contract_changes_to_a_new_digest(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "plan-contract-digest.sqlite")
    task_id = TaskRepository(db).create("contract digest")
    plans = PlanRepository(db)
    steps = [
        SimpleNamespace(
            id="inspect", title="Inspect", description="", profile="planner"
        )
    ]
    common = {
        "goal": "build a safe feature",
        "requirements": ["preserve existing work"],
        "acceptance_criteria": ["tests pass"],
        "test_commands": ["pytest"],
        "coverage_command": "coverage report",
    }
    plans.save_execution_plan(task_id, "first", {}, steps, {"title": "t", **common})
    first = plans.latest(task_id)["payload"]["task_contract_sha256"]
    plans.save_execution_plan(
        task_id,
        "revised",
        {},
        steps,
        common | {"title": "t", "requirements": ["preserve data and work"]},
    )
    second = plans.latest(task_id)["payload"]["task_contract_sha256"]
    assert first != second


def test_plan_fingerprint_matches_then_detects_changed_task_contract(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "plan-contract-current.sqlite")
    metadata = {
        "title": "contract",
        "description": "initial",
        "goal": "safe delivery",
        "requirements": ["preserve data"],
        "acceptance_criteria": [{"id": "tests-pass"}],
        "test_commands": ["pytest"],
        "coverage_command": "coverage report",
    }
    task_id = TaskRepository(db).create(
        metadata["title"], metadata["description"], metadata=metadata
    )
    plans = PlanRepository(db)
    plans.save_execution_plan(
        task_id,
        "initial plan",
        {},
        [SimpleNamespace(id="step", title="Step", description="", profile="coding")],
        metadata,
    )

    assert plans.inspect_latest_contract(task_id)["current"] is True
    changed = dict(metadata) | {"requirements": ["preserve all data"]}
    TaskRepository(db).update(task_id, metadata=changed)
    report = plans.inspect_latest_contract(task_id)
    assert report["current"] is False
    assert report["stored_sha256"] != report["current_sha256"]


def test_plan_fingerprint_fails_closed_for_legacy_or_missing_plan(tmp_path):
    db = Database(tmp_path / "plan-contract-legacy.sqlite")
    tasks = TaskRepository(db)
    task_id = tasks.create("legacy plan")
    plans = PlanRepository(db)
    assert plans.inspect_latest_contract(task_id)["current"] is False
    plan_id = plans.save(task_id, "legacy", {"subtasks": []})
    result = plans.inspect_latest_contract(task_id)
    assert result["current"] is False
    assert result["plan_id"] == plan_id
    assert result["stored_sha256"] is None


def test_plan_fingerprint_fails_closed_for_malformed_stored_contract(tmp_path):
    db = Database(tmp_path / "plan-contract-malformed.sqlite")
    task_id = TaskRepository(db).create("malformed contract")
    plans = PlanRepository(db)
    plan_id = plans.save(task_id, "plan", {"task_contract_sha256": "abc"})

    with db.connect() as connection:
        connection.execute("UPDATE tasks SET metadata='[]' WHERE id=?", (task_id,))
    assert plans.inspect_latest_contract(task_id)["current"] is False

    with db.connect() as connection:
        connection.execute("UPDATE tasks SET metadata='{}' WHERE id=?", (task_id,))
        connection.execute("UPDATE plans SET payload='[]' WHERE id=?", (plan_id,))
    assert plans.inspect_latest_contract(task_id)["current"] is False

    with db.connect() as connection:
        connection.execute("UPDATE plans SET payload='{' WHERE id=?", (plan_id,))
    result = plans.inspect_latest_contract(task_id)
    assert result["current"] is False
    assert result["plan_id"] == plan_id
    assert result["stored_sha256"] is None


def test_plan_contract_digest_rejects_non_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        PlanRepository.contract_digest([])


@pytest.mark.parametrize(
    "summary,steps,metadata",
    [
        ("", [], {}),
        ("plan", ["duplicate", "duplicate"], {}),
        ("plan", [""], {}),
        ("plan", [], []),
    ],
)
def test_execution_plan_persistence_rejects_invalid_contract(
    tmp_path, summary, steps, metadata
):
    from types import SimpleNamespace

    db = Database(tmp_path / "invalid-atomic-plan.sqlite")
    task_id = TaskRepository(db).create("atomic plan")
    with pytest.raises(ValueError, match="contract"):
        PlanRepository(db).save_execution_plan(
            task_id,
            summary,
            {},
            [SimpleNamespace(id=item) for item in steps],
            metadata,
        )


def test_execution_plan_persistence_rolls_back_every_write_on_step_failure(tmp_path):
    from types import SimpleNamespace

    db = Database(tmp_path / "rollback-plan.sqlite")
    task_id = TaskRepository(db).create("atomic plan", metadata={"original": True})
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER fail_plan_step BEFORE INSERT ON subtasks "
            "BEGIN SELECT RAISE(ABORT,'step insert failed'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="step insert failed"):
        PlanRepository(db).save_execution_plan(
            task_id,
            "new plan",
            {"subtasks": ["one"]},
            [SimpleNamespace(id="one", title="one", description="", profile="coding")],
            {"new": True},
        )

    assert PlanRepository(db).latest(task_id) is None
    assert SubtaskRepository(db).list(task_id) == []
    assert json.loads(TaskRepository(db).get(task_id)["metadata"]) == {"original": True}


def test_execution_plan_persistence_rejects_missing_task_without_partial_rows(tmp_path):
    db = Database(tmp_path / "missing-atomic-plan.sqlite")
    with pytest.raises(ValueError, match="task not found"):
        PlanRepository(db).save_execution_plan(
            999,
            "plan",
            {},
            [],
            {},
        )
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0


def test_execution_plan_persistence_rolls_back_if_task_snapshot_cannot_update(
    tmp_path,
):
    from types import SimpleNamespace

    db = Database(tmp_path / "ignored-task-update.sqlite")
    task_id = TaskRepository(db).create("atomic plan", metadata={"original": True})
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER ignore_task_snapshot BEFORE UPDATE ON tasks "
            "BEGIN SELECT RAISE(IGNORE); END"
        )

    with pytest.raises(ValueError, match="task not found"):
        PlanRepository(db).save_execution_plan(
            task_id,
            "new plan",
            {"subtasks": ["one"]},
            [SimpleNamespace(id="one", title="one", description="", profile="coding")],
            {"new": True},
        )

    assert PlanRepository(db).latest(task_id) is None
    assert SubtaskRepository(db).list(task_id) == []
    assert json.loads(TaskRepository(db).get(task_id)["metadata"]) == {"original": True}


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


def test_correction_retry_budget_claims_open_findings_atomically(tmp_path):
    db = Database(tmp_path / "correction-budget.sqlite")
    task_id = TaskRepository(db).create("correction")
    repo = CorrectionRepository(db)
    first = repo.record(task_id, _finding(rule="first"))
    second = repo.record(task_id, _finding(rule="second"))

    claimed = repo.start_open_for_task(task_id, max_attempts=1)

    assert {item["id"] for item in claimed} == {first["id"], second["id"]}
    assert {item["attempts"] for item in claimed} == {1}
    assert {item["status"] for item in claimed} == {"in_progress"}
    with pytest.raises(ValueError, match="retry budget exhausted"):
        repo.set_status(first["id"], "open")
        repo.start_open_for_task(task_id, max_attempts=1)


def test_correction_retry_budget_survives_repository_reopen(tmp_path):
    path = tmp_path / "correction-budget-reopen.sqlite"
    db = Database(path)
    task_id = TaskRepository(db).create("durable correction budget")
    repo = CorrectionRepository(db)
    item = repo.record(task_id, _finding())
    claimed = repo.start_open_for_task(task_id, max_attempts=2)[0]
    assert claimed["attempts"] == 1

    # Model a process restart: the lease-like in-progress state is reopened,
    # but the consumed correction attempt must not be reset.
    reopened = CorrectionRepository(Database(path))
    reopened.set_status(item["id"], "open")
    again = reopened.start_open_for_task(task_id, max_attempts=2)[0]
    assert again["attempts"] == 2
    reopened.set_status(item["id"], "open")

    with pytest.raises(ValueError, match="retry budget exhausted"):
        CorrectionRepository(Database(path)).start_open_for_task(
            task_id, max_attempts=2
        )
    persisted = CorrectionRepository(Database(path)).get(item["id"])
    assert persisted["attempts"] == 2
    assert persisted["status"] == "open"


def test_correction_retry_budget_does_not_partially_claim_findings(tmp_path):
    db = Database(tmp_path / "correction-budget-rollback.sqlite")
    task_id = TaskRepository(db).create("correction")
    repo = CorrectionRepository(db)
    first = repo.record(task_id, _finding(rule="first"))
    second = repo.record(task_id, _finding(rule="second"))
    repo.set_status(second["id"], "in_progress")
    repo.set_status(second["id"], "open")

    with pytest.raises(ValueError, match="retry budget exhausted"):
        repo.start_open_for_task(task_id, max_attempts=1)

    assert repo.get(first["id"])["status"] == "open"
    assert repo.get(first["id"])["attempts"] == 0
    assert repo.get(second["id"])["status"] == "open"
    assert repo.get(second["id"])["attempts"] == 1


def test_correction_retry_budget_claim_rolls_back_if_audit_event_fails(tmp_path):
    db = Database(tmp_path / "correction-budget-event-failure.sqlite")
    task_id = TaskRepository(db).create("correction")
    repo = CorrectionRepository(db)
    item = repo.record(task_id, _finding())
    with db.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_retry_event BEFORE INSERT ON events "
            "WHEN NEW.kind='correction.item_status' "
            "BEGIN SELECT RAISE(ABORT,'event unavailable'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event unavailable"):
        repo.start_open_for_task(task_id, max_attempts=1)

    current = repo.get(item["id"])
    assert current["status"] == "open"
    assert current["attempts"] == 0


def test_correction_retry_budget_claim_without_pending_findings_is_empty(tmp_path):
    repo = CorrectionRepository(Database(tmp_path / "empty-correction-budget.sqlite"))
    task_id = TaskRepository(repo.db).create("no corrections")

    assert repo.start_open_for_task(task_id, max_attempts=0) == []


def test_sqlite_writer_contention_fails_without_partial_task_write(tmp_path):
    import sqlite3

    path = tmp_path / "writer-contention.sqlite"
    db = Database(path, timeout=0.01)
    blocker = sqlite3.connect(path, timeout=0.01)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            TaskRepository(db).create("must not partially persist")
    finally:
        blocker.rollback()
        blocker.close()

    assert TaskRepository(db).list() == []


@pytest.mark.parametrize("max_attempts", [-1, True, 11, "2"])
def test_correction_retry_budget_rejects_invalid_limits(tmp_path, max_attempts):
    repo = CorrectionRepository(Database(tmp_path / "bad-correction-budget.sqlite"))
    with pytest.raises(ValueError, match="max_attempts"):
        repo.start_open_for_task(1, max_attempts=max_attempts)


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


def test_supersede_obsolete_conflict_questions_is_audited_and_not_answerable(tmp_path):
    db = Database(tmp_path / "question-supersede.sqlite")
    task_id = TaskRepository(db).create("conflict refresh")
    questions = QuestionRepository(db)
    obsolete = questions.ask(
        task_id,
        "Choose old evidence",
        "requirements:conflict:goal:oldhash",
        purpose="decision",
    )
    retained = questions.ask(
        task_id,
        "Choose current evidence",
        "requirements:conflict:goal:newhash",
        purpose="decision",
    )
    unrelated = questions.ask(task_id, "Provide input", "requirements:incomplete")

    assert questions.supersede_conflict_questions(
        task_id, {"requirements:conflict:goal:newhash"}
    ) == [obsolete]
    assert questions.get(obsolete)["status"] == "superseded"
    assert questions.get(retained)["status"] == "open"
    assert questions.get(unrelated)["status"] == "open"
    assert not questions.answer(obsolete, "{}", task_id)
    with db.connect() as connection:
        event = connection.execute(
            "SELECT kind,payload FROM events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert event["kind"] == EventKind.QUESTION_SUPERSEDED.value
    assert json.loads(event["payload"]) == {
        "question_id": obsolete,
        "reason_kind": "requirements:conflict",
    }


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
