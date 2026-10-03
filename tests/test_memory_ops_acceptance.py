import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from harness.database import Database
from harness.memory_ops_acceptance import (
    MIN_OBSERVATION_WINDOW_SECONDS,
    evaluate_memory_ops_acceptance,
    required_snapshot_count,
)

SUBJECT = "a" * 64
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)
CHECKS = {
    "sqlite": "healthy",
    "vault": "healthy",
    "qdrant": "healthy",
    "embedding_evidence": "healthy",
}


def snapshots_for(interval=60, *, count=None, age=None, healthy=True, checks=None):
    count = count or required_snapshot_count(interval)
    age = age if age is not None else timedelta(seconds=interval)
    return [
        {
            "id": index + 1,
            "observed_at": (
                NOW - age - timedelta(seconds=index * interval)
            ).isoformat(),
            "healthy": healthy,
            "checks": dict(CHECKS if checks is None else checks),
            "source_sha256": SUBJECT,
        }
        for index in range(count)
    ]


def launchd(loaded=True):
    return {"installed": loaded, "loaded": loaded}


def test_required_snapshot_count_covers_five_minute_window_and_validates_interval():
    assert required_snapshot_count(60) == 6
    assert required_snapshot_count(300) == 3
    assert required_snapshot_count(3600) == 3
    for value in (True, 4, 3601, 60.0):
        with pytest.raises(ValueError, match="interval"):
            required_snapshot_count(value)


def test_acceptance_passes_with_loaded_launchd_and_fresh_healthy_five_minute_history():
    report = evaluate_memory_ops_acceptance(
        snapshots_for(),
        launchd_status=launchd(),
        source_sha256=SUBJECT,
        interval_seconds=60,
        now=NOW,
    )
    assert report["passed"] is True
    assert report["checks"] == {
        "launchd_loaded": True,
        "snapshot_count_sufficient": True,
        "observation_window_met": True,
        "snapshots_healthy": True,
        "required_checks_present": True,
        "latest_snapshot_fresh": True,
        "monitor_source_current": True,
    }
    assert report["snapshot_count"] == 6
    assert report["observation_window_seconds"] == MIN_OBSERVATION_WINDOW_SECONDS


@pytest.mark.parametrize(
    ("kwargs", "failed_check"),
    [
        ({"launchd_status": {"installed": True, "loaded": False}}, "launchd_loaded"),
        ({"snapshots": []}, "snapshot_count_sufficient"),
        ({"window_seconds": 30}, "observation_window_met"),
        ({"healthy": False}, "snapshots_healthy"),
        ({"checks": {"sqlite": "healthy"}}, "required_checks_present"),
        ({"age": timedelta(minutes=3)}, "latest_snapshot_fresh"),
        ({"source_mismatch": True}, "monitor_source_current"),
    ],
)
def test_acceptance_fails_closed_for_unmet_operational_conditions(kwargs, failed_check):
    values = {
        "snapshots": snapshots_for(),
        "launchd_status": launchd(),
        "source_sha256": SUBJECT,
        "interval_seconds": 60,
        "now": NOW,
    }
    if "snapshots" in kwargs:
        values["snapshots"] = kwargs["snapshots"]
    else:
        snap_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"healthy", "checks", "age"}
        }
        values["snapshots"] = snapshots_for(**snap_kwargs)
        if "launchd_status" in kwargs:
            values["launchd_status"] = kwargs["launchd_status"]
    if "window_seconds" in kwargs:
        values["snapshots"] = snapshots_for()
        for index, snapshot in enumerate(values["snapshots"]):
            snapshot["observed_at"] = (
                NOW - timedelta(seconds=60 + index * 30)
            ).isoformat()
    if kwargs.get("source_mismatch"):
        values["snapshots"] = snapshots_for()
        for snapshot in values["snapshots"]:
            snapshot["source_sha256"] = "b" * 64
    report = evaluate_memory_ops_acceptance(**values)
    assert report["passed"] is False
    assert report["checks"][failed_check] is False


def test_acceptance_rejects_invalid_hash_interval_and_snapshot_timestamp():
    args = {
        "snapshots": snapshots_for(),
        "launchd_status": launchd(),
        "source_sha256": "bad",
        "interval_seconds": 60,
        "now": NOW,
    }
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_memory_ops_acceptance(**args)
    args["source_sha256"] = None
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_memory_ops_acceptance(**args)
    args["source_sha256"] = SUBJECT
    args["interval_seconds"] = True
    with pytest.raises(ValueError, match="interval"):
        evaluate_memory_ops_acceptance(**args)
    args["interval_seconds"] = 60
    args["snapshots"] = [{"observed_at": "bad"}]
    with pytest.raises(ValueError, match="timestamp"):
        evaluate_memory_ops_acceptance(**args)
    args["snapshots"] = None
    with pytest.raises(TypeError, match="snapshots"):
        evaluate_memory_ops_acceptance(**args)
    args["snapshots"] = snapshots_for()
    args["now"] = NOW.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        evaluate_memory_ops_acceptance(**args)
    args["now"] = NOW
    args["snapshots"] = snapshots_for()
    args["snapshots"][0]["observed_at"] = NOW.replace(tzinfo=None).isoformat()
    with pytest.raises(ValueError, match="timezone"):
        evaluate_memory_ops_acceptance(**args)
    args["snapshots"] = snapshots_for()
    args.pop("now")
    evaluate_memory_ops_acceptance(**args)


def test_memory_ops_evidence_is_success_only_and_append_only(tmp_path):
    database = Database(tmp_path / "state.sqlite")
    from harness.database import MemoryOpsEvidenceRepository

    repository = MemoryOpsEvidenceRepository(database)
    checks = {
        name: True
        for name in (
            "launchd_loaded",
            "snapshot_count_sufficient",
            "observation_window_met",
            "snapshots_healthy",
            "required_checks_present",
            "latest_snapshot_fresh",
            "monitor_source_current",
        )
    }
    saved = repository.record(subject_sha256=SUBJECT, observed_at=NOW, checks=checks)
    assert saved["passed"] is True
    assert repository.list() == [saved]
    with closing(sqlite3.connect(database.path)) as connection, connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM memory_ops_evidence WHERE id=?", (saved["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE memory_ops_evidence SET passed=0 WHERE id=?", (saved["id"],)
            )
    with pytest.raises(ValueError, match="checks"):
        repository.record(subject_sha256=SUBJECT, observed_at=NOW, checks={})
    with pytest.raises(ValueError, match="all checks"):
        repository.record(
            subject_sha256=SUBJECT,
            observed_at=NOW,
            checks=checks | {"snapshots_healthy": False},
        )
    with pytest.raises(ValueError, match="SHA-256"):
        repository.record(subject_sha256="nope", observed_at=NOW, checks=checks)
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.record(
            subject_sha256=SUBJECT,
            observed_at=NOW.replace(tzinfo=None),
            checks=checks,
        )
    with pytest.raises(ValueError, match="checks"):
        repository.record(subject_sha256=SUBJECT, observed_at=NOW, checks={"x": 1})
    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=True)
    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=0)


def test_recent_snapshots_validate_limit_and_preserve_descending_order(tmp_path):
    from harness.database import OperationalSnapshotRepository

    repository = OperationalSnapshotRepository(Database(tmp_path / "state.sqlite"))
    assert repository.recent_memory_health(limit=1) == []
    for state in ("healthy", "unhealthy"):
        repository.record_memory_health(
            {"healthy": state == "healthy", "checks": {"sqlite": state}}
        )
    assert [
        row["checks"]["sqlite"] for row in repository.recent_memory_health(limit=2)
    ] == ["unhealthy", "healthy"]
    for limit in (True, 0, 501):
        with pytest.raises(ValueError, match="limit"):
            repository.recent_memory_health(limit=limit)
