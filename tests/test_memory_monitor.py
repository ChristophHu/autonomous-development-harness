import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from harness.core import Config, Store
from harness.database import OperationalSnapshotRepository
from harness.evidence import EvidenceRepository
from harness.memory_monitor import (
    MemoryHealthMonitor,
    collect_memory_health,
    sqlite_health,
)

SUBJECT = "a" * 64
NOW = datetime(2026, 10, 2, tzinfo=UTC)


def runtime(tmp_path):
    config = Config(tmp_path / "config.yaml")
    config.data["paths"]["database"] = str(tmp_path / "state.sqlite")
    config.data["paths"]["obsidian_vault"] = str(tmp_path / "vault")
    config.data["paths"]["workspace"] = str(tmp_path / "workspace")
    store = Store(config)
    return config, store, OperationalSnapshotRepository(store.database)


def record_embedding(store, *, observed_at=NOW, subject=SUBJECT):
    return EvidenceRepository(store.database).record(
        {
            "kind": "embedding",
            "source_id": f"monitor-test:{observed_at.timestamp()}",
            "observed_at": observed_at,
            "subject_sha256": subject,
            "passed": True,
            "checks": {"round_trip": True},
        }
    )


def test_sqlite_health_reports_healthy_missing_corrupt_and_foreign_key_failure(
    tmp_path,
):
    _config, store, _snapshots = runtime(tmp_path)
    healthy = store.db
    assert sqlite_health(healthy) == "healthy"
    assert sqlite_health(tmp_path / "missing.sqlite") == "missing"

    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("not sqlite", encoding="utf-8")
    assert sqlite_health(corrupt) == "unavailable"

    broken = tmp_path / "foreign-key.sqlite"
    with closing(sqlite3.connect(broken)) as connection, connection:
        connection.executescript(
            "CREATE TABLE parent(id INTEGER PRIMARY KEY);"
            "CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));"
            "INSERT INTO child VALUES(7);"
        )
    assert sqlite_health(broken) == "unhealthy"


def test_collect_memory_health_handles_disabled_optional_services(tmp_path):
    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["obsidian"]["enabled"] = False
    config.data["memory"]["qdrant"]["enabled"] = False

    report = collect_memory_health(
        config,
        store,
        SimpleNamespace(),
        source_sha256=SUBJECT,
        now=NOW,
    )

    assert report == {
        "healthy": True,
        "checks": {
            "sqlite": "healthy",
            "vault": "disabled",
            "qdrant": "disabled",
            "embedding_evidence": "not_required",
        },
        "source_sha256": SUBJECT,
    }


def test_collect_memory_health_accepts_fresh_evidence_and_vault(tmp_path, monkeypatch):
    from harness import vault_audit

    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["qdrant"]["enabled"] = True
    config.data["memory"]["obsidian"]["enabled"] = True
    record_embedding(store)
    monkeypatch.setattr(
        vault_audit, "audit_vault", lambda *_args, **_kwargs: {"healthy": True}
    )
    qdrant = SimpleNamespace(health_report=lambda: {"healthy": True})

    report = collect_memory_health(
        config,
        store,
        qdrant,
        source_sha256=SUBJECT,
        now=NOW,
    )

    assert report["healthy"] is True
    assert report["source_sha256"] == SUBJECT
    assert report["checks"] == {
        "sqlite": "healthy",
        "vault": "healthy",
        "qdrant": "healthy",
        "embedding_evidence": "healthy",
    }
    assert "note text" not in json.dumps(report)


def test_collect_memory_health_reports_unhealthy_vault(tmp_path, monkeypatch):
    from harness import vault_audit

    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["obsidian"]["enabled"] = True
    monkeypatch.setattr(
        vault_audit, "audit_vault", lambda *_args, **_kwargs: {"healthy": False}
    )

    report = collect_memory_health(
        config, store, SimpleNamespace(), source_sha256=SUBJECT, now=NOW
    )

    assert report["checks"]["vault"] == "unhealthy"
    assert report["healthy"] is False


def test_collect_memory_health_reports_missing_and_stale_embedding_evidence(
    tmp_path,
):
    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["qdrant"]["enabled"] = True
    config.data["memory"].setdefault("monitoring", {})["evidence_max_age_hours"] = 24
    qdrant = SimpleNamespace(health_report=lambda: {"healthy": True})

    missing = collect_memory_health(
        config, store, qdrant, source_sha256=SUBJECT, now=NOW
    )
    assert missing["checks"]["embedding_evidence"] == "missing"
    assert missing["healthy"] is False

    record_embedding(store, observed_at=NOW - timedelta(days=2))
    stale = collect_memory_health(config, store, qdrant, source_sha256=SUBJECT, now=NOW)
    assert stale["checks"]["embedding_evidence"] == "stale"
    assert stale["healthy"] is False


def test_collect_memory_health_redacts_probe_failures(tmp_path, monkeypatch):
    from harness import evidence, vault_audit

    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["qdrant"]["enabled"] = True
    config.data["memory"]["obsidian"]["enabled"] = True
    monkeypatch.setattr(
        vault_audit,
        "audit_vault",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private vault path")),
    )
    monkeypatch.setattr(
        evidence.EvidenceRepository,
        "list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    qdrant = SimpleNamespace(
        health_report=lambda: (_ for _ in ()).throw(RuntimeError("private URL"))
    )

    report = collect_memory_health(
        config, store, qdrant, source_sha256=SUBJECT, now=NOW
    )

    assert report["healthy"] is False
    assert report["checks"] == {
        "sqlite": "healthy",
        "vault": "unavailable",
        "qdrant": "unavailable",
        "embedding_evidence": "unavailable",
    }
    assert "private" not in json.dumps(report)


def test_collect_memory_health_detects_unhealthy_qdrant_and_source_mismatch(
    tmp_path,
):
    config, store, _snapshots = runtime(tmp_path)
    config.data["memory"]["qdrant"]["enabled"] = True
    record_embedding(store)
    qdrant = SimpleNamespace(health_report=lambda: {"healthy": False})

    report = collect_memory_health(
        config, store, qdrant, source_sha256="b" * 64, now=NOW
    )

    assert report["checks"]["qdrant"] == "unhealthy"
    assert report["checks"]["embedding_evidence"] == "subject_mismatch"
    assert report["healthy"] is False


@pytest.mark.parametrize("interval", [True, 4, 3601, 1.5])
def test_memory_monitor_rejects_invalid_intervals(tmp_path, interval):
    _config, _store, snapshots = runtime(tmp_path)
    with pytest.raises(ValueError, match="interval"):
        MemoryHealthMonitor(dict, snapshots, interval_seconds=interval)


@pytest.mark.parametrize("max_checks", [True, 0, -1, 1.5])
def test_memory_monitor_rejects_invalid_check_counts(tmp_path, max_checks):
    _config, _store, snapshots = runtime(tmp_path)
    monitor = MemoryHealthMonitor(dict, snapshots, interval_seconds=5)
    with pytest.raises(ValueError, match="max_checks"):
        monitor.run(max_checks=max_checks)


def test_memory_monitor_persists_each_probe_and_emits_only_state_changes(tmp_path):
    _config, _store, snapshots = runtime(tmp_path)
    healthy = {"healthy": True, "checks": {"sqlite": "healthy"}}
    unhealthy = {"healthy": False, "checks": {"qdrant": "unhealthy"}}
    reports = iter([healthy, healthy, unhealthy])
    emitted = []
    sleeps = []
    monitor = MemoryHealthMonitor(
        lambda: next(reports),
        snapshots,
        interval_seconds=7,
        sleep=sleeps.append,
        emit=emitted.append,
    )

    assert monitor.run(max_checks=3) == 3
    assert emitted == [healthy, unhealthy]
    assert sleeps == [7, 7]
    latest = snapshots.latest_memory_health()
    assert latest["healthy"] is False
    assert latest["checks"] == {"qdrant": "unhealthy"}


@pytest.mark.parametrize(
    "report",
    [
        {"healthy": 1, "checks": {}},
        [],
        {"healthy": True, "checks": []},
    ],
)
def test_memory_monitor_rejects_invalid_probe_reports(tmp_path, report):
    _config, _store, snapshots = runtime(tmp_path)
    monitor = MemoryHealthMonitor(lambda: report, snapshots, interval_seconds=5)
    with pytest.raises(TypeError, match="probe"):
        monitor.run(max_checks=1)


def test_memory_monitor_runs_until_interrupted_by_its_waiter(tmp_path):
    _config, _store, snapshots = runtime(tmp_path)
    calls = []

    def stop(_seconds):
        raise KeyboardInterrupt

    monitor = MemoryHealthMonitor(
        lambda: (
            calls.append(True) or {"healthy": True, "checks": {"sqlite": "healthy"}}
        ),
        snapshots,
        interval_seconds=5,
        sleep=stop,
    )
    with pytest.raises(KeyboardInterrupt):
        monitor.run()
    assert calls == [True]
    assert snapshots.latest_memory_health()["healthy"] is True


@pytest.mark.parametrize(
    "report",
    [
        None,
        {"healthy": 1, "checks": {"sqlite": "healthy"}},
        {"healthy": True, "checks": []},
        {"healthy": True, "checks": {}},
        {"healthy": True, "checks": {1: "healthy"}},
        {"healthy": True, "checks": {"sqlite": False}},
        {"healthy": True, "checks": {"sqlite": "healthy"}, "source_sha256": "bad"},
    ],
)
def test_memory_health_repository_rejects_invalid_reports(tmp_path, report):
    _config, _store, snapshots = runtime(tmp_path)
    expected = "source SHA-256" if report and "source_sha256" in report else "report"
    with pytest.raises(ValueError, match=expected):
        snapshots.record_memory_health(report)


def test_memory_health_repository_returns_none_before_first_snapshot(tmp_path):
    _config, _store, snapshots = runtime(tmp_path)
    assert snapshots.latest_memory_health() is None
