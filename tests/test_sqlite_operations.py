"""Operational SQLite checks must be observational and fail closed."""

import json
import sqlite3
from contextlib import closing

from typer.testing import CliRunner

from harness import cli
from harness.database import Database
from harness.sqlite_operations import inspect_sqlite


def test_sqlite_inspection_reports_current_schema_without_writing(tmp_path):
    path = tmp_path / "database with space.sqlite"
    Database(path)
    before = path.read_bytes()

    report = inspect_sqlite(path)

    assert report["status"] == "available"
    assert report["schema_version"] == Database.CURRENT_SCHEMA_VERSION
    assert report["version_history_complete"] is True
    assert report["integrity"] and report["foreign_keys"]
    assert report["findings"] == []
    assert path.read_bytes() == before


def test_sqlite_inspection_rejects_missing_symlink_and_invalid_file(tmp_path):
    missing = tmp_path / "missing.db"
    assert inspect_sqlite(missing)["status"] == "missing"
    target = tmp_path / "target.db"
    Database(target)
    link = tmp_path / "link.db"
    link.symlink_to(target)
    assert inspect_sqlite(link)["findings"] == ["database_symlink"]
    invalid = tmp_path / "invalid.db"
    invalid.write_text("not a database", encoding="utf-8")
    assert inspect_sqlite(invalid)["status"] == "unavailable"


def test_sqlite_inspection_flags_foreign_keys_and_schema_history(tmp_path):
    path = tmp_path / "incomplete.db"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "INSERT INTO events(task_id,kind,payload,created_at) VALUES(999,'task.created','{}','now')"
        )
        connection.execute("DELETE FROM schema_versions WHERE version=7")

    report = inspect_sqlite(path)

    assert report["status"] == "unhealthy"
    assert report["foreign_keys"] is False
    assert report["version_history_complete"] is False
    assert report["findings"] == ["foreign_key_violation", "schema_history_invalid"]


def test_sqlite_health_command_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / "cli.db"
    Database(path)
    monkeypatch.setattr(
        cli,
        "Config",
        lambda: type("ConfigFixture", (), {"path": lambda _self, _key: path})(),
    )
    before = path.read_bytes()
    result = CliRunner().invoke(cli.app, ["sqlite-health"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "available"
    assert path.read_bytes() == before


def test_sqlite_inspection_reports_locked_database_without_altering_it(tmp_path):
    path = tmp_path / "locked.db"
    Database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN EXCLUSIVE")
        report = inspect_sqlite(path)
        assert report["status"] == "unavailable"
        assert report["findings"] == ["database_probe_failed"]
        connection.rollback()
    assert inspect_sqlite(path)["status"] == "available"


def test_sqlite_inspection_handles_connection_failure(tmp_path, monkeypatch):
    path = tmp_path / "open-failure.db"
    Database(path)
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("private filesystem detail")
        ),
    )
    report = inspect_sqlite(path)
    assert report["status"] == "unavailable"
    assert report["findings"] == ["database_probe_failed"]
    assert "private filesystem detail" not in str(report)
