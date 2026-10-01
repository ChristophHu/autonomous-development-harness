import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from harness import cli
from harness.database import Database
from harness.evidence import EvidenceInput, EvidenceRepository, audit_evidence

SUBJECT = "a" * 64
NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)


def payload(**updates):
    value = {
        "kind": "ci",
        "source_id": "ci:run-17",
        "observed_at": NOW,
        "subject_sha256": SUBJECT,
        "passed": True,
        "checks": {"unit_tests": True, "coverage": True},
    }
    return EvidenceInput.model_validate(value | updates)


def test_evidence_input_rejects_unknown_fields_secrets_and_untyped_checks():
    with pytest.raises(ValidationError):
        EvidenceInput.model_validate(payload().model_dump() | {"raw_log": "secret"})
    with pytest.raises(ValidationError):
        payload(checks={"unit_tests": "passed"})
    with pytest.raises(ValidationError):
        payload(source_id="run id with spaces")
    with pytest.raises(ValidationError):
        payload(checks={"../../log": True})
    with pytest.raises(ValidationError):
        EvidenceInput.model_validate(
            {
                "kind": "ci",
                "source_id": "ci:bad-time",
                "observed_at": "not-a-timestamp",
                "subject_sha256": SUBJECT,
                "passed": True,
                "checks": {"unit_tests": True},
            }
        )


def test_evidence_repository_persists_idempotence_and_append_only_guards(tmp_path):
    database = Database(tmp_path / "evidence.sqlite")
    repository = EvidenceRepository(database)
    stored = repository.record(payload())
    assert stored["kind"] == "ci"
    assert len(stored["digest"]) == 64
    assert repository.list(kind="ci") == [stored]
    with pytest.raises(ValueError, match="already exists"):
        repository.record(payload())
    with sqlite3.connect(database.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM verification_evidence WHERE id=?", (stored["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE verification_evidence SET passed=0 WHERE id=?", (stored["id"],)
            )


def test_evidence_repository_rejects_invalid_kind_and_limit(tmp_path):
    repository = EvidenceRepository(Database(tmp_path / "evidence.sqlite"))
    with pytest.raises(ValueError, match="kind"):
        repository.list(kind="other")
    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=0)
    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=True)


def test_evidence_audit_accepts_fresh_matching_record_and_rejects_stale_or_mismatch(
    tmp_path,
):
    repository = EvidenceRepository(Database(tmp_path / "evidence.sqlite"))
    good = repository.record(payload())
    stale = repository.record(
        payload(source_id="ci:old", observed_at=NOW - timedelta(days=8))
    )
    mismatched = repository.record(
        payload(source_id="ci:other", subject_sha256="b" * 64)
    )
    failed = repository.record(payload(source_id="ci:failed", passed=False))
    report = audit_evidence(repository.list(), expected_subject_sha256=SUBJECT, now=NOW)
    statuses = {item["id"]: item for item in report["items"]}
    assert statuses[good["id"]]["valid"] is True
    assert statuses[stale["id"]]["reason"] == "stale"
    assert statuses[mismatched["id"]]["reason"] == "subject_mismatch"
    assert statuses[failed["id"]]["reason"] == "checks_failed"
    assert report["healthy"] is False
    with pytest.raises(ValueError, match="SHA-256"):
        audit_evidence([], expected_subject_sha256="invalid", now=NOW)
    with pytest.raises(ValueError, match="max_age"):
        audit_evidence([], expected_subject_sha256=SUBJECT, now=NOW, max_age_hours=0)


def test_evidence_audit_rejects_future_time_and_empty_evidence(tmp_path):
    repository = EvidenceRepository(Database(tmp_path / "future.sqlite"))
    future = repository.record(
        payload(source_id="ci:future", observed_at=NOW + timedelta(minutes=1))
    )
    report = audit_evidence([future], expected_subject_sha256=SUBJECT, now=NOW)
    assert report["items"][0]["reason"] == "timestamp_in_future"
    empty = audit_evidence([], expected_subject_sha256=SUBJECT, now=NOW)
    assert empty["healthy"] is False
    naive = audit_evidence(
        [
            {
                "id": 7,
                "kind": "ci",
                "observed_at": NOW.replace(tzinfo=None).isoformat(),
                "subject_sha256": SUBJECT,
                "passed": True,
                "checks": {"unit_tests": True},
            }
        ],
        expected_subject_sha256=SUBJECT,
        now=NOW,
    )
    assert naive["items"][0]["reason"] == "timestamp_not_aware"


def test_evidence_cli_import_list_and_audit(tmp_path, monkeypatch, capsys):
    database = tmp_path / "cli-evidence.sqlite"
    monkeypatch.setattr(
        cli,
        "Config",
        lambda: SimpleNamespace(path=lambda _name: database),
    )
    source = tmp_path / "evidence.json"
    source.write_text(
        json.dumps(
            {
                "kind": "ci",
                "source_id": "ci:cli-run",
                "observed_at": datetime.now(UTC).isoformat(),
                "subject_sha256": SUBJECT,
                "passed": True,
                "checks": {"unit_tests": True},
            }
        ),
        encoding="utf-8",
    )
    cli.evidence_import(source)
    assert json.loads(capsys.readouterr().out)["source_id"] == "ci:cli-run"
    cli.evidence_list(kind="ci", limit=20)
    assert json.loads(capsys.readouterr().out)[0]["kind"] == "ci"
    cli.evidence_audit(subject_sha256=SUBJECT, max_age_hours=168)
    assert json.loads(capsys.readouterr().out)["healthy"] is True


def test_evidence_cli_rejects_malformed_import_and_audit_hash(
    tmp_path, monkeypatch, capsys
):
    database = tmp_path / "invalid-cli.sqlite"
    monkeypatch.setattr(
        cli,
        "Config",
        lambda: SimpleNamespace(path=lambda _name: database),
    )
    source = tmp_path / "invalid.json"
    source.write_text('{"unexpected": true}', encoding="utf-8")
    with pytest.raises(cli.typer.Exit) as exit_error:
        cli.evidence_import(source)
    assert exit_error.value.exit_code == 2
    assert "Evidence import rejected" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit) as exit_error:
        cli.evidence_audit(subject_sha256="bad", max_age_hours=168)
    assert exit_error.value.exit_code == 2
    assert "Evidence audit rejected" in capsys.readouterr().out


def test_evidence_cli_rejects_bad_kind_and_empty_audit(tmp_path, monkeypatch):
    database = tmp_path / "empty-cli.sqlite"
    monkeypatch.setattr(
        cli,
        "Config",
        lambda: SimpleNamespace(path=lambda _name: database),
    )
    with pytest.raises(cli.typer.Exit) as exit_error:
        cli.evidence_list(kind="unknown", limit=100)
    assert exit_error.value.exit_code == 2
    result = CliRunner().invoke(
        cli.app, ["evidence", "audit", "--subject-sha256", SUBJECT]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["healthy"] is False
