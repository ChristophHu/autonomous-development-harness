import sqlite3
import subprocess

import pytest

from harness.errors import FailureCategory, failure_category, failure_record


class ProviderError(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PermissionError("no"), FailureCategory.PERMISSION),
        (TimeoutError("late"), FailureCategory.TIMEOUT),
        (
            subprocess.TimeoutExpired(["tool"], 1),
            FailureCategory.TIMEOUT,
        ),
        (sqlite3.OperationalError("locked"), FailureCategory.PERSISTENCE),
        (ValueError("invalid"), FailureCategory.VALIDATION),
        (ProviderError("unavailable"), FailureCategory.PROVIDER),
        (RuntimeError("unexpected"), FailureCategory.EXECUTION),
    ],
)
def test_failure_category_classifies_stable_operational_classes(error, expected):
    assert failure_category(error) is expected


def test_failure_category_recognizes_cancellation_by_contract_name():
    class TaskCancelled(Exception):
        pass

    assert failure_category(TaskCancelled()) is FailureCategory.CANCELLATION


def test_failure_category_recognizes_http_transport_module(monkeypatch):
    class TransportFailure(Exception):
        pass

    monkeypatch.setattr(TransportFailure, "__module__", "httpx")
    error = TransportFailure("offline")
    assert failure_category(error) is FailureCategory.TRANSPORT


def test_failure_category_recognizes_provider_module(monkeypatch):
    class RemoteFailure(Exception):
        pass

    monkeypatch.setattr(RemoteFailure, "__module__", "harness.providers")
    assert failure_category(RemoteFailure()) is FailureCategory.PROVIDER


def test_failure_category_recognizes_validation_module(monkeypatch):
    class SchemaFailure(Exception):
        pass

    monkeypatch.setattr(SchemaFailure, "__module__", "pydantic.errors")
    assert failure_category(SchemaFailure()) is FailureCategory.VALIDATION


def test_failure_record_sanitizes_message_and_keeps_type():
    record = failure_record(
        RuntimeError("secret-value"),
        lambda text: text.replace("secret-value", "[REDACTED]"),
    )

    assert record == {
        "category": "execution",
        "error_type": "RuntimeError",
        "message": "[REDACTED]",
    }


def test_failure_record_without_sanitizer_preserves_diagnostic_message():
    assert failure_record(ValueError("bad input")) == {
        "category": "validation",
        "error_type": "ValueError",
        "message": "bad input",
    }
