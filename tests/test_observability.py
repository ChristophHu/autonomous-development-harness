import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.audit import CURRENT_RUN
from harness.observability import (
    JsonLogFormatter,
    ObservabilityService,
    configure_logging,
    redact_log_message,
)


def test_log_redaction_covers_assignments_bearer_and_configured_values():
    message = redact_log_message(
        "Authorization: Bearer abc password=plain api_key: qwerty inline-value",
        ["inline-value", ""],
    )
    assert "abc" not in message
    assert "plain" not in message
    assert "qwerty" not in message
    assert "inline-value" not in message
    assert message.count("[REDACTED]") == 4
    assert redact_log_message(12) == "12"


def test_json_log_formatter_adds_correlation_and_exception_type():
    record = logging.LogRecord(
        "harness.test", logging.ERROR, __file__, 1, "token=secret", (), None
    )
    record.created = datetime(2026, 9, 30, tzinfo=UTC).timestamp()
    token = CURRENT_RUN.set(
        {"task_id": 9, "agent_run_id": 4, "agent": "executor", "profile": "coding"}
    )
    try:
        payload = json.loads(JsonLogFormatter(["secret"]).format(record))
    finally:
        CURRENT_RUN.reset(token)
    assert payload["task_id"] == 9
    assert payload["agent_run_id"] == 4
    assert payload["message"] == "token=[REDACTED]"
    assert "exception_type" not in payload


def test_json_log_formatter_omits_empty_context_and_limits_exception_details():
    try:
        raise RuntimeError("credential=must-not-appear")
    except RuntimeError:
        record = logging.LogRecord(
            "harness.test",
            logging.WARNING,
            __file__,
            1,
            "safe",
            (),
            __import__("sys").exc_info(),
        )
    payload = json.loads(JsonLogFormatter().format(record))
    assert "task_id" not in payload
    assert payload["exception_type"] == "RuntimeError"
    assert "must-not-appear" not in json.dumps(payload)


def test_configure_logging_is_rotating_structured_and_idempotent(tmp_path):
    path = tmp_path / "logs" / "harness.jsonl"
    external = logging.NullHandler()
    logging.getLogger("harness").addHandler(external)
    logger = configure_logging(
        path, level="DEBUG", max_bytes=1024, backup_count=2, secrets=["supersecret"]
    )
    owned = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_harness_managed", False)
    ]
    assert len(owned) == 2
    assert external in logger.handlers
    assert any(getattr(handler, "maxBytes", None) == 1024 for handler in owned)
    logger.info("provider secret=supersecret")
    assert "supersecret" not in path.read_text()
    configure_logging(path, level="WARNING", max_bytes=2048, backup_count=1)
    owned = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_harness_managed", False)
    ]
    assert len(owned) == 2
    assert logger.level == logging.WARNING
    logger.removeHandler(external)
    external.close()


@pytest.mark.parametrize("absolute", [False, True])
def test_runtime_build_honors_configured_log_path_and_redacts_secrets(
    tmp_path, monkeypatch, absolute
):
    from types import SimpleNamespace

    from harness import core

    path = tmp_path / "absolute.jsonl" if absolute else Path("logs/custom.jsonl")

    class Settings:
        def __init__(self):
            self.logging = SimpleNamespace(
                file=str(path), level="INFO", max_bytes=4096, backup_count=2
            )

    class ConfigFixture:
        def __init__(self):
            self._path = tmp_path / "config.yaml"
            self.settings = Settings()
            self.data = {"secrets": {"SERVICE_API_KEY": "private-canary"}}

        def path(self, _key):
            return tmp_path / "logs"

    monkeypatch.setattr(core, "Config", ConfigFixture)
    monkeypatch.setattr(core, "Store", lambda _config: "store")
    monkeypatch.setattr(core, "Orchestrator", lambda _store, _config: "runtime")

    config, store, runtime = core.build()
    assert isinstance(config, ConfigFixture)
    assert store == "store" and runtime == "runtime"
    logger = logging.getLogger("harness")
    logger.info("auth private-canary")
    destination = path if path.is_absolute() else tmp_path / path
    assert "private-canary" not in destination.read_text()
    for handler in tuple(logger.handlers):
        if getattr(handler, "_harness_managed", False):
            logger.removeHandler(handler)
            handler.close()


def test_metrics_are_derived_from_sqlite_and_include_event_catalogue(tmp_path):
    from test_evidence_workflow import ready_runtime

    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    store.event(created.id, "task.created", {})
    with store.database.connect() as connection:
        connection.execute(
            "INSERT INTO model_runs(agent_run_id,provider,model,prompt_tokens,completion_tokens,cost,created_at) "
            "VALUES(NULL,'fixture','unit',4,5,0.02,?)",
            ("2026-09-30T00:00:00+00:00",),
        )
    report = orchestrator.observability.metrics()
    assert report["tasks"] == {"total": 1, "by_status": {"pending": 1}}
    assert report["events"]["total"] == 1
    assert report["events"]["by_kind"] == {"task.created": 1}
    assert "task.created" in report["events"]["catalogue"]
    assert report["models"]["runs"] == 1
    assert report["models"]["prompt_tokens"] == 4
    assert report["models"]["completion_tokens"] == 5
    assert report["models"]["cost"] == 0.02
    assert report["models"]["missing_usage"] == 0


def test_metrics_represent_missing_model_usage_as_zero_totals(tmp_path):
    from harness.core import Config, Store
    from harness.domain import EventKind

    config = Config(tmp_path / "missing.yaml")
    config.data["paths"]["database"] = str(tmp_path / "empty.sqlite")
    store = Store(config)
    report = ObservabilityService(store.database, EventKind).metrics()
    assert report["tasks"] == {"total": 0, "by_status": {}}
    assert report["events"] == {
        "total": 0,
        "by_kind": {},
        "catalogue": sorted(item.value for item in EventKind),
    }
    assert report["models"]["runs"] == 0
    assert report["models"]["prompt_tokens"] == 0
    assert report["models"]["cost"] == 0
