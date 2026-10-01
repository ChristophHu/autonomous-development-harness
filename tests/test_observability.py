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
        connection.execute(
            "INSERT INTO agent_runs(task_id,agent,profile,status,started_at) VALUES(?,?,?,?,?)",
            (
                created.id,
                "planner",
                "planner",
                "completed",
                "2026-09-30T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO tool_calls(task_id,tool,status,input,started_at) VALUES(?,?,?,?,?)",
            (created.id, "test.run", "completed", "{}", "2026-09-30T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO validations(task_id,valid,report,created_at) VALUES(?,?,?,?)",
            (created.id, 1, "{}", "2026-09-30T00:00:00+00:00"),
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
    assert report["agents"] == {"total": 1, "by_status": {"completed": 1}}
    assert report["tools"] == {
        "total": 1,
        "by_status": {"completed": 1},
        "by_name": {"test.run": {"total": 1, "by_status": {"completed": 1}}},
        "by_transport": {"native": {"total": 1, "by_status": {"completed": 1}}},
        "latency_ms": {"samples": 0, "average": None, "maximum": None},
    }
    assert report["validations"] == {"valid": 1, "invalid": 0}
    assert report["qdrant"] == {
        "probes": 0,
        "healthy": 0,
        "latest_latency_ms": None,
        "points": 0,
        "status": "unknown",
        "probe_age_seconds": None,
    }


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
    assert report["agents"] == {"total": 0, "by_status": {}}
    assert report["tools"] == {
        "total": 0,
        "by_status": {},
        "by_name": {},
        "by_transport": {},
        "latency_ms": {"samples": 0, "average": None, "maximum": None},
    }
    assert report["validations"] == {"valid": 0, "invalid": 0}
    assert report["qdrant"] == {
        "probes": 0,
        "healthy": 0,
        "latest_latency_ms": None,
        "points": 0,
        "status": "unknown",
        "probe_age_seconds": None,
    }


def test_prometheus_export_is_aggregate_stable_and_does_not_emit_names(tmp_path):
    from harness.core import Config, Store
    from harness.domain import EventKind

    config = Config(tmp_path / "metrics-export.yaml")
    config.data["paths"]["database"] = str(tmp_path / "metrics-export.sqlite")
    store = Store(config)
    with store.database.connect() as connection:
        connection.execute(
            "INSERT INTO tool_calls(tool,status,input,started_at) VALUES(?,?,?,?)",
            ("mcp.secret-canary", "completed", "{}", "2026-09-30T00:00:00+00:00"),
        )
    exported = ObservabilityService(store.database, EventKind).prometheus()
    assert "harness_tool_calls_total 1" in exported
    assert "# TYPE harness_model_cost_total gauge" in exported
    assert "secret-canary" not in exported
    assert 'tool="' not in exported
    assert exported.endswith("\n")


def test_prometheus_number_formats_missing_and_numeric_values():
    assert ObservabilityService._prometheus_number(None) == "NaN"
    assert ObservabilityService._prometheus_number(12) == "12"
    assert ObservabilityService._prometheus_number(float("inf")) == "NaN"


def test_qdrant_probe_snapshots_are_exposed_as_aggregate_metrics(tmp_path):
    from harness.core import Config, Store
    from harness.database import OperationalSnapshotRepository
    from harness.domain import EventKind

    config = Config(tmp_path / "probe.yaml")
    config.data["paths"]["database"] = str(tmp_path / "probe.sqlite")
    store = Store(config)
    snapshots = OperationalSnapshotRepository(store.database)
    snapshots.record_qdrant_probe(
        {
            "healthy": True,
            "service": "available",
            "collection_exists": True,
            "dimension": 1024,
            "points_count": 7,
            "errors": [],
        },
        12.5,
    )
    service = ObservabilityService(store.database, EventKind)
    assert service.metrics()["qdrant"] == {
        "probes": 1,
        "healthy": 1,
        "latest_latency_ms": 12.5,
        "points": 7,
        "status": "available",
        "probe_age_seconds": pytest.approx(0, abs=1),
    }
    exported = service.prometheus()
    assert "harness_qdrant_probes_total 1" in exported
    assert "harness_qdrant_collection_points 7" in exported


@pytest.mark.parametrize(
    "platform_name,rss,expected_rss",
    [("darwin", 2048, 2048), ("linux", 2, 2048)],
)
def test_runtime_resource_metrics_normalize_rss_and_database_size(
    tmp_path, monkeypatch, platform_name, rss, expected_rss
):
    from types import SimpleNamespace

    import harness.observability as module

    database_path = tmp_path / "runtime.sqlite"
    database_path.write_bytes(b"abc")
    clock_values = iter((10.0, 12.5))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(module.sys, "platform", platform_name)
    monkeypatch.setattr(
        module.resource,
        "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=rss),
    )
    service = ObservabilityService(SimpleNamespace(path=database_path), ())
    assert service.runtime_metrics() == {
        "uptime_seconds": 2.5,
        "process_max_rss_bytes": expected_rss,
        "database_bytes": 3,
    }


def test_runtime_metrics_fail_safe_for_missing_database_and_clock_regression(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import harness.observability as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 5.0)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(
        module.resource,
        "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=1),
    )
    service = ObservabilityService(
        SimpleNamespace(path=tmp_path / "missing.sqlite"), ()
    )
    assert service.runtime_metrics() == {
        "uptime_seconds": 0.0,
        "process_max_rss_bytes": 1,
        "database_bytes": 0,
    }


def test_tool_metrics_include_bounded_transport_names_and_latency(tmp_path):
    from test_evidence_workflow import ready_runtime

    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    with store.database.connect() as connection:
        connection.executemany(
            "INSERT INTO tool_calls(task_id,tool,status,input,started_at,finished_at) "
            "VALUES(?,?,?,?,?,?)",
            [
                (
                    created.id,
                    "mcp.vault.read_note",
                    "completed",
                    "{}",
                    "2026-09-30T00:00:00+00:00",
                    "2026-09-30T00:00:01+00:00",
                ),
                (
                    created.id,
                    "filesystem.read",
                    "failed",
                    "{}",
                    "2026-09-30T00:00:00+00:00",
                    "2026-09-30T00:00:02+00:00",
                ),
                (
                    created.id,
                    "mcp.vault.read_note",
                    "running",
                    "{}",
                    "2026-09-30T00:00:00+00:00",
                    None,
                ),
            ],
        )

    tools = orchestrator.observability.metrics()["tools"]
    assert tools["total"] == 3
    assert tools["by_name"] == {
        "filesystem.read": {"total": 1, "by_status": {"failed": 1}},
        "mcp.vault.read_note": {
            "total": 2,
            "by_status": {"completed": 1, "running": 1},
        },
    }
    assert tools["by_transport"] == {
        "mcp": {"total": 2, "by_status": {"completed": 1, "running": 1}},
        "native": {"total": 1, "by_status": {"failed": 1}},
    }
    assert tools["latency_ms"]["samples"] == 2
    assert tools["latency_ms"]["average"] == pytest.approx(1500, abs=5)
    assert tools["latency_ms"]["maximum"] == pytest.approx(2000, abs=5)
