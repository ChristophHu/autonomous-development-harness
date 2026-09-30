import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from harness import cli
from harness.core import Config, Orchestrator, Store, Task
from harness.providers import ProviderHealth
from harness.service_lifecycle import ServiceLifecycle


def _write_pid_record(path, pid, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "pid": pid,
                "process_fingerprint": fingerprint,
                "started_at": "2026-09-29T10:00:00+00:00",
                "host": "127.0.0.1",
                "port": 8080,
            }
        )
    )


@pytest.fixture
def harness_context(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data["paths"]["database"] = str(tmp_path / "harness.db")
    cfg.data["paths"]["logs"] = str(tmp_path / "logs")
    cfg.data["paths"]["obsidian_vault"] = str(tmp_path / "vault")
    cfg.data["paths"]["workspace"] = str(tmp_path / "workspace")
    for path in (cfg.path("obsidian_vault"), cfg.path("workspace")):
        path.mkdir(parents=True, exist_ok=True)
    cfg.data["profiles"] = {
        name: {"model": {"primary": "local"}}
        for name in ["planner", "software-architect", "coding", "validator"]
    }
    store = Store(cfg)
    orchestrator = Orchestrator(store, cfg)
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    return cfg, store, orchestrator, tmp_path


def test_start_stop_paths(harness_context, monkeypatch, capsys):
    _, _, _, root = harness_context
    calls = []
    lifecycle = ServiceLifecycle(
        root / "data" / "harness.pid",
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="stable test process identity"
        ),
    )
    monkeypatch.setattr(cli, "_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(
        cli,
        "_start_preflight",
        lambda *_args: [
            {"name": "Configuration", "status": "available", "critical": True},
            {"name": "SQLite", "status": "available", "critical": True},
            {
                "name": "Provider primary",
                "status": "unavailable",
                "critical": False,
                "role": "primary",
            },
        ],
    )
    monkeypatch.setattr(cli, "_serve_api", lambda *args: calls.append(args) or True)
    monkeypatch.setattr(cli.os, "kill", lambda *_args: None)
    cli.start()
    assert calls and not lifecycle.pidfile.exists()
    with pytest.raises(cli.typer.BadParameter):
        cli.start(host="0.0.0.0")
    monkeypatch.setattr(cli.os, "kill", lambda *_args: None)
    lifecycle.register_current("127.0.0.1", 8080, pid=123)
    with pytest.raises(cli.typer.BadParameter):
        cli.start()
    monkeypatch.setattr(
        cli.os,
        "kill",
        lambda pid, _sig: (
            (_ for _ in ()).throw(ProcessLookupError()) if pid == 123 else None
        ),
    )
    cli.start()
    cli.stop()
    output = capsys.readouterr().out
    assert "not running" in output
    assert "Provider primary [primary]: unavailable (optional)" in output


def test_stop_stale_and_active(harness_context, monkeypatch, capsys):
    _, _, _, root = harness_context
    lifecycle = ServiceLifecycle(root / "data" / "harness.pid")
    monkeypatch.setattr(cli, "_lifecycle", lambda: lifecycle)
    lifecycle.register_current = lambda *_args: None
    pidfile = lifecycle.pidfile
    _write_pid_record(pidfile, 999, "a" * 64)
    monkeypatch.setattr(lifecycle, "process_fingerprint", lambda _pid: None)
    cli.stop()
    assert not pidfile.exists()
    _write_pid_record(pidfile, 123, "a" * 64)
    fingerprints = iter(["a" * 64, None])
    monkeypatch.setattr(
        lifecycle, "process_fingerprint", lambda _pid: next(fingerprints)
    )
    signals = []
    monkeypatch.setattr(
        lifecycle,
        "stop",
        lambda: signals.append("terminated") or "stopped",
    )
    cli.stop()
    assert signals == ["terminated"]
    assert "stopped gracefully" in capsys.readouterr().out


def test_start_removes_its_record_when_api_never_becomes_ready(
    harness_context, monkeypatch
):
    _, _, _, root = harness_context
    lifecycle = ServiceLifecycle(
        root / "data" / "harness.pid",
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="stable test process identity"
        ),
    )
    monkeypatch.setattr(cli, "_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(cli.os, "kill", lambda *_args: None)
    monkeypatch.setattr(cli, "_serve_api", lambda *_args: False)

    with pytest.raises(cli.typer.Exit):
        cli.start()

    assert lifecycle.read_record() is None


def test_start_converts_registration_errors(harness_context, monkeypatch):
    class FailingLifecycle:
        def register_current(self, *_args):
            raise RuntimeError("unsafe pid file")

    monkeypatch.setattr(cli, "_lifecycle", FailingLifecycle)
    with pytest.raises(cli.typer.BadParameter, match="unsafe pid file"):
        cli.start()


def test_start_reports_build_failure_without_sensitive_details(monkeypatch, capsys):
    for exception in (
        ValueError("secret config detail"),
        cli.yaml.YAMLError("secret yaml detail"),
        cli.sqlite3.OperationalError("secret database detail"),
    ):
        monkeypatch.setattr(
            cli, "build", lambda error=exception: (_ for _ in ()).throw(error)
        )
        with pytest.raises(cli.typer.Exit) as error:
            cli.start()
        output = capsys.readouterr().out
        assert error.value.exit_code == 1
        assert "configuration or local database unavailable" in output
        assert "secret" not in output


def test_start_blocks_critical_preflight_before_pid_or_server(
    harness_context, monkeypatch, capsys
):
    _, _, _, root = harness_context
    lifecycle = ServiceLifecycle(root / "data" / "harness.pid")
    monkeypatch.setattr(cli, "_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(
        cli,
        "_start_preflight",
        lambda *_args: [
            {"name": "Configuration", "status": "unavailable", "critical": True},
            {"name": "Provider local", "status": "unavailable", "critical": False},
        ],
    )
    served = []
    monkeypatch.setattr(cli, "_serve_api", lambda *_args: served.append(True))

    with pytest.raises(cli.typer.Exit) as error:
        cli.start()

    assert error.value.exit_code == 1
    assert not served
    assert not lifecycle.pidfile.exists()
    output = capsys.readouterr().out
    assert "Configuration: unavailable (required)" in output
    assert "Provider local: unavailable (optional)" in output


def test_start_preflight_reports_roles_without_secrets(harness_context, monkeypatch):
    cfg, store, orchestrator, _ = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    cfg.data["models"]["registry"] = {
        "primary-alias": {"provider": "alpha", "model": "private-model"},
        "fallback-alias": {"provider": "beta", "model": "fallback-model"},
    }
    cfg.data["profiles"] = {
        "planner": {
            "model": {
                "primary": "primary-alias",
                "fallback": ["", "fallback-alias", "ghost"],
            }
        }
    }
    cfg.data["models"]["providers"] = {
        "alpha": {"enabled": True, "api_key": "DO-NOT-PRINT"},
        "beta": {"enabled": True},
        "unused": {"enabled": True},
        "disabled": {"enabled": False},
        "ghost": {"enabled": True},
    }
    providers = {
        "alpha": SimpleNamespace(health=lambda: False),
        "beta": SimpleNamespace(health=lambda: True),
        "unused": SimpleNamespace(health=lambda: pytest.fail("unused provider probed")),
    }
    orchestrator.models.providers = providers
    orchestrator.qdrant.health = lambda: pytest.fail("disabled qdrant probed")
    monkeypatch.setattr(cli, "_config_health", lambda _conf: True)
    monkeypatch.setattr(cli, "_sqlite_health", lambda _store: "available")

    checks = cli._start_preflight(cfg, store, orchestrator)
    by_name = {check["name"]: check for check in checks}

    assert by_name["Provider alpha"]["role"] == "primary"
    assert by_name["Provider alpha"]["status"] == "unavailable"
    assert by_name["Provider alpha"]["critical"] is False
    assert by_name["Provider beta"]["role"] == "fallback"
    assert by_name["Provider beta"]["status"] == "available"
    assert by_name["Provider unused"]["status"] == "unused"
    assert by_name["Provider disabled"]["status"] == "disabled"
    assert by_name["Provider ghost"]["status"] == "unavailable"
    assert by_name["Qdrant"]["status"] == "disabled"
    assert all("DO-NOT-PRINT" not in str(check) for check in checks)


def test_start_preflight_handles_invalid_critical_and_optional_failures(
    harness_context, monkeypatch
):
    cfg, store, orchestrator, _ = harness_context
    cfg.data["profiles"] = {"planner": {"model": {"primary": "alpha"}}}
    cfg.data["models"]["providers"] = {"alpha": {"enabled": True}}
    orchestrator.models.providers = {
        "alpha": SimpleNamespace(
            health=lambda: (_ for _ in ()).throw(RuntimeError("secret detail"))
        )
    }
    cfg.data["memory"]["qdrant"]["enabled"] = True
    orchestrator.qdrant.health = lambda: False
    monkeypatch.setattr(cli, "_config_health", lambda _conf: False)
    monkeypatch.setattr(cli, "_sqlite_health", lambda _store: "unhealthy")

    checks = cli._start_preflight(cfg, store, orchestrator)
    by_name = {check["name"]: check for check in checks}

    assert by_name["Configuration"]["critical"] is True
    assert by_name["Configuration"]["status"] == "unavailable"
    assert by_name["SQLite"]["critical"] is True
    assert by_name["Provider alpha"]["status"] == "unavailable"
    assert by_name["Qdrant"]["critical"] is False
    assert "secret detail" not in str(checks)


def test_start_preflight_marks_provider_probe_timeout(harness_context, monkeypatch):
    cfg, store, orchestrator, _ = harness_context
    cfg.data["profiles"] = {"planner": {"model": {"primary": "alpha"}}}
    cfg.data["models"]["providers"] = {"alpha": {"enabled": True}}
    orchestrator.models.providers = {"alpha": SimpleNamespace(health=lambda: True)}
    monkeypatch.setattr(cli, "_config_health", lambda _conf: True)
    monkeypatch.setattr(cli, "_sqlite_health", lambda _store: "available")

    class PendingFuture:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    class Executor:
        def __init__(self, **_kwargs):
            self.future = PendingFuture()
            self.shutdown_called = False

        def submit(self, *_args):
            return self.future

        def shutdown(self, **_kwargs):
            self.shutdown_called = True

    executor = Executor()
    monkeypatch.setattr(cli, "ThreadPoolExecutor", lambda **_kwargs: executor)
    monkeypatch.setattr(cli, "wait", lambda futures, **_kwargs: (set(), set(futures)))

    checks = cli._start_preflight(cfg, store, orchestrator)

    assert (
        next(item for item in checks if item["name"] == "Provider alpha")["status"]
        == "unavailable"
    )
    assert executor.future.cancelled
    assert executor.shutdown_called


def test_stop_reports_shutdown_failure(monkeypatch, capsys):
    class FailingLifecycle:
        def stop(self):
            raise TimeoutError("shutdown timed out")

    monkeypatch.setattr(cli, "_lifecycle", FailingLifecycle)
    with pytest.raises(cli.typer.Exit):
        cli.stop()
    assert "shutdown failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("server_started", "health_ready", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_serve_api_reports_ready_only_after_startup_and_health(
    monkeypatch, capsys, server_started, health_ready, expected
):
    class FakeServer:
        def __init__(self, _config):
            self.started = False
            self.should_exit = False

        async def startup(self, _sockets=None):
            self.started = server_started

        def run(self):
            cli.asyncio.run(self.startup())

    class Lifecycle:
        def wait_until_ready(self, *_args):
            return health_ready

    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *_args, **_kwargs: object())

    result = cli._serve_api("127.0.0.1", 8080, Lifecycle(), timeout=0.1)

    assert result is expected
    assert ("Harness ready" in capsys.readouterr().out) is expected


def test_serve_api_stops_when_startup_times_out(monkeypatch):
    class FakeServer:
        def __init__(self, _config):
            self.started = False
            self.should_exit = False

        def run(self):
            return None

    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(cli.uvicorn, "Config", lambda *_args, **_kwargs: object())

    assert cli._serve_api("127.0.0.1", 8080, object(), timeout=0.001) is False


def test_health_helpers_are_fail_safe_and_sqlite_check_is_read_only(
    harness_context, monkeypatch
):
    _, store, _, _ = harness_context
    assert cli._component_health(lambda: True) == "available"
    assert cli._component_health(lambda: False) == "unavailable"
    assert cli._component_health(lambda: True, enabled=False) == "disabled"
    assert (
        cli._component_health(lambda: (_ for _ in ()).throw(RuntimeError()))
        == "unavailable"
    )
    assert cli._sqlite_health(store) == "available"
    assert (
        cli._sqlite_health(SimpleNamespace(db=store.db.with_name("missing.db")))
        == "missing"
    )
    store.db.write_text("not sqlite")
    assert cli._sqlite_health(store) == "unavailable"

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _query):
            return SimpleNamespace(fetchone=lambda: ("corrupt",))

        def close(self):
            return None

    monkeypatch.setattr(cli.sqlite3, "connect", lambda *_args, **_kwargs: Connection())
    assert cli._sqlite_health(store) == "unhealthy"
    assert cli._config_health(SimpleNamespace(validate=lambda: True)) is True
    assert (
        cli._config_health(
            SimpleNamespace(validate=lambda: (_ for _ in ()).throw(ValueError()))
        )
        is False
    )
    assert cli._provider_health({"z": SimpleNamespace(health=lambda: True)}) == {
        "z": "available"
    }


@pytest.mark.parametrize(
    ("result", "raise_error", "expected"),
    [
        ("ok", False, "available"),
        ("corrupt", False, "unhealthy"),
        (None, True, "unavailable"),
    ],
)
def test_sqlite_health_always_closes_read_only_connection(
    tmp_path, monkeypatch, result, raise_error, expected
):
    database = tmp_path / "health.db"
    database.touch()
    close_calls = []

    class Connection:
        def execute(self, statement):
            assert statement == "PRAGMA quick_check"
            if raise_error:
                raise cli.sqlite3.OperationalError("database detail")
            return SimpleNamespace(fetchone=lambda: (result,))

        def close(self):
            close_calls.append(True)

    connection = Connection()
    calls = []

    def connect(database_uri, **kwargs):
        calls.append((database_uri, kwargs))
        return connection

    monkeypatch.setattr(cli.sqlite3, "connect", connect)

    status = cli._sqlite_health(SimpleNamespace(db=database))

    assert calls
    assert status == expected
    assert close_calls == [True]
    assert calls[0][0].endswith("?mode=ro")
    assert calls[0][1] == {"uri": True, "timeout": 1}


def test_status_and_doctor(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    cli.status()
    assert "stopped" in capsys.readouterr().out

    class LifecycleFixture:
        def __init__(self):
            self.state = {"state": "running", "record": {"pid": 123}, "ready": True}

        def inspect(self):
            return self.state

    lifecycle = LifecycleFixture()
    monkeypatch.setattr(cli, "_lifecycle", lambda: lifecycle)
    cli.status()
    assert "running (ready) pid=123" in capsys.readouterr().out
    lifecycle.state = {"state": "invalid_record", "record": None, "ready": False}
    cli.status()
    assert "invalid_record" in capsys.readouterr().out
    lifecycle.state = {"state": "running", "record": {"pid": 123}, "ready": True}
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli.shutil, "which", lambda x: "/usr/bin/" + x)
    cfg.path("logs").mkdir(exist_ok=True)
    cfg.data["memory"] = {"qdrant": {"enabled": True}}
    orchestrator.qdrant.health = lambda: True
    orchestrator.qdrant.health_report = lambda: {
        "healthy": True,
        "service": "available",
        "collection_exists": True,
    }
    cli.status()
    status_output = capsys.readouterr().out
    assert "Qdrant evidence:" in status_output
    orchestrator.models.providers.clear()
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    docker_actions = []
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "run",
        lambda _self, action: (
            docker_actions.append(action) or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.setattr(
        cli.socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(OSError()),
    )
    cli.doctor()
    assert "✓" in capsys.readouterr().out


def test_status_and_doctor_fail_safe_diagnostics(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, _ = harness_context
    monkeypatch.setattr(
        cli,
        "_lifecycle",
        lambda: SimpleNamespace(
            inspect=lambda: (_ for _ in ()).throw(RuntimeError("bad pid record"))
        ),
    )
    cli.status()
    assert "unknown" in capsys.readouterr().out
    cfg.data["memory"] = {"qdrant": {"enabled": True}}
    orchestrator.models.providers.clear()
    orchestrator.qdrant.health = lambda: False
    orchestrator.qdrant.health_report = lambda: {
        "healthy": False,
        "service": "unavailable",
        "collection_exists": False,
        "errors": ["offline"],
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )

    class OpenSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        cli.socket, "create_connection", lambda *_args, **_kwargs: OpenSocket()
    )
    with pytest.raises(cli.typer.Exit):
        cli.doctor()
    output = capsys.readouterr().out
    assert "Docker Compose" in output
    assert "API Port Available" in output


def test_doctor_handles_missing_docker_and_broker_failure(harness_context, monkeypatch):
    cfg, _, orchestrator, _ = harness_context
    cfg.data["memory"] = {"qdrant": {"enabled": True}}
    orchestrator.models.providers.clear()
    orchestrator.qdrant.health = lambda: False
    orchestrator.qdrant.health_report = lambda: {
        "healthy": False,
        "service": "unavailable",
        "collection_exists": False,
        "errors": ["offline"],
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: None if name == "docker" else "/usr/bin/git"
    )
    monkeypatch.setattr(
        cli,
        "_lifecycle",
        lambda: SimpleNamespace(
            inspect=lambda: {"state": "stopped", "record": None, "ready": False}
        ),
    )
    with pytest.raises(cli.typer.Exit):
        cli.doctor()

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("daemon down")),
    )
    monkeypatch.setattr(
        cli.socket,
        "create_connection",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(cli.typer.Exit):
        cli.doctor()


def test_doctor_skips_optional_docker_checks_when_qdrant_is_disabled(
    harness_context, monkeypatch
):
    cfg, _, orchestrator, _ = harness_context
    cfg.data["memory"] = {"qdrant": {"enabled": False}}
    cfg.path("logs").mkdir(exist_ok=True)
    orchestrator.models.providers.clear()
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        cli,
        "_lifecycle",
        lambda: SimpleNamespace(
            inspect=lambda: {"state": "stopped", "record": None, "ready": False}
        ),
    )
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1)
    )
    monkeypatch.setattr(
        cli.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    cli.doctor()


def test_model_usage_outputs_report_and_validation_error(
    harness_context, monkeypatch, capsys
):
    _, store, _orchestrator, _ = harness_context
    monkeypatch.setattr(store.model_usage, "report", lambda **_kwargs: {"count": 1})
    cli.model_usage()
    assert '"count": 1' in capsys.readouterr().out
    monkeypatch.setattr(
        store.model_usage,
        "report",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad range")),
    )
    with pytest.raises(cli.typer.Exit):
        cli.model_usage()
    assert "bad range" in capsys.readouterr().out


def test_qdrant_smoke_requires_explicit_confirmation(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "live_smoke_test",
        lambda _self: (
            calls.append("smoke")
            or {
                "healthy": True,
                "persisted_after_restart": True,
                "cleaned": True,
            }
        ),
    )
    with pytest.raises(cli.typer.Exit):
        cli.qdrant_smoke(confirm=False)
    assert calls == []
    assert "No action taken" in capsys.readouterr().out
    cli.qdrant_smoke(confirm=True)
    assert calls == ["smoke"]
    assert "persisted_after_restart=True" in capsys.readouterr().out


def test_qdrant_smoke_reports_daemon_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "live_smoke_test",
        lambda _self: (_ for _ in ()).throw(PermissionError("daemon unavailable")),
    )
    with pytest.raises(cli.typer.Exit):
        cli.qdrant_smoke(confirm=True)
    assert "daemon unavailable" in capsys.readouterr().out


def test_qdrant_report_is_fail_safe_and_does_not_probe_when_disabled():
    orchestrator = SimpleNamespace(
        qdrant=SimpleNamespace(
            health_report=lambda: (_ for _ in ()).throw(RuntimeError("secret detail"))
        )
    )
    assert cli._qdrant_report(orchestrator, enabled=False)["service"] == "disabled"
    assert cli._qdrant_report(orchestrator, enabled=True)["errors"] == [
        "qdrant_probe_failed"
    ]
    orchestrator.qdrant.health_report = lambda: "invalid"
    assert cli._qdrant_report(orchestrator, enabled=True)["errors"] == [
        "invalid_health_report"
    ]


def test_task_commands(harness_context, capsys):
    _, store, _, _ = harness_context
    cli.task_create("new", "description")
    task_id = store.tasks.list()[0]["id"]
    store.event(task_id, "task.created", {})
    cli.task_list()
    cli.task_show(task_id)
    cli.task_events(task_id)
    cli.task_run(task_id)
    assert "new" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit):
        cli.task_show(99999)


def test_task_abort_command_serializes_service_result(harness_context, capsys):
    _, store, orchestrator, _ = harness_context
    task = store.create(Task(title="abort serialization"))
    orchestrator.service.abort = lambda task_id: store.get(task_id)
    cli.task_abort(task.id)
    assert '"title":"abort serialization"' in capsys.readouterr().out


def test_model_and_config_commands(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, _ = harness_context
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    orchestrator.models.providers.clear()
    cfg.data["models"]["registry"] = {
        "configured": {"provider": "local", "model": "fixture", "tier": "local"}
    }
    cli.model_list()
    orchestrator.models.register(
        "remote",
        type(
            "P",
            (),
            {"health": lambda self: False, "models": lambda self: []},
        )(),
    )
    cli.model_list()
    cli.model_status()
    cli.config_show()
    cli.config_resolved()
    cli.config_validate()
    assert "configuration valid" in capsys.readouterr().out
    cfg.data.pop("secrets", None)
    assert cli._safe_config() == cfg.data


def test_config_show_is_source_view_and_resolved_adds_defaults(
    harness_context, monkeypatch, capsys
):
    cfg, _, _, _ = harness_context
    cfg.data["secrets"]["OPENAI_API_KEY"] = "sensitive-value"
    cfg.configured = {
        "harness": {"name": "fixture"},
        "secrets": {"OPENAI_API_KEY": "sensitive-value"},
    }
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    cli.config_show()
    shown = capsys.readouterr().out
    assert "sensitive-value" not in shown
    assert "********" in shown
    assert "name: fixture" in shown and "port: 8080" not in shown
    cli.config_resolved()
    resolved = capsys.readouterr().out
    assert "api:" in resolved and "port: 8080" in resolved
    assert "sensitive-value" not in resolved


def test_model_discovery_and_status_use_provider_contracts(harness_context, capsys):
    _, _, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()

    class Provider:
        name = "local"
        model = "fixture-v1"

        def health(self):
            return True

        def models(self):
            return ["fixture-v1", "fixture-v2", None, ""]

    orchestrator.models.register("local", Provider())
    orchestrator.models.models = {
        "coding-model": {
            "provider": "local",
            "model": "fixture-v1",
            "capabilities": ["tools"],
        }
    }
    cli.model_list()
    output = capsys.readouterr().out
    assert "local\tavailable" in output
    assert "fixture-v2" in output
    assert "fixture-v1\tlocal\tcoding-model" in output
    assert orchestrator.models.discovered["local"] == ("fixture-v1", "fixture-v2")
    cli.status()
    assert "Provider local: available" in capsys.readouterr().out


def test_model_test_runs_configured_model_and_fails_closed(harness_context, capsys):
    _, _, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()

    class Provider:
        model = "fixture-v1"

        def complete(self, prompt, model=None):
            assert prompt == "Reply with exactly: OK"
            assert model == "fixture-v1"
            return "OK"

    orchestrator.models.register("local", Provider())
    orchestrator.models.models = {
        "coding-model": {"provider": "local", "model": "fixture-v1"}
    }
    cli.model_test("coding-model")
    assert "coding-model" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit):
        cli.model_test("not-configured")
    assert "Model test failed (ValueError)" in capsys.readouterr().out


def test_model_discovery_failure_is_reported_without_aborting(harness_context, capsys):
    _, _, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()

    class Unreachable:
        def health(self):
            raise RuntimeError("offline")

        def models(self):
            raise RuntimeError("offline")

    orchestrator.models.register("offline", Unreachable())
    cli.model_list()
    output = capsys.readouterr().out
    assert "offline\tunavailable" in output
    assert "offline\t<discovery failed>" in output
    cli.status()
    assert "Provider offline: unavailable" in capsys.readouterr().out


def test_model_inventory_shows_tier_and_unavailable_configured_model(
    harness_context, capsys
):
    _cfg, _store, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()
    provider = SimpleNamespace(
        health=lambda: True, models=lambda: ["live-v1", "unlisted-v2"]
    )
    orchestrator.models.register("local", provider)
    orchestrator.models.models = {
        "coding": {"provider": "local", "model": "live-v1", "tier": "advanced"},
        "fallback": {"provider": "local", "model": "missing-v1", "tier": "local"},
        "untiered": {"provider": "local", "model": "legacy-v1"},
    }
    cli.model_list()
    listed = capsys.readouterr().out
    assert "live-v1\tlocal\tcoding\tadvanced\tavailable" in listed
    assert "missing-v1\tlocal\tfallback\tlocal\tunavailable" in listed
    assert "legacy-v1\tlocal\tuntiered\tunknown\tunavailable" in listed
    assert "unlisted-v2\tlocal\t-\tunknown\tavailable" in listed
    cli.model_status()
    assert capsys.readouterr().out == listed


def test_model_inventory_reports_unknown_provider_without_secret_or_prompt(
    harness_context, capsys
):
    _cfg, _store, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()
    orchestrator.models.models = {
        "remote-model": {"provider": "absent", "model": "vendor-v1", "tier": "premium"}
    }
    cli.model_list()
    output = capsys.readouterr().out
    assert "vendor-v1\tabsent\tremote-model\tpremium\tunavailable" in output


def test_lmstudio_health_report_drives_inventory_status_and_preflight(
    harness_context, capsys, monkeypatch
):
    cfg, store, orchestrator, _ = harness_context
    cfg.data["models"]["providers"] = {"local": {"enabled": True}}
    orchestrator.models.providers.clear()
    report = ProviderHealth(
        "lmstudio", True, True, ("loaded", "downloaded"), ("loaded",)
    )
    probes = []
    orchestrator.models.register(
        "local",
        SimpleNamespace(
            health_report=lambda: probes.append(True) or report,
            health=lambda: pytest.fail("legacy health called"),
            models=lambda: pytest.fail("duplicate discovery called"),
        ),
    )
    orchestrator.models.models = {
        "chosen": {"provider": "local", "model": "loaded", "tier": "local"},
        "cold": {"provider": "local", "model": "downloaded", "tier": "local"},
    }
    cfg.data["models"]["registry"] = orchestrator.models.models
    cfg.data["profiles"] = {"planner": {"model": {"primary": "chosen"}}}
    cli.model_list()
    output = capsys.readouterr().out
    assert "local\tavailable" in output
    assert "loaded\tlocal\tchosen\tlocal\tavailable" in output
    assert "downloaded\tlocal\tcold\tlocal\tunavailable" in output
    assert orchestrator.models.discovered["local"] == ("downloaded", "loaded")
    assert cli._provider_health(orchestrator.models.providers) == {"local": "available"}
    monkeypatch.setattr(cli, "_config_health", lambda _conf: True)
    monkeypatch.setattr(cli, "_sqlite_health", lambda _store: "available")
    checks = cli._start_preflight(cfg, store, orchestrator)
    assert (
        next(c for c in checks if c["name"] == "Provider local")["status"]
        == "available"
    )
    cli.status()
    assert "Provider local: available" in capsys.readouterr().out
    assert len(probes) == 4
    report = ProviderHealth("lmstudio", True, True, ("loaded",), ())
    assert cli._provider_health(orchestrator.models.providers) == {
        "local": "not_loaded"
    }
    checks = cli._start_preflight(cfg, store, orchestrator)
    assert (
        next(c for c in checks if c["name"] == "Provider local")["status"]
        == "not_loaded"
    )
    cli.model_status()
    assert "loaded\tlocal\tchosen\tlocal\tunavailable" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit):
        cli.doctor()
    assert "Provider local status: not_loaded" in capsys.readouterr().out
    report = ProviderHealth("lmstudio", True, False)
    cli.model_list()
    assert "local\t<discovery failed>" in capsys.readouterr().out


def test_failed_health_report_is_redacted(harness_context, capsys):
    _cfg, _store, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()
    orchestrator.models.register(
        "local",
        SimpleNamespace(
            health_report=lambda: (_ for _ in ()).throw(RuntimeError("private")),
            models=lambda: ["downloaded"],
        ),
    )
    cli.model_list()
    output = capsys.readouterr().out
    assert "local\tunavailable" in output
    assert "private" not in output


def test_cli_command_groups_match_prompt_contract():
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "start",
        "stop",
        "status",
        "doctor",
        "config",
        "models",
        "tasks",
        "secrets",
    ):
        assert command in result.output


def test_model_test_accepts_usage_tuple_and_rejects_empty_response(
    harness_context, capsys
):
    _, _, orchestrator, _ = harness_context
    orchestrator.models.providers.clear()

    class Provider:
        def __init__(self, response):
            self.response = response

        def complete(self, prompt, model=None):
            return self.response

    orchestrator.models.register("local", Provider(("OK", object())))
    orchestrator.models.models = {
        "coding-model": {"provider": "local", "model": "fixture-v1"}
    }
    cli.model_test("coding-model")
    assert "successful" in capsys.readouterr().out
    orchestrator.models.providers["local"] = Provider("  ")
    with pytest.raises(cli.typer.Exit):
        cli.model_test("coding-model")
    assert "ValueError" in capsys.readouterr().out


def test_secret_commands(harness_context, monkeypatch, capsys):
    names = SimpleNamespace(
        list_names=lambda: ["TOKEN"],
        set=lambda *a: None,
        delete=lambda _: True,
        exists=lambda _: True,
    )
    monkeypatch.setattr(cli, "SecretResolver", lambda: names)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "value")
    cli.secrets_list()
    cli.secret_set("TOKEN")
    cli.secret_delete("TOKEN")
    assert "TOKEN" in capsys.readouterr().out
    names.delete = lambda _: False
    with pytest.raises(cli.typer.Exit):
        cli.secret_delete("missing")


def test_secret_exists_cli_reports_only_boolean_status(monkeypatch):
    from typer.testing import CliRunner

    resolver = SimpleNamespace(exists=lambda _name: True)
    monkeypatch.setattr(cli, "SecretResolver", lambda: resolver)
    runner = CliRunner()
    found = runner.invoke(cli.app, ["secrets", "exists", "TOKEN"])
    assert found.exit_code == 0
    assert found.output.strip() == "exists"
    resolver.exists = lambda _name: False
    absent = runner.invoke(cli.app, ["secrets", "exists", "TOKEN"])
    assert absent.exit_code == 1
    assert absent.output.strip() == "not found"


def test_secret_cli_errors_do_not_render_exception_or_secret(monkeypatch):
    from typer.testing import CliRunner

    secret_value = "never-print-this-secret"
    resolver = SimpleNamespace(
        list_names=lambda: (_ for _ in ()).throw(RuntimeError(secret_value)),
        set=lambda *_: (_ for _ in ()).throw(RuntimeError(secret_value)),
        delete=lambda _: (_ for _ in ()).throw(RuntimeError(secret_value)),
        exists=lambda _: (_ for _ in ()).throw(RuntimeError(secret_value)),
    )
    monkeypatch.setattr(cli, "SecretResolver", lambda: resolver)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: secret_value)
    runner = CliRunner()
    for args in (
        ["secrets", "list"],
        ["secrets", "set", "TOKEN"],
        ["secrets", "delete", "TOKEN"],
        ["secrets", "exists", "TOKEN"],
    ):
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 1
        assert "secret operation failed" in result.output
        assert secret_value not in result.output


def test_task_events_cli_redacts_existing_event_payloads(
    harness_context, monkeypatch, capsys
):
    _config, store, orchestrator, _root = harness_context
    canary = "SS1-CLI-CANARY"
    store.audit.secrets["SS1_CLI_TOKEN"] = canary
    task = store.tasks.create("legacy event")
    store.events.append(task, "task.failed", {"message": canary, "apiKey": "other"})
    monkeypatch.setattr(cli, "build", lambda: (None, store, orchestrator))

    cli.task_events(task)

    output = capsys.readouterr().out
    assert canary not in output
    assert "other" not in output
    assert "[REDACTED]" in output


def test_task_events_cli_reports_missing_task(harness_context, capsys):
    with pytest.raises(cli.typer.Exit) as result:
        cli.task_events(999999)
    assert result.value.exit_code == 1
    assert capsys.readouterr().out == "task not found\n"


def test_memory_sync_command_projects_sqlite_decisions(harness_context):
    _cfg, store, _orchestrator, root = harness_context
    from typer.testing import CliRunner

    store.decisions.save(None, "Use SQLite", "Canonical persistence")

    result = CliRunner().invoke(cli.app, ["memory", "sync"])

    assert result.exit_code == 0, result.output
    assert "created=1" in result.output
    assert (root / "vault" / "_harness" / "decisions" / "1.md").read_text().find(
        "Use SQLite"
    ) >= 0


def test_task_create_accepts_structured_yaml_spec(harness_context, tmp_path):
    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    spec = tmp_path / "task.yaml"
    spec.write_text(
        "title: Structured task\ngoal: Build feature\nrequirements: [Use tests]\nacceptance_criteria:\n  - {id: done, description: Feature works}\n"
    )
    result = CliRunner().invoke(cli.app, ["tasks", "create", "--file", str(spec)])
    assert result.exit_code == 0, result.stdout
    item = store.get(1)
    assert item.title == "Structured task"
    assert item.goal == "Build feature"
    assert item.requirements == ["Use tests"]


@pytest.mark.parametrize(
    "contents",
    [
        "{bad-json",
        "- not-a-mapping",
        "title: x\nstatus: completed",
        "title: x\nunknown: private-secret",
    ],
)
def test_task_create_rejects_invalid_or_protected_spec(
    harness_context, tmp_path, contents
):
    from typer.testing import CliRunner

    spec = tmp_path / "task.yaml"
    spec.write_text(contents)
    result = CliRunner().invoke(cli.app, ["tasks", "create", "--file", str(spec)])
    assert result.exit_code != 0
    assert "private-secret" not in result.stdout


def test_task_watch_follows_sse_and_resumes_cursor(harness_context, monkeypatch):
    from contextlib import contextmanager

    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    task_id = store.create(Task(title="Watching")).id
    cursors = []

    @contextmanager
    def stream(_method, _url, **kwargs):
        cursors.append(kwargs["headers"]["Last-Event-ID"])
        if len(cursors) == 1:
            lines = [
                "id: 5",
                "event: task.started",
                f"data: {json.dumps({'id': 5, 'task_id': task_id, 'kind': 'task.started', 'created_at': '09:31:01', 'payload': {}})}",
                "",
            ]
        else:
            lines = [
                ": keep-alive",
                "",
                "id: 6",
                "event: task.completed",
                f"data: {json.dumps({'id': 6, 'task_id': task_id, 'kind': 'task.completed', 'created_at': '09:31:03', 'payload': {}})}",
                "",
            ]
        yield SimpleNamespace(
            raise_for_status=lambda: None, iter_lines=lambda: iter(lines)
        )

    monkeypatch.setattr(cli.httpx, "stream", stream)
    result = CliRunner().invoke(cli.app, ["tasks", "watch", str(task_id)])
    assert result.exit_code == 0, result.stdout
    assert "task.started" in result.stdout and "task.completed" in result.stdout
    assert cursors == ["0", "5"]


def test_task_create_json_and_conflicting_options(harness_context, tmp_path):
    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    spec = tmp_path / "task.json"
    spec.write_text(json.dumps({"title": "JSON task", "requirements": ["Checked"]}))
    runner = CliRunner()
    created = runner.invoke(cli.app, ["tasks", "create", "--file", str(spec)])
    assert created.exit_code == 0, created.output
    assert store.get(1).requirements == ["Checked"]
    conflict = runner.invoke(cli.app, ["tasks", "create", "title", "--file", str(spec)])
    assert conflict.exit_code == 1
    assert store.get(2) is None
    missing_title = runner.invoke(cli.app, ["tasks", "create"])
    assert missing_title.exit_code == 1
    assert "title is required" in missing_title.output


@pytest.mark.parametrize(
    "name,contents",
    [
        ("task.txt", "title: Unsupported"),
        ("task.json", "{broken"),
        ("task.yaml", "title: x\nstatus: completed"),
    ],
)
def test_task_create_spec_errors_are_redacted(
    harness_context, tmp_path, name, contents
):
    from typer.testing import CliRunner

    spec = tmp_path / name
    spec.write_text(contents)
    result = CliRunner().invoke(cli.app, ["tasks", "create", "--file", str(spec)])
    assert result.exit_code == 1
    assert result.output.strip() == "invalid task specification"


def test_task_create_rejects_oversized_or_missing_file(harness_context, tmp_path):
    from typer.testing import CliRunner

    spec = tmp_path / "large.yaml"
    spec.write_text("x" * 1_000_001)
    runner = CliRunner()
    for path in (spec, tmp_path / "missing.yaml"):
        result = runner.invoke(cli.app, ["tasks", "create", "--file", str(path)])
        assert result.exit_code == 1
        assert result.output.strip() == "invalid task specification"


def test_task_watch_ignores_replayed_events_and_stops_on_cancel(
    harness_context, monkeypatch
):
    from contextlib import contextmanager

    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    task_id = store.create(Task(title="Watching")).id
    calls = []

    @contextmanager
    def stream(_method, url, **kwargs):
        calls.append((url, kwargs["headers"]["Last-Event-ID"]))
        if len(calls) == 1:
            rows = [(4, "task.started", {})]
        else:
            rows = [
                (4, "task.started", {}),
                (5, "task.status", {"status": "cancelled"}),
            ]
        lines = []
        for event_id, kind, payload in rows:
            event = {
                "id": event_id,
                "task_id": task_id,
                "kind": kind,
                "payload": payload,
            }
            lines.extend([f"id: {event_id}", f"data: {json.dumps(event)}", ""])
        yield SimpleNamespace(
            raise_for_status=lambda: None, iter_lines=lambda: iter(lines)
        )

    monkeypatch.setattr(cli.httpx, "stream", stream)
    result = CliRunner().invoke(cli.app, ["tasks", "watch", str(task_id)])
    assert result.exit_code == 0, result.output
    assert result.output.count("task.started") == 1
    assert "task.status" in result.output
    assert calls == [
        ("http://127.0.0.1:8080/api/events/stream", "0"),
        ("http://127.0.0.1:8080/api/events/stream", "4"),
    ]


def test_task_watch_missing_and_disconnected(harness_context, monkeypatch):
    from contextlib import contextmanager

    from typer.testing import CliRunner

    runner = CliRunner()
    missing = runner.invoke(cli.app, ["tasks", "watch", "999"])
    assert missing.exit_code == 1
    assert "task not found" in missing.output
    _cfg, store, _orchestrator, _ = harness_context
    task_id = store.create(Task(title="Watching")).id

    @contextmanager
    def disconnected(*_args, **_kwargs):
        raise cli.httpx.ConnectError("private server error")
        yield

    monkeypatch.setattr(cli.httpx, "stream", disconnected)
    result = runner.invoke(cli.app, ["tasks", "watch", str(task_id)])
    assert result.exit_code == 1
    assert result.output.count("TASK-") == 1
    assert "task event stream unavailable" in result.output
    assert "private server error" not in result.output


def test_task_create_service_error_is_redacted(harness_context, monkeypatch):
    from typer.testing import CliRunner

    _cfg, _store, orchestrator, _ = harness_context
    monkeypatch.setattr(
        orchestrator.service,
        "create",
        lambda _item: (_ for _ in ()).throw(ValueError("private service detail")),
    )
    result = CliRunner().invoke(cli.app, ["tasks", "create", "a title"])
    assert result.exit_code == 1
    assert result.output.strip() == "task creation failed"


def test_task_watch_interrupt_maps_to_exit_130(harness_context, monkeypatch):
    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    task_id = store.create(Task(title="Watching")).id
    monkeypatch.setattr(
        cli,
        "_watch_task_events",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    result = CliRunner().invoke(cli.app, ["tasks", "watch", str(task_id)])
    assert result.exit_code == 130


@pytest.mark.parametrize(
    "lines",
    [
        ["id: nope", ""],
        ["id: 1", 'data: {"id": 1, "task_id": 1}', ""],
    ],
)
def test_task_watch_rejects_malformed_sse(harness_context, monkeypatch, lines):
    from contextlib import contextmanager

    from typer.testing import CliRunner

    _cfg, store, _orchestrator, _ = harness_context
    task_id = store.create(Task(title="Watching")).id
    attempts = []

    @contextmanager
    def stream(*_args, **_kwargs):
        attempts.append(True)
        yield SimpleNamespace(
            raise_for_status=lambda: None, iter_lines=lambda: iter(lines)
        )

    monkeypatch.setattr(cli.httpx, "stream", stream)
    result = CliRunner().invoke(cli.app, ["tasks", "watch", str(task_id)])
    assert result.exit_code == 1
    assert len(attempts) == 3
    assert result.output.endswith("task event stream unavailable\n")


def test_memory_status_reports_explicit_mcp_opt_in(harness_context, capsys):
    cfg, _store, _orchestrator, _root = harness_context
    cfg.data["memory"]["obsidian"]["enabled"] = True
    cfg.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False

    cli.memory_status()

    report = json.loads(capsys.readouterr().out)
    assert report["obsidian_enabled"] is True
    assert report["mcp_enabled"] is False
    assert report["markdown_notes"] == 0


def test_memory_status_does_not_create_a_missing_vault(harness_context, capsys):
    cfg, _store, _orchestrator, _root = harness_context
    vault = cfg.path("obsidian_vault")
    vault.rmdir()

    cli.memory_status()

    report = json.loads(capsys.readouterr().out)
    assert report["exists"] is False
    assert not vault.exists()


def test_memory_audit_reports_read_only_health(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "vault"
    vault.mkdir()
    for relative in (
        "Willkommen.md",
        "Vault-Übersicht.md",
        "rules/Harness-Prinzipien.md",
        "architecture/Systemarchitektur.md",
        "architecture/Vault und Memory.md",
        "decisions/Entscheidungsregister.md",
        "agents/Agentenprofile.md",
        "tasks/Task-Register.md",
    ):
        path = vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\nlast_reviewed: {datetime.now(UTC).date().isoformat()}\n---\n# Note",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        cli, "Config", lambda: SimpleNamespace(path=lambda _name: vault)
    )

    cli.memory_audit()

    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["audited_notes"] == 8
    assert all(
        (vault / relative).is_file()
        for relative in (
            "Willkommen.md",
            "Vault-Übersicht.md",
            "rules/Harness-Prinzipien.md",
            "architecture/Systemarchitektur.md",
            "architecture/Vault und Memory.md",
            "decisions/Entscheidungsregister.md",
            "agents/Agentenprofile.md",
            "tasks/Task-Register.md",
        )
    )


def test_memory_audit_fails_closed_for_missing_vault(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "missing-vault"
    monkeypatch.setattr(
        cli, "Config", lambda: SimpleNamespace(path=lambda _name: vault)
    )

    with pytest.raises(cli.typer.Exit) as exc:
        cli.memory_audit()

    report = json.loads(capsys.readouterr().out)
    assert exc.value.exit_code == 1
    assert report["healthy"] is False
    assert not vault.exists()


def test_memory_audit_reports_configuration_error(monkeypatch, capsys):
    def unavailable_config():
        raise OSError("private configuration detail")

    monkeypatch.setattr(cli, "Config", unavailable_config)
    with pytest.raises(cli.typer.Exit) as exc:
        cli.memory_audit()
    assert exc.value.exit_code == 2
    assert capsys.readouterr().out == "Vault audit unavailable: OSError\n"


def test_qdrant_status_reports_disabled_service_and_exits_on_bad_health(
    harness_context, monkeypatch, capsys
):
    _cfg, _store, orchestrator, _root = harness_context
    _cfg.data["memory"]["qdrant"]["enabled"] = False
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": False, "service": "unavailable", "errors": ["offline"]},
    )
    cli.qdrant_status()
    report = json.loads(capsys.readouterr().out)
    assert report["enabled"] is False
    assert report["service"] == "disabled"
    assert report["healthy"] is True

    _cfg.data["memory"]["qdrant"]["enabled"] = True
    with pytest.raises(cli.typer.Exit):
        cli.qdrant_status()
    assert json.loads(capsys.readouterr().out)["errors"] == ["offline"]


def test_qdrant_reconcile_requires_confirmation_and_enabled_configuration(
    harness_context, monkeypatch, capsys
):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    calls = []
    monkeypatch.setattr(
        orchestrator.memory_service,
        "reconcile",
        lambda: calls.append(True) or {"notes": 2, "chunks": 3, "removed_points": 1},
    )
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_reconcile(confirm=False)
    assert result.value.exit_code == 2
    assert not calls
    assert "No action taken" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_reconcile(confirm=True)
    assert result.value.exit_code == 1
    assert not calls
    capsys.readouterr()
    cfg.data["memory"]["qdrant"]["enabled"] = True
    cli.qdrant_reconcile(confirm=True)
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["removed_points"] == 1


def test_qdrant_reconcile_reports_only_error_type(harness_context, monkeypatch, capsys):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(
        orchestrator.memory_service,
        "reconcile",
        lambda: (_ for _ in ()).throw(RuntimeError("secret-bearing detail")),
    )
    with pytest.raises(cli.typer.Exit):
        cli.qdrant_reconcile(confirm=True)
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "secret-bearing detail" not in output


def test_qdrant_init_requires_confirmation_and_enabled_service(
    harness_context, monkeypatch, capsys
):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    calls = []
    monkeypatch.setattr(cli, "build", lambda: (cfg, _store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant, "ensure_collection", lambda: calls.append(True)
    )

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_init(confirm=False)
    assert result.value.exit_code == 2
    assert not calls
    assert "No action taken" in capsys.readouterr().out

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_init(confirm=True)
    assert result.value.exit_code == 1
    assert not calls
    capsys.readouterr()

    cfg.data["memory"]["qdrant"]["enabled"] = True
    cli.qdrant_init(confirm=True)
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["initialized"] is True


def test_qdrant_init_reports_only_error_type(harness_context, monkeypatch, capsys):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(cli, "build", lambda: (cfg, _store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant,
        "ensure_collection",
        lambda: (_ for _ in ()).throw(RuntimeError("private response body")),
    )

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_init(confirm=True)
    assert result.value.exit_code == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "private response body" not in output


def test_qdrant_search_is_internal_and_requires_enabled_service(
    harness_context, monkeypatch, capsys
):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    calls = []
    monkeypatch.setattr(cli, "build", lambda: (cfg, _store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda query, limit: calls.append((query, limit)) or [{"id": "point-1"}],
    )

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_search("query", limit=3)
    assert result.value.exit_code == 1
    assert not calls
    capsys.readouterr()
    cfg.data["memory"]["qdrant"]["enabled"] = True
    cli.qdrant_search("query", limit=3)
    assert calls == [("query", 3)]
    assert json.loads(capsys.readouterr().out) == [{"id": "point-1"}]


@pytest.mark.parametrize(
    ("query", "limit"), [(" ", 5), ("query", 0), ("query", 21), ("query", True)]
)
def test_qdrant_search_rejects_invalid_query_and_limit(
    harness_context, query, limit, capsys
):
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_search(query, limit=limit)
    assert result.value.exit_code == 2
    assert capsys.readouterr().out


def test_qdrant_search_reports_only_error_type(harness_context, monkeypatch, capsys):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(cli, "build", lambda: (cfg, _store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private response body")
        ),
    )

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_search("query")
    assert result.value.exit_code == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "private response body" not in output


def test_metrics_command_prints_structured_durable_counters(harness_context, capsys):
    _cfg, store, _orchestrator, _root = harness_context
    task = store.create(Task(title="metrics"))
    store.event(task.id, "task.created", {})

    cli.runtime_metrics()

    report = json.loads(capsys.readouterr().out)
    assert report["tasks"]["by_status"]["pending"] == 1
    assert report["events"]["total"] == 1


def test_completion_command_reports_gap_count_and_returns_incomplete(
    harness_context, capsys
):
    import typer

    _cfg, _store, _orchestrator, root = harness_context
    lines = ["| Nr. | Name | Status | Tiefe |\n|---:|---|---|---|\n"]
    lines.extend(
        f"| {number} | item | {'Offen' if number == 2 else 'Erfüllt'} | status |\n"
        for number in range(1, 110)
    )
    (root / "GAP_MATRIX.md").write_text("".join(lines))

    with pytest.raises(typer.Exit) as error:
        cli.completion_status()

    assert error.value.exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["remaining"] == [2]
    assert report["complete"] is False


def test_completion_command_accepts_a_fully_satisfied_matrix(harness_context, capsys):
    _cfg, _store, _orchestrator, root = harness_context
    rows = "".join(
        f"| {number} | item | Erfüllt | evidence |\n" for number in range(1, 110)
    )
    matrix_path = root / "GAP_MATRIX.md"
    matrix_path.write_text(
        "| Nr. | Name | Status | Tiefe |\n|---:|---|---|---|\n" + rows
    )
    from test_completion import coverage_report

    from harness.verification import write_verification_report

    coverage_path = root / "coverage.json"
    coverage_path.write_text(json.dumps(coverage_report()))
    junit_path = root / "junit.xml"
    junit_path.write_text('<testsuite tests="1"><testcase name="ok" /></testsuite>')
    write_verification_report(
        matrix_path, coverage_path, junit_path, root / "data" / "verification.json"
    )

    cli.completion_status()

    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_completion_command_fails_closed_on_invalid_matrix(harness_context, capsys):
    import typer

    _cfg, _store, _orchestrator, root = harness_context
    (root / "GAP_MATRIX.md").write_text("not a matrix")

    with pytest.raises(typer.Exit) as error:
        cli.completion_status()

    assert error.value.exit_code == 2
    assert "unverifiable" in capsys.readouterr().out
