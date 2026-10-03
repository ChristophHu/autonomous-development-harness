import hashlib
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
    cfg.data.setdefault("tools", {})["mcp"] = {
        "servers": {
            "files": {"enabled": False, "builtin": "filesystem"},
            "vault": {"enabled": False, "builtin": "obsidian"},
        }
    }
    store = Store(cfg)
    orchestrator = Orchestrator(store, cfg)
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    monkeypatch.setattr(
        cli,
        "_build_embedding_acceptance",
        lambda: (cfg, store, orchestrator.qdrant, orchestrator.memory_service),
    )
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

        def execute(self, query):
            if query == "PRAGMA foreign_key_check":
                return SimpleNamespace(fetchall=list)
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


def test_sqlite_health_rejects_foreign_key_violations(tmp_path):
    database = tmp_path / "foreign-key.db"
    connection = cli.sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE parent(id INTEGER PRIMARY KEY);"
        "CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));"
        "INSERT INTO child VALUES(999);"
    )
    connection.close()
    assert cli._sqlite_health(SimpleNamespace(db=database)) == "unhealthy"


def test_decisions_cli_query_and_get(harness_context, monkeypatch):
    from typer.testing import CliRunner

    _config, store, orchestrator, _root = harness_context
    task = store.create(Task(title="decision CLI"))
    from harness.decisions import DecisionService

    decision = DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use SQLite",
            "rationale": "Local persistence",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "tags": ["storage"],
        }
    )
    monkeypatch.setattr(cli, "build", lambda: (_config, store, orchestrator))
    runner = CliRunner()
    query = runner.invoke(
        cli.app, ["decisions", "--task-id", str(task.id), "--tag", "storage"]
    )
    assert query.exit_code == 0 and json.loads(query.stdout)[0]["id"] == decision.id
    get = runner.invoke(cli.app, ["decisions", "--id", str(decision.id)])
    assert get.exit_code == 0 and json.loads(get.stdout)[0]["decision"] == "Use SQLite"
    missing = runner.invoke(cli.app, ["decisions", "--id", "999"])
    assert missing.exit_code == 1 and missing.stdout.strip() == "decision not found"


def test_artifact_cli_lists_latest_and_history(harness_context, monkeypatch):
    from typer.testing import CliRunner

    _config, store, orchestrator, _root = harness_context
    task = store.create(Task(title="artifact CLI"))
    store.save_artifact(task.id, "agent/step-1", "first", expected_version=0)
    store.save_artifact(task.id, "agent/step-1", "second", expected_version=1)
    monkeypatch.setattr(cli, "build", lambda: (_config, store, orchestrator))
    runner = CliRunner()
    latest = runner.invoke(cli.app, ["artifacts", "list", "--task-id", str(task.id)])
    assert latest.exit_code == 0
    assert json.loads(latest.stdout)[0]["content"] == "second"
    history = runner.invoke(
        cli.app,
        ["artifacts", "history", "--task-id", str(task.id), "--key", "agent/step-1"],
    )
    assert history.exit_code == 0
    assert [row["version"] for row in json.loads(history.stdout)] == [1, 2]


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
            assert statement in {"PRAGMA quick_check", "PRAGMA foreign_key_check"}
            if raise_error:
                raise cli.sqlite3.OperationalError("database detail")
            if statement == "PRAGMA foreign_key_check":
                return SimpleNamespace(fetchall=list)
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
    initial_status = capsys.readouterr().out
    assert "stopped" in initial_status
    assert "SQLite schema: 15/15 (available)" in initial_status

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
    assert "✓ SQLite Schema" in capsys.readouterr().out
    monkeypatch.setattr(
        cli,
        "inspect_sqlite",
        lambda _path: {"version_history_complete": False},
    )
    with pytest.raises(cli.typer.Exit):
        cli.doctor()
    assert "✗ SQLite Schema" in capsys.readouterr().out


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


def test_task_knowledge_search_cli_uses_shared_task_service(harness_context, capsys):
    _, store, orchestrator, _root = harness_context
    from typer.testing import CliRunner

    task = store.create(Task(title="CLI knowledge task"))
    vault = orchestrator.context.memory.vault
    (vault / "CLI.md").write_text(
        "---\ntype: architecture\nreviewed_on: 2026-10-03\n---\n"
        "# Architecture\nCLI task routing uses shared services.\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli.app, ["tasks", "knowledge-search", str(task.id), "routing"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["task_id"] == task.id
    assert "context:vault/CLI.md#Architecture" in result.output

    missing = CliRunner().invoke(
        cli.app, ["tasks", "knowledge-search", "999999", "routing"]
    )
    assert missing.exit_code == 1
    assert "task not found" in missing.output


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


def test_memory_sync_command_projects_sqlite_decisions(harness_context, monkeypatch):
    cfg, store, _orchestrator, root = harness_context
    from typer.testing import CliRunner

    store.decisions.save(None, "Use SQLite", "Canonical persistence")
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    monkeypatch.setattr(
        cli,
        "build",
        lambda: (_ for _ in ()).throw(AssertionError("Orchestrator must not start")),
    )

    result = CliRunner().invoke(cli.app, ["memory", "sync"])

    assert result.exit_code == 0, result.output
    assert "created=1" in result.output
    assert (root / "vault" / "_harness" / "decisions" / "1.md").read_text().find(
        "Use SQLite"
    ) >= 0


@pytest.mark.parametrize("database_contents", [None, "not a SQLite database"])
def test_memory_sync_fails_closed_when_canonical_database_is_unavailable(
    harness_context, monkeypatch, capsys, database_contents
):
    cfg, _store, _orchestrator, root = harness_context
    database_path = root / "missing.db"
    cfg.data["paths"]["database"] = str(database_path)
    if database_contents is not None:
        database_path.write_text(database_contents, encoding="utf-8")
    monkeypatch.setattr(cli, "Config", lambda: cfg)

    with pytest.raises(cli.typer.Exit) as result:
        cli.memory_sync()

    assert result.value.exit_code == 2
    assert capsys.readouterr().out == (
        "Obsidian decision projection unavailable: "
        + ("DatabaseError" if database_contents else "FileNotFoundError")
        + "\n"
    )
    assert not (root / "vault" / "_harness").exists()


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


def test_memory_status_reports_explicit_mcp_opt_in(
    harness_context, monkeypatch, capsys
):
    cfg, store, _orchestrator, _root = harness_context
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    cfg.data["memory"]["obsidian"]["enabled"] = True
    cfg.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False

    cli.memory_status()

    report = json.loads(capsys.readouterr().out)
    assert report["obsidian_enabled"] is True
    assert report["mcp_enabled"] is False
    assert report["markdown_notes"] == 0
    assert report["last_memory_monitor"] is None
    assert report["last_memory_ops_acceptance"] is None


def test_memory_service_factories_avoid_orchestrator_initialization(
    harness_context, monkeypatch
):
    from harness import core
    from harness.memory import QdrantMemory

    cfg, _store, _orchestrator, _root = harness_context
    created = []
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    monkeypatch.setattr(core, "Store", lambda config: created.append(config) or "store")

    assert cli._build_memory_store() == (cfg, "store")
    assert created == [cfg]

    cfg.data["memory"]["qdrant"].update(
        {"url": "http://127.0.0.1:6333", "collection": "memory-test", "timeout": 7}
    )
    cfg.data["memory"]["embeddings"]["dimensions"] = 1024
    cfg.data["secrets"]["QDRANT__SERVICE__API_KEY"] = "test-key"
    client = cli._memory_qdrant(cfg)

    assert isinstance(client, QdrantMemory)
    assert client.url == "http://127.0.0.1:6333"
    assert client.collection == "memory-test"
    assert client.dimension == 1024
    assert client.timeout == 7
    assert client.api_key == "test-key"
    assert client.embedder is None


def test_memory_status_does_not_create_a_missing_vault(
    harness_context, monkeypatch, capsys
):
    cfg, store, _orchestrator, _root = harness_context
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    vault = cfg.path("obsidian_vault")
    vault.rmdir()

    cli.memory_status()

    report = json.loads(capsys.readouterr().out)
    assert report["exists"] is False
    assert not vault.exists()


def test_memory_status_does_not_initialize_mcp_registry(
    harness_context, monkeypatch, capsys
):
    cfg, store, _orchestrator, _root = harness_context
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(
        cli,
        "build",
        lambda: (_ for _ in ()).throw(AssertionError("MCP registry initialized")),
    )

    cli.memory_status()

    assert json.loads(capsys.readouterr().out)["mcp_enabled"] is False


def test_memory_watch_requires_explicit_configuration(
    harness_context, monkeypatch, capsys
):
    cfg, store, _orchestrator, _root = harness_context
    cfg.data["memory"].setdefault("monitoring", {})["enabled"] = False
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    with pytest.raises(cli.typer.Exit) as result:
        cli.memory_watch(once=True)
    assert result.value.exit_code == 2
    assert capsys.readouterr().out == "Memory monitoring is disabled in configuration\n"


@pytest.mark.parametrize("healthy", [True, False])
def test_memory_watch_once_persists_and_reports_health(
    harness_context, monkeypatch, capsys, healthy
):
    cfg, store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    report = {"healthy": healthy, "checks": {"sqlite": "healthy"}}
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(cli, "_memory_qdrant", lambda _config: _orchestrator.qdrant)
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "b" * 64)
    monkeypatch.setattr(
        "harness.memory_monitor.collect_memory_health", lambda *_args, **_kwargs: report
    )

    if not healthy:
        with pytest.raises(cli.typer.Exit) as result:
            cli.memory_watch(once=True)
        assert result.value.exit_code == 1
    else:
        cli.memory_watch(once=True)

    assert json.loads(capsys.readouterr().out) == report
    from harness.database import OperationalSnapshotRepository

    assert (
        OperationalSnapshotRepository(store.database).latest_memory_health()["healthy"]
        is healthy
    )


def test_memory_watch_stops_cleanly_on_keyboard_interrupt(
    harness_context, monkeypatch, capsys
):
    cfg, store, orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(cli, "_memory_qdrant", lambda _config: orchestrator.qdrant)
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "b" * 64)
    monkeypatch.setattr(
        "harness.memory_monitor.MemoryHealthMonitor.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    cli.memory_watch()

    assert capsys.readouterr().out == "Memory monitoring stopped\n"


def test_memory_watch_runs_until_monitor_returns(harness_context, monkeypatch, capsys):
    cfg, store, orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(cli, "_memory_qdrant", lambda _config: orchestrator.qdrant)
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "b" * 64)
    monkeypatch.setattr(
        "harness.memory_monitor.MemoryHealthMonitor.run", lambda *_args, **_kwargs: 3
    )

    cli.memory_watch(once=False)

    assert capsys.readouterr().out == ""


def test_memory_service_commands_are_confirmation_gated_and_redacted(
    harness_context, monkeypatch, capsys
):
    cfg, _store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    calls = []
    fake = SimpleNamespace(
        install=lambda **kwargs: (
            calls.append(("install", kwargs)) or {"installed": True, "loaded": True}
        ),
        status=lambda: {"installed": True, "loaded": True},
        uninstall=lambda **kwargs: (
            calls.append(("uninstall", kwargs)) or {"installed": False, "removed": True}
        ),
    )
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    monkeypatch.setattr(cli, "_launchd_memory_watch_service", lambda: fake)

    cli.memory_service_install(confirm=True)
    assert json.loads(capsys.readouterr().out) == {
        "installed": True,
        "loaded": True,
    }
    cli.memory_service_status()
    assert json.loads(capsys.readouterr().out) == {
        "installed": True,
        "loaded": True,
    }
    cli.memory_service_uninstall(confirm=True)
    assert json.loads(capsys.readouterr().out) == {
        "installed": False,
        "removed": True,
    }
    assert calls == [
        ("install", {"confirm": True, "monitoring_enabled": True}),
        ("uninstall", {"confirm": True}),
    ]


def test_memory_ops_acceptance_requires_confirmation_and_enabled_config(
    harness_context, monkeypatch, capsys
):
    import typer

    cfg, _store, _orchestrator, _root = harness_context
    cfg.data["memory"].setdefault("monitoring", {})["enabled"] = False
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, _store))
    with pytest.raises(typer.Exit) as result:
        cli.memory_ops_acceptance(confirm=False)
    assert result.value.exit_code == 2
    assert "--confirm" in capsys.readouterr().out
    with pytest.raises(typer.Exit) as result:
        cli.memory_ops_acceptance(confirm=True)
    assert result.value.exit_code == 2
    assert "disabled" in capsys.readouterr().out


def test_memory_ops_acceptance_persists_successful_evidence(
    harness_context, monkeypatch, capsys
):
    from datetime import UTC, datetime, timedelta

    from harness.database import OperationalSnapshotRepository

    cfg, store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {}).update(
        {"enabled": True, "interval_seconds": 60}
    )
    now = datetime.now(UTC)
    snapshots = [
        {
            "observed_at": (now - timedelta(seconds=60 * (index + 1))).isoformat(),
            "healthy": True,
            "checks": {
                "sqlite": "healthy",
                "vault": "healthy",
                "qdrant": "healthy",
                "embedding_evidence": "healthy",
            },
            "source_sha256": "a" * 64,
        }
        for index in range(6)
    ]
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(
        cli,
        "_launchd_memory_watch_service",
        lambda: SimpleNamespace(status=lambda: {"installed": True, "loaded": True}),
    )
    monkeypatch.setattr(
        OperationalSnapshotRepository,
        "recent_memory_health",
        lambda _self, *, limit: snapshots[:limit],
    )
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "a" * 64)

    cli.memory_ops_acceptance(confirm=True)

    output = json.loads(capsys.readouterr().out)
    assert output["acceptance"]["passed"] is True
    assert output["evidence_id"] == 1
    from harness.database import MemoryOpsEvidenceRepository

    evidence = MemoryOpsEvidenceRepository(store.database).list()
    assert len(evidence) == 1
    assert evidence[0]["passed"] is True
    cli.memory_status()
    status = json.loads(capsys.readouterr().out)
    assert status["last_memory_ops_acceptance"]["id"] == evidence[0]["id"]


def test_memory_ops_acceptance_failure_and_launchd_error_do_not_write_evidence(
    harness_context, monkeypatch, capsys
):
    import typer

    cfg, store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {}).update(
        {"enabled": True, "interval_seconds": 60}
    )
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(
        cli,
        "_launchd_memory_watch_service",
        lambda: SimpleNamespace(status=lambda: {"installed": False, "loaded": False}),
    )
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "a" * 64)
    with pytest.raises(typer.Exit) as result:
        cli.memory_ops_acceptance(confirm=True)
    assert result.value.exit_code == 1
    assert json.loads(capsys.readouterr().out)["checks"]["launchd_loaded"] is False

    monkeypatch.setattr(
        cli,
        "_launchd_memory_watch_service",
        lambda: SimpleNamespace(
            status=lambda: (_ for _ in ()).throw(RuntimeError("private details"))
        ),
    )
    with pytest.raises(typer.Exit) as result:
        cli.memory_ops_acceptance(confirm=True)
    assert result.value.exit_code == 2
    assert (
        capsys.readouterr().out
        == "Memory operational acceptance failed: RuntimeError\n"
    )


def test_memory_service_command_redacts_install_errors(
    harness_context, monkeypatch, capsys
):
    import typer

    cfg, _store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    monkeypatch.setattr(
        cli,
        "_launchd_memory_watch_service",
        lambda: SimpleNamespace(
            install=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("secret user path")
            )
        ),
    )
    with pytest.raises(typer.Exit) as result:
        cli.memory_service_install(confirm=True)
    assert result.value.exit_code == 2
    assert (
        capsys.readouterr().out
        == "Memory watcher service install failed: RuntimeError\n"
    )


def test_memory_service_typer_options_and_error_paths(harness_context, monkeypatch):
    from typer.testing import CliRunner

    cfg, _store, _orchestrator, _root = harness_context
    cfg.data.setdefault("memory", {}).setdefault("monitoring", {})["enabled"] = True
    calls = []

    def fake_install(**kwargs):
        calls.append(kwargs)
        if not kwargs["confirm"]:
            raise ValueError("confirmation required")
        return {"installed": True, "loaded": True}

    fake = SimpleNamespace(
        install=fake_install,
        status=lambda: (_ for _ in ()).throw(OSError("private path")),
        uninstall=lambda **_kwargs: (_ for _ in ()).throw(ValueError("private path")),
    )
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    monkeypatch.setattr(cli, "_launchd_memory_watch_service", lambda: fake)
    runner = CliRunner()

    missing_confirmation = runner.invoke(cli.app, ["memory", "service", "install"])
    assert missing_confirmation.exit_code == 2
    assert "ValueError" in missing_confirmation.output
    installed = runner.invoke(cli.app, ["memory", "service", "install", "--confirm"])
    assert installed.exit_code == 0
    assert json.loads(installed.output) == {"installed": True, "loaded": True}
    status = runner.invoke(cli.app, ["memory", "service", "status"])
    assert status.exit_code == 2
    assert "OSError" in status.output
    uninstalled = runner.invoke(
        cli.app, ["memory", "service", "uninstall", "--confirm"]
    )
    assert uninstalled.exit_code == 2
    assert "ValueError" in uninstalled.output
    assert calls == [
        {"confirm": False, "monitoring_enabled": True},
        {"confirm": True, "monitoring_enabled": True},
    ]


def test_memory_service_factory_uses_project_and_executable_paths(
    harness_context, monkeypatch
):
    _cfg, _store, _orchestrator, root = harness_context
    monkeypatch.setattr(cli.Path, "home", lambda: root / "home")
    monkeypatch.setattr(cli.os.sys, "argv", [str(root / ".venv" / "bin" / "harness")])
    service = cli._launchd_memory_watch_service()
    assert service.home == root / "home"
    assert service.root == root
    assert service.executable == root / ".venv" / "bin" / "harness"


def test_memory_audit_reports_read_only_health(tmp_path, monkeypatch, capsys):
    from harness.vault_audit import CURATED_NOTE_TYPES

    vault = tmp_path / "vault"
    vault.mkdir()
    source_digest = hashlib.sha256((cli.ROOT / "README.md").read_bytes()).hexdigest()
    for relative, note_type in CURATED_NOTE_TYPES.items():
        path = vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntype: {note_type}\nlast_reviewed: {datetime.now(UTC).date().isoformat()}\nsources:\n  - path: README.md\n    sha256: {source_digest}\n---\n# Note",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        cli, "Config", lambda: SimpleNamespace(path=lambda _name: vault)
    )

    cli.memory_audit()

    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["audited_notes"] == len(CURATED_NOTE_TYPES)
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


def test_memory_audit_compares_read_only_sqlite_decisions(
    tmp_path, monkeypatch, capsys
):
    from harness.database import Database, DecisionRepository

    vault = tmp_path / "vault"
    database = tmp_path / "harness.sqlite"
    repository = DecisionRepository(Database(database))
    repository.record(
        None,
        "architecture",
        "human",
        "SQLite is authoritative",
        "local state",
        [],
        [],
        None,
    )
    monkeypatch.setattr(
        cli,
        "Config",
        lambda: SimpleNamespace(
            path=lambda name: database if name == "database" else vault
        ),
    )
    with pytest.raises(cli.typer.Exit) as exc:
        cli.memory_audit()
    report = json.loads(capsys.readouterr().out)
    assert exc.value.exit_code == 1
    assert any(
        item["code"] == "decision_projection_missing" for item in report["findings"]
    )
    assert not (vault / "_harness").exists()


@pytest.mark.parametrize("command", ["sync", "audit"])
def test_memory_cli_closes_read_only_sqlite_connections(
    tmp_path, monkeypatch, capsys, command
):
    import sqlite3
    from types import SimpleNamespace

    from harness.database import Database

    database = Database(tmp_path / "harness.sqlite")
    vault = tmp_path / "vault"
    config = SimpleNamespace(
        path=lambda name: database.path if name == "database" else vault
    )
    monkeypatch.setattr(cli, "Config", lambda: config)
    if command == "sync":
        monkeypatch.setattr(
            "harness.memory_projection.DecisionProjection.sync",
            lambda _self, _rows: {"created": 0, "updated": 0, "removed": 0},
        )
    else:
        monkeypatch.setattr(
            "harness.vault_audit.audit_vault",
            lambda *_args, **_kwargs: {"healthy": True},
        )

    original_connect = sqlite3.connect
    opened = []

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(cli.sqlite3, "connect", tracked_connect)
    getattr(cli, f"memory_{command}")()
    capsys.readouterr()

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


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


def test_qdrant_config_audit_reports_redacted_findings(tmp_path, capsys):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n  qdrant:\n    image: qdrant/qdrant:latest\n"
        '    ports: ["6333:6333"]\n'
        "    environment:\n      QDRANT__SERVICE__API_KEY: never-print-this\n"
    )
    config = tmp_path / "production.yml"
    config.write_text("service:\n  enable_cors: true\n")
    with pytest.raises(cli.typer.Exit) as exc:
        cli.qdrant_config_audit(compose, config)
    output = capsys.readouterr().out
    assert exc.value.exit_code == 1
    assert "literal_api_key" in output
    assert "never-print-this" not in output


def test_qdrant_config_audit_reports_unavailable_without_details(tmp_path, capsys):
    with pytest.raises(cli.typer.Exit) as exc:
        cli.qdrant_config_audit(tmp_path / "missing.yml")
    assert exc.value.exit_code == 2
    assert (
        capsys.readouterr().out
        == "Qdrant configuration audit unavailable: ValueError\n"
    )


def test_qdrant_config_audit_cli_accepts_compose_and_optional_config(tmp_path):
    from typer.testing import CliRunner

    compose = tmp_path / "compose.yml"
    compose.write_text(
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n"
        '    ports: ["127.0.0.1:6333:6333"]\n'
    )
    config = tmp_path / "production.yml"
    config.write_text("service:\n  enable_cors: false\n")
    result = CliRunner().invoke(
        cli.app,
        ["memory", "qdrant-config-audit", str(compose), "--config", str(config)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"healthy": True, "findings": []}


def test_knowledge_proposal_cli_requires_review_and_applies_only_confirmed_content(
    tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    root = tmp_path / "repo"
    root.mkdir()
    source = root / "source.py"
    source.write_text("def verified():\n    return True\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    knowledge = vault / "knowledge"
    knowledge.mkdir()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (knowledge / "Projektwissen.md").write_text(
        "---\ntype: project-knowledge\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: source.py, sha256: {source_digest}}}]\n---\n# Wissen\n"
    )
    (vault / "Willkommen.md").write_text(
        "---\ntype: vault-home\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: source.py, sha256: {source_digest}}}]\n---\n"
        "# Willkommen\n\n[[knowledge/Projektwissen]]\n"
    )
    body_file = tmp_path / "proposal.md"
    body_file.write_text("Never persist api_key=private-value in notes.\n")
    config = SimpleNamespace(
        path=lambda name: vault if name == "obsidian_vault" else tmp_path / "db"
    )
    audit = SimpleNamespace(
        sanitize=lambda value: value.replace("private-value", "[REDACTED]")
    )
    store = SimpleNamespace(audit=audit)
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "Config", lambda: config)
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (config, store))
    runner = CliRunner()
    args = [
        "memory",
        "knowledge-propose",
        "--task-id",
        "TASK-42",
        "--title",
        "Verified lesson",
        "--category",
        "lessons-learned",
        "--source",
        "source.py",
        "--body-file",
        str(body_file),
    ]
    proposed = runner.invoke(cli.app, args)
    assert proposed.exit_code == 0, proposed.output
    proposal = json.loads(proposed.output)
    assert not (vault / proposal["target"]).exists()
    listed = runner.invoke(cli.app, ["memory", "knowledge-proposals"])
    assert listed.exit_code == 0, listed.output
    assert "[REDACTED]" in listed.output
    assert "private-value" not in listed.output
    no_confirm = runner.invoke(
        cli.app,
        ["memory", "knowledge-approve", proposal["id"], "--sha256", proposal["digest"]],
    )
    assert no_confirm.exit_code != 0
    approved = runner.invoke(
        cli.app,
        [
            "memory",
            "knowledge-approve",
            proposal["id"],
            "--sha256",
            proposal["digest"],
            "--confirm",
        ],
    )
    assert approved.exit_code == 0, approved.output
    note = (vault / proposal["target"]).read_text()
    assert "[REDACTED]" in note
    assert "private-value" not in note


def test_knowledge_proposal_cli_rejects_invalid_source_and_records_rejection(
    tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    root = tmp_path / "repo"
    root.mkdir()
    source = root / "source.py"
    source.write_text("source\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    knowledge = vault / "knowledge"
    knowledge.mkdir()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (knowledge / "Projektwissen.md").write_text(
        "---\ntype: project-knowledge\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: source.py, sha256: {source_digest}}}]\n---\n# Wissen\n"
    )
    (vault / "Willkommen.md").write_text(
        "---\ntype: vault-home\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: source.py, sha256: {source_digest}}}]\n---\n"
        "# Willkommen\n\n[[knowledge/Projektwissen]]\n"
    )
    body_file = tmp_path / "proposal.md"
    body_file.write_text("Body\n")
    config = SimpleNamespace(path=lambda _name: vault)
    store = SimpleNamespace(audit=SimpleNamespace(sanitize=lambda value: value))
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "Config", lambda: config)
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (config, store))
    runner = CliRunner()
    bad = runner.invoke(
        cli.app,
        [
            "memory",
            "knowledge-propose",
            "--task-id",
            "TASK-1",
            "--title",
            "Bad",
            "--category",
            "lessons-learned",
            "--source",
            "../outside.py",
            "--body-file",
            str(body_file),
        ],
    )
    assert bad.exit_code == 2
    proposal = runner.invoke(
        cli.app,
        [
            "memory",
            "knowledge-propose",
            "--task-id",
            "TASK-2",
            "--title",
            "Rejected lesson",
            "--category",
            "project-knowledge",
            "--source",
            "source.py",
            "--body-file",
            str(body_file),
        ],
    )
    result = json.loads(proposal.output)
    rejected = runner.invoke(
        cli.app,
        [
            "memory",
            "knowledge-reject",
            result["id"],
            "--sha256",
            result["digest"],
            "--confirm",
        ],
    )
    assert rejected.exit_code == 0, rejected.output
    assert json.loads(rejected.output)["status"] == "rejected"


@pytest.mark.parametrize(
    ("proposal_state", "expected_exit", "expected_ready"),
    [("current", 0, True), ("stale_or_unavailable", 1, False)],
)
def test_knowledge_review_combines_vault_health_and_stale_proposals(
    tmp_path, monkeypatch, proposal_state, expected_exit, expected_ready
):
    from typer.testing import CliRunner

    from harness import vault_audit, vault_steward

    vault = tmp_path / "vault"
    root = tmp_path / "repo"
    database = tmp_path / "missing.db"
    vault.mkdir()
    root.mkdir()
    config = SimpleNamespace(
        path=lambda name: {
            "obsidian_vault": vault,
            "database": database,
        }[name]
    )
    monkeypatch.setattr(cli, "Config", lambda: config)
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(
        vault_audit,
        "audit_vault",
        lambda *_args, **_kwargs: {
            "healthy": True,
            "findings": [],
            "audited_notes": 19,
        },
    )

    class Steward:
        def __init__(self, *_args, **_kwargs):
            pass

        def pending(self):
            return [
                {
                    "id": "a" * 24,
                    "status": "pending",
                    "source_state": proposal_state,
                }
            ]

    monkeypatch.setattr(vault_steward, "VaultKnowledgeSteward", Steward)
    result = CliRunner().invoke(cli.app, ["memory", "knowledge-review"])
    report = json.loads(result.output)
    assert result.exit_code == expected_exit
    assert report["ready"] is expected_ready
    assert report["proposal_counts"]["pending"] == 1
    assert report["stale_pending_proposals"] == (
        [] if proposal_state == "current" else ["a" * 24]
    )


def test_qdrant_upgrade_smoke_requires_confirmation(capsys):
    with pytest.raises(cli.typer.Exit) as exc:
        cli.qdrant_upgrade_smoke(
            "qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0", confirm=False
        )
    assert exc.value.exit_code == 2
    assert "No action taken" in capsys.readouterr().out


def test_qdrant_upgrade_smoke_cli_requires_explicit_confirmation():
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        cli.app,
        [
            "qdrant-upgrade-smoke",
            "--baseline-image",
            "qdrant/qdrant:v1.19.0",
            "--candidate-image",
            "qdrant/qdrant:v1.20.0",
        ],
    )
    assert result.exit_code == 2
    assert "No action taken" in result.output


def test_qdrant_upgrade_smoke_reports_result_and_redacted_errors(monkeypatch, capsys):
    result = {
        "baseline_image": "qdrant/qdrant:v1.19.0",
        "candidate_image": "qdrant/qdrant:v1.20.0",
        "collection_readable": True,
        "persisted_after_restart": True,
        "cleaned": True,
    }
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "upgrade_smoke_test",
        lambda _self, _baseline, _candidate: result,
    )
    cli.qdrant_upgrade_smoke(
        "qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0", confirm=True
    )
    output = capsys.readouterr().out
    assert "baseline=qdrant/qdrant:v1.19.0" in output
    assert "candidate=qdrant/qdrant:v1.20.0" in output
    assert "persisted_after_restart=True" in output

    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "upgrade_smoke_test",
        lambda *_args: (_ for _ in ()).throw(ValueError("sensitive detail")),
    )
    with pytest.raises(cli.typer.Exit) as exc:
        cli.qdrant_upgrade_smoke(
            "qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0", confirm=True
        )
    assert exc.value.exit_code == 1
    assert (
        capsys.readouterr().out == "Isolated Qdrant upgrade test failed: ValueError\n"
    )


def test_qdrant_status_reports_disabled_service_and_exits_on_bad_health(
    harness_context, monkeypatch, capsys
):
    _cfg, _store, orchestrator, _root = harness_context
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (_cfg, _store))
    monkeypatch.setattr(cli, "_memory_qdrant", lambda _conf: orchestrator.qdrant)
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


def test_qdrant_acceptance_requires_explicit_confirmation(capsys):
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_acceptance(confirm=False)
    assert result.value.exit_code == 2
    assert "No provider request sent" in capsys.readouterr().out


def test_embedding_acceptance_factory_does_not_initialize_mcp(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data["paths"]["database"] = str(tmp_path / "acceptance.db")
    cfg.data["paths"]["obsidian_vault"] = str(tmp_path / "vault")
    store = Store(cfg)
    monkeypatch.setattr(cli, "_build_memory_store", lambda: (cfg, store))
    monkeypatch.setattr(
        cli,
        "build",
        lambda: (_ for _ in ()).throw(AssertionError("orchestrator started")),
    )

    conf, result_store, qdrant, service = cli._build_embedding_acceptance()

    assert conf is cfg and result_store is store
    assert qdrant.embedder.model == cfg.data["memory"]["embeddings"]["model"]
    assert service.vectors is qdrant
    assert not (tmp_path / "vault").exists()


def test_qdrant_acceptance_checks_enabled_health_and_read_only_search(
    harness_context, monkeypatch, capsys
):
    cfg, store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    cfg.data["memory"]["embeddings"].update(
        {"model": "local-embed", "dimensions": 1024}
    )
    orchestrator.qdrant.embedder.model = "local-embed"
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "a" * 64)
    calls = []
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": True, "dimension": 1024},
    )
    orchestrator.context.memory.write("acceptance", "verified source")
    source_hash = orchestrator.memory_service.digest("verified source")
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda query, limit: (
            calls.append((query, limit))
            or [
                {
                    "id": "not-output",
                    "payload": {"source": "acceptance", "source_hash": source_hash},
                }
            ]
        ),
    )

    cli.qdrant_acceptance(confirm=True)

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "healthy": True,
        "read_only": True,
        "stage": "embedding_search",
        "collection": cfg.data["memory"]["qdrant"]["collection"],
        "model": "local-embed",
        "dimension": 1024,
        "matches": 1,
        "candidates": 1,
        "evidence_id": 1,
        "evidence_kind": "embedding",
    }
    assert calls == [("Harness live embedding acceptance probe", 1)]
    assert "not-output" not in json.dumps(report)
    evidence = store.database
    from harness.evidence import EvidenceRepository

    item = EvidenceRepository(evidence).list(kind="embedding")[0]
    assert item["subject_sha256"] == "a" * 64
    assert item["passed"] is True
    assert item["checks"] == {
        "embedding_provider_response": True,
        "embedding_dimension_match": True,
        "qdrant_collection_healthy": True,
        "read_only_search": True,
        "source_hash_validated": True,
    }
    assert "not-output" not in json.dumps(item)


def test_qdrant_acceptance_fails_if_live_evidence_cannot_be_persisted(
    harness_context, monkeypatch, capsys
):
    import sqlite3

    cfg, store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": True, "dimension": 1024},
    )
    orchestrator.context.memory.write("acceptance", "current source")
    source_hash = orchestrator.memory_service.digest("current source")
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda *_args, **_kwargs: [
            {"payload": {"source": "acceptance", "source_hash": source_hash}}
        ],
    )
    monkeypatch.setattr(cli, "source_tree_sha256", lambda _root: "a" * 64)

    from harness.evidence import EvidenceRepository

    monkeypatch.setattr(
        EvidenceRepository,
        "record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError()),
    )
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_acceptance(confirm=True)

    assert result.value.exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "healthy": False,
        "stage": "evidence_persist",
        "error_type": "OperationalError",
    }


def test_qdrant_acceptance_does_not_record_evidence_without_current_source(
    harness_context, monkeypatch, capsys
):
    cfg, store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": True, "dimension": 1024},
    )
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda *_args, **_kwargs: [
            {"payload": {"source": "missing", "source_hash": "stale"}}
        ],
    )
    with pytest.raises(cli.typer.Exit) as error:
        cli.qdrant_acceptance(confirm=True)
    assert error.value.exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "healthy": False,
        "stage": "source_validation",
        "errors": ["no_current_source_match"],
        "candidates": 1,
    }
    from harness.evidence import EvidenceRepository

    assert EvidenceRepository(store.database).list(kind="embedding") == []


def test_qdrant_acceptance_rejects_embedding_contract_drift_before_search(
    harness_context, monkeypatch, capsys
):
    cfg, _store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(orchestrator.qdrant.embedder, "model", "wrong-model")
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": True, "dimension": orchestrator.qdrant.dimension},
    )
    monkeypatch.setattr(
        orchestrator.qdrant,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("searched")),
    )
    with pytest.raises(cli.typer.Exit) as error:
        cli.qdrant_acceptance(confirm=True)
    assert error.value.exit_code == 1
    assert json.loads(capsys.readouterr().out)["stage"] == "embedding_contract"


def test_qdrant_acceptance_fails_closed_when_service_is_disabled_or_unhealthy(
    harness_context, monkeypatch, capsys
):
    cfg, store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = False
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_acceptance(confirm=True)
    assert result.value.exit_code == 1
    assert "disabled" in capsys.readouterr().out

    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": False, "errors": ["collection_missing"]},
    )
    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_acceptance(confirm=True)
    assert result.value.exit_code == 1
    assert json.loads(capsys.readouterr().out)["stage"] == "qdrant_health"


@pytest.mark.parametrize("search_result", [RuntimeError("private"), {"bad": "shape"}])
def test_qdrant_acceptance_redacts_errors_and_rejects_invalid_results(
    harness_context, monkeypatch, capsys, search_result
):
    cfg, store, orchestrator, _root = harness_context
    cfg.data["memory"]["qdrant"]["enabled"] = True
    monkeypatch.setattr(cli, "build", lambda: (cfg, store, orchestrator))
    monkeypatch.setattr(
        orchestrator.qdrant,
        "health_report",
        lambda: {"healthy": True, "dimension": 1024},
    )
    if isinstance(search_result, Exception):

        def fail_search(*_args, **_kwargs):
            raise search_result

        search = fail_search
    else:
        search = lambda *_args, **_kwargs: search_result
    monkeypatch.setattr(orchestrator.qdrant, "search", search)

    with pytest.raises(cli.typer.Exit) as result:
        cli.qdrant_acceptance(confirm=True)
    assert result.value.exit_code == 1
    output = capsys.readouterr().out
    assert "embedding_search" in output
    assert "private" not in output


def test_metrics_command_prints_structured_durable_counters(harness_context, capsys):
    _cfg, store, _orchestrator, _root = harness_context
    task = store.create(Task(title="metrics"))
    store.event(task.id, "task.created", {})

    cli.runtime_metrics(format="json")

    report = json.loads(capsys.readouterr().out)
    assert report["tasks"]["by_status"]["pending"] == 1
    assert report["events"]["total"] == 1


def test_metrics_command_prints_prometheus_or_rejects_unknown_format(
    harness_context, capsys
):
    cli.runtime_metrics(format="prometheus")
    assert "harness_tasks_total 0" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit) as error:
        cli.runtime_metrics(format="xml")
    assert error.value.exit_code == 2
    assert "json or 'prometheus'" in capsys.readouterr().out


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


def test_completion_command_accepts_a_fully_satisfied_matrix(
    harness_context, monkeypatch, capsys
):
    cfg, store, _orchestrator, root = harness_context
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
    from datetime import UTC, datetime

    from harness.evidence import EvidenceInput, EvidenceRepository
    from harness.verification import source_tree_sha256

    cfg.data["verification"] = {"required_evidence": ["qdrant"]}
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    EvidenceRepository(store.database).record(
        EvidenceInput(
            kind="qdrant",
            source_id="test-qdrant",
            observed_at=datetime.now(UTC),
            subject_sha256=source_tree_sha256(root),
            passed=True,
            checks={"health": True},
        )
    )

    cli.completion_status()

    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is True
    assert result["evidence_policy_valid"] is True


def test_completion_command_fails_closed_on_invalid_matrix(harness_context, capsys):
    import typer

    _cfg, _store, _orchestrator, root = harness_context
    (root / "GAP_MATRIX.md").write_text("not a matrix")

    with pytest.raises(typer.Exit) as error:
        cli.completion_status()

    assert error.value.exit_code == 2
    assert "unverifiable" in capsys.readouterr().out


def test_completion_enforces_configured_evidence_policy(
    harness_context, monkeypatch, capsys
):
    import typer
    from test_completion import coverage_report

    from harness.verification import write_verification_report

    cfg, _store, _orchestrator, root = harness_context
    cfg.data["verification"] = {"required_evidence": ["qdrant"], "max_age_hours": 168}
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    matrix_path = root / "GAP_MATRIX.md"
    rows = "".join(f"| {n} | item | Erfüllt | evidence |\n" for n in range(1, 110))
    matrix_path.write_text(
        "| Nr. | Name | Status | Tiefe |\n|---:|---|---|---|\n" + rows
    )
    coverage_path = root / "coverage.json"
    coverage_path.write_text(json.dumps(coverage_report()))
    junit_path = root / "junit.xml"
    junit_path.write_text('<testsuite tests="1"><testcase name="ok" /></testsuite>')
    write_verification_report(
        matrix_path, coverage_path, junit_path, root / "data" / "verification.json"
    )
    cfg.path("database").unlink()
    with pytest.raises(typer.Exit) as error:
        cli.completion_status()
    report = json.loads(capsys.readouterr().out)
    assert error.value.exit_code == 1
    assert report["required_evidence"] == {"qdrant": False}
    assert report["evidence_policy_valid"] is False
