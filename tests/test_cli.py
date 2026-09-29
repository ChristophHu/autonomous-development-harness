import json
from types import SimpleNamespace

import pytest

from harness import cli
from harness.core import Config, Orchestrator, Store, Task
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
    assert "not running" in capsys.readouterr().out


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


def test_status_and_doctor(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, _root = harness_context
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
        list_names=lambda: ["TOKEN"], set=lambda *a: None, delete=lambda _: True
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
