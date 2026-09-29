from types import SimpleNamespace

import pytest

from harness import cli
from harness.core import Config, Orchestrator, Store, Task


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
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    cli.start()
    assert calls and not (root / "data" / "harness.pid").exists()
    with pytest.raises(cli.typer.BadParameter):
        cli.start(host="0.0.0.0")
    pidfile = root / "data" / "harness.pid"
    pidfile.write_text("123")
    monkeypatch.setattr(cli.os, "kill", lambda *args: None)
    with pytest.raises(cli.typer.BadParameter):
        cli.start()
    monkeypatch.setattr(
        cli.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError())
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)
    cli.start()
    cli.stop()
    assert "not running" in capsys.readouterr().out


def test_stop_stale_and_active(harness_context, monkeypatch, capsys):
    _, _, _, root = harness_context
    pidfile = root / "data" / "harness.pid"
    pidfile.parent.mkdir()
    pidfile.write_text("999")
    monkeypatch.setattr(
        cli.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError())
    )
    cli.stop()
    assert not pidfile.exists()
    pidfile.write_text("123")
    monkeypatch.setattr(cli.os, "kill", lambda *a: None)
    cli.stop()
    assert "shutdown" in capsys.readouterr().out


def test_status_and_doctor(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, root = harness_context
    cli.status()
    assert "stopped" in capsys.readouterr().out
    pidfile = root / "data" / "harness.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text("123")
    monkeypatch.setattr(cli.os, "kill", lambda *args: None)
    cli.status()
    assert "running" in capsys.readouterr().out
    pidfile.write_text("bad")
    cli.status()
    assert not pidfile.exists()
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli.shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(cli.os, "system", lambda _: 0)
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
    assert docker_actions == ["status"]
    cfg.data["memory"] = {"qdrant": {"enabled": True}}
    orchestrator.qdrant.health = lambda: False
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    with pytest.raises(cli.typer.Exit):
        cli.doctor()

    docker_actions.clear()
    monkeypatch.setattr(
        cli.DockerComposeBroker,
        "run",
        lambda _self, action: (
            docker_actions.append(action) or SimpleNamespace(returncode=127)
        ),
    )
    with pytest.raises(cli.typer.Exit):
        cli.doctor()
    assert docker_actions == ["status"]

    class BoundSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    cfg.data["memory"] = {"qdrant": {"enabled": False}}
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.socket, "create_connection", lambda *a, **k: BoundSocket())
    with pytest.raises(cli.typer.Exit):
        cli.doctor()


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
