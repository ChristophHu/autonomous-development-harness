from types import SimpleNamespace

import pytest

from harness import cli
from harness.core import Config, Orchestrator, Store


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
    monkeypatch.setattr(
        cli.socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(OSError()),
    )
    cli.doctor()
    assert "✓" in capsys.readouterr().out
    cfg.data["memory"] = {"qdrant": {"enabled": True}}
    orchestrator.qdrant.health = lambda: False
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    with pytest.raises(cli.typer.Exit):
        cli.doctor()

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
    store.event(task_id, "created", {})
    cli.task_list()
    cli.task_show(task_id)
    cli.task_events(task_id)
    cli.task_run(task_id)
    assert "new" in capsys.readouterr().out
    with pytest.raises(cli.typer.Exit):
        cli.task_show(99999)


def test_model_and_config_commands(harness_context, monkeypatch, capsys):
    cfg, _, orchestrator, _ = harness_context
    monkeypatch.setattr(cli, "Config", lambda: cfg)
    cfg.data["models"]["registry"] = {
        "configured": {"provider": "local", "tier": "local"}
    }
    cli.model_list()
    orchestrator.models.register(
        "remote", type("P", (), {"health": lambda self: False})()
    )
    cli.model_list()
    cli.model_status()
    cli.config_show()
    cli.config_resolved()
    cli.config_validate()
    assert "configuration valid" in capsys.readouterr().out
    cfg.data.pop("secrets", None)
    assert cli._safe_config() == cfg.data


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
