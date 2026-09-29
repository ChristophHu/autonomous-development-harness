"""Thin Typer commands that call the same stores and orchestrator as REST."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import threading

import httpx
import typer
import uvicorn
import yaml

from .core import ROOT, Config, Task, build
from .docker_broker import DockerComposeBroker
from .security import SecretResolver
from .service_lifecycle import ServiceLifecycle

app = typer.Typer(no_args_is_help=True)
tasks = typer.Typer(no_args_is_help=True)
models = typer.Typer(no_args_is_help=True)
config = typer.Typer(no_args_is_help=True)
secrets = typer.Typer(no_args_is_help=True)
memory = typer.Typer(no_args_is_help=True)
app.add_typer(tasks, name="tasks")
app.add_typer(models, name="models")
app.add_typer(config, name="config")
app.add_typer(secrets, name="secrets")
app.add_typer(memory, name="memory")


def _lifecycle():
    return ServiceLifecycle(ROOT / "data" / "harness.pid")


def _serve_api(host, port, lifecycle, timeout=10.0):
    startup_finished = threading.Event()
    startup_succeeded = threading.Event()
    readiness_confirmed = threading.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            try:
                await super().startup(sockets)
                if self.started:
                    startup_succeeded.set()
            finally:
                startup_finished.set()

    server = ReadyServer(
        uvicorn.Config("harness.api:app", host=host, port=port, reload=False)
    )

    def announce_readiness():
        if not startup_finished.wait(timeout):
            server.should_exit = True
            return
        if startup_succeeded.is_set() and lifecycle.wait_until_ready(
            host, port, timeout
        ):
            readiness_confirmed.set()
            typer.echo(
                f"Harness ready at http://{host}:{port}; Swagger: http://{host}:{port}/docs"
            )
        else:
            server.should_exit = True

    monitor = threading.Thread(target=announce_readiness, daemon=True)
    monitor.start()
    server.run()
    monitor.join(timeout + 0.2)
    return readiness_confirmed.is_set()


def _component_health(call, *, enabled=True):
    if not enabled:
        return "disabled"
    try:
        return "available" if bool(call()) else "unavailable"
    except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
        return "unavailable"


def _sqlite_health(store):
    if not store.db.is_file():
        return "missing"
    try:
        uri = f"file:{store.db.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1) as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()
        return "available" if result and result[0] == "ok" else "unhealthy"
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return "unavailable"


def _provider_health(providers):
    return {
        name: _component_health(provider.health)
        for name, provider in sorted(providers.items())
    }


def _config_health(conf):
    try:
        return conf.validate()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


@app.command()
def start(host: str | None = None, port: int | None = None):
    settings = Config().data.get("api", {})
    host = host or settings.get("host", "127.0.0.1")
    port = settings.get("port", 8080) if port is None else port
    lifecycle = _lifecycle()
    try:
        record = lifecycle.register_current(host, port)
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    try:
        if not _serve_api(host, port, lifecycle):
            raise typer.Exit(1)
    finally:
        lifecycle.remove_record(record)


@app.command()
def stop():
    lifecycle = _lifecycle()
    try:
        state = lifecycle.stop()
    except (OSError, RuntimeError, TimeoutError) as error:
        typer.echo(f"Harness shutdown failed: {error}")
        raise typer.Exit(1) from error
    messages = {
        "not_running": "Harness is not running",
        "stale": "Removed stale process record",
        "stopped": "Harness stopped gracefully",
    }
    typer.echo(messages.get(state, "Harness stopped gracefully"))


@app.command()
def status():
    conf, store, orchestrator = build()
    try:
        service = _lifecycle().inspect()
    except (OSError, RuntimeError, ValueError):
        service = {"state": "unknown", "record": None, "ready": False}
    running = service["state"] == "running"
    counts = {
        name: len(store.tasks.list(name))
        for name in (
            "pending",
            "analyzing",
            "planning",
            "ready",
            "executing",
            "testing",
            "validating",
            "correcting",
            "recovering",
            "waiting_human",
            "failed",
            "blocked",
            "cancelled",
            "completed",
        )
    }
    service_description = service["state"]
    if running:
        record = service.get("record") or {}
        readiness = "ready" if service.get("ready") else "not ready"
        service_description = (
            f"running ({readiness}) pid={record.get('pid', 'unknown')}"
        )
    typer.echo(f"Harness: {service_description}")
    typer.echo(f"SQLite: {_sqlite_health(store)}")
    vault = conf.path("obsidian_vault")
    typer.echo(
        f"Obsidian: {_component_health(lambda: vault.is_dir() and os.access(vault, os.R_OK | os.W_OK))}"
    )
    qdrant_enabled = conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
    typer.echo(
        f"Qdrant: {_component_health(orchestrator.qdrant.health, enabled=qdrant_enabled)}"
    )
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")
    for name, provider in orchestrator.models.providers.items():
        typer.echo(f"Provider {name}: {_component_health(provider.health)}")


@app.command()
def doctor():
    conf, store, orchestrator = build()
    qdrant_enabled = conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
    docker_available = shutil.which("docker") is not None
    compose_available = False
    if docker_available and qdrant_enabled:
        try:
            compose_available = (
                subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    timeout=3,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            compose_available = False
    docker_daemon_available = True
    if qdrant_enabled and docker_available and compose_available:
        try:
            docker_daemon_available = (
                DockerComposeBroker().run("status").returncode == 0
            )
        except (OSError, RuntimeError, ValueError):
            docker_daemon_available = False
    elif qdrant_enabled:
        docker_daemon_available = False
    try:
        service = _lifecycle().inspect()
    except (OSError, RuntimeError, ValueError):
        service = {"state": "unknown", "record": None, "ready": False}
    workspace = conf.path("workspace")
    vault = conf.path("obsidian_vault")
    log_dir = conf.path("logs")
    providers = _provider_health(orchestrator.models.providers)
    checks = {
        "macOS": platform.system() == "Darwin",
        "Apple Silicon": platform.machine() == "arm64",
        "Configuration": _config_health(conf),
        "SQLite": _sqlite_health(store) == "available",
        "Git": shutil.which("git") is not None,
        "Docker": docker_available or not qdrant_enabled,
        "Docker Compose": compose_available or not qdrant_enabled,
        "Docker Daemon / Qdrant Compose": docker_daemon_available,
        "Obsidian Vault": vault.is_dir() and os.access(vault, os.R_OK | os.W_OK),
        "Workspace": workspace.is_dir() and os.access(workspace, os.R_OK | os.W_OK),
        "Logs Directory": log_dir.is_dir() and os.access(log_dir, os.W_OK),
        "Qdrant": _component_health(orchestrator.qdrant.health, enabled=qdrant_enabled)
        in {"available", "disabled"},
        "API Service": service["state"] == "running" and service["ready"],
    }
    checks.update(
        {f"Provider {name}": state == "available" for name, state in providers.items()}
    )
    if service["state"] == "running" and service["ready"]:
        port_available = True
    else:
        try:
            with socket.create_connection(
                (
                    conf.data.get("api", {}).get("host", "127.0.0.1"),
                    conf.data.get("api", {}).get("port", 8080),
                ),
                timeout=0.2,
            ):
                port_available = False
        except OSError:
            port_available = True
    checks["API Port Available"] = port_available
    checks["API Service"] = service["state"] in {"stopped", "stale"} or (
        service["state"] == "running" and service["ready"]
    )
    for label, ok in checks.items():
        typer.echo(f"{'✓' if ok else '✗'} {label}")
    if not all(checks.values()):
        raise typer.Exit(1)


@app.command("qdrant-smoke")
def qdrant_smoke(confirm: bool = typer.Option(False, "--confirm")):
    """Run a disposable isolated Qdrant health/persistence/restart check."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm to start an isolated test stack."
        )
        raise typer.Exit(2)
    try:
        result = DockerComposeBroker().live_smoke_test()
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Isolated Qdrant smoke test failed: {error}")
        raise typer.Exit(1) from error
    typer.echo(
        "Isolated Qdrant smoke test passed: "
        f"healthy={result['healthy']} "
        f"persisted_after_restart={result['persisted_after_restart']} "
        f"cleaned={result['cleaned']}"
    )


@memory.command("sync")
def memory_sync():
    """Rebuild the Obsidian decision-note projection from canonical SQLite."""
    conf, store, _ = build()
    from .memory_projection import DecisionProjection

    result = DecisionProjection(conf.path("obsidian_vault")).sync(
        store.decisions.list_all()
    )
    typer.echo(
        "Obsidian decision projection: "
        + ", ".join(f"{key}={value}" for key, value in result.items())
    )


@tasks.command("create")
def task_create(title: str, description: str = ""):
    _, _, orchestrator = build()
    item = orchestrator.service.create(Task(title=title, description=description))
    typer.echo(f"created task {item.id}")


@tasks.command("list")
def task_list(status: str | None = None):
    _, _, orchestrator = build()
    for task in orchestrator.service.list(status):
        typer.echo(f"{task.id}\t{task.status}\t{task.title}")


@tasks.command("abort")
def task_abort(task_id: int):
    _, _, orchestrator = build()
    typer.echo(orchestrator.service.abort(task_id).model_dump_json())


@tasks.command("show")
def task_show(task_id: int):
    _, store, _ = build()
    item = store.get(task_id)
    if not item:
        raise typer.Exit(1)
    typer.echo(item.model_dump_json(indent=2))


@tasks.command("run")
def task_run(task_id: int):
    _, _, orchestrator = build()
    typer.echo(asyncio.run(orchestrator.run(task_id)).model_dump_json())


@tasks.command("events")
def task_events(task_id: int):
    _, store, _ = build()
    for event in store.events.list(task_id):
        typer.echo(f"{event['created_at']} {event['kind']} {event['payload']}")


@models.command("list")
def model_list():
    _, _, orchestrator = build()
    for name, provider in orchestrator.models.providers.items():
        try:
            available = bool(provider.health())
        except (OSError, RuntimeError, ValueError):
            available = False
        typer.echo(f"{name}\t{'available' if available else 'unavailable'}")
        try:
            discovered = provider.models()
        except (OSError, RuntimeError, ValueError):
            typer.echo(f"{name}\t<discovery failed>")
            continue
        aliases = {
            alias: definition
            for alias, definition in orchestrator.models.models.items()
            if definition.get("provider") == name
        }
        for model_id in discovered:
            if not isinstance(model_id, str) or not model_id:
                continue
            matching = [
                alias
                for alias, definition in aliases.items()
                if definition.get("model") == model_id
            ]
            label = ",".join(matching) if matching else "-"
            typer.echo(f"{model_id}\t{name}\t{label}")


@models.command("status")
def model_status():
    model_list()


@models.command("usage")
def model_usage(
    task_id: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Show persisted model-call usage without exposing prompts or secrets."""
    _, store, _ = build()
    try:
        report = store.model_usage.report(
            task_id=task_id,
            provider=provider,
            model=model,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    except ValueError as error:
        typer.echo(f"Usage report failed: {error}")
        raise typer.Exit(2) from error
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@models.command("test")
def model_test(model_name: str):
    """Send a minimal completion request through a configured model/provider."""
    _, _, orchestrator = build()
    try:
        provider, model_id = orchestrator.models.resolve(model_name)
        response = provider.complete(
            "Reply with exactly: OK", **({"model": model_id} if model_id else {})
        )
        text = response[0] if isinstance(response, tuple) else response
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty model response")
    except Exception as exc:
        typer.echo(f"Model test failed ({type(exc).__name__})")
        raise typer.Exit(1) from exc
    typer.echo(f"Model test successful: {model_name}")


def _safe_config():
    return Config().redacted(resolved=True)


@config.command("show")
def config_show():
    typer.echo(yaml.safe_dump(Config().redacted(), sort_keys=False).rstrip())


@config.command("resolved")
def config_resolved():
    typer.echo(
        yaml.safe_dump(Config().redacted(resolved=True), sort_keys=False).rstrip()
    )


@config.command("validate")
def config_validate():
    Config().validate()
    typer.echo("configuration valid")


@secrets.command("list")
def secrets_list():
    typer.echo("\n".join(SecretResolver().list_names()))


@secrets.command("set")
def secret_set(name: str):
    value = typer.prompt("Secret", hide_input=True, confirmation_prompt=True)
    SecretResolver().set(name, value)
    typer.echo(f"stored {name} in macOS Keychain")


@secrets.command("delete")
def secret_delete(name: str):
    if not SecretResolver().delete(name):
        raise typer.Exit(1)
    typer.echo(f"deleted {name} from macOS Keychain")
