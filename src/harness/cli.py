"""Thin Typer commands that call the same stores and orchestrator as REST."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import signal
import socket

import typer
import uvicorn
import yaml

from .core import ROOT, Config, Task, build
from .docker_broker import DockerComposeBroker
from .security import SecretResolver

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


@app.command()
def start(host: str | None = None, port: int | None = None):
    settings = Config().data.get("api", {})
    host = host or settings.get("host", "127.0.0.1")
    port = port or settings.get("port", 8080)
    if host not in {"127.0.0.1", "localhost"}:
        raise typer.BadParameter("API host must remain local")
    pidfile = ROOT / "data" / "harness.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
            raise typer.BadParameter("Harness already appears to be running")
        except ProcessLookupError:
            pidfile.unlink(missing_ok=True)
    pidfile.write_text(str(os.getpid()))
    typer.echo(
        f"Harness started at http://{host}:{port}; Swagger: http://{host}:{port}/docs"
    )
    try:
        uvicorn.run("harness.api:app", host=host, port=port, reload=False)
    finally:
        pidfile.unlink(missing_ok=True)


@app.command()
def stop():
    pidfile = ROOT / "data" / "harness.pid"
    if not pidfile.exists():
        typer.echo("Harness is not running")
        return
    pid = int(pidfile.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        typer.echo("Graceful shutdown requested")
    except ProcessLookupError:
        pidfile.unlink(missing_ok=True)
        typer.echo("Removed stale process record")


@app.command()
def status():
    conf, store, orchestrator = build()
    pidfile = ROOT / "data" / "harness.pid"
    running = False
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
            running = True
        except (ValueError, ProcessLookupError):
            pidfile.unlink(missing_ok=True)
    counts = {
        name: len(store.tasks.list(name))
        for name in (
            "pending",
            "planning",
            "executing",
            "validating",
            "waiting_human",
            "failed",
        )
    }
    typer.echo(
        f"Harness: {'running' if running else 'stopped'}\nSQLite: connected\nObsidian: {'available' if conf.path('obsidian_vault').is_dir() else 'missing'}\nQdrant: {'available' if orchestrator.qdrant.health() else 'unavailable'}"
    )
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")
    for name, provider in orchestrator.models.providers.items():
        try:
            available = bool(provider.health())
        except (OSError, RuntimeError, ValueError):
            available = False
        typer.echo(f"Provider {name}: {'available' if available else 'unavailable'}")


@app.command()
def doctor():
    conf, store, orchestrator = build()
    docker_status = DockerComposeBroker().run("status")
    checks = {
        "macOS": platform.system() == "Darwin",
        "Apple Silicon": platform.machine() == "arm64",
        "Configuration": conf.validate(),
        "SQLite": store.db.exists(),
        "Git": shutil.which("git") is not None,
        "Docker": shutil.which("docker") is not None,
        "Docker Compose": bool(shutil.which("docker"))
        and os.system("docker compose version >/dev/null 2>&1") == 0,
        "Docker Daemon / Qdrant Compose": docker_status.returncode == 0,
        "Obsidian Vault": conf.path("obsidian_vault").is_dir(),
        "Workspace": conf.path("workspace").is_dir(),
        "Qdrant": not conf.data.get("memory", {}).get("qdrant", {}).get("enabled")
        or orchestrator.qdrant.health(),
    }
    try:
        with socket.create_connection(
            (
                conf.data.get("api", {}).get("host", "127.0.0.1"),
                conf.data.get("api", {}).get("port", 8080),
            ),
            timeout=0.2,
        ):
            port_free = False
    except OSError:
        port_free = True
    checks["API Port Available"] = port_free
    for label, ok in checks.items():
        typer.echo(f"{'✓' if ok else '✗'} {label}")
    if not all(checks.values()):
        raise typer.Exit(1)


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
