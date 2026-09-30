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
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn
import yaml
from pydantic import ValidationError as PydanticValidationError

from .completion import audit_gap_matrix, audit_harness_completion
from .core import ROOT, Config, Task, build
from .docker_broker import DockerComposeBroker
from .providers import ProviderHealth
from .security import SecretResolver
from .service_lifecycle import ServiceLifecycle
from .verification import source_tree_sha256

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


@app.command("completion")
def completion_status():
    """Audit matrix, latest verification report, tests, and coverage evidence."""
    matrix = ROOT / "GAP_MATRIX.md"
    try:
        markdown = matrix.read_text(encoding="utf-8")
        coverage_path = ROOT / "coverage.json"
        coverage_text = (
            coverage_path.read_text(encoding="utf-8")
            if coverage_path.is_file()
            else None
        )
        verification_path = ROOT / "data" / "verification.json"
        try:
            evidence = json.loads(verification_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evidence = None
        matrix_report = audit_gap_matrix(markdown)
        report = audit_harness_completion(
            markdown,
            coverage_text,
            evidence,
            source_sha256=source_tree_sha256(ROOT),
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Harness completion: unverifiable ({type(error).__name__})")
        raise typer.Exit(2) from None
    report["fulfilled"] = matrix_report["fulfilled"]
    report["partial"] = matrix_report["partial"]
    report["open"] = matrix_report["open"]
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise typer.Exit(1)


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


def _qdrant_report(orchestrator, *, enabled):
    if not enabled:
        return {
            "healthy": True,
            "service": "disabled",
            "collection_exists": False,
            "errors": [],
        }
    try:
        report = orchestrator.qdrant.health_report()
        return (
            report
            if isinstance(report, dict)
            else {
                "healthy": False,
                "service": "unavailable",
                "errors": ["invalid_health_report"],
            }
        )
    except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
        return {
            "healthy": False,
            "service": "unavailable",
            "errors": ["qdrant_probe_failed"],
        }


def _sqlite_health(store):
    if not store.db.is_file():
        return "missing"
    try:
        uri = f"file:{store.db.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        try:
            result = connection.execute("PRAGMA quick_check").fetchone()
        finally:
            connection.close()
        return "available" if result and result[0] == "ok" else "unhealthy"
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return "unavailable"


def _provider_health(providers):
    return {
        name: _provider_status(_provider_probe(provider))
        for name, provider in sorted(providers.items())
    }


def _provider_probe(provider):
    if hasattr(provider, "health_report"):
        try:
            return provider.health_report()
        except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
            return "unavailable"
    return _component_health(provider.health)


def _provider_status(report):
    return report.status if isinstance(report, ProviderHealth) else report


def _model_available(report, model_id, discovered):
    if isinstance(report, ProviderHealth):
        return report.model_available(model_id)
    return report == "available" and model_id in discovered


def _config_health(conf):
    try:
        return conf.validate()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _provider_roles(conf):
    roles = {}
    models = conf.data.get("models", {})
    registry = models.get("registry", {})
    for profile in conf.data.get("profiles", {}).values():
        model = profile.get("model", {})
        for role in ("primary", "fallback"):
            references = (
                model.get(role, []) if role == "fallback" else [model.get(role)]
            )
            for reference in references:
                if not reference:
                    continue
                provider = registry.get(reference, {}).get("provider", reference)
                previous = roles.get(provider)
                if previous != "primary":
                    roles[provider] = role
    return roles


def _start_preflight(conf, store, orchestrator, *, provider_timeout=5.0):
    """Check mandatory local startup dependencies and report optional services."""
    checks = [
        {
            "name": "Configuration",
            "status": "available" if _config_health(conf) else "unavailable",
            "critical": True,
        },
        {
            "name": "SQLite",
            "status": "available"
            if _sqlite_health(store) == "available"
            else "unavailable",
            "critical": True,
        },
    ]
    provider_config = conf.data.get("models", {}).get("providers", {})
    providers = orchestrator.models.providers
    roles = _provider_roles(conf)
    enabled = {name for name, value in provider_config.items() if value.get("enabled")}
    probes = {name: providers[name] for name in roles.keys() & providers.keys()}
    results = {}
    executor = ThreadPoolExecutor(max_workers=max(1, min(len(probes), 8)))
    futures = {
        executor.submit(_provider_probe, item): name for name, item in probes.items()
    }
    try:
        completed, pending = wait(futures, timeout=provider_timeout)
        for future in completed:
            result = future.result()
            results[futures[future]] = _provider_status(result)
        for future in pending:
            future.cancel()
            results[futures[future]] = "unavailable"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for name in sorted(provider_config.keys() | roles.keys()):
        role = roles.get(name, "unused")
        if name in provider_config and name not in enabled:
            state = "disabled"
        elif name not in provider_config:
            state = "not_configured"
        elif name not in roles:
            state = "unused"
        elif name not in providers:
            state = "unavailable"
        else:
            state = results.get(name, "unavailable")
        checks.append(
            {
                "name": f"Provider {name}",
                "status": state,
                "critical": False,
                "role": role,
            }
        )

    qdrant_enabled = conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
    qdrant_state = _component_health(orchestrator.qdrant.health, enabled=qdrant_enabled)
    checks.append({"name": "Qdrant", "status": qdrant_state, "critical": False})
    return checks


def _print_preflight(checks):
    for check in checks:
        state = check["status"]
        critical = check["critical"]
        symbol = "✓" if state == "available" else ("✗" if critical else "⚠")
        classification = "required" if critical else "optional"
        role = f" [{check['role']}]" if "role" in check else ""
        typer.echo(f"{symbol} {check['name']}{role}: {state} ({classification})")


@app.command()
def start(host: str | None = None, port: int | None = None):
    try:
        conf, store, orchestrator = build()
    except (OSError, RuntimeError, ValueError, sqlite3.Error, yaml.YAMLError):
        typer.echo(
            "Start preflight failed: configuration or local database unavailable"
        )
        raise typer.Exit(1) from None
    checks = _start_preflight(conf, store, orchestrator)
    _print_preflight(checks)
    if any(check["critical"] and check["status"] != "available" for check in checks):
        typer.echo("Harness start blocked by a required preflight check")
        raise typer.Exit(1)
    settings = conf.data.get("api", {})
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
    counts = orchestrator.service.status_counts()
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
    qdrant_report = _qdrant_report(orchestrator, enabled=qdrant_enabled)
    typer.echo(f"Qdrant: {qdrant_report['service']}")
    if qdrant_enabled:
        typer.echo("Qdrant evidence: " + json.dumps(qdrant_report, sort_keys=True))
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")
    for name, provider in orchestrator.models.providers.items():
        report = _provider_probe(provider)
        typer.echo(f"Provider {name}: {_provider_status(report)}")


@app.command("metrics")
def runtime_metrics():
    """Print durable task, event-catalogue, and model-usage counters."""
    _conf, _store, orchestrator = build()
    typer.echo(json.dumps(orchestrator.observability.metrics(), indent=2))


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
    qdrant_report = _qdrant_report(orchestrator, enabled=qdrant_enabled)
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
        "Qdrant": qdrant_report["healthy"],
        "Qdrant Collection": not qdrant_enabled
        or qdrant_report.get("collection_exists", False),
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
    for name, state in providers.items():
        typer.echo(f"Provider {name} status: {state}")
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


@memory.command("status")
def memory_status():
    """Report configured Obsidian and MCP vault state without writing files."""
    conf, _, _ = build()
    from .memory_projection import DecisionProjection

    settings = conf.data.get("memory", {}).get("obsidian", {})
    servers = conf.data.get("tools", {}).get("mcp", {}).get("servers", {})
    status = DecisionProjection(conf.path("obsidian_vault")).status(
        obsidian_enabled=settings.get("enabled", False),
        mcp_enabled=servers.get("vault", {}).get("enabled", False),
    )
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))


@memory.command("audit")
def memory_audit():
    """Audit curated Vault notes, links, and review dates without writing."""
    from .vault_audit import audit_vault

    try:
        conf = Config()
        report = audit_vault(conf.path("obsidian_vault"))
    except (OSError, ValueError) as error:
        typer.echo(f"Vault audit unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise typer.Exit(1)


@memory.command("qdrant-status")
def qdrant_status():
    """Show Qdrant health, collection contract, and persisted point counts."""
    conf, _store, orchestrator = build()
    enabled = conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
    report = _qdrant_report(orchestrator, enabled=enabled)
    report["enabled"] = enabled
    if not enabled:
        report["healthy"] = True
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if enabled and not report["healthy"]:
        raise typer.Exit(1)


@memory.command("qdrant-init")
def qdrant_init(confirm: bool = typer.Option(False, "--confirm")):
    """Create the configured collection or validate its vector contract."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm to initialize the collection."
        )
        raise typer.Exit(2)
    conf, _store, orchestrator = build()
    qdrant = conf.data.get("memory", {}).get("qdrant", {})
    if not qdrant.get("enabled", False):
        typer.echo("Qdrant is disabled in configuration")
        raise typer.Exit(1)
    try:
        orchestrator.qdrant.ensure_collection()
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Qdrant collection initialization failed: {type(error).__name__}")
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "initialized": True,
                "collection": qdrant.get("collection", "harness-memory"),
                "dimension": conf.data.get("memory", {})
                .get("embeddings", {})
                .get("dimensions", 1024),
                "distance": "Cosine",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@memory.command("qdrant-search")
def qdrant_search(query: str, limit: int = 5):
    """Search indexed memory using the configured embedding provider."""
    if not query.strip():
        typer.echo("Query must not be empty")
        raise typer.Exit(2)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        typer.echo("Limit must be an integer between 1 and 20")
        raise typer.Exit(2)
    conf, _store, orchestrator = build()
    if not conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False):
        typer.echo("Qdrant is disabled in configuration")
        raise typer.Exit(1)
    try:
        result = orchestrator.qdrant.search(query, limit=limit)
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Qdrant search failed: {type(error).__name__}")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@memory.command("qdrant-reconcile")
def qdrant_reconcile(confirm: bool = typer.Option(False, "--confirm")):
    """Reindex vault notes and remove stale Harness-owned Qdrant vectors."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm to reconcile the Qdrant index."
        )
        raise typer.Exit(2)
    conf, _store, orchestrator = build()
    if not conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False):
        typer.echo("Qdrant is disabled in configuration")
        raise typer.Exit(1)
    try:
        result = orchestrator.memory_service.reconcile()
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Qdrant reconciliation failed: {type(error).__name__}")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@tasks.command("create")
def task_create(
    title: str | None = typer.Argument(None),
    description: str = typer.Argument(""),
    spec_file: Annotated[Path | None, typer.Option("--file")] = None,
):
    if spec_file is not None:
        if title or description:
            typer.echo("choose title/description or --file")
            raise typer.Exit(1)
        try:
            item = _load_task_spec(spec_file)
        except (OSError, ValueError, PydanticValidationError, yaml.YAMLError):
            typer.echo("invalid task specification")
            raise typer.Exit(1) from None
    else:
        if not title:
            typer.echo("task title is required")
            raise typer.Exit(1)
        item = Task(title=title, description=description)
    _, _, orchestrator = build()
    try:
        item = orchestrator.service.create(item)
    except ValueError:
        typer.echo("task creation failed")
        raise typer.Exit(1) from None
    typer.echo(f"created task {item.id}")


def _load_task_spec(path: Path) -> Task:
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("unsupported task specification format")
    if path.stat().st_size > 1_000_000:
        raise ValueError("task specification is too large")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict) or set(data) & {
        "id",
        "status",
        "created_at",
        "updated_at",
        "result",
        "plan",
        "validation_result",
        "test_result",
        "git_state",
        "decisions",
    }:
        raise ValueError("task specification contains protected fields")
    return Task.model_validate(data)


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
    _, _store, orchestrator = build()
    try:
        events = orchestrator.service.events(task_id)
    except ValueError:
        typer.echo("task not found")
        raise typer.Exit(1) from None
    for event in events:
        payload = json.loads(event["payload"])
        typer.echo(
            f"{event['created_at']} {event['kind']} "
            f"{json.dumps(payload, sort_keys=True)}"
        )


@tasks.command("watch")
def task_watch(task_id: int):
    conf, store, _ = build()
    if store.get(task_id) is None:
        typer.echo("task not found")
        raise typer.Exit(1)
    typer.echo(f"TASK-{task_id}")
    try:
        _watch_task_events(conf, task_id)
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (httpx.HTTPError, ValueError, RuntimeError):
        typer.echo("task event stream unavailable")
        raise typer.Exit(1) from None


def _watch_task_events(conf, task_id: int):
    api = conf.settings.api
    url = f"http://{api.host}:{api.port}/api/events/stream"
    cursor = 0
    failures = 0
    while failures < 3:
        previous = cursor
        try:
            with httpx.stream(
                "GET",
                url,
                params={"task_id": task_id},
                headers={"Last-Event-ID": str(cursor)},
                timeout=30,
            ) as response:
                response.raise_for_status()
                event_id = None
                data_lines = []
                for line in response.iter_lines():
                    if line.startswith(":"):
                        continue
                    if line == "":
                        if data_lines:
                            event = json.loads("\n".join(data_lines))
                            if (
                                not isinstance(event, dict)
                                or not isinstance(event_id, int)
                                or event.get("id") != event_id
                                or event.get("task_id") != task_id
                                or not isinstance(event.get("kind"), str)
                            ):
                                raise ValueError("invalid task event")
                            if event_id <= cursor:
                                event_id, data_lines = None, []
                                continue
                            cursor = event_id
                            typer.echo(
                                f"[{event.get('created_at', '')}] {event['kind']}"
                            )
                            if event["kind"] in {"task.completed", "task.failed"} or (
                                event["kind"] == "task.status"
                                and isinstance(event.get("payload"), dict)
                                and event["payload"].get("status") == "cancelled"
                            ):
                                return
                        event_id, data_lines = None, []
                    elif line.startswith("id:"):
                        try:
                            event_id = int(line[3:].strip())
                        except ValueError as exc:
                            raise ValueError("invalid event cursor") from exc
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
        except (httpx.HTTPError, ValueError):
            failures += 1
        else:
            failures = 0 if cursor > previous else failures + 1
    raise RuntimeError("event stream disconnected")


@models.command("list")
def model_list():
    _, _, orchestrator = build()
    registry = orchestrator.models
    providers = registry.providers
    definitions = registry.models
    names = set(providers) | {item.get("provider") for item in definitions.values()}
    for name in sorted(item for item in names if isinstance(item, str)):
        provider = providers.get(name)
        report = _provider_probe(provider) if provider is not None else "unavailable"
        state = _provider_status(report)
        typer.echo(f"{name}\t{state}")
        discovered = []
        if isinstance(report, ProviderHealth):
            discovered = report.models
            registry.discovered[name] = tuple(sorted(set(discovered)))
            if not report.api_available:
                typer.echo(f"{name}\t<discovery failed>")
        elif provider is not None:
            try:
                discovered = registry.discover(name)
            except (OSError, RuntimeError, ValueError, httpx.HTTPError):
                typer.echo(f"{name}\t<discovery failed>")
        configured = {
            alias: item
            for alias, item in definitions.items()
            if item.get("provider") == name
        }
        for alias, item in sorted(configured.items()):
            model_id = item.get("model") or "-"
            tier = item.get("tier") or "unknown"
            model_state = (
                "available"
                if _model_available(report, model_id, discovered)
                else "unavailable"
            )
            typer.echo(f"{model_id}\t{name}\t{alias}\t{tier}\t{model_state}")
        for model_id in sorted(
            set(discovered) - {item.get("model") for item in configured.values()}
        ):
            model_state = (
                "available"
                if _model_available(report, model_id, discovered)
                else "unavailable"
            )
            typer.echo(f"{model_id}\t{name}\t-\tunknown\t{model_state}")


@models.command("status")
def model_status():
    model_list()


@models.command("usage")
def model_usage(
    task_id: int | None = None,
    agent: str | None = None,
    profile: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    group_by: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Show persisted model-call usage without exposing prompts or secrets."""
    _, _store, orchestrator = build()
    try:
        report = orchestrator.service.model_usage(
            task_id=task_id,
            agent=agent,
            profile=profile,
            provider=provider,
            model=model,
            group_by=group_by,
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
    try:
        typer.echo("\n".join(SecretResolver().list_names()))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        typer.echo("secret operation failed")
        raise typer.Exit(1) from None


@secrets.command("set")
def secret_set(name: str):
    value = typer.prompt("Secret", hide_input=True, confirmation_prompt=True)
    try:
        SecretResolver().set(name, value)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        typer.echo("secret operation failed")
        raise typer.Exit(1) from None
    typer.echo(f"stored {name} in macOS Keychain")


@secrets.command("delete")
def secret_delete(name: str):
    try:
        deleted = SecretResolver().delete(name)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        typer.echo("secret operation failed")
        raise typer.Exit(1) from None
    if not deleted:
        typer.echo("secret not found")
        raise typer.Exit(1)
    typer.echo(f"deleted {name} from macOS Keychain")


@secrets.command("exists")
def secret_exists(name: str):
    try:
        found = SecretResolver().exists(name)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        typer.echo("secret operation failed")
        raise typer.Exit(1) from None
    typer.echo("exists" if found else "not found")
    if not found:
        raise typer.Exit(1)
