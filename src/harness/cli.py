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
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import httpx
import typer
import uvicorn
import yaml
from pydantic import ValidationError as PydanticValidationError

from .completion import audit_gap_matrix, audit_harness_completion
from .core import ROOT, Config, Task, build
from .database import Database, OperationalSnapshotRepository
from .docker_broker import DockerComposeBroker
from .isolation_diagnostics import probe_sandbox_capability
from .mcp_status import MCPStatusApplicationService
from .prometheus_acceptance import check_prometheus_integration
from .prometheus_audit import audit_prometheus_scrape, audit_prometheus_ui
from .security import SecretResolver
from .service_lifecycle import ServiceLifecycle
from .services import (
    ModelOperationsService,
    OperationalDiagnosticsService,
)
from .sqlite_operations import inspect_sqlite
from .verification import source_tree_sha256
from .verify_clusters import write_failure_clusters

app = typer.Typer(no_args_is_help=True)
tasks = typer.Typer(no_args_is_help=True)
models = typer.Typer(no_args_is_help=True)
config = typer.Typer(no_args_is_help=True)
secrets = typer.Typer(no_args_is_help=True)
artifacts = typer.Typer(no_args_is_help=True)
memory = typer.Typer(no_args_is_help=True)
memory_service = typer.Typer(no_args_is_help=True)
evidence = typer.Typer(no_args_is_help=True)
mcp = typer.Typer(no_args_is_help=True)
isolation = typer.Typer(no_args_is_help=True)
observability = typer.Typer(no_args_is_help=True)
app.add_typer(tasks, name="tasks")
app.add_typer(models, name="models")
app.add_typer(config, name="config")
app.add_typer(secrets, name="secrets")
app.add_typer(artifacts, name="artifacts")
app.add_typer(memory, name="memory")
memory.add_typer(memory_service, name="service")
app.add_typer(evidence, name="evidence")
app.add_typer(mcp, name="mcp")
app.add_typer(isolation, name="isolation")
app.add_typer(observability, name="observability")


@isolation.command("doctor")
def isolation_doctor(
    output: Annotated[Path | None, typer.Option("--output")] = None,
):
    """Probe nested sandbox support in a disposable directory without skipping tests."""
    try:
        report = probe_sandbox_capability()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        typer.echo(f"Isolation capability probe failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["supported"]:
        raise typer.Exit(1)


@app.command("verify-clusters")
def verify_clusters(
    junit: Annotated[Path, typer.Option("--junit")] = Path(
        "data/verification-junit.xml"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path(
        "data/verification-failure-clusters.json"
    ),
):
    """Summarize failed tests from a JUnit report; never changes their result."""
    try:
        report = write_failure_clusters(junit, output)
    except (OSError, ET.ParseError, ValueError) as error:
        typer.echo(f"Could not cluster verification failures: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(f"JUnit failures: {report['failed']}")
    for cluster in report["clusters"]:
        typer.echo(f"{cluster['category']}: {cluster['count']}")
    typer.echo(f"Detailed report: {output}")


@mcp.command("status")
def mcp_status():
    """Show configured MCP servers without starting them."""
    from .mcp_manager import MCPServerManager

    conf = Config()
    manager = MCPServerManager(conf.data.get("tools", {}).get("mcp", {}))
    service = MCPStatusApplicationService(
        manager.report,
        lambda: OperationalSnapshotRepository.read_latest_mcp_statuses(
            conf.path("database")
        ),
    )
    typer.echo(json.dumps(service.status(), ensure_ascii=False, indent=2))


@mcp.command("doctor")
def mcp_doctor():
    """Probe each enabled MCP server independently."""
    from .core import Permissions
    from .tools import ToolRegistry

    try:
        conf = Config()
        registry = ToolRegistry(Permissions(conf))
        reports = registry.mcp_status(probe=True)
        OperationalSnapshotRepository(
            Database(conf.path("database"))
        ).record_mcp_status(reports)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"MCP doctor could not initialize: {type(error).__name__}")
        raise typer.Exit(2) from None
    reports = MCPStatusApplicationService(
        list,
        lambda: OperationalSnapshotRepository.read_latest_mcp_statuses(
            conf.path("database")
        ),
    ).status(reports)
    typer.echo(json.dumps(reports, ensure_ascii=False, indent=2))
    if any(item["state"] in {"unavailable", "invalid_config"} for item in reports):
        raise typer.Exit(1)


@app.command("decisions")
def decision_query(
    task_id: int | None = typer.Option(None, "--task-id", min=1),
    category: str | None = typer.Option(None, "--category"),
    source: str | None = typer.Option(None, "--source"),
    tag: str | None = typer.Option(None, "--tag"),
    decision_id: int | None = typer.Option(None, "--id", min=1),
):
    """Read decisions by ID or with exact task/category/source/tag filters."""
    from .decisions import DecisionService

    _config, store, _orchestrator = build()
    service = DecisionService(store)
    if decision_id is not None:
        decision = service.get(decision_id)
        if decision is None:
            typer.echo("decision not found")
            raise typer.Exit(1)
        rows = [decision]
    else:
        rows = service.query(task_id, category, source, tag)
    typer.echo(
        json.dumps([row.model_dump(mode="json") for row in rows], ensure_ascii=False)
    )


@artifacts.command("list")
def artifact_list(task_id: int = typer.Option(..., "--task-id", min=1)):
    """List the latest version of each task artifact."""
    _config, _store, orchestrator = build()
    typer.echo(json.dumps(orchestrator.service.artifacts(task_id), ensure_ascii=False))


@artifacts.command("history")
def artifact_history(
    task_id: int = typer.Option(..., "--task-id", min=1),
    key: str = typer.Option(..., "--key"),
):
    """Show append-only versions of one task artifact."""
    _config, _store, orchestrator = build()
    typer.echo(
        json.dumps(
            orchestrator.service.artifact_history(task_id, key), ensure_ascii=False
        )
    )


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
        settings = Config().settings.verification
        required = settings.required_evidence
        if required:
            from .database import Database
            from .services import VerificationEvidenceService

            database_path = Config().path("database")
            if database_path.is_file():
                evidence_service = VerificationEvidenceService(Database(database_path))
                valid_by_kind = {
                    kind: evidence_service.audit_kind(
                        kind,
                        subject_sha256=source_tree_sha256(ROOT),
                        max_age_hours=settings.max_age_hours,
                    )["healthy"]
                    for kind in required
                }
            else:
                valid_by_kind = {kind: False for kind in required}
            report["required_evidence"] = valid_by_kind
            report["evidence_policy_valid"] = all(valid_by_kind.values())
            if not report["evidence_policy_valid"]:
                report["complete"] = False
                report["reasons"].append(
                    "required external verification evidence is missing or invalid"
                )
        else:
            report["required_evidence"] = {}
            report["evidence_policy_valid"] = True
    except (OSError, ValueError, sqlite3.Error) as error:
        typer.echo(f"Harness completion: unverifiable ({type(error).__name__})")
        raise typer.Exit(2) from None
    report["fulfilled"] = matrix_report["fulfilled"]
    report["partial"] = matrix_report["partial"]
    report["open"] = matrix_report["open"]
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise typer.Exit(1)


def _evidence_repository():
    from .database import Database
    from .services import VerificationEvidenceService

    return VerificationEvidenceService(Database(Config().path("database")))


@evidence.command("import")
def evidence_import(source: Path):
    """Import one strict JSON verification record into append-only SQLite."""
    from .evidence import EvidenceInput

    try:
        payload = EvidenceInput.model_validate_json(source.read_text(encoding="utf-8"))
        result = _evidence_repository().record(payload)
    except (OSError, ValueError) as error:
        typer.echo(f"Evidence import rejected: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@evidence.command("list")
def evidence_list(
    kind: str | None = typer.Option(None, "--kind"),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
):
    """List metadata-only imported evidence records."""
    try:
        records = _evidence_repository().list(kind=kind, limit=limit)
    except ValueError as error:
        typer.echo(f"Evidence query rejected: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(records, ensure_ascii=False, indent=2))


@evidence.command("audit")
def evidence_audit(
    subject_sha256: str = typer.Option(..., "--subject-sha256"),
    max_age_hours: int = typer.Option(168, "--max-age-hours", min=1, max=8760),
):
    """Check evidence freshness and exact source-tree hash binding."""
    try:
        report = _evidence_repository().audit(
            subject_sha256=subject_sha256, max_age_hours=max_age_hours
        )
    except ValueError as error:
        typer.echo(f"Evidence audit rejected: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise typer.Exit(1)


def _lifecycle():
    return ServiceLifecycle(ROOT / "data" / "harness.pid")


def _serve_api(
    host,
    port,
    lifecycle,
    timeout=10.0,
    *,
    metrics_listener=None,
    metrics_token=None,
    metrics_provider=None,
):
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

    metrics_server = None
    metrics_thread = None
    if metrics_listener is not None and metrics_listener.enabled:
        if not metrics_token or not callable(metrics_provider):
            return False
        from .metrics_api import create_metrics_app

        metrics_started = threading.Event()

        class ReadyMetricsServer(uvicorn.Server):
            async def startup(self, sockets=None):
                try:
                    await super().startup(sockets)
                    if self.started:
                        startup_succeeded.set()
                finally:
                    metrics_started.set()

        startup_succeeded = threading.Event()
        metrics_server = ReadyMetricsServer(
            uvicorn.Config(
                create_metrics_app(metrics_provider, metrics_token),
                host=metrics_listener.host,
                port=metrics_listener.port,
                access_log=False,
                log_config=None,
            )
        )
        metrics_thread = threading.Thread(target=metrics_server.run, daemon=True)
        metrics_thread.start()
        if not metrics_started.wait(timeout) or not startup_succeeded.is_set():
            metrics_server.should_exit = True
            metrics_thread.join(timeout)
            return False

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
    try:
        server.run()
    finally:
        if metrics_server is not None:
            metrics_server.should_exit = True
            metrics_thread.join(timeout)
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
            foreign_key_issues = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        finally:
            connection.close()
        return (
            "available"
            if result and result[0] == "ok" and not foreign_key_issues
            else "unhealthy"
        )
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return "unavailable"


def _provider_health(providers):
    return ModelOperationsService.providers_health(providers)


def _provider_probe(provider):
    return ModelOperationsService.probe_provider(provider)


def _provider_status(report):
    return ModelOperationsService.provider_status(report)


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


def _diagnostics_service(conf, store, orchestrator):
    return OperationalDiagnosticsService(
        conf,
        store,
        orchestrator,
        lifecycle_factory=_lifecycle,
        config_health=_config_health,
        sqlite_health=_sqlite_health,
        qdrant_reporter=_qdrant_report,
        docker_broker_factory=DockerComposeBroker,
        sqlite_inspector=inspect_sqlite,
        which=shutil.which,
        run=subprocess.run,
        system=platform.system,
        machine=platform.machine,
        access=os.access,
        connect=socket.create_connection,
        provider_roles=_provider_roles,
        provider_probe=_provider_probe,
        provider_status=_provider_status,
        component_health=_component_health,
        executor_factory=ThreadPoolExecutor,
        wait_for=wait,
    )


def _start_preflight(conf, store, orchestrator, *, provider_timeout=5.0):
    """Compatibility wrapper; start-check ownership lives in the application service."""
    return _diagnostics_service(conf, store, orchestrator).start_preflight(
        provider_timeout=provider_timeout
    )


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
    metrics_listener = conf.settings.api.metrics_listener
    metrics_token = None
    if metrics_listener.enabled:
        from .metrics_api import read_bearer_token

        try:
            metrics_token = read_bearer_token(metrics_listener.bearer_file)
        except (OSError, ValueError):
            typer.echo("Start blocked: metrics listener token file is unavailable")
            raise typer.Exit(1)
    lifecycle = _lifecycle()
    try:
        record = lifecycle.register_current(host, port)
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    try:
        if metrics_listener.enabled:
            ready = _serve_api(
                host,
                port,
                lifecycle,
                metrics_listener=metrics_listener,
                metrics_token=metrics_token,
                metrics_provider=orchestrator.observability_operations.prometheus,
            )
        else:
            ready = _serve_api(host, port, lifecycle)
        if not ready:
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
    report = _diagnostics_service(conf, store, orchestrator).status_report()
    details = report.details
    typer.echo(f"Harness: {details['service']}")
    typer.echo(f"SQLite: {details['sqlite']}")
    schema = details["sqlite_schema"]
    typer.echo(
        f"SQLite schema: {schema['schema_version']}/"
        f"{schema['expected_schema_version']} ({schema['status']})"
    )
    typer.echo(f"Obsidian: {_component_health(lambda: details['vault'])}")
    qdrant_report = details["qdrant"]
    typer.echo(f"Qdrant: {qdrant_report['service']}")
    if conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False):
        typer.echo("Qdrant evidence: " + json.dumps(qdrant_report, sort_keys=True))
    for name, count in details["counts"].items():
        typer.echo(f"{name}: {count}")
    for name, state in details["providers"].items():
        typer.echo(f"Provider {name}: {state}")


@app.command("metrics")
def runtime_metrics(
    format: str = typer.Option("json", "--format", case_sensitive=False),
):
    """Print durable task, event-catalogue, and model-usage counters."""
    _conf, _store, orchestrator = build()
    if format.casefold() == "prometheus":
        typer.echo(orchestrator.observability_operations.prometheus(), nl=False)
        return
    if format.casefold() != "json":
        typer.echo("Metrics format must be json or 'prometheus'.")
        raise typer.Exit(2)
    typer.echo(json.dumps(orchestrator.observability_operations.metrics(), indent=2))


@observability.command("audit-prometheus-ui")
def audit_prometheus_ui_command(
    compose_file: Annotated[
        Path, typer.Option(..., "--compose-file", exists=True, dir_okay=False)
    ],
    web_config_file: Annotated[
        Path, typer.Option(..., "--web-config-file", exists=True, dir_okay=False)
    ],
):
    """Read-only audit of Prometheus UI exposure, TLS, auth, and config mount."""
    try:
        compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        web_config = yaml.safe_load(web_config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        typer.echo(
            f"Prometheus UI audit could not read configuration: {type(error).__name__}"
        )
        raise typer.Exit(2) from None
    report = audit_prometheus_ui(compose, web_config)
    typer.echo(json.dumps(report, sort_keys=True))
    if report["status"] in {"unsafe", "unknown"}:
        raise typer.Exit(1)


@observability.command("audit-prometheus-scrape")
def audit_prometheus_scrape_command(
    scrape_config_file: Annotated[
        Path, typer.Option(..., "--scrape-config-file", exists=True, dir_okay=False)
    ],
):
    """Read-only audit of the documented Prometheus-to-Harness scrape contract."""
    try:
        scrape_config = yaml.safe_load(scrape_config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        typer.echo(
            f"Prometheus scrape audit could not read configuration: {type(error).__name__}"
        )
        raise typer.Exit(2) from None
    report = audit_prometheus_scrape(scrape_config)
    typer.echo(json.dumps(report, sort_keys=True))
    if report["status"] != "valid":
        raise typer.Exit(1)


@observability.command("check-prometheus")
def check_prometheus_command(
    prometheus_url: Annotated[str, typer.Option(..., "--prometheus-url")],
    harness_metrics_url: Annotated[str, typer.Option(..., "--harness-metrics-url")],
):
    """Live-check Prometheus readiness, Harness scrape, query, and 401 contract."""
    report = check_prometheus_integration(prometheus_url, harness_metrics_url)
    typer.echo(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise typer.Exit(1)


@app.command()
def doctor():
    conf, store, orchestrator = build()
    report = _diagnostics_service(conf, store, orchestrator).doctor_report()
    providers = report.details["providers"]
    mcp_servers = report.details["mcp"]
    for label, ok in report.checks.items():
        typer.echo(f"{'✓' if ok else '✗'} {label}")
    for name, state in providers.items():
        typer.echo(f"Provider {name} status: {state}")
    for item in mcp_servers:
        typer.echo(f"MCP {item['name']} status: {item['state']}")
    if not report.healthy:
        raise typer.Exit(1)


@app.command("sqlite-health")
def sqlite_health():
    """Inspect the configured SQLite database without opening it for writes."""
    report = inspect_sqlite(Config().path("database"))
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"] != "available":
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


@app.command("qdrant-upgrade-smoke")
def qdrant_upgrade_smoke(
    baseline_image: Annotated[str, typer.Option("--baseline-image")],
    candidate_image: Annotated[str, typer.Option("--candidate-image")],
    confirm: bool = typer.Option(False, "--confirm"),
):
    """Test a pinned Qdrant candidate against an isolated throwaway volume."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm to pull and test the pinned image transition."
        )
        raise typer.Exit(2)
    try:
        result = DockerComposeBroker().upgrade_smoke_test(
            baseline_image, candidate_image
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Isolated Qdrant upgrade test failed: {type(error).__name__}")
        raise typer.Exit(1) from None
    typer.echo(
        "Isolated Qdrant upgrade test passed: "
        f"baseline={result['baseline_image']} "
        f"candidate={result['candidate_image']} "
        f"collection_readable={result['collection_readable']} "
        f"persisted_after_restart={result['persisted_after_restart']} "
        f"cleaned={result['cleaned']}"
    )


@memory.command("sync")
def memory_sync():
    """Rebuild the Obsidian decision-note projection from canonical SQLite."""
    try:
        conf = Config()
        database_path = conf.path("database")
        if not database_path.is_file():
            raise FileNotFoundError("canonical SQLite database does not exist")
        with closing(
            sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        decisions = []
        for row in rows:
            item = dict(row)
            for field in ("evidence", "field_names", "alternatives", "tags"):
                item[field] = json.loads(item[field])
            decisions.append(item)
    except (OSError, sqlite3.Error, ValueError) as error:
        typer.echo(f"Obsidian decision projection unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    from .memory_projection import DecisionProjection

    result = DecisionProjection(conf.path("obsidian_vault")).sync(decisions)
    typer.echo(
        "Obsidian decision projection: "
        + ", ".join(f"{key}={value}" for key, value in result.items())
    )


def _build_memory_store():
    """Build only configuration and SQLite services for memory operations."""
    from .core import Store

    conf = Config()
    return conf, Store(conf)


def _memory_qdrant(conf):
    """Construct the Qdrant health client without initializing MCP tools."""
    from .memory import QdrantMemory

    memory = conf.data.get("memory", {})
    qdrant = memory.get("qdrant", {})
    embeddings = memory.get("embeddings", {})
    return QdrantMemory(
        qdrant.get("url", "http://127.0.0.1:6333"),
        qdrant.get("collection", "harness-memory"),
        embeddings.get("dimensions", 1024),
        timeout=qdrant.get("timeout", 5),
        api_key=conf.data.get("secrets", {}).get("QDRANT__SERVICE__API_KEY"),
    )


def _build_embedding_acceptance():
    """Build the live search path without loading the MCP tool registry."""
    from .memory import EmbeddingProvider, ObsidianMemory
    from .memory_service import MemoryService

    conf, store = _build_memory_store()
    embeddings = conf.data.get("memory", {}).get("embeddings", {})
    qdrant = _memory_qdrant(conf)
    qdrant.embedder = EmbeddingProvider(
        embeddings.get("base_url", "http://127.0.0.1:1234/v1"),
        embeddings.get("model", "text-embedding-model"),
        timeout=embeddings.get("timeout", 30),
        dimension=embeddings.get("dimensions", 1024),
        batch_size=embeddings.get("batch_size", 32),
    )
    service = MemoryService(
        ObsidianMemory(conf.path("obsidian_vault")),
        qdrant,
        batch_size=embeddings.get("batch_size", 32),
    )
    return conf, store, qdrant, service


@memory.command("status")
def memory_status():
    """Report configured Obsidian and MCP vault state without writing files."""
    conf, store = _build_memory_store()
    from .database import MemoryOpsEvidenceRepository, OperationalSnapshotRepository
    from .memory_projection import DecisionProjection

    settings = conf.data.get("memory", {}).get("obsidian", {})
    servers = conf.data.get("tools", {}).get("mcp", {}).get("servers", {})
    status = DecisionProjection(conf.path("obsidian_vault")).status(
        obsidian_enabled=settings.get("enabled", False),
        mcp_enabled=servers.get("vault", {}).get("enabled", False),
    )
    status["last_memory_monitor"] = OperationalSnapshotRepository(
        store.database
    ).latest_memory_health()
    latest_acceptance = MemoryOpsEvidenceRepository(store.database).list(limit=1)
    status["last_memory_ops_acceptance"] = (
        latest_acceptance[0] if latest_acceptance else None
    )
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))


@memory.command("watch")
def memory_watch(once: bool = typer.Option(False, "--once")):
    """Run opt-in, read-only memory health probes and persist their summaries."""
    conf, store = _build_memory_store()
    settings = conf.data.get("memory", {}).get("monitoring", {})
    if not settings.get("enabled", False):
        typer.echo("Memory monitoring is disabled in configuration")
        raise typer.Exit(2)

    from .database import OperationalSnapshotRepository
    from .memory_monitor import MemoryHealthMonitor, collect_memory_health

    snapshots = OperationalSnapshotRepository(store.database)
    source_hash = source_tree_sha256(ROOT)
    monitor = MemoryHealthMonitor(
        lambda: collect_memory_health(
            conf,
            store,
            _memory_qdrant(conf),
            source_sha256=source_hash,
        ),
        snapshots,
        interval_seconds=settings.get("interval_seconds", 60),
        emit=lambda report: typer.echo(json.dumps(report, ensure_ascii=False)),
    )
    try:
        monitor.run(max_checks=1 if once else None)
    except KeyboardInterrupt:
        typer.echo("Memory monitoring stopped")
        return
    if once:
        latest = snapshots.latest_memory_health()
        if latest is not None and not latest["healthy"]:
            raise typer.Exit(1)


def _launchd_memory_watch_service():
    from .memory_watch_service import LaunchdMemoryWatchService

    return LaunchdMemoryWatchService(
        home=Path.home(),
        root=ROOT,
        executable=Path(os.path.abspath(os.sys.argv[0])),
    )


@memory_service.command("install")
def memory_service_install(confirm: bool = typer.Option(False, "--confirm")):
    """Install and start the explicitly enabled memory watcher LaunchAgent."""
    conf = Config()
    enabled = conf.data.get("memory", {}).get("monitoring", {}).get("enabled", False)
    try:
        result = _launchd_memory_watch_service().install(
            confirm=confirm, monitoring_enabled=enabled
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Memory watcher service install failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result))


@memory_service.command("status")
def memory_service_status():
    """Report whether the managed memory watcher LaunchAgent is installed."""
    try:
        result = _launchd_memory_watch_service().status()
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Memory watcher service status failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result))


@memory_service.command("uninstall")
def memory_service_uninstall(confirm: bool = typer.Option(False, "--confirm")):
    """Stop and remove only the managed memory watcher LaunchAgent."""
    try:
        result = _launchd_memory_watch_service().uninstall(confirm=confirm)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Memory watcher service uninstall failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result))


@memory.command("acceptance")
def memory_ops_acceptance(confirm: bool = typer.Option(False, "--confirm")):
    """Record operational evidence after five healthy monitored minutes."""
    if confirm is not True:
        typer.echo("Memory operational acceptance requires --confirm")
        raise typer.Exit(2)
    conf, store = _build_memory_store()
    settings = conf.data.get("memory", {}).get("monitoring", {})
    if settings.get("enabled", False) is not True:
        typer.echo("Memory monitoring is disabled in configuration")
        raise typer.Exit(2)

    from .database import MemoryOpsEvidenceRepository, OperationalSnapshotRepository
    from .memory_ops_acceptance import (
        evaluate_memory_ops_acceptance,
        required_snapshot_count,
    )

    interval = settings.get("interval_seconds", 60)
    try:
        limit = required_snapshot_count(interval)
        launchd_status = _launchd_memory_watch_service().status()
        snapshots = OperationalSnapshotRepository(store.database).recent_memory_health(
            limit=limit
        )
        report = evaluate_memory_ops_acceptance(
            snapshots,
            launchd_status=launchd_status,
            source_sha256=source_tree_sha256(ROOT),
            interval_seconds=interval,
        )
        if not report["passed"]:
            typer.echo(json.dumps(report, ensure_ascii=False))
            raise typer.Exit(1)
        evidence = MemoryOpsEvidenceRepository(store.database).record(
            subject_sha256=report["source_sha256"],
            observed_at=datetime.now(UTC),
            checks=report["checks"],
        )
    except typer.Exit:
        raise
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        typer.echo(f"Memory operational acceptance failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(
        json.dumps(
            {
                "acceptance": report,
                "evidence_id": evidence["id"],
                "digest": evidence["digest"],
            },
            ensure_ascii=False,
        )
    )


@memory.command("audit")
def memory_audit():
    """Audit curated Vault notes, links, and review dates without writing."""
    from .vault_audit import audit_vault

    try:
        conf = Config()
        decision_rows = None
        database_path = conf.path("database")
        if database_path.is_file():
            with closing(
                sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            ) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT * FROM decisions ORDER BY id"
                ).fetchall()
            decision_rows = []
            for row in rows:
                item = dict(row)
                for field in ("evidence", "field_names", "alternatives", "tags"):
                    item[field] = json.loads(item[field])
                decision_rows.append(item)
        report = audit_vault(
            conf.path("obsidian_vault"),
            decision_rows=decision_rows,
            content_governance=True,
            source_root=ROOT,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Vault audit unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise typer.Exit(1)


@memory.command("knowledge-propose")
def memory_knowledge_propose(
    body_file: Annotated[
        Path, typer.Option("--body-file", exists=True, dir_okay=False, readable=True)
    ],
    task_id: str = typer.Option(..., "--task-id"),
    title: str = typer.Option(..., "--title"),
    category: str = typer.Option(..., "--category"),
    source: str = typer.Option(..., "--source"),
):
    """Create a source-hash-bound proposal; this does not edit curated notes."""
    from .vault_steward import VaultKnowledgeSteward

    try:
        _conf, store = _build_memory_store()
        body = body_file.read_text(encoding="utf-8")
        service = VaultKnowledgeSteward(
            _conf.path("obsidian_vault"), ROOT, redactor=store.audit.sanitize
        )
        result = service.propose(
            task_id=task_id,
            title=title,
            body=body,
            category=category,
            source_path=source,
        )
    except (OSError, UnicodeError, ValueError) as error:
        typer.echo(f"Knowledge proposal failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@memory.command("knowledge-proposals")
def memory_knowledge_proposals():
    """List knowledge proposals and their source evidence for human review."""
    from .vault_steward import VaultKnowledgeSteward

    try:
        conf = Config()
        service = VaultKnowledgeSteward(
            conf.path("obsidian_vault"), ROOT, create_inbox=False
        )
        proposals = service.pending()
    except (OSError, ValueError) as error:
        typer.echo(f"Knowledge proposals unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(proposals, ensure_ascii=False, indent=2))


@memory.command("knowledge-review")
def memory_knowledge_review():
    """Audit Vault health and report pending proposals with stale source evidence."""
    from .vault_audit import audit_vault
    from .vault_steward import VaultKnowledgeSteward

    try:
        conf = Config()
        database_path = conf.path("database")
        decision_rows = None
        if database_path.is_file():
            with closing(
                sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            ) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT * FROM decisions ORDER BY id"
                ).fetchall()
            decision_rows = []
            for row in rows:
                item = dict(row)
                for field in ("evidence", "field_names", "alternatives", "tags"):
                    item[field] = json.loads(item[field])
                decision_rows.append(item)
        vault_report = audit_vault(
            conf.path("obsidian_vault"),
            decision_rows=decision_rows,
            content_governance=True,
            source_root=ROOT,
        )
        proposals = VaultKnowledgeSteward(
            conf.path("obsidian_vault"), ROOT, create_inbox=False
        ).pending()
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        typer.echo(f"Knowledge review unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    stale_pending = [
        item["id"]
        for item in proposals
        if item["status"] == "pending" and item["source_state"] != "current"
    ]
    result = {
        "ready": vault_report["healthy"] and not stale_pending,
        "vault_healthy": vault_report["healthy"],
        "findings": vault_report["findings"],
        "curated_notes": vault_report["audited_notes"],
        "proposal_counts": {
            status: sum(item["status"] == status for item in proposals)
            for status in ("pending", "approved", "rejected")
        },
        "stale_pending_proposals": stale_pending,
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise typer.Exit(1)


@memory.command("knowledge-approve")
def memory_knowledge_approve(
    proposal_id: str,
    sha256: str = typer.Option(..., "--sha256"),
    confirm: bool = typer.Option(False, "--confirm"),
):
    """Apply exactly the reviewed proposal after explicit digest confirmation."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm after reviewing the proposal."
        )
        raise typer.Exit(2)
    from .vault_steward import VaultKnowledgeSteward

    try:
        conf = Config()
        result = VaultKnowledgeSteward(conf.path("obsidian_vault"), ROOT).approve(
            proposal_id, sha256
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Knowledge approval failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@memory.command("knowledge-reject")
def memory_knowledge_reject(
    proposal_id: str,
    sha256: str = typer.Option(..., "--sha256"),
    confirm: bool = typer.Option(False, "--confirm"),
):
    """Record rejection of exactly the reviewed proposal."""
    if not confirm:
        typer.echo(
            "No action taken. Repeat with --confirm after reviewing the proposal."
        )
        raise typer.Exit(2)
    from .vault_steward import VaultKnowledgeSteward

    try:
        conf = Config()
        result = VaultKnowledgeSteward(conf.path("obsidian_vault"), ROOT).reject(
            proposal_id, sha256
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Knowledge rejection failed: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@memory.command("qdrant-config-audit")
def qdrant_config_audit(
    compose_path: Annotated[
        Path, typer.Argument(..., readable=True, exists=True, dir_okay=False)
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", readable=True, exists=True, dir_okay=False),
    ] = None,
):
    """Read-only audit of operator-managed Qdrant Compose/config files."""
    from .qdrant_config_audit import audit_qdrant_compose

    try:
        report = audit_qdrant_compose(compose_path, config_path)
    except (OSError, TypeError, ValueError) as error:
        typer.echo(f"Qdrant configuration audit unavailable: {type(error).__name__}")
        raise typer.Exit(2) from None
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise typer.Exit(1)


@memory.command("qdrant-status")
def qdrant_status():
    """Show Qdrant health, collection contract, and persisted point counts."""
    conf, _store = _build_memory_store()
    enabled = conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
    started = time.monotonic()
    qdrant = _memory_qdrant(conf) if enabled else None
    report = _qdrant_report(SimpleNamespace(qdrant=qdrant), enabled=enabled)
    elapsed_ms = (time.monotonic() - started) * 1000
    if enabled:
        from .database import OperationalSnapshotRepository

        OperationalSnapshotRepository(_store.database).record_qdrant_probe(
            report, elapsed_ms
        )
    report["enabled"] = enabled
    if not enabled:
        report["healthy"] = True
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if enabled and not report["healthy"]:
        raise typer.Exit(1)


@memory.command("qdrant-acceptance")
def qdrant_acceptance(confirm: bool = typer.Option(False, "--confirm")):
    """Read-only live check of the configured embedding→Qdrant search path."""
    if not confirm:
        typer.echo(
            "No provider request sent. Repeat with --confirm to run the live acceptance probe."
        )
        raise typer.Exit(2)
    conf, store, qdrant, memory_service = _build_embedding_acceptance()
    qdrant_config = conf.data.get("memory", {}).get("qdrant", {})
    if not qdrant_config.get("enabled", False):
        typer.echo("Qdrant is disabled in configuration")
        raise typer.Exit(1)
    health = qdrant.health_report()
    if not health.get("healthy"):
        typer.echo(
            json.dumps(
                {
                    "healthy": False,
                    "stage": "qdrant_health",
                    "errors": health.get("errors", []),
                },
                ensure_ascii=False,
            )
        )
        raise typer.Exit(1)
    embedding = conf.data.get("memory", {}).get("embeddings", {})
    expected_model = embedding.get("model")
    expected_dimension = embedding.get("dimensions", 1024)
    provider = qdrant.embedder
    if (
        not isinstance(expected_model, str)
        or not expected_model.strip()
        or provider is None
        or getattr(provider, "model", None) != expected_model
        or getattr(provider, "dimension", None) != expected_dimension
        or qdrant.dimension != expected_dimension
        or health.get("dimension") != expected_dimension
    ):
        typer.echo(
            json.dumps(
                {
                    "healthy": False,
                    "stage": "embedding_contract",
                    "errors": ["configuration_mismatch"],
                }
            )
        )
        raise typer.Exit(1)
    try:
        candidates = qdrant.search("Harness live embedding acceptance probe", limit=1)
        results = memory_service.verified_search_points(candidates)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as error:
        typer.echo(
            json.dumps(
                {
                    "healthy": False,
                    "stage": "embedding_search",
                    "error_type": type(error).__name__,
                },
                ensure_ascii=False,
            )
        )
        raise typer.Exit(1) from None
    if not results:
        typer.echo(
            json.dumps(
                {
                    "healthy": False,
                    "stage": "source_validation",
                    "errors": ["no_current_source_match"],
                    "candidates": len(candidates),
                }
            )
        )
        raise typer.Exit(1)
    from .evidence import EvidenceRepository

    try:
        evidence = EvidenceRepository(store.database).record(
            {
                "kind": "embedding",
                "source_id": f"memory-live-acceptance:{uuid.uuid4().hex}",
                "observed_at": datetime.now(UTC),
                "subject_sha256": source_tree_sha256(ROOT),
                "passed": True,
                "checks": {
                    "embedding_provider_response": True,
                    "embedding_dimension_match": True,
                    "qdrant_collection_healthy": True,
                    "read_only_search": True,
                    "source_hash_validated": True,
                },
            }
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        typer.echo(
            json.dumps(
                {
                    "healthy": False,
                    "stage": "evidence_persist",
                    "error_type": type(error).__name__,
                },
                ensure_ascii=False,
            )
        )
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "healthy": True,
                "read_only": True,
                "stage": "embedding_search",
                "collection": qdrant_config.get("collection", "harness-memory"),
                "model": embedding.get("model"),
                "dimension": embedding.get("dimensions", 1024),
                "matches": len(results),
                "candidates": len(candidates),
                "evidence_id": evidence["id"],
                "evidence_kind": evidence["kind"],
            },
            ensure_ascii=False,
        )
    )


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


@memory.command("qdrant-reconcile-plan")
def qdrant_reconcile_plan():
    """Preview Vault/Qdrant drift without embedding or writing vectors."""
    conf, _store, orchestrator = build()
    if not conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False):
        typer.echo("Qdrant is disabled in configuration")
        raise typer.Exit(1)
    try:
        plan = orchestrator.memory_service.plan_reconciliation()
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Qdrant reconciliation preview failed: {type(error).__name__}")
        raise typer.Exit(1) from None
    # Point IDs are an implementation detail; only actionable counts/sources
    # are exposed. Applying always computes a fresh plan.
    typer.echo(
        json.dumps(
            {
                "notes": plan["notes"],
                "chunks": plan["chunks"],
                "needs_index": plan["needs_index"],
                "orphan_sources": plan["orphan_sources"],
                "orphan_points": sum(map(len, plan["orphan_points"].values())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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


@tasks.command("knowledge-search")
def task_knowledge_search(
    task_id: int,
    query: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=20)] = 5,
):
    """Search Vault knowledge for a task without changing the Vault."""
    _, _, orchestrator = build()
    try:
        result = orchestrator.service.knowledge_search(task_id, query, limit=limit)
    except ValueError as error:
        typer.echo(f"Task knowledge search failed: {error}")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


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
    inventory = orchestrator.model_operations.model_inventory()
    for provider in inventory:
        name = provider["provider"]
        typer.echo(f"{name}\t{provider['status']}")
        if provider["discovery_failed"]:
            typer.echo(f"{name}\t<discovery failed>")
        for model in provider["models"]:
            typer.echo(
                f"{model['id']}\t{name}\t{model['alias']}\t"
                f"{model['tier']}\t{model['status']}"
            )


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
    result = orchestrator.model_operations.test_model(model_name)
    if result["status"] != "successful":
        typer.echo(
            f"Model test failed ({result['error_category']}/{result['error_type']})"
        )
        raise typer.Exit(1)
    typer.echo(f"Model test successful: {model_name}")


def _safe_config():
    from .services import ConfigurationApplicationService

    return ConfigurationApplicationService(Config()).view(resolved=True)


@config.command("show")
def config_show():
    from .services import ConfigurationApplicationService

    typer.echo(
        yaml.safe_dump(
            ConfigurationApplicationService(Config()).view(), sort_keys=False
        ).rstrip()
    )


@config.command("resolved")
def config_resolved():
    from .services import ConfigurationApplicationService

    typer.echo(
        yaml.safe_dump(
            ConfigurationApplicationService(Config()).view(resolved=True),
            sort_keys=False,
        ).rstrip()
    )


@config.command("validate")
def config_validate():
    from .services import ConfigurationApplicationService

    ConfigurationApplicationService(Config()).validate()
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
