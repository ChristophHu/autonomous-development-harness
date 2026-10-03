"""Shared application boundary for API and command-line operations."""

import asyncio
import os
import platform
import shutil
import socket
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import httpx

from .domain import EventKind, Status, Task
from .evidence import EvidenceRepository, audit_evidence
from .process_control import RunControl, TaskCancelled, use_run_control
from .providers import ProviderHealth
from .sqlite_operations import inspect_sqlite


@dataclass(frozen=True)
class OperationalDiagnosticReport:
    """Typed, stable result shared by command and API diagnostics callers."""

    checks: dict[str, bool]
    details: dict

    @property
    def healthy(self):
        return all(self.checks.values())


class OperationalDiagnosticsService:
    """Collect health checks once and return render-neutral typed diagnostics."""

    def __init__(
        self,
        configuration,
        store,
        orchestrator,
        *,
        lifecycle_factory,
        config_health,
        sqlite_health,
        qdrant_reporter,
        docker_broker_factory,
        sqlite_inspector=inspect_sqlite,
        which=shutil.which,
        run=subprocess.run,
        system=platform.system,
        machine=platform.machine,
        access=os.access,
        connect=socket.create_connection,
        provider_roles=None,
        provider_probe=None,
        provider_status=None,
        component_health=None,
        executor_factory=ThreadPoolExecutor,
        wait_for=wait,
    ):
        self.configuration = configuration
        self.store = store
        self.orchestrator = orchestrator
        self.lifecycle_factory = lifecycle_factory
        self.config_health = config_health
        self.sqlite_health = sqlite_health
        self.qdrant_reporter = qdrant_reporter
        self.docker_broker_factory = docker_broker_factory
        self.sqlite_inspector = sqlite_inspector
        self.which = which
        self.run = run
        self.system = system
        self.machine = machine
        self.access = access
        self.connect = connect
        self.provider_roles = provider_roles or (lambda _conf: {})
        self.provider_probe = provider_probe
        self.provider_status = provider_status
        self.component_health = component_health or (
            lambda call, *, enabled=True: (
                "disabled" if not enabled else "available" if call() else "unavailable"
            )
        )
        self.executor_factory = executor_factory
        self.wait_for = wait_for

    def start_preflight(self, *, provider_timeout=5.0):
        """Collect start gates in the same service boundary as status and doctor."""
        provider_probe = self.provider_probe or ModelOperationsService.probe_provider
        provider_status = self.provider_status or ModelOperationsService.provider_status
        conf, store, orchestrator = self.configuration, self.store, self.orchestrator
        checks = [
            {
                "name": "Configuration",
                "status": "available" if self.config_health(conf) else "unavailable",
                "critical": True,
            },
            {
                "name": "SQLite",
                "status": "available"
                if self.sqlite_health(store) == "available"
                else "unavailable",
                "critical": True,
            },
        ]
        provider_config = conf.data.get("models", {}).get("providers", {})
        providers = orchestrator.models.providers
        roles = self.provider_roles(conf)
        enabled = {
            name for name, value in provider_config.items() if value.get("enabled")
        }
        probes = {name: providers[name] for name in roles.keys() & providers.keys()}
        results = {}
        executor = self.executor_factory(max_workers=max(1, min(len(probes), 8)))
        futures = {
            executor.submit(provider_probe, item): name for name, item in probes.items()
        }
        try:
            completed, pending = self.wait_for(futures, timeout=provider_timeout)
            for future in completed:
                results[futures[future]] = provider_status(future.result())
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
        qdrant_enabled = (
            conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
        )
        qdrant_state = self.component_health(
            orchestrator.qdrant.health, enabled=qdrant_enabled
        )
        checks.append({"name": "Qdrant", "status": qdrant_state, "critical": False})
        return checks

    def status_report(self):
        try:
            service = self.lifecycle_factory().inspect()
        except (OSError, RuntimeError, ValueError):
            service = {"state": "unknown", "record": None, "ready": False}
        state = service["state"]
        description = state
        if state == "running":
            record = service.get("record") or {}
            readiness = "ready" if service.get("ready") else "not ready"
            description = f"running ({readiness}) pid={record.get('pid', 'unknown')}"
        vault = self.configuration.path("obsidian_vault")
        vault_available = vault.is_dir() and self.access(vault, os.R_OK | os.W_OK)
        qdrant_enabled = (
            self.configuration.data.get("memory", {})
            .get("qdrant", {})
            .get("enabled", False)
        )
        qdrant = self.qdrant_reporter(self.orchestrator, enabled=qdrant_enabled)
        provider_states = {
            name: ModelOperationsService.provider_status(
                ModelOperationsService.probe_provider(provider)
            )
            for name, provider in self.orchestrator.models.providers.items()
        }
        checks = {
            "API Service": state == "running" and service.get("ready", False),
            "SQLite": self.sqlite_health(self.store) == "available",
            "Obsidian Vault": vault_available,
            "Qdrant": qdrant["healthy"],
            **{
                f"Provider {name}": value == "available"
                for name, value in provider_states.items()
            },
        }
        return self.status(
            checks,
            details={
                "service": description,
                "sqlite": self.sqlite_health(self.store),
                "sqlite_schema": self.sqlite_inspector(self.store.db),
                "vault": vault_available,
                "qdrant": qdrant,
                "counts": self.orchestrator.service.status_counts(),
                "providers": provider_states,
            },
        )

    def doctor_report(self):
        conf, store, orchestrator = self.configuration, self.store, self.orchestrator
        qdrant_enabled = (
            conf.data.get("memory", {}).get("qdrant", {}).get("enabled", False)
        )
        docker_available = self.which("docker") is not None
        compose_available = False
        if docker_available and qdrant_enabled:
            try:
                compose_available = (
                    self.run(
                        ["docker", "compose", "version"],
                        capture_output=True,
                        timeout=3,
                        check=False,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.TimeoutExpired):
                compose_available = False
        daemon_available = True
        if qdrant_enabled and docker_available and compose_available:
            try:
                daemon_available = (
                    self.docker_broker_factory().run("status").returncode == 0
                )
            except (OSError, RuntimeError, ValueError):
                daemon_available = False
        elif qdrant_enabled:
            daemon_available = False
        try:
            service = self.lifecycle_factory().inspect()
        except (OSError, RuntimeError, ValueError):
            service = {"state": "unknown", "record": None, "ready": False}
        providers = ModelOperationsService.providers_health(
            orchestrator.models.providers
        )
        qdrant = self.qdrant_reporter(orchestrator, enabled=qdrant_enabled)
        vault, workspace, logs = (
            conf.path(key) for key in ("obsidian_vault", "workspace", "logs")
        )
        checks = {
            "macOS": self.system() == "Darwin",
            "Apple Silicon": self.machine() == "arm64",
            "Configuration": self.config_health(conf),
            "SQLite": self.sqlite_health(store) == "available",
            "SQLite Schema": self.sqlite_inspector(store.db)[
                "version_history_complete"
            ],
            "Git": self.which("git") is not None,
            "Docker": docker_available or not qdrant_enabled,
            "Docker Compose": compose_available or not qdrant_enabled,
            "Docker Daemon / Qdrant Compose": daemon_available,
            "Obsidian Vault": vault.is_dir() and self.access(vault, os.R_OK | os.W_OK),
            "Workspace": workspace.is_dir()
            and self.access(workspace, os.R_OK | os.W_OK),
            "Logs Directory": logs.is_dir() and self.access(logs, os.W_OK),
            "Qdrant": qdrant["healthy"],
            "Qdrant Collection": not qdrant_enabled
            or qdrant.get("collection_exists", False),
            "API Service": service["state"] == "running" and service["ready"],
        }
        checks.update(
            {
                f"Provider {name}": state == "available"
                for name, state in providers.items()
            }
        )
        mcp = orchestrator.tools.mcp_status()
        checks.update(
            {
                f"MCP {item['name']}": item["state"]
                not in {"unavailable", "invalid_config"}
                for item in mcp
                if item["enabled"]
            }
        )
        if service["state"] == "running" and service["ready"]:
            port_available = True
        else:
            try:
                with self.connect(
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
        return self.doctor(
            checks, details={"providers": providers, "mcp": mcp, "qdrant": qdrant}
        )

    @staticmethod
    def status(checks, *, details=None):
        if not isinstance(checks, dict) or any(
            not isinstance(name, str) or type(ok) is not bool
            for name, ok in checks.items()
        ):
            raise ValueError("diagnostic checks must map names to booleans")
        if details is not None and not isinstance(details, dict):
            raise ValueError("diagnostic details must be a mapping")
        return OperationalDiagnosticReport(dict(checks), dict(details or {}))

    @classmethod
    def doctor(cls, checks, *, details=None):
        return cls.status(checks, details=details)


class VerificationEvidenceService:
    """Application boundary shared by HTTP and command-line evidence workflows."""

    def __init__(self, database):
        self.repository = EvidenceRepository(database)

    def record(self, payload):
        return self.repository.record(payload)

    def list(self, *, kind=None, limit=100):
        return self.repository.list(kind=kind, limit=limit)

    def audit(self, *, subject_sha256, max_age_hours=168):
        return audit_evidence(
            self.list(limit=500),
            expected_subject_sha256=subject_sha256,
            max_age_hours=max_age_hours,
        )

    def audit_kind(self, kind, *, subject_sha256, max_age_hours=168):
        return audit_evidence(
            self.list(kind=kind, limit=500),
            expected_subject_sha256=subject_sha256,
            max_age_hours=max_age_hours,
        )


class ConfigurationApplicationService:
    """Safe configuration operations shared by the CLI and HTTP API."""

    def __init__(self, configuration):
        self.configuration = configuration

    def view(self, *, resolved=False):
        return self.configuration.redacted(resolved=resolved)

    def validate(self):
        self.configuration.validate()
        settings = self.configuration.settings
        return {
            "valid": True,
            "environment": settings.harness.environment,
            "max_parallel_steps": settings.harness.max_parallel_steps,
        }


class ModelOperationsService:
    """Shared provider-health and model-inventory application operations."""

    def __init__(self, registry, snapshots=None):
        self.registry = registry
        self.snapshots = snapshots

    @staticmethod
    def probe_provider(provider):
        if hasattr(provider, "health_report"):
            try:
                return provider.health_report()
            except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
                return "unavailable"
        try:
            health = getattr(provider, "health", None)
            return (
                "available" if health is not None and bool(health()) else "unavailable"
            )
        except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
            return "unavailable"

    @staticmethod
    def provider_status(report):
        return report.status if isinstance(report, ProviderHealth) else report

    @staticmethod
    def model_available(report, model_id, discovered):
        if isinstance(report, ProviderHealth):
            return report.model_available(model_id)
        return report == "available" and model_id in discovered

    def provider_health(self):
        return self.providers_health(self.registry.providers)

    @classmethod
    def providers_health(cls, providers):
        return {
            name: cls.provider_status(cls.probe_provider(provider))
            for name, provider in sorted(providers.items())
        }

    def model_inventory(self):
        definitions = self.registry.models
        names = set(self.registry.providers) | {
            item.get("provider") for item in definitions.values()
        }
        inventory = []
        for name in sorted(item for item in names if isinstance(item, str)):
            provider = self.registry.providers.get(name)
            report = (
                self.probe_provider(provider) if provider is not None else "unavailable"
            )
            discovered = []
            discovery_failed = False
            if isinstance(report, ProviderHealth):
                discovered = list(report.models)
                self.registry.discovered[name] = tuple(sorted(set(discovered)))
                discovery_failed = not report.api_available
            elif provider is not None:
                try:
                    discovered = list(self.registry.discover(name))
                except (OSError, RuntimeError, ValueError, httpx.HTTPError):
                    discovery_failed = True
            if self.snapshots is not None:
                if discovery_failed:
                    previous = self.snapshots.latest_model_discovery(name)
                    if previous is not None:
                        discovered = previous["models"]
                self.snapshots.record_model_discovery(
                    name, self.provider_status(report), discovery_failed, discovered
                )
            configured = {
                alias: definition
                for alias, definition in definitions.items()
                if definition.get("provider") == name
            }
            models = []
            for alias, definition in sorted(configured.items()):
                model_id = definition.get("model") or "-"
                models.append(
                    {
                        "id": model_id,
                        "alias": alias,
                        "tier": definition.get("tier") or "unknown",
                        "status": (
                            "stale"
                            if discovery_failed and model_id in discovered
                            else "available"
                            if self.model_available(report, model_id, discovered)
                            else "unavailable"
                        ),
                    }
                )
            configured_ids = {item.get("model") for item in configured.values()}
            for model_id in sorted(set(discovered) - configured_ids):
                models.append(
                    {
                        "id": model_id,
                        "alias": "-",
                        "tier": "unknown",
                        "status": (
                            "stale"
                            if discovery_failed
                            else "available"
                            if self.model_available(report, model_id, discovered)
                            else "unavailable"
                        ),
                    }
                )
            inventory.append(
                {
                    "provider": name,
                    "status": self.provider_status(report),
                    "discovery_failed": discovery_failed,
                    "models": models,
                }
            )
        return inventory

    def test_model(self, model_name):
        """Run a minimal model check and return a secret-safe outcome."""
        try:
            provider, model_id = self.registry.resolve(model_name)
            response = provider.complete(
                "Reply with exactly: OK", **({"model": model_id} if model_id else {})
            )
            text = response[0] if isinstance(response, tuple) else response
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty model response")
        except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as error:
            return {
                "model": model_name,
                "status": "failed",
                "error_type": type(error).__name__,
            }
        return {"model": model_name, "status": "successful"}


class VaultKnowledgeApplicationService:
    """Shared, read-only and redacted Vault retrieval for task/API/CLI callers."""

    def __init__(self, vault, source_root, sanitizer=None):
        self.vault = vault
        self.source_root = source_root
        self._knowledge = None
        self.sanitizer = sanitizer or (lambda value: value)

    @property
    def knowledge(self):
        if self._knowledge is None:
            from .vault_knowledge import VaultKnowledgeService

            self._knowledge = VaultKnowledgeService(
                self.vault, source_root=self.source_root
            )
        return self._knowledge

    def search(self, query, *, limit=5):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            raise ValueError("limit must be between 1 and 20")
        return self.sanitizer(
            {
                "query": query,
                "hits": self.knowledge.search(query, limit=limit),
            }
        )


class TaskService:
    def __init__(self, store, orchestrator):
        self.store = store
        self.orchestrator = orchestrator
        self.active_controls = {}
        self._controls_lock = threading.Lock()

    def get(self, task_id):
        task = self.store.get(task_id)
        if task is None:
            raise ValueError("task not found")
        return task

    def create(self, task):
        if task.id is not None or task.status != Status.PENDING or task.git_state:
            raise ValueError(
                "new tasks must be pending without an ID or internal Git state"
            )
        for dependency in task.dependencies + (
            [task.parent_task_id] if task.parent_task_id else []
        ):
            self.get(dependency)
        return self.store.create(task)

    def patch(self, task_id, payload):
        self.get(task_id)
        protected = {
            "id",
            "status",
            "result",
            "created_at",
            "updated_at",
            "plan",
            "validation_result",
            "test_result",
            "git_state",
        }
        if not payload or protected.intersection(payload):
            raise ValueError("invalid editable task fields")
        updated = Task.model_validate(self.get(task_id).model_dump() | payload)
        pending = updated.dependencies + (
            [updated.parent_task_id] if updated.parent_task_id else []
        )
        visited = set()
        while pending:
            dependency = pending.pop()
            if dependency == task_id:
                raise ValueError("task dependency cycle")
            if dependency not in visited:
                visited.add(dependency)
                linked = self.get(dependency)
                pending.extend(linked.dependencies)
                if linked.parent_task_id:
                    pending.append(linked.parent_task_id)
        if not self.store.update_task_if_idle(task_id, updated):
            raise ValueError("task is currently running")
        return self.get(task_id)

    def list(self, status=None):
        return [self.get(row["id"]) for row in self.store.tasks.list(status)]

    def status_counts(self):
        return {
            status.value: len(self.store.tasks.list(status.value)) for status in Status
        }

    def knowledge_search(self, task_id, query, *, limit=5):
        self.get(task_id)
        result = self.orchestrator.knowledge_operations.search(query, limit=limit)
        return {"task_id": task_id, **result}

    def events(self, task_id, *filters):
        self.get(task_id)
        return self.store.list_events(task_id, *filters)

    def event_feed(self, *filters):
        return self.store.list_events(*filters)

    def events_after(self, cursor, task_id=None):
        if task_id is not None:
            self.get(task_id)
        return self.store.events.after(cursor, task_id)

    def questions(self, task_id):
        self.get(task_id)
        return self.store.list_questions(task_id)

    def plan(self, task_id):
        self.get(task_id)
        return self.store.latest_plan(task_id)

    def validation(self, task_id):
        self.get(task_id)
        return self.store.latest_validation(task_id)

    def save_artifact(self, task_id, key, content, expected_version=None):
        self.get(task_id)
        return self.store.save_artifact(task_id, key, content, expected_version)

    def artifacts(self, task_id):
        self.get(task_id)
        return self.store.artifacts_for_task(task_id)

    def artifact(self, task_id, key):
        self.get(task_id)
        return self.store.artifact(task_id, key)

    def artifact_history(self, task_id, key):
        self.get(task_id)
        return self.store.artifact_history(task_id, key)

    def corrections(self, task_id, status=None):
        self.get(task_id)
        return self.store.corrections.list_for_task(task_id, status)

    def update_correction(self, task_id, item_id, status):
        self.get(task_id)
        item = self.store.corrections.get(item_id)
        if item is None or item["task_id"] != task_id:
            raise ValueError("correction not found")
        return self.store.corrections.set_status(item_id, status)

    def ask_question(
        self, task_id, question, reason, options=None, required=True, purpose="input"
    ):
        self.get(task_id)
        return self.store.ask(task_id, question, reason, options, required, purpose)

    def model_usage(self, **filters):
        return self.store.model_usage.report(**filters)

    def abort(self, task_id):
        self.get(task_id)
        self.store.tasks.transition(task_id, Status.CANCELLED)
        with self._controls_lock:
            control = self.active_controls.get(task_id)
        if control is not None:
            control.request_stop("task_aborted")
        return self.get(task_id)

    def delete(self, task_id):
        self.get(task_id)
        if not self.store.tasks.delete_if_idle(task_id):
            raise ValueError("task is currently running")

    async def start(self, task_id):
        task = self.get(task_id)
        if any(
            self.get(dependency).status != Status.COMPLETED
            for dependency in task.dependencies
        ):
            raise ValueError("task dependencies are incomplete")
        owner = uuid.uuid4().hex
        if not self.store.tasks.claim(
            task_id, owner, exclusive=self.orchestrator.git_enabled
        ):
            raise ValueError("task is terminal or currently running")
        control = RunControl()
        with self._controls_lock:
            self.active_controls[task_id] = control

        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                if not self.store.tasks.renew(task_id, owner):
                    control.request_stop("lease_lost")
                    return

        renewal = asyncio.create_task(heartbeat())
        try:
            with use_run_control(control):
                try:
                    return await self.orchestrator._run(task_id, owner)
                except TaskCancelled:
                    return self.get(task_id)
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            with self._controls_lock:
                if self.active_controls.get(task_id) is control:
                    del self.active_controls[task_id]
            self.store.tasks.release(task_id, owner)

    async def answer(self, task_id, question_id, answer):
        task = self.get(task_id)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must not be blank")
        if not self.store.answer(question_id, answer, task_id):
            raise ValueError("open question not found or invalid answer")
        self.store.event(
            task_id, EventKind.QUESTION_ANSWERED, {"question_id": question_id}
        )
        if task.git_state.get("cleanup_question_id") == question_id:
            return await asyncio.to_thread(
                self.orchestrator._invoke,
                task,
                "git-cleanup",
                "coding",
                self.orchestrator.git_service.cleanup,
                task_id,
                question_id,
            )
        if task.status not in {
            Status.WAITING_HUMAN,
            Status.WAITING_DECISION,
            Status.WAITING_APPROVAL,
        } or self.store.questions.has_open_required(task_id):
            return self.get(task_id)
        return await self.start(task_id)
