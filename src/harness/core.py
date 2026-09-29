"""Core domain, persistence, configuration, orchestration and API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel

from .agents import (
    Executor,
    ModelRegistry,
    ModelRouter,
    Planner,
)
from .approvals import ApprovalService
from .audit import AuditRecorder
from .database import (
    AgentRunRepository,
    CorrectionRepository,
    Database,
    DecisionRepository,
    EventRepository,
    ModelRunRepository,
    PlanRepository,
    QuestionRepository,
    SubtaskRepository,
    TaskRepository,
    ValidationRepository,
)
from .domain import EventKind, Status, Task
from .memory import (
    EMBEDDING_BATCH_SIZE_DEFAULT,
    EMBEDDING_DIMENSION_DEFAULT,
    ContextBuilder,
    EmbeddingProvider,
    ObsidianMemory,
    QdrantMemory,
)
from .memory_service import MemoryService
from .security import SecretResolver
from .tools import ToolRegistry
from .usage import ModelUsageReportService
from .workflows import GitWorkflow

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("harness")

CONFIG_DEFAULTS = {
    "harness": {"name": "autonomous-development-harness", "environment": "development"},
    "paths": {
        "workspace": "./workspace",
        "obsidian_vault": "./vault",
        "logs": "./logs",
        "database": "./data/harness.db",
    },
    "database": {"type": "sqlite"},
    "git": {"enabled": False, "main": "main", "dev": "dev", "remote": None},
    "memory": {
        "obsidian": {"enabled": True},
        "qdrant": {
            "enabled": False,
            "url": "http://127.0.0.1:6333",
            "collection": "harness-memory",
        },
        "embeddings": {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "text-embedding-model",
            "dimensions": EMBEDDING_DIMENSION_DEFAULT,
            "batch_size": EMBEDDING_BATCH_SIZE_DEFAULT,
        },
    },
    "models": {
        "defaults": {"provider": "lmstudio"},
        "providers": {},
        "registry": {},
        "rates": {},
    },
    "profiles": {},
    "api": {"host": "127.0.0.1", "port": 8080, "swagger": True},
    "logging": {"level": "INFO", "file": "./logs/harness.log"},
    "tools": {"permissions": {}},
    "secrets": {},
}


_SECRET_FIELDS = ("secret", "password", "token", "api_key", "private_key", "credential")


class Event(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "task_id": 7,
                    "kind": "TASK_COMPLETED",
                    "payload": {"status": "completed"},
                    "created_at": "2026-09-28T12:00:00+00:00",
                }
            ]
        }
    }

    id: int | None = None
    task_id: int | None = None
    kind: str
    payload: dict[str, Any] = {}
    created_at: str = ""


class TaskLifecycle:
    STARTABLE: ClassVar[set[Status]] = {Status.PENDING, Status.FAILED, Status.BLOCKED}

    @classmethod
    def can_start(cls, status, has_open_question=False):
        state = Status(status)
        return not has_open_question and state in cls.STARTABLE | {Status.WAITING_HUMAN}


class Config:
    def __init__(self, path: Path = ROOT / "config.yaml"):
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a mapping")  # noqa: TRY004
        env = self._env()
        raw = deepcopy(data)
        secrets = raw.setdefault("secrets", {})
        if not isinstance(secrets, dict):
            raise ValueError("secrets must be a mapping")  # noqa: TRY004
        secrets.update(env)
        resolver = SecretResolver()
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LMSTUDIO_API_KEY"):
            keychain = resolver.get(name)
            if keychain:
                secrets[name] = keychain
        self.configured = raw
        self.data = self._merge(CONFIG_DEFAULTS, raw)
        self.validate()

    @classmethod
    def _merge(cls, defaults, overrides):
        result = deepcopy(defaults)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = cls._merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    @classmethod
    def _redact(cls, value, key=""):
        if isinstance(value, dict):
            return {name: cls._redact(item, name) for name, item in value.items()}
        if any(secret_field in key.lower() for secret_field in _SECRET_FIELDS):
            return "********"
        if isinstance(value, list):
            return [cls._redact(item, key) for item in value]
        return deepcopy(value)

    def resolved(self):
        """Return the effective configuration after defaults and source overrides."""
        return deepcopy(self.data)

    def redacted(self, resolved=False):
        """Return a safe display copy; never mutate effective configuration."""
        source = self.data if resolved else self.configured
        return self._redact(source)

    def _env(self):
        result = {}
        p = ROOT / ".env"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    result[k.strip()] = v.strip().strip("'\"")
        result.update(
            {k: v for k, v in os.environ.items() if k.endswith(("_API_KEY", "_MODEL"))}
        )
        return result

    def path(self, key: str) -> Path:
        value = self.data.get("paths", {}).get(key, f"./{key}")
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    def validate(self):
        def mapping(value, name):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")  # noqa: TRY004
            return value

        def text(value, name, *, optional=False):
            if optional and value is None:
                return
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        def boolean(value, name):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")  # noqa: TRY004

        def valid_url(value, name):
            text(value, name)
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"{name} must use http or https")

        def timeout(value, name):
            if isinstance(value, dict):
                if "total" not in value or set(value) - {"total", "connect", "read"}:
                    raise ValueError(f"{name} requires total and permits connect/read")
                phases = value.items()
            else:
                phases = (("total", value),)
            for phase, seconds in phases:
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not 0 < seconds <= 300
                ):
                    raise ValueError(
                        f"{name}.{phase} must be between 0 and 300 seconds"
                    )
                if phase != "total" and seconds > value["total"]:
                    raise ValueError(f"{name}.{phase} cannot exceed total")

        def provider_retry(value, name):
            retry = mapping(value, name)
            if set(retry) - {"max_attempts", "base_delay", "max_delay"}:
                raise ValueError(f"{name} has unknown retry settings")
            attempts = retry.get("max_attempts", 3)
            if isinstance(attempts, bool) or not isinstance(attempts, int):
                raise ValueError(  # noqa: TRY004
                    f"{name}.max_attempts must be an integer"
                )
            if not 1 <= attempts <= 5:
                raise ValueError(f"{name}.max_attempts must be between 1 and 5")
            base = retry.get("base_delay", 0.25)
            maximum = retry.get("max_delay", 4.0)
            for field, seconds in (("base_delay", base), ("max_delay", maximum)):
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not 0 <= seconds <= 60
                ):
                    raise ValueError(f"{name}.{field} must be between 0 and 60 seconds")
            if base > maximum:
                raise ValueError(f"{name}.base_delay cannot exceed max_delay")

        root = mapping(self.data, "configuration root")
        api = mapping(root.get("api", {}), "api")
        port = api.get("port", 8080)
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError("api.port must be an integer")  # noqa: TRY004
        if not 1 <= port <= 65535:
            raise ValueError("api.port must be between 1 and 65535")
        text(api.get("host", "127.0.0.1"), "api.host")
        if api.get("host", "127.0.0.1") not in {"127.0.0.1", "localhost"}:
            raise ValueError("api.host must remain local")
        boolean(api.get("swagger", True), "api.swagger")

        paths = mapping(root.get("paths", {}), "paths")
        for name, value in paths.items():
            text(value, f"paths.{name}")
        database = mapping(root.get("database", {}), "database")
        if database.get("type", "sqlite") != "sqlite":
            raise ValueError("database.type must be sqlite")

        git = mapping(root.get("git", {}), "git")
        boolean(git.get("enabled", False), "git.enabled")
        for name in ("main", "dev"):
            text(git.get(name, name), f"git.{name}")
        text(git.get("remote"), "git.remote", optional=True)

        memory = mapping(root.get("memory", {}), "memory")
        for section in ("obsidian", "qdrant"):
            section_data = mapping(memory.get(section, {}), f"memory.{section}")
            boolean(
                section_data.get("enabled", section == "obsidian"),
                f"memory.{section}.enabled",
            )
        qdrant = memory.get("qdrant", {})
        valid_url(qdrant.get("url", "http://127.0.0.1:6333"), "memory.qdrant.url")
        text(qdrant.get("collection", "harness-memory"), "memory.qdrant.collection")
        timeout(qdrant.get("timeout", 5), "memory.qdrant.timeout")
        embeddings = mapping(memory.get("embeddings", {}), "memory.embeddings")
        text(embeddings.get("provider", "lmstudio"), "memory.embeddings.provider")
        text(embeddings.get("model", "text-embedding-model"), "memory.embeddings.model")
        valid_url(
            embeddings.get("base_url", "http://127.0.0.1:1234/v1"),
            "memory.embeddings.base_url",
        )
        dimensions = embeddings.get("dimensions", EMBEDDING_DIMENSION_DEFAULT)
        batch_size = embeddings.get("batch_size", EMBEDDING_BATCH_SIZE_DEFAULT)
        timeout(embeddings.get("timeout", 30), "memory.embeddings.timeout")
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions < 1
        ):
            raise ValueError("memory.embeddings.dimensions must be a positive integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 256
        ):
            raise ValueError("memory.embeddings.batch_size must be between 1 and 256")

        models = mapping(root.get("models", {}), "models")
        providers = mapping(models.get("providers", {}), "models.providers")
        for name, provider in providers.items():
            provider = mapping(provider, f"models.providers.{name}")
            enabled = provider.get("enabled", False)
            boolean(enabled, f"models.providers.{name}.enabled")
            if enabled and not provider.get("base_url"):
                raise ValueError(f"models.providers.{name}.base_url is required")
            if provider.get("base_url"):
                valid_url(provider["base_url"], f"models.providers.{name}.base_url")
            if provider.get("model") is not None:
                text(provider["model"], f"models.providers.{name}.model")
            timeout(provider.get("timeout", 120), f"models.providers.{name}.timeout")
            provider_retry(provider.get("retry", {}), f"models.providers.{name}.retry")
        registry = mapping(models.get("registry", {}), "models.registry")
        for name, definition in registry.items():
            definition = mapping(definition, f"models.registry.{name}")
            for required in ("provider", "model"):
                text(definition.get(required), f"models.registry.{name}.{required}")
            capabilities = definition.get("capabilities", [])
            if not isinstance(capabilities, list) or any(
                not isinstance(item, str) for item in capabilities
            ):
                raise ValueError(
                    f"models.registry.{name}.capabilities must be a list of strings"
                )

        profiles = mapping(root.get("profiles", {}), "profiles")
        for name, profile in profiles.items():
            profile = mapping(profile, f"profiles.{name}")
            model = mapping(profile.get("model", {}), f"profiles.{name}.model")
            if "primary" in model:
                text(model["primary"], f"profiles.{name}.model.primary")
            fallback = model.get("fallback", [])
            if not isinstance(fallback, list) or any(
                not isinstance(item, str) for item in fallback
            ):
                raise ValueError(
                    f"profiles.{name}.model.fallback must be a list of strings"
                )
            for field in ("tools", "permissions"):
                values = profile.get(field, [])
                if not isinstance(values, list) or any(
                    not isinstance(item, str) for item in values
                ):
                    raise ValueError(
                        f"profiles.{name}.{field} must be a list of strings"
                    )

        logging_config = mapping(root.get("logging", {}), "logging")
        level = logging_config.get("level", "INFO")
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "logging.level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        text(logging_config.get("file", "./logs/harness.log"), "logging.file")
        tools = mapping(root.get("tools", {}), "tools")
        http = mapping(tools.get("http", {}), "tools.http")
        timeout(http.get("timeout", 30), "tools.http.timeout")
        mapping(root.get("secrets", {}), "secrets")
        return True


class Store:
    def __init__(self, config: Config):
        self.config = config
        self.db = config.path("database")
        self.database = Database(self.db)
        self.tasks = TaskRepository(self.database)
        self.events = EventRepository(self.database)
        self.plans = PlanRepository(self.database)
        self.validations = ValidationRepository(self.database)
        self.corrections = CorrectionRepository(self.database)
        self.subtasks = SubtaskRepository(self.database)
        self.questions = QuestionRepository(self.database)
        self.decisions = DecisionRepository(self.database)
        self.agent_runs = AgentRunRepository(self.database)
        self.model_runs = ModelRunRepository(self.database)
        self.model_usage = ModelUsageReportService(self.model_runs)
        self.audit = AuditRecorder(self.database, config.data.get("secrets", {}))
        self.db.parent.mkdir(parents=True, exist_ok=True)

    def create(self, task: Task) -> Task:
        task.id = self.tasks.create(
            task.title,
            task.description,
            task.status.value,
            task.model_dump(mode="json"),
        )
        return self.get(task.id)

    def get(self, task_id: int) -> Task | None:
        row = self.tasks.get(task_id)
        return (
            Task(
                **{
                    k: v
                    for k, v in json.loads(row["metadata"]).items()
                    if k
                    not in {
                        "id",
                        "title",
                        "description",
                        "status",
                        "result",
                        "created_at",
                        "updated_at",
                    }
                },
                id=row["id"],
                title=row["title"],
                description=row["description"],
                status=row["status"],
                result=row["result"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            if row
            else None
        )

    def update(self, task: Task):
        self.tasks.update(
            task.id,
            title=task.title,
            description=task.description,
            status=task.status.value,
            result=task.result,
            metadata=task.model_dump(mode="json"),
        )

    def event(self, task_id: int | None, kind: EventKind | str, payload: dict):
        event_kind = EventKind(kind)
        now = datetime.now(UTC).isoformat()
        payload = self.audit.sanitize(payload)
        self.events.append(task_id, event_kind, payload)
        return Event(
            task_id=task_id, kind=event_kind.value, payload=payload, created_at=now
        )

    def ask(self, task_id, question, reason, options=None, required=True):
        if not question.strip():
            raise ValueError("question must not be blank")
        payload = self.audit.sanitize({"question": question, "required": required})
        question_id = self.questions.ask(
            task_id, question, reason, options, required, payload
        )
        return question_id

    def answer(self, question_id, answer, task_id=None):
        return self.questions.answer(question_id, answer, task_id)


class Permissions:
    def __init__(self, config: Config):
        self.config = config
        self.rules = config.data.get("tools", {}).get("permissions", {})

    def require(self, tool: str, write=False):
        rule = self.rules.get(tool, "denied")
        if rule == "denied" or (write and rule != "write"):
            raise PermissionError(f"tool '{tool}' is not permitted")


class Orchestrator:
    def __init__(self, store: Store, config: Config | None = None):
        self.store = store
        self.config = config or store.config
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        registry = ModelRegistry(self.config)
        self.audit = store.audit
        router = ModelRouter(registry, self.config, audit=self.audit)
        self.models = registry
        self.router = router
        self.planner = Planner(router)
        workspace = self.config.path("workspace")
        self.tools = ToolRegistry(Permissions(self.config), self._tool_event, workspace)
        self.approvals = ApprovalService(self.store.questions, self.store)
        self.git_workflow = GitWorkflow(self.tools)
        from .git_service import GitWorkflowService

        self.git_service = GitWorkflowService(store, self.git_workflow, self.tools)
        self.git_enabled = self.config.data.get("git", {}).get("enabled", False)
        self.executor = Executor(router, self.tools)
        from .services import TaskService
        from .validation import EvidenceValidator

        self.service = TaskService(store, self)
        self.validator = EvidenceValidator(self.tools, router)
        qdrant = self.config.data.get("memory", {}).get("qdrant", {})
        self.qdrant_enabled = qdrant.get("enabled", False)
        embeddings = self.config.data.get("memory", {}).get("embeddings", {})
        embedder = EmbeddingProvider(
            embeddings.get("base_url", "http://127.0.0.1:1234/v1"),
            embeddings.get("model", "text-embedding-model"),
            timeout=embeddings.get("timeout", 30),
            dimension=embeddings.get("dimensions", EMBEDDING_DIMENSION_DEFAULT),
            batch_size=embeddings.get("batch_size", EMBEDDING_BATCH_SIZE_DEFAULT),
        )
        self.qdrant = QdrantMemory(
            qdrant.get("url", "http://127.0.0.1:6333"),
            qdrant.get("collection", "harness-memory"),
            embeddings.get("dimensions", EMBEDDING_DIMENSION_DEFAULT),
            embedder,
            timeout=qdrant.get("timeout", 5),
        )
        notes = ObsidianMemory(self.config.path("obsidian_vault"))
        self.memory_service = MemoryService(
            notes,
            self.qdrant,
            batch_size=embeddings.get("batch_size", EMBEDDING_BATCH_SIZE_DEFAULT),
        )
        self.context = ContextBuilder(
            notes,
            self.tools,
            self.memory_service if qdrant.get("enabled", False) else None,
        )

    def _tool_event(self, kind, payload):
        self.audit.tool_event(kind, payload)

    def _invoke(self, task, agent, profile, operation, *args):
        with self.audit.agent(task, agent, profile) as span:
            result = operation(*args)
            span["output"] = (
                result.model_dump(mode="json")
                if isinstance(result, BaseModel)
                else {"result_type": type(result).__name__}
            )
            if (
                getattr(result, "success", True) is False
                or getattr(result, "valid", True) is False
            ):
                span["status"] = "failed"
            return result

    def _record_model_usage(self, usage):
        self.store.model_runs.record(usage)

    def _execute_step(self, task, step, context, recovery_scope):
        repair = task.git_state.get("repair", {})
        repair_paths = (
            set(repair.get("paths", [])) if repair.get("authorized") else None
        )
        if repair_paths is not None and not set(step.write_paths) <= repair_paths:
            raise ValueError("repair plan writes outside the approved artifact paths")
        guard = (
            self.tools.recovery_writes(step.write_paths)
            if recovery_scope or repair_paths is not None
            else nullcontext()
        )
        with guard:
            return self._invoke(
                task,
                step.assigned_agent,
                step.profile,
                self.executor.execute,
                step,
                context,
            )

    async def emit(self, e: Event):
        self.store.event(e.task_id, e.kind, e.payload)
        await self.queue.put(e)

    async def run(self, task_id: int):
        return await self.service.start(task_id)

    async def _run(self, task_id, owner):
        from .domain import TERMINAL
        from .process_control import TaskCancelled, current_run_control

        run_control = current_run_control()

        def transition(status):
            if run_control is not None:
                run_control.check()
            self.store.tasks.transition(task_id, status, owner)

        try:
            task = self.store.get(task_id)
            # A previous process may have stopped after claiming a correction.
            # Re-open it so the next leased run can safely retry it.
            for item in self.store.corrections.list_for_task(
                task_id, status="in_progress"
            ):
                self.store.corrections.set_status(item["id"], "open")
            reconciliation = None
            recovery_scope = None
            interrupted = task.status not in {
                Status.PENDING,
                Status.FAILED,
                Status.BLOCKED,
                Status.WAITING_HUMAN,
            }
            if interrupted or (
                task.status in {Status.FAILED, Status.BLOCKED}
                and self.store.plans.latest(task_id) is not None
            ):
                from .reconciliation import ReconciliationService

                reconciliation = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "recovery",
                    "validator",
                    ReconciliationService(
                        self.store, self.tools, self.validator
                    ).inspect,
                    task,
                )
                self.store.event(
                    task_id,
                    EventKind.RECOVERY_RECONCILED,
                    reconciliation.model_dump(mode="json"),
                )
                self.store.tasks.transition(task_id, Status.BLOCKED, owner)
                self.store.event(
                    task_id,
                    EventKind.RECOVERY_INSPECTED,
                    {
                        "previous_state": str(task.status),
                        "action": "replan",
                        "previous_steps": [
                            dict(row) for row in self.store.subtasks.list(task_id)
                        ],
                    },
                )
            transition(Status.ANALYZING)
            task = self.store.get(task_id)
            from .requirements import RequirementCompleter

            task, context = await asyncio.to_thread(
                self._invoke,
                task,
                "requirements",
                "planner",
                RequirementCompleter(self.store, self.router).complete,
                task,
                lambda: self.context.build(task, str(self.config.path("workspace"))),
            )
            missing = [
                name
                for name in (
                    "goal",
                    "requirements",
                    "acceptance_criteria",
                    "test_commands",
                    "coverage_command",
                )
                if not getattr(task, name)
            ]
            if missing:
                self.store.ask(
                    task_id,
                    "Bitte ergänze die fehlenden Taskfelder: " + ", ".join(missing),
                    "requirements:incomplete",
                )
                return self.store.get(task_id)
            if self.git_enabled:
                started = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "git-workflow",
                    "coding",
                    self.git_service.begin,
                    task,
                )
                if not started:
                    current = self.store.get(task_id)
                    if current.status == Status.ANALYZING:
                        transition(Status.BLOCKED)
                        self.store.event(
                            task_id,
                            EventKind.TASK_BLOCKED,
                            {
                                "reason": "Git repair was declined or requires reconciliation."
                            },
                        )
                    return self.store.get(task_id)
                task = self.store.get(task_id)
                repair = task.git_state.get("repair", {})
                if repair.get("authorized"):
                    context += "\nAUTHORIZED_GIT_REPAIR_JSON:\n" + json.dumps(repair)
            transition(Status.PLANNING)
            if reconciliation is not None:
                from .reconciliation import RecoveryScope

                recovery_scope = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "recovery-review",
                    "validator",
                    RecoveryScope.assess,
                    task,
                    reconciliation,
                    self.router,
                    self.validator,
                )
                self.store.event(
                    task_id,
                    EventKind.RECOVERY_SCOPE,
                    recovery_scope.model_dump(mode="json"),
                )
                context += reconciliation.planner_context()
                context += recovery_scope.planner_context()
            plan = await asyncio.to_thread(
                self._invoke,
                task,
                "planner",
                "planner",
                self.planner.plan,
                task,
                context,
            )
            task.plan = plan.model_dump(mode="json")
            if recovery_scope is not None:
                recovery_scope.validate_plan(plan, self.tools)
                recovery_scope.verify_preserved()
            task.complexity = str(plan.complexity)
            plan_id = self.store.plans.save(
                task_id, plan.summary, plan.model_dump(mode="json")
            )
            self.store.subtasks.save_plan(task_id, plan.subtasks, plan_id)
            self.context.memory.write(
                f"tasks/{task_id}/plan", plan.model_dump_json(indent=2)
            )
            if self.qdrant_enabled:
                try:
                    await asyncio.to_thread(
                        self.memory_service.index,
                        f"tasks/{task_id}/plan",
                    )
                except (httpx.HTTPError, OSError) as exc:
                    self.store.event(
                        task_id,
                        EventKind.MEMORY_INDEX_FAILED,
                        {"error": type(exc).__name__},
                    )
            self.store.tasks.update(
                task_id,
                metadata=task.model_dump(mode="json")
                | {"plan": plan.model_dump(mode="json")},
            )
            transition(Status.READY)
            attempt = 0
            while True:
                transition(Status.EXECUTING)
                open_corrections = self.store.corrections.list_for_task(
                    task_id, status="open"
                )
                for item in open_corrections:
                    self.store.corrections.set_status(item["id"], "in_progress")
                active_corrections = self.store.corrections.list_for_task(
                    task_id, status="in_progress"
                )
                correction_context = [
                    {
                        "category": item["category"],
                        "rule": item["rule"],
                        "message": item["message"],
                        "subtask_id": item["subtask_id"],
                        "affected_paths": item["affected_paths"],
                        "evidence": item["evidence"],
                        "expected": item["expected"],
                    }
                    for item in active_corrections
                ]
                workspace_before = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "validator",
                    "validator",
                    self.validator.workspace_snapshot,
                )
                outputs = []
                for step in plan.ordered_steps():
                    if any(
                        not output.success
                        for output in outputs
                        if output.subtask_id in step.dependencies
                    ):
                        from .agents import ExecutorOutput

                        output = ExecutorOutput(
                            subtask_id=step.id,
                            success=False,
                            output="dependency failed",
                        )
                    else:
                        self.store.subtasks.update(
                            task_id, step.id, "running", plan_id=plan_id
                        )
                        output = await asyncio.to_thread(
                            self._execute_step,
                            task,
                            step,
                            context
                            + "\\nCorrections: "
                            + json.dumps(correction_context),
                            recovery_scope,
                        )
                    outputs.append(output)
                    self.store.subtasks.update(
                        task_id,
                        step.id,
                        "completed" if output.success else "failed",
                        output.model_dump_json(),
                        plan_id,
                    )
                workspace_after = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "validator",
                    "validator",
                    self.validator.workspace_snapshot,
                )
                transition(Status.TESTING)
                tests = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "tester",
                    "validator",
                    self.validator.run_tests,
                    task,
                )
                self.store.event(task_id, EventKind.TESTS_COMPLETED, tests)
                if recovery_scope is not None:
                    recovery_scope.verify_preserved()
                transition(Status.VALIDATING)
                validation = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "validator",
                    "validator",
                    self.validator.validate,
                    task,
                    outputs,
                    tests,
                    self.store.questions.has_open_required(task_id),
                    workspace_before,
                    workspace_after,
                )
                self.store.validations.record(
                    task_id, validation.valid, validation.model_dump_json()
                )
                self.store.tasks.update(
                    task_id,
                    metadata=self.store.get(task_id).model_dump(mode="json")
                    | {
                        "test_result": tests,
                        "validation_result": validation.model_dump(mode="json"),
                    },
                )
                validation_findings = [
                    finding.model_dump(mode="json") for finding in validation.findings
                ]
                if (
                    not validation.valid
                    and not validation_findings
                    and validation.errors
                ):
                    # Compatibility for custom validators that still return only
                    # the legacy errors field; never infer category from its text.
                    validation_findings = [
                        {
                            "category": "validation",
                            "source": "legacy_validator",
                            "rule": "validator.unclassified",
                            "message": "; ".join(validation.errors),
                            "evidence": {"errors": validation.errors},
                        }
                    ]
                for finding in validation_findings:
                    self.store.corrections.record(task_id, finding, plan_id)
                if validation.valid:
                    for item in self.store.corrections.list_for_task(
                        task_id, status="in_progress"
                    ):
                        self.store.corrections.set_status(item["id"], "resolved")
                    if recovery_scope is not None:
                        recovery_scope.verify_preserved()
                    if self.git_enabled:

                        def validate_merged_target(branch, step_outputs=outputs):
                            target_tests = self._invoke(
                                task,
                                "target-tester",
                                "validator",
                                self.validator.run_tests,
                                task,
                            )
                            target_workspace = self._invoke(
                                task,
                                "target-validator",
                                "validator",
                                self.validator.workspace_snapshot,
                            )
                            self.store.event(
                                task_id,
                                EventKind.GIT_TARGET_TESTS,
                                {"branch": branch, "tests": target_tests},
                            )
                            target_validation = self._invoke(
                                task,
                                "target-validator",
                                "validator",
                                self.validator.validate,
                                task,
                                step_outputs,
                                target_tests,
                                self.store.questions.has_open_required(task_id),
                                None,
                                target_workspace,
                                False,
                            )
                            self.store.validations.record(
                                task_id,
                                target_validation.valid,
                                target_validation.model_dump_json(),
                            )
                            current = self.store.get(task_id)
                            self.store.tasks.update(
                                task_id,
                                metadata=current.model_dump(mode="json")
                                | {
                                    "test_result": target_tests,
                                    "validation_result": target_validation.model_dump(
                                        mode="json"
                                    ),
                                },
                            )
                            self.store.event(
                                task_id,
                                EventKind.GIT_TARGET_VALIDATION,
                                {
                                    "branch": branch,
                                    "valid": target_validation.valid,
                                    "errors": target_validation.errors,
                                },
                            )
                            return target_validation.valid

                        merged = await asyncio.to_thread(
                            self._invoke,
                            task,
                            "git-workflow",
                            "coding",
                            self.git_service.finish,
                            task_id,
                            [
                                name
                                for output in outputs
                                for name in output.changed_files
                            ],
                            validate_merged_target,
                        )
                        if not merged:
                            return self.store.get(task_id)
                    transition(Status.COMPLETED)
                    self.store.tasks.update(
                        task_id,
                        result="Tests, coverage and independent acceptance validation passed.",
                    )
                    self.store.event(
                        task_id, EventKind.TASK_COMPLETED, {"attempt": attempt}
                    )
                    return self.store.get(task_id)
                for item in self.store.corrections.list_for_task(
                    task_id, status="in_progress"
                ):
                    self.store.corrections.set_status(item["id"], "open")
                findings = validation.errors
                if attempt == int(
                    self.config.data.get("harness", {}).get(
                        "max_correction_attempts", 2
                    )
                ):
                    raise RuntimeError("; ".join(findings))
                transition(Status.CORRECTING)
                attempt += 1
                self.store.event(
                    task_id,
                    EventKind.CORRECTION_STARTED,
                    {"attempt": attempt, "findings": findings},
                )
        except TaskCancelled:
            if run_control is not None and run_control.reason == "lease_lost":
                self.store.event(task_id, EventKind.TASK_LEASE_LOST, {"owner": owner})
            raise
        except Exception as exc:
            task = self.store.get(task_id)
            if task.status not in TERMINAL and task.status != Status.WAITING_HUMAN:
                transition(Status.FAILED)
                self.store.tasks.update(task_id, result=str(exc))
                self.store.event(task_id, EventKind.TASK_FAILED, {"error": str(exc)})
            raise


def build():
    config = Config()
    config.path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=config.data.get("logging", {}).get("level", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(config.path("logs") / "harness.log"),
            logging.StreamHandler(),
        ],
    )
    store = Store(config)
    return config, store, Orchestrator(store, config)
