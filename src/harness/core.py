"""Core domain, persistence, configuration, orchestration and API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

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
    Database,
    EventRepository,
    ModelRunRepository,
    PlanRepository,
    QuestionRepository,
    SubtaskRepository,
    TaskRepository,
)
from .domain import Status, Task
from .memory import ContextBuilder, EmbeddingProvider, ObsidianMemory, QdrantMemory
from .memory_service import MemoryService
from .security import SecretResolver
from .tools import ToolRegistry
from .workflows import GitWorkflow

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("harness")


class Event(BaseModel):
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
        self.data = data or {}
        self.data.setdefault("paths", {})
        env = self._env()
        secrets = self.data.setdefault("secrets", {})
        secrets.update(env)
        resolver = SecretResolver()
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LMSTUDIO_API_KEY"):
            keychain = resolver.get(name)
            if keychain:
                secrets[name] = keychain
        self.validate()

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
        api = self.data.get("api", {})
        port = api.get("port", 8080)
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("api.port must be between 1 and 65535")
        for name, provider in self.data.get("models", {}).get("providers", {}).items():
            if provider.get("enabled") and not provider.get("base_url"):
                raise ValueError(f"models.providers.{name}.base_url is required")
        return True


class Store:
    def __init__(self, config: Config):
        self.config = config
        self.db = config.path("database")
        self.database = Database(self.db)
        self.tasks = TaskRepository(self.database)
        self.events = EventRepository(self.database)
        self.plans = PlanRepository(self.database)
        self.subtasks = SubtaskRepository(self.database)
        self.questions = QuestionRepository(self.database)
        self.agent_runs = AgentRunRepository(self.database)
        self.model_runs = ModelRunRepository(self.database)
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

    def event(self, task_id: int | None, kind: str, payload: dict):
        now = datetime.now(UTC).isoformat()
        payload = self.audit.sanitize(payload)
        self.events.append(task_id, kind, payload)
        return Event(task_id=task_id, kind=kind, payload=payload, created_at=now)

    def ask(self, task_id, question, reason, options=None, required=True):
        if not question.strip():
            raise ValueError("question must not be blank")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if not task:
                raise ValueError("task not found")
            if required and task["status"] in {"completed", "cancelled"}:
                raise ValueError("terminal task cannot be reopened by a question")
            existing = connection.execute(
                "SELECT id FROM questions WHERE task_id=? AND question=? AND reason=? AND status='open'",
                (task_id, question, reason),
            ).fetchone()
            if existing:
                return existing["id"]
            question_id = connection.execute(
                "INSERT INTO questions(task_id,question,reason,options,required,created_at) VALUES(?,?,?,?,?,?)",
                (
                    task_id,
                    question,
                    reason,
                    json.dumps(options or []),
                    int(required),
                    self.database.now(),
                ),
            ).lastrowid
            if required:
                connection.execute(
                    "UPDATE tasks SET status='waiting_human',updated_at=? WHERE id=?",
                    (self.database.now(), task_id),
                )
        self.event(
            task_id,
            "QUESTION_ASKED",
            {"question_id": question_id, "question": question, "required": required},
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
        )
        self.qdrant = QdrantMemory(
            qdrant.get("url", "http://127.0.0.1:6333"),
            qdrant.get("collection", "harness-memory"),
            embeddings.get("dimensions", 32),
            embedder,
        )
        notes = ObsidianMemory(self.config.path("obsidian_vault"))
        self.memory_service = MemoryService(notes, self.qdrant)
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

        def transition(status):
            self.store.tasks.transition(task_id, status, owner)

        try:
            task = self.store.get(task_id)
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
                    "recovery.reconciled",
                    reconciliation.model_dump(mode="json"),
                )
                self.store.tasks.transition(task_id, Status.BLOCKED, owner)
                self.store.event(
                    task_id,
                    "recovery.inspected",
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
                            "task.blocked",
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
                    task_id, "recovery.scope", recovery_scope.model_dump(mode="json")
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
                        task_id, "memory.index_failed", {"error": type(exc).__name__}
                    )
            self.store.tasks.update(
                task_id,
                metadata=task.model_dump(mode="json")
                | {"plan": plan.model_dump(mode="json")},
            )
            transition(Status.READY)
            findings = []
            attempt = 0
            while True:
                transition(Status.EXECUTING)
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
                            context + "\\nCorrections: " + json.dumps(findings),
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
                transition(Status.TESTING)
                tests = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "tester",
                    "validator",
                    self.validator.run_tests,
                    task,
                )
                self.store.event(task_id, "tests.completed", tests)
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
                )
                with self.store.database.connect() as connection:
                    connection.execute(
                        "INSERT INTO validations(task_id,valid,report,created_at) VALUES(?,?,?,?)",
                        (
                            task_id,
                            int(validation.valid),
                            validation.model_dump_json(),
                            self.store.database.now(),
                        ),
                    )
                self.store.tasks.update(
                    task_id,
                    metadata=self.store.get(task_id).model_dump(mode="json")
                    | {
                        "test_result": tests,
                        "validation_result": validation.model_dump(mode="json"),
                    },
                )
                if validation.valid:
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
                            self.store.event(
                                task_id,
                                "git.target_tests",
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
                            )
                            with self.store.database.connect() as connection:
                                connection.execute(
                                    "INSERT INTO validations(task_id,valid,report,created_at) VALUES(?,?,?,?)",
                                    (
                                        task_id,
                                        int(target_validation.valid),
                                        target_validation.model_dump_json(),
                                        self.store.database.now(),
                                    ),
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
                                "git.target_validation",
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
                    self.store.event(task_id, "task.completed", {"attempt": attempt})
                    return self.store.get(task_id)
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
                    "correction.started",
                    {"attempt": attempt, "findings": findings},
                )
        except Exception as exc:
            task = self.store.get(task_id)
            if task.status not in TERMINAL and task.status != Status.WAITING_HUMAN:
                transition(Status.FAILED)
                self.store.tasks.update(task_id, result=str(exc))
                self.store.event(task_id, "task.failed", {"error": str(exc)})
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
