"""Core domain, persistence, configuration, orchestration and API."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from .agents import (
    Executor,
    ModelRegistry,
    ModelRouter,
    Planner,
    ProfileRegistry,
    execute_plan_dag,
)
from .approvals import ApprovalDenied, ApprovalRequired, ApprovalService
from .audit import AuditRecorder
from .configuration import HarnessConfig
from .database import (
    AgentRunRepository,
    ArtifactRepository,
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
from .errors import failure_record
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
        "context": {"max_bytes": 65536},
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
    STARTABLE: ClassVar[set[Status]] = {
        Status.PENDING,
        Status.FAILED,
        Status.BLOCKED,
        Status.RECOVERING,
    }

    @classmethod
    def can_start(cls, status, has_open_question=False):
        state = Status(status)
        return not has_open_question and state in cls.STARTABLE | {
            Status.WAITING_HUMAN,
            Status.WAITING_DECISION,
            Status.WAITING_APPROVAL,
        }


class ConfigurationService:
    def __init__(self, path: Path | None = None):
        if path is None:
            path = os.environ.get("HARNESS_CONFIG_PATH", ROOT / "config.yaml")
        self._path = Path(path)
        self.reload()

    def reload(self):
        """Reload all sources atomically, retaining the previous valid snapshot on failure."""
        previous_configured = getattr(self, "configured", None)
        previous_data = getattr(self, "data", None)
        path = self._path
        try:
            data = yaml.safe_load(path.read_text()) if path.exists() else {}
        except yaml.YAMLError:
            raise ValueError(f"invalid YAML in {path.name}") from None
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a mapping")  # noqa: TRY004
        if "secrets" in data and not isinstance(data["secrets"], dict):
            raise ValueError("secrets must be a mapping")
        environment = self._env()
        source_models = data.get("models", {})
        source_model_defaults = (
            source_models.get("defaults", {}) if isinstance(source_models, dict) else {}
        )
        default_provider = (
            source_model_defaults.get("provider", "lmstudio")
            if isinstance(source_model_defaults, dict)
            else "lmstudio"
        )
        raw = self._merge(
            data,
            self._environment_overrides(environment, default_provider=default_provider),
        )
        secrets = raw.setdefault("secrets", {})
        resolver = SecretResolver()
        secret_names = set(secrets) | {
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "LMSTUDIO_API_KEY",
        }
        for name in sorted(secret_names):
            keychain = resolver.get(name)
            if keychain:
                secrets[name] = keychain
        self.configured = raw
        self.data = self._merge(CONFIG_DEFAULTS, raw)
        try:
            self.validate()
        except Exception:
            if previous_configured is not None and previous_data is not None:
                self.configured = previous_configured
                self.data = previous_data
            raise
        return self

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
            for line_number, line in enumerate(p.read_text().splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[7:].lstrip()
                if "=" not in stripped:
                    raise ValueError(f".env line {line_number} must be KEY=VALUE")
                key, value = stripped.split("=", 1)
                key = key.strip()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise ValueError(
                        f".env line {line_number} has an invalid variable name"
                    )
                result[key] = self._dotenv_value(value)
        result.update(os.environ)
        return result

    @staticmethod
    def _dotenv_value(value):
        value = value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0:
                raise ValueError(".env contains an unterminated quoted value")
            if value[end + 1 :].strip() and not value[end + 1 :].lstrip().startswith(
                "#"
            ):
                raise ValueError(".env contains invalid text after a quoted value")
            return value[1:end]
        return value.split(" #", 1)[0].rstrip()

    @classmethod
    def _environment_overrides(cls, environment, *, default_provider="lmstudio"):
        result = {}
        paths = {
            "HARNESS_NAME": ("harness", "name"),
            "HARNESS_ENVIRONMENT": ("harness", "environment"),
            "HARNESS_WORKSPACE": ("paths", "workspace"),
            "OBSIDIAN_VAULT_PATH": ("paths", "obsidian_vault"),
            "HARNESS_LOGS_PATH": ("paths", "logs"),
            "HARNESS_DATABASE_PATH": ("paths", "database"),
            "QDRANT_URL": ("memory", "qdrant", "url"),
            "QDRANT_COLLECTION": ("memory", "qdrant", "collection"),
            "GIT_REMOTE_URL": ("git", "remote"),
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
        }
        selected = {key: value for key, value in environment.items() if key in paths}
        selected.update(
            {
                key: value
                for key, value in environment.items()
                if (key.endswith("_API_KEY") and str(value).strip())
                or (key != "LLM_MODEL" and re.fullmatch(r"[A-Z][A-Z0-9_]*_MODEL", key))
            }
        )
        for key, value in selected.items():
            if key in paths:
                target = result
                for part in paths[key][:-1]:
                    target = target.setdefault(part, {})
                if key == "API_PORT":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        pass
                target[paths[key][-1]] = value
            elif key.endswith("_API_KEY"):
                result.setdefault("secrets", {})[key] = value
            else:
                provider = key[:-6].lower()
                result.setdefault("models", {}).setdefault("providers", {}).setdefault(
                    provider, {}
                )["model"] = value
        if "LLM_PROVIDER" in environment:
            result.setdefault("models", {}).setdefault("defaults", {})["provider"] = (
                environment["LLM_PROVIDER"]
            )
        if "LLM_MODEL" in environment:
            result.setdefault("models", {}).setdefault("defaults", {})["model"] = (
                environment["LLM_MODEL"]
            )
        if "LLM_API_URL" in environment:
            provider = str(environment.get("LLM_PROVIDER", default_provider)).lower()
            result.setdefault("models", {}).setdefault("providers", {}).setdefault(
                provider, {}
            )["base_url"] = environment["LLM_API_URL"]
        return result

    def path(self, key: str) -> Path:
        paths = self.settings.paths
        value = getattr(paths, key, None)
        if value is None:
            value = (paths.model_extra or {}).get(key, f"./{key}")
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def settings(self) -> HarnessConfig:
        """Return a fresh typed view, so legacy data mutations cannot stale it."""
        try:
            return HarnessConfig.model_validate(self.data)
        except ValidationError as exc:
            location = exc.errors(include_input=False)[0]["loc"]
            path = ".".join(str(part) for part in location)
            raise ValueError(
                f"{path} has an invalid configuration type or value"
            ) from None

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

        def integer(value, name, *, minimum, maximum):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be an integer between {minimum} and {maximum}"
                )

        def secret_value(value, name):
            if isinstance(value, dict):
                for child, item in value.items():
                    secret_value(item, f"{name}.{child}")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

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
        harness = mapping(root.get("harness", {}), "harness")
        text(harness.get("name", "autonomous-development-harness"), "harness.name")
        text(harness.get("environment", "development"), "harness.environment")
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
        for name in ("workspace", "obsidian_vault", "logs", "database"):
            if name in paths:
                text(paths[name], f"paths.{name}")
        database = mapping(root.get("database", {}), "database")
        if database.get("type", "sqlite") != "sqlite":
            raise ValueError("database.type must be sqlite")

        git = mapping(root.get("git", {}), "git")
        boolean(git.get("enabled", False), "git.enabled")
        for name in ("main", "dev"):
            text(git.get(name, name), f"git.{name}")
        text(git.get("remote"), "git.remote", optional=True)
        branches = mapping(git.get("branches", {}), "git.branches")
        for name, value in branches.items():
            text(value, f"git.branches.{name}")
        workflow = mapping(git.get("workflow", {}), "git.workflow")
        for name, value in workflow.items():
            text(value, f"git.workflow.{name}")

        docker = mapping(root.get("docker", {}), "docker")
        if "enabled" in docker:
            boolean(docker["enabled"], "docker.enabled")
        if "compose_preferred" in docker:
            boolean(docker["compose_preferred"], "docker.compose_preferred")
        if "compose_file" in docker:
            text(docker["compose_file"], "docker.compose_file")

        api_enabled = root.get("api", {}).get("enabled")
        if api_enabled is not None:
            boolean(api_enabled, "api.enabled")

        memory = mapping(root.get("memory", {}), "memory")
        for section in ("obsidian", "qdrant"):
            section_data = mapping(memory.get(section, {}), f"memory.{section}")
            boolean(
                section_data.get("enabled", section == "obsidian"),
                f"memory.{section}.enabled",
            )
        monitoring = mapping(memory.get("monitoring", {}), "memory.monitoring")
        boolean(monitoring.get("enabled", False), "memory.monitoring.enabled")
        integer(
            monitoring.get("interval_seconds", 60),
            "memory.monitoring.interval_seconds",
            minimum=5,
            maximum=3600,
        )
        integer(
            monitoring.get("evidence_max_age_hours", 168),
            "memory.monitoring.evidence_max_age_hours",
            minimum=1,
            maximum=8760,
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
        defaults = mapping(models.get("defaults", {}), "models.defaults")
        for field in ("provider", "model"):
            if field in defaults:
                text(defaults[field], f"models.defaults.{field}")
        mapping(models.get("rates", {}), "models.rates")
        strategies = mapping(models.get("strategies", {}), "models.strategies")
        for name, strategy in strategies.items():
            strategy = mapping(strategy, f"models.strategies.{name}")
            for field in ("provider", "model", "profile"):
                if field in strategy:
                    text(strategy[field], f"models.strategies.{name}.{field}")

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
            if "instructions" in profile:
                text(profile["instructions"], f"profiles.{name}.instructions")
            if "max_steps" in profile and (
                isinstance(profile["max_steps"], bool)
                or not isinstance(profile["max_steps"], int)
                or profile["max_steps"] < 1
            ):
                raise ValueError(
                    f"profiles.{name}.max_steps must be a positive integer"
                )

        logging_config = mapping(root.get("logging", {}), "logging")
        level = logging_config.get("level", "INFO")
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "logging.level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        text(logging_config.get("file", "./logs/harness.log"), "logging.file")
        testing = mapping(root.get("testing", {}), "testing")
        if "tdd" in testing:
            boolean(testing["tdd"], "testing.tdd")
        coverage = mapping(testing.get("coverage", {}), "testing.coverage")
        for name, value in coverage.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 100
            ):
                raise ValueError(f"testing.coverage.{name} must be between 0 and 100")
        tools = mapping(root.get("tools", {}), "tools")
        permissions = mapping(tools.get("permissions", {}), "tools.permissions")
        for name, value in permissions.items():
            text(value, f"tools.permissions.{name}")
        http = mapping(tools.get("http", {}), "tools.http")
        timeout(http.get("timeout", 30), "tools.http.timeout")
        allowed_hosts = http.get("allowed_hosts", [])
        if not isinstance(allowed_hosts, list) or any(
            not isinstance(host, str) or not host.strip() for host in allowed_hosts
        ):
            raise ValueError("tools.http.allowed_hosts must be a list of strings")
        docker_tools = mapping(tools.get("docker", {}), "tools.docker")
        if "compose_file" in docker_tools:
            text(docker_tools["compose_file"], "tools.docker.compose_file")
        if "socket_path" in docker_tools:
            text(docker_tools["socket_path"], "tools.docker.socket_path", optional=True)
        git_tools = mapping(tools.get("git", {}), "tools.git")
        for hosts_key in ("allowed_hosts",):
            hosts = git_tools.get(hosts_key, [])
            if not isinstance(hosts, list) or any(
                not isinstance(host, str) or not host.strip() for host in hosts
            ):
                raise ValueError(f"tools.git.{hosts_key} must be a list of strings")
        mapping(git_tools.get("credentials", {}), "tools.git.credentials")
        ssh = mapping(git_tools.get("ssh", {}), "tools.git.ssh")
        hosts = ssh.get("allowed_hosts", [])
        if not isinstance(hosts, list) or any(
            not isinstance(host, str) or not host.strip() for host in hosts
        ):
            raise ValueError("tools.git.ssh.allowed_hosts must be a list of strings")
        ports = ssh.get("allowed_ports", [22])
        if not isinstance(ports, list) or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
            for port in ports
        ):
            raise ValueError("tools.git.ssh.allowed_ports must contain valid ports")
        mapping(ssh.get("host_keys", {}), "tools.git.ssh.host_keys")
        mapping(ssh.get("credentials", {}), "tools.git.ssh.credentials")
        secrets_config = mapping(root.get("secrets", {}), "secrets")
        for name, value in secrets_config.items():
            secret_value(value, f"secrets.{name}")
        _typed_settings = self.settings
        return True


# Backwards-compatible public name used throughout the runtime.
Config = ConfigurationService


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
        self.artifacts = ArtifactRepository(self.database)
        self.subtasks = SubtaskRepository(self.database)
        self.questions = QuestionRepository(self.database)
        self.decisions = DecisionRepository(self.database)
        self.agent_runs = AgentRunRepository(self.database)
        self.model_runs = ModelRunRepository(self.database)
        self.model_usage = ModelUsageReportService(self.model_runs)
        self.audit = AuditRecorder(self.database, config.data.get("secrets", {}))
        self.db.parent.mkdir(parents=True, exist_ok=True)

    def create(self, task: Task) -> Task:
        safe = self.audit.sanitize(task.model_dump(mode="json"))
        task.id = self.tasks.create(
            safe["title"],
            safe["description"],
            safe["status"],
            safe,
        )
        return self.get(task.id)

    def get(self, task_id: int) -> Task | None:
        row = self.tasks.get(task_id)
        if row is None:
            return None
        metadata = self.audit.sanitize(json.loads(row["metadata"]))
        fields = self.audit.sanitize(
            {
                "title": row["title"],
                "description": row["description"],
                "status": row["status"],
                "result": row["result"],
            }
        )
        return Task(
            **{
                key: value
                for key, value in metadata.items()
                if key
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
            title=fields["title"],
            description=fields["description"],
            status=fields["status"],
            result=fields["result"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update(self, task: Task):
        fields = self.audit.sanitize(
            {
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "result": task.result,
                "metadata": task.model_dump(mode="json"),
            }
        )
        self.tasks.update(task.id, **fields)

    def update_task_fields(self, task_id, **fields):
        self.tasks.update(task_id, **self.audit.sanitize(fields))

    def update_task_if_idle(self, task_id, task):
        fields = self.audit.sanitize(task.model_dump(mode="json"))
        return self.tasks.update_if_idle(
            task_id,
            fields["title"],
            fields["description"],
            fields,
        )

    def save_plan(self, task_id, summary, payload):
        return self.plans.save(
            task_id,
            self.audit.sanitize(summary),
            self.audit.sanitize(payload),
        )

    def save_subtasks(self, task_id, subtasks, plan_id=None):
        from .agents import Subtask

        safe = []
        for step in subtasks:
            fields = self.audit.sanitize(step.model_dump(mode="json"))
            safe.append(Subtask.model_validate(fields))
        return self.subtasks.save_plan(task_id, safe, plan_id)

    def update_subtask(self, task_id, external_id, status, output=None, plan_id=None):
        safe_output = (
            json.dumps(self.audit.sanitize(output)) if output is not None else None
        )
        return self.subtasks.update(task_id, external_id, status, safe_output, plan_id)

    def record_validation(self, task_id, valid, report):
        return self.validations.record(task_id, valid, self.audit.sanitize(report))

    def record_correction(self, task_id, finding, plan_id=None):
        return self.corrections.record(task_id, self.audit.sanitize(finding), plan_id)

    def save_artifact(self, task_id, key, content, expected_version=None):
        row = self.artifacts.save(
            task_id,
            self.audit.sanitize(key),
            self.audit.sanitize(content),
            expected_version,
        )
        return self._sanitize_artifact(row)

    def artifact(self, task_id, key):
        row = self.artifacts.latest(task_id, key)
        return self._sanitize_artifact(row) if row else None

    def artifact_history(self, task_id, key):
        return [
            self._sanitize_artifact(row) for row in self.artifacts.history(task_id, key)
        ]

    def artifacts_for_task(self, task_id):
        return [
            self._sanitize_artifact(row)
            for row in self.artifacts.latest_for_task(task_id)
        ]

    def _sanitize_artifact(self, row):
        row = dict(row)
        safe = self.audit.sanitize(
            {"key": row["artifact_key"], "content": row["content"]}
        )
        row["artifact_key"] = safe["key"]
        row["content"] = safe["content"]
        return row

    def list_events(self, *filters):
        rows = []
        for row in self.events.list(*filters):
            item = dict(row)
            item["payload"] = json.dumps(
                self.audit.sanitize(json.loads(item["payload"]))
            )
            rows.append(item)
        return rows

    def list_questions(self, task_id):
        return [self._sanitize_question(row) for row in self.questions.list(task_id)]

    def get_question(self, question_id):
        row = self.questions.get(question_id)
        return self._sanitize_question(row) if row is not None else None

    def _sanitize_question(self, row):
        item = dict(row)
        item.update(
            self.audit.sanitize(
                {
                    "question": item["question"],
                    "reason": item["reason"],
                    "options": json.loads(item["options"]),
                    "answer": item["answer"],
                }
            )
        )
        item["options"] = json.dumps(item["options"])
        return item

    def latest_plan(self, task_id):
        return self.audit.sanitize(self.plans.latest(task_id))

    def latest_validation(self, task_id):
        return self.audit.sanitize(self.validations.latest(task_id))

    def event(self, task_id: int | None, kind: EventKind | str, payload: dict):
        event_kind = EventKind(kind)
        now = datetime.now(UTC).isoformat()
        payload = self.audit.sanitize(payload)
        self.events.append(task_id, event_kind, payload)
        return Event(
            task_id=task_id, kind=event_kind.value, payload=payload, created_at=now
        )

    def ask(
        self, task_id, question, reason, options=None, required=True, purpose="input"
    ):
        if not question.strip():
            raise ValueError("question must not be blank")
        safe_fields = self.audit.sanitize(
            {"question": question, "reason": reason, "options": options or []}
        )
        question, reason, options = (
            safe_fields["question"],
            safe_fields["reason"],
            safe_fields["options"],
        )
        payload = self.audit.sanitize({"question": question, "required": required})
        question_id = self.questions.ask(
            task_id, question, reason, options, required, payload, purpose
        )
        return question_id

    def answer(self, question_id, answer, task_id=None):
        return self.questions.answer(question_id, self.audit.sanitize(answer), task_id)


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
        from .observability import ObservabilityService

        self.observability = ObservabilityService(store.database, EventKind)
        self.role_profiles = ProfileRegistry(self.config)
        self.role_profiles.role_profiles()
        self.planner_profile = self.role_profiles.for_role("planner").name
        self.requirements_profile = self.role_profiles.for_role("requirements").name
        self.executor_profile = self.role_profiles.for_role("executor").name
        self.tester_profile = self.role_profiles.for_role("tester").name
        self.validator_profile = self.role_profiles.for_role("independent-review").name
        self.recovery_profile = self.role_profiles.for_role("recovery-inspector").name
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        registry = ModelRegistry(self.config)
        self.audit = store.audit
        router = ModelRouter(registry, self.config, audit=self.audit)
        self.models = registry
        from .database import OperationalSnapshotRepository
        from .services import ModelOperationsService

        self.model_operations = ModelOperationsService(
            registry, OperationalSnapshotRepository(store.database)
        )
        self.router = router
        self.planner = Planner(router)
        workspace = self.config.path("workspace")
        self.tools = ToolRegistry(Permissions(self.config), self._tool_event, workspace)
        self.approvals = ApprovalService(self.store.questions, self.store)
        self.tools.approvals = self.approvals
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
            api_key=self.config.data.get("secrets", {}).get("QDRANT__SERVICE__API_KEY"),
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
            max_bytes=self.config.data.get("memory", {})
            .get("context", {})
            .get("max_bytes", 65536),
        )
        from .services import VaultKnowledgeApplicationService

        self.knowledge_operations = VaultKnowledgeApplicationService(
            self.config.path("obsidian_vault"),
            workspace,
            sanitizer=self.store.audit.sanitize,
        )

    def _tool_event(self, kind, payload):
        self.audit.tool_event(kind, payload)

    def _invoke(self, task, agent, profile, operation, *args, **kwargs):
        with self.audit.agent(task, agent, profile) as span:
            result = operation(*args, **kwargs)
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
            execute = self.executor.execute
            parameters = inspect.signature(execute).parameters.values()
            accepts_complexity = any(
                parameter.name == "complexity"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            return self._invoke(
                task,
                step.assigned_agent,
                step.profile,
                execute,
                step,
                context,
                **({"complexity": task.complexity} if accepts_complexity else {}),
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
                Status.WAITING_DECISION,
                Status.WAITING_APPROVAL,
            }
            if interrupted or (
                task.status in {Status.FAILED, Status.BLOCKED}
                and self.store.latest_plan(task_id) is not None
            ):
                if task.status != Status.RECOVERING:
                    transition(Status.RECOVERING)
                from .reconciliation import ReconciliationService

                reconciliation = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "recovery",
                    self.recovery_profile,
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
            from .claim_evidence import RoutedClaimVerifier
            from .requirements import RequirementCompleter

            task, context = await asyncio.to_thread(
                self._invoke,
                task,
                "requirements",
                self.requirements_profile,
                RequirementCompleter(
                    self.store,
                    self.router,
                    claim_verifier=RoutedClaimVerifier(
                        self.router,
                        author_profile=self.requirements_profile,
                        verifier_profile=self.validator_profile,
                    ),
                ).complete,
                task,
                lambda: self.context.build_evidence(
                    task, str(self.config.path("workspace"))
                ),
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
                pending_conflict_decision = any(
                    item["status"] == "open"
                    and item.get("required")
                    and item.get("purpose") == "decision"
                    and item.get("reason", "").startswith("requirements:conflict:")
                    for item in self.store.list_questions(task_id)
                )
                if pending_conflict_decision:
                    return self.store.get(task_id)
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
                    self.validator_profile,
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
                self.planner_profile,
                self.planner.plan,
                task,
                context,
            )
            task.plan = plan.model_dump(mode="json")
            if recovery_scope is not None:
                recovery_scope.validate_plan(plan, self.tools)
                recovery_scope.verify_preserved()
            task.complexity = str(plan.complexity)
            safe_plan = self.store.audit.sanitize(plan.model_dump(mode="json"))
            plan_id = self.store.save_plan(task_id, plan.summary, safe_plan)
            self.store.save_subtasks(task_id, plan.subtasks, plan_id)
            self.context.memory.write(
                f"tasks/{task_id}/plan", json.dumps(safe_plan, indent=2)
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
            self.store.update_task_fields(
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
                    self.validator_profile,
                    self.validator.workspace_snapshot,
                )
                ordered_steps = plan.ordered_steps()

                async def execute_step(step, corrections=correction_context):
                    self.store.subtasks.update(
                        task_id, step.id, "running", plan_id=plan_id
                    )
                    output = await asyncio.to_thread(
                        self._execute_step,
                        task,
                        step,
                        context + "\\nCorrections: " + json.dumps(corrections),
                        recovery_scope,
                    )

                    return output

                def persist_step(step, output):
                    self.store.update_subtask(
                        task_id,
                        step.id,
                        "completed" if output.success else "failed",
                        output.model_dump(mode="json"),
                        plan_id,
                    )
                    artifact_key = f"agent/{step.id}"
                    prior_artifact = self.store.artifact(task_id, artifact_key)
                    self.store.save_artifact(
                        task_id,
                        artifact_key,
                        json.dumps(output.model_dump(mode="json"), ensure_ascii=False),
                        expected_version=(
                            prior_artifact["version"] if prior_artifact else 0
                        ),
                    )

                outputs = await execute_plan_dag(
                    ordered_steps,
                    execute_step,
                    max_parallel_steps=self.config.data.get("harness", {}).get(
                        "max_parallel_steps", 4
                    ),
                    on_complete=persist_step,
                )
                workspace_after = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "validator",
                    self.validator_profile,
                    self.validator.workspace_snapshot,
                )
                transition(Status.TESTING)
                tests = await asyncio.to_thread(
                    self._invoke,
                    task,
                    "tester",
                    self.tester_profile,
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
                    self.validator_profile,
                    self.validator.validate,
                    task,
                    outputs,
                    tests,
                    self.store.questions.has_open_required(task_id),
                    workspace_before,
                    workspace_after,
                )
                self.store.record_validation(
                    task_id,
                    validation.valid,
                    validation.model_dump(mode="json"),
                )
                self.store.update_task_fields(
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
                    self.store.record_correction(task_id, finding, plan_id)
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
                                self.tester_profile,
                                self.validator.run_tests,
                                task,
                            )
                            target_workspace = self._invoke(
                                task,
                                "target-validator",
                                self.validator_profile,
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
                                self.validator_profile,
                                self.validator.validate,
                                task,
                                step_outputs,
                                target_tests,
                                self.store.questions.has_open_required(task_id),
                                None,
                                target_workspace,
                                False,
                            )
                            self.store.record_validation(
                                task_id,
                                target_validation.valid,
                                target_validation.model_dump(mode="json"),
                            )
                            current = self.store.get(task_id)
                            self.store.update_task_fields(
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
                    self.store.update_task_fields(
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
        except ApprovalRequired:
            # ApprovalService.ask atomically persisted the exact action and
            # transitioned this task to WAITING_HUMAN; return that durable state.
            return self.store.get(task_id)
        except ApprovalDenied:
            task = self.store.get(task_id)
            if task.status not in TERMINAL:
                self.store.tasks.transition(task_id, Status.BLOCKED, owner)
                self.store.event(
                    task_id,
                    EventKind.TASK_BLOCKED,
                    {"reason": "human denied a specific tool action"},
                )
            return self.store.get(task_id)
        except TaskCancelled:
            if run_control is not None and run_control.reason == "lease_lost":
                self.store.event(task_id, EventKind.TASK_LEASE_LOST, {"owner": owner})
            raise
        except Exception as exc:
            task = self.store.get(task_id)
            if task.status not in TERMINAL and task.status not in {
                Status.WAITING_HUMAN,
                Status.WAITING_DECISION,
                Status.WAITING_APPROVAL,
            }:
                transition(Status.FAILED)
                failure = failure_record(exc, self.store.audit.sanitize)
                self.store.update_task_fields(task_id, result=failure["message"])
                self.store.event(
                    task_id,
                    EventKind.TASK_FAILED,
                    {
                        "error": failure["message"],
                        "error_type": failure["error_type"],
                        "category": failure["category"],
                    },
                )
                exc.args = (failure["message"],)
            raise


def build():
    config = Config()
    log_settings = config.settings.logging
    log_path = Path(log_settings.file)
    if not log_path.is_absolute():
        log_path = config._path.parent / log_path
    from .observability import configure_logging

    configure_logging(
        log_path,
        level=log_settings.level,
        max_bytes=log_settings.max_bytes,
        backup_count=log_settings.backup_count,
        secrets=config.data.get("secrets", {}).values(),
    )
    store = Store(config)
    return config, store, Orchestrator(store, config)
