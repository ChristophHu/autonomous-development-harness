"""Context-local provenance and redaction without recording model prompts."""

import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

from .database import AgentRunRepository, AuditRepository

CURRENT_RUN = ContextVar("harness_run", default=None)
LOGGER = logging.getLogger("harness")


def _sensitive_key(key):
    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    return any(
        marker in normalized
        for marker in (
            "authorization",
            "password",
            "secret",
            "token",
            "apikey",
            "credential",
            "privatekey",
            "cookie",
        )
    )


class AuditRecorder:
    def __init__(self, database, secrets=None):
        self.repository = AuditRepository(database)
        self.agents = AgentRunRepository(database)
        self.secrets = secrets or {}

    def sanitize(self, value):
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if _sensitive_key(key) else self.sanitize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        if isinstance(value, str):
            for key, secret in self.secrets.items():
                if (
                    isinstance(secret, str)
                    and secret
                    and not str(key).endswith("_MODEL")
                ):
                    value = value.replace(secret, "[REDACTED]")
        return value

    @contextmanager
    def agent(self, task, agent, profile):
        agent, profile = self.sanitize(agent), self.sanitize(profile)
        run_id = self.agents.start(task.id, agent, profile)
        context = {
            "task_id": task.id,
            "agent_run_id": run_id,
            "agent": agent,
            "profile": profile,
            "complexity": task.complexity,
        }
        token = CURRENT_RUN.set(self.sanitize(context))
        span = {"status": "completed", "output": None}
        try:
            yield span
        except BaseException as exc:
            span.update(status="failed", output={"error_type": type(exc).__name__})
            raise
        finally:
            self.agents.finish(
                run_id, span["status"], json.dumps(self.sanitize(span["output"]))
            )
            CURRENT_RUN.reset(token)
            LOGGER.info(
                "agent.finished task=%s agent=%s profile=%s status=%s",
                task.id,
                agent,
                profile,
                span["status"],
            )

    @contextmanager
    def model(self, provider, model, fallback_index=0):
        run_id = self.repository.start_model(
            self.sanitize(provider),
            self.sanitize(model),
            CURRENT_RUN.get() or {},
            fallback_index,
        )
        started = time.monotonic()
        span = {"usage": None, "provider": provider, "model": model}
        try:
            yield span
        except BaseException as exc:
            self._finish_model(run_id, "failed", started, span, type(exc).__name__)
            raise
        else:
            self._finish_model(run_id, "completed", started, span)

    def _finish_model(self, run_id, status, started, span, error_type=None):
        usage = span["usage"]
        if usage is not None:
            usage = replace(
                usage,
                provider=self.sanitize(usage.provider),
                model=self.sanitize(usage.model),
            )
        self.repository.finish_model(
            run_id,
            status,
            (time.monotonic() - started) * 1000,
            usage,
            error_type,
            self.sanitize(span["provider"]),
            self.sanitize(span["model"]),
        )

    def tool_event(self, kind, payload):
        context = CURRENT_RUN.get() or {}
        self.repository.tool_event(
            context.get("task_id"), kind, self.sanitize(payload), context
        )
