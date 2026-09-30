"""Structured, correlated and redacted runtime logs plus durable metrics."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ASSIGNMENT = re.compile(
    r"(?i)(authorization|password|secret|api[_-]?key|access[_-]?token)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def redact_log_message(message, secrets=()):
    """Redact common inline credentials and configured secret values."""
    if not isinstance(message, str):
        message = str(message)
    message = _ASSIGNMENT.sub(r"\1\2[REDACTED]", message)
    message = _BEARER.sub("Bearer [REDACTED]", message)
    for secret in secrets:
        if isinstance(secret, str) and secret:
            message = message.replace(secret, "[REDACTED]")
    return message


class JsonLogFormatter(logging.Formatter):
    def __init__(self, secrets=()):
        super().__init__()
        self.secrets = tuple(secrets)

    def format(self, record):
        from .audit import CURRENT_RUN

        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_log_message(record.getMessage(), self.secrets),
        }
        context = CURRENT_RUN.get() or {}
        for field in ("task_id", "agent_run_id", "agent", "profile"):
            if context.get(field) is not None:
                payload[field] = context[field]
        if record.exc_info and record.exc_info[0]:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    path, *, level="INFO", max_bytes=10_485_760, backup_count=3, secrets=()
):
    """Install idempotent rotating JSON handlers on the harness logger."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("harness")
    logger.setLevel(getattr(logging, level))
    for handler in tuple(logger.handlers):
        if getattr(handler, "_harness_managed", False):
            logger.removeHandler(handler)
            handler.close()
    handlers = [
        RotatingFileHandler(
            destination,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ]
    for handler in handlers:
        handler._harness_managed = True
        handler.setFormatter(JsonLogFormatter(secrets))
        logger.addHandler(handler)
    return logger


class ObservabilityService:
    """Read-only, durable counters derived from the canonical SQLite store."""

    def __init__(self, database, event_kinds):
        self.database = database
        self.event_kinds = tuple(sorted(str(item.value) for item in event_kinds))

    def metrics(self):
        with self.database.connect() as connection:
            tasks = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
                )
            }
            events = {
                row["kind"]: row["count"]
                for row in connection.execute(
                    "SELECT kind, COUNT(*) AS count FROM events GROUP BY kind"
                )
            }
            model = connection.execute(
                "SELECT COUNT(*) AS runs, SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, SUM(cost) AS cost, "
                "SUM(CASE WHEN prompt_tokens IS NULL OR completion_tokens IS NULL "
                "THEN 1 ELSE 0 END) AS missing_usage "
                "FROM model_runs"
            ).fetchone()
        model_runs = dict(model)
        for key in ("prompt_tokens", "completion_tokens", "cost"):
            if model_runs[key] is None:
                model_runs[key] = 0
        return {
            "tasks": {"total": sum(tasks.values()), "by_status": tasks},
            "events": {
                "total": sum(events.values()),
                "by_kind": events,
                "catalogue": list(self.event_kinds),
            },
            "models": model_runs,
        }
