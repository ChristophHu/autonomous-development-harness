"""Structured, correlated and redacted runtime logs plus durable metrics."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import resource
import sys
import time
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
        self.started_monotonic = time.monotonic()

    def runtime_metrics(self):
        """Return bounded process and local SQLite resource gauges."""
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
        try:
            database_bytes = os.stat(self.database.path).st_size
        except OSError:
            database_bytes = 0
        return {
            "uptime_seconds": round(
                max(0.0, time.monotonic() - self.started_monotonic), 3
            ),
            "process_max_rss_bytes": rss_bytes,
            "database_bytes": database_bytes,
        }

    @staticmethod
    def _prometheus_number(value):
        if value is None or not math.isfinite(float(value)):
            return "NaN"
        return format(float(value), ".12g")

    def prometheus(self):
        """Export aggregate metrics only; never turn persisted names into labels."""
        report = self.metrics()
        counters = {
            "harness_tasks_total": report["tasks"]["total"],
            "harness_events_total": report["events"]["total"],
            "harness_model_runs_total": report["models"]["runs"],
            "harness_prompt_tokens_total": report["models"]["prompt_tokens"],
            "harness_completion_tokens_total": report["models"]["completion_tokens"],
            "harness_tool_calls_total": report["tools"]["total"],
            "harness_validations_total": sum(report["validations"].values()),
            "harness_validations_valid_total": report["validations"]["valid"],
            "harness_validations_invalid_total": report["validations"]["invalid"],
        }
        gauges = {
            "harness_model_cost_total": report["models"]["cost"],
            "harness_tool_call_duration_ms_average": report["tools"]["latency_ms"][
                "average"
            ],
            "harness_tool_call_duration_ms_maximum": report["tools"]["latency_ms"][
                "maximum"
            ],
            "harness_qdrant_probe_latency_ms": report["qdrant"]["latest_latency_ms"],
            "harness_qdrant_collection_points": report["qdrant"]["points"],
            "harness_qdrant_probe_age_seconds": report["qdrant"]["probe_age_seconds"],
        }
        counters["harness_qdrant_probes_total"] = report["qdrant"]["probes"]
        gauges["harness_qdrant_healthy"] = report["qdrant"]["healthy"]
        gauges["harness_process_uptime_seconds"] = report["runtime"]["uptime_seconds"]
        gauges["harness_process_max_rss_bytes"] = report["runtime"][
            "process_max_rss_bytes"
        ]
        gauges["harness_sqlite_database_bytes"] = report["runtime"]["database_bytes"]
        lines = []
        for name, value in (*counters.items(), *gauges.items()):
            metric_type = "counter" if name in counters else "gauge"
            lines.extend(
                (
                    f"# TYPE {name} {metric_type}",
                    f"{name} {self._prometheus_number(value)}",
                )
            )
        return "\n".join(lines) + "\n"

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
            qdrant = connection.execute(
                "SELECT COUNT(*) AS probes, MAX(id) AS latest_id FROM qdrant_probe_snapshots"
            ).fetchone()
            latest_qdrant = (
                connection.execute(
                    "SELECT healthy,latency_ms,points,status,"
                    "MAX(0,(julianday('now')-julianday(observed_at))*86400) AS age_seconds "
                    "FROM qdrant_probe_snapshots WHERE id=?",
                    (qdrant["latest_id"],),
                ).fetchone()
                if qdrant["latest_id"] is not None
                else None
            )
            agent_runs = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM agent_runs GROUP BY status"
                )
            }
            tool_calls = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM tool_calls GROUP BY status"
                )
            }
            tool_rows = connection.execute(
                "SELECT tool, status, COUNT(*) AS count, "
                "AVG(CASE WHEN finished_at IS NOT NULL THEN "
                "MAX(0, (julianday(finished_at)-julianday(started_at))*86400000) END) "
                "AS average_ms, "
                "MAX(CASE WHEN finished_at IS NOT NULL THEN "
                "MAX(0, (julianday(finished_at)-julianday(started_at))*86400000) END) "
                "AS maximum_ms, "
                "SUM(CASE WHEN finished_at IS NOT NULL THEN 1 ELSE 0 END) AS samples "
                "FROM tool_calls GROUP BY tool, status ORDER BY tool, status"
            ).fetchall()
            validations = {
                "valid": connection.execute(
                    "SELECT COUNT(*) FROM validations WHERE valid=1"
                ).fetchone()[0],
                "invalid": connection.execute(
                    "SELECT COUNT(*) FROM validations WHERE valid=0"
                ).fetchone()[0],
            }
        model_runs = dict(model)
        tools_by_name = {}
        tools_by_transport = {}
        latencies = []
        for row in tool_rows:
            summary = tools_by_name.setdefault(
                row["tool"], {"total": 0, "by_status": {}}
            )
            summary["total"] += row["count"]
            summary["by_status"][row["status"]] = row["count"]
            transport = "mcp" if row["tool"].startswith("mcp.") else "native"
            transport_summary = tools_by_transport.setdefault(
                transport, {"total": 0, "by_status": {}}
            )
            transport_summary["total"] += row["count"]
            transport_summary["by_status"][row["status"]] = row["count"]
            if row["samples"]:
                latencies.append(
                    (
                        row["average_ms"] * row["samples"],
                        row["maximum_ms"],
                        row["samples"],
                    )
                )
        latency_samples = sum(item[2] for item in latencies)
        latency_average = (
            sum(item[0] for item in latencies) / latency_samples
            if latency_samples
            else None
        )
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
            "agents": {"total": sum(agent_runs.values()), "by_status": agent_runs},
            "tools": {
                "total": sum(tool_calls.values()),
                "by_status": tool_calls,
                "by_name": tools_by_name,
                "by_transport": tools_by_transport,
                "latency_ms": {
                    "samples": latency_samples,
                    "average": round(latency_average, 3)
                    if latency_average is not None
                    else None,
                    "maximum": round(max((item[1] for item in latencies), default=0), 3)
                    if latency_samples
                    else None,
                },
            },
            "validations": validations,
            "qdrant": {
                "probes": qdrant["probes"],
                "healthy": int(latest_qdrant["healthy"]) if latest_qdrant else 0,
                "latest_latency_ms": round(latest_qdrant["latency_ms"], 3)
                if latest_qdrant
                else None,
                "points": latest_qdrant["points"]
                if latest_qdrant and latest_qdrant["points"] is not None
                else 0,
                "status": latest_qdrant["status"] if latest_qdrant else "unknown",
                "probe_age_seconds": round(latest_qdrant["age_seconds"], 3)
                if latest_qdrant
                else None,
            },
            "runtime": self.runtime_metrics(),
        }
