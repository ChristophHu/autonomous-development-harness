"""Opt-in, read-only probes for SQLite, Obsidian, Qdrant, and live evidence."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import httpx


def sqlite_health(path: str | Path) -> str:
    """Inspect SQLite integrity through a read-only connection."""
    database = Path(path)
    if not database.is_file():
        return "missing"
    connection = None
    try:
        connection = sqlite3.connect(
            f"file:{database.resolve()}?mode=ro", uri=True, timeout=1
        )
        result = connection.execute("PRAGMA quick_check").fetchone()
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        return (
            "healthy"
            if result and result[0] == "ok" and not violations
            else "unhealthy"
        )
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return "unavailable"
    finally:
        if connection is not None:
            connection.close()


def collect_memory_health(
    config,
    store,
    qdrant,
    *,
    source_sha256,
    now=None,
):
    """Return status labels only; do not include note or query contents."""
    from .core import ROOT
    from .evidence import EvidenceRepository, audit_evidence
    from .vault_audit import audit_vault

    memory = config.data.get("memory", {})
    checks = {"sqlite": sqlite_health(store.db)}
    obsidian_enabled = memory.get("obsidian", {}).get("enabled", False)
    if obsidian_enabled:
        try:
            report = audit_vault(
                config.path("obsidian_vault"),
                decision_rows=store.decisions.list_all(),
                content_governance=True,
                source_root=ROOT,
                today=now.date() if now is not None else None,
            )
            checks["vault"] = "healthy" if report["healthy"] else "unhealthy"
        except (OSError, ValueError, TypeError):
            checks["vault"] = "unavailable"
    else:
        checks["vault"] = "disabled"

    qdrant_enabled = memory.get("qdrant", {}).get("enabled", False)
    if qdrant_enabled:
        try:
            report = qdrant.health_report()
            checks["qdrant"] = (
                "healthy"
                if isinstance(report, dict) and report.get("healthy") is True
                else "unhealthy"
            )
        except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
            checks["qdrant"] = "unavailable"
        try:
            rows = EvidenceRepository(store.database).list(kind="embedding", limit=1)
            evidence = audit_evidence(
                rows,
                expected_subject_sha256=source_sha256,
                now=now,
                max_age_hours=memory.get("monitoring", {}).get(
                    "evidence_max_age_hours", 168
                ),
            )
            checks["embedding_evidence"] = (
                "healthy"
                if evidence["healthy"]
                else (evidence["items"][0]["reason"] or "unhealthy")
                if evidence["items"]
                else "missing"
            )
        except (OSError, ValueError, sqlite3.Error):
            checks["embedding_evidence"] = "unavailable"
    else:
        checks["qdrant"] = "disabled"
        checks["embedding_evidence"] = "not_required"

    return {
        "healthy": all(
            state in {"healthy", "disabled", "not_required"}
            for state in checks.values()
        ),
        "checks": checks,
        "source_sha256": source_sha256,
    }


class MemoryHealthMonitor:
    """Persist every probe, emitting only initial state and state transitions."""

    def __init__(
        self, probe, snapshots, *, interval_seconds, sleep=time.sleep, emit=None
    ):
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, int)
            or not 5 <= interval_seconds <= 3600
        ):
            raise ValueError(
                "memory monitor interval must be between 5 and 3600 seconds"
            )
        self.probe = probe
        self.snapshots = snapshots
        self.interval_seconds = interval_seconds
        self.sleep = sleep
        self.emit = emit or (lambda _report: None)

    def run(self, *, max_checks=None):
        if max_checks is not None and (
            isinstance(max_checks, bool)
            or not isinstance(max_checks, int)
            or max_checks < 1
        ):
            raise ValueError("max_checks must be a positive integer")
        previous = None
        completed = 0
        while max_checks is None or completed < max_checks:
            report = self.probe()
            if (
                not isinstance(report, dict)
                or not isinstance(report.get("healthy"), bool)
                or not isinstance(report.get("checks"), dict)
            ):
                raise TypeError("memory monitor probe returned an invalid report")
            self.snapshots.record_memory_health(report)
            if report != previous:
                self.emit(report)
            previous = json.loads(json.dumps(report, sort_keys=True))
            completed += 1
            if max_checks is None or completed < max_checks:
                self.sleep(self.interval_seconds)
        return completed
