"""Evaluate durable memory-monitor operation without changing Vault or Qdrant."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta

MIN_OBSERVATION_WINDOW_SECONDS = 300
REQUIRED_MEMORY_CHECKS = {
    "sqlite",
    "vault",
    "qdrant",
    "embedding_evidence",
}


def required_snapshot_count(interval_seconds):
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or not 5 <= interval_seconds <= 3600
    ):
        raise ValueError("memory monitoring interval is invalid")
    return max(
        3,
        math.ceil(MIN_OBSERVATION_WINDOW_SECONDS / interval_seconds) + 1,
    )


def evaluate_memory_ops_acceptance(
    snapshots,
    *,
    launchd_status,
    source_sha256,
    interval_seconds,
    now=None,
):
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}", source_sha256
    ):
        raise ValueError("source SHA-256 is invalid")
    required = required_snapshot_count(interval_seconds)
    if not isinstance(snapshots, list):
        raise TypeError("memory health snapshots are invalid")
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        raise ValueError("acceptance timestamp must include a timezone")
    enough = len(snapshots) >= required
    selected = snapshots[:required] if enough else snapshots
    parsed = []
    for snapshot in selected:
        try:
            observed = datetime.fromisoformat(snapshot["observed_at"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("memory snapshot timestamp is invalid") from None
        if observed.tzinfo is None:
            raise ValueError("memory snapshot timestamp must include a timezone")
        parsed.append(observed)

    window = 0
    if len(parsed) >= 2:
        window = int((max(parsed) - min(parsed)).total_seconds())
    latest = max(parsed) if parsed else None
    max_age = timedelta(seconds=max(2 * interval_seconds, 30))
    latest_fresh = (
        latest is not None and latest <= reference and reference - latest <= max_age
    )
    required_checks_present = enough and all(
        isinstance(snapshot.get("checks"), dict)
        and REQUIRED_MEMORY_CHECKS.issubset(snapshot["checks"])
        for snapshot in selected
    )
    monitor_source_current = enough and all(
        snapshot.get("source_sha256") == source_sha256 for snapshot in selected
    )
    snapshots_healthy = enough and all(
        snapshot.get("healthy") is True
        and isinstance(snapshot.get("checks"), dict)
        and all(
            snapshot["checks"].get(check) == "healthy"
            for check in REQUIRED_MEMORY_CHECKS
        )
        for snapshot in selected
    )
    checks = {
        "launchd_loaded": isinstance(launchd_status, dict)
        and launchd_status.get("installed") is True
        and launchd_status.get("loaded") is True,
        "snapshot_count_sufficient": enough,
        "observation_window_met": window >= MIN_OBSERVATION_WINDOW_SECONDS,
        "snapshots_healthy": snapshots_healthy,
        "required_checks_present": required_checks_present,
        "latest_snapshot_fresh": latest_fresh,
        "monitor_source_current": monitor_source_current,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "snapshot_count": len(selected),
        "required_snapshot_count": required,
        "observation_window_seconds": window,
        "required_observation_window_seconds": MIN_OBSERVATION_WINDOW_SECONDS,
        "source_sha256": source_sha256,
    }
