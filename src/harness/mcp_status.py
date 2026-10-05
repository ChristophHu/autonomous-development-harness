"""Shared read-only MCP status and latest-probe presentation."""

from __future__ import annotations

from datetime import UTC, datetime


class MCPStatusApplicationService:
    """Share read-only MCP status plus last-probe freshness across callers."""

    def __init__(self, report_provider, snapshot_provider, *, clock=None):
        self.report_provider = report_provider
        self.snapshot_provider = snapshot_provider
        self.clock = clock or (lambda: datetime.now(UTC))

    def status(self, reports=None):
        current = self.report_provider() if reports is None else reports
        snapshots = {
            item["server"]: item
            for item in self.snapshot_provider()
            if isinstance(item, dict) and isinstance(item.get("server"), str)
        }
        now = self.clock()
        result = []
        for report in current:
            item = dict(report)
            previous = snapshots.get(item.get("name"))
            item["last_probe"] = None
            if previous is not None:
                last_probe = {
                    "observed_at": previous.get("observed_at"),
                    "state": previous.get("state"),
                    "error_type": previous.get("error_type"),
                }
                try:
                    observed = datetime.fromisoformat(previous["observed_at"])
                    if observed.tzinfo is None:
                        observed = observed.replace(tzinfo=UTC)
                    last_probe["age_seconds"] = max(
                        0, int((now - observed).total_seconds())
                    )
                except (KeyError, TypeError, ValueError):
                    last_probe["age_seconds"] = None
                item["last_probe"] = last_probe
            result.append(item)
        return result
