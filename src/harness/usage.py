"""Validated, shared read service for persisted model usage reports."""

from datetime import UTC, datetime


class ModelUsageReportService:
    def __init__(self, repository):
        self.repository = repository

    @staticmethod
    def _text(value, name):
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a non-empty string")
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _date(value, name):
        if value is None:
            return None
        try:
            result = (
                value if isinstance(value, datetime) else datetime.fromisoformat(value)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an ISO-8601 datetime") from error
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
        return result.astimezone(UTC).isoformat()

    def report(
        self,
        *,
        task_id=None,
        agent=None,
        profile=None,
        provider=None,
        model=None,
        group_by=None,
        since=None,
        until=None,
        limit=50,
        offset=0,
    ):
        if task_id is not None and (
            isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 1
        ):
            raise ValueError("task_id must be a positive integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 200
        ):
            raise ValueError("limit must be an integer between 1 and 200")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        provider = self._text(provider, "provider")
        model = self._text(model, "model")
        agent = self._text(agent, "agent")
        profile = self._text(profile, "profile")
        if group_by is not None and group_by not in {
            "task_id",
            "agent",
            "profile",
            "provider",
            "model",
            "day",
        }:
            raise ValueError("invalid group_by dimension")
        since = self._date(since, "since")
        until = self._date(until, "until")
        if since is not None and until is not None and since > until:
            raise ValueError("since must not be later than until")
        report = self.repository.usage_report(
            task_id=task_id,
            agent=agent,
            profile=profile,
            provider=provider,
            model=model,
            group_by=group_by,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
        return report | {"limit": limit, "offset": offset}
