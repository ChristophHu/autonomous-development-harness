"""Shared review policy for curated Vault audits and knowledge retrieval."""

from datetime import date, datetime, timedelta


def classify_review(metadata, today: date, max_age_days: int):
    """Return (state, ISO date) without trusting malformed review metadata."""
    reviewed = metadata.get("last_reviewed", metadata.get("reviewed_on"))
    if reviewed is None:
        return "unreviewed", None
    if isinstance(reviewed, datetime):
        reviewed = reviewed.date()
    elif isinstance(reviewed, str):
        try:
            reviewed = date.fromisoformat(reviewed)
        except ValueError:
            try:
                reviewed = datetime.fromisoformat(reviewed).date()
            except ValueError:
                return "invalid", None
    if not isinstance(reviewed, date):
        return "invalid", None
    normalized = reviewed.isoformat()
    if reviewed > today:
        return "future", normalized
    if today - reviewed > timedelta(days=max_age_days):
        return "stale", normalized
    return "current", normalized
