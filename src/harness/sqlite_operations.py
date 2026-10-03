"""Read-only operational inspection of the persistent SQLite database."""

import sqlite3
from pathlib import Path
from urllib.parse import quote

from .database import Database


def inspect_sqlite(path: Path) -> dict:
    """Report bounded health data without creating or migrating a database."""
    report = {
        "status": "missing",
        "schema_version": None,
        "expected_schema_version": Database.CURRENT_SCHEMA_VERSION,
        "journal_mode": None,
        "integrity": False,
        "foreign_keys": False,
        "version_history_complete": False,
        "findings": [],
    }
    path = Path(path)
    if path.is_symlink():
        report.update(status="unhealthy", findings=["database_symlink"])
        return report
    if not path.is_file():
        return report
    connection = None
    try:
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.execute("PRAGMA query_only=ON")
        report["journal_mode"] = connection.execute("PRAGMA journal_mode").fetchone()[0]
        report["integrity"] = (
            connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        )
        report["foreign_keys"] = (
            connection.execute("PRAGMA foreign_key_check").fetchone() is None
        )
        rows = connection.execute(
            "SELECT version FROM schema_versions ORDER BY version"
        ).fetchall()
        versions = [row[0] for row in rows]
        report["schema_version"] = versions[-1] if versions else None
        report["version_history_complete"] = versions == list(
            range(1, Database.CURRENT_SCHEMA_VERSION + 1)
        )
        for ok, finding in (
            (report["integrity"], "integrity_check_failed"),
            (report["foreign_keys"], "foreign_key_violation"),
            (report["version_history_complete"], "schema_history_invalid"),
        ):
            if not ok:
                report["findings"].append(finding)
        report["status"] = "available" if not report["findings"] else "unhealthy"
    except (OSError, sqlite3.Error, IndexError, TypeError, ValueError):
        report.update(status="unavailable", findings=["database_probe_failed"])
    finally:
        if connection is not None:
            connection.close()
    return report
