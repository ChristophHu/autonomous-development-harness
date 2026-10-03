"""Versioned SQLite persistence and repository abstractions."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar


class Database:
    CURRENT_SCHEMA_VERSION: ClassVar[int] = 17

    def __init__(self, path: Path, *, timeout: float = 5.0):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            raise ValueError("SQLite timeout must be a non-negative number")
        self.path = path
        self.timeout = float(timeout)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=self.timeout)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def migrate(self):
        with self.connect() as c:
            c.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS subtasks(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, external_id TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL, profile TEXT NOT NULL, status TEXT NOT NULL, output TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plans(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, summary TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL, decision TEXT NOT NULL, rationale TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS agent_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER REFERENCES tasks(id), agent TEXT NOT NULL, profile TEXT NOT NULL, status TEXT NOT NULL, output TEXT, started_at TEXT NOT NULL, finished_at TEXT);
            CREATE TABLE IF NOT EXISTS model_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, agent_run_id INTEGER REFERENCES agent_runs(id), provider TEXT NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, question TEXT NOT NULL, reason TEXT NOT NULL, options TEXT NOT NULL DEFAULT '[]', required INTEGER NOT NULL DEFAULT 1, answer TEXT, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, answered_at TEXT);
            CREATE TABLE IF NOT EXISTS validations(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, valid INTEGER NOT NULL, report TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tool_calls(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, tool TEXT NOT NULL, status TEXT NOT NULL, input TEXT NOT NULL, output TEXT, started_at TEXT NOT NULL, finished_at TEXT);
            """)
            columns = {r[1] for r in c.execute("PRAGMA table_info(tasks)")}
            for name in ("created_at", "updated_at"):
                if name not in columns:
                    c.execute(
                        f"ALTER TABLE tasks ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                    )
            self._validate_migration_history(c)
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=1"
            ).fetchone():
                c.execute("INSERT INTO schema_versions VALUES(1,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=2"
            ).fetchone():
                c.execute(
                    "ALTER TABLE tasks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
                )
                c.execute(
                    "ALTER TABLE subtasks ADD COLUMN plan_id INTEGER REFERENCES plans(id)"
                )
                c.execute(
                    "CREATE TABLE task_leases(task_id INTEGER PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,owner TEXT NOT NULL,expires_at REAL NOT NULL)"
                )
                c.execute("INSERT INTO schema_versions VALUES(2,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=3"
            ).fetchone():
                for column, kind in (
                    ("call_id", "TEXT"),
                    ("agent_run_id", "INTEGER REFERENCES agent_runs(id)"),
                    ("profile", "TEXT"),
                    ("risk", "TEXT"),
                ):
                    c.execute(f"ALTER TABLE tool_calls ADD COLUMN {column} {kind}")
                c.execute(
                    "CREATE UNIQUE INDEX tool_calls_identity ON tool_calls(call_id)"
                )
                for column, kind in (
                    ("task_id", "INTEGER REFERENCES tasks(id)"),
                    ("agent", "TEXT"),
                    ("profile", "TEXT"),
                    ("complexity", "TEXT"),
                    ("started_at", "TEXT"),
                    ("finished_at", "TEXT"),
                    ("status", "TEXT"),
                    ("latency_ms", "REAL"),
                    ("cached_tokens", "INTEGER"),
                    ("reasoning_tokens", "INTEGER"),
                    ("fallback_index", "INTEGER"),
                    ("error_type", "TEXT"),
                ):
                    c.execute(f"ALTER TABLE model_runs ADD COLUMN {column} {kind}")
                c.execute("INSERT INTO schema_versions VALUES(3,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=4"
            ).fetchone():
                for column, kind in (
                    ("category", "TEXT NOT NULL DEFAULT 'legacy'"),
                    ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
                    ("evidence", "TEXT NOT NULL DEFAULT '[]'"),
                    ("field_names", "TEXT NOT NULL DEFAULT '[]'"),
                    (
                        "question_id",
                        "INTEGER REFERENCES questions(id) ON DELETE SET NULL",
                    ),
                ):
                    c.execute(f"ALTER TABLE decisions ADD COLUMN {column} {kind}")
                c.execute("INSERT INTO schema_versions VALUES(4,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=5"
            ).fetchone():
                c.execute(
                    """CREATE TABLE IF NOT EXISTS correction_items(
                        id TEXT PRIMARY KEY CHECK(length(id)=64),
                        task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                        plan_id INTEGER REFERENCES plans(id) ON DELETE SET NULL,
                        subtask_id TEXT,
                        category TEXT NOT NULL,
                        source TEXT NOT NULL,
                        rule TEXT NOT NULL,
                        message TEXT NOT NULL,
                        affected_paths TEXT NOT NULL,
                        evidence TEXT NOT NULL,
                        expected TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open'
                            CHECK(status IN ('open','in_progress','resolved')),
                        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        resolved_at TEXT,
                        UNIQUE(task_id,id)
                    )"""
                )
                c.execute(
                    "CREATE INDEX IF NOT EXISTS correction_items_task_status "
                    "ON correction_items(task_id,status,created_at,id)"
                )
                c.execute("INSERT INTO schema_versions VALUES(5,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=6"
            ).fetchone():
                for column, kind in (
                    ("alternatives", "TEXT NOT NULL DEFAULT '[]'"),
                    ("outcome", "TEXT"),
                    ("tags", "TEXT NOT NULL DEFAULT '[]'"),
                ):
                    c.execute(f"ALTER TABLE decisions ADD COLUMN {column} {kind}")
                c.execute("INSERT INTO schema_versions VALUES(6,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=7"
            ).fetchone():
                c.execute(
                    """CREATE TABLE artifacts(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                        artifact_key TEXT NOT NULL,
                        version INTEGER NOT NULL CHECK(version > 0),
                        content TEXT NOT NULL,
                        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
                        created_at TEXT NOT NULL,
                        UNIQUE(task_id,artifact_key,version)
                    )"""
                )
                c.execute(
                    "CREATE INDEX artifacts_task_key ON artifacts(task_id,artifact_key,version)"
                )
                c.execute("INSERT INTO schema_versions VALUES(7,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=8"
            ).fetchone():
                c.execute(
                    "ALTER TABLE decisions ADD COLUMN supersedes_id INTEGER REFERENCES decisions(id)"
                )
                c.execute(
                    "CREATE UNIQUE INDEX decisions_supersedes ON decisions(supersedes_id) WHERE supersedes_id IS NOT NULL"
                )
                c.execute("INSERT INTO schema_versions VALUES(8,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=9"
            ).fetchone():
                c.execute(
                    """CREATE TABLE verification_evidence(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL CHECK(kind IN ('ci','provider','qdrant','embedding','http_tls')),
                        source_id TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        subject_sha256 TEXT NOT NULL CHECK(length(subject_sha256)=64),
                        passed INTEGER NOT NULL CHECK(passed IN (0,1)),
                        checks_json TEXT NOT NULL,
                        digest TEXT NOT NULL UNIQUE CHECK(length(digest)=64)
                    )"""
                )
                c.execute(
                    "CREATE INDEX verification_evidence_kind_time ON verification_evidence(kind,observed_at DESC)"
                )
                c.execute(
                    "CREATE TRIGGER verification_evidence_no_update BEFORE UPDATE ON verification_evidence BEGIN SELECT RAISE(ABORT,'verification evidence is append-only'); END"
                )
                c.execute(
                    "CREATE TRIGGER verification_evidence_no_delete BEFORE DELETE ON verification_evidence BEGIN SELECT RAISE(ABORT,'verification evidence is append-only'); END"
                )
                c.execute("INSERT INTO schema_versions VALUES(9,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=10"
            ).fetchone():
                c.execute("""CREATE TABLE model_discovery_snapshots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL,
                    observed_at TEXT NOT NULL, status TEXT NOT NULL,
                    discovery_failed INTEGER NOT NULL CHECK(discovery_failed IN (0,1)),
                    models_json TEXT NOT NULL)""")
                c.execute(
                    "CREATE INDEX model_discovery_latest ON model_discovery_snapshots(provider,id DESC)"
                )
                c.execute("""CREATE TABLE qdrant_probe_snapshots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL,
                    healthy INTEGER NOT NULL CHECK(healthy IN (0,1)), status TEXT NOT NULL,
                    latency_ms REAL NOT NULL CHECK(latency_ms >= 0),
                    collection_exists INTEGER NOT NULL CHECK(collection_exists IN (0,1)),
                    vector_size INTEGER, points INTEGER, error_category TEXT)""")
                c.execute("INSERT INTO schema_versions VALUES(10,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=11"
            ).fetchone():
                c.execute("""CREATE TABLE memory_health_snapshots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL,
                    healthy INTEGER NOT NULL CHECK(healthy IN (0,1)),
                    checks_json TEXT NOT NULL)""")
                c.execute(
                    "CREATE INDEX memory_health_latest ON memory_health_snapshots(id DESC)"
                )
                c.execute("INSERT INTO schema_versions VALUES(11,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=12"
            ).fetchone():
                c.execute("""CREATE TABLE memory_ops_evidence(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    subject_sha256 TEXT NOT NULL CHECK(length(subject_sha256)=64),
                    passed INTEGER NOT NULL CHECK(passed=1),
                    checks_json TEXT NOT NULL,
                    digest TEXT NOT NULL UNIQUE CHECK(length(digest)=64))""")
                c.execute(
                    "CREATE INDEX memory_ops_evidence_latest ON memory_ops_evidence(id DESC)"
                )
                c.execute(
                    "CREATE TRIGGER memory_ops_evidence_no_update BEFORE UPDATE ON memory_ops_evidence BEGIN SELECT RAISE(ABORT,'memory ops evidence is append-only'); END"
                )
                c.execute(
                    "CREATE TRIGGER memory_ops_evidence_no_delete BEFORE DELETE ON memory_ops_evidence BEGIN SELECT RAISE(ABORT,'memory ops evidence is append-only'); END"
                )
                c.execute("INSERT INTO schema_versions VALUES(12,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=13"
            ).fetchone():
                c.execute(
                    "ALTER TABLE memory_health_snapshots ADD COLUMN source_sha256 TEXT"
                )
                c.execute("INSERT INTO schema_versions VALUES(13,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=14"
            ).fetchone():
                c.execute(
                    "ALTER TABLE questions ADD COLUMN purpose TEXT NOT NULL DEFAULT 'input' CHECK(purpose IN ('input','decision','approval'))"
                )
                c.execute("INSERT INTO schema_versions VALUES(14,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=15"
            ).fetchone():
                c.execute(
                    "ALTER TABLE model_runs ADD COLUMN error_category TEXT "
                    "CHECK(error_category IS NULL OR error_category IN "
                    "('cancellation','execution','permission','persistence',"
                    "'provider','timeout','transport','validation'))"
                )
                c.execute("INSERT INTO schema_versions VALUES(15,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=16"
            ).fetchone():
                c.execute("""CREATE TABLE mcp_status_snapshots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('available','unavailable','invalid_config','disabled','not_started')),
                    error_type TEXT,
                    CHECK(error_type IS NULL OR error_type GLOB '[A-Za-z]*'))""")
                c.execute(
                    "CREATE INDEX mcp_status_latest ON mcp_status_snapshots(server,id DESC)"
                )
                c.execute("INSERT INTO schema_versions VALUES(16,?)", (self.now(),))
            if not c.execute(
                "SELECT 1 FROM schema_versions WHERE version=17"
            ).fetchone():
                c.execute("""CREATE TRIGGER mcp_status_no_update
                    BEFORE UPDATE ON mcp_status_snapshots
                    BEGIN SELECT RAISE(ABORT,'MCP status snapshots are append-only'); END""")
                c.execute("""CREATE TRIGGER mcp_status_no_delete
                    BEFORE DELETE ON mcp_status_snapshots
                    BEGIN SELECT RAISE(ABORT,'MCP status snapshots are append-only'); END""")
                c.execute("INSERT INTO schema_versions VALUES(17,?)", (self.now(),))

    @classmethod
    def _validate_migration_history(cls, connection):
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            )
        ]
        if versions and versions[-1] > cls.CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {versions[-1]} is newer than this Harness "
                f"(maximum supported: {cls.CURRENT_SCHEMA_VERSION})"
            )
        if versions and versions != list(range(1, versions[-1] + 1)):
            raise RuntimeError("Database migration history is incomplete")

    @staticmethod
    def now():
        return datetime.now(UTC).isoformat()


class OperationalSnapshotRepository:
    """Persist bounded model discovery, Qdrant probes, and memory health."""

    def __init__(self, database):
        self.database = database

    def record_mcp_status(self, reports):
        """Append per-server status without retaining exception messages or secrets."""
        allowed = {
            "available",
            "unavailable",
            "invalid_config",
            "disabled",
            "not_started",
        }
        if not isinstance(reports, list):
            raise TypeError("MCP status reports must be a list")
        rows = []
        for report in reports:
            if not isinstance(report, dict):
                raise TypeError("MCP status report is invalid")
            name, state = report.get("name"), report.get("state")
            error_type = report.get("error")
            if state == "invalid_config":
                error_type = None
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 128
                or state not in allowed
                or (
                    error_type is not None
                    and (
                        not isinstance(error_type, str)
                        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", error_type)
                    )
                )
            ):
                raise ValueError("MCP status report is invalid")
            rows.append((name, Database.now(), state, error_type))
        with self.database.connect() as connection:
            connection.executemany(
                "INSERT INTO mcp_status_snapshots(server,observed_at,state,error_type) VALUES(?,?,?,?)",
                rows,
            )

    def latest_mcp_statuses(self):
        with self.database.connect() as connection:
            rows = connection.execute("""SELECT snapshot.server,snapshot.observed_at,
                snapshot.state,snapshot.error_type FROM mcp_status_snapshots AS snapshot
                JOIN (SELECT server,MAX(id) AS id FROM mcp_status_snapshots GROUP BY server) AS latest
                ON snapshot.id=latest.id ORDER BY snapshot.server""").fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def read_latest_mcp_statuses(path):
        """Read snapshots without creating a database or applying migrations."""
        from urllib.parse import quote

        database_path = Path(path)
        if not database_path.is_file():
            return []
        uri = f"file:{quote(str(database_path), safe='/')}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=1)
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute("""SELECT snapshot.server,snapshot.observed_at,
                    snapshot.state,snapshot.error_type FROM mcp_status_snapshots AS snapshot
                    JOIN (SELECT server,MAX(id) AS id FROM mcp_status_snapshots GROUP BY server) AS latest
                    ON snapshot.id=latest.id ORDER BY snapshot.server""").fetchall()
                return [dict(row) for row in rows]
            finally:
                connection.close()
        except sqlite3.OperationalError:
            return []

    def record_model_discovery(self, provider, status, discovery_failed, models):
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO model_discovery_snapshots(provider,observed_at,status,discovery_failed,models_json) VALUES(?,?,?,?,?)",
                (
                    provider,
                    Database.now(),
                    str(status),
                    int(discovery_failed),
                    json.dumps(sorted(set(models))),
                ),
            )

    def latest_model_discovery(self, provider):
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_discovery_snapshots WHERE provider=? ORDER BY id DESC LIMIT 1",
                (provider,),
            ).fetchone()
        return (
            None
            if row is None
            else {
                "status": row["status"],
                "discovery_failed": bool(row["discovery_failed"]),
                "models": json.loads(row["models_json"]),
                "observed_at": row["observed_at"],
            }
        )

    def record_qdrant_probe(self, report, latency_ms):
        errors = report.get("errors", [])
        category = str(errors[0])[:64] if errors else None
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO qdrant_probe_snapshots(observed_at,healthy,status,latency_ms,collection_exists,vector_size,points,error_category) VALUES(?,?,?,?,?,?,?,?)",
                (
                    Database.now(),
                    int(bool(report.get("healthy"))),
                    str(report.get("service", "unknown"))[:32],
                    max(0.0, float(latency_ms)),
                    int(bool(report.get("collection_exists"))),
                    report.get("dimension"),
                    report.get("points_count"),
                    category,
                ),
            )

    def record_memory_health(self, report):
        if (
            not isinstance(report, dict)
            or not isinstance(report.get("healthy"), bool)
            or not isinstance(report.get("checks"), dict)
            or not report["checks"]
            or any(
                not isinstance(name, str) or not isinstance(state, str)
                for name, state in report["checks"].items()
            )
        ):
            raise ValueError("memory health report is invalid")
        source_sha256 = report.get("source_sha256")
        if source_sha256 is not None and (
            not isinstance(source_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", source_sha256)
        ):
            raise ValueError("memory health source SHA-256 is invalid")
        with self.database.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_health_snapshots(observed_at,healthy,checks_json,source_sha256) VALUES(?,?,?,?)",
                (
                    Database.now(),
                    int(report["healthy"]),
                    json.dumps(report["checks"], sort_keys=True),
                    source_sha256,
                ),
            )
        return cursor.lastrowid

    def latest_memory_health(self):
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_health_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "observed_at": row["observed_at"],
            "healthy": bool(row["healthy"]),
            "checks": json.loads(row["checks_json"]),
            "source_sha256": row["source_sha256"],
        }

    def recent_memory_health(self, *, limit):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            raise ValueError("memory snapshot limit must be between 1 and 500")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_health_snapshots ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "observed_at": row["observed_at"],
                "healthy": bool(row["healthy"]),
                "checks": json.loads(row["checks_json"]),
                "source_sha256": row["source_sha256"],
            }
            for row in rows
        ]


class MemoryOpsEvidenceRepository:
    """Store successful, append-only evidence for an operational memory run."""

    REQUIRED_CHECKS: ClassVar[set[str]] = {
        "launchd_loaded",
        "snapshot_count_sufficient",
        "observation_window_met",
        "snapshots_healthy",
        "required_checks_present",
        "latest_snapshot_fresh",
        "monitor_source_current",
    }

    def __init__(self, database):
        self.database = database

    def record(self, *, subject_sha256, observed_at, checks):
        import hashlib
        import re

        if not isinstance(subject_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", subject_sha256
        ):
            raise ValueError("memory ops evidence subject SHA-256 is invalid")
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            raise ValueError("memory ops evidence timestamp must be timezone-aware")
        if (
            not isinstance(checks, dict)
            or set(checks) != self.REQUIRED_CHECKS
            or any(type(value) is not bool for value in checks.values())
        ):
            raise ValueError("memory ops evidence checks are invalid")
        if not all(checks.values()):
            raise ValueError("memory ops evidence requires all checks to pass")
        normalized_at = observed_at.astimezone(UTC).isoformat()
        canonical = json.dumps(
            {
                "observed_at": normalized_at,
                "subject_sha256": subject_sha256,
                "passed": True,
                "checks": checks,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.database.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_ops_evidence(observed_at,subject_sha256,passed,checks_json,digest) VALUES(?,?,?,?,?)",
                (
                    normalized_at,
                    subject_sha256,
                    1,
                    json.dumps(checks, sort_keys=True),
                    digest,
                ),
            )
        return {
            "id": cursor.lastrowid,
            "observed_at": normalized_at,
            "subject_sha256": subject_sha256,
            "passed": True,
            "checks": checks,
            "digest": digest,
        }

    def list(self, *, limit=100):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            raise ValueError("memory ops evidence limit must be between 1 and 500")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_ops_evidence ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "observed_at": row["observed_at"],
                "subject_sha256": row["subject_sha256"],
                "passed": bool(row["passed"]),
                "checks": json.loads(row["checks_json"]),
                "digest": row["digest"],
            }
            for row in rows
        ]


class TaskRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, title, description="", status="pending", metadata=None):
        now = self.db.now()
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO tasks(title,description,status,created_at,updated_at,metadata) VALUES(?,?,?,?,?,?)",
                (title, description, status, now, now, json.dumps(metadata or {})),
            )
            return cur.lastrowid

    def get(self, task_id):
        with self.db.connect() as c:
            return c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()

    def update(self, task_id, **fields):
        if not set(fields) <= {"title", "description", "status", "result", "metadata"}:
            raise ValueError("unknown task repository field")
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        fields["updated_at"] = self.db.now()
        sql = ", ".join(f"{k}=?" for k in fields)
        with self.db.connect() as c:
            c.execute(f"UPDATE tasks SET {sql} WHERE id=?", (*fields.values(), task_id))

    def update_if_idle(self, task_id, title, description, metadata):
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM task_leases WHERE task_id=? AND expires_at>strftime('%s','now')",
                (task_id,),
            ).fetchone():
                return False
            return (
                connection.execute(
                    "UPDATE tasks SET title=?,description=?,metadata=?,updated_at=? WHERE id=?",
                    (title, description, json.dumps(metadata), self.db.now(), task_id),
                ).rowcount
                == 1
            )

    def delete_if_idle(self, task_id):
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM task_leases WHERE task_id=? AND expires_at>strftime('%s','now')",
                (task_id,),
            ).fetchone():
                return False
            connection.execute(
                "DELETE FROM model_runs WHERE agent_run_id IN (SELECT id FROM agent_runs WHERE task_id=?)",
                (task_id,),
            )
            connection.execute("DELETE FROM agent_runs WHERE task_id=?", (task_id,))
            connection.execute("DELETE FROM tool_calls WHERE task_id=?", (task_id,))
            return (
                connection.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount
                == 1
            )

    def list(self, status=None):
        with self.db.connect() as c:
            return c.execute(
                "SELECT * FROM tasks WHERE status IS ? OR ? IS NULL ORDER BY id",
                (status, status),
            ).fetchall()

    def claim(self, task_id, owner, ttl=300, exclusive=False):
        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "DELETE FROM task_leases WHERE task_id=? AND expires_at<=?",
                (task_id, time.time()),
            )
            row = c.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if not row or row["status"] in {"completed", "cancelled"}:
                return False
            if (
                exclusive
                and c.execute(
                    "SELECT 1 FROM task_leases WHERE task_id<>? AND expires_at>? LIMIT 1",
                    (task_id, time.time()),
                ).fetchone()
            ):
                return False
            if c.execute(
                "SELECT 1 FROM questions WHERE task_id=? AND required=1 AND status='open'",
                (task_id,),
            ).fetchone():
                raise ValueError("required question is still open")
            changed = c.execute(
                "INSERT OR IGNORE INTO task_leases VALUES(?,?,?)",
                (task_id, owner, time.time() + ttl),
            ).rowcount
            return changed == 1

    def renew(self, task_id, owner, ttl=300):
        with self.db.connect() as c:
            return (
                c.execute(
                    "UPDATE task_leases SET expires_at=? WHERE task_id=? AND owner=? AND expires_at>?",
                    (time.time() + ttl, task_id, owner, time.time()),
                ).rowcount
                == 1
            )

    def release(self, task_id, owner):
        with self.db.connect() as c:
            return (
                c.execute(
                    "DELETE FROM task_leases WHERE task_id=? AND owner=?",
                    (task_id, owner),
                ).rowcount
                == 1
            )

    def transition(self, task_id, target, owner=None):
        from .domain import EventKind, may_transition

        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if not row:
                raise ValueError("task not found")
            if (
                owner
                and not c.execute(
                    "SELECT 1 FROM task_leases WHERE task_id=? AND owner=? AND expires_at>?",
                    (task_id, owner, time.time()),
                ).fetchone()
            ):
                raise ValueError("task lease is no longer owned")
            if not may_transition(row["status"], target):
                raise ValueError(
                    f"invalid task transition: {row['status']} -> {target}"
                )
            if (
                target in {"executing", "completed"}
                and c.execute(
                    "SELECT 1 FROM questions WHERE task_id=? AND required=1 AND status='open'",
                    (task_id,),
                ).fetchone()
            ):
                raise ValueError("required question is still open")
            if (
                target == "completed"
                and c.execute(
                    "SELECT 1 FROM correction_items WHERE task_id=? AND status IN ('open','in_progress') LIMIT 1",
                    (task_id,),
                ).fetchone()
            ):
                raise ValueError("required correction is still open or unverified")
            now = self.db.now()
            c.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                (str(target), now, task_id),
            )
            c.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (
                    task_id,
                    EventKind.TASK_STATUS.value,
                    json.dumps({"from": row["status"], "status": str(target)}),
                    now,
                ),
            )


class EventRepository:
    def __init__(self, db):
        self.db = db

    def append(self, task_id, kind, payload):
        from .domain import EventKind

        event_kind = EventKind(kind)
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (task_id, event_kind.value, json.dumps(payload), self.db.now()),
            )
            return cur.lastrowid

    def list(self, task_id=None, event_type=None, since=None, until=None, actor=None):
        clauses = []
        args = []
        for column, value, operator in (
            ("task_id", task_id, "="),
            ("kind", event_type, "="),
            ("created_at", since, ">="),
            ("created_at", until, "<="),
        ):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                args.append(value)
        if actor is not None:
            clauses.append(
                "(json_extract(payload,'$.actor')=? OR "
                "json_extract(payload,'$.agent')=? OR json_extract(payload,'$.profile')=?)"
            )
            args.extend((actor, actor, actor))
        sql = (
            "SELECT * FROM events"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY id"
        )
        with self.db.connect() as c:
            return c.execute(sql, args).fetchall()

    def after(self, event_id=0, task_id=None):
        sql = "SELECT * FROM events WHERE id>?"
        args = [event_id]
        if task_id is not None:
            sql += " AND task_id=?"
            args.append(task_id)
        with self.db.connect() as c:
            return c.execute(sql + " ORDER BY id", args).fetchall()


class PlanRepository:
    def __init__(self, db):
        self.db = db

    def save(self, task_id, summary, payload):
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO plans(task_id,summary,payload,created_at) VALUES(?,?,?,?)",
                (task_id, summary, json.dumps(payload), self.db.now()),
            )
            return cur.lastrowid

    def latest(self, task_id):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT * FROM plans WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["payload"] = json.loads(result["payload"])
            return result


class ArtifactRepository:
    """Append-only, content-addressed task artifacts with optimistic versions."""

    def __init__(self, db):
        self.db = db

    def save(self, task_id, key, content, expected_version=None):
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 1:
            raise ValueError("task_id must be a positive integer")
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValueError(
                "artifact key must be a non-empty string of at most 200 characters"
            )
        if not isinstance(content, str):
            raise TypeError("artifact content must be a string")
        if expected_version is not None and (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 0
        ):
            raise ValueError("expected_version must be a non-negative integer or null")
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM tasks WHERE id=?", (task_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("task not found")
            row = connection.execute(
                "SELECT MAX(version) AS version FROM artifacts WHERE task_id=? AND artifact_key=?",
                (task_id, key),
            ).fetchone()
            current = row["version"] or 0
            if expected_version is not None and current != expected_version:
                raise ValueError("artifact version conflict")
            version = current + 1
            cursor = connection.execute(
                "INSERT INTO artifacts(task_id,artifact_key,version,content,sha256,created_at) VALUES(?,?,?,?,?,?)",
                (
                    task_id,
                    key,
                    version,
                    content,
                    hashlib.sha256(content.encode()).hexdigest(),
                    self.db.now(),
                ),
            )
            from .domain import EventKind

            connection.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (
                    task_id,
                    EventKind.ARTIFACT_RECORDED.value,
                    json.dumps(
                        {
                            "key": key,
                            "version": version,
                            "sha256": hashlib.sha256(content.encode()).hexdigest(),
                        }
                    ),
                    self.db.now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id=?", (cursor.lastrowid,)
            ).fetchone()
            return dict(row)

    def latest(self, task_id, key):
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE task_id=? AND artifact_key=? ORDER BY version DESC LIMIT 1",
                (task_id, key),
            ).fetchone()
            return dict(row) if row else None

    def history(self, task_id, key):
        with self.db.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM artifacts WHERE task_id=? AND artifact_key=? ORDER BY version",
                    (task_id, key),
                )
            ]

    def latest_for_task(self, task_id):
        with self.db.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT a.* FROM artifacts a JOIN (SELECT artifact_key,MAX(version) version FROM artifacts WHERE task_id=? GROUP BY artifact_key) latest ON latest.artifact_key=a.artifact_key AND latest.version=a.version WHERE a.task_id=? ORDER BY a.artifact_key",
                    (task_id, task_id),
                )
            ]


class ValidationRepository:
    def __init__(self, db):
        self.db = db

    def record(self, task_id, valid, report):
        serialized = report if isinstance(report, str) else json.dumps(report)
        with self.db.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO validations(task_id,valid,report,created_at) VALUES(?,?,?,?)",
                (task_id, int(valid), serialized, self.db.now()),
            )
            return cursor.lastrowid

    def latest(self, task_id):
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM validations WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["valid"] = bool(result["valid"])
        result["report"] = json.loads(result["report"])
        return result


class CorrectionRepository:
    """Persist idempotent, task-scoped correction findings and transitions."""

    STATUSES: ClassVar[set[str]] = {"open", "in_progress", "resolved"}
    TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "open": {"in_progress", "resolved"},
        "in_progress": {"open", "resolved"},
        "resolved": {"open"},
    }
    REQUIRED_FIELDS: ClassVar[set[str]] = {"category", "source", "rule", "message"}
    OPTIONAL_FIELDS: ClassVar[set[str]] = {
        "subtask_id",
        "affected_paths",
        "evidence",
        "expected",
    }

    def __init__(self, db):
        self.db = db

    @staticmethod
    def _task_id(task_id):
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 1:
            raise ValueError("task_id must be a positive integer")

    @classmethod
    def _normalize_finding(cls, finding):
        if not isinstance(finding, dict) or not cls.REQUIRED_FIELDS <= set(finding):
            raise ValueError("finding fields are invalid")
        if not set(finding) <= cls.REQUIRED_FIELDS | cls.OPTIONAL_FIELDS:
            raise ValueError("finding fields are invalid")
        normalized = {}
        for name in ("category", "source", "rule", "message"):
            value = finding[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"finding {name} must be a nonempty string")
            normalized[name] = value.strip()
        subtask_id = finding.get("subtask_id")
        if subtask_id is not None and (
            not isinstance(subtask_id, str) or not subtask_id.strip()
        ):
            raise ValueError("finding subtask_id must be a nonempty string or null")
        normalized["subtask_id"] = subtask_id.strip() if subtask_id else None
        paths = finding.get("affected_paths", [])
        if (
            not isinstance(paths, list)
            or any(not isinstance(path, str) or not path.strip() for path in paths)
            or len([path.strip() for path in paths])
            != len({path.strip() for path in paths})
        ):
            raise ValueError("affected_paths must contain unique nonempty strings")
        normalized["affected_paths"] = sorted(path.strip() for path in paths)
        for name in ("evidence", "expected"):
            value = finding.get(name, {})
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a mapping")
            try:
                json.dumps(value, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be JSON serializable") from exc
            normalized[name] = value
        return normalized

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        for name in ("affected_paths", "evidence", "expected"):
            result[name] = json.loads(result[name])
        return result

    @staticmethod
    def _stable_id(task_id, finding):
        identity = {
            "task_id": task_id,
            "category": finding["category"],
            "source": finding["source"],
            "rule": finding["rule"],
            "subtask_id": finding["subtask_id"],
            "affected_paths": sorted(finding["affected_paths"]),
        }
        serialized = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _append_event(connection, task_id, kind, payload, now):
        from .domain import EventKind

        connection.execute(
            "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
            (task_id, EventKind(kind).value, json.dumps(payload), now),
        )

    def record(self, task_id, finding, plan_id=None):
        """Create or refresh a finding atomically; repeated reports keep one ID."""
        self._task_id(task_id)
        if plan_id is not None and (
            not isinstance(plan_id, int) or isinstance(plan_id, bool) or plan_id < 1
        ):
            raise ValueError("plan_id must be a positive integer or null")
        finding = self._normalize_finding(finding)
        item_id = self._stable_id(task_id, finding)
        now = self.db.now()
        values = (
            plan_id,
            finding["subtask_id"],
            finding["category"],
            finding["source"],
            finding["rule"],
            finding["message"],
            json.dumps(finding["affected_paths"]),
            json.dumps(finding["evidence"], sort_keys=True, allow_nan=False),
            json.dumps(finding["expected"], sort_keys=True, allow_nan=False),
        )
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if task is None:
                raise ValueError("task not found")
            if task["status"] == "completed":
                raise ValueError("cannot record a correction for a completed task")
            if plan_id is not None:
                plan = connection.execute(
                    "SELECT task_id FROM plans WHERE id=?", (plan_id,)
                ).fetchone()
                if plan is None or plan["task_id"] != task_id:
                    raise ValueError("plan does not belong to task")
            existing = connection.execute(
                "SELECT * FROM correction_items WHERE id=?", (item_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO correction_items(
                        id,task_id,plan_id,subtask_id,category,source,rule,message,
                        affected_paths,evidence,expected,status,attempts,created_at,
                        updated_at,resolved_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'open',0,?,?,NULL)""",
                    (item_id, task_id, *values, now, now),
                )
                status = "open"
                changed = True
            else:
                status = (
                    "open" if existing["status"] == "resolved" else existing["status"]
                )
                changed = (
                    status != existing["status"]
                    or tuple(
                        existing[name]
                        for name in (
                            "plan_id",
                            "subtask_id",
                            "category",
                            "source",
                            "rule",
                            "message",
                            "affected_paths",
                            "evidence",
                            "expected",
                        )
                    )
                    != values
                )
                if changed:
                    connection.execute(
                        """UPDATE correction_items SET plan_id=?,subtask_id=?,
                            category=?,source=?,rule=?,message=?,affected_paths=?,
                            evidence=?,expected=?,status=?,updated_at=?,resolved_at=?
                            WHERE id=?""",
                        (
                            *values,
                            status,
                            now,
                            None if status == "open" else existing["resolved_at"],
                            item_id,
                        ),
                    )
            if changed:
                self._append_event(
                    connection,
                    task_id,
                    "correction.item_recorded",
                    {"item_id": item_id, "status": status, "source": finding["source"]},
                    now,
                )
            row = connection.execute(
                "SELECT * FROM correction_items WHERE id=?", (item_id,)
            ).fetchone()
            return self._decode(row)

    def get(self, item_id):
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM correction_items WHERE id=?", (item_id,)
            ).fetchone()
        return self._decode(row)

    def list_for_task(self, task_id, status=None):
        self._task_id(task_id)
        if status is not None and (
            not isinstance(status, str) or status not in self.STATUSES
        ):
            raise ValueError("unsupported correction status")
        query = "SELECT * FROM correction_items WHERE task_id=?"
        parameters = [task_id]
        if status is not None:
            query += " AND status=?"
            parameters.append(status)
        query += " ORDER BY created_at,id"
        with self.db.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode(row) for row in rows]

    def set_status(self, item_id, status):
        if not isinstance(status, str) or status not in self.STATUSES:
            raise ValueError("unsupported correction status")
        now = self.db.now()
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM correction_items WHERE id=?", (item_id,)
            ).fetchone()
            if row is None:
                return None
            current = row["status"]
            if current == status:
                return self._decode(row)
            if status not in self.TRANSITIONS.get(current, set()):
                raise ValueError("invalid correction status transition")
            attempts = row["attempts"] + (status == "in_progress")
            resolved_at = now if status == "resolved" else None
            connection.execute(
                "UPDATE correction_items SET status=?,attempts=?,updated_at=?,resolved_at=? WHERE id=?",
                (status, attempts, now, resolved_at, item_id),
            )
            self._append_event(
                connection,
                row["task_id"],
                "correction.item_status",
                {"item_id": item_id, "from": current, "status": status},
                now,
            )
            updated = connection.execute(
                "SELECT * FROM correction_items WHERE id=?", (item_id,)
            ).fetchone()
            return self._decode(updated)


class SubtaskRepository:
    """Versioned execution artifacts for auditing; never replay checkpoints."""

    def __init__(self, db: Database):
        self.db = db

    def save_plan(self, task_id, subtasks, plan_id=None):
        with self.db.connect() as c:
            if plan_id is None:
                c.execute(
                    "DELETE FROM subtasks WHERE task_id=? AND plan_id IS NULL",
                    (task_id,),
                )
            for subtask in subtasks:
                c.execute(
                    "INSERT INTO subtasks(task_id,external_id,title,description,profile,status,plan_id) VALUES(?,?,?,?,?,'pending',?)",
                    (
                        task_id,
                        subtask.id,
                        subtask.title,
                        subtask.description,
                        subtask.profile,
                        plan_id,
                    ),
                )

    def update(self, task_id, external_id, status, output=None, plan_id=None):
        with self.db.connect() as c:
            return (
                c.execute(
                    "UPDATE subtasks SET status=?,output=? WHERE task_id=? AND external_id=? AND plan_id IS ?",
                    (status, output, task_id, external_id, plan_id),
                ).rowcount
                == 1
            )

    def list(self, task_id):
        with self.db.connect() as c:
            return c.execute(
                "SELECT * FROM subtasks WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()


class DecisionRepository:
    def __init__(self, db):
        self.db = db

    def save(self, task_id, decision, rationale):
        return self.record(
            task_id,
            "legacy",
            "legacy",
            decision,
            rationale,
            (),
            (),
            None,
        )

    def record(
        self,
        task_id,
        category,
        source,
        decision,
        rationale,
        evidence,
        field_names,
        question_id,
        alternatives=(),
        outcome=None,
        tags=(),
        supersedes_id=None,
    ):
        from .domain import EventKind

        event_kind = EventKind.DECISION_RECORDED
        now = self.db.now()
        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            cur = c.execute(
                "INSERT INTO decisions(task_id,decision,rationale,created_at,category,source,evidence,field_names,question_id,alternatives,outcome,tags,supersedes_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    decision,
                    rationale,
                    now,
                    category,
                    source,
                    json.dumps(evidence),
                    json.dumps(field_names),
                    question_id,
                    json.dumps(alternatives),
                    outcome,
                    json.dumps(tags),
                    supersedes_id,
                ),
            )
            decision_id = cur.lastrowid
            c.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (
                    task_id,
                    event_kind.value,
                    json.dumps(
                        {
                            "decision_id": decision_id,
                            "category": category,
                            "source": source,
                            "question_id": question_id,
                            "field_names": field_names,
                            "alternatives": alternatives,
                            "outcome": outcome,
                            "tags": tags,
                            "evidence_refs": [item["ref"] for item in evidence],
                            **(
                                {"supersedes_id": supersedes_id}
                                if supersedes_id is not None
                                else {}
                            ),
                        }
                    ),
                    now,
                ),
            )
            return decision_id

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        result["evidence"] = json.loads(result["evidence"])
        result["field_names"] = json.loads(result["field_names"])
        result["alternatives"] = json.loads(result["alternatives"])
        result["tags"] = json.loads(result["tags"])
        return result

    def get(self, decision_id):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT * FROM decisions WHERE id=?", (decision_id,)
            ).fetchone()
        return self._decode(row)

    def list(self, task_id):
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM decisions WHERE task_id IS ? OR task_id IS NULL ORDER BY id",
                (task_id,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def list_all(self):
        with self.db.connect() as c:
            rows = c.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        return [self._decode(row) for row in rows]


class QuestionRepository:
    def __init__(self, db):
        self.db = db

    def create(self, task_id, question, reason, options=None, required=True):
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO questions(task_id,question,reason,options,required,created_at) VALUES(?,?,?,?,?,?)",
                (
                    task_id,
                    question,
                    reason,
                    json.dumps(options or []),
                    int(required),
                    self.db.now(),
                ),
            )
            return cur.lastrowid

    def ask(
        self,
        task_id,
        question,
        reason,
        options=None,
        required=True,
        event_payload=None,
        purpose="input",
    ):
        from .domain import EventKind

        status_by_purpose = {
            "input": "waiting_human",
            "decision": "waiting_decision",
            "approval": "waiting_approval",
        }
        if purpose not in status_by_purpose:
            raise ValueError("unsupported question purpose")

        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if task is None:
                raise ValueError("task not found")
            if required and task["status"] in {"completed", "cancelled"}:
                raise ValueError("terminal task cannot be reopened by a question")
            existing = connection.execute(
                "SELECT id FROM questions WHERE task_id=? AND question=? AND reason=? AND status='open'",
                (task_id, question, reason),
            ).fetchone()
            if existing is not None:
                return existing["id"]
            question_id = connection.execute(
                "INSERT INTO questions(task_id,question,reason,options,required,created_at,purpose) VALUES(?,?,?,?,?,?,?)",
                (
                    task_id,
                    question,
                    reason,
                    json.dumps(options or []),
                    int(required),
                    self.db.now(),
                    purpose,
                ),
            ).lastrowid
            if required:
                status = status_by_purpose[purpose]
                previous = task["status"]
                connection.execute(
                    "UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                    (status, self.db.now(), task_id),
                )
                if previous != status:
                    connection.execute(
                        "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                        (
                            task_id,
                            EventKind.TASK_STATUS.value,
                            json.dumps({"from": previous, "status": status}),
                            self.db.now(),
                        ),
                    )
            connection.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (
                    task_id,
                    EventKind.QUESTION_ASKED.value,
                    json.dumps({"question_id": question_id, **(event_payload or {})}),
                    self.db.now(),
                ),
            )
            return question_id

    def list(self, task_id):
        with self.db.connect() as c:
            return c.execute(
                "SELECT * FROM questions WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()

    def get(self, question_id):
        with self.db.connect() as c:
            return c.execute(
                "SELECT * FROM questions WHERE id=?", (question_id,)
            ).fetchone()

    def has_open_required(self, task_id):
        with self.db.connect() as c:
            return (
                c.execute(
                    "SELECT 1 FROM questions WHERE task_id=? AND status='open' AND required=1 LIMIT 1",
                    (task_id,),
                ).fetchone()
                is not None
            )

    def supersede_conflict_questions(self, task_id, keep_reasons=()):
        """Close obsolete, unanswered requirement-conflict questions atomically."""
        from .domain import EventKind

        keep = set(keep_reasons)
        superseded = []
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id,reason FROM questions WHERE task_id=? AND status='open' "
                "AND required=1 AND purpose='decision' AND reason LIKE 'requirements:conflict:%'",
                (task_id,),
            ).fetchall()
            for row in rows:
                if row["reason"] in keep:
                    continue
                connection.execute(
                    "UPDATE questions SET status='superseded' WHERE id=? AND status='open'",
                    (row["id"],),
                )
                connection.execute(
                    "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                    (
                        task_id,
                        EventKind.QUESTION_SUPERSEDED.value,
                        json.dumps(
                            {
                                "question_id": row["id"],
                                "reason_kind": "requirements:conflict",
                            }
                        ),
                        self.db.now(),
                    ),
                )
                superseded.append(row["id"])
        return superseded

    def answer(self, question_id, answer, task_id=None):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT options,task_id,required FROM questions WHERE id=? AND status='open'",
                (question_id,),
            ).fetchone()
            if not row or (task_id is not None and row["task_id"] != task_id):
                return False
            options = json.loads(row["options"])
            if options and answer not in options:
                return False
            changed = (
                c.execute(
                    "UPDATE questions SET answer=?,status='answered',answered_at=? WHERE id=? AND status='open'",
                    (answer, self.db.now(), question_id),
                ).rowcount
                == 1
            )
            if changed and row["required"]:
                pending = c.execute(
                    "SELECT purpose FROM questions WHERE task_id=? AND status='open' AND required=1 ORDER BY id DESC LIMIT 1",
                    (row["task_id"],),
                ).fetchone()
                if pending is not None:
                    status = {
                        "input": "waiting_human",
                        "decision": "waiting_decision",
                        "approval": "waiting_approval",
                    }[pending["purpose"]]
                    task = c.execute(
                        "SELECT status FROM tasks WHERE id=?", (row["task_id"],)
                    ).fetchone()
                    if task is not None and task["status"] != status:
                        c.execute(
                            "UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                            (status, self.db.now(), row["task_id"]),
                        )
                        c.execute(
                            "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                            (
                                row["task_id"],
                                "task.status",
                                json.dumps({"from": task["status"], "status": status}),
                                self.db.now(),
                            ),
                        )
            return changed

    def consume_answer(self, question_id):
        with self.db.connect() as c:
            return (
                c.execute(
                    "UPDATE questions SET status='consumed' WHERE id=? AND status='answered'",
                    (question_id,),
                ).rowcount
                == 1
            )

    def consume_approval(self, question_id, reason):
        with self.db.connect() as connection:
            return (
                connection.execute(
                    "UPDATE questions SET status='executed' WHERE id=? AND reason=? AND status='consumed' AND answer='approve'",
                    (question_id, reason),
                ).rowcount
                == 1
            )


class AgentRunRepository:
    def __init__(self, db):
        self.db = db

    def start(self, task_id, agent, profile):
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO agent_runs(task_id,agent,profile,status,started_at) VALUES(?,?,?,'running',?)",
                (task_id, agent, profile, self.db.now()),
            )
            return cur.lastrowid

    def finish(self, run_id, status, output):
        with self.db.connect() as c:
            return (
                c.execute(
                    "UPDATE agent_runs SET status=?,output=?,finished_at=? WHERE id=?",
                    (status, output, self.db.now(), run_id),
                ).rowcount
                == 1
            )


class ModelRunRepository:
    def __init__(self, db):
        self.db = db

    def record(self, usage, agent_run_id=None):
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO model_runs(agent_run_id,provider,model,prompt_tokens,completion_tokens,cost,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    agent_run_id,
                    usage.provider,
                    usage.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost,
                    self.db.now(),
                ),
            )
            return cur.lastrowid

    @staticmethod
    def _usage_totals(row):
        runs = row["runs"]
        totals = {"runs": runs}
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "cost",
        ):
            reported = row[f"{field}_reported"]
            totals[field] = row[field]
            totals[f"{field}_reported"] = reported
            totals[f"{field}_missing"] = runs - reported
        return totals

    def usage_report(
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
        grouping = {
            "task_id": "task_id",
            "agent": "agent",
            "profile": "profile",
            "provider": "provider",
            "model": "model",
            "day": "date(COALESCE(started_at,created_at))",
        }
        if group_by is not None and group_by not in grouping:
            raise ValueError("invalid group_by dimension")
        clauses, parameters = [], []
        for column, value in (
            ("task_id", task_id),
            ("agent", agent),
            ("profile", profile),
            ("provider", provider),
            ("model", model),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        if since is not None:
            clauses.append("COALESCE(started_at,created_at)>=?")
            parameters.append(since)
        if until is not None:
            clauses.append("COALESCE(started_at,created_at)<=?")
            parameters.append(until)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        aggregates = (
            "COUNT(*) AS runs, "
            "SUM(prompt_tokens) AS prompt_tokens, "
            "COUNT(prompt_tokens) AS prompt_tokens_reported, "
            "SUM(completion_tokens) AS completion_tokens, "
            "COUNT(completion_tokens) AS completion_tokens_reported, "
            "SUM(cached_tokens) AS cached_tokens, "
            "COUNT(cached_tokens) AS cached_tokens_reported, "
            "SUM(reasoning_tokens) AS reasoning_tokens, "
            "COUNT(reasoning_tokens) AS reasoning_tokens_reported, "
            "SUM(cost) AS cost, COUNT(cost) AS cost_reported"
        )
        with self.db.connect() as connection:
            totals_row = connection.execute(
                f"SELECT {aggregates} FROM model_runs{where}",
                parameters,
            ).fetchone()
            totals = self._usage_totals(totals_row)
            groups = []
            if group_by is not None:
                expression = grouping[group_by]
                groups = [
                    {"value": row["value"], "totals": self._usage_totals(row)}
                    for row in connection.execute(
                        f"SELECT {expression} AS value, {aggregates} "
                        f"FROM model_runs{where} GROUP BY {expression} ORDER BY value",
                        parameters,
                    ).fetchall()
                ]
            rows = connection.execute(
                "SELECT id,task_id,agent,profile,complexity,provider,model,status,"
                "started_at,finished_at,latency_ms,fallback_index,prompt_tokens,"
                "completion_tokens,cached_tokens,reasoning_tokens,cost,error_type,error_category "
                f"FROM model_runs{where} "
                "ORDER BY COALESCE(started_at,created_at) DESC,id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()
        return {
            "total": totals["runs"],
            "totals": totals,
            "items": [dict(row) for row in rows],
            "group_by": group_by,
            "groups": groups,
        }


class AuditRepository:
    """Atomic persistence of correlated tool events and model invocation spans."""

    def __init__(self, db):
        self.db = db

    def tool_event(self, task_id, kind, payload, context):
        from .domain import EventKind

        event_kind = EventKind(kind)
        if event_kind not in {
            EventKind.TOOL_CALL_STARTED,
            EventKind.TOOL_CALL_COMPLETED,
            EventKind.TOOL_CALL_FAILED,
        }:
            raise ValueError("unsupported tool event kind")
        kind = event_kind.value
        with self.db.connect() as connection:
            now = self.db.now()
            if kind == "TOOL_CALL_STARTED":
                connection.execute(
                    "INSERT INTO tool_calls(task_id,agent_run_id,profile,call_id,tool,risk,status,input,started_at) VALUES(?,?,?,?,?,?,'running',?,?)",
                    (
                        task_id,
                        context.get("agent_run_id"),
                        context.get("profile"),
                        payload["call_id"],
                        payload["tool"],
                        payload["risk"],
                        json.dumps(payload["input"]),
                        now,
                    ),
                )
            else:
                status = "completed" if kind == "TOOL_CALL_COMPLETED" else "failed"
                changed = connection.execute(
                    "UPDATE tool_calls SET status=?,output=?,finished_at=? WHERE call_id=? AND status='running'",
                    (
                        status,
                        json.dumps(payload.get("output", payload.get("error"))),
                        now,
                        payload["call_id"],
                    ),
                ).rowcount
                if changed != 1:
                    raise ValueError("tool completion has no active invocation")
            connection.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (task_id, kind, json.dumps(payload | context), now),
            )

    def start_model(self, provider, model, context, fallback_index):
        with self.db.connect() as connection:
            now = self.db.now()
            return connection.execute(
                "INSERT INTO model_runs(agent_run_id,task_id,agent,profile,complexity,provider,model,status,started_at,created_at,fallback_index,prompt_tokens,completion_tokens,cost) VALUES(?,?,?,?,?,?,?,'running',?,?,?,NULL,NULL,NULL)",
                (
                    context.get("agent_run_id"),
                    context.get("task_id"),
                    context.get("agent"),
                    context.get("profile"),
                    context.get("complexity"),
                    provider,
                    model,
                    now,
                    now,
                    fallback_index,
                ),
            ).lastrowid

    def finish_model(
        self,
        run_id,
        status,
        elapsed,
        usage=None,
        error_type=None,
        error_category=None,
        provider=None,
        model=None,
    ):
        if error_category is not None:
            from .errors import FailureCategory

            try:
                error_category = FailureCategory(error_category).value
            except ValueError as error:
                raise ValueError("unsupported model failure category") from error
        with self.db.connect() as connection:
            values = (
                (
                    usage.provider,
                    usage.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost,
                    usage.cached_tokens,
                    usage.reasoning_tokens,
                )
                if usage
                else (provider, model, None, None, None, None, None)
            )
            connection.execute(
                "UPDATE model_runs SET status=?,finished_at=?,latency_ms=?,provider=COALESCE(?,provider),model=COALESCE(?,model),prompt_tokens=?,completion_tokens=?,cost=?,cached_tokens=?,reasoning_tokens=?,error_type=?,error_category=? WHERE id=?",
                (
                    status,
                    self.db.now(),
                    elapsed,
                    *values,
                    error_type,
                    error_category,
                    run_id,
                ),
            )
