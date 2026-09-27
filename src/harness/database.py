"""Versioned SQLite persistence and repository abstractions."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
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

    @staticmethod
    def now():
        return datetime.now(UTC).isoformat()


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
        from .domain import may_transition

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
            now = self.db.now()
            c.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                (str(target), now, task_id),
            )
            c.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (
                    task_id,
                    "task.status",
                    json.dumps({"from": row["status"], "status": str(target)}),
                    now,
                ),
            )


class EventRepository:
    def __init__(self, db):
        self.db = db

    def append(self, task_id, kind, payload):
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                (task_id, kind, json.dumps(payload), self.db.now()),
            )
            return cur.lastrowid

    def list(self, task_id=None, event_type=None, since=None, until=None):
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
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO decisions(task_id,decision,rationale,created_at) VALUES(?,?,?,?)",
                (task_id, decision, rationale, self.db.now()),
            )
            return cur.lastrowid

    def list(self, task_id):
        with self.db.connect() as c:
            return c.execute(
                "SELECT * FROM decisions WHERE task_id IS ? OR task_id IS NULL ORDER BY id",
                (task_id,),
            ).fetchall()


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

    def answer(self, question_id, answer, task_id=None):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT options,task_id FROM questions WHERE id=? AND status='open'",
                (question_id,),
            ).fetchone()
            if not row or (task_id is not None and row["task_id"] != task_id):
                return False
            options = json.loads(row["options"])
            if options and answer not in options:
                return False
            return (
                c.execute(
                    "UPDATE questions SET answer=?,status='answered',answered_at=? WHERE id=? AND status='open'",
                    (answer, self.db.now(), question_id),
                ).rowcount
                == 1
            )

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


class AuditRepository:
    """Atomic persistence of correlated tool events and model invocation spans."""

    def __init__(self, db):
        self.db = db

    def tool_event(self, task_id, kind, payload, context):
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
        provider=None,
        model=None,
    ):
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
                "UPDATE model_runs SET status=?,finished_at=?,latency_ms=?,provider=COALESCE(?,provider),model=COALESCE(?,model),prompt_tokens=?,completion_tokens=?,cost=?,cached_tokens=?,reasoning_tokens=?,error_type=? WHERE id=?",
                (status, self.db.now(), elapsed, *values, error_type, run_id),
            )
