"""Secret-safe, read-only extraction of task/Git event breadcrumbs for failed tests."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

_SAFE_STATES = {
    "queued",
    "planning",
    "executing",
    "testing",
    "validating",
    "completed",
    "failed",
    "cancelled",
    "waiting_human",
    "waiting_decision",
    "waiting_approval",
    "running",
    "stopped",
    "stale",
}
_SAFE_WORKFLOWS = {"feature", "bugfix", "hotfix", "release", "other"}
_SAFE_PHASE = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")


def _safe_payload(payload):
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("status", "from"):
        item = value.get(key)
        if isinstance(item, str) and item in _SAFE_STATES:
            result[key] = item
    workflow = value.get("workflow")
    if isinstance(workflow, str) and workflow in _SAFE_WORKFLOWS:
        result["workflow"] = workflow
    phase = value.get("phase")
    if isinstance(phase, str) and _SAFE_PHASE.fullmatch(phase):
        result["phase"] = phase
    purpose = value.get("purpose")
    if isinstance(purpose, str) and purpose in {"input", "decision", "approval"}:
        result["purpose"] = purpose
    question_id = value.get("question_id")
    if type(question_id) is int and question_id > 0:
        result["question_id"] = question_id
    return result


def _question_failure_class(question):
    """Return a bounded diagnostic code, never the stored question text."""
    if not isinstance(question, str):
        return None
    folded = question.casefold()
    if "sandbox_apply: operation not permitted" in folded:
        return "host_sandbox_blocked"
    if "sandbox-exec:" in folded and "operation not permitted" in folded:
        return "sandbox_execution_denied"
    return None


def read_git_workflow_trace(database_path, *, limit=30):
    """Read only bounded Git/lifecycle breadcrumbs; never return raw payloads."""
    path = Path(database_path)
    if not path.is_file() or isinstance(limit, bool) or not isinstance(limit, int):
        return []
    if not 1 <= limit <= 100:
        return []
    try:
        with closing(
            sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=0.2)
        ) as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "events" not in tables:
                return []
            rows = db.execute(
                "SELECT id, task_id, kind, payload FROM events "
                "WHERE kind LIKE 'git.%' OR kind IN "
                "('task.status','question.asked','QUESTION_ASKED') "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            question_ids = sorted(
                {
                    question_id
                    for row in rows
                    for question_id in [_safe_payload(row[3]).get("question_id")]
                    if type(question_id) is int
                }
            )
            question_details = {}
            if question_ids and "questions" in tables:
                placeholders = ",".join("?" for _ in question_ids)
                columns = {row[1] for row in db.execute("PRAGMA table_info(questions)")}
                question_expression = "question" if "question" in columns else "NULL"
                question_details = {
                    row[0]: (row[1], row[2])
                    for row in db.execute(
                        f"SELECT id, reason, {question_expression} FROM questions "
                        f"WHERE id IN ({placeholders})",
                        question_ids,
                    ).fetchall()
                }
    except (OSError, sqlite3.Error, ValueError):
        return []
    trace = []
    for row in reversed(rows):
        state = _safe_payload(row[3])
        details = question_details.get(state.get("question_id"), (None, None))
        reason, question = details
        if isinstance(reason, str) and re.fullmatch(r"git:[a-z_]{1,40}", reason):
            state["reason_category"] = reason
        failure_class = _question_failure_class(question)
        if failure_class is not None:
            state["failure_class"] = failure_class
        trace.append(
            {
                "event_id": row[0],
                "task_id": row[1],
                "kind": row[2],
                "state": state,
            }
        )
    return trace
