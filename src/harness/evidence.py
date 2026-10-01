"""Strict append-only external verification evidence with freshness binding."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
)


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["ci", "provider", "qdrant", "embedding", "http_tls"]
    source_id: StrictStr = Field(min_length=1, max_length=120)
    observed_at: AwareDatetime
    subject_sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    passed: StrictBool
    checks: dict[StrictStr, StrictBool] = Field(min_length=1, max_length=64)

    @field_validator("observed_at", mode="before")
    @classmethod
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("observed_at must be an ISO-8601 timestamp") from None
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}", value):
            raise ValueError("evidence source_id contains unsupported characters")
        return value

    @field_validator("checks")
    @classmethod
    def validate_check_names(cls, value):
        if any(not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", key) for key in value):
            raise ValueError("evidence check names must be bounded identifiers")
        return value


def _canonical(model: EvidenceInput) -> str:
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


def evidence_digest(model: EvidenceInput) -> str:
    return hashlib.sha256(_canonical(model).encode("utf-8")).hexdigest()


class EvidenceRepository:
    def __init__(self, database):
        self.database = database

    def record(self, payload):
        item = (
            payload
            if isinstance(payload, EvidenceInput)
            else EvidenceInput.model_validate(payload)
        )
        source_id = item.source_id
        digest = evidence_digest(item)
        with self.database.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO verification_evidence(kind,source_id,observed_at,subject_sha256,passed,checks_json,digest) VALUES(?,?,?,?,?,?,?)",
                    (
                        item.kind,
                        source_id,
                        item.observed_at.isoformat(),
                        item.subject_sha256,
                        int(item.passed),
                        json.dumps(item.checks, sort_keys=True),
                        digest,
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError("verification evidence already exists") from None
        return {
            "id": cursor.lastrowid,
            **item.model_dump(mode="json"),
            "observed_at": item.observed_at.isoformat(),
            "digest": digest,
        }

    def list(self, *, kind=None, limit=100):
        if kind is not None and kind not in {
            "ci",
            "provider",
            "qdrant",
            "embedding",
            "http_tls",
        }:
            raise ValueError("unsupported evidence kind")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            raise ValueError("limit must be between 1 and 500")
        query = "SELECT * FROM verification_evidence"
        params = []
        if kind is not None:
            query += " WHERE kind=?"
            params.append(kind)
        query += " ORDER BY observed_at DESC,id DESC LIMIT ?"
        params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "source_id": row["source_id"],
                "observed_at": row["observed_at"],
                "subject_sha256": row["subject_sha256"],
                "passed": bool(row["passed"]),
                "checks": json.loads(row["checks_json"]),
                "digest": row["digest"],
            }
            for row in rows
        ]


def audit_evidence(
    rows,
    *,
    expected_subject_sha256,
    now=None,
    max_age_hours=168,
):
    if not re.fullmatch(r"[a-f0-9]{64}", expected_subject_sha256 or ""):
        raise ValueError("expected subject must be a SHA-256 digest")
    if (
        isinstance(max_age_hours, bool)
        or not isinstance(max_age_hours, int)
        or max_age_hours < 1
    ):
        raise ValueError("max_age_hours must be positive")
    reference = now or datetime.now(UTC)
    cutoff = reference - timedelta(hours=max_age_hours)
    reports = []
    for row in rows:
        observed = datetime.fromisoformat(row["observed_at"])
        if observed.tzinfo is None:
            reason = "timestamp_not_aware"
        elif observed > reference:
            reason = "timestamp_in_future"
        elif observed < cutoff:
            reason = "stale"
        elif row["subject_sha256"] != expected_subject_sha256:
            reason = "subject_mismatch"
        elif not row["passed"] or not all(
            row.get("checks", json.loads(row.get("checks_json", "{}"))).values()
        ):
            reason = "checks_failed"
        else:
            reason = None
        reports.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "valid": reason is None,
                "reason": reason,
            }
        )
    return {
        "expected_subject_sha256": expected_subject_sha256,
        "max_age_hours": max_age_hours,
        "items": reports,
        "healthy": bool(reports) and all(item["valid"] for item in reports),
    }
