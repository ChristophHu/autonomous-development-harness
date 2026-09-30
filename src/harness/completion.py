"""Fail-closed audit of the numbered requirements matrix."""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta

from .coverage_gate import assert_full_coverage

_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|[^|]*\|\s*(Erfüllt|Teilweise|Offen)\s*\|", re.MULTILINE
)


def audit_gap_matrix(markdown, expected=109):
    """Return a deterministic completion report; malformed matrices fail closed."""
    rows = [(int(number), state) for number, state in _ROW.findall(markdown)]
    numbers = [number for number, _ in rows]
    if len(numbers) != len(set(numbers)):
        raise ValueError("gap matrix contains duplicate requirement numbers")
    if sorted(numbers) != list(range(1, expected + 1)):
        raise ValueError("gap matrix does not contain the complete numbered scope")
    states = dict(rows)
    remaining = sorted(number for number, state in rows if state != "Erfüllt")
    return {
        "expected": expected,
        "fulfilled": sum(state == "Erfüllt" for _, state in rows),
        "partial": sum(state == "Teilweise" for _, state in rows),
        "open": sum(state == "Offen" for _, state in rows),
        "remaining": remaining,
        "complete": not remaining,
        "matrix_valid": len(states) == expected,
    }


def audit_harness_completion(
    markdown, coverage_text, evidence, *, source_sha256=None, now=None
):
    """Require a complete matrix and fresh, hash-bound successful verification."""
    matrix = audit_gap_matrix(markdown)
    reasons = []
    coverage_valid = False
    if isinstance(coverage_text, str):
        try:
            coverage = json.loads(coverage_text)
            assert_full_coverage(coverage)
            coverage_valid = True
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            reasons.append("coverage evidence is missing or invalid")
    else:
        reasons.append("coverage evidence is missing or invalid")

    evidence_valid = isinstance(evidence, dict)
    if not evidence_valid:
        reasons.append("verification evidence is missing")
    else:
        expected_hashes = {
            "matrix_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            "coverage_sha256": hashlib.sha256(coverage_text.encode()).hexdigest()
            if isinstance(coverage_text, str)
            else None,
            "source_sha256": source_sha256,
        }
        if any(evidence.get(key) != value for key, value in expected_hashes.items()):
            evidence_valid = False
            reasons.append("verification evidence does not match current inputs")
        counts = evidence.get("tests", {})
        count_keys = {"total", "passed", "failed", "errors", "skipped"}
        counts_valid = (
            isinstance(counts, dict)
            and set(counts) == count_keys
            and all(type(counts[key]) is int for key in count_keys)
        )
        if (
            evidence.get("schema_version") != 1
            or evidence.get("passed") is not True
            or evidence.get("checks") != ["pytest", "coverage", "ruff", "format"]
            or not counts_valid
            or counts.get("total", 0) < 1
            or counts.get("failed", 0) != 0
            or counts.get("errors", 0) != 0
            or counts.get("skipped", 0) != 0
            or counts.get("passed", 0) != counts.get("total", 0)
        ):
            evidence_valid = False
            reasons.append("verification checks or test results are incomplete")
        try:
            completed_at = datetime.fromisoformat(evidence["completed_at"])
            current = now or datetime.now(UTC)
            if (
                completed_at.tzinfo is None
                or completed_at > current
                or completed_at < current - timedelta(days=7)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            evidence_valid = False
            reasons.append("verification evidence is stale or has no valid timestamp")

    if not matrix["complete"]:
        reasons.append("GAP matrix still has partial or open requirements")
    complete = matrix["complete"] and coverage_valid and evidence_valid
    return {
        **matrix,
        "coverage_valid": coverage_valid,
        "verification_valid": evidence_valid,
        "complete": complete,
        "reasons": reasons,
    }
