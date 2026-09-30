import pytest

from harness.completion import audit_gap_matrix, audit_harness_completion


def matrix(states):
    header = "| Nr. | Name | Status | Tiefe |\n|---:|---|---|---|\n"
    return header + "".join(
        f"| {number} | requirement | {state} | evidence |\n"
        for number, state in enumerate(states, 1)
    )


def test_audit_reports_partial_and_open_points_without_false_completion():
    result = audit_gap_matrix(matrix(["Erfüllt", "Teilweise", "Offen"]), expected=3)

    assert result == {
        "expected": 3,
        "fulfilled": 1,
        "partial": 1,
        "open": 1,
        "remaining": [2, 3],
        "complete": False,
        "matrix_valid": True,
    }


def test_audit_accepts_only_all_fulfilled_numbered_requirements():
    result = audit_gap_matrix(matrix(["Erfüllt", "Erfüllt"]), expected=2)

    assert result["complete"] is True
    assert result["remaining"] == []


@pytest.mark.parametrize(
    "document", [matrix(["Erfüllt", "Erfüllt"]).replace("| 2 |", "| 1 |"), ""]
)
def test_audit_rejects_duplicate_or_missing_rows(document):
    with pytest.raises(ValueError, match="complete numbered scope|duplicate"):
        audit_gap_matrix(document, expected=2)


def test_audit_rejects_unsupported_status():
    with pytest.raises(ValueError, match="complete numbered scope"):
        audit_gap_matrix(matrix(["unknown"]), expected=1)


def coverage_report():
    summary = {
        "missing_lines": 0,
        "missing_branches": 0,
        "excluded_lines": 0,
        "num_statements": 1,
        "num_branches": 1,
    }
    return {
        "meta": {"branch_coverage": True},
        "files": {
            "module": {"summary": summary, "functions": {"f": {"summary": summary}}}
        },
    }


def test_completion_requires_hash_bound_current_verification():
    import hashlib
    import json
    from datetime import UTC, datetime

    markdown = matrix(["Erfüllt"] * 109)
    coverage = json.dumps(coverage_report())
    completed = datetime.now(UTC)
    evidence = {
        "schema_version": 1,
        "passed": True,
        "checks": ["pytest", "coverage", "ruff", "format"],
        "completed_at": completed.isoformat(),
        "matrix_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "coverage_sha256": hashlib.sha256(coverage.encode()).hexdigest(),
        "source_sha256": "source-current",
        "tests": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0},
    }

    result = audit_harness_completion(
        markdown, coverage, evidence, source_sha256="source-current", now=completed
    )

    assert result["complete"] is True
    assert result["coverage_valid"] is True
    assert result["verification_valid"] is True


def test_completion_rejects_matrix_or_missing_proof():
    markdown = matrix(["Erfüllt"] * 108 + ["Teilweise"])

    result = audit_harness_completion(markdown, None, None)

    assert result["complete"] is False
    assert result["remaining"] == [109]
    assert "verification evidence is missing" in result["reasons"]
    assert "coverage evidence is missing or invalid" in result["reasons"]


@pytest.mark.parametrize("coverage", ["not-json", "{}", 3])
def test_completion_rejects_invalid_coverage_text(coverage):
    result = audit_harness_completion(matrix(["Erfüllt"] * 109), coverage, None)

    assert result["coverage_valid"] is False


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "source",
        "checks",
        "count",
        "bad-count",
        "schema",
        "timestamp",
        "stale",
        "future",
        "passed",
    ],
)
def test_completion_rejects_stale_or_incomplete_verification(change):
    import hashlib
    import json
    from datetime import UTC, datetime

    markdown = matrix(["Erfüllt"] * 109)
    coverage = json.dumps(coverage_report())
    current = datetime.now(UTC)
    evidence = {
        "schema_version": 1,
        "passed": True,
        "checks": ["pytest", "coverage", "ruff", "format"],
        "completed_at": current.isoformat(),
        "matrix_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "coverage_sha256": hashlib.sha256(coverage.encode()).hexdigest(),
        "source_sha256": "source-current",
        "tests": {"total": 2, "passed": 2, "failed": 0, "errors": 0, "skipped": 0},
    }
    if change == "hash":
        evidence["matrix_sha256"] = "stale"
    elif change == "source":
        evidence["source_sha256"] = "stale"
    elif change == "checks":
        evidence["checks"] = []
    elif change == "count":
        evidence["tests"]["failed"] = 1
    elif change == "bad-count":
        evidence["tests"]["total"] = "two"
    elif change == "schema":
        evidence["schema_version"] = 2
    elif change == "timestamp":
        evidence["completed_at"] = "invalid"
    elif change == "stale":
        from datetime import timedelta

        evidence["completed_at"] = (current - timedelta(days=8)).isoformat()
    elif change == "future":
        from datetime import timedelta

        evidence["completed_at"] = (current + timedelta(seconds=1)).isoformat()
    else:
        evidence["passed"] = False

    result = audit_harness_completion(
        markdown, coverage, evidence, source_sha256="source-current", now=current
    )

    assert result["complete"] is False
    assert result["verification_valid"] is False
