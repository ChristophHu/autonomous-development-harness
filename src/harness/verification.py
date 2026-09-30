"""Create hash-bound evidence after the complete verification script succeeds."""

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from .coverage_gate import assert_full_coverage


def source_tree_sha256(root):
    """Hash implementation, tests, and the verification contract deterministically."""
    base = Path(root)
    candidates = []
    for directory in (base / "src" / "harness", base / "tests"):
        if directory.is_dir():
            candidates.extend(directory.rglob("*.py"))
    candidates.extend(
        path
        for path in (base / "scripts" / "verify.sh", base / "pyproject.toml")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(base).as_posix()):
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _counts(junit_path):
    root = ET.parse(junit_path).getroot()
    cases = root.findall(".//testcase")
    total = len(cases)
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "total": total,
        "passed": total - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
    }


def write_verification_report(matrix_path, coverage_path, junit_path, output_path):
    """Atomically write a report; caller invokes only after every verify check passes."""
    matrix = Path(matrix_path).read_bytes()
    coverage = Path(coverage_path).read_bytes()
    assert_full_coverage(json.loads(coverage))
    counts = _counts(junit_path)
    if not counts["total"]:
        raise ValueError("verification report requires at least one test")
    if counts["passed"] != counts["total"]:
        raise ValueError("verification report cannot record failed or skipped tests")
    report = {
        "schema_version": 1,
        "passed": True,
        "checks": ["pytest", "coverage", "ruff", "format"],
        "completed_at": datetime.now(UTC).isoformat(),
        "matrix_sha256": hashlib.sha256(matrix).hexdigest(),
        "coverage_sha256": hashlib.sha256(coverage).hexdigest(),
        "source_sha256": source_tree_sha256(Path(matrix_path).parent),
        "tests": counts,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return report
