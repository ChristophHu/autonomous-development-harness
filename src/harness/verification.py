"""Create hash-bound evidence after the complete verification script succeeds."""

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from .coverage_gate import assert_full_coverage

_MATRIX_REPORT_START = "<!-- VERIFY-RESULT:START -->"
_MATRIX_REPORT_END = "<!-- VERIFY-RESULT:END -->"


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


def _matrix_status_counts(content):
    statuses = re.findall(
        r"^\|\s*\d+\s*\|[^\n|]*\|\s*(Erfüllt|Teilweise|Offen)\s*\|",
        content,
        flags=re.MULTILINE,
    )
    return {
        status: statuses.count(status) for status in ("Erfüllt", "Teilweise", "Offen")
    }


def _refresh_matrix(path, report_line):
    content = path.read_text(encoding="utf-8")
    # Older matrix revisions repeated status totals in an unmanaged paragraph.
    # Keep explanatory text, but make the managed block the only source of totals.
    content = re.sub(
        r"^Erfüllt:\s*\d+/\d+\s*\([^)]+\)\.\s*Teilweise:\s*\d+/\d+\.\s*"
        r"Offen:\s*\d+/\d+\.\s*",
        "Aktuelle GAP-Zahlen stehen ausschließlich im automatisch gepflegten "
        "Verifikationsblock oben. ",
        content,
        flags=re.MULTILINE,
    )
    block = f"{_MATRIX_REPORT_START}\n{report_line}\n{_MATRIX_REPORT_END}"
    markers_present = _MATRIX_REPORT_START in content or _MATRIX_REPORT_END in content
    if markers_present:
        if (
            content.count(_MATRIX_REPORT_START) != 1
            or content.count(_MATRIX_REPORT_END) != 1
        ):
            raise ValueError("verification report markers are malformed")
        start = content.index(_MATRIX_REPORT_START)
        end = content.index(_MATRIX_REPORT_END) + len(_MATRIX_REPORT_END)
        if end < start:
            raise ValueError("verification report markers are malformed")
        content = content[:start] + block + content[end:]
    else:
        heading_end = content.find("\n")
        if heading_end < 0:
            content = content + "\n\n" + block + "\n"
        else:
            content = (
                content[: heading_end + 1]
                + "\n"
                + block
                + "\n\n"
                + content[heading_end + 1 :]
            )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return content


def write_verification_report(matrix_path, coverage_path, junit_path, output_path):
    """Atomically write a report; caller invokes only after every verify check passes."""
    matrix_file = Path(matrix_path)
    coverage = Path(coverage_path).read_bytes()
    modules = assert_full_coverage(json.loads(coverage))
    counts = _counts(junit_path)
    if not counts["total"]:
        raise ValueError("verification report requires at least one test")
    if counts["passed"] != counts["total"]:
        raise ValueError("verification report cannot record failed or skipped tests")
    matrix_text = matrix_file.read_text(encoding="utf-8")
    statuses = _matrix_status_counts(matrix_text)
    coverage_data = json.loads(coverage)
    statement_total = sum(
        module["summary"]["num_statements"]
        for module in coverage_data["files"].values()
    )
    branch_total = sum(
        module["summary"]["num_branches"] for module in coverage_data["files"].values()
    )
    report_line = (
        f"Aktueller automatischer Verifikationsstand: {counts['passed']} Tests bestanden, "
        f"0 fehlgeschlagen/übersprungen; 100 % Statements/Branches/Funktionen "
        f"({statement_total} Statements, {branch_total} Branches, {len(modules)} Module, "
        f"{sum(module['functions'] for module in modules.values())} Funktionen); "
        f"Ruff und Formatcheck bestanden. GAP-Zählung aus Matrixzeilen: "
        f"{statuses['Erfüllt']} erfüllt, {statuses['Teilweise']} teilweise, {statuses['Offen']} offen."
    )
    matrix_text = _refresh_matrix(matrix_file, report_line)
    report = {
        "schema_version": 1,
        "passed": True,
        "checks": ["pytest", "coverage", "ruff", "format"],
        "completed_at": datetime.now(UTC).isoformat(),
        "matrix_sha256": hashlib.sha256(matrix_text.encode()).hexdigest(),
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
