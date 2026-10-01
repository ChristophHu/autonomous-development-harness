import hashlib
import json
import xml.etree.ElementTree as ET

import pytest
from test_completion import coverage_report

from harness.verification import source_tree_sha256, write_verification_report


@pytest.mark.parametrize("matrix_content", ["matrix", "matrix\nbody"])
def test_verification_report_hashes_matrix_and_coverage(tmp_path, matrix_content):
    matrix = tmp_path / "GAP_MATRIX.md"
    matrix.write_text(matrix_content)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps(coverage_report()))
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite tests="2"><testcase name="one" /><testcase name="two" /></testsuite>'
    )
    output = tmp_path / "data" / "verification.json"

    report = write_verification_report(matrix, coverage, junit, output)

    assert report["passed"] is True
    assert report["tests"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert json.loads(output.read_text()) == report
    assert report["source_sha256"] == source_tree_sha256(tmp_path)
    assert report["matrix_sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()
    assert "2 Tests bestanden" in matrix.read_text()
    assert list(output.parent.iterdir()) == [output]


def test_verification_report_replaces_managed_summary_and_counts_matrix_rows(tmp_path):
    matrix = tmp_path / "GAP_MATRIX.md"
    matrix.write_text(
        "# Gap\n\n<!-- VERIFY-RESULT:START -->\nstale\n<!-- VERIFY-RESULT:END -->\n"
        "| 1 | One | Erfüllt | evidence |\n| 2 | Two | Teilweise | evidence |\n"
        "| 3 | Three | Offen | evidence |\n"
    )
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps(coverage_report()))
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite tests="1"><testcase /></testsuite>')

    report = write_verification_report(
        matrix, coverage, junit, tmp_path / "report.json"
    )
    content = matrix.read_text()

    assert content.count("VERIFY-RESULT:START") == 1
    assert "stale" not in content
    assert "1 erfüllt, 1 teilweise, 1 offen" in content
    assert report["matrix_sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "markers",
    [
        "<!-- VERIFY-RESULT:START -->\n",
        "<!-- VERIFY-RESULT:END -->\n<!-- VERIFY-RESULT:START -->",
        (
            "<!-- VERIFY-RESULT:START --><!-- VERIFY-RESULT:START -->\n"
            "<!-- VERIFY-RESULT:END -->"
        ),
    ],
)
def test_verification_report_rejects_malformed_matrix_markers(tmp_path, markers):
    matrix = tmp_path / "GAP_MATRIX.md"
    matrix.write_text("# Gap\n" + markers)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps(coverage_report()))
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite tests="1"><testcase /></testsuite>')

    with pytest.raises(ValueError, match="markers are malformed"):
        write_verification_report(matrix, coverage, junit, tmp_path / "report.json")


def test_source_hash_is_stable_and_tracks_contract_files(tmp_path):
    implementation = tmp_path / "src" / "harness" / "module.py"
    tests = tmp_path / "tests" / "test_module.py"
    script = tmp_path / "scripts" / "verify.sh"
    project = tmp_path / "pyproject.toml"
    for path in (implementation, tests, script, project):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version one")

    first = source_tree_sha256(tmp_path)
    assert source_tree_sha256(tmp_path) == first
    tests.write_text("version two")
    assert source_tree_sha256(tmp_path) != first
    assert len(source_tree_sha256(tmp_path / "empty")) == 64


@pytest.mark.parametrize(
    "testcase",
    [
        "<failure />",
        "<error />",
        "<skipped />",
    ],
)
def test_verification_report_refuses_nonpassing_suites(tmp_path, testcase):
    matrix = tmp_path / "matrix"
    matrix.write_text("matrix")
    coverage = tmp_path / "coverage"
    coverage.write_text(json.dumps(coverage_report()))
    junit = tmp_path / "junit.xml"
    junit.write_text(
        f'<testsuite tests="1"><testcase>{testcase}</testcase></testsuite>'
    )

    with pytest.raises(ValueError, match="failed or skipped"):
        write_verification_report(matrix, coverage, junit, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    "contents,error", [("<not-closed>", ET.ParseError), ("<testsuite />", ValueError)]
)
def test_verification_report_rejects_malformed_or_empty_junit(
    tmp_path, contents, error
):
    matrix = tmp_path / "matrix"
    matrix.write_text("matrix")
    coverage = tmp_path / "coverage"
    coverage.write_text(json.dumps(coverage_report()))
    junit = tmp_path / "junit.xml"
    junit.write_text(contents)

    with pytest.raises(error):
        write_verification_report(matrix, coverage, junit, tmp_path / "out.json")


def test_verification_report_requires_complete_coverage(tmp_path):
    matrix = tmp_path / "matrix"
    matrix.write_text("matrix")
    coverage = tmp_path / "coverage"
    coverage.write_text("{}")
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite tests="1"><testcase /></testsuite>')

    with pytest.raises(ValueError):
        write_verification_report(matrix, coverage, junit, tmp_path / "out.json")
