import json
import sqlite3

import pytest

from harness.database import Database, OperationalSnapshotRepository
from harness.services import OperationalDiagnosticsService
from harness.verify_clusters import cluster_junit, write_failure_clusters


def test_mcp_snapshots_append_and_return_only_latest(tmp_path):
    repository = OperationalSnapshotRepository(Database(tmp_path / "db.sqlite"))
    repository.record_mcp_status(
        [{"name": "vault", "state": "unavailable", "error": "MCPError"}]
    )
    repository.record_mcp_status(
        [{"name": "vault", "state": "available", "error": None}]
    )
    repository.record_mcp_status(
        [
            {
                "name": "misconfigured",
                "state": "invalid_config",
                "error": "invalid server configuration",
            }
        ]
    )
    snapshots = {item["server"]: item for item in repository.latest_mcp_statuses()}
    assert snapshots["vault"]["state"] == "available"
    assert snapshots["misconfigured"]["error_type"] is None
    assert len(repository.read_latest_mcp_statuses(tmp_path / "db.sqlite")) == 2
    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        repository.database.connect() as connection,
    ):
        connection.execute("DELETE FROM mcp_status_snapshots")
    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        repository.database.connect() as connection,
    ):
        connection.execute("UPDATE mcp_status_snapshots SET state='disabled'")


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        {"name": "x"},
        {"name": "x", "state": "bogus"},
        {"name": "x" * 129, "state": "available"},
        {"name": "x", "state": "available", "error": "e" * 65},
        {"name": "x", "state": "unavailable", "error": "secret token"},
    ],
)
def test_mcp_snapshot_input_validation(tmp_path, report):
    repository = OperationalSnapshotRepository(Database(tmp_path / "db.sqlite"))
    with pytest.raises((TypeError, ValueError)):
        repository.record_mcp_status(report if isinstance(report, list) else [report])
    with pytest.raises(TypeError, match="must be a list"):
        repository.record_mcp_status((report,))


def test_mcp_snapshot_read_is_noncreating_and_schema_tolerant(tmp_path):
    missing = tmp_path / "missing.sqlite"
    assert OperationalSnapshotRepository.read_latest_mcp_statuses(missing) == []
    assert not missing.exists()
    old = tmp_path / "old.sqlite"
    import sqlite3

    sqlite3.connect(old).close()
    assert OperationalSnapshotRepository.read_latest_mcp_statuses(old) == []


def test_operational_diagnostics_returns_typed_report_and_validates():
    report = OperationalDiagnosticsService.doctor(
        {"SQLite": True, "MCP": False}, details={"safe": "data"}
    )
    assert report.healthy is False
    assert report.details == {"safe": "data"}
    assert OperationalDiagnosticsService.status({"ok": True}).healthy is True
    for checks, details in (([], None), ({"wrong": 1}, None), ({}, [])):
        with pytest.raises(ValueError):
            OperationalDiagnosticsService.status(checks, details=details)


def test_failure_cluster_classifies_and_writes_report(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<testsuite>
      <testcase nodeid="test::sandbox"><failure message="sandbox-exec operation not permitted"/></testcase>
      <testcase nodeid="test::network"><error>Connection refused</error></testcase>
      <testcase nodeid="test::coverage"><failure>coverage failure</failure></testcase>
      <testcase nodeid="test::dependency"><failure>No module named x</failure></testcase>
      <testcase nodeid="tests.test_git_resume::test_retry"><failure>AssertionError: KeyError: 'phase'; expected waiting_human</failure><system-out>HARNESS_GIT_WORKFLOW_TRACE=[{"event_id":9,"task_id":3,"kind":"git.workflow","state":{"phase":"branch_created","workflow":"feature","branch":"secret"}}]</system-out></testcase>
      <testcase nodeid="tests.test_git_ssh::test_agent_identities"><error>PermissionError: socket [Errno 1] Operation not permitted</error></testcase>
      <testcase nodeid="tests.test_http_integration::test_tls"><error>PermissionError: socket [Errno 1] Operation not permitted</error></testcase>
      <testcase nodeid="test::other"><failure>unexpected failure</failure></testcase>
      <testcase nodeid="test::pass"/><testcase nodeid="test::skip"><skipped/></testcase>
    </testsuite>""",
        encoding="utf-8",
    )
    report = cluster_junit(junit)
    assert report["failed"] == 8
    assert {item["category"] for item in report["clusters"]} == {
        "sandbox",
        "network",
        "coverage",
        "dependency",
        "assertion",
        "other",
    }
    destination = tmp_path / "nested" / "clusters.json"
    written = write_failure_clusters(junit, destination)
    assert json.loads(destination.read_text()) == written
    assert written["breakdowns"] == {
        "network": {
            "http_tls_loopback": 1,
            "ssh_agent_socket": 1,
            "other_socket_or_network": 1,
        },
        "git_workflow_assertions": {"git_workflow_resume": 1},
        "git_workflow_assertions_with_event_trace": 1,
    }
    workflow_failure = next(
        test
        for cluster in written["clusters"]
        for test in cluster["tests"]
        if test["subsystem"] == "git_workflow_resume"
    )
    assert workflow_failure["missing_state_fields"] == ["phase"]
    assert workflow_failure["state_markers"] == ["waiting_human"]
    assert workflow_failure["workflow_event_trace"] == [
        {
            "event_id": 9,
            "task_id": 3,
            "kind": "git.workflow",
            "state": {
                "phase": "branch_created",
                "workflow": "feature",
            },
        }
    ]
    assert "test::sandbox" in written["clusters"][4]["tests"][0]["rerun"] or any(
        "test::sandbox" in test["rerun"]
        for cluster in written["clusters"]
        for test in cluster["tests"]
    )


def test_failure_cluster_accepts_safe_git_question_reason(tmp_path):
    junit = tmp_path / "question.xml"
    junit.write_text(
        '<testsuite><testcase nodeid="tests.test_git_resume::test_case">'
        "<failure>AssertionError</failure><system-out>"
        'HARNESS_GIT_WORKFLOW_TRACE=[{"event_id":1,"task_id":2,'
        '"kind":"QUESTION_ASKED","state":{"question_id":4,'
        '"reason_category":"git:reconciliation","secret":"hidden"}}]'
        "</system-out></testcase></testsuite>"
    )
    report = cluster_junit(junit)
    trace = report["clusters"][0]["tests"][0]["workflow_event_trace"]
    assert trace[0]["kind"] == "QUESTION_ASKED"
    assert trace[0]["state"] == {
        "question_id": 4,
        "reason_category": "git:reconciliation",
    }
    assert "hidden" not in repr(trace)


def test_failure_cluster_reads_trace_from_junit_property(tmp_path):
    junit = tmp_path / "trace-property.xml"
    junit.write_text(
        '<testsuite><testcase nodeid="tests.test_git_resume::test_case">'
        "<failure>AssertionError</failure><properties><property "
        'name="HARNESS_GIT_WORKFLOW_TRACE" value="[{&quot;event_id&quot;:1,'
        "&quot;kind&quot;:&quot;QUESTION_ASKED&quot;,&quot;state&quot;:{"
        '&quot;reason_category&quot;:&quot;git:reconciliation&quot;}}]"/>'
        "</properties></testcase></testsuite>"
    )
    report = cluster_junit(junit)
    trace = report["clusters"][0]["tests"][0]["workflow_event_trace"]
    assert trace[0]["state"] == {"reason_category": "git:reconciliation"}


def test_failure_cluster_falls_back_to_classname_and_handles_no_failures(tmp_path):
    junit = tmp_path / "empty.xml"
    junit.write_text('<testsuite><testcase classname="C" name="ok"/></testsuite>')
    assert cluster_junit(junit)["failed"] == 0


def test_failure_cluster_searches_full_traceback_but_bounds_saved_message(tmp_path):
    junit = tmp_path / "long.xml"
    detail = "x" * 2500 + " sandbox_apply: Operation not permitted"
    junit.write_text(
        f'<testsuite><testcase classname="C" name="test">'
        f"<failure>{detail}</failure></testcase></testsuite>"
    )
    report = cluster_junit(junit)
    case = report["clusters"][0]["tests"][0]
    assert report["clusters"][0]["category"] == "sandbox"
    assert case["signals"] == ["sandbox"]
    assert len(case["message"]) == 2000
