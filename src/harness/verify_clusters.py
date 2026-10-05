"""Classify pytest JUnit failures into actionable, non-authoritative clusters."""

from __future__ import annotations

import json
import re
import shlex
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

_SIGNALS = (
    (
        "sandbox",
        (
            "sandbox_apply",
            "sandbox-exec",
            "host sandbox blocked",
            "sandbox operation not permitted",
            "host_sandbox_blocked",
            "sandbox_execution_denied",
        ),
    ),
    ("network", ("connection refused", "network is unreachable", "loopback", "socket")),
    ("coverage", ("coverage failure", "coverage report", "not covered")),
    ("dependency", ("modulenotfounderror", "no module named", "importerror")),
    ("assertion", ("assertionerror", "assert ")),
)
_TRACE_STATES = {
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
_TRACE_WORKFLOWS = {"feature", "bugfix", "hotfix", "release", "other"}


def _failure(case):
    failure = case.find("failure")
    if failure is None:
        failure = case.find("error")
    if failure is None:
        return None
    trace_properties = " ".join(
        f"HARNESS_GIT_WORKFLOW_TRACE={prop.get('value', '')}"
        for prop in case.findall("./properties/property")
        if prop.get("name") == "HARNESS_GIT_WORKFLOW_TRACE"
    )
    sections = (
        failure.get("message", ""),
        failure.text or "",
        case.findtext("system-out", ""),
        trace_properties,
    )
    return " ".join(sections).strip()


def _display_message(message):
    """Remove raw task traces from persisted cluster excerpts."""
    return re.sub(
        r"(?m)HARNESS_GIT_WORKFLOW_TRACE=.*$",
        "HARNESS_GIT_WORKFLOW_TRACE=[redacted]",
        message,
    )


def _category(message):
    folded = message.casefold()
    for category, signals in _SIGNALS:
        if any(signal in folded for signal in signals):
            return category
    return "other"


def _subsystem(nodeid):
    module = nodeid.split("::", 1)[0].rsplit(".", 1)[-1]
    return {
        "test_git_http": "git_https_proxy",
        "test_git_ssh": "git_ssh",
        "test_http_integration": "http_tls_loopback",
        "test_git_resume": "git_workflow_resume",
        "test_git_integration": "git_workflow_integration",
        "test_git_broker": "git_broker",
        "test_process_isolation": "process_isolation",
    }.get(module, module.removeprefix("test_"))


def _workflow_signals(nodeid, message):
    subsystem = _subsystem(nodeid)
    if subsystem not in {"git_workflow_resume", "git_workflow_integration"}:
        return {}
    fields = sorted(set(re.findall(r"KeyError: ['\"]([^'\"]+)['\"]", message)))
    states = sorted(
        set(
            re.findall(
                r"\b(waiting_human|waiting_approval|waiting_decision|completed|failed)\b",
                message,
                flags=re.IGNORECASE,
            )
        )
    )
    signals = {}
    if fields:
        signals["missing_state_fields"] = fields
    if states:
        signals["state_markers"] = [state.lower() for state in states]
    marker = "HARNESS_GIT_WORKFLOW_TRACE="
    line = next((item for item in message.splitlines() if marker in item), None)
    if line is not None:
        try:
            trace = json.loads(line.split(marker, 1)[1])
        except (IndexError, json.JSONDecodeError):
            trace = None
        if isinstance(trace, list):
            safe_trace = []
            for event in trace[:30]:
                if not isinstance(event, dict):
                    continue
                kind = event.get("kind")
                state = event.get("state")
                if (
                    type(event.get("event_id")) is not int
                    or not isinstance(kind, str)
                    or not (
                        kind in {"task.status", "question.asked", "QUESTION_ASKED"}
                        or re.fullmatch(r"git\.[a-z_]{1,48}", kind)
                    )
                    or not isinstance(state, dict)
                ):
                    continue
                safe_state = {}
                for key in ("status", "from"):
                    if isinstance(state.get(key), str) and state[key] in _TRACE_STATES:
                        safe_state[key] = state[key]
                if (
                    isinstance(state.get("workflow"), str)
                    and state["workflow"] in _TRACE_WORKFLOWS
                ):
                    safe_state["workflow"] = state["workflow"]
                phase = state.get("phase")
                if isinstance(phase, str) and re.fullmatch(
                    r"[a-z][a-z0-9_]{0,39}", phase
                ):
                    safe_state["phase"] = phase
                if isinstance(state.get("purpose"), str) and state["purpose"] in {
                    "input",
                    "decision",
                    "approval",
                }:
                    safe_state["purpose"] = state["purpose"]
                question_id = state.get("question_id")
                if type(question_id) is int and question_id > 0:
                    safe_state["question_id"] = question_id
                reason_category = state.get("reason_category")
                if isinstance(reason_category, str) and re.fullmatch(
                    r"git:[a-z_]{1,40}", reason_category
                ):
                    safe_state["reason_category"] = reason_category
                failure_class = state.get("failure_class")
                if failure_class in {
                    "host_sandbox_blocked",
                    "sandbox_execution_denied",
                }:
                    safe_state["failure_class"] = failure_class
                safe_trace.append(
                    {
                        "event_id": event["event_id"],
                        "task_id": event.get("task_id")
                        if type(event.get("task_id")) is int
                        else None,
                        "kind": kind,
                        "state": safe_state,
                    }
                )
            if safe_trace:
                signals["workflow_event_trace"] = safe_trace
    return signals


def _network_subcategory(nodeid):
    folded = nodeid.casefold()
    if "test_git_ssh" in folded:
        if "agent" in folded:
            return "ssh_agent_socket"
        if "proxy" in folded:
            return "ssh_loopback_proxy"
        return "ssh_transport_or_socket"
    if "test_git_http" in folded:
        if "proxy" in folded:
            return "git_https_proxy_listener"
        return "https_git_transport"
    if "test_http_integration" in folded:
        return "http_tls_loopback"
    if "daemon_socket" in folded:
        return "daemon_unix_socket"
    return "other_socket_or_network"


def cluster_junit(junit_path):
    """Return cluster counts and reproducible node ids from a pytest JUnit file."""
    root = ET.parse(junit_path).getroot()
    clusters = defaultdict(list)
    for case in root.findall(".//testcase"):
        message = _failure(case)
        if message is None:
            continue
        nodeid = case.get("nodeid") or "::".join(
            value for value in (case.get("classname"), case.get("name")) if value
        )
        category = _category(message)
        entry = {
            "nodeid": nodeid,
            "subsystem": _subsystem(nodeid),
            "message": _display_message(message)[:2000],
            "signals": [
                signal_category
                for signal_category, signals in _SIGNALS
                if any(signal in message.casefold() for signal in signals)
            ],
            "rerun": ".venv/bin/pytest -q " + shlex.quote(nodeid),
        }
        if category == "network":
            entry["network_subcategory"] = _network_subcategory(nodeid)
        if entry["subsystem"] in {
            "git_workflow_resume",
            "git_workflow_integration",
        }:
            entry.update(_workflow_signals(nodeid, message))
        clusters[category].append(entry)
    network_breakdown = defaultdict(int)
    workflow_breakdown = defaultdict(int)
    workflow_trace_count = 0
    for category, entries in clusters.items():
        for entry in entries:
            if category == "network":
                network_breakdown[entry["network_subcategory"]] += 1
            if category == "assertion" and entry["subsystem"].startswith(
                "git_workflow_"
            ):
                workflow_breakdown[entry["subsystem"]] += 1
                workflow_trace_count += bool(entry.get("workflow_event_trace"))
    return {
        "schema_version": 1,
        "source": str(junit_path),
        "failed": sum(map(len, clusters.values())),
        "breakdowns": {
            "network": dict(sorted(network_breakdown.items())),
            "git_workflow_assertions": dict(sorted(workflow_breakdown.items())),
            "git_workflow_assertions_with_event_trace": workflow_trace_count,
        },
        "clusters": [
            {"category": key, "count": len(clusters[key]), "tests": clusters[key]}
            for key in sorted(clusters)
        ],
    }


def write_failure_clusters(junit_path, output_path):
    report = cluster_junit(junit_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
