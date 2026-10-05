"""Keep test collection independent from operator-owned runtime configuration."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from harness.failure_trace import read_git_workflow_trace

_temporary_config = None
_previous_config_path = None


def pytest_configure():
    global _temporary_config, _previous_config_path

    root = Path(__file__).resolve().parents[1]
    operator_config = yaml.safe_load((root / "config.yaml").read_text())
    operator_config.pop("secrets", None)
    servers = operator_config.get("tools", {}).get("mcp", {}).get("servers", {})
    if isinstance(servers, dict):
        for server in servers.values():
            if isinstance(server, dict):
                server["enabled"] = False
    api = operator_config.get("api", {})
    if isinstance(api, dict):
        metrics_listener = api.get("metrics_listener", {})
        if isinstance(metrics_listener, dict):
            metrics_listener["enabled"] = False

    _previous_config_path = os.environ.get("HARNESS_CONFIG_PATH")
    _temporary_config = TemporaryDirectory(prefix="harness-pytest-config-")
    config_path = Path(_temporary_config.name) / "config.yaml"
    config_path.write_text(yaml.safe_dump(operator_config, sort_keys=True))
    os.environ["HARNESS_CONFIG_PATH"] = str(config_path)


def pytest_unconfigure():
    global _temporary_config

    if _previous_config_path is None:
        os.environ.pop("HARNESS_CONFIG_PATH", None)
    else:
        os.environ["HARNESS_CONFIG_PATH"] = _previous_config_path

    if _temporary_config is not None:
        _temporary_config.cleanup()
        _temporary_config = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.when != "call" or "git_" not in item.nodeid:
        return
    workspace = item.funcargs.get("tmp_path")
    if workspace is None:
        return
    for database in sorted((*workspace.glob("*.sqlite"), *workspace.glob("*.db"))):
        trace = read_git_workflow_trace(database)
        if trace:
            item.user_properties.append(
                (
                    "HARNESS_GIT_WORKFLOW_TRACE",
                    json.dumps(trace, ensure_ascii=True, separators=(",", ":")),
                )
            )
            return
