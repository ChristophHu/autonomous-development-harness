"""Keep test collection independent from operator-owned runtime configuration."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

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
