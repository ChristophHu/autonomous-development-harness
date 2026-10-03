import importlib.util
import os
from pathlib import Path


def load_isolation_hooks():
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("test_config_isolation_hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_config_override_survives_collection_and_restores_after_session(
    monkeypatch,
):
    configured_path = "/operator/config.yaml"
    monkeypatch.setenv("HARNESS_CONFIG_PATH", configured_path)
    hooks = load_isolation_hooks()

    hooks.pytest_configure()
    test_config_path = Path(os.environ["HARNESS_CONFIG_PATH"])
    config_data = __import__("yaml").safe_load(test_config_path.read_text())

    assert test_config_path != Path(configured_path)
    assert "secrets" not in config_data
    assert all(
        not server["enabled"]
        for server in config_data["tools"]["mcp"]["servers"].values()
    )
    assert "lmstudio" in config_data["models"]["providers"]

    # The same override remains active for test bodies after collection.
    assert os.environ["HARNESS_CONFIG_PATH"] == str(test_config_path)

    hooks.pytest_unconfigure()

    assert os.environ["HARNESS_CONFIG_PATH"] == configured_path
    assert not test_config_path.exists()


def test_pytest_config_override_removes_environment_value_when_none_existed(
    monkeypatch,
):
    monkeypatch.delenv("HARNESS_CONFIG_PATH", raising=False)
    hooks = load_isolation_hooks()

    hooks.pytest_configure()
    assert Path(os.environ["HARNESS_CONFIG_PATH"]).is_file()

    hooks.pytest_unconfigure()

    assert "HARNESS_CONFIG_PATH" not in os.environ
