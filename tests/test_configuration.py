"""Typed configuration contract tests."""

import pytest
import yaml
from pydantic import ValidationError

from harness.configuration import HarnessConfig
from harness.core import ConfigurationService


def test_typed_config_accepts_current_runtime_configuration():
    settings = HarnessConfig.model_validate(
        {
            "harness": {"name": "harness", "environment": "development"},
            "paths": {"workspace": "./work", "database": "./data/db.sqlite"},
            "database": {"type": "sqlite"},
            "memory": {
                "qdrant": {
                    "enabled": False,
                    "url": "http://127.0.0.1:6333",
                    "collection": "mem",
                    "timeout": {"total": 5, "connect": 2, "read": 5},
                },
                "embeddings": {
                    "provider": "lmstudio",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "model": "embed-v1",
                    "dimensions": 1024,
                    "batch_size": 32,
                    "timeout": 30,
                },
            },
            "git": {
                "enabled": True,
                "branches": {"main": "main", "development": "dev"},
                "workflow": {"feature": "feature/", "release": "release/"},
            },
            "docker": {"enabled": True, "compose_preferred": True},
            "api": {"enabled": True, "host": "127.0.0.1", "port": 8080},
            "logging": {"level": "INFO", "file": "./logs/harness.log"},
            "models": {
                "defaults": {"provider": "openai"},
                "providers": {
                    "openai": {
                        "enabled": True,
                        "base_url": "https://api.openai.com/v1",
                        "retry": {
                            "max_attempts": 3,
                            "base_delay": 0.25,
                            "max_delay": 4,
                        },
                    }
                },
                "registry": {
                    "openai_sol": {
                        "provider": "openai",
                        "model": "vendor-model-id",
                        "tier": "advanced",
                    }
                },
                "strategies": {
                    "fast": {"provider": "openai", "model": "vendor-model-id"}
                },
            },
            "profiles": {
                "coding": {
                    "model": {"primary": "openai_sol", "fallback": ["local"]},
                    "instructions": "Implement safely",
                    "tools": ["filesystem.read"],
                    "permissions": ["filesystem"],
                    "max_steps": 20,
                }
            },
            "testing": {"tdd": True, "coverage": {"statements": 100, "branches": 100}},
            "tools": {
                "permissions": {"filesystem": "write"},
                "http": {"allowed_hosts": ["api.example.test"], "timeout": 30},
                "git": {
                    "ssh": {
                        "allowed_hosts": ["git.example.test"],
                        "allowed_ports": [22],
                    }
                },
            },
        }
    )

    assert settings.models.providers["openai"].retry.max_attempts == 3
    assert settings.models.registry["openai_sol"].model == "vendor-model-id"
    assert settings.profiles["coding"].model.primary == "openai_sol"
    assert settings.testing.coverage.branches == 100


def test_typed_config_preserves_extensions_at_root_and_known_sections():
    settings = HarnessConfig.model_validate(
        {
            "x_org_extension": {"mode": "custom"},
            "models": {
                "x_registry_source": "catalog",
                "providers": {"local": {"enabled": False, "x_transport": "unix"}},
            },
            "profiles": {"coding": {"model": {}, "x_prompt_revision": 4}},
        }
    )

    assert settings.model_extra["x_org_extension"] == {"mode": "custom"}
    assert settings.models.model_extra["x_registry_source"] == "catalog"
    assert settings.models.providers["local"].model_extra["x_transport"] == "unix"
    assert settings.profiles["coding"].model_extra["x_prompt_revision"] == 4


def test_configuration_service_exposes_fresh_typed_settings_and_path(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        core, "SecretResolver", lambda: SimpleNamespace(get=lambda _key: None)
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "paths": {"workspace": "./initial", "extension_data": "./extra"},
                "api": {"port": 8090},
            }
        )
    )
    config = ConfigurationService(path)

    assert config.settings.api.port == 8090
    assert config.path("workspace") == tmp_path / "initial"
    assert config.path("extension_data") == tmp_path / "extra"
    assert config.path("unconfigured") == tmp_path / "unconfigured"

    config.data["api"]["port"] = "private-invalid-value"
    with pytest.raises(
        ValueError, match="api.port has an invalid configuration type or value"
    ) as error:
        _ = config.settings
    assert "private-invalid-value" not in str(error.value)

    config.data["api"]["port"] = 8090
    config.data["models"]["registry"] = {
        "model": {"provider": "local", "model": "x", "tier": 42}
    }
    with pytest.raises(
        ValueError,
        match="models.registry.model.tier has an invalid configuration type or value",
    ):
        config.validate()


def test_configuration_service_reload_is_atomic_and_refreshes_sources(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        core, "SecretResolver", lambda: SimpleNamespace(get=lambda _key: None)
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"api": {"port": 8100}}))
    config = ConfigurationService(path)

    path.write_text(yaml.safe_dump({"api": {"port": 8200}}))
    assert config.reload() is config
    assert config.settings.api.port == 8200

    path.write_text(yaml.safe_dump({"api": {"port": "invalid-secret-value"}}))
    with pytest.raises(ValueError, match="api.port must be an integer") as error:
        config.reload()

    assert "invalid-secret-value" not in str(error.value)
    assert config.settings.api.port == 8200


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ({"api": {"port": True}}, "api.port"),
        ({"api": {"swagger": "true"}}, "api.swagger"),
        ({"git": {"enabled": 1}}, "git.enabled"),
        ({"profiles": {"coding": {"max_steps": 0}}}, "profiles.coding.max_steps"),
        (
            {"models": {"registry": {"x": {"provider": 1, "model": "m"}}}},
            "models.registry.x.provider",
        ),
        (
            {"memory": {"embeddings": {"batch_size": 257}}},
            "memory.embeddings.batch_size",
        ),
        (
            {"testing": {"coverage": {"branches": 101}}},
            "testing.coverage.branches",
        ),
        (
            {"tools": {"git": {"ssh": {"allowed_ports": [True]}}}},
            "tools.git.ssh.allowed_ports.0",
        ),
    ],
)
def test_typed_config_rejects_invalid_known_fields(payload, path):
    with pytest.raises(ValidationError) as error:
        HarnessConfig.model_validate(payload)

    assert path in str(error.value)
