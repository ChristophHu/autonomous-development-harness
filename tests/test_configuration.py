"""Typed configuration contract tests."""

import pytest
import yaml
from pydantic import ValidationError

from harness.configuration import HarnessConfig
from harness.core import ConfigurationService


def test_complete_example_config_is_valid_and_has_specified_sections(tmp_path):
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "config.example.yaml"
    data = yaml.safe_load(sample.read_text())
    settings = HarnessConfig.model_validate(data)
    assert set(settings.models.registry) >= {
        "openai_astra",
        "openai_sol",
        "openai_luna",
        "deepseek_flash",
        "qwen_local",
    }
    assert {item.tier for item in settings.models.registry.values()} >= {
        "premium",
        "advanced",
        "standard",
        "economical",
        "local",
    }
    assert set(settings.profiles) >= {
        "software-architect",
        "planner",
        "coding",
        "validator",
        "test-engineer",
        "documentation",
        "classification",
    }
    assert settings.git.workflow["feature"] == "feature/"
    assert settings.testing.coverage.branches == 100
    assert settings.tools.permissions["filesystem.delete"] == "denied"
    assert ConfigurationService(sample).validate()


def test_provider_kind_is_explicit_and_validated():
    settings = HarnessConfig.model_validate(
        {"models": {"providers": {"local": {"kind": "lmstudio"}}}}
    )
    assert settings.models.providers["local"].kind == "lmstudio"
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"providers": {"local": {"kind": "unknown"}}}}
        )


def test_mcp_streamable_http_requires_https_exact_host_binding_and_allowlist():
    server = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test/rpc",
        "allowed_hosts": ["mcp.example.test"],
        "allow_tools": ["lookup"],
    }
    settings = HarnessConfig.model_validate(
        {"tools": {"mcp": {"servers": {"remote": server}}}}
    )
    assert settings.tools.mcp.servers["remote"].transport == "streamable_http"
    for invalid in (
        server | {"url": "http://mcp.example.test/rpc"},
        server | {"url": "https://attacker.test/rpc"},
        server | {"url": "https://user:pass@mcp.example.test/rpc"},
        server | {"allowed_hosts": []},
        server | {"auth_secret": "bad-name"},
        server | {"url": "https://mcp.example.test:bad/rpc"},
        server | {"read_only": False},
        server | {"trusted_local": True},
        {
            "command": ["mcp-server"],
            "trusted_local": True,
            "url": "https://mcp.example.test/rpc",
            "allow_tools": ["lookup"],
        },
    ):
        with pytest.raises(ValidationError):
            HarnessConfig.model_validate(
                {"tools": {"mcp": {"servers": {"remote": invalid}}}}
            )


@pytest.mark.parametrize(
    "logging_settings",
    [
        {"max_bytes": 1023},
        {"max_bytes": 104_857_601},
        {"max_bytes": True},
        {"backup_count": -1},
        {"backup_count": 11},
    ],
)
def test_logging_rotation_settings_are_bounded_and_strict(logging_settings):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"logging": logging_settings})


@pytest.mark.parametrize(
    "tier", ["premium", "advanced", "standard", "economical", "local"]
)
def test_model_tier_accepts_each_supported_configurable_tier(tier):
    settings = HarnessConfig.model_validate(
        {"models": {"registry": {"m": {"provider": "p", "model": "id", "tier": tier}}}}
    )
    assert settings.models.registry["m"].tier == tier


@pytest.mark.parametrize("tier", ["Premium", "unknown", "", 1])
def test_model_tier_rejects_values_outside_supported_tiers(tier):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {
                "models": {
                    "registry": {"m": {"provider": "p", "model": "id", "tier": tier}}
                }
            }
        )


def test_model_tier_remains_optional_for_legacy_models():
    settings = HarnessConfig.model_validate(
        {"models": {"registry": {"m": {"provider": "p", "model": "id"}}}}
    )
    assert settings.models.registry["m"].tier is None


def test_model_fallback_policy_is_bounded_and_validated():
    settings = HarnessConfig.model_validate(
        {
            "models": {
                "routing": {
                    "fallback": {"max_attempts": 3, "base_delay": 0.1, "max_delay": 1.0}
                }
            }
        }
    )
    assert settings.models.routing.fallback.max_attempts == 3
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"routing": {"fallback": {"max_attempts": 9}}}}
        )


@pytest.mark.parametrize(
    "strategy",
    [
        {"primary": "a", "fallback": ["a"]},
        {"primary": "a", "fallback": ["b", "b"]},
        {"primary": " ", "fallback": []},
        {"primary": "a", "fallback": [""]},
    ],
)
def test_profile_model_strategy_rejects_blank_or_duplicate_candidates(strategy):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"profiles": {"coding": {"model": strategy}}})


def test_profile_model_strategy_preserves_primary_and_fallback_order():
    settings = HarnessConfig.model_validate(
        {"profiles": {"coding": {"model": {"primary": "a", "fallback": ["b", "c"]}}}}
    )
    assert settings.profiles["coding"].model.primary == "a"
    assert settings.profiles["coding"].model.fallback == ["b", "c"]


def test_model_routing_policy_validates_all_complexity_preferences():
    settings = HarnessConfig.model_validate(
        {
            "models": {
                "routing": {
                    "tier_preferences": {
                        "low": [
                            "local",
                            "economical",
                            "standard",
                            "advanced",
                            "premium",
                        ],
                        "medium": [
                            "standard",
                            "local",
                            "economical",
                            "advanced",
                            "premium",
                        ],
                        "high": [
                            "advanced",
                            "standard",
                            "premium",
                            "economical",
                            "local",
                        ],
                        "critical": [
                            "premium",
                            "advanced",
                            "standard",
                            "economical",
                            "local",
                        ],
                    },
                    "profile_tier_preferences": {
                        "coding": {
                            "critical": [
                                "premium",
                                "advanced",
                                "standard",
                                "economical",
                                "local",
                            ]
                        }
                    },
                }
            },
            "profiles": {"coding": {"model": {"primary": "model"}}},
        }
    )
    assert [str(tier) for tier in settings.models.routing.tier_preferences["low"]] == [
        "local",
        "economical",
        "standard",
        "advanced",
        "premium",
    ]
    assert (
        settings.models.routing.profile_tier_preferences["coding"]["critical"][0]
        == "premium"
    )


@pytest.mark.parametrize(
    "preferences",
    [
        {
            "low": ["local", "local"],
            "medium": ["standard"],
            "high": ["advanced"],
            "critical": ["premium"],
        },
        {
            "low": ["unknown"],
            "medium": ["standard"],
            "high": ["advanced"],
            "critical": ["premium"],
        },
        {"low": ["local"], "medium": ["standard"], "high": ["advanced"]},
        {"low": []},
    ],
)
def test_model_routing_policy_rejects_duplicate_unknown_or_missing_preferences(
    preferences,
):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"routing": {"tier_preferences": preferences}}}
        )


def test_model_routing_profile_override_requires_known_profile():
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {
                "models": {
                    "routing": {
                        "profile_tier_preferences": {
                            "missing": {
                                "low": [
                                    "local",
                                    "economical",
                                    "standard",
                                    "advanced",
                                    "premium",
                                ]
                            }
                        }
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "payload,path",
    [
        (
            {"models": {"rates": {"m": {"input": -1, "output": 2}}}},
            "models.rates.m.input",
        ),
        (
            {"models": {"rates": {"m": {"input": 1, "output": "bad"}}}},
            "models.rates.m.output",
        ),
        (
            {
                "tools": {
                    "git": {
                        "credentials": {
                            "example.com": {"mode": "bearer", "secret_name": "bad name"}
                        }
                    }
                }
            },
            "tools.git.credentials.example.com",
        ),
        (
            {
                "tools": {
                    "git": {
                        "ssh": {
                            "credentials": {
                                "example.com": {"username": "git", "fingerprint": "bad"}
                            }
                        }
                    }
                }
            },
            "tools.git.ssh.credentials.example.com",
        ),
    ],
)
def test_operational_config_objects_reject_invalid_shapes(payload, path):
    with pytest.raises(ValidationError) as error:
        HarnessConfig.model_validate(payload)
    assert path in str(error.value).replace("`", "")


@pytest.mark.parametrize(
    "credential",
    [
        {"username": "oauth2", "secret_name": "GIT_TOKEN"},
        {"mode": "basic", "username": "oauth2", "secret_name": "GIT_TOKEN"},
        {"mode": "bearer", "secret_name": "GIT_TOKEN"},
    ],
)
def test_operational_config_accepts_supported_credential_modes(credential):
    settings = HarnessConfig.model_validate(
        {
            "models": {"rates": {"model-id": {"input": 1, "output": 2}}},
            "tools": {
                "git": {
                    "credentials": {"example.com": credential},
                    "ssh": {
                        "credentials": {
                            "ssh.example.com": {
                                "username": "git",
                                "fingerprint": "SHA256:" + "A" * 43,
                            }
                        },
                        "host_keys": {"ssh.example.com": ["ssh-ed25519 AAAA"]},
                    },
                }
            },
        }
    )
    assert settings.models.rates["model-id"].input == 1
    assert settings.tools.git.credentials["example.com"].secret_name == "GIT_TOKEN"


@pytest.mark.parametrize(
    "credential",
    [
        {"secret_name": "GIT_TOKEN"},
        {"username": "bad:name", "secret_name": "GIT_TOKEN"},
        {"username": "bad\nname", "secret_name": "GIT_TOKEN"},
        {"mode": "bearer", "username": "extra", "secret_name": "GIT_TOKEN"},
    ],
)
def test_operational_config_rejects_invalid_credential_combinations(credential):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"tools": {"git": {"credentials": {"example.com": credential}}}}
        )


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
