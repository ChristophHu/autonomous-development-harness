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
    assert settings.harness.max_correction_attempts == 2
    assert settings.harness.max_correction_elapsed_seconds == 1800
    assert ConfigurationService(sample).validate()


def test_metrics_listener_defaults_to_disabled_and_accepts_explicit_authenticated_setup():
    defaults = HarnessConfig.model_validate({})
    assert defaults.api.metrics_listener.enabled is False
    assert defaults.api.metrics_listener.host == "127.0.0.1"
    enabled = HarnessConfig.model_validate(
        {
            "api": {
                "metrics_listener": {
                    "enabled": True,
                    "host": "0.0.0.0",
                    "port": 9091,
                    "bearer_file": "/private/tmp/metrics-token",
                }
            }
        }
    )
    assert enabled.api.metrics_listener.enabled is True
    assert enabled.api.metrics_listener.host == "0.0.0.0"


@pytest.mark.parametrize(
    "api",
    [
        {
            "port": 8080,
            "metrics_listener": {
                "enabled": True,
                "port": 8080,
                "bearer_file": "/tmp/metrics-token",
            },
        },
        {
            "port": 9091,
            "metrics_listener": {
                "enabled": True,
                "port": 9091,
                "bearer_file": "/tmp/metrics-token",
            },
        },
        {"metrics_listener": {"host": "192.168.1.10"}},
        {"metrics_listener": {"enabled": True, "bearer_file": ""}},
        {"metrics_listener": {"port": True}},
        {"metrics_listener": {"enabled": True}},
    ],
)
def test_metrics_listener_rejects_unsafe_or_invalid_settings(api):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"api": api})


@pytest.mark.parametrize(
    "budget",
    [
        {"calibration_min_samples": 0},
        {"calibration_min_samples": True},
        {"calibration_max_multiplier": 0.9},
        {"calibration_max_multiplier": 2.1},
    ],
)
def test_token_calibration_limits_are_strict_and_bounded(budget):
    payload = {
        "models": {
            "input_token_budgets": {
                "model": {
                    "characters_per_token": 3.5,
                    "max_input_tokens": 1000,
                    **budget,
                }
            }
        }
    }
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(payload)


def test_configuration_extensions_are_explicit_and_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HarnessConfig.model_validate({"harness": {"untyped_option": True}})

    settings = HarnessConfig.model_validate(
        {
            "extensions": {"vendor": {"mode": "custom"}},
            "harness": {"extensions": {"label": "local"}},
        }
    )

    assert settings.extensions == {"vendor": {"mode": "custom"}}
    assert settings.harness.extensions == {"label": "local"}


def test_default_configuration_path_can_be_overridden_for_isolated_tests(
    tmp_path, monkeypatch
):
    path = tmp_path / "isolated-config.yaml"
    path.write_text("harness:\n  name: isolated-test-config\n")
    monkeypatch.setenv("HARNESS_CONFIG_PATH", str(path))

    config = ConfigurationService()

    assert config._path == path
    assert config.resolved()["harness"]["name"] == "isolated-test-config"


def test_known_path_settings_may_be_omitted_after_defaults_are_loaded():
    config = ConfigurationService()
    config.data["paths"].pop("logs")

    assert config.validate()


def test_provider_kind_is_explicit_and_validated():
    settings = HarnessConfig.model_validate(
        {"models": {"providers": {"local": {"kind": "lmstudio"}}}}
    )
    assert settings.models.providers["local"].kind == "lmstudio"
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"providers": {"local": {"kind": "unknown"}}}}
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "secret"},
        {"Cookie": "session"},
        {"X-Title": "bad\r\ninjected: yes"},
        {"X-Other": "value"},
    ],
)
def test_provider_custom_headers_are_restricted_to_safe_metadata(headers):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"providers": {"p": {"headers": headers}}}}
        )


def test_provider_accepts_safe_metadata_headers_and_rejects_them_for_lmstudio():
    settings = HarnessConfig.model_validate(
        {
            "models": {
                "providers": {
                    "p": {
                        "headers": {
                            "HTTP-Referer": "https://harness.test",
                            "X-Title": "Harness",
                        }
                    }
                }
            }
        }
    )
    assert settings.models.providers["p"].headers["X-Title"] == "Harness"
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {
                "models": {
                    "providers": {
                        "p": {"kind": "lmstudio", "headers": {"X-Title": "Harness"}}
                    }
                }
            }
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


@pytest.mark.parametrize(
    "server",
    [
        {
            "command": ["mcp"],
            "allow_tools": ["read"],
            "trusted_local": True,
            "read_roots": [""],
        },
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test/rpc",
            "allowed_hosts": ["mcp.example.test"],
            "allow_tools": ["read"],
            "read_roots": ["/tmp"],
        },
        {"builtin": "filesystem", "read_roots": ["/tmp"]},
    ],
)
def test_mcp_read_roots_are_transport_and_builtin_scoped(server):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"tools": {"mcp": {"servers": {"s": server}}}})


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


def test_model_input_token_budget_accepts_one_explicit_local_tokenizer():
    settings = HarnessConfig.model_validate(
        {
            "models": {
                "input_token_budgets": {
                    "model-id": {
                        "encoding": "cl100k_base",
                        "max_input_tokens": 12000,
                        "framing_tokens": 256,
                    }
                }
            }
        }
    )
    assert settings.models.input_token_budgets["model-id"].max_input_tokens == 12000
    assert settings.models.input_token_budgets["model-id"].framing_tokens == 256
    assert settings.models.input_token_budgets["model-id"].safety_margin_percent == 20
    approximate = HarnessConfig.model_validate(
        {
            "models": {
                "input_token_budgets": {
                    "approx": {
                        "characters_per_token": 3.5,
                        "max_input_tokens": 4000,
                        "safety_margin_percent": 30,
                    }
                }
            }
        }
    )
    assert approximate.models.input_token_budgets["approx"].characters_per_token == 3.5
    assert approximate.models.input_token_budgets["approx"].safety_margin_percent == 30
    local = HarnessConfig.model_validate(
        {
            "models": {
                "input_token_budgets": {
                    "local-model": {
                        "tokenizer_file": "/models/local/tokenizer.json",
                        "max_input_tokens": 8000,
                    }
                }
            }
        }
    )
    assert local.models.input_token_budgets["local-model"].tokenizer_file.endswith(
        "tokenizer.json"
    )


@pytest.mark.parametrize(
    "budget",
    [
        {"max_input_tokens": 10},
        {"encoding": "x", "tokenizer_file": "local.json", "max_input_tokens": 10},
        {"encoding": "x", "max_input_tokens": 0},
        {"encoding": "x", "max_input_tokens": 10, "framing_tokens": -1},
        {"tokenizer_file": "bad\x00path", "max_input_tokens": 10},
        {"characters_per_token": 0, "max_input_tokens": 10},
        {"characters_per_token": 4, "encoding": "x", "max_input_tokens": 10},
        {
            "characters_per_token": 4,
            "max_input_tokens": 10,
            "safety_margin_percent": 101,
        },
    ],
)
def test_model_input_token_budget_rejects_incomplete_or_unsafe_settings(budget):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"models": {"input_token_budgets": {"model-id": budget}}}
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
    "payload",
    [
        {"agents": {"contract_mode": "sometimes"}},
        {"agents": {"contract_mode": True}},
        {"profiles": {"p": {"model": {"primary": "local"}, "contract_mode": "loose"}}},
    ],
)
def test_agent_contract_modes_are_strictly_typed(payload):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(payload)


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
    assert settings.harness.max_parallel_steps == 4
    assert settings.models.providers["openai"].retry.max_attempts == 3
    assert settings.models.registry["openai_sol"].model == "vendor-model-id"
    assert settings.profiles["coding"].model.primary == "openai_sol"
    assert settings.testing.coverage.branches == 100


@pytest.mark.parametrize("value", [0, 33, True, "4"])
def test_parallel_step_limit_is_strict_and_bounded(value):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"harness": {"max_parallel_steps": value}})


@pytest.mark.parametrize("value", [-1, 11, True, "2"])
def test_correction_attempt_limit_is_strict_and_bounded(value):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({"harness": {"max_correction_attempts": value}})


@pytest.mark.parametrize("value", [0, 86401, True, "30"])
def test_correction_elapsed_budget_is_strict_and_bounded(value):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"harness": {"max_correction_elapsed_seconds": value}}
        )


@pytest.mark.parametrize("value", [0, True, "5"])
def test_correction_input_token_budget_is_strict(value):
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(
            {"harness": {"max_correction_input_tokens": value}}
        )


def test_typed_config_preserves_extensions_at_root_and_known_sections():
    settings = HarnessConfig.model_validate(
        {
            "extensions": {"x_org_extension": {"mode": "custom"}},
            "models": {
                "extensions": {"x_registry_source": "catalog"},
                "providers": {
                    "local": {"enabled": False, "extensions": {"x_transport": "unix"}}
                },
            },
            "profiles": {
                "coding": {"model": {}, "extensions": {"x_prompt_revision": 4}}
            },
        }
    )

    assert settings.extensions["x_org_extension"] == {"mode": "custom"}
    assert settings.models.extensions["x_registry_source"] == "catalog"
    assert settings.models.providers["local"].extensions["x_transport"] == "unix"
    assert settings.profiles["coding"].extensions["x_prompt_revision"] == 4


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
                "paths": {
                    "workspace": "./initial",
                    "extensions": {"extension_data": "./extra"},
                },
                "api": {"port": 8090},
            }
        )
    )
    config = ConfigurationService(path)

    assert config.settings.api.port == 8090
    assert config.path("workspace") == tmp_path / "initial"
    assert config.settings.paths.extensions["extension_data"] == "./extra"
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
            {"profiles": {"coding": {"capabilities": ["superuser"]}}},
            "profiles.coding.capabilities.0",
        ),
        (
            {"models": {"registry": {"x": {"provider": 1, "model": "m"}}}},
            "models.registry.x.provider",
        ),
        (
            {"memory": {"embeddings": {"batch_size": 257}}},
            "memory.embeddings.batch_size",
        ),
        ({"memory": {"context": {"max_bytes": 255}}}, "memory.context.max_bytes"),
        (
            {"memory": {"context": {"max_bytes": 1048577}}},
            "memory.context.max_bytes",
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


def test_memory_context_budget_is_typed_and_defaulted():
    assert HarnessConfig.model_validate({}).memory.context.max_bytes == 65536
    assert (
        HarnessConfig.model_validate(
            {"memory": {"context": {"max_bytes": 2048}}}
        ).memory.context.max_bytes
        == 2048
    )
