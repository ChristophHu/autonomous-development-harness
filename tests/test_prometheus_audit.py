from harness.prometheus_audit import audit_prometheus_scrape, audit_prometheus_ui


def scrape_config():
    return {
        "scrape_configs": [
            {
                "job_name": "harness",
                "metrics_path": "/metrics/prometheus",
                "authorization": {
                    "type": "Bearer",
                    "credentials_file": "/run/secrets/harness_metrics_token",
                },
                "static_configs": [{"targets": ["host.docker.internal:9091"]}],
            }
        ]
    }


def test_prometheus_scrape_accepts_documented_bridge_contract_without_echoing_secret():
    config = scrape_config()
    config["scrape_configs"][0]["authorization"]["credentials_file"] = "TOKEN-CANARY"
    result = audit_prometheus_scrape(config)
    assert result == {"status": "valid", "findings": []}
    assert "TOKEN-CANARY" not in str(result)


def test_prometheus_scrape_reports_invalid_root_and_missing_job_configuration():
    assert audit_prometheus_scrape(None) == {
        "status": "unknown",
        "findings": ["PROM_SCRAPE_CONFIG_INVALID"],
    }
    assert audit_prometheus_scrape({}) == {
        "status": "invalid",
        "findings": ["PROM_SCRAPE_JOBS_MISSING"],
    }
    assert audit_prometheus_scrape(
        {"scrape_configs": [None, {"job_name": "other"}]}
    ) == {
        "status": "invalid",
        "findings": ["PROM_SCRAPE_JOB_MISSING"],
    }


def test_prometheus_scrape_rejects_duplicate_harness_jobs():
    config = scrape_config()
    config["scrape_configs"].append(config["scrape_configs"][0])
    assert audit_prometheus_scrape(config) == {
        "status": "invalid",
        "findings": ["PROM_SCRAPE_JOB_DUPLICATED"],
    }


def test_prometheus_scrape_detects_path_auth_and_target_drift():
    config = scrape_config()
    job = config["scrape_configs"][0]
    job["metrics_path"] = "/api/metrics/prometheus"
    job["authorization"] = {"type": "Basic", "credentials": "SECRET-CANARY"}
    job["static_configs"] = [{"targets": ["127.0.0.1:8080"]}]
    result = audit_prometheus_scrape(config)
    assert result == {
        "status": "invalid",
        "findings": [
            "PROM_SCRAPE_PATH_MISMATCH",
            "PROM_SCRAPE_CREDENTIAL_FILE_INVALID",
            "PROM_SCRAPE_TARGET_MISMATCH",
        ],
    }
    assert "SECRET-CANARY" not in str(result)


def test_prometheus_scrape_rejects_missing_or_blank_credentials_file():
    for authorization in (None, {"type": "Bearer"}, {"credentials_file": "  "}):
        config = scrape_config()
        config["scrape_configs"][0]["authorization"] = authorization
        assert (
            "PROM_SCRAPE_CREDENTIAL_FILE_INVALID"
            in audit_prometheus_scrape(config)["findings"]
        )


def test_prometheus_scrape_rejects_non_list_target_shapes():
    for static_configs in (None, [None, {"targets": "host.docker.internal:9091"}]):
        config = scrape_config()
        config["scrape_configs"][0]["static_configs"] = static_configs
        assert (
            "PROM_SCRAPE_TARGET_MISMATCH" in audit_prometheus_scrape(config)["findings"]
        )


def test_cli_scrape_audit_reports_only_safe_codes_and_sanitizes_parse_errors(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from harness.cli import app

    path = tmp_path / "prometheus.yml"
    path.write_text(yaml.safe_dump(scrape_config()), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["observability", "audit-prometheus-scrape", "--scrape-config-file", str(path)],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "valid"' in result.output
    assert "/run/secrets/" not in result.output

    path.write_text("scrape_configs: [SECRET-CANARY\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["observability", "audit-prometheus-scrape", "--scrape-config-file", str(path)],
    )
    assert result.exit_code == 2
    assert "SECRET-CANARY" not in result.output
    assert "ParserError" in result.output

    invalid = scrape_config()
    invalid["scrape_configs"][0]["static_configs"][0]["targets"] = ["127.0.0.1:8080"]
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "observability",
            "audit-prometheus-scrape",
            "--scrape-config-file",
            str(path),
        ],
    )
    assert result.exit_code == 1
    assert "PROM_SCRAPE_TARGET_MISMATCH" in result.output


def secure_compose():
    return {
        "services": {
            "prometheus": {
                "ports": ["0.0.0.0:9090:9090"],
                "command": ["--web.config.file=/etc/prometheus/web.yml"],
                "volumes": [
                    {
                        "type": "bind",
                        "source": "./web.yml",
                        "target": "/etc/prometheus/web.yml",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": "./tls",
                        "target": "/etc/prometheus/tls",
                        "read_only": True,
                    },
                ],
            }
        }
    }


def secure_web_config():
    return {
        "tls_server_config": {
            "cert_file": "/etc/prometheus/tls/cert.pem",
            "key_file": "/etc/prometheus/tls/key.pem",
        },
        "basic_auth_users": {"operator": "$2b$12$hashed-value"},
    }


def test_public_ui_requires_mounted_readonly_web_config_with_tls_and_auth():
    compose = secure_compose()
    compose["services"]["prometheus"]["volumes"].insert(0, 42)
    result = audit_prometheus_ui(compose, secure_web_config())
    assert result == {"status": "protected", "findings": []}


def test_public_ui_without_protection_reports_only_safe_finding_codes():
    result = audit_prometheus_ui(
        {"services": {"prometheus": {"ports": ["9090:9090"]}}}, {}
    )
    assert result == {
        "status": "unsafe",
        "findings": [
            "PROM_UI_TLS_MISSING",
            "PROM_UI_AUTH_MISSING",
            "PROM_UI_WEB_CONFIG_NOT_LOADED",
        ],
    }


def test_loopback_ui_does_not_require_remote_authentication():
    result = audit_prometheus_ui(
        {
            "services": {
                "prometheus": {
                    "network_mode": "host",
                    "command": ["--web.listen-address=127.0.0.1:9090"],
                }
            }
        },
        {},
    )
    assert result == {"status": "loopback_only", "findings": []}


def test_public_ui_requires_readonly_web_config_bind_mount():
    compose = secure_compose()
    compose["services"]["prometheus"]["volumes"][0]["read_only"] = False
    result = audit_prometheus_ui(compose, secure_web_config())
    assert "PROM_UI_WEB_CONFIG_NOT_READONLY" in result["findings"]


def test_public_ui_detects_host_network_listener_exposure():
    compose = {
        "services": {
            "prometheus": {
                "network_mode": "host",
                "command": [
                    "--web.listen-address=0.0.0.0:9090",
                    "--web.config.file=/etc/prometheus/web.yml",
                ],
                "volumes": secure_compose()["services"]["prometheus"]["volumes"],
            }
        }
    }
    result = audit_prometheus_ui(compose, secure_web_config())
    assert result == {"status": "protected", "findings": []}


def test_missing_compose_service_is_reported_safely():
    assert audit_prometheus_ui({}, {}) == {
        "status": "unknown",
        "findings": ["PROM_UI_SERVICE_MISSING"],
    }


def test_command_options_accept_separate_values_and_mount_strings():
    compose = {
        "services": {
            "prometheus": {
                "ports": ["127.0.0.1:9090:9090", "9090:9090"],
                "command": [
                    "--web.listen-address",
                    "0.0.0.0:9090",
                    "--web.config.file",
                    "/etc/prometheus/web.yml",
                ],
                "volumes": [
                    "./web.yml:/etc/prometheus/web.yml:ro",
                    "./tls:/etc/prometheus/tls:ro",
                ],
            }
        }
    }
    assert audit_prometheus_ui(compose, secure_web_config()) == {
        "status": "protected",
        "findings": [],
    }


def test_unset_or_empty_values_generate_tls_and_auth_findings():
    compose = {
        "services": {
            "prometheus": {
                "ports": [{"target": 9090}],
                "command": ["--web.config.file=/etc/prometheus/web.yml"],
                "volumes": ["./web.yml:/etc/prometheus/web.yml:rw"],
            }
        }
    }
    result = audit_prometheus_ui(
        compose,
        {
            "tls_server_config": {"cert_file": "", "key_file": None},
            "basic_auth_users": {"": "", "operator": 1},
        },
    )
    assert result["findings"] == [
        "PROM_UI_TLS_MISSING",
        "PROM_UI_AUTH_MISSING",
        "PROM_UI_WEB_CONFIG_NOT_READONLY",
    ]


def test_plain_password_is_not_accepted_as_bcrypt_hash():
    web_config = secure_web_config()
    web_config["basic_auth_users"]["operator"] = "plaintext-canary"
    result = audit_prometheus_ui(secure_compose(), web_config)
    assert "PROM_UI_AUTH_MISSING" in result["findings"]


def test_tls_file_mount_must_be_read_only():
    compose = secure_compose()
    compose["services"]["prometheus"]["volumes"][1]["read_only"] = False
    result = audit_prometheus_ui(compose, secure_web_config())
    assert "PROM_UI_TLS_FILES_NOT_READONLY" in result["findings"]


def test_unknown_volume_and_command_shapes_are_treated_as_unprotected():
    compose = {
        "services": {
            "prometheus": {
                "network_mode": "host",
                "command": "--web.listen-address=0.0.0.0:9090 --web.config.file",
                "volumes": None,
            }
        }
    }
    result = audit_prometheus_ui(compose, secure_web_config())
    assert result["findings"] == [
        "PROM_UI_TLS_FILES_NOT_READONLY",
        "PROM_UI_WEB_CONFIG_NOT_LOADED",
    ]


def test_localhost_and_ipv4_loopback_are_recognized():
    for host in ("localhost", "127.0.0.1", "127.8.9.10"):
        compose = {
            "services": {
                "prometheus": {
                    "ports": [f"{host}:9090:9090"],
                    "command": [],
                }
            }
        }
        assert audit_prometheus_ui(compose, {})["status"] == "loopback_only"


def test_ipv6_loopback_port_and_dynamic_public_port_are_classified():
    for ports, expected in (
        (["[::1]:9090:9090"], "loopback_only"),
        ([{"target": 9090, "host_ip": "::1"}], "loopback_only"),
        (["9090"], "unsafe"),
    ):
        compose = {
            "services": {
                "prometheus": {
                    "ports": ports,
                    "command": [],
                }
            }
        }
        assert audit_prometheus_ui(compose, {})["status"] == expected


def test_non_mapping_inputs_and_non_matching_published_ports_fail_closed():
    assert audit_prometheus_ui(None, {})["status"] == "unknown"
    compose = {
        "services": {
            "prometheus": {
                "ports": [{"target": 8080, "host_ip": "0.0.0.0"}, 9090],
                "command": [],
            }
        }
    }
    assert audit_prometheus_ui(compose, {})["status"] == "loopback_only"


def test_missing_config_mount_and_loopback_mapping_are_reported_correctly():
    compose = {
        "services": {
            "prometheus": {
                "ports": [
                    "127.0.0.1:9090:9090",
                    {"target": 9090, "host_ip": "127.0.0.1"},
                    "9091:9091",
                ],
                "command": ["--web.config.file=/etc/prometheus/web.yml"],
                "volumes": None,
            }
        }
    }
    result = audit_prometheus_ui(compose, {})
    assert result == {"status": "loopback_only", "findings": []}


def test_mount_state_reports_a_list_without_the_requested_target_as_missing():
    compose = {
        "services": {
            "prometheus": {
                "ports": ["9090:9090"],
                "command": ["--web.config.file=/etc/prometheus/web.yml"],
                "volumes": ["./data:/prometheus:rw", 42],
            }
        }
    }
    result = audit_prometheus_ui(compose, {})
    assert "PROM_UI_WEB_CONFIG_NOT_MOUNTED" in result["findings"]


def test_public_ui_without_web_config_mount_is_reported():
    compose = {
        "services": {
            "prometheus": {
                "ports": ["9090:9090"],
                "command": ["--web.config.file=/etc/prometheus/web.yml"],
                "volumes": None,
            }
        }
    }
    result = audit_prometheus_ui(compose, {})
    assert "PROM_UI_WEB_CONFIG_NOT_MOUNTED" in result["findings"]


def test_cli_audit_reads_yaml_and_never_echoes_auth_values(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from harness.cli import app

    compose_path = tmp_path / "compose.yml"
    web_path = tmp_path / "web.yml"
    compose_path.write_text(yaml.safe_dump(secure_compose()), encoding="utf-8")
    web = secure_web_config()
    web_path.write_text(yaml.safe_dump(web), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "observability",
            "audit-prometheus-ui",
            "--compose-file",
            str(compose_path),
            "--web-config-file",
            str(web_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "protected"' in result.output
    assert "hashed-value" not in result.output


def test_cli_audit_returns_failure_for_public_unauthenticated_ui(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from harness.cli import app

    compose_path = tmp_path / "compose.yml"
    web_path = tmp_path / "web.yml"
    compose_path.write_text(
        yaml.safe_dump({"services": {"prometheus": {"ports": ["9090:9090"]}}}),
        encoding="utf-8",
    )
    web_path.write_text("{}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "observability",
            "audit-prometheus-ui",
            "--compose-file",
            str(compose_path),
            "--web-config-file",
            str(web_path),
        ],
    )
    assert result.exit_code == 1
    assert "PROM_UI_TLS_MISSING" in result.output


def test_cli_audit_sanitizes_invalid_yaml_error(tmp_path):
    from typer.testing import CliRunner

    from harness.cli import app

    compose_path = tmp_path / "compose.yml"
    web_path = tmp_path / "web.yml"
    compose_path.write_text("service: [SECRET-CANARY\n", encoding="utf-8")
    web_path.write_text("{}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "observability",
            "audit-prometheus-ui",
            "--compose-file",
            str(compose_path),
            "--web-config-file",
            str(web_path),
        ],
    )
    assert result.exit_code == 2
    assert "SECRET-CANARY" not in result.output
    assert "ParserError" in result.output
