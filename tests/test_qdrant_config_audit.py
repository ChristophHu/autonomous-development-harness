import json

import pytest

from harness.qdrant_config_audit import audit_qdrant_compose


def write(path, value):
    path.write_text(value)
    return path


def test_audit_clean_compose_and_config(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        """services:
  qdrant:
    image: qdrant/qdrant:v1.19.0
    ports: [\"127.0.0.1:6333:6333\"]
    environment: [\"QDRANT__SERVICE__API_KEY=${QDRANT__SERVICE__API_KEY:?required}\"]
    volumes: [\"./config:/qdrant/config:ro\"]
""",
    )
    config = write(tmp_path / "production.yml", "service:\n  enable_cors: false\n")
    assert audit_qdrant_compose(compose, config) == {"healthy": True, "findings": []}


@pytest.mark.parametrize(
    ("compose_text", "config_text", "expected"),
    [
        (
            'services:\n  qdrant:\n    image: qdrant/qdrant:latest\n    ports: ["6333:6333"]\n',
            "service:\n  enable_cors: true\n",
            {"image_not_pinned", "qdrant_port_unbound", "cors_enabled_review"},
        ),
        (
            "services:\n  qdrant:\n    environment:\n      QDRANT__SERVICE__API_KEY: do-not-print-this\n",
            None,
            {"image_missing", "literal_api_key"},
        ),
        (
            'services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    volumes: ["./config:/qdrant/config"]\n',
            None,
            {"config_mount_writable"},
        ),
        (
            "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    ports: [{target: 6333, published: 6333}]\n",
            None,
            {"qdrant_port_unbound"},
        ),
        (
            'services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    environment: ["QDRANT__SERVICE__API_KEY=another-secret"]\n',
            None,
            {"literal_api_key"},
        ),
        (
            'services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    environment: ["QDRANT__SERVICE__API_KEY=${QDRANT__SERVICE__API_KEY:-fallback-secret}"]\n',
            None,
            {"literal_api_key"},
        ),
    ],
)
def test_audit_reports_only_safe_finding_codes(
    tmp_path, compose_text, config_text, expected
):
    compose = write(tmp_path / "compose.yml", compose_text)
    config = write(tmp_path / "config.yml", config_text) if config_text else None
    report = audit_qdrant_compose(compose, config)
    assert {finding["code"] for finding in report["findings"]} == expected
    assert report["healthy"] is False
    assert "secret" not in json.dumps(report).lower()


@pytest.mark.parametrize(
    "compose_text",
    [
        "invalid: [yaml",
        "[]",
        "services: []",
        "services:\n  other: {}\n",
        "services:\n  qdrant: []\n",
    ],
)
def test_invalid_compose_fails_closed(tmp_path, compose_text):
    compose = write(tmp_path / "compose.yml", compose_text)
    with pytest.raises((TypeError, ValueError)):
        audit_qdrant_compose(compose)


@pytest.mark.parametrize("config_text", ["not: [yaml", "[]"])
def test_invalid_optional_config_fails_closed(tmp_path, config_text):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n",
    )
    config = write(tmp_path / "config.yml", config_text)
    with pytest.raises((TypeError, ValueError)):
        audit_qdrant_compose(compose, config)


def test_symlink_is_rejected(tmp_path):
    source = write(tmp_path / "source.yml", "services: {}\n")
    link = tmp_path / "link.yml"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        audit_qdrant_compose(link)


def test_unbound_ipv6_and_non_qdrant_ports_are_not_misclassified(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        """services:
  qdrant:
    image: qdrant/qdrant:v1.19.0
    ports: [\"[::1]:6333:6333\", \"8080:8080\"]
""",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


def test_audit_handles_unsupported_shapes_without_false_findings(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        """services:
  qdrant:
    image: qdrant/qdrant@sha256:0123456789abcdef
    ports: false
    environment: false
    volumes: false
""",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


@pytest.mark.parametrize(
    "port",
    [7, "6333", "8080:8080", "127.0.0.1:6333:6333", "0.0.0.0:8080:8080"],
)
def test_port_parser_ignores_nonmatching_or_loopback_ports(tmp_path, port):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    ports:\n      - "
        + json.dumps(port)
        + "\n",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


def test_secret_parser_ignores_empty_reference_and_malformed_environment(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        """services:
  qdrant:
    image: qdrant/qdrant:v1.19.0
    environment: [\"QDRANT__SERVICE__API_KEY=\", 12, \"BROKEN\"]
""",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


def test_secret_parser_ignores_non_string_values(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n"
        "    environment:\n      QDRANT__SERVICE__API_KEY: 17\n",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


@pytest.mark.parametrize(
    "reference",
    [
        "${QDRANT__SERVICE__API_KEY}",
        "${QDRANT__SERVICE__API_KEY:?required}",
        "${QDRANT__SERVICE__API_KEY?required}",
    ],
)
def test_secret_parser_accepts_required_environment_references(tmp_path, reference):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n"
        f'    environment: ["QDRANT__SERVICE__API_KEY={reference}"]\n',
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


def test_example_secret_is_blank_and_env_file_is_gitignored():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    key_lines = [
        line.partition("=")[2]
        for line in (root / ".env.example").read_text().splitlines()
        if line.startswith("QDRANT__SERVICE__API_KEY=")
    ]
    assert key_lines == [""]
    assert ".env" in (root / ".gitignore").read_text().splitlines()


@pytest.mark.parametrize(
    "mount",
    [
        "data:/qdrant/storage",
        {
            "type": "bind",
            "source": "./config",
            "target": "/qdrant/config",
            "read_only": True,
        },
    ],
)
def test_mount_parser_accepts_storage_and_readonly_config(tmp_path, mount):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n    volumes:\n      - "
        + json.dumps(mount)
        + "\n",
    )
    assert audit_qdrant_compose(compose) == {"healthy": True, "findings": []}


def test_mount_parser_detects_long_syntax_writable_config(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n"
        "    volumes:\n      - {type: bind, source: ./config, target: /qdrant/config}\n",
    )
    report = audit_qdrant_compose(compose)
    assert report["findings"] == [{"code": "config_mount_writable"}]


def test_cors_audit_ignores_missing_or_invalid_service_mapping(tmp_path):
    compose = write(
        tmp_path / "compose.yml",
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\n",
    )
    config = write(tmp_path / "config.yml", "service: []\n")
    assert audit_qdrant_compose(compose, config) == {"healthy": True, "findings": []}
