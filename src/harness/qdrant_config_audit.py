"""Read-only, value-redacting audit for operator-managed Qdrant Compose files."""

from pathlib import Path

import yaml

_SECRET_NAMES = {"QDRANT__SERVICE__API_KEY", "QDRANT_API_KEY", "API_KEY"}


def _is_safe_secret_reference(value):
    if not value.startswith("${") or not value.endswith("}"):
        return False
    expression = value[2:-1]
    for separator in (":?", "?"):
        if separator in expression:
            name, message = expression.split(separator, 1)
            return name.upper() in _SECRET_NAMES and bool(message)
    return expression.upper() in _SECRET_NAMES


def _load(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("configuration file must not be a symlink")
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("configuration file is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise TypeError("configuration root must be a mapping")
    return value


def _ports(service):
    ports = service.get("ports", [])
    if not isinstance(ports, list):
        return []
    return ports


def _port_is_unbound(port):
    if isinstance(port, dict):
        host_ip = port.get("host_ip")
        return port.get("published") in {6333, "6333", 6334, "6334"} and host_ip in {
            None,
            "",
            "0.0.0.0",
            "::",
        }
    if not isinstance(port, str):
        return False
    parts = port.split(":")
    if len(parts) == 2:
        return parts[1].split("/")[0] in {"6333", "6334"}
    if len(parts) >= 3:
        return parts[1].split("/")[0] in {"6333", "6334"} and parts[0] in {
            "",
            "0.0.0.0",
            "::",
        }
    return False


def _literal_secret(environment):
    if isinstance(environment, dict):
        entries = environment.items()
    elif isinstance(environment, list):
        entries = (item.split("=", 1) for item in environment if isinstance(item, str))
    else:
        return False
    for entry in entries:
        if len(entry) != 2:
            continue
        name, value = entry
        if (
            str(name).upper() in _SECRET_NAMES
            and isinstance(value, str)
            and value.strip()
            and not _is_safe_secret_reference(value.strip())
        ):
            return True
    return False


def audit_qdrant_compose(compose_path, config_path=None):
    """Return safe finding codes only; never include configuration values."""
    compose = _load(compose_path)
    services = compose.get("services")
    if not isinstance(services, dict):
        raise TypeError("Compose services must be a mapping")
    service = services.get("qdrant")
    if not isinstance(service, dict):
        raise TypeError("Compose qdrant service is missing or invalid")

    findings = []

    def add(code):
        findings.append({"code": code})

    image = service.get("image")
    if not isinstance(image, str) or not image.strip():
        add("image_missing")
    elif "@sha256:" not in image and (
        image.rsplit("/", 1)[-1].endswith(":latest")
        or ":" not in image.rsplit("/", 1)[-1]
    ):
        add("image_not_pinned")

    if any(_port_is_unbound(port) for port in _ports(service)):
        add("qdrant_port_unbound")
    if _literal_secret(service.get("environment")):
        add("literal_api_key")

    volumes = service.get("volumes", [])
    if isinstance(volumes, list):
        for mount in volumes:
            if (
                isinstance(mount, str)
                and mount.split(":")[0].endswith("config")
                and (mount.count(":") < 2 or mount.rsplit(":", 1)[-1] != "ro")
            ):
                add("config_mount_writable")
                break
            if (
                isinstance(mount, dict)
                and mount.get("target") == "/qdrant/config"
                and mount.get("read_only") is not True
            ):
                add("config_mount_writable")
                break

    if config_path is not None:
        config = _load(config_path)
        cors = config.get("service", {})
        if isinstance(cors, dict) and cors.get("enable_cors") is True:
            add("cors_enabled_review")

    return {"healthy": not findings, "findings": findings}
