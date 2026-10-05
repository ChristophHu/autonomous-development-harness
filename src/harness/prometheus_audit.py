"""Read-only safety audit for an external Prometheus web UI configuration."""

from __future__ import annotations


def _command_items(command):
    if isinstance(command, str):
        return command.split()
    if isinstance(command, list):
        return [str(item) for item in command]
    return []


def _listen_address(command):
    for item in _command_items(command):
        if item.startswith("--web.listen-address="):
            return item.split("=", 1)[1]
        if item == "--web.listen-address":
            items = _command_items(command)
            index = items.index(item)
            return items[index + 1] if index + 1 < len(items) else ""
    return "0.0.0.0:9090"


def _web_config_path(command):
    items = _command_items(command)
    for index, item in enumerate(items):
        if item.startswith("--web.config.file="):
            return item.split("=", 1)[1]
        if item == "--web.config.file" and index + 1 < len(items):
            return items[index + 1]
    return None


def _is_loopback(value):
    if value.startswith("[") and "]" in value:
        host = value[1 : value.index("]")]
    elif value.count(":") > 1:
        host = value
    else:
        host = value.rsplit(":", 1)[0] if ":" in value else value
    return host.casefold() == "localhost" or host.startswith("127.") or host == "::1"


def _published_externally(ports):
    if not isinstance(ports, list):
        return False
    for port in ports:
        if isinstance(port, str):
            parts = port.rsplit(":", 2)
            if len(parts) == 1 and parts[0].split("/", 1)[0] == "9090":
                return True
            container_port = parts[-1].split("/", 1)[0]
            if (
                len(parts) == 3
                and container_port == "9090"
                and not _is_loopback(parts[0])
            ):
                return True
            if len(parts) == 2 and container_port == "9090":
                return True
        elif isinstance(port, dict) and port.get("target") == 9090:
            host_ip = port.get("host_ip", "0.0.0.0")
            if not _is_loopback(str(host_ip)):
                return True
    return False


def _config_mount_state(volumes, target):
    if not isinstance(volumes, list):
        return "missing"
    for volume in volumes:
        if isinstance(volume, dict) and volume.get("target") == target:
            return "readonly" if volume.get("read_only") is True else "writable"
        if isinstance(volume, str):
            parts = volume.split(":")
            if len(parts) >= 2 and parts[1] == target:
                return "readonly" if "ro" in parts[2:] else "writable"
    return "missing"


def _readonly_mount_covers(volumes, path):
    if not isinstance(volumes, list):
        return False
    for volume in volumes:
        if isinstance(volume, dict):
            target = volume.get("target")
            readonly = volume.get("read_only") is True
        elif isinstance(volume, str):
            parts = volume.split(":")
            target = parts[1] if len(parts) >= 2 else None
            readonly = "ro" in parts[2:]
        else:
            continue
        if (
            readonly
            and isinstance(target, str)
            and (path == target or path.startswith(target.rstrip("/") + "/"))
        ):
            return True
    return False


def audit_prometheus_ui(compose, web_config):
    """Return status and non-sensitive finding codes; never echo config values."""
    services = compose.get("services") if isinstance(compose, dict) else None
    service = services.get("prometheus") if isinstance(services, dict) else None
    if not isinstance(service, dict):
        return {"status": "unknown", "findings": ["PROM_UI_SERVICE_MISSING"]}

    command = service.get("command")
    host_network = service.get("network_mode") == "host"
    address = _listen_address(command)
    loopback = _is_loopback(address)
    published = _published_externally(service.get("ports"))
    exposed = (host_network and not loopback) or published
    if not exposed:
        return {"status": "loopback_only", "findings": []}

    findings = []
    tls = web_config.get("tls_server_config") if isinstance(web_config, dict) else None
    if not isinstance(tls, dict) or not tls.get("cert_file") or not tls.get("key_file"):
        findings.append("PROM_UI_TLS_MISSING")
    elif not all(
        _readonly_mount_covers(service.get("volumes"), tls[path_key])
        for path_key in ("cert_file", "key_file")
    ):
        findings.append("PROM_UI_TLS_FILES_NOT_READONLY")
    auth = web_config.get("basic_auth_users") if isinstance(web_config, dict) else None
    if (
        not isinstance(auth, dict)
        or not auth
        or any(
            not isinstance(user, str)
            or not user
            or not isinstance(hashed, str)
            or not hashed.startswith(("$2a$", "$2b$", "$2y$"))
            for user, hashed in auth.items()
        )
    ):
        findings.append("PROM_UI_AUTH_MISSING")

    config_path = _web_config_path(command)
    if config_path is None:
        findings.append("PROM_UI_WEB_CONFIG_NOT_LOADED")
    else:
        mount_state = _config_mount_state(service.get("volumes"), config_path)
        if mount_state == "missing":
            findings.append("PROM_UI_WEB_CONFIG_NOT_MOUNTED")
        elif mount_state != "readonly":
            findings.append("PROM_UI_WEB_CONFIG_NOT_READONLY")
    return {
        "status": "protected" if not findings else "unsafe",
        "findings": findings,
    }


def audit_prometheus_scrape(scrape_config):
    """Check the documented Docker Desktop scrape contract without echoing secrets."""
    if not isinstance(scrape_config, dict):
        return {"status": "unknown", "findings": ["PROM_SCRAPE_CONFIG_INVALID"]}
    jobs = scrape_config.get("scrape_configs")
    if not isinstance(jobs, list):
        return {"status": "invalid", "findings": ["PROM_SCRAPE_JOBS_MISSING"]}
    matches = [
        job
        for job in jobs
        if isinstance(job, dict) and job.get("job_name") == "harness"
    ]
    if not matches:
        return {"status": "invalid", "findings": ["PROM_SCRAPE_JOB_MISSING"]}
    if len(matches) != 1:
        return {"status": "invalid", "findings": ["PROM_SCRAPE_JOB_DUPLICATED"]}

    job = matches[0]
    findings = []
    if job.get("metrics_path", "/metrics") != "/metrics/prometheus":
        findings.append("PROM_SCRAPE_PATH_MISMATCH")
    authorization = job.get("authorization")
    if (
        not isinstance(authorization, dict)
        or str(authorization.get("type", "Bearer")).casefold() != "bearer"
        or not isinstance(authorization.get("credentials_file"), str)
        or not authorization["credentials_file"].strip()
        or "credentials" in authorization
    ):
        findings.append("PROM_SCRAPE_CREDENTIAL_FILE_INVALID")

    targets = []
    static_configs = job.get("static_configs")
    if isinstance(static_configs, list):
        for static_config in static_configs:
            if isinstance(static_config, dict):
                configured_targets = static_config.get("targets")
                if isinstance(configured_targets, list):
                    targets.extend(configured_targets)
    if "host.docker.internal:9091" not in targets:
        findings.append("PROM_SCRAPE_TARGET_MISMATCH")
    return {
        "status": "valid" if not findings else "invalid",
        "findings": findings,
    }
