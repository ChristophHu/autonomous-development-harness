"""Read-only live acceptance checks for a Prometheus scrape integration."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx


def _valid_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and (port is None or 1 <= port <= 65535)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _prometheus_base(value):
    return value.rstrip("/")


def check_prometheus_integration(
    prometheus_url,
    harness_metrics_url,
    *,
    client_factory=httpx.Client,
):
    """Check readiness, target health, query result, and unauthenticated 401.

    Reports contain stable check statuses only; upstream response bodies and
    target errors are deliberately never returned.
    """
    if not _valid_url(prometheus_url) or not _valid_url(harness_metrics_url):
        return {
            "status": "invalid_input",
            "checks": {},
            "findings": ["PROM_ACCEPTANCE_URL_INVALID"],
        }

    checks = {
        "prometheus_ready": "failed",
        "harness_target": "failed",
        "up_query": "failed",
        "metrics_auth": "failed",
    }
    findings = []
    try:
        with client_factory(timeout=3.0, follow_redirects=False) as client:
            try:
                ready = client.get(f"{_prometheus_base(prometheus_url)}/-/ready")
            except httpx.HTTPError:
                return {
                    "status": "unreachable",
                    "checks": checks,
                    "findings": ["PROM_ACCEPTANCE_UNREACHABLE"],
                }
            if ready.status_code == 200:
                checks["prometheus_ready"] = "passed"
            else:
                findings.append("PROM_ACCEPTANCE_NOT_READY")

            try:
                response = client.get(
                    f"{_prometheus_base(prometheus_url)}/api/v1/targets",
                    params={"state": "active"},
                )
                payload = response.json()
                targets = payload.get("data", {}).get("activeTargets", [])
                matching = [
                    target
                    for target in targets
                    if isinstance(target, dict)
                    and isinstance(target.get("labels"), dict)
                    and target["labels"].get("job") == "harness"
                ]
                target_ok = (
                    response.status_code == 200
                    and payload.get("status") == "success"
                    and len(matching) == 1
                    and matching[0].get("health") == "up"
                )
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                target_ok = False
            if target_ok:
                checks["harness_target"] = "passed"
            else:
                findings.append("PROM_ACCEPTANCE_TARGET_NOT_UP")

            try:
                response = client.get(
                    f"{_prometheus_base(prometheus_url)}/api/v1/query",
                    params={"query": 'up{job="harness"}'},
                )
                payload = response.json()
                results = payload.get("data", {}).get("result", [])
                values = [
                    item.get("value", [None, None])[1]
                    for item in results
                    if isinstance(item, dict)
                    and isinstance(item.get("value"), list)
                    and len(item["value"]) == 2
                ]
                query_ok = (
                    response.status_code == 200
                    and payload.get("status") == "success"
                    and len(values) == 1
                    and values[0] == "1"
                )
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                query_ok = False
            if query_ok:
                checks["up_query"] = "passed"
            else:
                findings.append("PROM_ACCEPTANCE_QUERY_NOT_ONE")

            try:
                auth_response = client.get(harness_metrics_url)
                auth_ok = auth_response.status_code == 401
            except httpx.HTTPError:
                auth_ok = False
            if auth_ok:
                checks["metrics_auth"] = "passed"
            else:
                findings.append("PROM_ACCEPTANCE_AUTH_CONTRACT_FAILED")
    except (OSError, RuntimeError, TypeError, ValueError):
        return {
            "status": "unreachable",
            "checks": checks,
            "findings": ["PROM_ACCEPTANCE_CLIENT_FAILED"],
        }

    return {
        "status": "passed" if not findings else "failed",
        "checks": checks,
        "findings": findings,
    }
