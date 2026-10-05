import httpx
import pytest

from harness.prometheus_acceptance import check_prometheus_integration

PROM_URL = "http://prometheus.test:9090"
METRICS_URL = "http://harness.test:9091/metrics/prometheus"


def _client_factory(handler):
    def factory(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    return factory


def _success_handler(request):
    if request.url.path == "/-/ready":
        return httpx.Response(200)
    if request.url.path == "/api/v1/targets":
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "activeTargets": [{"labels": {"job": "harness"}, "health": "up"}]
                },
            },
        )
    if request.url.path == "/api/v1/query":
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {"result": [{"value": ["now", "1"]}]},
            },
        )
    return httpx.Response(401, json={"detail": "secret response must stay hidden"})


def test_live_acceptance_passes_all_four_checks_without_leaking_response_content():
    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(_success_handler)
    )

    assert result == {
        "status": "passed",
        "checks": {
            "prometheus_ready": "passed",
            "harness_target": "passed",
            "up_query": "passed",
            "metrics_auth": "passed",
        },
        "findings": [],
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "ftp://prometheus.test",
        "http://user:password@host",
        "http://host?q=secret",
        "http://host/#fragment",
        "http://[invalid",
        "http://host:bad",
        "http://host:70000",
    ],
)
def test_invalid_urls_are_rejected_before_any_request(value):
    def forbidden_factory(**_kwargs):
        raise AssertionError("invalid URL must not cause a request")

    result = check_prometheus_integration(
        value, METRICS_URL, client_factory=forbidden_factory
    )

    assert result["status"] == "invalid_input"
    assert result["findings"] == ["PROM_ACCEPTANCE_URL_INVALID"]


def test_invalid_harness_url_is_rejected():
    result = check_prometheus_integration(PROM_URL, "file:///metrics")
    assert result["status"] == "invalid_input"


def test_prometheus_connection_failure_is_reported_without_response_details():
    def handler(_request):
        raise httpx.ConnectError("private hostname and token")

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(handler)
    )

    assert result["status"] == "unreachable"
    assert result["findings"] == ["PROM_ACCEPTANCE_UNREACHABLE"]
    assert "private" not in str(result)


@pytest.mark.parametrize(
    ("path", "response", "expected_finding"),
    [
        ("/-/ready", httpx.Response(503), "PROM_ACCEPTANCE_NOT_READY"),
        (
            "/api/v1/targets",
            httpx.Response(200, json={"status": "error"}),
            "PROM_ACCEPTANCE_TARGET_NOT_UP",
        ),
        (
            "/api/v1/query",
            httpx.Response(200, json={"status": "success", "data": {"result": []}}),
            "PROM_ACCEPTANCE_QUERY_NOT_ONE",
        ),
        (
            "/metrics/prometheus",
            httpx.Response(200),
            "PROM_ACCEPTANCE_AUTH_CONTRACT_FAILED",
        ),
    ],
)
def test_nonpassing_responses_produce_stable_finding_codes(
    path, response, expected_finding
):
    def handler(request):
        if request.url.path == path:
            return response
        return _success_handler(request)

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(handler)
    )

    assert result["status"] == "failed"
    assert expected_finding in result["findings"]
    assert "secret response" not in str(result)


def test_redirects_are_not_followed():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/-/ready":
            return httpx.Response(302, headers={"location": "http://other.test/"})
        return _success_handler(request)

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(handler)
    )

    assert result["checks"]["prometheus_ready"] == "failed"
    assert len(requests) == 4
    assert all(request.url.host != "other.test" for request in requests)


def test_invalid_payload_shapes_and_bad_json_fail_closed():
    def handler(request):
        if request.url.path == "/-/ready":
            return httpx.Response(200)
        if request.url.path == "/api/v1/targets":
            return httpx.Response(200, json={"status": "success", "data": None})
        if request.url.path == "/api/v1/query":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(401)

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(handler)
    )
    assert result["checks"]["harness_target"] == "failed"
    assert result["checks"]["up_query"] == "failed"


def test_unreachable_harness_listener_fails_auth_check():
    def handler(request):
        if request.url.path == "/metrics/prometheus":
            raise httpx.ConnectError("private listener detail")
        return _success_handler(request)

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=_client_factory(handler)
    )
    assert result["checks"]["metrics_auth"] == "failed"
    assert result["findings"] == ["PROM_ACCEPTANCE_AUTH_CONTRACT_FAILED"]


def test_client_construction_failure_is_bounded():
    def broken_factory(**_kwargs):
        raise RuntimeError("private client detail")

    result = check_prometheus_integration(
        PROM_URL, METRICS_URL, client_factory=broken_factory
    )
    assert result["status"] == "unreachable"
    assert result["findings"] == ["PROM_ACCEPTANCE_CLIENT_FAILED"]


def test_cli_live_check_uses_report_and_returns_failure_exit(monkeypatch):
    from typer.testing import CliRunner

    from harness import cli

    monkeypatch.setattr(
        cli,
        "check_prometheus_integration",
        lambda *_args: {
            "status": "failed",
            "checks": {},
            "findings": ["PROM_ACCEPTANCE_TARGET_NOT_UP"],
        },
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "observability",
            "check-prometheus",
            "--prometheus-url",
            PROM_URL,
            "--harness-metrics-url",
            METRICS_URL,
        ],
    )

    assert result.exit_code == 1
    assert "PROM_ACCEPTANCE_TARGET_NOT_UP" in result.stdout
