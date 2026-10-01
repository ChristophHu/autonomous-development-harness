import httpx
import pytest

import harness.providers as providers_module
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import (
    CostCalculator,
    ModelUsage,
    OpenAICompatibleProvider,
    ProviderError,
    UsageTracker,
)


def test_provider_no_auth_and_missing_usage():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )
    )
    provider = OpenAICompatibleProvider(
        "local", "http://model/v1", model="fixture-model", transport=transport
    )
    assert provider.headers() == {} and provider.complete("prompt")[0] == "ok"


def test_provider_safe_metadata_headers_merge_with_auth():
    provider = OpenAICompatibleProvider(
        "remote",
        "https://example.test/v1",
        api_key="secret",
        headers={"HTTP-Referer": "https://harness.test", "X-Title": "Harness"},
    )
    assert provider.headers() == {
        "Authorization": "Bearer secret",
        "HTTP-Referer": "https://harness.test",
        "X-Title": "Harness",
    }


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer forged"},
        {"Cookie": "x=y"},
        {"X-Title": "bad\r\ninjected: yes"},
        {"X-Unknown": "value"},
    ],
)
def test_provider_rejects_unsafe_custom_headers(headers):
    with pytest.raises(ValueError, match="safe single-line allowlist"):
        OpenAICompatibleProvider("remote", "https://example.test/v1", headers=headers)


def test_lmstudio_provider_rejects_custom_headers():
    with pytest.raises(ValueError, match="not supported for LM Studio"):
        OpenAICompatibleProvider(
            "local",
            "http://127.0.0.1:1234/v1",
            kind="lmstudio",
            headers={"X-Title": "Harness"},
        )


def test_provider_merges_partial_retry_settings_with_defaults():
    provider = OpenAICompatibleProvider("p", "http://model", retry={"max_attempts": 1})
    assert provider.retry == {
        "max_attempts": 1,
        "base_delay": 0.25,
        "max_delay": 4.0,
    }


def test_provider_errors_and_health():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="unavailable")
    )
    provider = OpenAICompatibleProvider(
        "bad", "http://model/v1", model="fixture-model", transport=transport
    )
    assert provider.health() is False
    with pytest.raises(ProviderError):
        provider.models()
    with pytest.raises(ProviderError):
        provider.complete("prompt")


@pytest.mark.parametrize(
    "payload",
    [{"data": "bad"}, {"data": [{"id": 3}]}, {"data": [{"id": "bad\nline"}]}],
)
def test_discovery_rejects_malformed_provider_inventory_without_leaking_body(payload):
    provider = OpenAICompatibleProvider(
        "local",
        "http://model/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )
    with pytest.raises(ProviderError, match="invalid model inventory") as error:
        provider.models()
    assert "bad" not in str(error.value)


def test_discovery_returns_sorted_provider_ids():
    provider = OpenAICompatibleProvider(
        "local",
        "http://model/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"id": "z"}, {"id": "a"}]}
            )
        ),
    )
    assert provider.models() == ["z", "a"]


def test_lmstudio_health_distinguishes_downloaded_and_loaded_models():
    requests = []
    deadlines = []

    def handler(request):
        requests.append(str(request.url))
        deadlines.append(request.extensions["timeout"])
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        return httpx.Response(
            200,
            json={
                "models": [
                    {"key": "qwen", "type": "llm", "loaded_instances": []},
                    {
                        "key": "embed",
                        "type": "embedding",
                        "loaded_instances": [{"id": "embed"}],
                    },
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        model="qwen",
        transport=httpx.MockTransport(handler),
    )
    report = provider.health_report()
    assert report.reachable and report.api_available
    assert report.models == ("qwen",)
    assert report.loaded_models == ()
    assert report.status == "not_loaded"
    assert not report.model_available("qwen")
    assert not provider.health()
    assert (
        requests
        == [
            "http://127.0.0.1:1234/v1/models",
            "http://127.0.0.1:1234/api/v1/models",
        ]
        * 2
    )
    assert all(limit["connect"] == limit["read"] == 2.5 for limit in deadlines)


def test_lmstudio_health_reports_loaded_configured_model():
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "qwen",
                        "type": "llm",
                        "loaded_instances": [{"id": "qwen"}],
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        model="qwen",
        transport=httpx.MockTransport(handler),
    )
    report = provider.health_report()
    assert report.status == "available"
    assert report.model_available("qwen")
    assert provider.health()


def test_lmstudio_loaded_model_must_also_be_visible_to_openai_api():
    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"data": [{"id": "visible"}]}
                if request.url.path == "/v1/models"
                else {
                    "models": [
                        {
                            "key": "other",
                            "type": "llm",
                            "loaded_instances": [{"id": "other"}],
                        }
                    ]
                },
            )
        ),
    )
    report = provider.health_report()
    assert report.models == ("visible",)
    assert report.loaded_models == ()
    assert report.status == "not_loaded"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (httpx.Response(503), None, "api_unavailable"),
        (httpx.Response(200, json={"wrong": []}), None, "api_unavailable"),
        (httpx.Response(200, json={"data": []}), None, "no_models"),
        (
            httpx.Response(200, json={"data": [{"id": "qwen"}]}),
            httpx.Response(503),
            "loaded_state_unknown",
        ),
        (
            httpx.Response(200, json={"data": [{"id": "qwen"}]}),
            httpx.Response(200, json={"models": "bad"}),
            "loaded_state_unknown",
        ),
    ],
)
def test_provider_health_fails_closed_on_api_and_native_errors(first, second, expected):
    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        transport=httpx.MockTransport(
            lambda request: first if request.url.path == "/v1/models" else second
        ),
    )
    assert provider.health_report().status == expected


def test_remote_provider_health_only_needs_valid_nonempty_model_api():
    provider = OpenAICompatibleProvider(
        "openai",
        "https://example.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": "model"}]})
        ),
    )
    report = provider.health_report()
    assert report.status == "available"
    assert report.loaded_models is None
    assert report.model_available("model")


def test_health_report_separates_transport_failure_and_configured_model_absence():
    def offline(request):
        raise httpx.ConnectError("private transport detail", request=request)

    provider = OpenAICompatibleProvider(
        "openai",
        "https://example.test/v1",
        transport=httpx.MockTransport(offline),
    )
    report = provider.health_report()
    assert report.status == "unreachable"
    assert not report.reachable and not provider.health()

    provider = OpenAICompatibleProvider(
        "openai",
        "https://example.test/v1",
        model="configured",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": "other"}]})
        ),
    )
    assert provider.health_report().status == "configured_model_unavailable"


@pytest.mark.parametrize(
    "native",
    [
        {"models": [42]},
        {"models": [{"type": "llm", "loaded_instances": []}]},
        {"models": [{"key": "q", "type": "other", "loaded_instances": []}]},
        {"models": [{"key": "q", "type": "llm", "loaded_instances": "bad"}]},
        {"models": [{"key": "q", "type": "llm", "loaded_instances": [42]}]},
        {"models": [{"key": "q", "type": "llm", "loaded_instances": [{"id": 3}]}]},
    ],
)
def test_lmstudio_invalid_native_inventory_cannot_claim_loaded_model(native):
    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"data": [{"id": "q"}]}
                if request.url.path == "/v1/models"
                else native,
            )
        ),
    )
    report = provider.health_report()
    assert report.status == "loaded_state_unknown"
    assert not report.model_available("q")


def test_lmstudio_probe_rejects_invalid_kind_url_and_native_transport_failure():
    with pytest.raises(ValueError, match="kind"):
        OpenAICompatibleProvider("x", "http://127.0.0.1:1234/v1", kind="unknown")
    with pytest.raises(ValueError, match="/v1"):
        OpenAICompatibleProvider("x", "http://127.0.0.1:1234/api", kind="lmstudio")

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "q"}]})
        raise httpx.ConnectError("offline", request=request)

    provider = OpenAICompatibleProvider(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        kind="lmstudio",
        transport=httpx.MockTransport(handler),
    )
    assert provider.health_report().status == "loaded_state_unknown"


def test_cost_unknown_model_and_usage_tracker():
    usage = ModelUsage("p", "unknown", 10, 10)
    assert CostCalculator().calculate(usage) is None
    tracker = UsageTracker()
    tracker.record(usage)
    assert tracker.total() is None
    tracker.record(ModelUsage("p", "known", 1, 1, cost=0.0))
    assert tracker.total() is None
    complete = UsageTracker()
    assert complete.total() is None
    complete.record(ModelUsage("p", "free", 1, 1, cost=0.0))
    assert complete.total() == 0.0


def test_provider_decodes_function_tool_calls():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "filesystem.read",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )
    )
    text, usage, calls = OpenAICompatibleProvider(
        "local", "http://model/v1", model="fixture-model", transport=transport
    ).complete(
        "inspect",
        tools=[
            {
                "name": "filesystem.read",
                "description": "read",
                "parameters": {"type": "object"},
            }
        ],
    )
    assert text == "" and usage.prompt_tokens == 4
    assert (calls[0].id, calls[0].name, calls[0].arguments) == (
        "call-1",
        "filesystem.read",
        {"path": "README.md"},
    )


def test_provider_retries_transient_status_with_capped_backoff(monkeypatch):
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "20"}),
            httpx.Response(503),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    delays = []
    monkeypatch.setattr(providers_module.time, "sleep", delays.append)
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(lambda request: next(responses)),
        retry={"max_attempts": 3, "base_delay": 0.5, "max_delay": 2},
    )

    assert provider.complete("prompt")[0] == "ok"
    assert delays == [2, 1]


def test_provider_ignores_invalid_retry_after(monkeypatch):
    responses = iter(
        [
            httpx.Response(503, headers={"Retry-After": "later"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    delays = []
    monkeypatch.setattr(providers_module.time, "sleep", delays.append)
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(lambda request: next(responses)),
        retry={"max_attempts": 2, "base_delay": 0.25, "max_delay": 1},
    )

    assert provider.complete("prompt")[0] == "ok"
    assert delays == [0.25]


@pytest.mark.parametrize("status", [400, 401, 404])
def test_provider_does_not_retry_permanent_http_errors(monkeypatch, status):
    calls = []
    monkeypatch.setattr(providers_module.time, "sleep", lambda delay: pytest.fail())
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(status)
        ),
        retry={"max_attempts": 3, "base_delay": 0, "max_delay": 1},
    )

    with pytest.raises(ProviderError, match=f"HTTP {status}"):
        provider.complete("prompt")
    assert len(calls) == 1


def test_provider_stops_after_retry_budget(monkeypatch):
    calls = []
    delays = []
    monkeypatch.setattr(providers_module.time, "sleep", delays.append)
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(503)
        ),
        retry={"max_attempts": 2, "base_delay": 0.25, "max_delay": 1},
    )

    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.complete("prompt")
    assert len(calls) == 2 and delays == [0.25]


def test_provider_does_not_retry_invalid_success_response(monkeypatch):
    calls = []
    monkeypatch.setattr(providers_module.time, "sleep", lambda delay: pytest.fail())
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(
            lambda request: (
                calls.append(request) or httpx.Response(200, json={"choices": []})
            )
        ),
        retry={"max_attempts": 3, "base_delay": 0.1, "max_delay": 1},
    )

    with pytest.raises(ProviderError, match="invalid provider response"):
        provider.complete("prompt")
    assert len(calls) == 1


def test_provider_does_not_retry_ambiguous_transport_error(monkeypatch):
    calls = []
    monkeypatch.setattr(providers_module.time, "sleep", lambda delay: pytest.fail())

    def fail_connection(request):
        calls.append(request)
        raise httpx.ConnectError("connection lost after request")

    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(fail_connection),
        retry={"max_attempts": 3, "base_delay": 0.1, "max_delay": 1},
    )

    with pytest.raises(httpx.ConnectError):
        provider.complete("prompt")
    assert len(calls) == 1


def test_provider_cancellation_during_backoff_prevents_another_attempt(monkeypatch):
    calls = []
    control = RunControl()
    wait = control.stop_event.wait

    def stop_during_backoff(delay):
        if delay > 0.02:
            control.request_stop("test cancellation")
            return True
        return wait(delay)

    monkeypatch.setattr(control.stop_event, "wait", stop_during_backoff)
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(503)
        ),
        retry={"max_attempts": 3, "base_delay": 0.1, "max_delay": 1},
    )

    with use_run_control(control), pytest.raises(TaskCancelled):
        provider.complete("prompt")
    assert len(calls) == 1


def test_provider_continues_after_controlled_backoff_without_cancellation():
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    provider = OpenAICompatibleProvider(
        "p",
        "http://model",
        model="m",
        transport=httpx.MockTransport(lambda request: next(responses)),
        retry={"max_attempts": 2, "base_delay": 0, "max_delay": 1},
    )

    with use_run_control(RunControl()):
        assert provider.complete("prompt")[0] == "ok"
