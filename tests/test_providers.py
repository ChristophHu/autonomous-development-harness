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


def test_cost_unknown_model_and_usage_tracker():
    usage = ModelUsage("p", "unknown", 10, 10)
    assert CostCalculator().calculate(usage) is None
    tracker = UsageTracker()
    tracker.record(usage)
    assert tracker.total() == 0


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
