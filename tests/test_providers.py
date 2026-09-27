import httpx
import pytest

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
