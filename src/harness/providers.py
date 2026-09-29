"""OpenAI-compatible providers, routing, health and usage accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from .http_control import request


@dataclass
class ModelUsage:
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(
        self, name, base_url, api_key=None, model=None, transport=None, timeout=120
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(transport=transport)
        self.transport = transport
        self.timeout = timeout

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self):
        try:
            return request(
                "GET",
                f"{self.base_url}/models",
                client=self.client,
                transport=self.transport,
                owned_client=True,
                headers=self.headers(),
                timeout=5,
            ).is_success
        except httpx.HTTPError:
            return False

    def models(self):
        r = request(
            "GET",
            f"{self.base_url}/models",
            client=self.client,
            transport=self.transport,
            owned_client=True,
            headers=self.headers(),
            timeout=10,
        )
        if not r.is_success:
            raise ProviderError(f"{self.name}: {r.status_code}")
        return [x.get("id") for x in r.json().get("data", [])]

    def complete(self, prompt, model=None, tools=None):
        selected = model or self.model
        if not selected:
            raise ProviderError(f"{self.name}: no model configured")
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        body = {
            "model": selected,
            "messages": messages,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
        r = request(
            "POST",
            f"{self.base_url}/chat/completions",
            client=self.client,
            transport=self.transport,
            owned_client=True,
            headers={**self.headers(), "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        if not r.is_success:
            raise ProviderError(f"{self.name}: HTTP {r.status_code}")
        try:
            data = r.json()
            message = data["choices"][0]["message"]
            calls = [
                ToolCall(
                    call["id"],
                    call["function"]["name"],
                    json.loads(call["function"]["arguments"]),
                )
                for call in message.get("tool_calls") or []
            ]
            if any(
                not call.id or not isinstance(call.arguments, dict) for call in calls
            ):
                raise ValueError("invalid tool call")
            content = message.get("content") or ""
            if not isinstance(content, str) or (not content.strip() and not calls):
                raise ValueError("empty or invalid response")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name}: invalid provider response") from exc
        usage = data.get("usage") or {}
        usage = ModelUsage(
            self.name,
            selected,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        )
        if calls:
            return content, usage, calls
        return content, usage


class UsageTracker:
    def __init__(self):
        self.runs = []

    def record(self, usage):
        self.runs.append(usage)

    def total(self):
        return sum(u.cost or 0 for u in self.runs)


class CostCalculator:
    def __init__(self, rates=None):
        self.rates = rates or {}

    def calculate(self, usage):
        rate = self.rates.get(usage.model)
        if (
            rate is None
            or usage.prompt_tokens is None
            or usage.completion_tokens is None
        ):
            usage.cost = None
            return None
        usage.cost = (
            usage.prompt_tokens * rate.get("input", 0)
            + usage.completion_tokens * rate.get("output", 0)
        ) / 1_000_000
        return usage.cost
