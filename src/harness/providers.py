"""OpenAI-compatible providers, routing, health and usage accounting."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from .http_control import request
from .process_control import current_run_control
from .retry_budget import current_retry_budget


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
    def __init__(
        self, message, *, category="provider_permanent", fallback_allowed=False
    ):
        super().__init__(message)
        self.category = category
        self.fallback_allowed = fallback_allowed


def parse_retry_after(value, *, now=None):
    """Parse RFC delta-seconds or HTTP-date; invalid/negative values are ignored."""
    if value is None:
        return None
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            current = now or datetime.now(UTC)
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            return max(0.0, (retry_at - current).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True)
class ProviderHealth:
    kind: str
    reachable: bool
    api_available: bool
    models: tuple[str, ...] = ()
    loaded_models: tuple[str, ...] | None = None
    configured_model: str | None = None

    @property
    def status(self):
        if not self.reachable:
            return "unreachable"
        if not self.api_available:
            return "api_unavailable"
        if not self.models:
            return "no_models"
        if self.kind == "lmstudio":
            if self.loaded_models is None:
                return "loaded_state_unknown"
            if not self.loaded_models:
                return "not_loaded"
        if self.configured_model and not self.model_available(self.configured_model):
            return "configured_model_unavailable"
        return "available"

    def model_available(self, model_id):
        available = self.loaded_models if self.kind == "lmstudio" else self.models
        return model_id in available if available is not None else False


class OpenAICompatibleProvider:
    def __init__(
        self,
        name,
        base_url,
        api_key=None,
        model=None,
        transport=None,
        timeout=120,
        retry=None,
        kind="openai_compatible",
        headers=None,
    ):
        if kind not in {"openai_compatible", "lmstudio"}:
            raise ValueError("unsupported provider kind")
        if kind == "lmstudio" and not base_url.rstrip("/").endswith("/v1"):
            raise ValueError("LM Studio base URL must end in /v1")
        headers = headers or {}
        if kind == "lmstudio" and headers:
            raise ValueError("custom provider headers are not supported for LM Studio")
        if any(
            not isinstance(key, str)
            or key.lower() not in {"http-referer", "x-title"}
            or not isinstance(value, str)
            or not value.strip()
            or any(char in value for char in "\r\n")
            for key, value in headers.items()
        ):
            raise ValueError("provider headers must use the safe single-line allowlist")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.kind = kind
        self.api_key = api_key
        self.custom_headers = dict(headers)
        self.model = model
        self.client = httpx.Client(transport=transport)
        self.transport = transport
        self.timeout = timeout
        self.retry = {
            "max_attempts": 3,
            "base_delay": 0.25,
            "max_delay": 4.0,
        }
        self.retry.update(retry or {})

    def headers(self):
        result = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        result.update(self.custom_headers)
        return result

    def _wait_retry(self, attempt, retry_after=None):
        budget = current_retry_budget()
        if budget is not None and not budget.claim():
            return False
        delay = min(self.retry["base_delay"] * (2**attempt), self.retry["max_delay"])
        if retry_after is not None:
            delay = min(max(delay, retry_after), self.retry["max_delay"])
        control = current_run_control()
        if budget is not None:
            budget.wait(delay)
        elif control is None:
            time.sleep(delay)
        elif control.stop_event.wait(delay):
            control.check()
        return True

    def health(self):
        return self.health_report().status == "available"

    @staticmethod
    def _model_ids(response):
        try:
            entries = response.json()["data"]
            if not isinstance(entries, list) or not all(
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
                and not any(char in item["id"] for char in "\r\n\t")
                for item in entries
            ):
                raise ValueError("invalid inventory")
            return tuple(item["id"] for item in entries)
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("invalid model inventory") from error

    @staticmethod
    def _loaded_ids(response):
        try:
            entries = response.json()["models"]
            if not isinstance(entries, list):
                raise ValueError("invalid native inventory")  # noqa: TRY004
            loaded = set()
            for item in entries:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("key"), str)
                    or item.get("type") not in {"llm", "embedding"}
                    or not isinstance(item.get("loaded_instances"), list)
                ):
                    raise ValueError("invalid native inventory")
                instances = item["loaded_instances"]
                if not all(
                    isinstance(instance, dict) and isinstance(instance.get("id"), str)
                    for instance in instances
                ):
                    raise ValueError("invalid native inventory")
                if item["type"] == "llm" and instances:
                    loaded.add(item["key"])
            return tuple(sorted(loaded))
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("invalid native model inventory") from error

    def health_report(self):
        report = ProviderHealth(self.kind, False, False, configured_model=self.model)
        try:
            response = request(
                "GET",
                f"{self.base_url}/models",
                client=self.client,
                transport=self.transport,
                owned_client=True,
                headers=self.headers(),
                timeout=2.5 if self.kind == "lmstudio" else 5,
            )
        except httpx.HTTPError:
            return report
        if not response.is_success:
            return ProviderHealth(self.kind, True, False, configured_model=self.model)
        try:
            models = self._model_ids(response)
        except ProviderError:
            return ProviderHealth(self.kind, True, False, configured_model=self.model)
        if self.kind != "lmstudio" or not models:
            return ProviderHealth(
                self.kind, True, True, models, configured_model=self.model
            )
        try:
            native = request(
                "GET",
                f"{self.base_url[:-3]}/api/v1/models",
                client=self.client,
                transport=self.transport,
                owned_client=True,
                headers=self.headers(),
                timeout=2.5,
            )
            loaded = (
                tuple(sorted(set(self._loaded_ids(native)) & set(models)))
                if native.is_success
                else None
            )
        except (httpx.HTTPError, ProviderError):
            loaded = None
        return ProviderHealth(self.kind, True, True, models, loaded, self.model)

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
        return list(self._model_ids(r))

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
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.retry["max_attempts"]):
            control = current_run_control()
            if control is not None:
                control.check()
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
            if r.is_success:
                break
            if r.status_code not in retryable_statuses:
                raise ProviderError(
                    f"{self.name}: HTTP {r.status_code}",
                    category="provider_permanent",
                )
            if attempt + 1 < self.retry["max_attempts"]:
                retry_after = parse_retry_after(r.headers.get("Retry-After"))
                if not self._wait_retry(attempt, retry_after):
                    break
        if not r.is_success:
            raise ProviderError(
                f"{self.name}: HTTP {r.status_code}",
                category="transient_http",
                fallback_allowed=True,
            )
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
        if not self.runs or any(usage.cost is None for usage in self.runs):
            return None
        return sum(usage.cost for usage in self.runs)


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
