"""Explicit model-bound token counters for prompt preflight."""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class TokenBudgetError(ValueError):
    """A token estimate cannot be produced or exceeds the configured budget."""


@dataclass(frozen=True)
class TokenEstimate:
    raw_tokens: int
    estimated_tokens: int
    method: str
    safety_margin_percent: int


class TokenCounterRegistry:
    """Map exact configured model IDs to trusted local count functions."""

    def __init__(self, counters=None):
        self._counters = {}
        self._estimators = {}
        for model, counter in (counters or {}).items():
            self.register(model, counter)

    def register(self, model, counter):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("token counter model ID must not be blank")
        if not callable(counter):
            raise TypeError("token counter must be callable")
        if model in self._counters or model in self._estimators:
            raise ValueError("token counter already registered for model")
        self._counters[model] = counter

    def register_estimator(self, model, characters_per_token, safety_margin_percent=20):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("token estimator model ID must not be blank")
        if model in self._counters or model in self._estimators:
            raise ValueError("token counter already registered for model")
        if (
            isinstance(characters_per_token, bool)
            or not isinstance(characters_per_token, (int, float))
            or not math.isfinite(characters_per_token)
            or characters_per_token <= 0
        ):
            raise ValueError("characters per token must be a positive finite number")
        if (
            isinstance(safety_margin_percent, bool)
            or not isinstance(safety_margin_percent, int)
            or not 0 <= safety_margin_percent <= 100
        ):
            raise ValueError("token estimate safety margin must be between 0 and 100")
        self._estimators[model] = (characters_per_token, safety_margin_percent)

    def estimate(
        self,
        model,
        messages,
        tools=None,
        *,
        framing_tokens=0,
        safety_margin_percent=0,
    ):
        if not isinstance(model, str) or not model:
            raise TokenBudgetError("a resolved model ID is required for token counting")
        counter = self._counters.get(model)
        estimator = self._estimators.get(model)
        if counter is None and estimator is None:
            raise TokenBudgetError(
                f"no exact token counter or approximation estimator registered for model: {model}"
            )
        if not isinstance(messages, (str, list)):
            raise TypeError("prompt messages must be text or a message list")
        if (
            isinstance(framing_tokens, bool)
            or not isinstance(framing_tokens, int)
            or framing_tokens < 0
        ):
            raise ValueError("framing token allowance must be a non-negative integer")
        if (
            isinstance(safety_margin_percent, bool)
            or not isinstance(safety_margin_percent, int)
            or not 0 <= safety_margin_percent <= 100
        ):
            raise ValueError("token estimate safety margin must be between 0 and 100")
        payload = {
            "model": model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in (tools or [])],
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if counter is not None:
            count = counter(serialized)
            method = "configured_tokenizer_estimate"
            margin = safety_margin_percent
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TokenBudgetError("token counter returned an invalid count")
        else:
            characters_per_token, margin = estimator
            count = math.ceil(len(serialized) / characters_per_token)
            method = "characters_per_token"
        raw_tokens = count + framing_tokens
        estimated_tokens = math.ceil(raw_tokens * (100 + margin) / 100)
        return TokenEstimate(raw_tokens, estimated_tokens, method, margin)

    def count(self, model, messages, tools=None, *, framing_tokens=0):
        return self.estimate(
            model, messages, tools, framing_tokens=framing_tokens
        ).estimated_tokens


def enforce_token_budget(
    registry,
    model,
    messages,
    budget,
    tools=None,
    *,
    framing_tokens=0,
    safety_margin_percent=0,
):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("token budget must be a positive integer")
    estimate = registry.estimate(
        model,
        messages,
        tools,
        framing_tokens=framing_tokens,
        safety_margin_percent=safety_margin_percent,
    )
    if estimate.estimated_tokens > budget:
        raise TokenBudgetError(
            f"estimated prompt token budget exceeded for model {model}: "
            f"{estimate.estimated_tokens}>{budget}"
        )
    return estimate.estimated_tokens


def count_with_tiktoken(encoding_name: str) -> Callable[[str], int]:
    """Create an explicitly selected tiktoken counter; no model guessing."""
    if not isinstance(encoding_name, str) or not encoding_name.strip():
        raise ValueError("tokenizer encoding name must not be blank")

    def count(text):
        try:
            import tiktoken
        except ImportError as error:
            raise TokenBudgetError("tiktoken support is not installed") from error
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except (KeyError, ValueError) as error:
            raise TokenBudgetError(
                "configured tiktoken encoding is unavailable"
            ) from error
        return len(encoding.encode(text, disallowed_special=()))

    return count


def count_with_tokenizers_file(tokenizer_file: str) -> Callable[[str], int]:
    """Load a local Hugging Face tokenizer.json without network/code loading."""
    path = Path(tokenizer_file)
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_size > 50_000_000
        ):
            raise TokenBudgetError("tokenizer file is not a safe regular file")
    except TokenBudgetError:
        raise
    except (OSError, ValueError) as error:
        raise TokenBudgetError("configured tokenizer file is unavailable") from error

    def count(text):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise TokenBudgetError("tokenizers support is not installed") from error
        try:
            tokenizer = Tokenizer.from_file(str(resolved))
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        except (OSError, ValueError) as error:
            raise TokenBudgetError("configured tokenizer file is invalid") from error

    return count
