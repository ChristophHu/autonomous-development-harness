import sys
from types import SimpleNamespace

import pytest

from harness.context_token_budget import (
    TokenBudgetError,
    TokenCounterRegistry,
    count_with_tiktoken,
    count_with_tokenizers_file,
    enforce_token_budget,
)


def test_registry_counts_serialized_messages_tools_and_framing():
    seen = []
    registry = TokenCounterRegistry({"model-a": lambda text: seen.append(text) or 4})
    count = registry.count(
        "model-a",
        [{"role": "user", "content": "hello"}],
        [{"name": "search"}],
        framing_tokens=3,
    )
    assert count == 7
    assert '"tools"' in seen[0] and '"type":"function"' in seen[0]
    assert '"model":"model-a"' in seen[0] and "hello" in seen[0]


def test_registry_estimates_characters_with_configured_margin():
    registry = TokenCounterRegistry()
    registry.register_estimator(
        "model-a", characters_per_token=2, safety_margin_percent=25
    )
    estimate = registry.estimate("model-a", "abcd")
    raw = (len('{"messages":"abcd","model":"model-a","tools":[]}') + 1) // 2
    assert estimate.raw_tokens == raw
    assert estimate.estimated_tokens == (raw * 125 + 99) // 100
    assert estimate.method == "characters_per_token"
    assert estimate.safety_margin_percent == 25


def test_estimator_budget_uses_conservative_estimate_and_reports_method():
    registry = TokenCounterRegistry()
    registry.register_estimator(
        "model-a", characters_per_token=1, safety_margin_percent=100
    )
    expected = registry.estimate("model-a", "abcd").estimated_tokens
    with pytest.raises(
        TokenBudgetError, match="estimated prompt token budget exceeded"
    ):
        enforce_token_budget(registry, "model-a", "abcd", expected - 1)
    assert enforce_token_budget(registry, "model-a", "abcd", expected) == expected


def test_registry_rejects_invalid_estimator_settings():
    registry = TokenCounterRegistry()
    for model, ratio, margin in (
        ("", 4, 20),
        ("m", 0, 20),
        ("m", 4, -1),
        ("m", 4, 101),
    ):
        with pytest.raises(ValueError):
            registry.register_estimator(model, ratio, margin)
    registry.register_estimator("model", 4)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("model", lambda _text: 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_estimator("model", 3)


def test_registry_rejects_invalid_registration_and_duplicate():
    registry = TokenCounterRegistry()
    for model, counter, error in (
        (" ", lambda _: 1, ValueError),
        ("model", None, TypeError),
    ):
        with pytest.raises(error):
            registry.register(model, counter)
    registry.register("model", lambda _: 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("model", lambda _: 1)


def test_hard_budget_fails_closed_for_unknown_model_and_invalid_inputs():
    registry = TokenCounterRegistry({"model": lambda _: 1})
    with pytest.raises(TokenBudgetError, match="no exact token counter"):
        registry.count("unknown", "prompt")
    with pytest.raises(TokenBudgetError, match="resolved model"):
        registry.count("", "prompt")
    with pytest.raises(TypeError, match="message list"):
        registry.count("model", {})
    for framing in (True, -1, 1.5):
        with pytest.raises(ValueError, match="framing"):
            TokenCounterRegistry({"model": lambda _: 0}).count(
                "model", "x", framing_tokens=framing
            )
    for margin in (True, -1, 101, 1.5):
        with pytest.raises(ValueError, match="safety margin"):
            TokenCounterRegistry({"model": lambda _text: 1}).estimate(
                "model", "x", safety_margin_percent=margin
            )
    with pytest.raises(ValueError, match="positive integer"):
        enforce_token_budget(registry, "model", "x", True)


def test_hard_budget_checks_counter_contract_and_limit():
    for output in (True, -1, "many"):
        registry = TokenCounterRegistry({"model": lambda _text, value=output: value})
        with pytest.raises(TokenBudgetError, match="invalid count"):
            registry.count("model", "prompt")
    registry = TokenCounterRegistry({"model": lambda text: len(text)})
    assert enforce_token_budget(registry, "model", "x", 100) > 0
    with pytest.raises(TokenBudgetError, match="budget exceeded"):
        enforce_token_budget(registry, "model", "x" * 100, 1)


def test_tiktoken_counter_requires_explicit_encoding_and_optional_dependency(
    monkeypatch,
):
    with pytest.raises(ValueError, match="encoding name"):
        count_with_tiktoken("")
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(TokenBudgetError, match="not installed"):
        count_with_tiktoken("cl100k_base")("prompt")


def test_tiktoken_counter_uses_explicit_encoding_without_special_token_rejection(
    monkeypatch,
):
    class Encoding:
        def encode(self, text, *, disallowed_special):
            assert disallowed_special == ()
            return text.split()

    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(get_encoding=lambda name: Encoding()),
    )
    counter = count_with_tiktoken("explicit-test-encoding")
    assert counter("a b") == 2
    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(get_encoding=lambda _name: (_ for _ in ()).throw(KeyError())),
    )
    with pytest.raises(TokenBudgetError, match="unavailable"):
        count_with_tiktoken("missing")("prompt")


def test_tokenizers_file_is_local_bounded_and_optional(tmp_path, monkeypatch):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("fixture")
    with pytest.raises(TokenBudgetError, match="unavailable"):
        count_with_tokenizers_file(str(tmp_path / "missing.json"))
    link = tmp_path / "link.json"
    link.symlink_to(tokenizer_file)
    with pytest.raises(TokenBudgetError, match="safe regular"):
        count_with_tokenizers_file(str(link))
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(TokenBudgetError, match="not installed"):
        count_with_tokenizers_file(str(tokenizer_file))("prompt")


def test_tokenizers_file_uses_local_tokenizer_and_rejects_bad_file(
    tmp_path, monkeypatch
):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("fixture")

    class Encoder:
        @staticmethod
        def encode(text, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=text.split())

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _path: Encoder())),
    )
    assert count_with_tokenizers_file(str(tokenizer_file))("one two") == 2
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(
            Tokenizer=SimpleNamespace(
                from_file=lambda _path: (_ for _ in ()).throw(ValueError("bad"))
            )
        ),
    )
    with pytest.raises(TokenBudgetError, match="invalid"):
        count_with_tokenizers_file(str(tokenizer_file))("prompt")
