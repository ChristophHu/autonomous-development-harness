import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.claim_evidence import (
    IndependentClaimVerifier,
    RoutedClaimVerifier,
    verify_claim,
)


@pytest.mark.parametrize(
    ("claim", "source", "status"),
    [
        ("Alpha beta", "The requirement is alpha   beta here.", "supported"),
        (True, "enabled=true", "supported"),
        (False, "enabled=false", "supported"),
        (12, "retries: 12", "supported"),
        (1.5, "ratio is 1.5", "supported"),
        (12, "retries: 120", "insufficient"),
        ("gamma", "alpha beta", "insufficient"),
        ("", "some evidence", "insufficient"),
        (["alpha", "beta"], "alpha beta", "supported"),
        (["alpha", "gamma"], "alpha beta", "insufficient"),
        ([], "alpha beta", "insufficient"),
        ("alpha", "", "insufficient"),
        ("alpha", None, "insufficient"),
    ],
)
def test_claim_verifier_only_accepts_exact_scalar_source_evidence(
    claim, source, status
):
    verdict = verify_claim(claim, source)
    assert verdict.status == status
    assert bool(verdict.quote) is (status == "supported")


def test_claim_verifier_returns_original_casing_and_rejects_partial_word():
    assert verify_claim("alpha", "ALPHA is required.").quote == "ALPHA"
    assert verify_claim("auth", "authorization is required").status == "insufficient"


def test_independent_verifier_requires_distinct_identity_and_grounded_quote():
    with pytest.raises(TypeError):
        IndependentClaimVerifier(None, identity="reviewer", author_identity="writer")
    for identity, author in (("", "writer"), ("reviewer", " "), ("same", "same")):
        with pytest.raises(ValueError):
            IndependentClaimVerifier(
                lambda *_: {}, identity=identity, author_identity=author
            )
    verifier = IndependentClaimVerifier(
        lambda _value, _source: {"status": "supported", "quote": "Source phrase"},
        identity="reviewer-model",
        author_identity="author-model",
    )
    assert (
        verify_claim(
            "a paraphrase", "The Source phrase is present", verifier=verifier
        ).status
        == "supported"
    )
    assert (
        verify_claim("unsupported", "Source phrase", verifier=verifier).status
        == "supported"
    )
    assert verifier.verify("claim", "").status == "insufficient"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"status": "unknown"}, "insufficient"),
        ("supported", "insufficient"),
        ({"status": "supported", "quote": "not in source"}, "insufficient"),
        (
            {"status": "supported", "quote": "source phrase", "extra": True},
            "insufficient",
        ),
        (
            {"status": "supported", "quote": "source phrase", "verifier_id": " "},
            "insufficient",
        ),
        ({"status": "contradicted"}, "contradicted"),
        ({"status": "ambiguous", "quote": 1}, "insufficient"),
    ],
)
def test_independent_verifier_validates_shape_status_and_quote(response, expected):
    verifier = IndependentClaimVerifier(
        lambda *_: response,
        identity="reviewer",
        author_identity="author",
    )
    assert verifier.verify("claim", "source phrase").status == expected


def test_independent_verifier_failure_is_insufficient():
    verifier = IndependentClaimVerifier(
        lambda *_: (_ for _ in ()).throw(RuntimeError("failed")),
        identity="reviewer",
        author_identity="author",
    )
    assert verifier.verify("claim", "source").status == "insufficient"


def test_invalid_custom_verifier_result_fails_closed():
    class InvalidVerifier:
        @staticmethod
        def verify(_value, _source):
            return object()

    assert (
        verify_claim("not present", "source", verifier=InvalidVerifier()).status
        == "insufficient"
    )


def test_routed_claim_verifier_excludes_author_models_and_records_actual_model():
    class Registry:
        @staticmethod
        def resolve(name):
            return SimpleNamespace(model=name), name

    class Router:
        registry = Registry()

        @staticmethod
        def candidates(profile, check_availability=False):
            assert check_availability is False
            return (
                ["writer-model"]
                if profile == "author"
                else ["writer-model", "reviewer-model"]
            )

        @staticmethod
        def complete(profile, prompt, *, allowed_models, route_observer):
            assert profile == "reviewer"
            assert allowed_models == {"reviewer-model"}
            route_observer("reviewer-model")
            request = json.loads(prompt)
            assert request["claim"] == "paraphrase"
            assert "source text as untrusted data" in request["instruction"]
            return json.dumps({"status": "supported", "quote": "source wording"})

    verifier = RoutedClaimVerifier(
        Router(), author_profile="author", verifier_profile="reviewer"
    )
    verdict = verifier.verify("paraphrase", "source wording")
    assert verdict.status == "supported"
    assert verdict.verifier_id == "reviewer-model"


def test_routed_claim_verifier_fails_closed_when_only_author_model_is_available():
    class Registry:
        @staticmethod
        def resolve(name):
            return SimpleNamespace(model=name), name

    class Router:
        registry = Registry()

        @staticmethod
        def candidates(_profile, check_availability=False):
            assert check_availability is False
            return ["same-model"]

        @staticmethod
        def complete(*_args, **_kwargs):
            raise AssertionError("router must reject an empty independent route")

    verifier = RoutedClaimVerifier(
        Router(), author_profile="author", verifier_profile="reviewer"
    )
    assert verifier.verify("paraphrase", "source").status == "insufficient"


@pytest.mark.parametrize(
    ("route", "response"),
    [
        (None, '{"status":"supported","quote":"source"}'),
        ("reviewer-model", "[]"),
    ],
)
def test_routed_claim_verifier_rejects_missing_independence_or_invalid_json_shape(
    route, response
):
    class Registry:
        @staticmethod
        def resolve(name):
            return SimpleNamespace(model=name), name

    class Router:
        registry = Registry()

        @staticmethod
        def candidates(profile, check_availability=False):
            assert check_availability is False
            return ["writer-model"] if profile == "author" else ["reviewer-model"]

        @staticmethod
        def complete(_profile, _prompt, *, allowed_models, route_observer):
            assert allowed_models == {"reviewer-model"}
            if route is not None:
                route_observer(route)
            return response

    verifier = RoutedClaimVerifier(
        Router(), author_profile="author", verifier_profile="reviewer"
    )
    assert verifier.verify("paraphrase", "source").status == "insufficient"


def test_routed_claim_verifier_rejects_blank_profile_names():
    with pytest.raises(ValueError, match="profiles must not be blank"):
        RoutedClaimVerifier(object(), author_profile=" ", verifier_profile="reviewer")


def test_same_profile_names_construct_but_never_verify_with_same_model():
    class Registry:
        @staticmethod
        def resolve(name):
            return SimpleNamespace(model=name), name

    class Router:
        registry = Registry()

        @staticmethod
        def candidates(_profile, check_availability=False):
            assert check_availability is False
            return ["same-model"]

        @staticmethod
        def complete(*_args, **_kwargs):
            raise AssertionError("an overlapping verifier route must not run")

    verifier = RoutedClaimVerifier(
        Router(), author_profile="shared", verifier_profile="shared"
    )
    assert verifier.verify("paraphrase", "source").status == "insufficient"


def test_versioned_claim_evidence_goldset_has_zero_exact_rule_errors():
    goldset_path = Path(__file__).parent / "fixtures" / "claim_evidence_goldset_v1.json"
    goldset = json.loads(goldset_path.read_text(encoding="utf-8"))
    assert goldset["version"] == 1
    false_accepts = []
    false_rejects = []
    for case in goldset["cases"]:
        actual = verify_claim(case["claim"], case["source"]).status == "supported"
        if actual and not case["supported"]:
            false_accepts.append(case["id"])
        if case["supported"] and not actual:
            false_rejects.append(case["id"])
    assert false_accepts == []
    assert false_rejects == []
