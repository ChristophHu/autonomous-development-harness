import pytest

from harness.knowledge_evaluation import (
    evaluate_context_evidence,
    evaluate_retrieval,
)


def test_retrieval_evaluation_reports_exact_precision_recall_and_gaps():
    report = evaluate_retrieval(["vault:rules", "repo:api"], ["repo:api", "stale"])

    assert report == {
        "expected": 2,
        "retrieved": 2,
        "relevant": 1,
        "missing": ["vault:rules"],
        "unexpected": ["stale"],
        "precision": 0.5,
        "recall": 0.5,
        "passed": False,
    }


@pytest.mark.parametrize(
    "expected,retrieved,precision,recall",
    [
        (["a"], ["a"], 1.0, 1.0),
        ([], [], 1.0, 1.0),
        (["a"], [], 0.0, 0.0),
        ([], ["a"], 0.0, 0.0),
    ],
)
def test_retrieval_evaluation_has_defined_empty_set_semantics(
    expected, retrieved, precision, recall
):
    report = evaluate_retrieval(expected, retrieved)
    assert report["precision"] == precision
    assert report["recall"] == recall
    assert report["passed"] == (expected == retrieved)


@pytest.mark.parametrize(
    "expected,retrieved,message",
    [
        ("a", [], "expected_refs must be a sequence"),
        ([], "a", "retrieved_refs must be a sequence"),
        ([None], [], "expected_refs must contain"),
        (["  "], [], "expected_refs must contain"),
        (["a", " a "], [], "expected_refs must not contain duplicate"),
        ([], ["a", "a"], "retrieved_refs must not contain duplicate"),
        (None, [], "expected_refs must be a sequence"),
    ],
)
def test_retrieval_evaluation_rejects_invalid_or_ambiguous_refs(
    expected, retrieved, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_retrieval(expected, retrieved)


def test_versioned_goldset_scores_real_context_builder_vault_and_qdrant(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from harness.memory import ContextBuilder, ObsidianMemory

    goldset = json.loads(
        (
            Path(__file__).parent / "fixtures" / "knowledge_retrieval_goldset_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert goldset["version"] == 3
    memory = ObsidianMemory(tmp_path / "vault")
    vault_text = "# task-context\nA unique persist-vault-marker decision."
    qdrant_text = "A unique current-vector-marker record."
    memory.write("knowledge/task-context", vault_text)
    memory.write("knowledge/qdrant-current", qdrant_text)

    class FakeQdrant:
        def search(self, query, limit):
            assert limit == 3
            if "current-vector-marker" in query:
                return [
                    {
                        "id": "vector-current",
                        "payload": {
                            "source": "knowledge/qdrant-current",
                            "text": qdrant_text,
                            "source_hash": hashlib.sha256(
                                memory.read("knowledge/qdrant-current").encode()
                            ).hexdigest(),
                        },
                    }
                ]
            if "stale-vector-marker" in query:
                return [
                    {
                        "id": "vector-stale",
                        "payload": {
                            "source": "knowledge/qdrant-current",
                            "text": qdrant_text,
                            "source_hash": "0" * 64,
                        },
                    }
                ]
            if "deleted-vector-marker" in query:
                return [
                    {
                        "id": "vector-deleted",
                        "payload": {
                            "source": "knowledge/deleted-note",
                            "text": "A unique deleted vector marker.",
                            "source_hash": hashlib.sha256(
                                b"A unique deleted vector marker."
                            ).hexdigest(),
                        },
                    }
                ]
            return []

    builder = ContextBuilder(memory, None, FakeQdrant())
    base_refs = [
        "context:task/title",
        "context:task/description",
        "context:task/workspace",
        "context:vault/decisions.md",
    ]
    actual = {}
    for case in goldset["cases"]:
        name = case["id"]
        query = {
            "vault-source": "persist-vault-marker",
            "current-qdrant-source": "current-vector-marker",
            "stale-qdrant-source": "stale-vector-marker",
            "no-match-query": "no-source-should-match",
            "deleted-qdrant-source": "deleted-vector-marker",
        }[name]
        evidence = builder.build_evidence(
            SimpleNamespace(title=query, description="goldset evaluation"),
            str(tmp_path),
        )
        expected = [*base_refs, *case["expected_refs"]]
        report = evaluate_context_evidence(
            expected,
            evidence,
            expected_rejections=case["expected_rejections"],
            expected_conflicts=case["expected_conflicts"],
        )
        actual[name] = report

    assert actual["vault-source"]["passed"] is True, json.dumps(
        actual["vault-source"], indent=2
    )
    assert actual["current-qdrant-source"]["passed"] is True, json.dumps(
        actual["current-qdrant-source"], indent=2
    )
    assert actual["stale-qdrant-source"]["passed"] is True
    assert actual["stale-qdrant-source"]["rejected_refs"] == [
        "context:qdrant/vector-stale"
    ]
    assert actual["no-match-query"]["passed"] is True
    assert actual["no-match-query"]["precision"] == 1.0
    assert actual["no-match-query"]["recall"] == 1.0
    assert actual["deleted-qdrant-source"]["passed"] is True, json.dumps(
        actual["deleted-qdrant-source"], indent=2
    )
    assert actual["deleted-qdrant-source"]["rejected_refs"] == [
        "context:qdrant/vector-deleted"
    ]
    assert actual["deleted-qdrant-source"]["rejected_count"] == 1
    assert actual["deleted-qdrant-source"]["rejected_sources"] == [
        {"ref": "context:qdrant/vector-deleted", "status": "source_unavailable"}
    ]


def test_context_goldset_detects_wrong_rejection_status_or_conflict():
    evidence = {
        "fragments": [{"ref": "vault:current"}],
        "rejected_sources": [{"ref": "vault:stale", "status": "source_hash_stale"}],
        "conflicts": {"goal": {"values": ["a", "b"]}},
    }
    expected = [{"ref": "vault:stale", "status": "source_hash_stale"}]
    good = evaluate_context_evidence(
        ["vault:current"],
        evidence,
        expected_rejections=expected,
        expected_conflicts=["goal"],
    )
    assert good["passed"] is True
    assert good["rejection_match"] is True
    assert good["conflict_match"] is True
    wrong_status = evaluate_context_evidence(
        ["vault:current"],
        evidence,
        expected_rejections=[{"ref": "vault:stale", "status": "stale"}],
        expected_conflicts=["goal"],
    )
    assert wrong_status["passed"] is False
    assert wrong_status["rejection_match"] is False
    missing_conflict = evaluate_context_evidence(
        ["vault:current"], evidence, expected_conflicts=[]
    )
    assert missing_conflict["passed"] is False
    assert missing_conflict["conflict_match"] is False


@pytest.mark.parametrize(
    "kwargs,evidence,message",
    [
        ({"expected_rejections": [None]}, {"fragments": []}, "expected rejections"),
        (
            {"expected_rejections": [{"ref": "x"}]},
            {"fragments": []},
            "expected rejections",
        ),
        ({"expected_conflicts": ["x", "x"]}, {"fragments": []}, "duplicate"),
        ({}, {"fragments": [], "conflicts": []}, "conflicts must be a mapping"),
    ],
)
def test_context_goldset_contracts(kwargs, evidence, message):
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_context_evidence([], evidence, **kwargs)


def test_goldset_detects_retrieval_false_positive_and_stale_duplicate():
    report = evaluate_retrieval(["vault:source"], ["vault:source", "stale:source"])
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0
    assert report["unexpected"] == ["stale:source"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_retrieval(["vault:source"], ["vault:source", "vault:source"])


@pytest.mark.parametrize(
    "evidence,error,message",
    [
        (None, TypeError, "mapping"),
        ({}, TypeError, "fragments"),
        ({"fragments": [None]}, TypeError, "mappings"),
        ({"fragments": [], "rejected_sources": {}}, TypeError, "must be a list"),
        (
            {"fragments": [], "rejected_sources": [None]},
            TypeError,
            "include a reference",
        ),
        (
            {"fragments": [], "rejected_sources": [{"ref": "qdrant:x"}]},
            TypeError,
            "include a status",
        ),
    ],
)
def test_context_evaluation_rejects_malformed_evidence(evidence, error, message):
    with pytest.raises(error, match=message):
        evaluate_context_evidence([], evidence)
