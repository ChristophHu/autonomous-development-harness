"""Deterministic evaluation helpers for versioned retrieval goldsets."""

from __future__ import annotations


def evaluate_retrieval(expected_refs, retrieved_refs):
    """Return exact-ref precision/recall counts for one goldset case.

    Inputs are ordered reference sequences; duplicates are rejected so a caller
    cannot inflate recall by repeating evidence. Empty denominators are reported
    as 1.0, which makes a correctly empty retrieval case a perfect match.
    """
    expected = _refs(expected_refs, "expected_refs")
    retrieved = _refs(retrieved_refs, "retrieved_refs")
    expected_set = set(expected)
    retrieved_set = set(retrieved)
    relevant = expected_set & retrieved_set
    missing = sorted(expected_set - retrieved_set)
    unexpected = sorted(retrieved_set - expected_set)
    precision = len(relevant) / len(retrieved) if retrieved else float(not expected)
    recall = len(relevant) / len(expected) if expected else float(not retrieved)
    return {
        "expected": len(expected),
        "retrieved": len(retrieved),
        "relevant": len(relevant),
        "missing": missing,
        "unexpected": unexpected,
        "precision": precision,
        "recall": recall,
        "passed": not missing and not unexpected,
    }


def evaluate_context_evidence(
    expected_refs, evidence, *, expected_rejections=None, expected_conflicts=None
):
    """Score the actual ContextBuilder evidence envelope and report rejections."""
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a mapping")
    fragments = evidence.get("fragments")
    if not isinstance(fragments, list):
        raise TypeError("evidence fragments must be a list")
    refs = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            raise TypeError("evidence fragments must be mappings")
        refs.append(fragment.get("ref"))
    rejected = evidence.get("rejected_sources", [])
    if not isinstance(rejected, list):
        raise TypeError("rejected_sources must be a list")
    rejected_refs = []
    rejected_sources = []
    for item in rejected:
        if not isinstance(item, dict) or not isinstance(item.get("ref"), str):
            raise TypeError("rejected source entries must include a reference")
        rejected_refs.append(item["ref"])
        status = item.get("status")
        if not isinstance(status, str) or not status:
            raise TypeError("rejected source entries must include a status")
        rejected_sources.append({"ref": item["ref"], "status": status})
    report = evaluate_retrieval(expected_refs, refs)
    rejection_expectations = []
    for item in expected_rejections or ():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("ref"), str)
            or not isinstance(item.get("status"), str)
        ):
            raise TypeError("expected rejections must include reference and status")
        rejection_expectations.append({"ref": item["ref"], "status": item["status"]})
    expected_conflict_names = _refs(expected_conflicts or (), "expected_conflicts")
    conflicts = evidence.get("conflicts", {})
    if not isinstance(conflicts, dict):
        raise TypeError("evidence conflicts must be a mapping")
    rejection_match = expected_rejections is None or sorted(
        rejected_sources, key=lambda item: (item["ref"], item["status"])
    ) == sorted(rejection_expectations, key=lambda item: (item["ref"], item["status"]))
    conflict_match = expected_conflicts is None or set(conflicts) == set(
        expected_conflict_names
    )
    return {
        **report,
        "passed": report["passed"] and rejection_match and conflict_match,
        "rejected_count": len(rejected_refs),
        "rejected_refs": sorted(rejected_refs),
        "rejected_sources": sorted(rejected_sources, key=lambda item: item["ref"]),
        "rejection_match": rejection_match,
        "conflict_match": conflict_match,
        "conflict_names": sorted(conflicts),
    }


def _refs(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of unique nonempty references")
    try:
        refs = list(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a sequence of unique nonempty references"
        ) from exc
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError(f"{name} must contain only nonempty string references")
    normalized = [ref.strip() for ref in refs]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicate references")
    return normalized
