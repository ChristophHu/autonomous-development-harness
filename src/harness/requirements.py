"""Grounded requirement completion before planning or human clarification."""

import hashlib
import json
import stat
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from .agents import ProfileRegistry
from .claim_evidence import verify_claim
from .decisions import DecisionEvidence, DecisionService
from .domain import EventKind, Task
from .memory import context_evidence_envelope, context_evidence_size
from .structured_output import parse_model_output


class RequirementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fields: dict[str, object]
    rationale: StrictStr
    evidence: dict[str, list[DecisionEvidence]]

    @model_validator(mode="after")
    def restrict_proposed_fields(self):
        if not self.rationale.strip():
            raise ValueError("requirement rationale must not be blank")
        allowed = set(FIELDS)
        if not set(self.fields) <= allowed or not set(self.evidence) <= allowed:
            raise ValueError("requirement output contains an unknown field")
        return self


FIELDS = (
    "goal",
    "requirements",
    "acceptance_criteria",
    "test_commands",
    "coverage_command",
)


class RequirementCompleter:
    def __init__(self, store, router, claim_verifier=None):
        self.store, self.router = store, router
        self.claim_verifier = claim_verifier
        self.decisions = DecisionService(store)
        self.profile = ProfileRegistry(router.config).for_role("requirements")

    @staticmethod
    def _ref(source, identifier):
        return f"{source}:{identifier}"

    def _repository_fragment_current(self, fragment):
        reference = fragment.get("ref")
        provenance = fragment.get("provenance")
        prefix = "context:repository/"
        if (
            not isinstance(reference, str)
            or not reference.startswith(prefix)
            or not isinstance(provenance, dict)
        ):
            return False
        relative = reference[len(prefix) :].split("#", 1)[0]
        expected = provenance.get("sha256")
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            return False
        try:
            root = Path(self.store.config.path("workspace")).resolve(strict=True)
            candidate = root / relative
            current = root
            for part in Path(relative).parts:
                current = current / part
                if current.is_symlink():
                    return False
            resolved = candidate.resolve(strict=True)
            if (
                not resolved.is_relative_to(root)
                or not stat.S_ISREG(resolved.stat().st_mode)
                or resolved.stat().st_size > 512_000
            ):
                return False
            with resolved.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
        except (OSError, TypeError, ValueError):
            return False
        return actual == expected

    def complete(self, task, context):
        task_ref = self._ref("task", task.id)
        sources = [
            {
                "source": "task",
                "ref": task_ref,
                "data": task.model_dump(mode="json"),
            }
        ]
        known_evidence = {task_ref: "task"}
        if task.parent_task_id:
            parent = self.store.get(task.parent_task_id)
            if parent:
                parent_ref = self._ref("parent", parent.id)
                known_evidence[parent_ref] = "parent"
                sources.append(
                    {
                        "source": "parent",
                        "ref": parent_ref,
                        "data": parent.model_dump(mode="json"),
                    }
                )
        answers = [
            dict(row)
            for row in self.store.list_questions(task.id)
            if row["status"] != "open"
        ]
        for answer in answers:
            answer_ref = self._ref("answer", answer["id"])
            answer["source_ref"] = answer_ref
            known_evidence[answer_ref] = "answer"
        sources.append({"source": "answers", "data": answers})
        decisions = [dict(row) for row in self.decisions.repository.list(task.id)]
        for decision in decisions:
            decision_ref = self._ref("decision", decision["id"])
            decision["source_ref"] = decision_ref
            known_evidence[decision_ref] = "decision"
        sources.append({"source": "decisions", "data": decisions})
        retrieved = context() if callable(context) else context
        context_claims = {}
        conflicts = {}
        context_refs = set()
        context_text_by_ref = {}
        claim_integrity_issues = {}
        supplied_claim_issues = {}
        if isinstance(retrieved, dict) and isinstance(retrieved.get("fragments"), list):
            valid_fragments = []
            stale_repository_refs = []
            for fragment in retrieved["fragments"]:
                if (
                    isinstance(fragment, dict)
                    and fragment.get("kind") == "repository_symbol"
                ):
                    if self._repository_fragment_current(fragment):
                        valid_fragments.append(fragment)
                    else:
                        stale_repository_refs.append(
                            {
                                "ref": fragment.get("ref", ""),
                                "status": "stale_or_unavailable",
                            }
                        )
                else:
                    valid_fragments.append(fragment)
            if stale_repository_refs:
                retrieved = {
                    **retrieved,
                    "fragments": valid_fragments,
                    "rejected_sources": [
                        *retrieved.get("rejected_sources", []),
                        *stale_repository_refs,
                    ],
                }
            incoming_budget = retrieved.get("budget_bytes")
            incoming_size = context_evidence_size(retrieved)
            if incoming_budget is not None and (
                isinstance(incoming_budget, bool)
                or not isinstance(incoming_budget, int)
                or incoming_budget < 1
                or incoming_size > incoming_budget
            ):
                raise ValueError("context evidence exceeds its declared byte budget")
            for fragment in retrieved["fragments"]:
                if not isinstance(fragment, dict):
                    continue
                reference = fragment.get("ref")
                if not isinstance(reference, str) or not reference.startswith(
                    "context:"
                ):
                    continue
                known_evidence[reference] = "context"
                context_refs.add(reference)
                if isinstance(fragment.get("text"), str):
                    context_text_by_ref[reference] = fragment["text"]
            context_claims = retrieved.get("claims", {})
            supplied_claim_issues = retrieved.get("claim_issues", {})
            retrieved = context_evidence_envelope(retrieved)
            retrieved["conflicts"] = {}
        claims_by_field = {}
        if isinstance(context_claims, dict):
            claim_bytes = 0
            claim_count = 0
            for name, entries in context_claims.items():
                if name not in FIELDS:
                    continue
                if not isinstance(entries, list):
                    claim_integrity_issues[name] = ["claims must be a list"]
                    continue
                claims_by_field.setdefault(name, [])
                for item in entries:
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("ref"), str)
                        or item.get("ref") not in context_refs
                        or "value" not in item
                    ):
                        claim_integrity_issues.setdefault(name, []).append(
                            "claim reference is absent from supplied context fragments"
                        )
                        continue
                    try:
                        encoded = json.dumps(
                            item["value"],
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    except (TypeError, ValueError):
                        claim_integrity_issues.setdefault(name, []).append(
                            "claim value is not valid JSON"
                        )
                        continue
                    if len(encoded.encode("utf-8")) > 4096:
                        claim_integrity_issues.setdefault(name, []).append(
                            "claim value exceeds the evidence size limit"
                        )
                        continue
                    verdict = verify_claim(
                        item["value"],
                        context_text_by_ref.get(item["ref"]),
                        verifier=self.claim_verifier,
                    )
                    if verdict.status != "supported":
                        claim_integrity_issues.setdefault(name, []).append(
                            "claim lacks exact support in its cited source"
                        )
                        continue
                    item = {
                        **item,
                        "evidence_quote": verdict.quote,
                        "verification_method": (
                            "independent"
                            if verdict.reason == "independent semantic verdict"
                            else "exact"
                        ),
                    }
                    if verdict.reason == "independent semantic verdict":
                        item["verifier_id"] = (
                            verdict.verifier_id or self.claim_verifier.identity
                        )
                    encoded_size = len(encoded.encode("utf-8"))
                    if claim_count >= 32 or claim_bytes + encoded_size > 8192:
                        claim_integrity_issues.setdefault(name, []).append(
                            "context claim payload exceeds the evidence limit"
                        )
                        continue
                    claims_by_field[name].append(item)
                    claim_count += 1
                    claim_bytes += encoded_size
        if isinstance(supplied_claim_issues, dict):
            for name, issues in supplied_claim_issues.items():
                if name not in FIELDS:
                    continue
                if isinstance(issues, list):
                    claim_integrity_issues.setdefault(name, []).extend(
                        item[:256] for item in issues[:32] if isinstance(item, str)
                    )
                else:
                    claim_integrity_issues.setdefault(name, []).append(
                        "claim integrity metadata is invalid"
                    )
        for answer in answers:
            if answer.get("reason") != "requirements:incomplete":
                continue
            try:
                payload = json.loads(answer["answer"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                for name in FIELDS:
                    if name in payload:
                        try:
                            value_size = len(
                                json.dumps(payload[name], allow_nan=False).encode(
                                    "utf-8"
                                )
                            )
                        except (TypeError, ValueError):
                            claim_integrity_issues.setdefault(name, []).append(
                                "human claim is not valid JSON"
                            )
                            continue
                        if value_size > 4096:
                            claim_integrity_issues.setdefault(name, []).append(
                                "human claim exceeds the evidence size limit"
                            )
                            continue
                        claims_by_field.setdefault(name, []).append(
                            {"ref": answer["source_ref"], "value": payload[name]}
                        )
        conflicts = {}
        for name, entries in claims_by_field.items():
            distinct = {
                json.dumps(item.get("value"), sort_keys=True, default=str)
                for item in entries
                if "value" in item
            }
            if len(distinct) > 1:
                conflicts[name] = {"sources": entries}
        for name, issues in claim_integrity_issues.items():
            conflicts[name] = {"integrity_issues": issues}
        if isinstance(retrieved, dict) and "fragments" in retrieved:
            retrieved["claims"] = claims_by_field
            retrieved["claim_issues"] = claim_integrity_issues
            retrieved["conflicts"] = conflicts
            if "budget_bytes" in retrieved:
                retrieved["used_bytes"] = context_evidence_size(retrieved)
                if retrieved["used_bytes"] > retrieved["budget_bytes"]:
                    raise ValueError(
                        "validated context evidence exceeds its byte budget"
                    )
        detected_conflict_fields = {
            name for name in conflicts if not getattr(task, name)
        }

        resolved_fields = set()
        for answer in answers:
            reason = answer.get("reason", "")
            if (
                not reason.startswith("requirements:conflict:")
                or answer.get("purpose") != "decision"
                or answer.get("status") != "answered"
            ):
                continue
            name = reason.split(":", 3)[2]
            if name not in conflicts or name not in FIELDS:
                continue
            try:
                if len(answer["answer"].encode("utf-8")) > 8192:
                    continue
                selection = json.loads(answer["answer"])
            except (TypeError, ValueError):
                continue
            if not isinstance(selection, dict):
                continue
            candidate = (
                next(
                    (
                        entry
                        for entry in claims_by_field.get(name, [])
                        if set(selection) == {"ref", "value"}
                        if entry.get("ref") == selection.get("ref")
                        and entry.get("value") == selection.get("value")
                    ),
                    None,
                )
                if isinstance(selection, dict)
                else None
            )
            if name not in FIELDS or getattr(task, name):
                continue
            if candidate is None and (
                set(selection) != {"value", "rationale"}
                or not isinstance(selection.get("rationale"), str)
                or not selection["rationale"].strip()
                or len(selection["rationale"]) > 4000
                or "value" not in selection
                or claim_integrity_issues.get(name)
            ):
                continue
            selected_value = (
                candidate["value"] if candidate is not None else selection["value"]
            )
            rationale = (
                "The human selected one of the currently cited conflicting claims."
                if candidate is not None
                else selection["rationale"].strip()
            )
            try:
                if (
                    len(json.dumps(selected_value, allow_nan=False).encode("utf-8"))
                    > 4096
                ):
                    continue
                updated = Task.model_validate(
                    task.model_dump() | {name: selected_value}
                )
            except (ValueError, TypeError):
                continue
            if getattr(updated, name) != getattr(task, name):
                task = updated
                resolved_fields.add(name)
                self.decisions.record(
                    {
                        "task_id": task.id,
                        "category": "requirement_resolution",
                        "source": "human",
                        "question_id": answer["id"],
                        "decision": f"Resolved conflicting requirement: {name}",
                        "rationale": rationale,
                        "field_names": [name],
                        "evidence": [
                            {
                                "source": "answer",
                                "ref": answer["source_ref"],
                            },
                            *(
                                [
                                    {
                                        "source": candidate["ref"].partition(":")[0],
                                        "ref": candidate["ref"],
                                    }
                                ]
                                if candidate is not None
                                else []
                            ),
                        ],
                    }
                )

        current_conflict_reasons = set()
        for name in detected_conflict_fields - resolved_fields:
            if name not in claims_by_field or claim_integrity_issues.get(name):
                continue
            candidates = claims_by_field[name]
            options = [
                json.dumps(
                    {"ref": item["ref"], "value": item["value"]},
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for item in candidates
            ]
            reason_digest = hashlib.sha256("\n".join(options).encode()).hexdigest()[:12]
            current_conflict_reasons.add(
                f"requirements:conflict:{name}:{reason_digest}"
            )
            rendered = "\n".join(
                f"- {item['ref']}: {json.dumps(item['value'], ensure_ascii=False, default=str)}"
                for item in candidates
            )
            question = (
                f"Widersprüchliche Belege für '{name}'. Antworte mit JSON "
                '{"ref": <Quellenreferenz>, "value": <belegter Wert>} '
                'oder {"value": <neuer Wert>, "rationale": <Begründung>}.\n' + rendered
            )
            self.store.ask(
                task.id,
                question,
                f"requirements:conflict:{name}:{reason_digest}",
                required=True,
                purpose="decision",
            )

        self.store.questions.supersede_conflict_questions(
            task.id, current_conflict_reasons
        )

        conflict_fields = detected_conflict_fields - resolved_fields
        sources.append({"source": "context_conflicts", "fields": conflicts})
        context_digest = hashlib.sha256(
            json.dumps(retrieved, sort_keys=True, default=str).encode()
        ).hexdigest()
        context_ref = self._ref("context", context_digest)
        known_evidence[context_ref] = "context"
        sources.append(
            {
                "source": "memory_and_repository",
                "ref": context_ref,
                "data": retrieved,
            }
        )

        for answer in answers:
            if answer["reason"] != "requirements:incomplete":
                continue
            try:
                payload = json.loads(answer["answer"])
                updates = {
                    name: payload[name]
                    for name in FIELDS
                    if name in payload
                    and not getattr(task, name)
                    and name not in detected_conflict_fields
                }
                updated = Task.model_validate(task.model_dump() | updates)
            except (ValueError, TypeError):
                continue
            changed_fields = [
                name
                for name in updates
                if getattr(task, name) != getattr(updated, name)
            ]
            if changed_fields:
                task = updated
                self.decisions.record(
                    {
                        "task_id": task.id,
                        "category": "requirement_resolution",
                        "source": "human",
                        "question_id": answer["id"],
                        "decision": "Requirements supplied by human answer",
                        "rationale": "An answered requirements question supplied these fields.",
                        "field_names": changed_fields,
                        "evidence": [
                            {
                                "source": "answer",
                                "ref": self._ref("answer", answer["id"]),
                            }
                        ],
                    }
                )

        missing = [name for name in FIELDS if not getattr(task, name)]
        if missing:
            contract_input = {"task": task.model_dump(mode="json"), "sources": sources}
            self.profile.validate_input(contract_input)
            prompt = (
                "REQUIREMENTS: Derive only missing fields from these sources; do not invent requirements or commands. "
                "Return JSON with fields, rationale, and evidence mapping each proposed field to one or more "
                "exact {source,ref} citations from the supplied sources. Unsupported fields must be omitted.\n"
                + json.dumps(
                    {
                        "missing": missing,
                        "blocked_by_conflict": conflicts,
                        "sources": sources,
                        "allowed_evidence": [
                            {"source": source, "ref": ref}
                            for ref, source in known_evidence.items()
                        ],
                    },
                    default=str,
                )
            )
            try:
                response = parse_model_output(
                    RequirementProposal,
                    self.router.complete(
                        self.profile.name, prompt, complexity=task.complexity
                    ),
                    agent="requirements",
                )
                self.profile.validate_output(response.model_dump(mode="json"))
                fields = response.fields
                citations = response.evidence
                rationale = response.rationale
                updates = {}
                evidence_by_field = {}
                for name in missing:
                    if name not in fields or name in conflict_fields:
                        continue
                    refs = citations.get(name, [])
                    if not refs or any(
                        known_evidence.get(item.ref) != item.source for item in refs
                    ):
                        continue
                    updates[name] = fields[name]
                    evidence_by_field[name] = refs
                if updates:
                    updated = Task.model_validate(task.model_dump() | updates)
                    changed_fields = [
                        name
                        for name in updates
                        if getattr(task, name) != getattr(updated, name)
                    ]
                    if changed_fields:
                        evidence = {
                            (item.source, item.ref): item
                            for name in changed_fields
                            for item in evidence_by_field[name]
                        }
                        task = updated
                        self.decisions.record(
                            {
                                "task_id": task.id,
                                "category": "requirement_resolution",
                                "source": "agent",
                                "decision": "Requirements derived for fields: "
                                + ", ".join(changed_fields),
                                "rationale": rationale,
                                "field_names": changed_fields,
                                "evidence": [
                                    item.model_dump() for item in evidence.values()
                                ],
                            }
                        )
            except (RuntimeError, ValueError, TypeError):
                pass
        self.store.update_task_fields(task.id, metadata=task.model_dump(mode="json"))
        self.store.event(
            task.id,
            EventKind.REQUIREMENTS_INSPECTED,
            {
                "sources": [source["source"] for source in sources],
                "missing": [name for name in FIELDS if not getattr(task, name)],
                "conflicts": sorted(conflict_fields),
                "resolved_conflicts": sorted(resolved_fields),
            },
        )
        return task, json.dumps(sources, default=str)
