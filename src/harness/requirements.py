"""Grounded requirement completion before planning or human clarification."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from .decisions import DecisionEvidence, DecisionService
from .domain import EventKind, Task
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
    def __init__(self, store, router):
        self.store, self.router = store, router
        self.decisions = DecisionService(store)

    @staticmethod
    def _ref(source, identifier):
        return f"{source}:{identifier}"

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
                    if name in payload and not getattr(task, name)
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
            prompt = (
                "REQUIREMENTS: Derive only missing fields from these sources; do not invent requirements or commands. "
                "Return JSON with fields, rationale, and evidence mapping each proposed field to one or more "
                "exact {source,ref} citations from the supplied sources. Unsupported fields must be omitted.\n"
                + json.dumps(
                    {
                        "missing": missing,
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
                    self.router.complete("planner", prompt, complexity=task.complexity),
                    agent="requirements",
                )
                fields = response.fields
                citations = response.evidence
                rationale = response.rationale
                updates = {}
                evidence_by_field = {}
                for name in missing:
                    if name not in fields:
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
            },
        )
        return task, json.dumps(sources, default=str)
