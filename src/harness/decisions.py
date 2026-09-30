"""Typed, append-only decisions with validated source provenance."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: Literal["task", "parent", "answer", "decision", "context"]
    ref: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def reference_matches_source(self):
        if not self.ref.startswith(f"{self.source}:"):
            raise ValueError("evidence reference must match its source")
        return self


class DecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: int | None
    category: Literal[
        "requirement_resolution",
        "human_choice",
        "architecture",
        "implementation_constraint",
        "operational",
    ]
    source: Literal["human", "agent", "task", "parent", "memory", "repository"]
    question_id: int | None = None
    decision: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=4000)
    field_names: list[str] = Field(default_factory=list, max_length=32)
    evidence: list[DecisionEvidence] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_decision_contract(self):
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) for name in self.field_names
        ):
            raise ValueError("decision field names must be valid identifiers")
        if len(self.field_names) != len(set(self.field_names)):
            raise ValueError("decision field names must be unique")
        if self.category == "requirement_resolution" and not self.field_names:
            raise ValueError("requirement decisions must name the resolved fields")
        if self.source == "human" and self.question_id is None:
            raise ValueError("human decisions require a question reference")
        return self


class Decision(DecisionInput):
    id: int = Field(gt=0)
    created_at: str = Field(min_length=1)


class DecisionService:
    def __init__(self, store):
        self.store = store
        self.repository = store.decisions

    def _validate_evidence(self, decision):
        task = (
            self.store.get(decision.task_id) if decision.task_id is not None else None
        )
        if decision.task_id is not None and task is None:
            raise ValueError("task not found")
        if decision.question_id is not None:
            question = self.store.get_question(decision.question_id)
            if question is None or question["task_id"] != decision.task_id:
                raise ValueError("question does not belong to decision task")
            if question["status"] not in {"answered", "consumed", "executed"}:
                raise ValueError("decision question is not answered")
        answer_refs = set()
        for item in decision.evidence:
            identifier = item.ref.partition(":")[2]
            if item.source in {"task", "parent", "answer", "decision"}:
                if not identifier.isdecimal():
                    raise ValueError("evidence reference ID must be an integer")
                reference_id = int(identifier)
            else:
                continue
            if item.source == "task":
                if decision.task_id != reference_id:
                    raise ValueError("task evidence does not match decision task")
            elif item.source == "parent":
                if task is None or task.parent_task_id != reference_id:
                    raise ValueError("parent evidence does not match task parent")
            elif item.source == "answer":
                question = self.store.get_question(reference_id)
                if (
                    question is None
                    or question["task_id"] != decision.task_id
                    or question["status"] not in {"answered", "consumed", "executed"}
                ):
                    raise ValueError("answer evidence is unavailable for decision task")
                answer_refs.add(reference_id)
            else:
                prior = self.repository.get(reference_id)
                if prior is None or prior["task_id"] not in {None, decision.task_id}:
                    raise ValueError(
                        "decision evidence is unavailable for decision task"
                    )
        if decision.source == "human" and decision.question_id not in answer_refs:
            raise ValueError("human decision evidence does not match its question")

    def record(self, payload):
        decision = DecisionInput.model_validate(payload)
        self._validate_evidence(decision)
        safe_decision = self.store.audit.sanitize(decision.decision)
        safe_rationale = self.store.audit.sanitize(decision.rationale)
        decision_id = self.repository.record(
            decision.task_id,
            decision.category,
            decision.source,
            safe_decision,
            safe_rationale,
            [item.model_dump() for item in decision.evidence],
            decision.field_names,
            decision.question_id,
        )
        return Decision.model_validate(self.repository.get(decision_id))

    def list(self, task_id):
        return [Decision.model_validate(row) for row in self.repository.list(task_id)]
