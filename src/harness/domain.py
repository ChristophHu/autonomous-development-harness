"""Typed task contracts and explicit lifecycle transitions."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    READY = "ready"
    EXECUTING = "executing"
    TESTING = "testing"
    VALIDATING = "validating"
    CORRECTING = "correcting"
    RECOVERING = "recovering"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    """Supported event names for new writes; historic database values stay readable."""

    TASK_CREATED = "task.created"
    TASK_STATUS = "task.status"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_BLOCKED = "task.blocked"
    TASK_FAILED = "task.failed"
    TASK_LEASE_LOST = "task.lease_lost"
    REQUIREMENTS_INSPECTED = "requirements.inspected"
    QUESTION_ASKED = "QUESTION_ASKED"
    QUESTION_ANSWERED = "question.answered"
    DECISION_RECORDED = "decision.recorded"
    RECOVERY_RECONCILED = "recovery.reconciled"
    RECOVERY_INSPECTED = "recovery.inspected"
    RECOVERY_SCOPE = "recovery.scope"
    MEMORY_INDEX_FAILED = "memory.index_failed"
    TESTS_COMPLETED = "tests.completed"
    CORRECTION_STARTED = "correction.started"
    CORRECTION_ITEM_RECORDED = "correction.item_recorded"
    CORRECTION_ITEM_STATUS = "correction.item_status"
    GIT_WORKFLOW = "git.workflow"
    GIT_WORKFLOW_CLASSIFIED = "git.workflow_classified"
    GIT_REPAIR_REQUESTED = "git.repair_requested"
    GIT_REPAIR_DECLINED = "git.repair_declined"
    GIT_REPAIR_AUTHORIZED = "git.repair_authorized"
    GIT_RECONCILED = "git.reconciled"
    GIT_TARGET_TESTS = "git.target_tests"
    GIT_TARGET_VALIDATION = "git.target_validation"
    AGENT_RUN = "agent.run"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_COMPLETED = "TOOL_CALL_COMPLETED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"


TERMINAL = {Status.COMPLETED, Status.CANCELLED}
TRANSITIONS = {
    Status.PENDING: {Status.ANALYZING},
    Status.ANALYZING: {Status.PLANNING},
    Status.PLANNING: {Status.READY},
    Status.READY: {Status.EXECUTING},
    Status.EXECUTING: {Status.TESTING, Status.CORRECTING},
    Status.TESTING: {Status.VALIDATING},
    Status.VALIDATING: {Status.COMPLETED, Status.CORRECTING},
    Status.CORRECTING: {Status.EXECUTING, Status.PLANNING},
    Status.RECOVERING: {Status.ANALYZING},
    Status.WAITING_HUMAN: {Status.ANALYZING, Status.RECOVERING},
    Status.FAILED: {Status.ANALYZING, Status.RECOVERING},
    Status.BLOCKED: {Status.ANALYZING, Status.RECOVERING},
}


def may_transition(source, target):
    source, target = Status(source), Status(target)
    if source in TERMINAL:
        return False
    if target in {
        Status.CANCELLED,
        Status.FAILED,
        Status.BLOCKED,
        Status.WAITING_HUMAN,
    }:
        return True
    return target in TRANSITIONS.get(source, set())


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: Literal["review", "file_exists", "file_contains", "command"] = "review"
    path: str = ""
    contains: str = ""
    command: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def executable_contract(self):
        if self.kind.startswith("file_") and not self.path:
            raise ValueError("file acceptance criterion requires path")
        if self.kind == "file_contains" and not self.contains:
            raise ValueError("file_contains requires expected content")
        if self.kind == "command" and not self.command:
            raise ValueError("command criterion requires an argument list")
        return self


class Task(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "title": "Implement feature",
                    "goal": "Add the requested feature",
                    "requirements": ["Use the existing architecture"],
                }
            ]
        },
    )
    id: int | None = None
    parent_task_id: int | None = None
    title: str = Field(min_length=1)
    description: str = ""
    goal: str = ""
    priority: int = 0
    status: Status = Status.PENDING
    created_at: str = ""
    updated_at: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    dependencies: list[int] = Field(default_factory=list)
    assigned_agent: str | None = None
    assigned_profile: str | None = None
    complexity: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    test_result: dict[str, Any] | None = None
    test_commands: list[list[str]] = Field(default_factory=list)
    lint_commands: list[list[str]] = Field(default_factory=list)
    coverage_command: list[str] = Field(default_factory=list)
    coverage_report: str = "coverage.json"
    coverage_threshold: float = Field(default=100, ge=0, le=100)
    result: str | None = None
    workflow: Literal["feature", "bugfix", "hotfix", "release", "other"] | None = None
    release_version: str = ""
    git_state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_acceptance_ids(self):
        ids = [criterion.id for criterion in self.acceptance_criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion IDs must be unique")
        if not self.title.strip():
            raise ValueError("task title must not be blank")
        return self
