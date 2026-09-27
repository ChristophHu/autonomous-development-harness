"""Single-use, persisted human approvals for destructive operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

_SEAL = object()


@dataclass(frozen=True)
class GitApprovalTarget:
    task_id: int
    repository: str
    arguments: tuple[str, ...]
    action: str = "branch.delete"
    remote_url: str = ""

    def __post_init__(self):
        if self.task_id < 1 or not self.arguments:
            raise ValueError("approval requires task and exact Git arguments")
        object.__setattr__(self, "repository", str(Path(self.repository).resolve()))
        object.__setattr__(self, "arguments", tuple(self.arguments))

    def reason(self):
        payload = json.dumps(self.__dict__, sort_keys=True)
        return f"approval:{self.action}:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ApprovalGrant:
    task_id: int
    action: str
    question_id: int
    _seal: object = field(repr=False, compare=False)
    target: GitApprovalTarget
    _used: bool = field(default=False, repr=False, compare=False)
    _consume: Callable[[int], bool] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def _issue(
        cls,
        task_id: int,
        action: str,
        question_id: int,
        seal: object,
        target=None,
        consumer=None,
    ):
        if seal is not _SEAL:
            raise PermissionError(
                "approval grants can only be issued by ApprovalService"
            )
        if (
            not isinstance(target, GitApprovalTarget)
            or target.task_id != task_id
            or target.action != action
        ):
            raise PermissionError("approval requires a matching immutable Git target")
        return cls(task_id, action, question_id, seal, target, _consume=consumer)

    def permits(self, action: str, target=None) -> bool:
        return (
            self._seal is _SEAL
            and self.action == action
            and self.target == target
            and not self._used
        )

    def consume(self, action: str, target=None) -> bool:
        if not self.permits(action, target):
            return False
        if self._consume is not None and not self._consume(self.question_id):
            return False
        object.__setattr__(self, "_used", True)
        return True


class ApprovalService:
    def __init__(self, questions, store=None):
        self.questions = questions
        self.store = store

    def request(self, target, required=False):
        if self.store is None:
            raise ValueError("approval requests require a task store")
        return self.store.ask(
            target.task_id,
            "Git-Freigabe: " + json.dumps(target.__dict__),
            target.reason(),
            ["approve", "deny"],
            required=required,
        )

    def issue(
        self, task_id: int, question_id: int, action: str, target=None
    ) -> ApprovalGrant:
        if (
            not isinstance(target, GitApprovalTarget)
            or target.task_id != task_id
            or target.action != action
        ):
            raise PermissionError(
                "matching recorded human approval requires an exact target"
            )
        question = self.questions.get(question_id)
        if (
            not question
            or question["task_id"] != task_id
            or question["reason"] != target.reason()
            or question["status"] != "answered"
            or question["answer"] != "approve"
        ):
            raise PermissionError("matching recorded human approval is required")
        if not self.questions.consume_answer(question_id):
            raise PermissionError("approval has already been consumed")
        return ApprovalGrant._issue(
            task_id,
            action,
            question_id,
            _SEAL,
            target,
            partial(self.questions.consume_approval, reason=target.reason()),
        )

    def resume_issue(self, task_id, question_id, action, target):
        """Reconstitute an unconsumed grant after a crash, without replaying approval."""
        question = self.questions.get(question_id)
        if (
            not isinstance(target, GitApprovalTarget)
            or target.task_id != task_id
            or target.action != action
            or not question
            or question["task_id"] != task_id
            or question["reason"] != target.reason()
            or question["status"] != "consumed"
            or question["answer"] != "approve"
        ):
            raise PermissionError("unconsumed matching approval is required")
        return ApprovalGrant._issue(
            task_id,
            action,
            question_id,
            _SEAL,
            target,
            partial(self.questions.consume_approval, reason=target.reason()),
        )
