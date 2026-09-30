"""Single-use, persisted human approvals for destructive operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

_SEAL = object()


class ApprovalRequired(RuntimeError):
    def __init__(self, question_id):
        super().__init__("human approval is required for this exact tool action")
        self.question_id = question_id


class ApprovalDenied(PermissionError):
    pass


@dataclass(frozen=True)
class GitApprovalTarget:
    task_id: int
    repository: str
    arguments: tuple[str, ...]
    action: str = "branch.delete"
    remote_url: str = ""
    remote_identity: tuple[int, int] | None = None
    ref_updates: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self):
        if self.task_id < 1 or not self.arguments:
            raise ValueError("approval requires task and exact Git arguments")
        object.__setattr__(self, "repository", str(Path(self.repository).resolve()))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(
            self, "ref_updates", tuple(tuple(update) for update in self.ref_updates)
        )

    def reason(self):
        fields = dict(self.__dict__)
        if self.remote_identity is None:
            del fields["remote_identity"]  # Preserve existing branch/repair grants.
        if not self.ref_updates:
            del fields["ref_updates"]
        payload = json.dumps(fields, sort_keys=True)
        return f"approval:{self.action}:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ToolApprovalTarget:
    """Approval bound to one task, tool, canonical argument set and affected paths."""

    task_id: int
    tool: str
    arguments: str
    paths: tuple[str, ...] = ()

    def __post_init__(self):
        if self.task_id < 1 or not self.tool.strip():
            raise ValueError("approval requires a task and exact tool")
        payload = json.loads(self.arguments)
        object.__setattr__(
            self,
            "arguments",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        object.__setattr__(
            self,
            "paths",
            tuple(sorted({str(Path(path).resolve()) for path in self.paths})),
        )

    @classmethod
    def create(cls, task_id, tool, arguments, paths=()):
        return cls(
            task_id,
            tool,
            json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            tuple(paths),
        )

    @property
    def action(self):
        return "tool:" + self.tool

    def reason(self):
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return (
            "approval:"
            + self.action
            + ":"
            + hashlib.sha256(payload.encode()).hexdigest()
        )


@dataclass(frozen=True)
class ApprovalGrant:
    task_id: int
    action: str
    question_id: int
    _seal: object = field(repr=False, compare=False)
    target: GitApprovalTarget | ToolApprovalTarget
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
        valid = (
            isinstance(target, GitApprovalTarget)
            and target.task_id == task_id
            and target.action == action
        ) or (
            isinstance(target, ToolApprovalTarget)
            and target.task_id == task_id
            and target.action == action
        )
        if not valid:
            raise PermissionError(
                "approval requires a matching immutable action target"
            )
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
        if isinstance(target, ToolApprovalTarget):
            safe_target = {
                "tool": target.tool,
                "arguments_sha256": hashlib.sha256(
                    target.arguments.encode()
                ).hexdigest(),
                "paths": target.paths,
            }
            label = "Tool-Freigabe: "
        else:
            safe_target = target.__dict__
            label = "Git-Freigabe: "
        return self.store.ask(
            target.task_id,
            label + json.dumps(safe_target, sort_keys=True),
            target.reason(),
            ["approve", "deny"],
            required=required,
        )

    def request_tool(self, target: ToolApprovalTarget, required=True):
        if not isinstance(target, ToolApprovalTarget):
            raise TypeError("tool approval requires ToolApprovalTarget")
        return self.request(target, required)

    def grant_tool_for_target(self, target: ToolApprovalTarget):
        """Return the exact answered grant, create a question, or fail on denial/replay."""
        if not isinstance(target, ToolApprovalTarget):
            raise TypeError("tool approval requires ToolApprovalTarget")
        matches = [
            row
            for row in self.questions.list(target.task_id)
            if row["reason"] == target.reason()
        ]
        if not matches:
            raise ApprovalRequired(self.request_tool(target))
        question = matches[-1]
        if question["status"] == "open":
            raise ApprovalRequired(question["id"])
        if question["status"] == "answered":
            if question["answer"] != "approve":
                raise ApprovalDenied("human denied this exact tool action")
            return self.issue_tool(target.task_id, question["id"], target)
        if question["status"] in {"consumed", "executed"}:
            raise PermissionError(
                "approval for this exact tool action was already consumed"
            )
        raise PermissionError("approval is not available for this exact tool action")

    def issue_tool(self, task_id, question_id, target):
        if not isinstance(target, ToolApprovalTarget):
            raise PermissionError("exact tool action target is required")
        return self.issue(task_id, question_id, target.action, target)

    def issue(
        self, task_id: int, question_id: int, action: str, target=None
    ) -> ApprovalGrant:
        if not isinstance(target, (GitApprovalTarget, ToolApprovalTarget)) or (
            target.task_id != task_id or target.action != action
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
