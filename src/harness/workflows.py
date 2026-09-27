import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .approvals import ApprovalGrant
from .git_policy import git_operation


class Workflow(StrEnum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    HOTFIX = "hotfix"
    RELEASE = "release"
    OTHER = "other"


@dataclass(frozen=True)
class WorkflowDecision:
    workflow: Workflow
    source: str
    evidence: str


class GitWorkflow:
    def __init__(self, tools):
        self.tools = tools

    def branch_name(self, workflow: Workflow, name: str):
        slug = re.sub(r"[^a-z0-9._/-]+", "-", name.lower()).strip("-./")
        branch = f"{workflow.value}/{slug}"
        self._validate_ref(branch)
        return branch

    @staticmethod
    def _validate_ref(ref):
        if (
            not ref
            or ref.startswith("-")
            or ref.endswith((".", ".lock", "/"))
            or ".." in ref
            or "//" in ref
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", ref)
        ):
            raise ValueError(f"invalid Git ref: {ref!r}")
        return ref

    def _git(self, args, cwd, approval=None, task_id=None):
        result = self.tools.git(args, cwd, approval=approval, task_id=task_id)
        if getattr(result, "returncode", 0):
            raise RuntimeError(
                f"git {' '.join(args)} failed ({result.returncode}): {getattr(result, 'stderr', '')}"
            )
        return result

    def create_branch(self, workflow: Workflow, name: str, cwd: str, base=None):
        branch = self.branch_name(workflow, name)
        args = ["switch", "-c", branch]
        if base:
            args.append(self._validate_ref(base))
        return self._git(args, cwd)

    def commit(self, paths: list[str], message: str, cwd: str):
        if not paths or not message.strip():
            raise ValueError("commit requires explicit paths and a non-empty message")
        safe_paths = []
        for path in paths:
            candidate = path.replace("\\", "/")
            if candidate.startswith("/") or any(
                part in {"", ".", ".."} for part in candidate.split("/")
            ):
                raise ValueError("commit paths must stay inside the repository")
            if (Path(cwd) / candidate).is_dir():
                raise ValueError(
                    "commit paths must name explicit files, not directories"
                )
            safe_paths.append(":(literal)" + candidate)
        self._git(["add", "--", *safe_paths], cwd)
        return self._git(
            ["commit", "-m", message.strip(), "--only", "--", *safe_paths], cwd
        )

    def merge(self, branch: str, cwd: str):
        self._validate_ref(branch)
        return self._git(["merge", "--no-ff", "--", branch], cwd)

    def tag_release(self, version: str, cwd: str):
        if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
            raise ValueError(
                "release tag must be a semantic version (for example 1.2.3)"
            )
        tag = version if version.startswith("v") else f"v{version}"
        return self._git(["tag", "-a", tag, "-m", f"Release {tag}"], cwd)

    def delete_branch(
        self, branch: str, cwd: str, approval: ApprovalGrant, task_id=None
    ):
        self._validate_ref(branch)
        if not isinstance(approval, ApprovalGrant):
            raise PermissionError(
                "branch deletion requires a persisted, single-use human approval"
            )
        return self.execute(["branch", "-d", branch], cwd, approval, task_id)

    def classify(self, title: str):
        lower = title.lower()
        return (
            Workflow.RELEASE
            if "release" in lower
            else Workflow.OTHER
            if lower.startswith("other:")
            else Workflow.HOTFIX
            if "hotfix" in lower
            else Workflow.BUGFIX
            if any(marker in lower for marker in ("bug", "fix", "defect"))
            else Workflow.FEATURE
        )

    def classify_task(self, task):
        if task.workflow:
            return WorkflowDecision(
                Workflow(task.workflow), "explicit", "task.workflow is authoritative"
            )
        workflow = self.classify(task.title)
        return WorkflowDecision(
            workflow,
            "title_rule",
            f"title matched deterministic {workflow.value} classification rule",
        )

    def execute(self, args, cwd, approval=None, task_id=None):
        _write, action, _destination = git_operation(args)
        if action:
            if not task_id or not isinstance(approval, ApprovalGrant):
                raise PermissionError(
                    "Git action requires explicit human approval and task"
                )
            target = self.tools.git_target(task_id, args, cwd)
            if not approval.permits(action, target):
                raise PermissionError(
                    "Git action requires matching explicit human approval"
                )
        return self._git(args, cwd, approval=approval, task_id=task_id)
