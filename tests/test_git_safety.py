import copy
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from test_evidence_workflow import ready_runtime

from harness.approvals import ApprovalGrant, ApprovalService, GitApprovalTarget
from harness.core import Config, Permissions
from harness.git_policy import (
    GitOperation,
    GitRisk,
    describe_git_operation,
    git_operation,
    validate_git_destination,
)
from harness.tools import ToolExecutor


@pytest.mark.parametrize(
    "args,write,action",
    [
        (["status"], False, None),
        (["branch"], False, None),
        (["branch", "--list", "feature/*"], False, None),
        (["branch", "new"], True, None),
        (["branch", "--delete", "topic"], True, "branch.delete"),
        (["branch", "-rd", "origin/topic"], True, "branch.delete"),
        (["branch", "-D", "topic"], True, "branch.delete"),
        (["push", "origin", "--delete", "topic"], True, "branch.delete"),
        (["push", "--delete", "origin", "topic"], True, "branch.delete"),
        (["push", "origin", ":refs/heads/topic"], True, "branch.delete"),
        (["push", "origin", "+:topic"], True, "branch.delete"),
        (["push", "origin", "topic"], True, "git.push"),
        (["remote"], False, None),
        (["remote", "get-url", "origin"], False, None),
        (["remote", "add", "origin", "/remote"], True, None),
        (["tag"], False, None),
        (["tag", "-l"], False, None),
        (["tag", "--list"], False, None),
        (["tag", "v1"], True, None),
    ],
)
def test_git_command_permission_contract(args, write, action):
    result = git_operation(args)
    assert result[:2] == (write, action)


@pytest.mark.parametrize(
    "args",
    [
        [],
        [""],
        [None],
        ["-C", "/other", "branch", "-d", "x"],
        ["config", "alias.rm", "branch -D"],
        ["log", "--output=../outside"],
        ["diff", "--output", "file"],
        ["push", "--mirror", "origin", "x"],
        ["push", "origin", "--prune", "x"],
        ["push"],
        ["push", "ssh://host/repo", "x"],
        ["push", "origin", "refs/heads/*"],
    ],
)
def test_unsafe_or_ambiguous_git_command_is_rejected(args):
    with pytest.raises((PermissionError, ValueError)):
        git_operation(args)


def test_approval_is_immutable_target_bound_and_not_burned_by_wrong_target(
    tmp_path, monkeypatch
):
    store, runtime, task = ready_runtime(tmp_path)
    runtime.tools.permissions.rules["git"] = "write"
    task = store.create(task)
    target = runtime.tools.git_target(task.id, ["branch", "-d", "topic"])
    question_id = runtime.approvals.request(target)
    assert store.answer(question_id, "approve", task.id)
    grant = runtime.approvals.issue(task.id, question_id, target.action, target)
    with pytest.raises(FrozenInstanceError):
        grant.action = "other"
    calls = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: (
            calls.append((args, kwargs)) or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )
    monkeypatch.setenv("GIT_DIR", "/other/repo")
    for args, task_id, cwd in (
        (["branch", "-D", "topic"], task.id, tmp_path),
        (["branch", "-d", "different"], task.id, tmp_path),
        (["branch", "-d", "topic"], task.id + 1, tmp_path),
        (["branch", "-d", "topic"], task.id, tmp_path / "other"),
    ):
        with pytest.raises(PermissionError, match="matching"):
            runtime.tools.executor.git(args, cwd, grant, task_id=task_id)
    assert not calls
    with pytest.raises(PermissionError, match="matching explicit"):
        runtime.git_workflow.execute(
            ["branch", "-D", "topic"], str(tmp_path), grant, task_id=task.id
        )
    from harness.audit import CURRENT_RUN

    token = CURRENT_RUN.set({"task_id": task.id})
    try:
        runtime.tools.git(["branch", "-d", "topic"], approval=grant)
    finally:
        CURRENT_RUN.reset(token)
    assert "GIT_DIR" not in calls[0][1]["env"]
    assert not grant.consume(target.action, target)
    with pytest.raises(PermissionError, match="recorded"):
        runtime.approvals.issue(task.id, question_id, target.action, target)


@pytest.mark.parametrize(
    "stdout,code",
    [
        ("", 0),
        ("", 1),
        ("https://token@host/repo", 0),
        ("ssh://user:password@host/repo", 0),
    ],
)
def test_push_target_rejects_unresolved_or_credential_urls(
    tmp_path, monkeypatch, stdout, code
):
    config = Config()

    def run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, "a" * 40, "")
        return subprocess.CompletedProcess(args, code, stdout, "failed")

    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        run,
    )
    with pytest.raises(PermissionError):
        ToolExecutor(Permissions(config)).git_target(
            1, ["push", "origin", "main:refs/heads/topic"], tmp_path
        )


def test_push_target_rejects_unavailable_approved_source(tmp_path, monkeypatch):
    config = Config()
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "missing"),
    )
    with pytest.raises(PermissionError, match="source branch is unavailable"):
        ToolExecutor(Permissions(config)).git_target(
            1, ["push", "origin", "main:refs/heads/topic"], tmp_path
        )


def test_approval_service_rejects_unbound_and_mismatching_targets(tmp_path):
    store, runtime, task = ready_runtime(tmp_path)
    task = store.create(task)
    target = runtime.tools.git_target(task.id, ["branch", "-d", "topic"])
    with pytest.raises(ValueError):
        GitApprovalTarget(0, str(tmp_path), ())
    with pytest.raises(ValueError, match="task store"):
        ApprovalService(store.questions).request(target)
    for action, candidate in (
        ("branch.delete", None),
        ("other", target),
        (
            "branch.delete",
            GitApprovalTarget(task.id + 1, str(tmp_path), target.arguments),
        ),
    ):
        with pytest.raises(PermissionError, match="exact target"):
            runtime.approvals.issue(task.id, 1, action, candidate)
    from harness import approvals

    with pytest.raises(PermissionError, match="immutable"):
        ApprovalGrant._issue(task.id, target.action, 1, approvals._SEAL)
    with pytest.raises(PermissionError, match="recorded"):
        runtime.approvals.issue(task.id, 9999, target.action, target)


def test_copied_concurrent_grants_share_persisted_single_use_claim(tmp_path):
    store, runtime, task = ready_runtime(tmp_path)
    task = store.create(task)
    target = runtime.tools.git_target(task.id, ["branch", "-d", "topic"])
    question_id = runtime.approvals.request(target)
    assert store.answer(question_id, "approve", task.id)
    grant = runtime.approvals.issue(task.id, question_id, target.action, target)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: item.consume(target.action, target),
                [grant, copy.copy(grant)],
            )
        )
    assert sorted(results) == [False, True]
    assert store.questions.get(question_id)["status"] == "executed"


def test_standard_remote_deletion_never_runs_without_approval(monkeypatch, tmp_path):
    config = Config()
    config.data["tools"] = {"permissions": {"git": "write"}}
    calls = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: (
            calls.append(args) or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )
    with pytest.raises(PermissionError):
        ToolExecutor(Permissions(config)).git(
            ["push", "origin", "--delete", "feature/x"], tmp_path
        )
    assert calls == []


def test_git_read_permission_cannot_commit(monkeypatch, tmp_path):
    config = Config()
    config.data["tools"] = {"permissions": {"git": "read"}}
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda *args, **kwargs: pytest.fail("write reached Git"),
    )
    with pytest.raises(PermissionError):
        ToolExecutor(Permissions(config)).git(["commit", "-m", "bypass"], tmp_path)


@pytest.mark.parametrize(
    "args",
    [
        ["init", "../outside"],
        ["init", "/tmp/outside-repository"],
        ["clone", "https://example.invalid/repo.git", "../outside"],
        ["clone", "https://example.invalid/repo.git", "/tmp/outside-repository"],
        ["clone", "--upload-pack=touch /tmp/pwned", "https://example.invalid/repo.git"],
    ],
)
def test_git_init_and_clone_cannot_escape_workspace(tmp_path, args):
    _store, runtime, _task = ready_runtime(tmp_path)
    runtime.tools.permissions.rules["git"] = "write"
    calls = []
    runtime.tools.executor.git = lambda *values, **kwargs: (
        calls.append(values) or SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    with pytest.raises((PermissionError, ValueError)):
        runtime.tools.git(args)
    assert calls == []


def test_git_init_and_clone_allow_bounded_workspace_destinations(tmp_path):
    _store, runtime, _task = ready_runtime(tmp_path)
    runtime.tools.permissions.rules["git"] = "write"
    calls = []
    runtime.tools.executor.git = lambda args, **kwargs: (
        calls.append(args) or SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    assert (
        runtime.tools.git(["init", "--initial-branch=main", "new-repo"]).returncode == 0
    )
    assert (
        runtime.tools.git(
            ["clone", "https://example.invalid/repo.git", "cloned"]
        ).returncode
        == 0
    )
    assert calls == [
        ["init", "--initial-branch=main", "new-repo"],
        ["clone", "https://example.invalid/repo.git", "cloned"],
    ]


def test_git_clone_rejects_symlink_destination_escape(tmp_path):
    _store, runtime, _task = ready_runtime(tmp_path)
    runtime.tools.permissions.rules["git"] = "write"
    external = tmp_path.parent / "external-repo"
    external.mkdir()
    (runtime.tools.workspace / "link").symlink_to(external, target_is_directory=True)
    calls = []
    runtime.tools.executor.git = lambda *args, **kwargs: calls.append(args)
    with pytest.raises(PermissionError, match="destination"):
        runtime.tools.git(["clone", "https://example.invalid/repo.git", "link/child"])
    assert calls == []


@pytest.mark.parametrize(
    "args,risk,action,remote",
    [
        (["status"], GitRisk.READ, None, None),
        (["commit", "-m", "x"], GitRisk.WRITE, None, None),
        (["branch", "-D", "x"], GitRisk.DESTRUCTIVE, "branch.delete", None),
        (["push", "origin", "main"], GitRisk.WRITE, "git.push", "origin"),
    ],
)
def test_git_operations_have_typed_risk_and_approval_contract(
    args, risk, action, remote
):
    operation = describe_git_operation(args)
    assert isinstance(operation, GitOperation)
    assert (
        operation.command,
        operation.risk,
        operation.approval_action,
        operation.remote,
    ) == (
        args[0],
        risk,
        action,
        remote,
    )


@pytest.mark.parametrize(
    "args",
    [["status"], ["init", "repo"], ["clone", "https://example.invalid/r.git", "r"]],
)
def test_git_destination_contract_is_only_for_init_and_clone(tmp_path, args):
    result = validate_git_destination(args, tmp_path)
    if args[0] == "status":
        assert result is None
    else:
        assert result.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    "args",
    [["clone", "--depth"], ["init", "--template=/tmp/hooks"]],
)
def test_git_init_clone_reject_unsupported_or_incomplete_options(tmp_path, args):
    with pytest.raises((PermissionError, ValueError)):
        validate_git_destination(args, tmp_path)


@pytest.mark.parametrize(
    "args",
    [
        ["init", "--bare"],
        ["init", "-b", "main", "repo"],
        ["init", "--initial-branch", "main", "repo"],
        ["clone", "--single-branch", "--quiet", "--no-checkout", "source.git"],
        ["clone", "--depth", "1", "--branch", "main", "https://host/repo.git"],
        ["clone", "-b", "main", "source.git", "target"],
        ["clone", "--depth=1", "--filter=blob:none", "source.git", "target"],
        ["clone", "https://host/repo.git", "@absolute@"],
    ],
)
def test_git_init_clone_support_bounded_documented_options(tmp_path, args):
    args = [
        str(tmp_path / "absolute") if item == "@absolute@" else item for item in args
    ]
    assert validate_git_destination(args, tmp_path).is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    "args",
    [
        ["init", "--initial-branch="],
        ["init", "-b"],
        ["init", "one", "two"],
        ["clone"],
        ["clone", "-invalid-source"],
        ["clone", ""],
        ["clone", "source", "one", "two"],
        ["clone", "/"],
        ["clone", "--filter="],
        ["clone", "--unknown", "source"],
    ],
)
def test_git_init_clone_reject_malformed_destinations_and_options(tmp_path, args):
    with pytest.raises((PermissionError, ValueError)):
        validate_git_destination(args, tmp_path)


@pytest.mark.parametrize(
    "args",
    [
        ["clone", "ext::sh -c touch${IFS}/tmp/pwned", "safe-name"],
        ["clone", "https://user:secret@example.invalid/repo.git", "safe-name"],
    ],
)
def test_git_clone_rejects_remote_helper_and_inline_credentials(tmp_path, args):
    _store, runtime, _task = ready_runtime(tmp_path)
    runtime.tools.permissions.rules["git"] = "write"
    calls = []
    runtime.tools.executor.git = lambda *values, **kwargs: calls.append(values)
    with pytest.raises(PermissionError):
        runtime.tools.git(args)
    assert calls == []


def test_tool_executor_itself_enforces_init_clone_destination_boundary(
    tmp_path, monkeypatch
):
    config = Config()
    config.data["tools"] = {"permissions": {"git": "write"}}
    calls = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: (
            calls.append(args) or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )
    executor = ToolExecutor(Permissions(config))
    with pytest.raises(PermissionError, match="destination"):
        executor.git(["init", "../outside"], tmp_path)
    assert calls == []
    assert executor.git(["init", "inside"], tmp_path).returncode == 0
    assert calls[0][4:] == ["-c", "core.hooksPath=/dev/null", "init", "inside"]


def test_push_approval_rejects_multiple_actual_destinations(monkeypatch, tmp_path):
    config = Config()
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "/repo/first.git\n/repo/second.git\n", ""
        ),
    )
    with pytest.raises(PermissionError, match="single"):
        ToolExecutor(Permissions(config)).git_target(
            1, ["push", "origin", "--delete", "topic"], tmp_path
        )
