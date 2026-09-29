import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_git_integration import git_runtime

from harness.git_broker import (
    LocalTransport,
    configure_push_upstream,
    fetch_plan,
    local_path,
    push_branch,
    push_execution_args,
    push_refs,
    push_upstream_branch,
    reconcile_push_upstream,
    record_tracking_verification,
    stop_group,
    transport_index,
    verify_push_tracking,
)
from harness.isolation import isolated_command, trusted_git_metadata


def native(git, cwd, *args):
    return subprocess.run(
        [git, *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_local_clone_fetch_and_fast_forward_pull(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    clone = repo / "clone"
    result = harness.tools.git(["clone", str(repo), "clone"])
    assert result.returncode == 0, result.stderr
    assert (clone / "addition.py").exists()
    (repo / "new.txt").write_text("update")
    native(git, repo, "add", "new.txt")
    native(git, repo, "commit", "-m", "update")
    result = harness.tools.git(["fetch", "origin", "main"], cwd="clone")
    assert result.returncode == 0, result.stderr
    result = harness.tools.git(["pull", "--ff-only", "origin", "main"], cwd="clone")
    assert result.returncode == 0, result.stderr
    assert (clone / "new.txt").read_text() == "update"


def test_fetch_plan_normalizes_multiple_branch_refs_and_prune():
    options, refspecs = fetch_plan(
        ["fetch", "--prune", "--quiet", "origin", "main", "refs/heads/release/v1"],
        3,
    )
    assert options == ("--prune", "--quiet")
    assert refspecs == (
        "+refs/heads/main:refs/remotes/origin/main",
        "+refs/heads/release/v1:refs/remotes/origin/release/v1",
    )


def test_fetch_plan_supports_all_heads_and_tag_options():
    assert fetch_plan(["fetch", "--tags", "origin"], 2) == (
        ("--tags",),
        ("+refs/heads/*:refs/remotes/origin/*",),
    )
    assert fetch_plan(["fetch", "origin", "refs/tags/v1"], 1) == (
        (),
        ("refs/tags/v1:refs/tags/v1",),
    )


@pytest.mark.parametrize(
    "args",
    [
        ["fetch", "--tags", "--prune", "origin"],
        ["fetch", "--prune", "origin", "refs/tags/v1"],
        ["fetch", "origin", "refs/remotes/origin/main"],
        ["fetch", "origin", "main", "main"],
        ["fetch", "origin", "main*"],
        ["pull", "--prune", "--ff-only", "origin", "main"],
    ],
)
def test_fetch_plan_rejects_ambiguous_or_unsafe_ref_actions(args):
    with pytest.raises(PermissionError):
        fetch_plan(args, transport_index(args))


@pytest.mark.parametrize(
    ("args", "index"),
    [
        (["push", "origin"], 1),
        (["fetch", "../outside"], 1),
        (["fetch", "--force", "origin"], 2),
        (["fetch", "--quiet", "--quiet", "origin"], 3),
        (["fetch", "--tags", "--no-tags", "origin"], 3),
        (["pull", "origin", "main"], 1),
        (["fetch", "origin", ""], 1),
        (["fetch", "origin", "--upload-pack=evil"], 1),
        (["fetch", "origin", "refs/remotes/origin/main"], 1),
        (["fetch", "origin", "bad..name"], 1),
        (["fetch", "origin", "main?"], 1),
        (["fetch", "origin", "main[0]"], 1),
        (["fetch", "origin", "main", "refs/heads/main"], 1),
        (["fetch", "--prune", "origin", "refs/tags/v1"], 2),
        (["pull", "--ff-only", "origin", "refs/tags/v1"], 2),
    ],
)
def test_fetch_plan_rejects_each_unsupported_contract(args, index):
    with pytest.raises(PermissionError):
        fetch_plan(args, index)


def test_local_fetch_updates_multiple_branches_tags_and_prunes_tracking_only(
    tmp_path, monkeypatch
):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    native(git, repo, "branch", "topic/second")
    clone_result = harness.tools.git(["clone", str(repo), "clone"])
    assert clone_result.returncode == 0, clone_result.stderr
    clone = repo / "clone"

    native(git, repo, "tag", "release-v1")
    fetch = harness.tools.git(
        ["fetch", "--no-tags", "origin", "main", "topic/second"], cwd="clone"
    )
    assert fetch.returncode == 0, fetch.stderr
    assert native(git, clone, "rev-parse", "refs/remotes/origin/main")
    assert native(git, clone, "rev-parse", "refs/remotes/origin/topic/second")

    tagged = harness.tools.git(["fetch", "--tags", "origin"], cwd="clone")
    assert tagged.returncode == 0, tagged.stderr
    assert native(git, clone, "rev-parse", "refs/tags/release-v1")

    native(git, clone, "branch", "local/keep")
    native(git, repo, "branch", "-D", "topic/second")
    pruned = harness.tools.git(["fetch", "--prune", "origin"], cwd="clone")
    assert pruned.returncode == 0, pruned.stderr
    assert native(git, clone, "show-ref", "--verify", "refs/remotes/origin/main")
    assert (
        subprocess.run(
            [git, "show-ref", "--verify", "refs/remotes/origin/topic/second"],
            cwd=clone,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )
    assert native(git, clone, "show-ref", "--verify", "refs/heads/local/keep")
    assert native(git, clone, "show-ref", "--verify", "refs/tags/release-v1")


def test_linked_worktree_can_commit_without_relaxing_shell(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    native(git, repo, "worktree", "add", "linked", "-b", "linked")
    linked = repo / "linked"
    (linked / "new.txt").write_text("worktree")
    result = harness.tools.git(["add", "new.txt"], cwd="linked")
    assert result.returncode == 0, result.stderr
    result = harness.tools.git(["commit", "-m", "worktree"], cwd="linked")
    assert result.returncode == 0, result.stderr
    result = harness.tools.executor.shell(
        [
            "python3",
            "-c",
            "from pathlib import Path; Path('../.git/refs/heads/linked').unlink()",
        ],
        linked,
    )
    assert result.returncode != 0
    assert native(git, repo, "rev-parse", "linked") == native(
        git, linked, "rev-parse", "HEAD"
    )


def test_preflight_failure_does_not_consume_approval(tmp_path, monkeypatch):
    store, harness, task, _git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    args = ["branch", "-D", "dev"]
    target = harness.tools.git_target(task.id, args)
    qid = harness.approvals.request(target)
    store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    original = harness.tools.executor.git
    monkeypatch.setattr(
        "harness.tools.isolated_command",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("preflight unavailable")),
    )
    with pytest.raises(PermissionError, match="preflight"):
        original(args, repo, grant, task.id)
    assert grant.permits(target.action, target)


@pytest.mark.parametrize(
    "url",
    [
        "https://host/repo",
        "ext::evil",
        "file://other/repo",
        "file:///repo?query",
        "file:///repo#fragment",
    ],
)
def test_nonlocal_and_ambiguous_transport_urls_are_denied(tmp_path, url):
    with pytest.raises(PermissionError, match="local Git"):
        local_path(url, tmp_path)


def test_local_file_urls_and_relative_paths_are_canonical(tmp_path):
    assert local_path("relative.git", tmp_path) == tmp_path / "relative.git"
    assert (
        local_path("file://localhost" + str(tmp_path) + "/has%20space", tmp_path)
        == tmp_path / "has space"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["fetch"],
        ["fetch", "--upload-pack=evil", "origin"],
        ["pull", "--rebase", "origin"],
    ],
)
def test_transport_options_cannot_select_arbitrary_helpers(args):
    with pytest.raises(PermissionError):
        transport_index(args)


def test_broker_clone_without_destination_and_default_fetch(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    source = tmp_path / "source.git"
    native(git, tmp_path, "clone", "--bare", str(repo), str(source))
    result = harness.tools.git(
        ["clone", "--quiet", "--depth", "1", "--branch", "main", source.as_uri()]
    )
    assert result.returncode == 0, result.stderr
    target = repo / "source"
    assert native(git, target, "remote", "get-url", "origin") == str(source)
    result = harness.tools.git(["fetch", "origin"], cwd="source")
    assert result.returncode == 0, result.stderr
    assert native(git, target, "rev-parse", "refs/remotes/origin/main") == native(
        git, repo, "rev-parse", "main"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["fetch", "/arbitrary"],
        ["pull", "origin", "main"],
        ["pull", "--ff-only", "origin"],
        ["fetch", "origin", "../escape"],
    ],
)
def test_invalid_remote_and_pull_contracts_are_denied(tmp_path, args):
    with pytest.raises((PermissionError, ValueError)):
        LocalTransport.prepare(args, tmp_path, {})


def test_missing_and_invalid_git_remote_are_denied(tmp_path):
    with pytest.raises(PermissionError, match="unavailable"):
        LocalTransport.prepare(["clone", "missing", "dest"], tmp_path, {})
    with pytest.raises(PermissionError, match="verified"):
        LocalTransport.prepare(["clone", str(tmp_path), "dest"], tmp_path, {})


def test_push_to_working_repository_is_denied(tmp_path, monkeypatch):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="bare"):
        LocalTransport.prepare(
            ["push", "origin", "main:refs/heads/main"],
            repo,
            harness.tools.executor._environment(),
            str(repo),
        )


@pytest.mark.parametrize("problem", ["missing", "different_image"])
def test_remote_fd_helper_must_be_the_trusted_git_image(tmp_path, monkeypatch, problem):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    if problem == "missing":
        original = Path.is_file
        monkeypatch.setattr(
            Path,
            "is_file",
            lambda path: False if path.name == "git-remote-fd" else original(path),
        )
    else:
        original = Path.resolve
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda path, *a, **k: (
                Path("/usr/bin/false")
                if path.name == "git-remote-fd"
                else original(path, *a, **k)
            ),
        )
    with pytest.raises(PermissionError, match="remote-fd"):
        LocalTransport.prepare(
            ["clone", str(repo), "dest"], repo, harness.tools.executor._environment()
        )


def test_failed_kernel_probe_preserves_grant(tmp_path, monkeypatch):
    store, harness, task, _git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    args = ["branch", "-D", "dev"]
    target = harness.tools.git_target(task.id, args)
    qid = harness.approvals.request(target)
    store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 71, "", "sandbox_apply denied"
        ),
    )
    with pytest.raises(PermissionError, match="preflight"):
        harness.tools.executor.git(args, repo, grant, task.id)
    assert grant.permits(target.action, target)
    assert store.questions.get(qid)["status"] == "consumed"


def test_claim_lost_after_probe_does_not_execute(tmp_path, monkeypatch):
    from harness.approvals import _SEAL, ApprovalGrant, GitApprovalTarget
    from harness.core import Config, Permissions
    from harness.tools import ToolExecutor

    cfg = Config()
    cfg.data["tools"] = {"permissions": {"git": "write"}}
    args = ["branch", "-D", "topic"]
    target = GitApprovalTarget(1, str(tmp_path), tuple(args))
    grant = ApprovalGrant._issue(1, target.action, 1, _SEAL, target, lambda _: False)
    calls = []
    monkeypatch.setattr(
        "harness.tools.subprocess.run",
        lambda command, **kw: (
            calls.append(command) or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )
    with pytest.raises(PermissionError, match="matching"):
        ToolExecutor(Permissions(cfg)).git(args, tmp_path, grant, 1)
    assert len(calls) == 1 and calls[0][-1] == "--version"


def test_remote_replaced_after_preflight_is_denied(tmp_path, monkeypatch):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    plan = LocalTransport.prepare(
        ["clone", str(repo), "dest"], tmp_path, harness.tools.executor._environment()
    )
    repo.rename(tmp_path / "old")
    repo.mkdir()
    with pytest.raises(PermissionError, match="identity"):
        plan.run()


@pytest.mark.parametrize(
    "problem", ["symlink", "pointer", "backlink", "common", "layout"]
)
def test_unverified_worktree_metadata_is_denied(tmp_path, monkeypatch, problem):
    _store, _harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    linked = repo / "linked"
    native(git, repo, "worktree", "add", str(linked), "-b", "linked")
    metadata = repo / ".git/worktrees/linked"
    if problem == "symlink":
        (linked / ".git").unlink()
        (linked / ".git").symlink_to(metadata)
    elif problem == "pointer":
        (linked / ".git").write_text("malformed")
    elif problem == "backlink":
        (metadata / "gitdir").write_text(str(repo / "other"))
    elif problem == "common":
        (metadata / "commondir").write_text(str(repo))
    else:
        (repo / ".git/HEAD").unlink()
    with pytest.raises(PermissionError):
        trusted_git_metadata(linked)


def test_bare_receiver_cannot_inherit_parent_metadata_writes(tmp_path, monkeypatch):
    _store, _harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    bare = repo / "nested.git"
    native(git, repo, "init", "--bare", str(bare))
    assert trusted_git_metadata(bare) == (bare,)
    command = isolated_command(["git", "--version"], bare, git=True)
    assert str(repo / ".git") not in command[2]


def test_stop_group_handles_concurrent_exit(monkeypatch):
    waits = []
    process = SimpleNamespace(
        pid=123, poll=lambda: None, wait=lambda: waits.append(True)
    )
    monkeypatch.setattr(
        "harness.git_broker.os.killpg",
        lambda *a: (_ for _ in ()).throw(ProcessLookupError()),
    )
    stop_group(process)
    assert waits == [True]


def approved_push(tmp_path, monkeypatch, args=None):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    native(git, tmp_path, "init", "--bare", str(remote))
    native(git, repo, "remote", "add", "origin", str(remote))
    task = store.create(task)
    args = args or ["push", "origin", "main:refs/heads/topic"]
    target = harness.tools.git_target(task.id, args)
    qid = harness.approvals.request(target)
    store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    return store, harness, task, git, repo, remote, args, target, grant


def test_approved_push_disables_hooks_on_both_sides(tmp_path, monkeypatch):
    store, harness, task, git, repo, remote, args, _target, grant = approved_push(
        tmp_path, monkeypatch
    )
    for hook in (repo / ".git/hooks/pre-push", remote / "hooks/pre-receive"):
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\necho corrupt > HEAD\necho corrupt > ../outside\n")
        hook.chmod(0o755)
    original = (remote / "HEAD").read_text()
    result = harness.tools.git(args, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    assert native(git, remote, "rev-parse", "topic") == native(
        git, repo, "rev-parse", "main"
    )
    assert (remote / "HEAD").read_text() == original
    assert not (tmp_path / "outside").exists()
    assert store.questions.get(grant.question_id)["status"] == "executed"


def test_successful_push_refreshes_named_remote_tracking_ref(tmp_path, monkeypatch):
    _store, harness, task, git, repo, remote, args, _target, grant = approved_push(
        tmp_path, monkeypatch
    )
    result = harness.tools.git(args, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    assert native(git, repo, "rev-parse", "refs/remotes/origin/topic") == native(
        git, remote, "rev-parse", "refs/heads/topic"
    )


@pytest.mark.parametrize("flag", ["-u", "--set-upstream"])
def test_approved_push_can_set_upstream_for_one_explicit_branch(
    tmp_path, monkeypatch, flag
):
    _store, harness, task, git, repo, remote, args, _target, grant = approved_push(
        tmp_path, monkeypatch, ["push", flag, "origin", "main:refs/heads/topic"]
    )
    result = harness.tools.git(args, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    assert native(git, repo, "config", "branch.main.remote") == "origin"
    assert native(git, repo, "config", "branch.main.merge") == "refs/heads/topic"
    assert native(git, repo, "rev-parse", "refs/remotes/origin/topic") == native(
        git, remote, "rev-parse", "refs/heads/topic"
    )


def test_push_delete_removes_only_its_named_tracking_ref(tmp_path, monkeypatch):
    store, harness, task, git, repo, remote, args, _target, grant = approved_push(
        tmp_path, monkeypatch
    )
    assert harness.tools.git(args, approval=grant, task_id=task.id).returncode == 0
    native(git, repo, "update-ref", "refs/remotes/origin/keep", "HEAD")
    delete = ["push", "origin", "--delete", "topic"]
    target = harness.tools.git_target(task.id, delete)
    qid = harness.approvals.request(target)
    store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    result = harness.tools.git(delete, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    assert (
        subprocess.run(
            [git, "rev-parse", "--verify", "refs/remotes/origin/topic"],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )
    assert native(git, repo, "rev-parse", "refs/remotes/origin/keep") == native(
        git, repo, "rev-parse", "HEAD"
    )
    assert (
        subprocess.run(
            [git, "rev-parse", "--verify", "refs/heads/topic"],
            cwd=remote,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )


@pytest.mark.parametrize(
    "args",
    [
        ["push", "origin", "main", "dev"],
        ["push", "origin", "main"],
        ["push", "origin", ":"],
        ["push", "origin", "main:refs/tags/unsafe"],
        ["push", "origin", "main:topic"],
        ["push", "origin", "--force-with-lease", "main"],
        ["push", "origin", "--tags", "main"],
    ],
)
def test_unbounded_push_contract_fails_before_approval_can_be_requested(
    tmp_path, monkeypatch, args
):
    _store, harness, task, _git, _repo, _remote, _old, _target, _grant = approved_push(
        tmp_path, monkeypatch
    )
    with pytest.raises(PermissionError, match="push branch"):
        harness.tools.git_target(task.id, args)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["push", "origin", "HEAD:refs/heads/topic"], ("topic", False)),
        (["push", "origin", "+main:refs/heads/topic"], ("topic", False)),
        (["push", "origin", ":refs/heads/topic"], ("topic", True)),
        (["push", "origin", "+:topic"], ("topic", True)),
        (["push", "origin", "--delete", "topic"], ("topic", True)),
    ],
)
def test_bounded_push_refspec_resolution(args, expected):
    assert push_branch(args, 1) == expected


def test_multi_push_plan_normalizes_explicit_branch_updates():
    assert push_refs(
        [
            "push",
            "origin",
            "main:refs/heads/stable",
            "refs/heads/release:refs/heads/release",
        ],
        1,
    ) == (("main", "stable"), ("release", "release"))


def test_multi_push_execution_is_atomic_and_uses_approved_object_ids():
    args, index = push_execution_args(
        ["push", "origin", "main:refs/heads/stable", "dev:refs/heads/dev"],
        1,
        (("main", "stable"), ("dev", "dev")),
        (("main", "stable", "a" * 40), ("dev", "dev", "b" * 40)),
    )
    assert args == [
        "push",
        "--atomic",
        "origin",
        f"{'a' * 40}:refs/heads/stable",
        f"{'b' * 40}:refs/heads/dev",
    ]
    assert index == 2


def test_multi_push_execution_rejects_mismatch_with_approved_refs():
    with pytest.raises(PermissionError, match="approved source commits"):
        push_execution_args(
            ["push", "origin", "main:refs/heads/stable"],
            1,
            (("main", "stable"),),
            (("main", "other", "a" * 40),),
        )


@pytest.mark.parametrize(
    "refspecs",
    [
        ("main:refs/heads/a", "dev:refs/heads/a"),
        ("main:refs/heads/a", "main:refs/heads/b"),
        ("HEAD:refs/heads/a", "dev:refs/heads/b"),
        ("+main:refs/heads/a", "dev:refs/heads/b"),
        ("main:refs/heads/a", "dev:refs/tags/b"),
        ("main:refs/heads/a", "dev:refs/heads/*"),
        ("main:refs/heads/a", ":refs/heads/b"),
    ],
)
def test_multi_push_plan_rejects_ambiguous_or_unsafe_refspecs(refspecs):
    with pytest.raises(PermissionError, match="push branch"):
        push_refs(["push", "origin", *refspecs], 1)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["push", "origin"], "exactly one refspec"),
        (["push", "origin", "main:refs/heads/main", "--unknown"], "option"),
        (["push", "origin", "--delete", "--set-upstream", "main"], "upstream"),
        (["push", "origin", "main"], "explicit source and destination"),
    ],
)
def test_legacy_single_branch_parser_error_contracts(args, message):
    with pytest.raises(PermissionError, match=message):
        push_branch(args, 1)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["push", "origin"], "explicit refspecs"),
        (["push", "-u", "origin", "main:refs/heads/a", "dev:refs/heads/b"], "upstream"),
        (["push", "origin", "bad..name:refs/heads/a"], "source is invalid"),
        (["push", "origin", "main:refs/heads/bad..name"], "destination is invalid"),
    ],
)
def test_multi_push_parser_reports_invalid_contracts(args, message):
    index = 2 if args[1] == "-u" else 1
    with pytest.raises(PermissionError, match=message):
        push_refs(args, index)


def test_tracking_verifier_ignores_deletion_targets():
    verify_push_tracking(((None, "removed"),), "origin", Path("."), {})


def test_tracking_verifier_requires_remote_oid_to_match_approval(tmp_path, monkeypatch):
    outputs = iter(("a" * 40, "b" * 40))

    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, next(outputs), "")

    monkeypatch.setattr("harness.git_broker.subprocess.run", run)
    with pytest.raises(RuntimeError, match="does not match"):
        verify_push_tracking(
            (("main", "stable"),),
            "origin",
            tmp_path,
            {},
            (("main", "stable", "c" * 40),),
        )


def test_tracking_verification_failure_is_reported_after_successful_push(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "harness.git_broker.verify_push_tracking",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("mismatch")),
    )
    result = subprocess.CompletedProcess(["git", "push"], 0, "", "")
    reported = record_tracking_verification(
        result,
        (("main", "stable"),),
        "origin",
        tmp_path,
        {},
        (("main", "stable", "a" * 40),),
    )
    assert reported.returncode == 1
    assert "tracking verification failed: mismatch" in reported.stderr


def test_approved_multi_ref_push_reconciles_every_tracking_ref(tmp_path, monkeypatch):
    store, harness, task, git, repo, remote, _old, _target, _grant = approved_push(
        tmp_path, monkeypatch
    )
    native(git, repo, "switch", "dev")
    (repo / "dev-only.txt").write_text("development branch\n")
    native(git, repo, "add", "dev-only.txt")
    native(git, repo, "commit", "-m", "development branch")
    native(git, repo, "switch", "main")
    args = ["push", "origin", "main:refs/heads/main", "dev:refs/heads/dev"]
    target = harness.tools.git_target(task.id, args)
    question_id = harness.approvals.request(target)
    store.answer(question_id, "approve", task.id)
    grant = harness.approvals.issue(task.id, question_id, target.action, target)
    result = harness.tools.git(args, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    approved_oids = {
        (source, destination): oid
        for source, destination, oid in grant.target.ref_updates
    }
    for branch in ("main", "dev"):
        remote_oid = native(git, remote, "rev-parse", f"refs/heads/{branch}")
        tracked_oid = native(git, repo, "rev-parse", f"refs/remotes/origin/{branch}")
        assert remote_oid == tracked_oid
        source = "dev" if branch == "dev" else "main"
        assert remote_oid == approved_oids[(source, branch)]


def test_push_approval_is_invalidated_when_approved_source_commit_changes(
    tmp_path, monkeypatch
):
    _store, harness, task, git, repo, _remote, args, target, grant = approved_push(
        tmp_path, monkeypatch
    )
    (repo / "after-approval.txt").write_text("changed after approval\n")
    native(git, repo, "add", "after-approval.txt")
    native(git, repo, "commit", "-m", "change approved source")
    with pytest.raises(PermissionError, match="matching recorded human approval"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert grant.permits(target.action, target)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["push", "-u", "origin", "main:refs/heads/topic"], "main"),
        (
            ["push", "origin", "--set-upstream", "refs/heads/main:refs/heads/topic"],
            "main",
        ),
        (["push", "origin", "main:refs/heads/topic"], None),
    ],
)
def test_upstream_requires_and_resolves_named_source(args, expected):
    index = 2 if args[1] == "-u" else 1
    assert push_upstream_branch(args, index) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["push", "-u", "origin", "HEAD:refs/heads/topic"],
        ["push", "-u", "origin", "main:refs/heads/../topic"],
        ["push", "-u", "origin", "bad..name:refs/heads/topic"],
        ["push", "--set-upstream", "origin", "--delete", "topic"],
    ],
)
def test_upstream_rejects_implicit_invalid_or_deleted_sources(args):
    with pytest.raises(PermissionError):
        from harness.git_broker import validate_push_upstream

        index = 2 if args[1] in {"-u", "--set-upstream"} else 1
        validate_push_upstream(args, index, push_branch(args, index))


def test_upstream_rejects_missing_source_and_delete_flag():
    with pytest.raises(PermissionError, match="exactly one explicit branch"):
        push_upstream_branch(["push", "-u", "origin"], 2)
    with pytest.raises(PermissionError, match="cannot be configured"):
        push_branch(["push", "origin", "-u", "--delete", "topic"], 1)
    with pytest.raises(PermissionError, match="cannot be configured"):
        configure_push_upstream(
            ["push", "-u", "origin", "main:refs/heads/topic"],
            2,
            ("topic", True),
            Path.cwd(),
            {},
        )


def test_upstream_wraps_invalid_source_ref(monkeypatch):
    from harness.workflows import GitWorkflow

    def reject_ref(_ref):
        raise ValueError("invalid ref")

    monkeypatch.setattr(GitWorkflow, "_validate_ref", reject_ref)
    with pytest.raises(PermissionError, match="source branch is invalid"):
        push_upstream_branch(["push", "-u", "origin", "main:refs/heads/topic"], 2)


def test_upstream_reconciliation_reports_configuration_failures(monkeypatch, tmp_path):
    args = ["push", "-u", "origin", "main:refs/heads/topic"]
    successful_push = SimpleNamespace(returncode=0, stdout="", stderr="")

    def raise_config(*_args, **_kwargs):
        raise OSError("git config unavailable")

    monkeypatch.setattr("harness.git_broker.subprocess.run", raise_config)
    result = reconcile_push_upstream(
        successful_push, args, 2, ("topic", False), tmp_path, {}
    )
    assert result.returncode == 1
    assert "push succeeded but upstream setup failed" in result.stderr

    failed_config = SimpleNamespace(returncode=128, stdout="", stderr="rejected")
    monkeypatch.setattr(
        "harness.git_broker.subprocess.run", lambda *_args, **_kwargs: failed_config
    )
    result = reconcile_push_upstream(
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        args,
        2,
        ("topic", False),
        tmp_path,
        {},
    )
    assert result.returncode == 128
    assert "rejected" in result.stderr
    assert "push succeeded but upstream setup failed" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["push", "origin", "--delete", "main:topic"],
        ["push", "origin", ":"],
        ["push", "origin", "HEAD:refs/remotes/other/topic"],
        ["push", "origin", ":refs/remotes/other/topic"],
        ["push", "origin", "HEAD:refs/heads/../topic"],
        ["push", "origin", "bad..name:refs/heads/topic"],
    ],
)
def test_bounded_push_refspec_rejects_invalid_branches(args):
    with pytest.raises(PermissionError, match="push branch"):
        push_branch(args, 1)


def test_push_reconciliation_reports_remote_mutation_before_refresh_failure(
    tmp_path, monkeypatch
):
    plan = LocalTransport(
        ("push", "origin", "main:refs/heads/topic"),
        tmp_path,
        tmp_path,
        1,
        (),
        {},
        (1, 2),
        ("topic", False),
    )
    push = subprocess.CompletedProcess(list(plan.args), 0, "pushed", "")
    monkeypatch.setattr(
        LocalTransport,
        "prepare",
        lambda *args, **kwargs: SimpleNamespace(identity=(3, 4)),
    )
    with pytest.raises(RuntimeError, match="push succeeded.*remote identity"):
        plan._reconcile_push(push)

    monkeypatch.setattr(
        LocalTransport,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("refresh preflight denied")
        ),
    )
    with pytest.raises(RuntimeError, match="push succeeded.*refresh preflight denied"):
        plan._reconcile_push(push)

    monkeypatch.setattr(
        LocalTransport,
        "prepare",
        lambda *args, **kwargs: SimpleNamespace(
            identity=(1, 2),
            run=lambda: subprocess.CompletedProcess([], 73, "", "refresh denied"),
        ),
    )
    result = plan._reconcile_push(push)
    assert result.returncode == 73
    assert "push succeeded but local tracking reconciliation failed" in result.stderr


def test_unsupported_network_does_not_burn_push_grant(tmp_path, monkeypatch):
    _store, harness, task, git, repo, _remote, args, _target, _grant = approved_push(
        tmp_path, monkeypatch
    )
    native(
        git,
        repo,
        "remote",
        "set-url",
        "--push",
        "origin",
        "https://example.invalid/remote.git",
    )
    target = harness.tools.git_target(task.id, args)
    qid = harness.approvals.request(target)
    harness.store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    with pytest.raises(PermissionError, match="allowlisted"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert grant.permits(target.action, target)


@pytest.mark.parametrize("change", ["replace_directory", "retarget_symlink"])
def test_changed_local_remote_identity_cannot_use_prior_approval(
    tmp_path, monkeypatch, change
):
    store, harness, task, git, repo, remote, args, target, grant = approved_push(
        tmp_path, monkeypatch
    )
    if change == "replace_directory":
        remote.rename(tmp_path / "old.git")
        native(git, tmp_path, "init", "--bare", str(remote))
    else:
        link = tmp_path / "alias.git"
        link.symlink_to(remote, target_is_directory=True)
        native(git, repo, "remote", "set-url", "--push", "origin", str(link))
        assert harness.tools.git_target(task.id, args) == target
        alternate = tmp_path / "other.git"
        native(git, tmp_path, "init", "--bare", str(alternate))
        link.unlink()
        link.symlink_to(alternate, target_is_directory=True)
    with pytest.raises(PermissionError, match="matching"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert grant.permits(target.action, target)
    assert store.questions.get(grant.question_id)["status"] == "consumed"


def test_file_push_url_has_same_canonical_identity(tmp_path, monkeypatch):
    _store, harness, task, git, repo, remote, args, target, _grant = approved_push(
        tmp_path, monkeypatch
    )
    native(git, repo, "remote", "set-url", "--push", "origin", remote.as_uri())
    assert harness.tools.git_target(task.id, args) == target


def test_upload_pack_cannot_run_configured_pack_hook(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    native(git, repo, "config", "uploadpack.packObjectsHook", "/usr/bin/touch marker")
    result = harness.tools.git(["clone", str(repo), "dest"])
    assert result.returncode == 0, result.stderr
    assert not (repo / "marker").exists()
    assert (repo / "dest/addition.py").exists()


def test_missing_clone_branch_is_reported_as_failure(tmp_path, monkeypatch):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    result = harness.tools.git(["clone", "--branch", "absent", str(repo), "dest"])
    assert result.returncode != 0
    assert "absent" in result.stderr


def test_pull_refuses_divergence_without_discarding_changes(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    assert harness.tools.git(["clone", str(repo), "dest"]).returncode == 0
    clone = repo / "dest"
    native(git, clone, "config", "user.name", "Test")
    native(git, clone, "config", "user.email", "test@example.invalid")
    for directory, name in ((clone, "local"), (repo, "remote")):
        (directory / name).write_text(name)
        native(git, directory, "add", name)
        native(git, directory, "commit", "-m", name)
    before = native(git, clone, "rev-parse", "HEAD")
    result = harness.tools.git(["pull", "--ff-only", "origin", "main"], cwd="dest")
    assert result.returncode != 0
    assert native(git, clone, "rev-parse", "HEAD") == before
    assert (clone / "local").read_text() == "local"


def test_client_timeout_terminates_both_process_groups(tmp_path, monkeypatch):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    plan = LocalTransport.prepare(
        ["clone", str(repo), "dest"], repo, harness.tools.executor._environment()
    )
    started = []
    original = subprocess.Popen

    def launch(*a, **kw):
        process = original(*a, **kw)
        started.append(process)
        if len(started) == 2:
            process.communicate = lambda **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(a[0], 120)
            )
        return process

    monkeypatch.setattr("harness.git_broker.subprocess.Popen", launch)
    with pytest.raises(subprocess.TimeoutExpired):
        plan.run()
    assert len(started) == 2
    assert all(process.poll() is not None for process in started)
    assert started[0].stdin.closed and started[0].stdout.closed


def test_server_failure_is_not_hidden_by_client_success(tmp_path, monkeypatch):
    _store, harness, _task, _git, repo = git_runtime(tmp_path, monkeypatch)
    plan = LocalTransport.prepare(
        ["clone", str(repo), "dest"], repo, harness.tools.executor._environment()
    )
    started = []
    original = subprocess.Popen

    def launch(*a, **kw):
        process = original(*a, **kw)
        started.append(process)
        if len(started) == 1:
            wait = process.wait

            def failed_wait(*args, **kwargs):
                wait(*args, **kwargs)
                process.returncode = 70
                return 70

            process.wait = failed_wait
        return process

    monkeypatch.setattr("harness.git_broker.subprocess.Popen", launch)
    result = plan.run()
    assert result.returncode == 70


def test_interruption_after_push_never_replays_consumed_grant(tmp_path, monkeypatch):
    store, harness, task, git, repo, remote, args, target, grant = approved_push(
        tmp_path, monkeypatch
    )
    original = LocalTransport.run

    def interrupted(plan):
        result = original(plan)
        assert result.returncode == 0
        raise RuntimeError("interrupted after remote mutation")

    monkeypatch.setattr(LocalTransport, "run", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert native(git, remote, "rev-parse", "topic") == native(
        git, repo, "rev-parse", "main"
    )
    assert store.questions.get(grant.question_id)["status"] == "executed"
    with pytest.raises(PermissionError, match="matching"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    with pytest.raises(PermissionError, match="unconsumed"):
        harness.approvals.resume_issue(
            task.id, grant.question_id, target.action, target
        )


def test_remote_directory_swap_at_spawn_cannot_redirect_push(tmp_path, monkeypatch):
    _store, harness, _task, git, repo, remote, args, _target, _grant = approved_push(
        tmp_path, monkeypatch
    )
    alternate = tmp_path / "alternate.git"
    native(git, tmp_path, "init", "--bare", str(alternate))
    plan = LocalTransport.prepare(
        args, repo, harness.tools.executor._environment(), str(remote)
    )
    original = subprocess.Popen
    swapped = []

    def launch(*a, **kw):
        if not swapped:
            swapped.append(True)
            remote.rename(tmp_path / "original.git")
            alternate.rename(remote)
        return original(*a, **kw)

    monkeypatch.setattr("harness.git_broker.subprocess.Popen", launch)
    result = plan.run()
    assert result.returncode != 0
    refs = subprocess.run(
        [git, "show-ref"], cwd=remote, capture_output=True, text=True, check=False
    )
    assert refs.returncode == 1 and not refs.stdout


def test_remote_changed_during_prepare_preserves_approval(tmp_path, monkeypatch):
    store, harness, task, git, _repo, remote, args, target, grant = approved_push(
        tmp_path, monkeypatch
    )
    alternate = tmp_path / "alternate.git"
    native(git, tmp_path, "init", "--bare", str(alternate))
    original = LocalTransport.prepare

    def changed(*a, **kw):
        remote.rename(tmp_path / "old.git")
        alternate.rename(remote)
        return original(*a, **kw)

    monkeypatch.setattr(LocalTransport, "prepare", changed)
    with pytest.raises(PermissionError, match="during preflight"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert grant.permits(target.action, target)
    assert store.questions.get(grant.question_id)["status"] == "consumed"


def test_feature_workflow_refreshes_dev_from_local_remote(tmp_path, monkeypatch):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    native(git, tmp_path, "clone", "--bare", str(repo), str(remote))
    native(git, repo, "remote", "add", "origin", str(remote))
    upstream = tmp_path / "upstream"
    native(git, tmp_path, "clone", "--branch", "dev", str(remote), str(upstream))
    native(git, upstream, "config", "user.name", "Test")
    native(git, upstream, "config", "user.email", "test@example.invalid")
    (upstream / "upstream.txt").write_text("updated dev")
    native(git, upstream, "add", "upstream.txt")
    native(git, upstream, "commit", "-m", "upstream update")
    native(git, upstream, "push", "origin", "dev")
    harness.git_service.settings["remote"] = "origin"
    task.workflow = "feature"
    task = store.create(task)
    result = asyncio.run(harness.run(task.id))
    assert result.status == "completed"
    assert (repo / "upstream.txt").read_text() == "updated dev"
    assert result.validation_result["valid"] is True


def test_legacy_branch_approval_hash_survives_broker_upgrade(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    target = harness.tools.git_target(task.id, ["branch", "-D", "dev"])
    legacy = {
        "task_id": target.task_id,
        "repository": target.repository,
        "arguments": target.arguments,
        "action": target.action,
        "remote_url": "",
    }
    reason = (
        "approval:branch.delete:"
        + hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    )
    qid = store.ask(
        task.id, "Legacy approval", reason, ["approve", "deny"], required=False
    )
    store.answer(qid, "approve", task.id)
    grant = harness.approvals.issue(task.id, qid, target.action, target)
    assert grant.permits(target.action, target)
