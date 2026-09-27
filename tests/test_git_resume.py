import asyncio
import subprocess

import pytest
from test_git_integration import git_runtime

from harness.agents import ExecutorOutput


@pytest.mark.parametrize("workflow", ["feature", "hotfix", "release"])
@pytest.mark.parametrize("interrupt", [1, 2])
def test_resume_proves_existing_merges_without_duplicate_commits(
    tmp_path, monkeypatch, workflow, interrupt
):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = workflow
    task.release_version = "2.3.4" if workflow == "release" else ""
    task = store.create(task)
    original = harness.git_workflow.merge
    calls = []

    def fail(branch, cwd):
        calls.append(branch)
        if len(calls) == interrupt:
            original(branch, cwd)
            raise RuntimeError("interrupted after successful merge")
        return original(branch, cwd)

    # Features have only one merge; exercise a failure before it for case two.
    if workflow == "feature" and interrupt == 2:

        def fail(branch, cwd):
            raise RuntimeError("interrupted before merge")

    harness.git_workflow.merge = fail
    assert asyncio.run(harness.run(task.id)).status == "waiting_human"
    before = harness.tools.git(
        ["rev-parse", "refs/heads/" + store.get(task.id).git_state["branch"]]
    ).stdout
    harness.git_workflow.merge = original
    harness.executor.execute = lambda step, context: ExecutorOutput(
        subtask_id=step.id,
        success=True,
        output="existing implementation verified",
        changed_files=[],
    )
    question = store.questions.list(task.id)[0]
    result = asyncio.run(harness.service.answer(task.id, question["id"], "retry"))
    assert result.status == "completed"
    assert (
        harness.tools.git(
            ["rev-parse", "refs/heads/" + result.git_state["branch"]]
        ).stdout
        == before
    )
    assert "return a+b" in harness.tools.git(["show", "dev:addition.py"]).stdout
    assert store.events.list(task.id, "git.reconciled")
    assert result.test_result["commands"][0]["returncode"] == 0
    assert result.validation_result["valid"] is True


def interrupted(tmp_path, monkeypatch, workflow="feature"):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = workflow
    task.release_version = "2.3.4" if workflow == "release" else ""
    task = store.create(task)
    harness.git_workflow.merge = lambda *args: (_ for _ in ()).throw(
        RuntimeError("pause")
    )
    assert asyncio.run(harness.run(task.id)).status == "waiting_human"
    return store, harness, store.get(task.id), git, repo


@pytest.mark.parametrize("problem", ["repository", "source", "dirty", "missing"])
def test_resume_blocks_unproven_or_dirty_sources(tmp_path, monkeypatch, problem):
    store, harness, task, _git, repo = interrupted(tmp_path, monkeypatch)
    if problem == "repository":
        task.git_state["repository"] = str(tmp_path)
    elif problem == "source":
        task.git_state["source_oid"] = "0" * 40
    elif problem == "dirty":
        (repo / "user.txt").write_text("must not be discarded")
    else:
        task.git_state["branch"] = "feature/missing"
    store.update(task)
    assert not harness.git_service.begin(task)
    assert store.get(task.id).status == "waiting_human"
    assert not store.events.list(task.id, "git.reconciled")


def test_resume_rechecks_source_at_finish(tmp_path, monkeypatch):
    _store, harness, task, _git, repo = interrupted(tmp_path, monkeypatch)
    assert harness.git_service.begin(task)
    (repo / "addition.py").write_text("unexpected concurrent edit")
    assert not harness.git_service.finish(task.id, ["addition.py"])
    assert (repo / "addition.py").read_text() == "unexpected concurrent edit"


@pytest.mark.parametrize("wrong_tag", [False, True])
def test_resume_tag_write_before_state_save(tmp_path, monkeypatch, wrong_tag):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = "release"
    task.release_version = "2.3.4"
    task = store.create(task)
    tag_release = harness.git_workflow.tag_release

    def fail(version, cwd):
        if wrong_tag:
            harness.git_workflow._git(
                ["tag", "-a", "v2.3.4", "-m", "wrong target", "main~1"], cwd
            )
        else:
            tag_release(version, cwd)
        raise RuntimeError("interrupted after tag")

    harness.git_workflow.tag_release = fail
    assert asyncio.run(harness.run(task.id)).status == "waiting_human"
    before = harness.tools.git(["rev-parse", "refs/tags/v2.3.4"]).stdout
    harness.git_workflow.tag_release = tag_release
    assert harness.git_service.begin(store.get(task.id)), [
        dict(row) for row in store.questions.list(task.id)
    ]
    assert harness.git_service.finish(task.id, ["addition.py"]) is (not wrong_tag)
    assert harness.tools.git(["rev-parse", "refs/tags/v2.3.4"]).stdout == before


def test_changed_primary_is_rejected_before_merge(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = interrupted(tmp_path, monkeypatch, "release")
    task.git_state.update(phase="merged_primary", primary_oid="0" * 40)
    store.update(task)
    assert harness.git_service.begin(task)
    assert not harness.git_service.finish(task.id, ["addition.py"])
    assert "Primärer Zielbranch" in store.questions.list(task.id)[-1]["question"]


def test_unresolved_merge_requires_manual_resolution(tmp_path, monkeypatch):
    store, harness, task, git, repo = interrupted(tmp_path, monkeypatch)

    def run(*args, check=True):
        return subprocess.run(
            [git, *args], cwd=repo, capture_output=True, text=True, check=check
        )

    run("switch", "dev")
    (repo / "addition.py").write_text("def add(a,b):\n    return 0\n")
    run("add", "addition.py")
    run("commit", "-m", "conflicting user work")
    assert (
        run("merge", "--no-ff", task.git_state["branch"], check=False).returncode == 1
    )
    markers = (repo / "addition.py").read_text()
    assert "<<<<<<<" in markers
    assert not harness.git_service.begin(task)
    assert (repo / "addition.py").read_text() == markers
    # Only the human fixture resolves the conflict; the harness never resets/aborts it.
    (repo / "addition.py").write_text("def add(a,b):\n    return a+b\n")
    run("add", "addition.py")
    run("commit", "-m", "human conflict resolution")
    assert harness.git_service.begin(store.get(task.id)), [
        dict(row) for row in store.questions.list(task.id)
    ] + [{"git_status": harness.tools.git(["status", "--porcelain=v1"]).stdout}]
    assert harness.git_service.finish(task.id, ["addition.py"])


def test_existing_cleanup_question_is_not_duplicated(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    result = asyncio.run(harness.run(task.id))
    question_id = result.git_state["cleanup_question_id"]
    assert harness.git_service.begin(result)
    assert harness.git_service.finish(task.id, ["addition.py"])
    assert len(store.questions.list(task.id)) == 1
    assert store.get(task.id).git_state["cleanup_question_id"] == question_id
    assert store.get(task.id).git_state["phase"] == "awaiting_cleanup"


def test_lightweight_tag_is_not_accepted_as_annotated_release(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = interrupted(tmp_path, monkeypatch, "release")
    original = harness.git_workflow.merge
    # Replace the injected merge failure with the actual implementation.
    from harness.workflows import GitWorkflow

    harness.git_workflow.merge = GitWorkflow.merge.__get__(harness.git_workflow)
    harness.git_workflow.tag_release = lambda version, cwd: harness.git_workflow._git(
        ["tag", "v2.3.4"], cwd
    )
    assert harness.git_service.begin(task)
    assert harness.git_service.finish(task.id, ["addition.py"])
    # A subsequent inspection rejects the existing lightweight tag, without deleting it.
    assert harness.git_service.begin(store.get(task.id))
    assert not harness.git_service.finish(task.id, ["addition.py"])
    assert harness.tools.git(["tag", "--list", "v2.3.4"]).stdout.strip() == "v2.3.4"
    harness.git_workflow.merge = original


def test_retry_runs_fresh_tests_and_never_merges_on_failure(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = interrupted(tmp_path, monkeypatch)
    before = harness.tools.git(["rev-parse", "dev"]).stdout
    harness.executor.execute = lambda step, context: ExecutorOutput(
        subtask_id=step.id, success=True, output="verified", changed_files=[]
    )
    original = harness.validator.run_tests
    calls = []

    def fail(task):
        reports = original(task)
        calls.append(reports)
        reports["commands"][0]["returncode"] = 1
        return reports

    harness.validator.run_tests = fail
    question = store.questions.list(task.id)[0]
    with pytest.raises(RuntimeError, match="command failed"):
        asyncio.run(harness.service.answer(task.id, question["id"], "retry"))
    result = store.get(task.id)
    assert result.status == "failed"
    assert calls
    assert harness.tools.git(["rev-parse", "dev"]).stdout == before
    assert result.git_state["phase"] == "committed"


def test_commit_intent_recovers_real_commit_if_state_save_is_interrupted(
    tmp_path, monkeypatch
):
    store, harness, task, _git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    assert harness.git_service.begin(task)
    (repo / "addition.py").write_text("def add(a,b):\n    return a+b\n")
    task = store.get(task.id)
    task.validation_result = {"valid": True}
    task.test_result = {
        "commands": [{"returncode": 0}],
        "coverage": {"totals": {"percent_covered": 100}},
    }
    store.update(task)
    original_save = harness.git_service.save

    def interrupted_save(task_id, state):
        if state.get("phase") == "committed":
            raise RuntimeError("simulated process stop before commit ID persistence")
        return original_save(task_id, state)

    harness.git_service.save = interrupted_save
    assert not harness.git_service.finish(task.id, ["addition.py"])
    assert store.get(task.id).git_state["phase"] == "commit_pending"
    harness.git_service.save = original_save
    assert harness.git_service.begin(store.get(task.id)), [
        dict(row) for row in store.questions.list(task.id)
    ] + [{"git_status": harness.tools.git(["status", "--porcelain=v1"]).stdout}]
    state = store.get(task.id).git_state
    assert state["phase"] == "committed"
    assert (
        state["source_oid"] == harness.tools.git(["rev-parse", "HEAD"]).stdout.strip()
    )
    recovered_intent = dict(state, phase="commit_pending")
    assert harness.git_service.resolve_commit_intent(
        recovered_intent, str(repo), task.id
    )
    task = store.get(task.id)
    task.git_state["phase"] = "commit_pending"
    store.update(task)
    assert harness.git_service.finish(task.id, ["addition.py"])
    assert store.get(task.id).git_state["phase"] == "awaiting_cleanup"


def test_commit_intent_rejects_unrelated_head_change(tmp_path, monkeypatch):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    assert harness.git_service.begin(task)
    state = store.get(task.id).git_state
    state.update(
        phase="commit_pending",
        commit_parent=harness.tools.git(["rev-parse", "HEAD"]).stdout.strip(),
        commit_paths=["addition.py"],
        commit_message=f"Task {task.id}: {task.title}",
    )
    harness.git_service.save(task.id, state)
    (repo / "user.txt").write_text("unrelated")
    subprocess.run([git, "add", "user.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [git, "commit", "-m", "unrelated user commit"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert not harness.git_service.begin(store.get(task.id))
    assert store.get(task.id).git_state["phase"] == "commit_pending"


@pytest.mark.parametrize(
    "problem",
    ["allowed_change", "no_changes", "wrong_path", "rename", "missing_parent"],
)
def test_uncommitted_commit_intent_only_resumes_allowed_files(
    tmp_path, monkeypatch, problem
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    assert harness.git_service.begin(task)
    state = store.get(task.id).git_state
    state.update(
        phase="commit_pending",
        commit_parent=harness.tools.git(["rev-parse", "HEAD"]).stdout.strip(),
        commit_paths=["addition.py"],
        commit_message=f"Task {task.id}: {task.title}",
    )
    if problem == "missing_parent":
        state.pop("commit_parent")
    elif problem == "allowed_change":
        (repo / "addition.py").write_text("def add(a,b):\n    return a+b\n")
    elif problem == "wrong_path":
        (repo / "check.py").write_text("user change")
    elif problem == "rename":
        subprocess.run(
            [git, "mv", "addition.py", "renamed.py"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    task.git_state = state
    store.update(task)
    allowed_commit_in_progress = problem == "allowed_change"
    assert harness.git_service.begin(store.get(task.id)) is allowed_commit_in_progress
