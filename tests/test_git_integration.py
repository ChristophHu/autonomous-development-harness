import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_evidence_workflow import runtime, specification

from harness.agents import ExecutorOutput
from harness.core import Orchestrator


def git_runtime(tmp_path, monkeypatch):
    git = shutil.which("git", path="/opt/homebrew/bin:" + os.environ.get("PATH", ""))
    assert git
    monkeypatch.setenv("PATH", str(Path(git).parent) + ":" + os.environ.get("PATH", ""))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    store, old = runtime(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    store.config.data["paths"]["workspace"] = str(repo)
    store.config.data["tools"]["permissions"]["git"] = "write"
    store.config.data["git"] = {"enabled": True}
    store.config.data["secrets"] = {}
    harness = Orchestrator(store, store.config)
    harness.models.register("fixture", old.models.get("fixture"))
    (repo / "addition.py").write_text("def add(a,b):\n    raise NotImplementedError\n")
    (repo / "check.py").write_text("from addition import add\nassert add(2,3)==5\n")
    (repo / ".gitignore").write_text(".coverage\ncoverage.json\n__pycache__/\n")
    for args in (
        ["-c", "init.templateDir=", "init", "--initial-branch=main"],
        ["config", "user.name", "Harness Test"],
        ["config", "user.email", "harness@example.invalid"],
        ["config", "commit.gpgsign", "false"],
        ["config", "core.hooksPath", "/dev/null"],
        ["add", "addition.py", "check.py", ".gitignore"],
        ["commit", "-m", "baseline"],
        ["branch", "dev"],
    ):
        subprocess.run(
            [git, *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    def execute(step, context):
        harness.tools.execute(
            "filesystem.write",
            {"path": "addition.py", "content": "def add(a,b):\n    return a+b\n"},
        )
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="implemented",
            changed_files=["addition.py"],
        )

    harness.executor.execute = execute
    task = specification()
    task.test_commands = [
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--source=addition",
            "check.py",
        ]
    ]
    return store, harness, task, git, repo


@pytest.mark.parametrize("workflow", ["feature", "bugfix", "hotfix", "release"])
def test_real_local_workflow_tests_merge_sync_tag_and_human_cleanup(
    tmp_path, monkeypatch, workflow
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = workflow
    task.release_version = "1.2.3" if workflow == "release" else ""
    task = store.create(task)
    result = asyncio.run(harness.run(task.id))
    assert result.status == "completed"
    classification = json.loads(
        store.events.list(task.id, "git.workflow_classified")[0]["payload"]
    )
    assert classification["workflow"] == workflow
    assert classification["source"] == "explicit"
    state = result.git_state
    branch = state["branch"]
    assert branch.startswith(workflow + "/")
    target_tests = store.events.list(task.id, "git.target_tests")
    expected_validations = 2 if workflow in {"hotfix", "release"} else 1
    assert len(target_tests) == expected_validations
    assert all(
        json.loads(event["payload"])["valid"]
        for event in store.events.list(task.id, "git.target_validation")
    )

    def read(ref):
        return subprocess.run(
            [git, "show", ref + ":addition.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert "return a+b" in read("dev")
    if workflow in {"hotfix", "release"}:
        assert "return a+b" in read("main")
    else:
        assert "NotImplementedError" in read("main")
    assert harness.tools.git(["branch", "--list", branch]).stdout.strip()
    if workflow == "release":
        assert harness.tools.git(["tag", "--list", "v1.2.3"]).stdout.strip() == "v1.2.3"
        assert not store.questions.list(task.id)
    else:
        question_id = state["cleanup_question_id"]
        answer = "deny" if workflow == "bugfix" else "approve"
        result = asyncio.run(harness.service.answer(task.id, question_id, answer))
        assert result.status == "completed"
        assert result.git_state["phase"] == (
            "cleanup_declined" if answer == "deny" else "cleaned"
        )
        assert bool(harness.tools.git(["branch", "--list", branch]).stdout.strip()) == (
            answer == "deny"
        )


def test_dirty_workspace_is_not_changed_and_invalid_release_is_rejected(
    tmp_path, monkeypatch
):
    store, harness, task, _git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    (repo / "user.txt").write_text("user-owned change")
    assert asyncio.run(harness.run(task.id)).status == "waiting_human"
    assert (repo / "user.txt").read_text() == "user-owned change"
    assert harness.tools.git(["branch", "--show-current"]).stdout.strip() == "main"
    with pytest.raises(ValueError, match="semantic version"):
        harness.git_service.begin(task.model_copy(update={"workflow": "release"}))


def test_remote_branch_delete_is_bound_to_exact_remote_and_single_use(
    tmp_path, monkeypatch
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    subprocess.run(
        [git, "-c", "init.templateDir=", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [git, "remote", "add", "origin", str(remote)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [git, "push", "origin", "main:refs/heads/topic"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    task = store.create(task)
    args = ["push", "origin", "--delete", "topic"]
    target = harness.tools.git_target(task.id, args)
    question_id = harness.approvals.request(target)
    assert store.answer(question_id, "approve", task.id)
    grant = harness.approvals.issue(task.id, question_id, "branch.delete", target)
    subprocess.run(
        [git, "remote", "set-url", "--push", "origin", str(tmp_path / "different.git")],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(PermissionError, match="matching"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    subprocess.run(
        [git, "remote", "set-url", "--push", "origin", str(remote)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(PermissionError, match="matching"):
        harness.tools.git(
            ["push", "origin", "--delete", "main"], approval=grant, task_id=task.id
        )
    result = harness.tools.git(args, approval=grant, task_id=task.id)
    assert result.returncode == 0, result.stderr
    with pytest.raises(PermissionError, match="matching"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    result = subprocess.run(
        [git, "--git-dir", str(remote), "show-ref", "refs/heads/topic"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_merge_failure_never_completes_or_discards_committed_work(
    tmp_path, monkeypatch
):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    harness.git_workflow.merge = lambda *args: (_ for _ in ()).throw(
        RuntimeError("merge conflict")
    )
    result = asyncio.run(harness.run(task.id))
    assert result.status == "waiting_human"
    assert result.git_state["phase"] == "committed"
    assert not store.events.list(task.id, "task.completed")
    assert store.questions.list(task.id)[0]["reason"] == "git:reconciliation"


def test_git_tasks_cannot_switch_a_workspace_owned_by_another_active_task(
    tmp_path, monkeypatch
):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    first = store.create(task)
    second = store.create(task.model_copy(update={"id": None}))
    assert store.tasks.claim(first.id, "first-owner")
    with pytest.raises(ValueError, match="currently running"):
        asyncio.run(harness.run(second.id))
    assert harness.tools.git(["branch", "--show-current"]).stdout.strip() == "main"


@pytest.mark.parametrize("workflow", ["feature", "hotfix"])
def test_failing_merged_target_is_revalidated_before_task_completion(
    tmp_path, monkeypatch, workflow
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = workflow
    task = store.create(task)
    executed = False

    def execute(step, context):
        nonlocal executed
        harness.tools.execute(
            "filesystem.write",
            {"path": "addition.py", "content": "def add(a,b):\n    return a+b\n"},
        )
        if not executed:
            executed = True
            subprocess.run(
                [git, "switch", "dev"], cwd=repo, check=True, capture_output=True
            )
            (repo / "check.py").write_text(
                "from addition import add\nassert add(2,3)==6\n"
            )
            subprocess.run(
                [git, "add", "check.py"], cwd=repo, check=True, capture_output=True
            )
            subprocess.run(
                [git, "commit", "-m", "independent dev change"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [git, "switch", store.get(task.id).git_state["branch"]],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="implemented",
            changed_files=["addition.py"],
        )

    harness.executor.execute = execute
    result = asyncio.run(harness.run(task.id))
    assert result.status == "waiting_approval"
    expected_phase = "merged_primary" if workflow == "feature" else "merged_secondary"
    assert result.git_state["phase"] == expected_phase
    repair_question = store.questions.list(task.id)[0]
    assert repair_question["options"] == '["approve", "deny"]'
    assert repair_question["reason"].startswith("approval:git.repair:")
    assert "==6" in (repo / "check.py").read_text()
    assert store.events.list(task.id, "git.target_tests")
    verdict = store.events.list(task.id, "git.target_validation")[-1]
    payload = json.loads(verdict["payload"])
    assert payload["branch"] == "dev"
    assert payload["valid"] is False
    events = [
        json.loads(event["payload"])
        for event in store.events.list(task.id, "git.target_validation")
    ]
    assert [event["valid"] for event in events] == (
        [False] if workflow == "feature" else [True, False]
    )
    assert [event["branch"] for event in events] == (
        ["dev"] if workflow == "feature" else ["main", "dev"]
    )
    assert not store.events.list(task.id, "task.completed")


def test_approved_target_repair_is_scoped_and_revalidated(tmp_path, monkeypatch):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = "feature"
    task = store.create(task)
    executed = False

    def execute(step, context):
        nonlocal executed
        if "AUTHORIZED_GIT_REPAIR_JSON:" in context:
            assert "check.py" in step.write_paths
            harness.tools.execute(
                "filesystem.write",
                {
                    "path": "check.py",
                    "content": "from addition import add\nassert add(2,3)==5\n",
                },
            )
            return ExecutorOutput(
                subtask_id=step.id,
                success=True,
                output="repaired",
                changed_files=["check.py"],
            )
        harness.tools.execute(
            "filesystem.write",
            {"path": "addition.py", "content": "def add(a,b):\n    return a+b\n"},
        )
        if not executed:
            executed = True
            subprocess.run(
                [git, "switch", "dev"], cwd=repo, check=True, capture_output=True
            )
            (repo / "check.py").write_text(
                "from addition import add\nassert add(2,3)==6\n"
            )
            subprocess.run(
                [git, "add", "check.py"], cwd=repo, check=True, capture_output=True
            )
            subprocess.run(
                [git, "commit", "-m", "independent dev change"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [git, "switch", store.get(task.id).git_state["branch"]],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="implemented",
            changed_files=["addition.py"],
        )

    harness.executor.execute = execute
    waiting = asyncio.run(harness.run(task.id))
    question = store.questions.list(task.id)[0]
    assert waiting.status == "waiting_approval"
    assert (
        "check.py"
        in json.loads(question["question"].removeprefix("Git-Freigabe: "))["arguments"]
    )
    completed = asyncio.run(harness.service.answer(task.id, question["id"], "approve"))
    assert completed.status == "completed"
    assert completed.git_state["repair"]["authorized"] is True
    assert completed.git_state["repair"]["branch_oid"] != ""
    assert (repo / "check.py").read_text().endswith("assert add(2,3)==5\n")
    assert json.loads(
        store.events.list(task.id, "git.target_validation")[-1]["payload"]
    )["valid"]
    assert len(store.events.list(task.id, "task.completed")) == 1


def test_denied_target_repair_stays_blocked_without_creating_branch(
    tmp_path, monkeypatch
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = "feature"
    task = store.create(task)
    ran = False

    def execute(step, context):
        nonlocal ran
        harness.tools.execute(
            "filesystem.write",
            {"path": "addition.py", "content": "def add(a,b):\n    return a+b\n"},
        )
        if not ran:
            ran = True
            subprocess.run(
                [git, "switch", "dev"], cwd=repo, check=True, capture_output=True
            )
            (repo / "check.py").write_text(
                "from addition import add\nassert add(2,3)==6\n"
            )
            subprocess.run(
                [git, "add", "check.py"], cwd=repo, check=True, capture_output=True
            )
            subprocess.run(
                [git, "commit", "-m", "bad target"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [git, "switch", store.get(task.id).git_state["branch"]],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="implemented",
            changed_files=["addition.py"],
        )

    harness.executor.execute = execute
    waiting = asyncio.run(harness.run(task.id))
    qid = store.questions.list(task.id)[0]["id"]
    denied = asyncio.run(harness.service.answer(task.id, qid, "deny"))
    assert waiting.status == "waiting_approval"
    assert denied.status == "blocked"
    assert denied.git_state["phase"] == "repair_declined"
    assert not store.events.list(task.id, "task.completed")
    assert not subprocess.run(
        [git, "branch", "--list", "feature/repair-"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def test_release_main_repair_requires_a_new_release_task(tmp_path, monkeypatch):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = "release"
    task.release_version = "1.2.3"
    task = store.create(task)
    harness.git_service.save(
        task.id, {"workflow": "release", "phase": "merged_primary"}
    )
    assert not harness.git_service.request_repair(task.id, "main", "failed", [])
    question = store.questions.list(task.id)[0]
    assert question["reason"] == "git:reconciliation"
    assert "neuen Release-Task" in question["question"]


@pytest.mark.parametrize("dirty", [False, True])
def test_repair_request_rejects_wrong_or_dirty_target(tmp_path, monkeypatch, dirty):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task.workflow = "feature"
    task = store.create(task)
    harness.git_service.save(
        task.id, {"workflow": "feature", "phase": "merged_primary"}
    )
    if dirty:
        subprocess.run(
            [git, "switch", "dev"], cwd=repo, check=True, capture_output=True
        )
        (repo / "user.txt").write_text("user-owned")
    result = harness.git_service.request_repair(task.id, "dev", "failed", [])
    assert result is False
    question = store.questions.list(task.id)[0]
    assert question["reason"] == "git:reconciliation"
    assert "manuell prüfen" in question["question"]


def test_repair_finish_rejects_target_changed_after_authorization(
    tmp_path, monkeypatch
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    branch = "feature/repair-finish"
    subprocess.run(
        [git, "switch", "-c", branch, "dev"], cwd=repo, check=True, capture_output=True
    )
    (repo / "addition.py").write_text("def add(a,b):\n    return a+b\n")
    subprocess.run(
        [git, "add", "addition.py"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        [git, "commit", "-m", "repair source"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    source_oid = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    state = {
        "workflow": "feature",
        "repository": str(repo),
        "branch": branch,
        "phase": "committed",
        "source_oid": source_oid,
        "repair": {"authorized": True, "branch": "dev", "branch_oid": "stale"},
    }
    task.git_state = state
    task.validation_result = {"valid": True}
    task.test_result = {
        "commands": [{"returncode": 0}],
        "coverage": {"totals": {"percent_covered": 100}},
    }
    store.update(task)
    assert not harness.git_service.finish(task.id, ["addition.py"])
    assert store.questions.list(task.id)
