from types import SimpleNamespace

import pytest
from test_evidence_workflow import ready_runtime

from harness.workflows import Workflow


def test_git_finish_rejects_empty_test_evidence_before_any_write(tmp_path):
    store, runtime, task = ready_runtime(tmp_path)
    task.validation_result = {"valid": True}
    task.test_result = {"commands": [], "coverage": None}
    task.git_state = {
        "phase": "branch_created",
        "workflow": "feature",
        "branch": "feature/x",
    }
    task = store.create(task)
    runtime.git_workflow._git = lambda *args, **kwargs: pytest.fail(
        "Git reached without real test evidence"
    )
    with pytest.raises(ValueError, match="successful tests"):
        runtime.git_service.finish(task.id, ["addition.py"])


@pytest.mark.parametrize(
    "workflow,title,expected,source",
    [
        ("hotfix", "ordinary task", Workflow.HOTFIX, "explicit"),
        (None, "Hotfix: API crash", Workflow.HOTFIX, "title_rule"),
        (None, "Release 2.0", Workflow.RELEASE, "title_rule"),
        (None, "Fix broken API", Workflow.BUGFIX, "title_rule"),
        (None, "Other: investigate", Workflow.OTHER, "title_rule"),
        (None, "Add new feature", Workflow.FEATURE, "title_rule"),
    ],
)
def test_git_workflow_classifier_is_typed_and_auditable(
    tmp_path, workflow, title, expected, source
):
    _store, runtime, task = ready_runtime(tmp_path)
    task.workflow = workflow
    task.title = title
    decision = runtime.git_workflow.classify_task(task)
    assert decision.workflow is expected
    assert decision.source == source
    assert decision.evidence


def prepared(tmp_path):
    store, runtime, task = ready_runtime(tmp_path)
    task.validation_result = {"valid": True}
    task.test_result = {
        "commands": [{"returncode": 0}],
        "coverage": {"totals": {"percent_covered": 100}},
    }
    task.git_state = {
        "phase": "branch_created",
        "workflow": "feature",
        "branch": "feature/x",
        "repository": str(runtime.tools.workspace),
    }
    task = store.create(task)
    runtime.git_workflow._git = lambda *args, **kwargs: SimpleNamespace(
        stdout="feature/x\n", returncode=0
    )
    return store, runtime, task


@pytest.mark.parametrize(
    "problem",
    ["validation", "invalid", "tests", "commands", "coverage", "threshold", "failed"],
)
def test_git_finish_requires_all_gates(tmp_path, problem):
    store, runtime, task = prepared(tmp_path)
    if problem == "validation":
        task.validation_result = None
    elif problem == "invalid":
        task.validation_result = {"valid": False}
    elif problem == "tests":
        task.test_result = None
    elif problem == "commands":
        task.test_result["commands"] = []
    elif problem == "coverage":
        task.test_result["coverage"] = None
    elif problem == "threshold":
        task.test_result["coverage"]["totals"]["percent_covered"] = 90
    else:
        task.test_result["commands"][0]["returncode"] = 1
    store.update(task)
    with pytest.raises(ValueError, match="successful tests"):
        runtime.git_service.finish(task.id, ["addition.py"])


@pytest.mark.parametrize("problem", ["phase", "branch", "repository"])
def test_git_resume_checks_actual_state_not_only_metadata(tmp_path, problem):
    store, runtime, task = prepared(tmp_path)
    assert runtime.git_service.begin(task)
    task.git_state[
        {"phase": "phase", "branch": "branch", "repository": "repository"}[problem]
    ] = "wrong"
    store.update(task)
    assert runtime.git_service.begin(task) is False
    assert store.get(task.id).status == "waiting_human"
    assert runtime.git_service.finish(task.id, ["addition.py"]) is False


def test_other_workflow_and_empty_artifacts(tmp_path):
    store, runtime, task = prepared(tmp_path)
    task.workflow = "other"
    assert runtime.git_service.begin(task)
    assert runtime.git_service.finish(task.id, [])
    task = store.get(task.id)
    task.git_state = {
        "phase": "branch_created",
        "workflow": "feature",
        "branch": "feature/x",
        "repository": str(runtime.tools.workspace),
    }
    task.acceptance_criteria = []
    store.update(task)
    with pytest.raises(ValueError, match="explicit task artifact"):
        runtime.git_service.finish(task.id, [])
    with pytest.raises(ValueError, match="cleanup question"):
        runtime.git_service.cleanup(task.id, 999)


def test_begin_records_git_failure_and_configured_fast_forward_pull(tmp_path):
    store, runtime, task = prepared(tmp_path)
    task.git_state = {}
    store.update(task)
    runtime.git_service.settings = {"remote": "origin"}
    calls = []

    def git(args, *arguments, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="", returncode=0)

    runtime.git_workflow._git = git
    assert runtime.git_service.begin(task)
    assert ["pull", "--ff-only", "origin", "dev"] in calls
    task.git_state = {}
    runtime.git_workflow._git = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Git unavailable")
    )
    assert not runtime.git_service.begin(task)
    assert store.get(task.id).status == "waiting_human"


def test_commit_cannot_stage_an_entire_directory(tmp_path):
    _store, runtime, task = prepared(tmp_path)
    (tmp_path / "directory").mkdir()
    with pytest.raises(ValueError, match="explicit files"):
        runtime.git_workflow.commit(["directory"], "not allowed", str(tmp_path))
    with pytest.raises(ValueError, match="invalid editable"):
        runtime.service.patch(task.id, {"git_state": {}})
    with pytest.raises(ValueError, match="internal Git state"):
        runtime.service.create(task.model_copy(update={"id": None}))
