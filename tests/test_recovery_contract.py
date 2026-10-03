"""Rest-work plans are contracts, not merely advisory prompts."""

import asyncio
import json

import pytest
from test_evidence_workflow import ready_runtime

from harness.agents import PlannerOutput
from harness.reconciliation import ReconciliationService, RecoveryScope


def test_recovery_rejects_replay_and_requires_complete_remaining_scope(tmp_path):
    from harness.reconciliation import RecoveryScope

    store, runtime, task = ready_runtime(tmp_path)
    task = store.create(task)
    report = ReconciliationService(store, runtime.tools, runtime.validator).inspect(
        task
    )
    scope = RecoveryScope.assess(task, report, runtime.router, runtime.validator)
    plan = PlannerOutput.model_validate(
        {
            "summary": "blind replay",
            "complexity": "simple",
            "subtasks": [
                {
                    "id": "replay",
                    "title": "rewrite addition",
                    "description": "replay",
                    "expected_result": "sum",
                    "acceptance_criteria": ["sum"],
                    "recovery_targets": ["criterion:sum"],
                    "write_paths": ["addition.py"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="outside remaining scope"):
        scope.validate_plan(plan, runtime.tools)


def test_restart_from_recovering_does_not_repeat_recovery_transition(
    tmp_path, monkeypatch
):
    from harness.reconciliation import ReconciliationService

    store, runtime, task = ready_runtime(tmp_path)
    task_id = store.create(task).id
    for status in ("analyzing", "planning", "ready", "executing", "recovering"):
        store.tasks.transition(task_id, status)

    def fail_inspection(*_args):
        raise RuntimeError("recovery inspection unavailable")

    monkeypatch.setattr(ReconciliationService, "inspect", fail_inspection)

    with pytest.raises(RuntimeError, match="recovery inspection unavailable"):
        asyncio.run(runtime.run(task_id))

    assert store.get(task_id).status == "failed"


def review_response(task, status="completed"):
    return json.dumps(
        {
            "requirements": {
                requirement: {
                    "status": status,
                    "criteria": ["sum"],
                    "evidence": "Observed addition.py and fresh executed arithmetic test.",
                }
                for requirement in task.requirements
            }
        }
    )


def assessed_runtime(tmp_path, status="completed"):
    store, runtime, task = ready_runtime(tmp_path)
    task = store.create(task)
    report = ReconciliationService(store, runtime.tools, runtime.validator).inspect(
        task
    )
    report.git = {"status": {"returncode": 0}, "diff": {"returncode": 0}}
    runtime.router.complete = lambda *args, **kwargs: review_response(task, status)
    return runtime, task, report


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "unknown",
        "no-evidence",
        "not-confirmed",
        "empty-rationale",
        "bad-json",
    ],
)
def test_recovery_assessment_fails_closed(tmp_path, problem):
    runtime, task, report = assessed_runtime(tmp_path)
    response = json.loads(review_response(task))
    result = response["requirements"][task.requirements[0]]
    if problem == "missing":
        response["requirements"] = {}
    elif problem == "unknown":
        result["criteria"] = ["invented"]
    elif problem == "no-evidence":
        result["criteria"] = []
    elif problem == "not-confirmed":
        report.confirmed_criteria = []
    elif problem == "empty-rationale":
        result["evidence"] = "   "
    runtime.router.complete = lambda *args, **kwargs: (
        "bad JSON" if problem == "bad-json" else json.dumps(response)
    )
    with pytest.raises(ValueError):
        RecoveryScope.assess(task, report, runtime.router, runtime.validator)


@pytest.mark.parametrize("problem", ["none", "git", "tests", "remaining"])
def test_recovery_assessment_keeps_unverified_requirements_open(tmp_path, problem):
    runtime, task, report = assessed_runtime(
        tmp_path, "remaining" if problem == "remaining" else "completed"
    )
    # This contract test exercises recovery assessment, not the nested test runner.
    # Make its baseline evidence explicit so an environment-dependent coverage
    # failure cannot silently change the expected recovery scope.
    report.confirmed_criteria = ["sum"]
    report.remaining_criteria = []
    report.uncertain_criteria = []
    report.test_findings = []
    if problem == "git":
        report.git["status"]["returncode"] = 1
    if problem == "tests":
        report.test_findings = ["failing test"]
    scope = RecoveryScope.assess(task, report, runtime.router, runtime.validator)
    assert ("requirement:" + task.requirements[0] in scope.remaining_targets) == (
        problem != "none"
    )
    assert ("tests" in scope.remaining_targets) == (problem == "tests")
    assert str(tmp_path / "addition.py") in scope.protected_files


def rest_plan(targets, paths=(), tools=()):
    return PlannerOutput.model_validate(
        {
            "summary": "rest",
            "complexity": "simple",
            "subtasks": [
                {
                    "id": "rest",
                    "title": "remaining",
                    "description": "inspect or implement remaining",
                    "expected_result": "remaining accepted",
                    "acceptance_criteria": ["observed"],
                    "recovery_targets": targets,
                    "write_paths": list(paths),
                    "required_tools": list(tools),
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "unknown",
        "verify-write",
        "escape",
        "absolute",
        "blank-path",
        "protected",
        "shell",
        "missing-target",
    ],
)
def test_recovery_plan_rejects_invalid_scope(tmp_path, problem):
    runtime, task, report = assessed_runtime(tmp_path, "remaining")
    scope = RecoveryScope.assess(task, report, runtime.router, runtime.validator)
    targets = scope.remaining_targets
    paths, tools = [], []
    if problem == "empty":
        targets = []
    elif problem == "unknown":
        targets = ["criterion:sum"]
    elif problem == "verify-write":
        targets, paths = ["verify"], ["other.txt"]
    elif problem == "escape":
        paths = ["../outside.txt"]
    elif problem == "absolute":
        paths = [str(tmp_path / "new.txt")]
    elif problem == "blank-path":
        paths = [""]
    elif problem == "protected":
        paths = ["addition.py"]
    elif problem == "shell":
        tools = ["shell.execute"]
    else:
        targets = ["verify"]
    with pytest.raises(ValueError):
        scope.validate_plan(rest_plan(targets, paths, tools), runtime.tools)


def test_recovery_valid_plan_preservation_and_runtime_write_guard(tmp_path):
    runtime, task, report = assessed_runtime(tmp_path, "remaining")
    scope = RecoveryScope.assess(task, report, runtime.router, runtime.validator)
    scope.validate_plan(
        rest_plan(
            scope.remaining_targets,
            ["new.txt"],
            ["filesystem.read", "filesystem.write"],
        ),
        runtime.tools,
    )
    scope.verify_preserved()
    with runtime.tools.recovery_writes(["new.txt"]):
        runtime.tools.execute("filesystem.write", {"path": "new.txt", "content": "new"})
        assert runtime.tools.execute("filesystem.read", {"path": "addition.py"})
        with pytest.raises(PermissionError, match="outside recovery"):
            runtime.tools.execute(
                "filesystem.write", {"path": "addition.py", "content": "overwrite"}
            )
        with pytest.raises(PermissionError, match="unrestricted mutation"):
            runtime.tools.execute("shell.execute", {"command": ["echo", "bypass"]})
        with runtime.tools.recovery_writes([]), pytest.raises(PermissionError):
            runtime.tools.execute(
                "filesystem.create", {"path": "newer.txt", "content": "x"}
            )
    assert runtime.tools.recovery_paths.get() is None
    (tmp_path / "addition.py").write_text("corrupted")
    with pytest.raises(ValueError, match="confirmed artifact changed"):
        scope.verify_preserved()

    (tmp_path / "other.py").write_text("source")
    (tmp_path / "addition.py").unlink()
    (tmp_path / "addition.py").symlink_to(tmp_path / "other.py")
    with pytest.raises(ValueError, match="confirmed artifact changed"):
        scope.verify_preserved()
    (tmp_path / "addition.py").unlink()
    with pytest.raises(ValueError, match="confirmed artifact changed"):
        scope.verify_preserved()


def test_recovery_path_shared_with_missing_criterion_is_not_frozen(tmp_path):
    from harness.domain import AcceptanceCriterion

    runtime, task, report = assessed_runtime(tmp_path, "remaining")
    task.acceptance_criteria.append(
        AcceptanceCriterion(
            id="extension",
            description="extend same file",
            kind="file_contains",
            path="addition.py",
            contains="new feature",
        )
    )
    report.remaining_criteria = ["extension"]
    scope = RecoveryScope.assess(task, report, runtime.router, runtime.validator)
    assert not scope.protected_files
    scope.validate_plan(
        rest_plan(scope.remaining_targets, ["addition.py"], ["filesystem.create"]),
        runtime.tools,
    )


@pytest.mark.parametrize("stage", ["executor", "validator"])
def test_recovery_cannot_complete_after_confirmed_artifact_tampering(tmp_path, stage):
    store, runtime, task = ready_runtime(tmp_path)
    task.status = "executing"
    task = store.create(task)

    def corrupt():
        with (tmp_path / "addition.py").open("a") as handle:
            handle.write("# unauthorized rewrite\n")

    if stage == "executor":
        original = runtime.executor.execute

        def execute(*args):
            corrupt()
            return original(*args)

        runtime.executor.execute = execute
    else:
        original = runtime.validator.validate

        def validate(*args):
            result = original(*args)
            corrupt()
            return result

        runtime.validator.validate = validate
    with pytest.raises(ValueError, match="confirmed artifact changed"):
        asyncio.run(runtime.run(task.id))
    assert store.get(task.id).status == "failed"
    assert not store.events.list(task.id, "task.completed")
