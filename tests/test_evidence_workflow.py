"""Real subprocess E2E: broken code, failing test, correction, coverage, review."""

import asyncio
import json
import sys

import pytest

from harness.agents import ExecutorOutput, ValidatorOutput
from harness.approvals import ApprovalDenied, ApprovalRequired, ToolApprovalTarget
from harness.core import Config, Orchestrator, Store
from harness.domain import AcceptanceCriterion, Task


def runtime(tmp_path):
    config = Config()
    config.data["paths"] = {
        "database": str(tmp_path / "db.sqlite"),
        "workspace": str(tmp_path),
        "obsidian_vault": str(tmp_path / "vault"),
    }
    config.data["models"] = {}
    config.data["memory"] = {}
    config.data["tools"] = {
        "permissions": {"filesystem": "write", "shell": "write", "git": "read"}
    }
    config.data["profiles"] = {
        name: {"model": {"primary": "fixture"}}
        for name in ("planner", "coding", "validator")
    }
    store = Store(config)
    orchestrator = Orchestrator(store, config)

    class Provider:
        def complete(self, prompt, **kwargs):
            if prompt.startswith("RECOVERY_REVIEW:"):
                return json.dumps(
                    {
                        "requirements": {
                            "add two integers": {
                                "status": "uncertain",
                                "criteria": ["sum"],
                                "evidence": "The addition criterion needs renewed independent inspection.",
                            }
                        }
                    }
                )
            if prompt.startswith("PLAN:"):
                try:
                    task_data = json.loads(
                        prompt.split("\nTask:\n", 1)[1].split("\nContext:\n", 1)[0]
                    )
                except (IndexError, json.JSONDecodeError):
                    task_data = {}
                requirement_ids = task_data.get("requirements", []) or [
                    "add two integers"
                ]
                criterion_ids = [
                    item.get("id")
                    for item in task_data.get("acceptance_criteria", [])
                    if isinstance(item, dict) and item.get("id")
                ]
                targets = []
                write_paths = []
                if "RECOVERY_SCOPE_JSON:\n" in prompt:
                    targets = json.loads(
                        prompt.split("RECOVERY_SCOPE_JSON:\n")[-1].splitlines()[0]
                    )["remaining_targets"]
                else:
                    write_paths = ["addition.py"]
                if "AUTHORIZED_GIT_REPAIR_JSON:\n" in prompt:
                    write_paths = json.loads(
                        prompt.split("AUTHORIZED_GIT_REPAIR_JSON:\n")[-1].splitlines()[
                            0
                        ]
                    )["paths"]
                return json.dumps(
                    {
                        "summary": "implement addition",
                        "complexity": "simple",
                        "subtasks": [
                            {
                                "id": "code",
                                "title": "addition",
                                "description": "implement",
                                "expected_result": "correct sum",
                                "acceptance_criteria": criterion_ids or ["sum"],
                                "requirement_ids": requirement_ids,
                                "recovery_targets": targets,
                                "write_paths": write_paths,
                            }
                        ],
                    }
                )
            if prompt.startswith("REQUIREMENTS:"):
                return json.dumps(
                    {
                        "fields": {},
                        "rationale": "No additional requirement facts are supported.",
                        "evidence": {},
                    }
                )
            payload = json.loads(prompt.split("\n", 1)[1])
            available = payload["available_evidence_ids"]
            requirements = payload["task"]["requirements"]
            criteria = [item["id"] for item in payload["task"]["acceptance_criteria"]]
            plan = payload["task"].get("plan") or {}
            steps = plan.get("subtasks", [])

            def references(key, requirement):
                field = "requirement_ids" if requirement else "acceptance_criteria"
                matching = [
                    f"plan-step:{step['id']}"
                    for step in steps
                    if key in step.get(field, [])
                ]
                observed = next(
                    (item for item in available if not item.startswith("plan-step:")),
                    None,
                )
                return [*matching[:1], *([observed] if observed else [])]

            return json.dumps(
                {
                    "requirements": dict.fromkeys(requirements, True),
                    "criteria": dict.fromkeys(criteria, True),
                    "evidence": "Observed real subprocess tests and source artifacts.",
                    "requirement_evidence": {
                        name: references(name, True) for name in requirements
                    },
                    "criterion_evidence": {
                        name: references(name, False) for name in criteria
                    },
                }
            )

    orchestrator.models.register("fixture", Provider())
    return store, orchestrator


def specification():
    return Task(
        title="addition",
        goal="correct arithmetic",
        requirements=["add two integers"],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="sum",
                description="addition exists",
                kind="file_contains",
                path="addition.py",
                contains="def add",
            )
        ],
        test_commands=[
            [sys.executable, "-c", "from addition import add; assert add(2,3)==5"]
        ],
        coverage_command=[
            sys.executable,
            "-m",
            "coverage",
            "json",
            "-o",
            "coverage.json",
        ],
    )


def ready_runtime(tmp_path):
    store, orchestrator = runtime(tmp_path)
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
    (tmp_path / "addition.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "check.py").write_text(
        "from addition import add\nassert add(2, 3) == 5\n"
    )
    orchestrator.executor = type(
        "ExecutorFixture",
        (),
        {
            "execute": lambda self, step, context: ExecutorOutput(
                subtask_id=step.id,
                success=True,
                output="existing implementation inspected",
            )
        },
    )()
    return store, orchestrator, task


def test_real_failure_is_corrected_and_independently_validated(tmp_path):
    store, orchestrator = runtime(tmp_path)
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
    (tmp_path / "check.py").write_text(
        "from addition import add\nassert add(2, 3) == 5\n"
    )

    class Executor:
        calls = 0

        def execute(self, step, context):
            self.calls += 1
            if self.calls == 2:
                assert "command failed" in context
            (tmp_path / "addition.py").write_text(
                "def add(a, b):\n    return "
                + ("a - b" if self.calls == 1 else "sum((a, b))")
                + "\n"
            )
            return ExecutorOutput(
                subtask_id=step.id,
                success=True,
                output="implemented",
                changed_files=["addition.py"],
                tool_evidence=[
                    {"tool": "filesystem.write", "changed_path": "addition.py"}
                ],
            )

    executor = Executor()
    orchestrator.executor = executor
    created = store.create(task)
    result = asyncio.run(orchestrator.run(created.id))
    assert result.status == "completed" and executor.calls == 2
    assert result.test_result["coverage"]["totals"]["percent_covered"] == 100
    assert result.validation_result["valid"] is True
    assert len(store.events.list(created.id, "correction.started")) == 1
    assert [item["status"] for item in store.corrections.list_for_task(created.id)] == [
        "resolved"
    ]
    assert store.corrections.list_for_task(created.id)[0]["attempts"] == 1
    assert store.tasks.claim(created.id, "other") is False


def test_persisted_correction_is_retried_and_resolved_without_subprocesses(tmp_path):
    from harness.agents import CorrectionFinding

    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    contexts = []

    class ValidatorFixture:
        def __init__(self):
            self.calls = 0

        def workspace_snapshot(self):
            return {"applicable": False, "filesystem": {}}

        def run_tests(self, _task):
            return {"commands": [], "coverage": {"totals": {}}}

        def validate(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ValidatorOutput(
                    valid=False,
                    errors=["acceptance criterion failed: sum"],
                    findings=[
                        CorrectionFinding(
                            category="acceptance",
                            source="acceptance_validator",
                            rule="acceptance.criterion_failed",
                            message="sum criterion failed",
                        )
                    ],
                )
            return ValidatorOutput(valid=True)

    orchestrator.validator = ValidatorFixture()

    def execute(_task, step, context, _recovery_scope):
        contexts.append(context)
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="corrected",
        )

    orchestrator._execute_step = execute
    result = asyncio.run(orchestrator.run(created.id))

    items = store.corrections.list_for_task(created.id)
    assert result.status == "completed"
    assert len(contexts) == 2
    assert '"rule": "acceptance.criterion_failed"' in contexts[1]
    assert len(items) == 1
    assert items[0]["status"] == "resolved"
    assert items[0]["attempts"] == 1


def test_unresolved_correction_stays_open_when_retry_budget_exhausts(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    orchestrator.config.data["harness"] = {"max_correction_attempts": 1}

    class ValidatorFixture:
        def workspace_snapshot(self):
            return {"applicable": False, "filesystem": {}}

        def run_tests(self, _task):
            return {"commands": [], "coverage": {"totals": {}}}

        def validate(self, *_args, **_kwargs):
            return ValidatorOutput(
                valid=False,
                errors=["acceptance criterion failed: sum"],
            )

    orchestrator.validator = ValidatorFixture()
    orchestrator._execute_step = lambda _task, step, _context, _scope: ExecutorOutput(
        subtask_id=step.id, success=True, output="unchanged"
    )

    with pytest.raises(RuntimeError, match="acceptance criterion failed"):
        asyncio.run(orchestrator.run(created.id))

    item = store.corrections.list_for_task(created.id)[0]
    assert item["status"] == "open"
    assert item["attempts"] == 1
    assert store.get(created.id).status == "failed"


def test_interrupted_in_progress_correction_is_reopened_on_restart(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    item = store.corrections.record(
        created.id,
        {
            "category": "tests",
            "source": "test_runner",
            "rule": "test_command.failed",
            "message": "previous run was interrupted",
        },
    )
    store.corrections.set_status(item["id"], "in_progress")

    class ValidatorFixture:
        def workspace_snapshot(self):
            return {"applicable": False, "filesystem": {}}

        def run_tests(self, _task):
            return {"commands": [], "coverage": {"totals": {}}}

        def validate(self, *_args, **_kwargs):
            return ValidatorOutput(valid=True)

    orchestrator.validator = ValidatorFixture()
    orchestrator._execute_step = lambda _task, step, _context, _scope: ExecutorOutput(
        subtask_id=step.id, success=True, output="verified"
    )

    result = asyncio.run(orchestrator.run(created.id))

    resumed_item = store.corrections.get(item["id"])
    assert result.status == "completed"
    assert resumed_item["status"] == "resolved"
    assert resumed_item["attempts"] == 2


def test_missing_requirements_pause_without_model_or_success(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="unspecified"))
    result = asyncio.run(orchestrator.run(task.id))
    assert result.status == "waiting_human"
    assert store.questions.has_open_required(task.id)
    with pytest.raises(ValueError, match="question"):
        asyncio.run(orchestrator.run(task.id))


def test_high_impact_tool_approval_pauses_task_and_records_exact_target(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(specification())
    target = ToolApprovalTarget.create(
        task.id, "http.request", {"method": "POST", "url": "https://example.test"}
    )

    def request_approval(*_args, **_kwargs):
        question_id = orchestrator.approvals.request_tool(target)
        raise ApprovalRequired(question_id)

    orchestrator._invoke = request_approval
    result = asyncio.run(orchestrator.run(task.id))
    assert result.status == "waiting_approval"
    question = store.questions.list(task.id)[0]
    assert question["reason"] == target.reason()
    assert "https://example.test" not in question["question"]


def test_persisted_approval_resumes_for_exact_action_once(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(specification())
    secret = "authorization-secret-do-not-store"
    arguments = {"method": "POST", "headers": {"Authorization": secret}}
    target = ToolApprovalTarget.create(task.id, "http.request", arguments)
    question_id = orchestrator.approvals.request_tool(target)
    question = store.questions.get(question_id)
    assert secret not in question["question"]
    assert store.get(task.id).status == "waiting_approval"
    assert store.answer(question_id, "approve", task.id)

    grant = orchestrator.approvals.grant_tool_for_target(target)
    assert grant.consume(target.action, target)
    with pytest.raises(PermissionError, match="already consumed"):
        orchestrator.approvals.grant_tool_for_target(target)
    changed = ToolApprovalTarget.create(
        task.id,
        "http.request",
        {"method": "POST", "headers": {"Authorization": "different"}},
    )
    with pytest.raises(ApprovalRequired):
        orchestrator.approvals.grant_tool_for_target(changed)


def test_denied_action_approval_blocks_task(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(specification())
    orchestrator._invoke = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ApprovalDenied("denied")
    )
    result = asyncio.run(orchestrator.run(task.id))
    assert result.status == "blocked"
    assert any(row["kind"] == "task.blocked" for row in store.events.list(task.id))


def test_denial_does_not_reopen_a_task_that_became_terminal(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(specification())

    def terminal_then_deny(*_args, **_kwargs):
        with store.database.connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='completed' WHERE id=?", (task.id,)
            )
        raise ApprovalDenied("denied")

    orchestrator._invoke = terminal_then_deny
    result = asyncio.run(orchestrator.run(task.id))
    assert result.status == "completed"


def test_service_metadata_and_concurrent_claim(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(specification())
    assert orchestrator.service.patch(task.id, {"priority": 7}).priority == 7
    assert store.tasks.claim(task.id, "first")
    assert not store.tasks.claim(task.id, "second")
    assert not store.tasks.release(task.id, "second")
    with pytest.raises(ValueError, match="running"):
        orchestrator.service.patch(task.id, {"goal": "other"})
    assert store.tasks.release(task.id, "first")


def test_real_executor_tool_loop_corrects_a_real_test_failure(tmp_path):
    from harness.agents import Executor
    from harness.providers import ModelUsage, ToolCall

    store, orchestrator, task = ready_runtime(tmp_path)
    orchestrator.config.data["profiles"]["coding"].update(
        {"tools": ["filesystem.write"], "permissions": ["filesystem"]}
    )

    class Provider:
        def complete(self, prompt, **kwargs):
            if isinstance(prompt, list):
                if prompt[-1]["role"] == "tool":
                    assert prompt[-1]["tool_call_id"] == "edit-call"
                    return '{"output":"Actual workspace edit performed"}', ModelUsage(
                        "fixture", "fixture"
                    )
                corrected = "command failed" in prompt[1]["content"]
                content = (
                    "def add(a, b):\n    return "
                    + ("sum((a, b))" if corrected else "a - b")
                    + "\n"
                )
                return (
                    "",
                    ModelUsage("fixture", "fixture"),
                    [
                        ToolCall(
                            "edit-call",
                            "filesystem.write",
                            {"path": "addition.py", "content": content},
                        )
                    ],
                )
            if prompt.startswith("PLAN:"):
                return json.dumps(
                    {
                        "summary": "implementation",
                        "complexity": "simple",
                        "subtasks": [
                            {
                                "id": "code",
                                "title": "code",
                                "description": "implement addition",
                                "expected_result": "sum",
                                "acceptance_criteria": ["sum"],
                                "requirement_ids": ["add two integers"],
                                "required_tools": ["filesystem.write"],
                                "write_paths": ["addition.py"],
                            }
                        ],
                    }
                )
            assert '"artifacts"' in prompt
            payload = json.loads(prompt.split("\n", 1)[1])
            available = payload["available_evidence_ids"]
            requirements = payload["task"]["requirements"]
            criteria = [item["id"] for item in payload["task"]["acceptance_criteria"]]
            plan = payload["task"].get("plan") or {}
            steps = plan.get("subtasks", [])

            def references(key, requirement):
                field = "requirement_ids" if requirement else "acceptance_criteria"
                matching = [
                    f"plan-step:{step['id']}"
                    for step in steps
                    if key in step.get(field, [])
                ]
                observed = next(
                    (item for item in available if not item.startswith("plan-step:")),
                    None,
                )
                return [*matching[:1], *([observed] if observed else [])]

            return json.dumps(
                {
                    "requirements": dict.fromkeys(requirements, True),
                    "criteria": dict.fromkeys(criteria, True),
                    "evidence": "Reviewed observed source artifacts and test reports",
                    "requirement_evidence": {
                        name: references(name, True) for name in requirements
                    },
                    "criterion_evidence": {
                        name: references(name, False) for name in criteria
                    },
                }
            )

    orchestrator.models.register("fixture", Provider())
    orchestrator.executor = Executor(orchestrator.router, orchestrator.tools)
    result = asyncio.run(orchestrator.run(store.create(task).id))
    assert result.status == "completed"
    assert result.test_result["coverage"]["totals"]["percent_covered"] == 100
    completed_tools = [
        json.loads(event["payload"])
        for event in store.events.list(event_type="TOOL_CALL_COMPLETED")
    ]
    assert sum(event["tool"] == "filesystem.write" for event in completed_tools) == 2
    assert sum(event["tool"] == "test.run_tests" for event in completed_tools) == 2
    assert sum(event["tool"] == "test.run_coverage" for event in completed_tools) == 2
    assert len(store.events.list(result.id, "correction.started")) == 1


def test_test_side_effect_blocks_task_completion_even_when_tests_pass(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
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
    task.coverage_command = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "-o",
        "coverage.json",
    ]
    (tmp_path / "check.py").write_text(
        "from pathlib import Path\n"
        "from addition import add\n"
        "assert add(2, 3) == 5\n"
        "Path('test-side-effect.txt').write_text('unauthorized')\n"
    )
    created = store.create(task)
    orchestrator.config.data["harness"] = {"max_correction_attempts": 1}

    def execute(_task, step, _context, _scope):
        (tmp_path / "addition.py").write_text(
            "def add(a, b):\n    return sum((a, b))\n"
        )
        return ExecutorOutput(
            subtask_id=step.id,
            success=True,
            output="implemented",
            changed_files=["addition.py"],
            tool_evidence=[{"tool": "filesystem.write", "changed_path": "addition.py"}],
        )

    orchestrator._execute_step = execute
    with pytest.raises(
        RuntimeError,
        match="workspace change is absent from executor results|test execution modified workspace",
    ):
        asyncio.run(orchestrator.run(created.id))
    failed = store.get(created.id)
    assert failed.status == "failed"
    assert any(
        finding["rule"] == "workspace.integrity"
        for finding in failed.validation_result["findings"]
    )
    assert store.events.list(created.id, "task.completed") == []


def test_validator_side_effect_invalidates_completion_evidence(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    orchestrator.config.data["harness"] = {"max_correction_attempts": 1}
    original = orchestrator.validator.validate
    calls = 0

    def changing_review(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        (tmp_path / "review-side-effect.txt").write_text(str(calls))
        return result

    orchestrator.validator.validate = changing_review
    with pytest.raises(
        RuntimeError, match="workspace changed during independent validation"
    ):
        asyncio.run(orchestrator.run(created.id))
    assert calls == 2
    assert store.events.list(created.id, "task.completed") == []
    assert any(
        item["rule"] == "validation.workspace_changed"
        for item in store.corrections.list_for_task(created.id)
    )


def test_workspace_change_after_validation_commit_prevents_completion(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    original = store.record_validation

    def record_then_mutate(task_id, valid, report):
        validation_id = original(task_id, valid, report)
        if valid:
            (tmp_path / "late-change.txt").write_text("changed after validation")
        return validation_id

    store.record_validation = record_then_mutate
    with pytest.raises(
        RuntimeError, match="workspace changed after independent validation"
    ):
        asyncio.run(orchestrator.run(created.id))
    assert store.events.list(created.id, "task.completed") == []


def test_workspace_fingerprint_rejects_non_snapshot():
    from harness.core import workspace_fingerprint

    with pytest.raises(TypeError, match="snapshot is invalid"):
        workspace_fingerprint(None)


def test_concurrent_application_starts_execute_exactly_once(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)

    async def start_both():
        return await asyncio.gather(
            orchestrator.run(created.id),
            orchestrator.run(created.id),
            return_exceptions=True,
        )

    results = asyncio.run(start_both())
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert (
        sum(
            isinstance(result, Task) and result.status == "completed"
            for result in results
        )
        == 1
    )
    assert len(store.events.list(created.id, "task.completed")) == 1
