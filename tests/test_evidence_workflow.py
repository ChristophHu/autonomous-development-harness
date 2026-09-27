"""Real subprocess E2E: broken code, failing test, correction, coverage, review."""

import asyncio
import json
import sys

import pytest

from harness.agents import ExecutorOutput
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
                                "criteria": [],
                                "evidence": "Current observations require renewed independent inspection.",
                            }
                        }
                    }
                )
            if prompt.startswith("PLAN:"):
                targets = []
                write_paths = []
                if "RECOVERY_SCOPE_JSON:\n" in prompt:
                    targets = json.loads(
                        prompt.split("RECOVERY_SCOPE_JSON:\n")[-1].splitlines()[0]
                    )["remaining_targets"]
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
                                "acceptance_criteria": ["sum is correct"],
                                "recovery_targets": targets,
                                "write_paths": write_paths,
                            }
                        ],
                    }
                )
            return json.dumps(
                {
                    "requirements": {"add two integers": True},
                    "criteria": {"sum": True},
                    "evidence": "Observed real subprocess tests and sum implementation.",
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
            )

    executor = Executor()
    orchestrator.executor = executor
    created = store.create(task)
    result = asyncio.run(orchestrator.run(created.id))
    assert result.status == "completed" and executor.calls == 2
    assert result.test_result["coverage"]["totals"]["percent_covered"] == 100
    assert result.validation_result["valid"] is True
    assert len(store.events.list(created.id, "correction.started")) == 1
    assert store.tasks.claim(created.id, "other") is False


def test_missing_requirements_pause_without_model_or_success(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="unspecified"))
    result = asyncio.run(orchestrator.run(task.id))
    assert result.status == "waiting_human"
    assert store.questions.has_open_required(task.id)
    with pytest.raises(ValueError, match="question"):
        asyncio.run(orchestrator.run(task.id))


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
                    return "Actual workspace edit performed", ModelUsage(
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
                                "required_tools": ["filesystem.write"],
                            }
                        ],
                    }
                )
            assert '"artifacts"' in prompt
            return json.dumps(
                {
                    "requirements": {"add two integers": True},
                    "criteria": {"sum": True},
                    "evidence": "Reviewed observed addition.py and actual test/coverage reports",
                }
            )

    orchestrator.models.register("fixture", Provider())
    orchestrator.executor = Executor(orchestrator.router, orchestrator.tools)
    result = asyncio.run(orchestrator.run(store.create(task).id))
    assert result.status == "completed"
    assert result.test_result["coverage"]["totals"]["percent_covered"] == 100
    assert len(store.events.list(event_type="TOOL_CALL_COMPLETED")) == 2
    assert len(store.events.list(result.id, "correction.started")) == 1


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
