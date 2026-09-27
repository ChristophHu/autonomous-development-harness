import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_evidence_workflow import ready_runtime

from harness.domain import AcceptanceCriterion
from harness.reconciliation import ReconciliationService


def test_restart_reconciles_actual_state_before_replanning(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    task.status = "executing"
    task.acceptance_criteria.append(
        AcceptanceCriterion(
            id="missing",
            description="remaining work",
            kind="file_exists",
            path="remaining.txt",
        )
    )
    created = store.create(task)
    contexts = []
    original = orchestrator.planner.plan

    def plan(task, context):
        contexts.append(context)
        (tmp_path / "remaining.txt").write_text("implemented remaining work")
        return original(task, context)

    orchestrator.planner.plan = plan
    # The fixture review must acknowledge the additional acceptance criterion.
    original_review = orchestrator.router.complete

    def complete(profile, prompt, **kwargs):
        result = original_review(profile, prompt, **kwargs)
        if prompt.startswith("REVIEW:"):
            review = json.loads(result)
            review["criteria"]["missing"] = True
            return json.dumps(review)
        return result

    orchestrator.router.complete = complete
    assert asyncio.run(orchestrator.run(created.id)).status == "completed"
    report = json.loads(
        store.events.list(created.id, "recovery.reconciled")[0]["payload"]
    )
    assert report["confirmed_criteria"] == ["sum"]
    assert report["remaining_criteria"] == ["missing"]
    assert report["tests"]["commands"][0]["returncode"] == 0
    assert "RECOVERY_RECONCILIATION" in contexts[0]
    assert "remaining work only" in contexts[0]
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT agent FROM agent_runs WHERE task_id=? ORDER BY id",
                (created.id,),
            ).fetchone()[0]
            == "recovery"
        )


def test_reconciliation_distrusts_history_and_inspects_commands(tmp_path):
    from harness.agents import PlannerOutput, Subtask

    store, orchestrator, task = ready_runtime(tmp_path)
    task.acceptance_criteria = [
        AcceptanceCriterion(id="review", description="human review"),
        AcceptanceCriterion(
            id="pass",
            description="command succeeds",
            kind="command",
            command=[sys.executable, "-c", "print('observed')"],
        ),
        AcceptanceCriterion(
            id="fail",
            description="command fails",
            kind="command",
            command=[sys.executable, "-c", "raise SystemExit(2)"],
        ),
        AcceptanceCriterion(
            id="file",
            description="missing despite old success",
            kind="file_contains",
            path="addition.py",
            contains="not implemented",
        ),
    ]
    task = store.create(task)
    steps = [Subtask(id=str(i), title="old", description="old") for i in range(4)]
    plan = PlannerOutput(summary="old", complexity="simple", subtasks=steps)
    plan_id = store.plans.save(task.id, plan.summary, plan.model_dump(mode="json"))
    store.subtasks.save_plan(task.id, steps, plan_id)
    for step, output in zip(
        steps,
        ["invalid-json", "[]", '{"changed_files":["addition.py",17]}', ""],
        strict=True,
    ):
        store.subtasks.update(task.id, step.id, "completed", output, plan_id)
    store.event(task.id, "old.completed", {"success": True})
    calls = []

    def git(args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=" M addition.py", stderr="")

    orchestrator.tools.git = git
    report = ReconciliationService(
        store, orchestrator.tools, orchestrator.validator
    ).inspect(task)
    assert report.previous_plan["summary"] == "old"
    assert report.events[0]["kind"] == "old.completed"
    assert report.confirmed_criteria == ["pass"]
    assert report.remaining_criteria == ["fail", "file"]
    assert report.uncertain_criteria == ["review"]
    assert report.uncertain_requirements == task.requirements
    assert report.files["addition.py"]["sha256"]
    assert calls == [
        ["status", "--porcelain=v1"],
        ["diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
    ]
    assert report.tests["acceptance_commands"]["fail"]["returncode"] == 2


def test_reconciliation_rejects_external_files_and_propagates_inspection_errors(
    tmp_path,
):
    store, orchestrator, task = ready_runtime(tmp_path)
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="escape", description="unsafe", kind="file_exists", path="../secret"
        )
    ]
    task = store.create(task)
    service = ReconciliationService(store, orchestrator.tools, orchestrator.validator)
    with pytest.raises(PermissionError, match="escapes"):
        service.inspect(task)
    task.acceptance_criteria = []
    orchestrator.tools.git = lambda args: (_ for _ in ()).throw(
        PermissionError("git denied")
    )
    with pytest.raises(PermissionError, match="git denied"):
        service.inspect(task)


def test_killed_process_and_new_process_read_persisted_state(tmp_path):
    store, _orchestrator, task = ready_runtime(tmp_path)
    task.status = "executing"
    task = store.create(task)
    # The first process actually dies without any Python cleanup/checkpoint.
    crash = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,signal,sys; from pathlib import Path; Path(sys.argv[1]).write_text('partial implementation'); os.kill(os.getpid(),signal.SIGKILL)",
            str(tmp_path / "partial.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert crash.returncode < 0
    task.acceptance_criteria.append(
        AcceptanceCriterion(
            id="partial",
            description="persisted artifact",
            kind="file_exists",
            path="partial.txt",
        )
    )
    store.update(task)
    script = """
import json,sys
from harness.core import Config,Store,Orchestrator
from harness.reconciliation import ReconciliationService
config=Config()
config.data=json.loads(sys.argv[1])
store=Store(config)
runtime=Orchestrator(store,config)
task=store.get(int(sys.argv[2]))
report=ReconciliationService(store,runtime.tools,runtime.validator).inspect(task)
store.event(task.id,'recovery.reconciled',report.model_dump(mode='json'))
print(report.model_dump_json())
"""
    restart = subprocess.run(
        [sys.executable, "-c", script, json.dumps(store.config.data), str(task.id)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert restart.returncode == 0, restart.stderr
    report = json.loads(restart.stdout)
    assert report["previous_state"] == "executing"
    assert report["confirmed_criteria"] == ["sum", "partial"]
    assert report["files"]["partial.txt"]["content"] == "partial implementation"
    assert store.events.list(task.id, "recovery.reconciled")


@pytest.mark.parametrize("coverage", [None, {"totals": {"percent_covered": 12}}])
def test_reconciliation_includes_failed_test_and_coverage_findings(tmp_path, coverage):
    store, orchestrator, task = ready_runtime(tmp_path)
    task.test_commands = []
    task.acceptance_criteria = []
    task = store.create(task)
    orchestrator.validator.run_tests = lambda task: {
        "commands": [{"command": ["test"], "returncode": 1}],
        "coverage": coverage,
    }
    report = ReconciliationService(
        store, orchestrator.tools, orchestrator.validator
    ).inspect(task)
    assert len(report.test_findings) == 3
    assert report.test_findings[0] == "no test commands configured"
    assert report.test_findings[1].startswith("failed command")
