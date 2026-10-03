import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from test_evidence_workflow import runtime

from harness.domain import AcceptanceCriterion, Task


def test_harness_crash_lease_expiry_restart_remaining_work_and_completion(tmp_path):
    store, _orchestrator = runtime(tmp_path)
    store.config.data["secrets"] = {}
    store.config.data["profiles"]["coding"].update(
        {"tools": ["filesystem.write"], "permissions": ["filesystem"]}
    )
    check = tmp_path / "check.py"
    check.write_text(
        "from addition import add\nfrom remainder import multiply\nassert add(2,3)==5\nassert multiply(2,3)==6\n"
    )
    git = shutil.which("git", path="/opt/homebrew/bin:" + os.environ.get("PATH", ""))
    assert git is not None
    environment = dict(
        os.environ,
        PATH=str(Path(git).parent) + ":" + os.environ.get("PATH", ""),
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    for arguments in (
        ["-c", "init.templateDir=", "init", "--initial-branch=main"],
        ["add", "check.py"],
        [
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "test baseline",
        ],
    ):
        subprocess.run(
            [git, *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    task = store.create(
        Task(
            title="two-stage arithmetic",
            goal="working arithmetic",
            requirements=["add integers", "multiply integers"],
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="add",
                    description="addition",
                    kind="file_contains",
                    path="addition.py",
                    contains="return a + b",
                ),
                AcceptanceCriterion(
                    id="multiply",
                    description="multiplication",
                    kind="file_contains",
                    path="remainder.py",
                    contains="return a * b",
                ),
            ],
            test_commands=[
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--branch",
                    "--source=addition,remainder",
                    "check.py",
                ]
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
    )
    helper = Path(__file__).parent / "fixtures" / "recovery_process.py"
    command = [sys.executable, str(helper), json.dumps(store.config.data), str(task.id)]
    crash = subprocess.run(
        [*command, "crash"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert crash.returncode == -9, crash.stderr
    assert (tmp_path / "crash.ready").exists()
    assert store.get(task.id).status == "executing"
    before = hashlib.sha256((tmp_path / "addition.py").read_bytes()).hexdigest()
    assert store.subtasks.list(task.id)[0]["status"] == "completed"
    assert store.subtasks.list(task.id)[1]["status"] == "running"
    assert not store.tasks.claim(task.id, "competing-restart")
    # Advance only the isolated test lease, instead of waiting five minutes.
    with store.database.connect() as connection:
        connection.execute(
            "UPDATE task_leases SET expires_at=0 WHERE task_id=?", (task.id,)
        )
    # Simulate a second crash after the next process had durably entered recovery.
    store.tasks.transition(task.id, "recovering")
    restart = subprocess.run(
        [*command, "restart"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert restart.returncode == 0, restart.stderr
    assert json.loads(restart.stdout)["status"] == "completed"
    assert store.get(task.id).status == "completed"
    assert hashlib.sha256((tmp_path / "addition.py").read_bytes()).hexdigest() == before
    plan = store.plans.latest(task.id)["payload"]
    assert [step["id"] for step in plan["subtasks"]] == ["remaining"]
    assert store.events.list(task.id, "recovery.scope")
    reconciliation = json.loads(
        store.events.list(task.id, "recovery.reconciled")[0]["payload"]
    )
    assert all(result["returncode"] == 0 for result in reconciliation["git"].values())
    assert "addition.py" in reconciliation["git"]["status"]["stdout"]
    assert store.events.list(task.id, "task.completed")
    tests = store.get(task.id).test_result
    assert all(result["returncode"] == 0 for result in tests["commands"])
    assert tests["coverage"]["totals"]["percent_covered"] == 100
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM task_leases WHERE task_id=?", (task.id,)
            ).fetchone()[0]
            == 0
        )
