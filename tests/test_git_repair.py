from types import SimpleNamespace

import pytest
from test_git_integration import git_runtime

from harness.approvals import GitApprovalTarget
from harness.database import QuestionRepository


def repair_case(tmp_path, monkeypatch, *, status="answered", answer="approve"):
    store, harness, task, _git, _repo = git_runtime(tmp_path, monkeypatch)
    task = store.create(task)
    target = GitApprovalTarget(
        task.id,
        str(tmp_path / "repo"),
        ("repair", "dev", "repair/1", "oid", "check.py"),
        "git.repair",
    )
    qid = harness.approvals.request(target, required=True)
    if answer:
        assert store.answer(qid, answer, task.id)
    if status == "consumed":
        assert store.questions.consume_answer(qid)
    elif status == "executed":
        with store.database.connect() as connection:
            connection.execute(
                "UPDATE questions SET status='executed' WHERE id=?", (qid,)
            )
    repair = {
        "repository": target.repository,
        "branch": "dev",
        "branch_oid": "oid",
        "branch_name": "repair/1",
        "paths": ["check.py"],
        "arguments": list(target.arguments),
        "target": target.__dict__,
        "question_id": qid,
    }
    task.git_state = {"workflow": "feature", "branch": "feature/1", "repair": repair}
    service = harness.git_service
    service.oid = lambda ref, cwd: "oid"
    service.save = lambda task_id, state: setattr(task, "git_state", state)
    calls = []

    class Git:
        current = "dev"
        dirty = ""
        branches = ""

        def _git(self, args, cwd):
            calls.append(args)
            if args[:2] == ["branch", "--list"]:
                return SimpleNamespace(stdout=self.branches)
            if args[:2] == ["branch", "--show-current"]:
                return SimpleNamespace(stdout=self.current)
            if "--porcelain=v1" in args:
                return SimpleNamespace(stdout=self.dirty)
            return SimpleNamespace(stdout="")

    service.workflow = Git()
    service.blocked = lambda task_id, error: False
    service.store = SimpleNamespace(
        questions=QuestionRepository(store.database),
        event=lambda *args: None,
    )

    class Grant:
        def consume(self, action, exact):
            return True

    service.approvals = SimpleNamespace(
        issue=lambda *args: Grant(), resume_issue=lambda *args: Grant()
    )
    return service, task, target, calls


def test_repair_approval_creates_exactly_bound_branch(tmp_path, monkeypatch):
    service, task, _target, calls = repair_case(tmp_path, monkeypatch)
    assert service.begin_repair(task)
    assert calls[-1] == ["switch", "-c", "repair/1", "dev"]
    assert task.git_state["repair"]["authorized"]


@pytest.mark.parametrize(
    "change",
    ["repository", "oid", "question", "arguments", "reason", "answer", "dirty"],
)
def test_repair_approval_validation_fails_closed(tmp_path, monkeypatch, change):
    service, task, _target, _calls = repair_case(tmp_path, monkeypatch)
    question = service.store.questions.get(task.git_state["repair"]["question_id"])
    if change == "repository":
        task.git_state["repair"]["repository"] = "/elsewhere"
    elif change == "oid":
        service.oid = lambda ref, cwd: "changed"
    elif change == "question":
        task.git_state["repair"]["question_id"] = 99999
    elif change == "arguments":
        task.git_state["repair"]["arguments"].append("extra.py")
    elif change == "reason":
        with service.store.questions.db.connect() as connection:
            connection.execute(
                "UPDATE questions SET reason='other' WHERE id=?", (question["id"],)
            )
    elif change == "answer":
        with service.store.questions.db.connect() as connection:
            connection.execute(
                "UPDATE questions SET status='open',answer=NULL WHERE id=?",
                (question["id"],),
            )
    else:
        service.workflow.dirty = " M check.py"
    assert not service.begin_repair(task)


@pytest.mark.parametrize("current", ["repair/1", "dev", "other"])
def test_executed_approval_recovers_branch_switch_crash(tmp_path, monkeypatch, current):
    service, task, _target, calls = repair_case(
        tmp_path, monkeypatch, status="executed"
    )
    service.workflow.current = current
    assert service.begin_repair(task) is (current != "other")
    if current == "dev":
        assert calls[-1] == ["switch", "-c", "repair/1", "dev"]


def test_consumed_approval_resumes_grant_once(tmp_path, monkeypatch):
    service, task, _target, calls = repair_case(
        tmp_path, monkeypatch, status="consumed"
    )
    assert service.begin_repair(task)
    assert calls[-1] == ["switch", "-c", "repair/1", "dev"]


def test_executed_repair_branch_with_changed_head_is_rejected(tmp_path, monkeypatch):
    service, task, _target, _calls = repair_case(
        tmp_path, monkeypatch, status="executed"
    )
    service.workflow.current = "repair/1"
    calls = 0

    def oid(ref, cwd):
        nonlocal calls
        calls += 1
        return "oid" if calls == 1 else "changed"

    service.oid = oid
    assert not service.begin_repair(task)


def test_executed_repair_branch_collision_is_rejected(tmp_path, monkeypatch):
    service, task, _target, _calls = repair_case(
        tmp_path, monkeypatch, status="executed"
    )
    service.workflow.branches = "repair/1"
    assert not service.begin_repair(task)


def test_repair_fails_if_grant_was_already_consumed(tmp_path, monkeypatch):
    service, task, _target, _calls = repair_case(tmp_path, monkeypatch)

    class SpentGrant:
        def consume(self, action, target):
            return False

    service.approvals.issue = lambda *args: SpentGrant()
    assert not service.begin_repair(task)


def test_repair_rejects_non_executable_question_status(tmp_path, monkeypatch):
    service, task, _target, _calls = repair_case(tmp_path, monkeypatch)
    with service.store.questions.db.connect() as connection:
        connection.execute(
            "UPDATE questions SET status='open' WHERE id=?",
            (task.git_state["repair"]["question_id"],),
        )
    assert not service.begin_repair(task)
