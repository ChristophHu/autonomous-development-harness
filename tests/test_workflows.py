import pytest

from harness.approvals import ApprovalService, GitApprovalTarget
from harness.workflows import GitWorkflow, Workflow


class GitTool:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def git(self, args, cwd, approval=None, task_id=None):
        self.calls.append((args, cwd, approval))
        if self.result is not None:
            return self.result
        return (args, cwd, approval)

    def git_target(self, task_id, args, cwd):
        return GitApprovalTarget(task_id, cwd, tuple(args))


class Questions:
    def __init__(self, target=None):
        self.target = target or GitApprovalTarget(1, ".", ("branch", "-d", "x"))
        self.row = {
            "task_id": 1,
            "reason": self.target.reason(),
            "status": "answered",
            "answer": "approve",
        }
        self.consumed = False

    def get(self, question_id):
        return self.row if question_id == 2 else None

    def consume_answer(self, question_id):
        if self.consumed:
            return False
        self.consumed = True
        return True

    def consume_approval(self, question_id, reason):
        return reason == self.target.reason()


def test_classify_all_workflows():
    workflow = GitWorkflow(GitTool())
    assert workflow.classify("hotfix outage") == Workflow.HOTFIX
    assert workflow.classify("bug in parser") == Workflow.BUGFIX
    assert workflow.classify("new feature") == Workflow.FEATURE
    assert workflow.classify("release 1.2.0") == Workflow.RELEASE
    assert workflow.classify("other: documentation") == Workflow.OTHER
    assert workflow.branch_name(Workflow.RELEASE, "1.2.0") == "release/1.2.0"


def test_git_delete_requires_matching_human_approval():
    workflow = GitWorkflow(GitTool())
    for args in (
        ["branch", "-d", "x"],
        ["branch", "-D", "x"],
        ["push", "--delete", "origin", "x"],
    ):
        with pytest.raises(PermissionError):
            workflow.execute(args, ".")
    target = Questions().target
    token = ApprovalService(Questions(target)).issue(1, 2, "branch.delete", target)
    assert workflow.execute(["branch", "-d", "x"], ".", token, task_id=1)[2] is token
    assert workflow.delete_branch("x", ".", token, task_id=1)[0] == [
        "branch",
        "-d",
        "x",
    ]
    with pytest.raises(PermissionError):
        workflow.execute(
            ["branch", "-d", "x"],
            ".",
            type("Forged", (), {"approved": True, "action": "branch.delete"})(),
        )
    with pytest.raises(PermissionError, match="persisted"):
        workflow.delete_branch(
            "feature/x",
            ".",
            type("Forged", (), {"approved": True, "action": "branch.delete"})(),
        )
    assert workflow.execute(["status"], ".")[0] == ["status"]


def test_approval_requires_matching_persisted_single_use_human_answer():
    from harness.approvals import ApprovalGrant

    questions = Questions()
    service = ApprovalService(questions)
    grant = service.issue(1, 2, "branch.delete", questions.target)
    assert grant.permits("branch.delete", questions.target) and questions.consumed
    with pytest.raises(PermissionError, match="consumed"):
        service.issue(1, 2, "branch.delete", questions.target)
    denied = Questions()
    denied.row["answer"] = "deny"
    with pytest.raises(PermissionError, match="recorded"):
        ApprovalService(denied).issue(1, 2, "branch.delete", denied.target)
    with pytest.raises(PermissionError, match="only be issued"):
        ApprovalGrant._issue(1, "branch.delete", 2, object())


def test_approval_service_uses_sqlite_question_record(tmp_path):
    from harness.database import Database, QuestionRepository, TaskRepository

    db = Database(tmp_path / "approvals.sqlite")
    task_id = TaskRepository(db).create("delete branch")
    target = GitApprovalTarget(task_id, str(tmp_path), ("branch", "-d", "topic"))
    questions = QuestionRepository(db)
    question_id = questions.create(
        task_id, "Delete?", target.reason(), ["approve", "deny"]
    )
    service = ApprovalService(questions)
    with pytest.raises(PermissionError, match="recorded"):
        service.issue(task_id, question_id, "branch.delete", target)
    assert questions.answer(question_id, "approve", task_id)
    grant = service.issue(task_id, question_id, "branch.delete", target)
    assert grant.permits("branch.delete", target)
    with pytest.raises(PermissionError):
        service.issue(task_id, question_id, "branch.delete", target)


def test_approval_resume_only_reissues_consumed_matching_approval(tmp_path):
    from harness.database import Database, QuestionRepository, TaskRepository

    db = Database(tmp_path / "resume-approval.sqlite")
    task_id = TaskRepository(db).create("repair branch")
    questions = QuestionRepository(db)
    target = GitApprovalTarget(task_id, str(tmp_path), ("repair", "dev"), "git.repair")
    question_id = questions.create(
        task_id, "Repair?", target.reason(), ["approve", "deny"]
    )
    service = ApprovalService(questions)
    with pytest.raises(PermissionError, match="unconsumed"):
        service.resume_issue(task_id, question_id, "git.repair", target)
    assert questions.answer(question_id, "approve", task_id)
    assert questions.consume_answer(question_id)
    grant = service.resume_issue(task_id, question_id, "git.repair", target)
    assert grant.consume("git.repair", target)
    with pytest.raises(PermissionError, match="unconsumed"):
        service.resume_issue(task_id, question_id, "git.repair", target)


def test_git_workflow_local_operations_and_validation():
    tool = GitTool()
    workflow = GitWorkflow(tool)
    workflow.create_branch(Workflow.FEATURE, "Add widget", "/repo")
    workflow.commit(
        ["src/widget.py", "tests/test_widget.py"], "Implement widget", "/repo"
    )
    workflow.merge("feature/add-widget", "/repo")
    workflow.tag_release("1.2.3", "/repo")
    assert [call[0][:2] for call in tool.calls] == [
        ["switch", "-c"],
        ["add", "--"],
        ["commit", "-m"],
        ["merge", "--no-ff"],
        ["tag", "-a"],
    ]
    for args in ((["../secret"], "ok"), (["/absolute"], "ok"), ([], "ok")):
        with pytest.raises(ValueError):
            workflow.commit(args[0], args[1], "/repo")
    with pytest.raises(ValueError, match="semantic version"):
        workflow.tag_release("latest", "/repo")
    with pytest.raises(ValueError, match="invalid Git ref"):
        workflow.merge("feature/../main", "/repo")


def test_git_workflow_propagates_git_failure():
    import subprocess

    tool = GitTool(subprocess.CompletedProcess(["git"], 1, "", "merge conflict"))
    with pytest.raises(RuntimeError, match="merge conflict"):
        GitWorkflow(tool).merge("feature/x", "/repo")
